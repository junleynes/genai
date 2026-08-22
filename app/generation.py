"""
Opensource Generative AI job runner — remote WanGP via MCP Streamable HTTP.

Verified against WanGP v1.10.1 / protocol 2025-03-26:
  - Endpoint MUST be http://HOST:PORT/mcp/  (trailing slash; /mcp → 307)
  - Handshake: initialize → notifications/initialized → tools/call
  - Header: Mcp-Session-Id
  - Tools: wangp_generate, wangp_get_job, wangp_list_models, ...
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from . import db

logger = logging.getLogger("genai.generation")

# Deployed-code fingerprint — GET /api/version must show this
BACKEND_ID = "mcp-streamable-http-v2-no-mock"
BACKEND_BUILT = "2026-08-23-outputs-http"

UPLOAD_DIR = Path(__file__).parent.parent / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(__file__).parent.parent / "static" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_mcp_req_id = 0
_mcp_sessions: dict[str, str] = {}  # endpoint -> session id


def _duration_to_frames(seconds: float, fps: int = 24) -> int:
    n = max(1, int(round(float(seconds) * fps)))
    return max(5, (n // 4) * 4 + 1)


def _map_job_to_settings(job: dict, defaults: dict) -> dict:
    params = job.get("params") or {}
    jtype = job.get("job_type", "t2v")
    prompt = job.get("prompt") or ""

    model = params.get("model_type") or params.get("model")
    if model in (None, "", "auto"):
        model = defaults.get("model_type") or "ltx2_22B_distilled"

    resolution = params.get("resolution") or defaults.get("resolution") or "1280x704"
    steps = int(params.get("steps") or params.get("num_inference_steps") or defaults.get("num_inference_steps") or 8)
    seed = int(params.get("seed", -1))
    duration = float(params.get("duration") or params.get("duration_seconds") or 4)
    video_length = params.get("video_length")
    if video_length is None:
        video_length = _duration_to_frames(duration)

    settings: dict[str, Any] = {
        "model_type": str(model),
        "prompt": prompt,
        "negative_prompt": params.get("negative_prompt") or "",
        "resolution": resolution,
        "num_inference_steps": steps,
        "seed": seed,
        "video_length": int(video_length),
        "duration_seconds": duration,
        "force_fps": str(params.get("force_fps") or params.get("fps") or defaults.get("force_fps") or "24"),
        "guidance_scale": float(params.get("guidance_scale") or params.get("cfg") or 5.0),
        "flow_shift": float(params.get("flow_shift") or 5.0),
        "_api": {"return_media": True},
    }

    image_path = (
        job.get("image_path") or params.get("image_path")
        or params.get("image_url") or job.get("image_url")
    )
    image_end = (
        job.get("image_end_path") or params.get("image_end_path")
        or params.get("end_image_url") or job.get("end_image_url")
    )
    video_path = (
        job.get("video_path") or params.get("video_path")
        or params.get("video_url") or job.get("video_url")
    )
    audio_path = (
        job.get("audio_path") or params.get("audio_path")
        or params.get("audio_url") or job.get("audio_url")
    )

    if jtype in ("i2v", "ia2v") and image_path:
        settings["image_start"] = str(image_path)
        settings["image_prompt_type"] = "S"
    if image_end:
        settings["image_end"] = str(image_end)
    if jtype == "v2v" and video_path:
        settings["video_source"] = str(video_path)
        settings["image_prompt_type"] = "V"
    if jtype == "ia2v" and audio_path:
        settings["audio_guide"] = str(audio_path)
        settings["audio_prompt_type"] = "A"
    if jtype in ("t2i", "i2i"):
        settings["image_mode"] = 1
        settings["video_length"] = 1
        if jtype == "i2i" and image_path:
            settings["image_start"] = str(image_path)

    quality = (params.get("quality") or "balanced").lower()
    if quality == "fast":
        settings["num_inference_steps"] = min(settings["num_inference_steps"], 6)
    elif quality == "quality":
        settings["num_inference_steps"] = max(settings["num_inference_steps"], 20)

    return settings


def _copy_result_into_static(src: str | Path, job_id: str) -> str:
    src = Path(src)
    if not src.exists():
        return str(src)
    ext = src.suffix or ".mp4"
    dest = RESULTS_DIR / f"{job_id}{ext}"
    shutil.copy2(src, dest)
    return f"/static/results/{dest.name}"


def _next_mcp_id() -> int:
    global _mcp_req_id
    _mcp_req_id += 1
    return _mcp_req_id


def _mcp_endpoint(base: str) -> str:
    """Always end with /mcp/ — WanGP returns 307 without trailing slash."""
    base = (base or "").strip()
    if not base:
        return ""
    # strip trailing slashes then normalize
    base = base.rstrip("/")
    if base.endswith("/mcp"):
        return base + "/"
    if "/mcp/" in base + "/":
        # already has mcp somewhere
        if not base.endswith("/mcp"):
            pass
    return base + "/mcp/"


def _parse_sse_or_json(body: str, content_type: str) -> dict:
    ctype = (content_type or "").lower()
    body = body or ""
    if (
        "text/event-stream" in ctype
        or body.lstrip().startswith("event:")
        or "\ndata:" in body
        or body.lstrip().startswith("data:")
    ):
        last = None
        for line in body.splitlines():
            if line.startswith("data:"):
                raw = line[5:].strip()
                if raw and raw != "[DONE]":
                    try:
                        last = json.loads(raw)
                    except Exception:
                        pass
        if last is None:
            raise RuntimeError(f"MCP SSE empty/unparseable: {body[:500]}")
        return last
    try:
        return json.loads(body)
    except Exception as e:
        raise RuntimeError(f"MCP non-JSON: {e}; body={body[:500]}")


def _parse_mcp_tool_result(data: dict) -> Any:
    if not isinstance(data, dict):
        return data
    if data.get("error"):
        err = data["error"]
        if isinstance(err, dict):
            msg = err.get("message") or str(err)
            if err.get("data") is not None:
                msg = f"{msg} | {err.get('data')}"
            raise RuntimeError(msg)
        raise RuntimeError(str(err))
    result = data.get("result", data)
    if not isinstance(result, dict):
        return result
    if result.get("isError"):
        content = result.get("content") or []
        texts = []
        for item in content:
            if isinstance(item, dict):
                texts.append(item.get("text") or str(item))
            else:
                texts.append(str(item))
        raise RuntimeError("; ".join(texts) or "MCP tool isError")
    if result.get("structuredContent") is not None:
        return result["structuredContent"]
    content = result.get("content")
    if isinstance(content, list) and content:
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text") or "")
            elif isinstance(item, str):
                texts.append(item)
        joined = "\n".join(texts).strip()
        if joined:
            try:
                return json.loads(joined)
            except Exception:
                return {"text": joined}
    return result


def mcp_raw_request(
    mcp_url: str,
    method: str,
    params: dict | None = None,
    *,
    timeout: float = 60.0,
    session_id: str | None = None,
    is_notification: bool = False,
) -> tuple[dict | None, str | None]:
    endpoint = _mcp_endpoint(mcp_url)
    if not endpoint:
        raise RuntimeError("MCP URL is empty")

    payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if not is_notification:
        payload["id"] = _next_mcp_id()
    if params is not None:
        payload["params"] = params

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    sid = session_id or _mcp_sessions.get(endpoint)
    if sid:
        headers["Mcp-Session-Id"] = sid

    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        r = client.post(endpoint, json=payload, headers=headers)
        new_sid = (
            r.headers.get("mcp-session-id")
            or r.headers.get("Mcp-Session-Id")
            or sid
        )
        if new_sid:
            _mcp_sessions[endpoint] = new_sid

        if is_notification:
            if r.status_code not in (200, 202, 204):
                raise RuntimeError(f"MCP notification HTTP {r.status_code}: {r.text[:400]}")
            return None, new_sid

        if r.status_code >= 400:
            raise RuntimeError(
                f"MCP HTTP {r.status_code} {endpoint} method={method}: {r.text[:500]}"
            )

        data = _parse_sse_or_json(r.text, r.headers.get("content-type", ""))
        return data, new_sid


def mcp_ensure_session(mcp_url: str, timeout: float = 30.0) -> str | None:
    endpoint = _mcp_endpoint(mcp_url)
    if endpoint in _mcp_sessions and _mcp_sessions[endpoint]:
        return _mcp_sessions[endpoint]

    data, sid = mcp_raw_request(
        mcp_url,
        "initialize",
        {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "Opensource Generative AI", "version": "1.1.0"},
        },
        timeout=timeout,
        session_id=None,
    )
    try:
        mcp_raw_request(
            mcp_url,
            "notifications/initialized",
            {},
            timeout=timeout,
            session_id=sid,
            is_notification=True,
        )
    except Exception as e:
        logger.debug("initialized notification: %s", e)

    if data and isinstance(data, dict) and data.get("error"):
        err = data["error"]
        raise RuntimeError(err.get("message") if isinstance(err, dict) else str(err))

    return sid or _mcp_sessions.get(endpoint)


def mcp_call_tool(mcp_url: str, name: str, arguments: dict, timeout: float = 120.0) -> Any:
    sid = mcp_ensure_session(mcp_url, timeout=min(30.0, timeout))
    data, _ = mcp_raw_request(
        mcp_url,
        "tools/call",
        {"name": name, "arguments": arguments or {}},
        timeout=timeout,
        session_id=sid,
    )
    if data is None:
        raise RuntimeError(f"Empty MCP response for {name}")
    return _parse_mcp_tool_result(data)


def mcp_upload_media(mcp_url: str, local_path: str) -> str:
    path = Path(local_path)
    if not path.is_file():
        return str(local_path)

    create = mcp_call_tool(
        mcp_url, "wangp_create_gallery_upload", {"filename": path.name}, timeout=60.0
    )
    if not isinstance(create, dict):
        raise RuntimeError(f"gallery upload create failed: {create}")

    upload_url = (
        create.get("url")
        or create.get("upload_url")
        or create.get("put_url")
        or create.get("href")
    )
    gallery_id = create.get("gallery_id") or create.get("id") or create.get("item_id")

    if upload_url and str(upload_url).startswith("/"):
        origin = _mcp_endpoint(mcp_url).rstrip("/")
        if origin.endswith("/mcp"):
            origin = origin[:-4]
        upload_url = origin.rstrip("/") + str(upload_url)

    if not upload_url:
        raise RuntimeError(f"No upload URL from wangp_create_gallery_upload: {create}")

    data = path.read_bytes()
    headers = {"Content-Type": create.get("content_type") or "application/octet-stream"}
    with httpx.Client(timeout=600.0, follow_redirects=True) as client:
        put = client.put(str(upload_url), content=data, headers=headers)
        if put.status_code >= 400:
            put = client.post(str(upload_url), content=data, headers=headers)
        if put.status_code >= 400:
            raise RuntimeError(f"Gallery PUT HTTP {put.status_code}: {put.text[:300]}")
        try:
            body = put.json()
            if isinstance(body, dict):
                gallery_id = (
                    body.get("gallery_id") or body.get("id") or body.get("item_id") or gallery_id
                )
        except Exception:
            pass

    if not gallery_id:
        raise RuntimeError(f"Upload OK but no gallery_id: {create}")
    return str(gallery_id)


def _resolve_local_media(p: str | None) -> Optional[str]:
    if not p:
        return None
    path = Path(p)
    if path.is_file():
        return str(path)
    rel = str(p).lstrip("/")
    for candidate in (
        Path(__file__).parent.parent / rel,
        UPLOAD_DIR / Path(p).name,
        UPLOAD_DIR / "jobs" / Path(p).name,
        Path(__file__).parent.parent / "static" / "uploads" / "jobs" / Path(p).name,
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def _prepare_mcp_source(mcp_url: str, job: dict, settings_cfg: dict) -> dict:
    defaults = {
        "model_type": settings_cfg.get("default_model_type") or "ltx2_22B_distilled",
        "resolution": settings_cfg.get("default_resolution") or "1280x704",
        "num_inference_steps": settings_cfg.get("default_steps") or 8,
        "force_fps": settings_cfg.get("default_fps") or "24",
    }
    source = _map_job_to_settings(job, defaults)
    for key in (
        "image_start", "image_end", "image_refs",
        "video_source", "video_guide", "audio_guide", "audio_guide2",
    ):
        val = source.get(key)
        if not val:
            continue
        if isinstance(val, list):
            out = []
            for item in val:
                local = _resolve_local_media(str(item))
                out.append(mcp_upload_media(mcp_url, local) if local else item)
            source[key] = out
        else:
            local = _resolve_local_media(str(val))
            if local:
                source[key] = mcp_upload_media(mcp_url, local)
    return source


def _mcp_origin(mcp_url: str) -> str:
    ep = _mcp_endpoint(mcp_url).rstrip("/")
    if ep.endswith("/mcp"):
        ep = ep[:-4]
    return ep.rstrip("/")


def _save_bytes_result(job_id: str, data: bytes, ext: str = ".png") -> str:
    if not ext.startswith("."):
        ext = "." + ext
    # sniff
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        ext = ".png"
    elif data[:3] == b"\xff\xd8\xff":
        ext = ".jpg"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        ext = ".webp"
    elif data[:4] == b"\x00\x00\x00\x18" or data[4:8] == b"ftyp":
        ext = ".mp4"
    elif data[:4] == b"\x1aE\xdf\xa3":
        ext = ".webm"
    dest = RESULTS_DIR / f"{job_id}{ext}"
    dest.write_bytes(data)
    return f"/static/results/{dest.name}"


def _rewrite_transfer_url(url: str, mcp_url: str) -> str:
    """Relative or localhost download URLs must use the MCP host GenAI can reach."""
    url = str(url).strip()
    origin = _mcp_origin(mcp_url)
    if not url:
        return url
    if url.startswith("/"):
        return origin.rstrip("/") + url
    try:
        from urllib.parse import urlparse, urlunparse
        u = urlparse(url)
        o = urlparse(origin)
        host = (u.hostname or "").lower()
        if host in ("127.0.0.1", "localhost", "0.0.0.0", "::1") or not host:
            # keep path/query, swap host/port to MCP origin
            return urlunparse((
                o.scheme or "http",
                o.netloc,
                u.path or "",
                u.params,
                u.query,
                u.fragment,
            ))
    except Exception:
        pass
    return url


def mcp_download_gallery_item(mcp_url: str, gallery_id: str, job_id: str) -> str:
    """Use wangp_create_gallery_download then GET the short-lived URL onto this server."""
    # try several argument shapes
    last_err = None
    create = None
    for args in (
        {"gallery_id": gallery_id},
        {"id": gallery_id},
        {"item_id": gallery_id},
        {"gallery_id": str(gallery_id)},
    ):
        try:
            create = mcp_call_tool(
                mcp_url, "wangp_create_gallery_download", args, timeout=60.0
            )
            if create:
                break
        except Exception as e:
            last_err = e
            create = None
    if not isinstance(create, dict):
        raise RuntimeError(f"gallery download create failed: {last_err or create}")

    url = (
        create.get("url")
        or create.get("download_url")
        or create.get("get_url")
        or create.get("href")
        or create.get("uri")
    )
    if not url and isinstance(create.get("data"), dict):
        d = create["data"]
        url = d.get("url") or d.get("download_url") or d.get("href")
    if not url:
        raise RuntimeError(f"No download URL from wangp_create_gallery_download: {create}")

    url = _rewrite_transfer_url(str(url), mcp_url)
    logger.info("Gallery download GET %s (id=%s)", url, gallery_id)

    with httpx.Client(timeout=600.0, follow_redirects=True) as client:
        r = client.get(str(url))
        if r.status_code >= 400:
            raise RuntimeError(f"Gallery download HTTP {r.status_code}: {r.text[:200]} url={url}")
        ct = (r.headers.get("content-type") or "").lower()
        ext = ".bin"
        if "png" in ct:
            ext = ".png"
        elif "jpeg" in ct or "jpg" in ct:
            ext = ".jpg"
        elif "webp" in ct:
            ext = ".webp"
        elif "mp4" in ct or "video" in ct:
            ext = ".mp4"
        elif "webm" in ct:
            ext = ".webm"
        cd = r.headers.get("content-disposition") or ""
        if "filename=" in cd:
            import re as _re
            mm = _re.search(r'filename\*?=(?:UTF-8\'\'\')?"?([^";]+)"?', cd, re.I)
            if mm and "." in mm.group(1):
                ext = "." + mm.group(1).strip().rsplit(".", 1)[-1].lower()[:8]
        if len(r.content) < 32:
            raise RuntimeError(f"Gallery download empty body from {url}")
        return _save_bytes_result(job_id, r.content, ext)


def _artifact_to_local_file(job_id: str, art: dict) -> str | None:
    """Decode MCP result.artifacts / embedded media into static/results."""
    if not isinstance(art, dict):
        return None
    # base64 fields common over JSON-RPC
    for key in ("image_base64", "data", "b64", "base64", "png_base64", "jpeg_base64"):
        val = art.get(key)
        if isinstance(val, str) and len(val) > 100:
            import base64
            raw = val
            if "," in raw and raw.strip().startswith("data:"):
                raw = raw.split(",", 1)[1]
            try:
                data = base64.b64decode(raw)
                if len(data) > 64:
                    media = str(art.get("media_type") or art.get("type") or "image")
                    ext = ".mp4" if "video" in media else ".png"
                    if data[:3] == b"\xff\xd8\xff":
                        ext = ".jpg"
                    return _save_bytes_result(job_id, data, ext)
            except Exception:
                pass
    # raw bytes as list of ints (unlikely but possible)
    for key in ("image_bytes", "bytes", "content"):
        val = art.get(key)
        if isinstance(val, list) and len(val) > 64 and all(isinstance(x, int) for x in val[:20]):
            data = bytes(val)
            return _save_bytes_result(job_id, data, ".png")
    return None


def _normalize_gallery_items(raw) -> list[dict]:
    """Accept many MCP gallery payload shapes."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if not isinstance(raw, dict):
        return []
    out = []
    for key in (
        "items", "gallery", "gallery_items", "images", "videos", "audio",
        "visual", "visuals", "records", "entries", "data", "result",
        "selected", "selection",
    ):
        val = raw.get(key)
        if isinstance(val, list):
            out.extend([x for x in val if isinstance(x, dict)])
        elif isinstance(val, dict):
            # nested container or single item
            if any(k in val for k in ("gallery_id", "id", "path", "filename")):
                out.append(val)
            else:
                out.extend(_normalize_gallery_items(val))
    # single item dict
    if not out and any(k in raw for k in ("gallery_id", "path", "filename")):
        out.append(raw)
    return out


def mcp_list_tool_names(mcp_url: str) -> list[str]:
    try:
        data, _ = mcp_raw_request(mcp_url, "tools/list", {}, timeout=20.0)
        if not isinstance(data, dict):
            return []
        result = data.get("result") or data
        tools = result.get("tools") if isinstance(result, dict) else []
        names = []
        for tool in tools or []:
            if isinstance(tool, dict) and tool.get("name"):
                names.append(str(tool["name"]))
        return names
    except Exception as e:
        logger.warning("tools/list failed: %s", e)
        return []


def mcp_download_via_outputs_http(base_url: str, remote_path: str, job_id: str) -> str:
    """Fetch by filename from a static HTTP root pointing at WanGP outputs/."""
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise RuntimeError("outputs HTTP base empty")
    name = Path(str(remote_path).replace("\\", "/")).name
    if not name:
        raise RuntimeError("no filename in path")
    from urllib.parse import quote
    candidates = [
        f"{base}/{quote(name)}",
        f"{base}/{name}",
        f"{base}/{quote(name, safe='')}",
    ]
    # also try URL-encoded spaces from original
    last_err = None
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        for u in candidates:
            try:
                r = client.get(u)
                if r.status_code == 200 and r.content and len(r.content) > 64:
                    ct = (r.headers.get("content-type") or "").lower()
                    if "text/html" in ct and b"<html" in r.content[:200].lower():
                        last_err = f"HTML at {u}"
                        continue
                    ext = Path(name).suffix or ".bin"
                    return _save_bytes_result(job_id, r.content, ext)
                last_err = f"HTTP {r.status_code} {u}"
            except Exception as e:
                last_err = str(e)
    raise RuntimeError(f"outputs HTTP fetch failed: {last_err}")


def mcp_fetch_result_via_gallery(mcp_url: str, job_id: str, remote_path: str | None = None) -> str:
    """
    MCP-only transfer (no Gradio):
      list gallery (with retries) → create_gallery_download → GET bytes
    """
    name = Path(str(remote_path).replace("\\", "/")).name if remote_path else None
    stem = Path(name).stem if name else None

    attempts = (
        ("wangp_list_gallery", {}),
        ("wangp_list_gallery", {"media_type": "image"}),
        ("wangp_list_gallery", {"media_type": "video"}),
        ("wangp_list_gallery", {"media_type": "all"}),
        ("wangp_get_gallery_selection", {"media_type": "all"}),
        ("wangp_get_gallery_selection", {"media_type": "image"}),
        ("wangp_get_gallery_selection", {}),
    )

    items: list[dict] = []
    call_errors: list[str] = []
    last_raw = None
    for attempt in range(4):
        if attempt:
            time.sleep(1.2 * attempt)
        for tool, args in attempts:
            try:
                raw = mcp_call_tool(mcp_url, tool, args, timeout=45.0)
                last_raw = raw
            except Exception as e:
                call_errors.append(f"{tool}: {e}")
                logger.warning("gallery tool %s failed: %s", tool, e)
                continue
            batch = _normalize_gallery_items(raw)
            for it in batch:
                if it not in items:
                    items.append(it)
        if items:
            break

    if not items:
        tool_names = mcp_list_tool_names(mcp_url)
        gallery_tools = [n for n in tool_names if "gallery" in n.lower()]
        raise RuntimeError(
            "wangp_list_gallery returned no items. "
            + (f"tool_errors=[{'; '.join(call_errors)[:350]}] " if call_errors else "")
            + (f"gallery_tools_on_server={gallery_tools or 'none'} " if tool_names else "tools/list unavailable ")
            + f"last_raw={str(last_raw)[:180]}"
        )

    def score(it: dict) -> int:
        s = 0
        p = str(it.get("path") or it.get("file") or it.get("filename") or it.get("name") or "")
        n = Path(p.replace("\\", "/")).name
        gid = it.get("gallery_id") or it.get("id") or it.get("item_id")
        if name and n.lower() == name.lower():
            s += 200
        if name and name.lower() in p.lower():
            s += 80
        if stem and stem.lower() in n.lower():
            s += 60
        if remote_path and remote_path.replace("\\", "/").lower() in p.replace("\\", "/").lower():
            s += 100
        if gid:
            s += 5
        return s

    items_sorted = sorted(items, key=score, reverse=True)
    errors = []
    for it in items_sorted[:12]:
        gid = it.get("gallery_id") or it.get("id") or it.get("item_id") or it.get("client_id")
        if not gid:
            continue
        gs = str(gid).strip()
        if not gs or "/" in gs or "\\" in gs or gs.lower().endswith((".png", ".jpg", ".jpeg", ".mp4", ".webp")):
            continue
        try:
            return mcp_download_gallery_item(mcp_url, gs, job_id)
        except Exception as e:
            errors.append(f"{gs}: {e}")
            logger.warning("gallery download failed id=%s: %s", gs, e)

    raise RuntimeError(
        "Gallery items found but download failed: "
        + ("; ".join(errors)[:400] if errors else "no usable gallery_id in items")
    )



def mcp_download_remote_file(mcp_url: str, remote_path: str, job_id: str, gradio_url: str = "") -> str:
    """
    Only used when Gradio URL is explicitly set and is NOT the MCP endpoint.
    MCP-only installs must use gallery download — never probe /file= on MCP port.
    """
    remote_path = str(remote_path).strip().strip('"').strip("'")
    if not remote_path:
        raise RuntimeError("empty remote path")

    if remote_path.startswith(("http://", "https://")):
        with httpx.Client(timeout=600.0, follow_redirects=True) as client:
            r = client.get(remote_path)
            if r.status_code >= 400:
                raise RuntimeError(f"Download HTTP {r.status_code}")
            ext = Path(remote_path.split("?")[0]).suffix or ".bin"
            return _save_bytes_result(job_id, r.content, ext)

    local = Path(remote_path)
    if local.is_file():
        return _copy_result_into_static(local, job_id)

    gurl = (gradio_url or "").strip().rstrip("/")
    if not gurl:
        raise RuntimeError("not local, gallery download needed (no Gradio URL configured)")

    # Refuse Gradio probes against the MCP origin (same host:port as /mcp/)
    try:
        from urllib.parse import urlparse
        mcp_o = urlparse(_mcp_origin(mcp_url))
        gr_o = urlparse(gurl)
        if mcp_o.hostname and gr_o.hostname and mcp_o.hostname == gr_o.hostname and (
            (mcp_o.port or (443 if mcp_o.scheme == "https" else 80))
            == (gr_o.port or (443 if gr_o.scheme == "https" else 80))
        ):
            raise RuntimeError(
                "Gradio URL matches MCP host:port — /file= will 404. "
                "Clear Gradio URL for MCP-only, or set real Gradio (e.g. :7860)."
            )
    except RuntimeError:
        raise
    except Exception:
        pass

    from urllib.parse import quote
    name = Path(remote_path.replace("\\", "/")).name
    path_variants = [remote_path, remote_path.replace("\\", "/")]
    candidates = []
    for pv in path_variants:
        candidates.append(f"{gurl}/file={pv}")
        candidates.append(f"{gurl}/gradio_api/file={pv}")
        candidates.append(f"{gurl}/file={quote(pv, safe=':/\\\\')}")

    last_err = None
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for u in candidates:
            try:
                r = client.get(u)
                if r.status_code == 200 and r.content and len(r.content) > 100:
                    ct = (r.headers.get("content-type") or "").lower()
                    if "text/html" in ct and b"<html" in r.content[:200].lower():
                        continue
                    ext = Path(name).suffix or ".bin"
                    return _save_bytes_result(job_id, r.content, ext)
                last_err = f"HTTP {r.status_code}"
            except Exception as e:
                last_err = str(e)
    raise RuntimeError(f"Gradio file fetch failed last={last_err}")



def _artifact_to_local_file(job_id: str, art: dict) -> str | None:
    """Decode MCP result.artifacts / embedded media into static/results."""
    if not isinstance(art, dict):
        return None
    # base64 fields common over JSON-RPC
    for key in ("image_base64", "data", "b64", "base64", "png_base64", "jpeg_base64"):
        val = art.get(key)
        if isinstance(val, str) and len(val) > 100:
            import base64
            raw = val
            if "," in raw and raw.strip().startswith("data:"):
                raw = raw.split(",", 1)[1]
            try:
                data = base64.b64decode(raw)
                if len(data) > 64:
                    media = str(art.get("media_type") or art.get("type") or "image")
                    ext = ".mp4" if "video" in media else ".png"
                    if data[:3] == b"\xff\xd8\xff":
                        ext = ".jpg"
                    return _save_bytes_result(job_id, data, ext)
            except Exception:
                pass
    # raw bytes as list of ints (unlikely but possible)
    for key in ("image_bytes", "bytes", "content"):
        val = art.get(key)
        if isinstance(val, list) and len(val) > 64 and all(isinstance(x, int) for x in val[:20]):
            data = bytes(val)
            return _save_bytes_result(job_id, data, ".png")
    return None



def _extract_gallery_id(obj: Any) -> str | None:
    if obj is None:
        return None
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return None
        # filesystem path → not a gallery id
        if "/" in s or "\\" in s or s.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm", ".gif")):
            return None
        if len(s) < 120:
            return s
        return None
    if isinstance(obj, dict):
        for k in ("gallery_id", "id", "item_id", "media_id", "client_id"):
            v = obj.get(k)
            if v is not None:
                gid = _extract_gallery_id(str(v) if not isinstance(v, dict) else None)
                if gid:
                    return gid
                if isinstance(v, dict):
                    gid = _extract_gallery_id(v)
                    if gid:
                        return gid
    return None


def _walk_gallery_ids(obj: Any, found: list | None = None) -> list[str]:
    if found is None:
        found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("gallery_id", "item_id", "media_id") and v is not None:
                s = str(v).strip()
                if s and s not in found and "/" not in s and "\\" not in s:
                    found.append(s)
            elif k == "id" and v is not None:
                s = str(v).strip()
                # skip job ids that look like uuid if path-like — keep short ids
                if s and s not in found and "/" not in s and "\\" not in s and len(s) < 80:
                    # avoid pure numeric progress fields
                    if not (k == "id" and isinstance(v, int) and v < 10000):
                        found.append(s)
            else:
                _walk_gallery_ids(v, found)
    elif isinstance(obj, list):
        for x in obj:
            _walk_gallery_ids(x, found)
    return found


def _walk_file_paths(obj: Any, found: list | None = None) -> list[str]:
    if found is None:
        found = []
    if isinstance(obj, str):
        s = obj.strip()
        low = s.lower()
        if any(low.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm", ".gif", ".mov")):
            if s not in found:
                found.append(s)
        elif ("\\outputs\\" in s.lower() or "/outputs/" in s.lower()) and s not in found:
            found.append(s)
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk_file_paths(v, found)
    elif isinstance(obj, list):
        for x in obj:
            _walk_file_paths(x, found)
    return found



def _merge_gallery_from_job_state(result: dict, st: dict) -> dict:
    """Pull gallery_items / generated_files from outer job poll into result."""
    if not isinstance(result, dict):
        result = {}
    if not isinstance(st, dict):
        return result
    for key in ("gallery_items", "generated_files", "files", "artifacts"):
        if key not in result or not result.get(key):
            if st.get(key):
                result[key] = st[key]
        # also nested under result
    nested = st.get("result") if isinstance(st.get("result"), dict) else None
    if nested:
        for key in ("gallery_items", "generated_files", "files", "artifacts"):
            if key not in result or not result.get(key):
                if nested.get(key):
                    result[key] = nested[key]
    return result

def _apply_mcp_result(job_id: str, result: dict, mcp_url: str = "", gradio_url: str = "") -> bool:
    """Copy/download result onto *this* server so the browser can preview it."""
    if not isinstance(result, dict):
        db.update_job(job_id, {"error": f"Bad MCP result type: {type(result)}"})
        return False

    success = result.get("success")
    files = result.get("generated_files") or result.get("files") or []
    gallery = result.get("gallery_items") or []
    errors = result.get("errors") or []

    if success is False or (not files and not gallery and errors and success is not True):
        # only fail hard if explicitly failed
        if success is False or errors:
            msgs = []
            for err in errors:
                msgs.append(err.get("message") if isinstance(err, dict) else str(err))
            if msgs:
                db.update_job(job_id, {"error": ("; ".join(msgs) or "MCP generation failed")[:800]})
                return False

    url = None
    transfer_errors: list[str] = []

    # 1) Explicit gallery list
    for g in gallery if isinstance(gallery, list) else []:
        gid = _extract_gallery_id(g)
        if gid and mcp_url:
            try:
                url = mcp_download_gallery_item(mcp_url, gid, job_id)
                break
            except Exception as e:
                transfer_errors.append(f"gallery {gid}: {e}")

    # 2) Any gallery-like ids nested in result
    if not url and mcp_url:
        for gid in _walk_gallery_ids(result):
            try:
                url = mcp_download_gallery_item(mcp_url, gid, job_id)
                if url:
                    break
            except Exception as e:
                transfer_errors.append(f"walk gallery {gid}: {e}")

    # 3) generated_files entries
    if not url:
        for f in files if isinstance(files, list) else []:
            gid = _extract_gallery_id(f)
            remote_path = None
            if isinstance(f, dict):
                remote_path = f.get("path") or f.get("file") or f.get("url") or f.get("filename")
            else:
                remote_path = str(f)
            if gid and mcp_url:
                try:
                    url = mcp_download_gallery_item(mcp_url, str(gid), job_id)
                    break
                except Exception as e:
                    transfer_errors.append(str(e))
            if remote_path and (gradio_url or "").strip():
                try:
                    url = mcp_download_remote_file(mcp_url, str(remote_path), job_id, gradio_url=gradio_url)
                    break
                except Exception as e:
                    transfer_errors.append(str(e))
            elif remote_path:
                transfer_errors.append(f"path={remote_path} (need gallery id; Gradio not configured)")

    # 4) Any media paths nested in result
    if not url and (gradio_url or "").strip():
        for path in _walk_file_paths(result):
            try:
                url = mcp_download_remote_file(mcp_url, path, job_id, gradio_url=gradio_url)
                break
            except Exception as e:
                transfer_errors.append(str(e))

    # 5) Artifacts (return_media / embedded base64)
    if not url:
        arts = result.get("artifacts") or result.get("media") or []
        if isinstance(arts, dict):
            arts = [arts]
        if isinstance(arts, list):
            for art in arts:
                try:
                    got = _artifact_to_local_file(job_id, art) if isinstance(art, dict) else None
                    if got:
                        url = got
                        break
                except Exception as e:
                    transfer_errors.append(f"artifact: {e}")

    # 6) MCP-only: list gallery & download (no Gradio required)
    if not url and mcp_url:
        paths = _walk_file_paths(result)
        hint = paths[0] if paths else None
        try:
            url = mcp_fetch_result_via_gallery(mcp_url, job_id, hint)
        except Exception as e:
            transfer_errors.append(f"list_gallery: {e}")
            logger.warning("gallery list transfer failed: %s", e)

    # 6b) Static HTTP server root for WanGP outputs/ (reliable when gallery is empty)
    if not url:
        try:
            settings = db.get_settings()
            out_base = (settings.get("wan2gp_outputs_http_base") or "").strip()
            paths = _walk_file_paths(result)
            hint = paths[0] if paths else None
            if not hint:
                # also check generated_files strings
                for f in (result.get("generated_files") or []):
                    if isinstance(f, str):
                        hint = f
                        break
                    if isinstance(f, dict) and f.get("path"):
                        hint = f["path"]
                        break
            if out_base and hint:
                url = mcp_download_via_outputs_http(out_base, hint, job_id)
        except Exception as e:
            transfer_errors.append(f"outputs_http: {e}")
            logger.warning("outputs http transfer failed: %s", e)

    if not url:
        msg = (
            "Generation finished on remote WanGP, but the file could not be transferred over MCP. "
            "No Gradio is required — the file must be available via Gallery download "
            "(wangp_list_gallery + wangp_create_gallery_download). "
            + ("; ".join(transfer_errors)[:450] if transfer_errors else "")
        )
        db.update_job(job_id, {"error": msg[:800], "status": "failed", "progress": 0})
        return False

    db.update_job(job_id, {
        "status": "completed",
        "progress": 100,
        "result_url": url,
        "preview_url": url,
        "error": None,
    })
    logger.info("Job %s result stored locally at %s", job_id, url)
    return True




def _extract_progress_pct(st: dict) -> int | None:
    """Best-effort progress 0-100 from wangp_get_job payloads."""
    if not isinstance(st, dict):
        return None

    def from_ratio(val) -> int | None:
        try:
            v = float(val)
        except (TypeError, ValueError):
            return None
        if v < 0:
            return None
        if v <= 1.0:
            return int(round(v * 100))
        if v <= 100:
            return int(round(v))
        return min(100, int(round(v)))

    # direct fields
    for key in ("progress", "percent", "percentage", "pct", "ratio"):
        if key in st and st[key] is not None:
            p = from_ratio(st[key])
            if p is not None:
                return p

    status = st.get("status")
    if isinstance(status, dict):
        for key in ("progress", "percent", "percentage"):
            if key in status and status[key] is not None:
                p = from_ratio(status[key])
                if p is not None:
                    return p
        cur = status.get("current_step") or status.get("step")
        tot = status.get("total_steps") or status.get("steps")
        if cur is not None and tot:
            try:
                return max(0, min(100, int(100 * float(cur) / float(tot))))
            except (TypeError, ValueError, ZeroDivisionError):
                pass
    elif isinstance(status, str):
        # e.g. "45%" or "step 3/10"
        import re as _re
        m = _re.search(r"(\d{1,3})\s*%", status)
        if m:
            return max(0, min(100, int(m.group(1))))
        m = _re.search(r"(\d+)\s*/\s*(\d+)", status)
        if m:
            try:
                return max(0, min(100, int(100 * int(m.group(1)) / int(m.group(2)))))
            except ZeroDivisionError:
                pass

    # step fields at top level
    cur = st.get("current_step") or st.get("step")
    tot = st.get("total_steps") or st.get("num_inference_steps") or st.get("steps")
    if cur is not None and tot:
        try:
            return max(0, min(100, int(100 * float(cur) / float(tot))))
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    # events list (newest last)
    events = st.get("events") or st.get("event_log") or []
    if isinstance(events, list):
        for ev in reversed(events):
            if not isinstance(ev, dict):
                continue
            data = ev.get("data") if isinstance(ev.get("data"), dict) else ev
            if not isinstance(data, dict):
                continue
            for key in ("progress", "percent", "percentage"):
                if data.get(key) is not None:
                    p = from_ratio(data[key])
                    if p is not None:
                        return p
            cur = data.get("current_step") or data.get("step")
            tot = data.get("total_steps") or data.get("steps")
            if cur is not None and tot:
                try:
                    return max(0, min(100, int(100 * float(cur) / float(tot))))
                except (TypeError, ValueError, ZeroDivisionError):
                    pass

    phase = st.get("phase")
    if isinstance(phase, (int, float)) and st.get("total_phases"):
        try:
            return max(0, min(100, int(100 * float(phase) / float(st["total_phases"]))))
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    return None


def _run_mcp_generate(job_id: str, job: dict, settings_cfg: dict) -> bool:
    mcp_url = (settings_cfg.get("wan2gp_mcp_url") or "").strip()
    if not mcp_url:
        db.update_job(job_id, {"error": "wan2gp_mcp_url is empty — set Admin → Wan2GP Server → MCP base URL"})
        return False

    try:
        mcp_ensure_session(mcp_url)
    except Exception as e:
        logger.exception("MCP initialize failed")
        db.update_job(job_id, {
            "error": (
                f"Cannot reach MCP at {_mcp_endpoint(mcp_url)}: {e}. "
                "Use trailing /mcp/ and ensure the GenAI host can reach that IP:port."
            )[:800]
        })
        return False

    try:
        source = _prepare_mcp_source(mcp_url, job, settings_cfg)
    except Exception as e:
        logger.exception("MCP media upload failed")
        db.update_job(job_id, {"error": f"MCP media upload: {e}"[:800]})
        return False

    logger.info("MCP wangp_generate → %s model=%s", _mcp_endpoint(mcp_url), source.get("model_type"))

    try:
        out = mcp_call_tool(
            mcp_url,
            "wangp_generate",
            {"source": source, "wait": False, "event_limit": 20},
            timeout=90.0,
        )
    except Exception as e:
        logger.exception("wangp_generate failed")
        db.update_job(job_id, {"error": f"wangp_generate: {e}"[:800]})
        return False

    remote_id = None
    if isinstance(out, dict):
        remote_id = out.get("job_id") or out.get("id")
        if not remote_id and isinstance(out.get("job"), dict):
            remote_id = out["job"].get("id") or out["job"].get("job_id")
        if not remote_id and (out.get("generated_files") is not None or out.get("success") is not None):
            return _apply_mcp_result(job_id, out, mcp_url, gradio_url=(settings_cfg.get('wan2gp_url') or '').strip())

    if not remote_id:
        db.update_job(job_id, {"error": f"wangp_generate no job_id: {str(out)[:500]}"})
        return False

    db.update_job(job_id, {
        "progress": 20,
        "status": "processing",
        "params": {**(job.get("params") or {}), "mcp_job_id": remote_id},
    })

    deadline = time.time() + float(settings_cfg.get("mcp_timeout_s") or 3600)
    last_err = None
    while time.time() < deadline:
        cur = db.get_job(job_id)
        if cur and cur.get("status") in ("cancelled", "canceled", "failed"):
            logger.info("Job %s stopped locally (%s)", job_id, cur.get("status"))
            return False
        try:
            st = mcp_call_tool(
                mcp_url,
                "wangp_get_job",
                {"job_id": remote_id, "event_limit": 20},
                timeout=30.0,
            )
        except Exception as e:
            last_err = str(e)
            time.sleep(2)
            continue

        if not isinstance(st, dict):
            time.sleep(2)
            continue

        pct = _extract_progress_pct(st)
        if pct is not None:
            db.update_job(job_id, {"progress": max(5, min(95, int(pct)))})

        done = st.get("done") or st.get("finished") or st.get("complete")
        status = str(st.get("status") or st.get("state") or "").lower()
        result = st.get("result") or st.get("generation_result")

        if done or status in ("completed", "success", "failed", "error", "cancelled", "canceled"):
            if result is None:
                result = st
            payload = result if isinstance(result, dict) else st
            if isinstance(payload, dict):
                payload = _merge_gallery_from_job_state(dict(payload), st if isinstance(st, dict) else {})
            return _apply_mcp_result(job_id, payload if isinstance(payload, dict) else st, mcp_url, gradio_url=(settings_cfg.get('wan2gp_url') or '').strip())

        time.sleep(2.0)

    db.update_job(job_id, {
        "error": f"MCP job timed out" + (f" (last: {last_err})" if last_err else "")
    })
    return False




# job_type → MCP list filters (main_output / inputs)
_JOB_TYPE_FILTERS = {
    "t2v": {"main_output": "video"},
    "i2v": {"main_output": "video", "inputs": "image"},
    "ia2v": {"main_output": "video", "inputs": "image"},  # audio-capable video models
    "v2v": {"main_output": "video", "inputs": "video"},
    "t2i": {"main_output": "image"},
    "i2i": {"main_output": "image", "inputs": "image"},
}


def _normalize_model_list(raw: Any) -> list[dict]:
    """Accept many MCP payload shapes and return a list of model dicts."""
    if raw is None:
        return []
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = (
            raw.get("models")
            or raw.get("items")
            or raw.get("data")
            or raw.get("result")
            or raw.get("model_defs")
            or raw.get("metadata")
        )
        if items is None and raw.get("model_type"):
            items = [raw]
        if isinstance(items, dict):
            # sometimes keyed by model_type
            items = list(items.values())
        if not isinstance(items, list):
            # text blob?
            text = raw.get("text")
            if isinstance(text, str) and text.strip().startswith(("[", "{")):
                import json as _json
                try:
                    parsed = _json.loads(text)
                    return _normalize_model_list(parsed)
                except Exception:
                    return []
            return []
    elif isinstance(raw, str):
        import json as _json
        try:
            return _normalize_model_list(_json.loads(raw))
        except Exception:
            return []
    else:
        return []

    out = []
    for m in items:
        if not isinstance(m, dict):
            continue
        meta = m.get("metadata") if isinstance(m.get("metadata"), dict) else {}
        model_type = (
            m.get("model_type")
            or meta.get("model_type")
            or m.get("id")
            or m.get("type")
            or ""
        )
        if not model_type:
            continue
        name = m.get("name") or meta.get("name") or meta.get("family_label") or model_type
        main_out = meta.get("main_output") or m.get("main_output") or []
        inputs = meta.get("inputs") or m.get("inputs") or []
        if isinstance(main_out, str):
            main_out = [main_out]
        if isinstance(inputs, str):
            inputs = [inputs]
        family = meta.get("family") or m.get("family") or ""
        out.append({
            "model_type": str(model_type),
            "name": str(name),
            "family": str(family),
            "main_output": list(main_out) if main_out else [],
            "inputs": list(inputs) if inputs else [],
        })
    return out


def _filter_models_for_job_type(models: list[dict], job_type: str) -> list[dict]:
    """Soft filter — never return empty if we had candidates."""
    if not models:
        return []

    def score(m: dict) -> int:
        mo = [str(x).lower() for x in (m.get("main_output") or [])]
        inp = [str(x).lower() for x in (m.get("inputs") or [])]
        blob = f"{m.get('model_type','')} {m.get('name','')} {m.get('family','')}".lower()
        s = 0
        if job_type in ("t2i", "i2i"):
            if "image" in mo:
                s += 3
            if "video" in mo and "image" not in mo:
                s -= 2
            if "t2i" in blob or "text2image" in blob or "text-to-image" in blob or "flux" in blob or "qwen" in blob:
                s += 1
            if job_type == "i2i":
                if "image" in inp:
                    s += 2
                if "i2i" in blob or "img2img" in blob or "edit" in blob:
                    s += 1
        elif job_type in ("t2v", "i2v", "ia2v", "v2v"):
            if "video" in mo:
                s += 3
            if "image" in mo and "video" not in mo:
                s -= 2
            if job_type == "i2v" and ("image" in inp or "i2v" in blob):
                s += 2
            if job_type == "v2v" and ("video" in inp or "v2v" in blob):
                s += 2
            if job_type == "t2v" and ("t2v" in blob or "text" in inp):
                s += 1
            if job_type == "ia2v" and ("audio" in inp or "s2v" in blob or "talk" in blob):
                s += 2
        return s

    ranked = sorted(models, key=score, reverse=True)
    positive = [m for m in ranked if score(m) > 0]
    return positive if positive else ranked


def list_models_for_job_type(mcp_url: str, job_type: str, limit: int = 120) -> list[dict]:
    """
    Load models from MCP. Tries filtered wangp_list_models, then unfiltered,
    then wangp_list_model_defs.
    """
    attempts: list[tuple[str, dict]] = []
    base_filter = dict(_JOB_TYPE_FILTERS.get(job_type) or {})
    if base_filter:
        attempts.append(("wangp_list_models", {**base_filter, "limit": limit}))
    attempts.append(("wangp_list_models", {"limit": limit}))
    # broader discovery
    attempts.append(("wangp_list_model_defs", {**base_filter, "limit": limit} if base_filter else {"limit": limit}))
    attempts.append(("wangp_list_model_defs", {"limit": limit}))

    errors = []
    collected: list[dict] = []
    seen = set()

    for tool, args in attempts:
        try:
            raw = mcp_call_tool(mcp_url, tool, args, timeout=60.0)
            batch = _normalize_model_list(raw)
            for m in batch:
                mt = m["model_type"]
                if mt not in seen:
                    seen.add(mt)
                    collected.append(m)
            if collected:
                break
        except Exception as e:
            errors.append(f"{tool}: {e}")
            logger.warning("list models %s failed: %s", tool, e)

    if not collected:
        raise RuntimeError(
            "No models returned from MCP. "
            + ("; ".join(errors) if errors else "Empty list.")
        )

    filtered = _filter_models_for_job_type(collected, job_type)
    return filtered[:limit]


# Optional callback set by main: callable(list_of_job_ids_to_start)
_on_job_finished = None


async def process_job(job_id: str) -> None:
    """Always MCP when configured. Never mock. Failures store error on the job."""
    job = db.get_job(job_id)
    if not job:
        return

    db.update_job(job_id, {"status": "processing", "progress": 5, "error": None, "backend": BACKEND_ID})
    settings_cfg = db.get_settings()
    mcp_url = (settings_cfg.get("wan2gp_mcp_url") or "").strip()
    enabled = bool(settings_cfg.get("wan2gp_enabled", False))

    try:
        if not enabled:
            db.update_job(job_id, {
                "status": "failed",
                "progress": 0,
                "error": "Wan2GP disabled. Enable in Admin → Wan2GP Server and set MCP URL ending with /mcp/",
            })
            return

        if not mcp_url:
            db.update_job(job_id, {
                "status": "failed",
                "progress": 0,
                "error": "MCP URL empty. Set e.g. http://YOUR_HOST:8080/mcp/ in Admin → Wan2GP Server.",
            })
            return

        ok = await asyncio.to_thread(_run_mcp_generate, job_id, job, settings_cfg)
        if ok:
            return
        cur = db.get_job(job_id) or {}
        if cur.get("status") in ("cancelled", "canceled"):
            return
        err = (cur.get("error") or "").strip() or "MCP generation failed"
        db.update_job(job_id, {"status": "failed", "progress": 0, "error": err[:800]})
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        db.update_job(job_id, {"status": "failed", "progress": 0, "error": str(e)[:800]})
    finally:
        # Kick queue after this job leaves processing
        try:
            next_ids = try_start_queued_jobs()
            if next_ids and _on_job_finished:
                _on_job_finished(next_ids)
        except Exception:
            logger.exception("queue kick failed")



async def test_wan2gp_connection(url: str = "", root: str = "", mcp_url: str = "") -> dict:
    """Validate real MCP handshake + wangp_list_models (not Gradio, not bare HTTP)."""
    mcp_url = (mcp_url or "").strip()
    if not mcp_url:
        return {
            "ok": False,
            "message": "Set MCP base URL (http://HOST:PORT/mcp/) — Gradio URL alone is not enough.",
            "version": None,
        }

    endpoint = _mcp_endpoint(mcp_url)
    try:
        _mcp_sessions.pop(endpoint, None)
        sid = await asyncio.to_thread(mcp_ensure_session, mcp_url, 20.0)
        out = await asyncio.to_thread(
            mcp_call_tool, mcp_url, "wangp_list_models", {"limit": 5}, 30.0
        )
        n = len(out) if isinstance(out, list) else "?"
        return {
            "ok": True,
            "message": f"MCP OK {endpoint} session={bool(sid)} wangp_list_models={n}",
            "version": "mcp-2025-03-26",
            "endpoint": endpoint,
        }
    except Exception as e:
        return {
            "ok": False,
            "message": (
                f"MCP failed at {endpoint}: {e}. "
                "Confirm trailing /mcp/, protocol 2025-03-26, and network from this host."
            ),
            "version": None,
            "endpoint": endpoint,
        }



def try_start_queued_jobs() -> list[str]:
    """Start queued jobs up to concurrency limits. Returns started job ids."""
    settings = db.get_settings()
    max_c = max(1, int(settings.get("max_concurrent_jobs") or 1))
    scope = (settings.get("concurrent_scope") or "overall").lower()
    started = []
    for job in db.list_queued_jobs(limit=100):
        uid = job.get("user_id") or ""
        if scope == "per_user":
            active = db.count_active_jobs(uid)
        else:
            active = db.count_active_jobs(None)
        if active >= max_c:
            if scope == "per_user":
                continue  # try another user's job
            break
        jid = job["id"]
        # claim: only if still queued
        cur = db.get_job(jid)
        if not cur or cur.get("status") != "queued":
            continue
        db.update_job(jid, {"status": "processing", "progress": 1, "error": None})
        started.append(jid)
        # fire async via asyncio if loop running — caller uses BackgroundTasks
    return started
