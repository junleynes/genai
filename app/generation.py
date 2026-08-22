"""
WanForge job runner — remote WanGP via MCP Streamable HTTP.

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

logger = logging.getLogger("wanforge.generation")

# Deployed-code fingerprint — GET /api/version must show this
BACKEND_ID = "mcp-streamable-http-v2-no-mock"
BACKEND_BUILT = "2026-08-23-file-transfer"

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
            "clientInfo": {"name": "WanForge", "version": "1.1.0"},
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


def mcp_download_gallery_item(mcp_url: str, gallery_id: str, job_id: str) -> str:
    """Use wangp_create_gallery_download then GET the short-lived URL onto this server."""
    create = mcp_call_tool(
        mcp_url,
        "wangp_create_gallery_download",
        {"gallery_id": gallery_id},
        timeout=60.0,
    )
    if not isinstance(create, dict):
        raise RuntimeError(f"gallery download create failed: {create}")

    url = (
        create.get("url")
        or create.get("download_url")
        or create.get("get_url")
        or create.get("href")
    )
    if not url:
        raise RuntimeError(f"No download URL from wangp_create_gallery_download: {create}")

    if str(url).startswith("/"):
        url = _mcp_origin(mcp_url) + str(url)

    with httpx.Client(timeout=600.0, follow_redirects=True) as client:
        r = client.get(str(url))
        if r.status_code >= 400:
            raise RuntimeError(f"Gallery download HTTP {r.status_code}: {r.text[:200]}")
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
        # filename from content-disposition
        cd = r.headers.get("content-disposition") or ""
        if "filename=" in cd:
            import re
            m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd, re.I)
            if m:
                name = m.group(1).strip()
                if "." in name:
                    ext = "." + name.rsplit(".", 1)[-1].lower()[:8]
        return _save_bytes_result(job_id, r.content, ext)


def mcp_download_remote_file(mcp_url: str, remote_path: str, job_id: str, gradio_url: str = "") -> str:
    """
    Remote WanGP paths are not on this disk. Try:
      1) direct http(s) URL
      2) local path (shared disk)
      3) Gradio /file= and /gradio_api/file= (same host or wan2gp_url)
      4) MCP origin heuristics
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

    name = Path(remote_path.replace("\\", "/")).name
    # Windows path → use as Gradio file= parameter (Gradio accepts absolute paths on server)
    # Encode carefully: Gradio often wants the path as-is after file=
    from urllib.parse import quote

    origins = []
    if gradio_url:
        origins.append(gradio_url.rstrip("/"))
    origin = _mcp_origin(mcp_url)
    if origin:
        origins.append(origin)
        # common: MCP on 8080, Gradio on 7860 same host
        try:
            from urllib.parse import urlparse
            u = urlparse(origin)
            if u.hostname:
                origins.append(f"{u.scheme}://{u.hostname}:7860")
                origins.append(f"{u.scheme}://{u.hostname}:7861")
        except Exception:
            pass

    # unique preserve order
    seen = set()
    uniq_origins = []
    for o in origins:
        if o and o not in seen:
            seen.add(o)
            uniq_origins.append(o)

    path_variants = [
        remote_path,
        remote_path.replace("\\", "/"),
        # Gradio on Windows sometimes wants forward slashes
        remote_path.replace("\\", "/"),
    ]
    # Also try just filename under outputs
    path_variants.append(name)

    candidates = []
    for o in uniq_origins:
        for pv in path_variants:
            # Gradio classic
            candidates.append(f"{o}/file={pv}")
            candidates.append(f"{o}/file={quote(pv, safe=':/\\\\')}")
            candidates.append(f"{o}/gradio_api/file={pv}")
            candidates.append(f"{o}/gradio_api/file={quote(pv, safe=':/\\\\')}")
        candidates.append(f"{o}/media/{name}")
        candidates.append(f"{o}/outputs/{name}")
        candidates.append(f"{o}/static/outputs/{name}")

    last_err = None
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for u in candidates:
            try:
                r = client.get(u)
                if r.status_code == 200 and r.content and len(r.content) > 100:
                    # skip HTML error pages
                    ct = (r.headers.get("content-type") or "").lower()
                    if "text/html" in ct and b"<html" in r.content[:200].lower():
                        continue
                    ext = Path(name).suffix or ".bin"
                    return _save_bytes_result(job_id, r.content, ext)
                last_err = f"HTTP {r.status_code} for {u[:80]}"
            except Exception as e:
                last_err = str(e)

    raise RuntimeError(
        f"Cannot fetch remote file {remote_path!r} onto WanForge host "
        f"(not local, gallery download needed). last={last_err}"
    )



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
            if remote_path:
                try:
                    url = mcp_download_remote_file(mcp_url, str(remote_path), job_id, gradio_url=gradio_url)
                    break
                except Exception as e:
                    transfer_errors.append(str(e))

    # 4) Any media paths nested in result
    if not url:
        for path in _walk_file_paths(result):
            try:
                url = mcp_download_remote_file(mcp_url, path, job_id, gradio_url=gradio_url)
                break
            except Exception as e:
                transfer_errors.append(str(e))

    if not url:
        msg = (
            "Generation finished on remote WanGP, but the file could not be transferred to this app. "
            "Set Gradio URL in Admin (e.g. http://WANGP_HOST:7860) so files can be fetched via /file=, "
            "or ensure MCP gallery download works. "
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
                "Use trailing /mcp/ and ensure the WanForge host can reach that IP:port."
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

        prog = st.get("progress")
        if prog is None and isinstance(st.get("status"), dict):
            prog = st["status"].get("progress")
        if isinstance(prog, (int, float)):
            pct = int(prog * 100) if float(prog) <= 1 else int(prog)
            db.update_job(job_id, {"progress": max(20, min(95, pct))})

        done = st.get("done") or st.get("finished") or st.get("complete")
        status = str(st.get("status") or st.get("state") or "").lower()
        result = st.get("result") or st.get("generation_result")

        if done or status in ("completed", "success", "failed", "error", "cancelled", "canceled"):
            if result is None:
                result = st
            return _apply_mcp_result(job_id, result if isinstance(result, dict) else st, mcp_url, gradio_url=(settings_cfg.get('wan2gp_url') or '').strip())

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
        err = (cur.get("error") or "").strip() or "MCP generation failed"
        db.update_job(job_id, {"status": "failed", "progress": 0, "error": err[:800]})
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        db.update_job(job_id, {"status": "failed", "progress": 0, "error": str(e)[:800]})


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
