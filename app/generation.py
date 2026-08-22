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


def _apply_mcp_result(job_id: str, result: dict) -> bool:
    if not isinstance(result, dict):
        db.update_job(job_id, {"error": f"Bad MCP result type: {type(result)}"})
        return False

    success = result.get("success")
    files = result.get("generated_files") or result.get("files") or []
    gallery = result.get("gallery_items") or []
    errors = result.get("errors") or []

    if success is False or (not files and not gallery and errors):
        msgs = []
        for err in errors:
            msgs.append(err.get("message") if isinstance(err, dict) else str(err))
        db.update_job(job_id, {"error": ("; ".join(msgs) or "MCP generation failed")[:800]})
        return False

    path = None
    if files:
        first = files[0]
        path = first.get("path") if isinstance(first, dict) else str(first)
    elif gallery:
        g0 = gallery[0]
        path = (g0.get("path") or g0.get("url") or g0.get("file")) if isinstance(g0, dict) else str(g0)
    path = path or result.get("path") or result.get("output")
    if not path:
        db.update_job(job_id, {"error": f"No output in MCP result: {str(result)[:500]}"})
        return False

    if str(path).startswith(("http://", "https://")):
        url = str(path)
    else:
        try:
            url = _copy_result_into_static(path, job_id)
        except Exception:
            url = str(path)

    db.update_job(job_id, {
        "status": "completed",
        "progress": 100,
        "result_url": url,
        "preview_url": url,
        "error": None,
    })
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
            return _apply_mcp_result(job_id, out)

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
            return _apply_mcp_result(job_id, result if isinstance(result, dict) else st)

        time.sleep(2.0)

    db.update_job(job_id, {
        "error": f"MCP job timed out" + (f" (last: {last_err})" if last_err else "")
    })
    return False


async def process_job(job_id: str) -> None:
    """Always MCP when configured. Never mock. Failures store error on the job."""
    job = db.get_job(job_id)
    if not job:
        return

    db.update_job(job_id, {"status": "processing", "progress": 5, "error": None})
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
