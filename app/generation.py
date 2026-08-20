"""
Job processing via the official WanGP Python API (shared.api).

Docs: https://github.com/deepbeepmeep/Wan2GP/blob/main/docs/API.md
Settings: https://github.com/deepbeepmeep/Wan2GP/blob/main/docs/SETTINGS.md

Preferred path (same machine as WanGP install):
  from shared.api import init
  session = init(root=Path(...))
  job = session.submit_task({...settings...})
  result = job.result()  # result.generated_files

Gradio HTTP /predict is NOT the documented integration path for WanGP.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

from . import db

logger = logging.getLogger("wanforge.generation")

UPLOAD_DIR = Path(__file__).parent.parent / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(__file__).parent.parent / "static" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_session = None
_session_root: Optional[str] = None


def _duration_to_frames(seconds: float, fps: int = 24) -> int:
    n = max(1, int(round(seconds * fps)))
    return max(5, (n // 4) * 4 + 1)


def _map_job_to_settings(job: dict, defaults: dict) -> dict:
    params = job.get("params") or {}
    jtype = job.get("job_type", "t2v")
    prompt = job.get("prompt") or ""

    model_type = (
        params.get("model_type")
        or defaults.get("model_type")
        or "ltx2_22B_distilled"
    )

    resolution = params.get("resolution") or defaults.get("resolution") or "1280x704"
    steps = int(params.get("steps") or params.get("num_inference_steps") or defaults.get("num_inference_steps") or 8)
    seed = int(params.get("seed", -1))
    duration = float(params.get("duration") or params.get("duration_seconds") or 4)
    video_length = params.get("video_length")
    if video_length is None:
        video_length = _duration_to_frames(duration)

    settings: dict[str, Any] = {
        "model_type": model_type,
        "prompt": prompt,
        "negative_prompt": params.get("negative_prompt") or "",
        "resolution": resolution,
        "num_inference_steps": steps,
        "seed": seed,
        "video_length": int(video_length),
        "duration_seconds": duration,
        "force_fps": str(params.get("force_fps") or defaults.get("force_fps") or "24"),
        "guidance_scale": float(params.get("guidance_scale") or params.get("cfg") or 5.0),
        "flow_shift": float(params.get("flow_shift") or 5.0),
        "_api": {"return_media": True},
    }

    image_path = job.get("image_path") or params.get("image_path")
    image_end = job.get("image_end_path") or params.get("image_end_path")
    video_path = job.get("video_path") or params.get("video_path")
    audio_path = job.get("audio_path") or params.get("audio_path")

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
        settings["audio_prompt_type"] = settings.get("audio_prompt_type") or "A"

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


def _get_session(root: str, cli_args: Optional[list] = None):
    global _session, _session_root
    root = str(Path(root).resolve())
    if _session is not None and _session_root == root:
        return _session

    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"WanGP root not found: {root}")

    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))

    from shared.api import init  # type: ignore

    args = cli_args or ["--attention", "sdpa", "--profile", "4"]
    _session = init(
        root=root_path,
        cli_args=args,
        console_output=False,
    )
    _session_root = root
    logger.info("WanGP session initialized at %s", root)
    return _session


def _copy_result_into_static(src: str | Path, job_id: str) -> str:
    src = Path(src)
    if not src.exists():
        return str(src)
    ext = src.suffix or ".mp4"
    dest = RESULTS_DIR / f"{job_id}{ext}"
    shutil.copy2(src, dest)
    return f"/static/results/{dest.name}"



# ---------------------------------------------------------------------------
# MCP Streamable HTTP client (API.md: connect to http://host:port/mcp)
# Tools: wangp_generate, wangp_get_job, wangp_cancel_job, wangp_list_models, ...
# ---------------------------------------------------------------------------

_mcp_req_id = 0


def _next_mcp_id() -> int:
    global _mcp_req_id
    _mcp_req_id += 1
    return _mcp_req_id


def _mcp_endpoint(base: str) -> str:
    base = (base or "").rstrip("/")
    if base.endswith("/mcp"):
        return base
    return base + "/mcp"


def _parse_mcp_tool_result(data: dict) -> Any:
    """Extract structured / text payload from MCP tools/call response."""
    if not isinstance(data, dict):
        return data
    # JSON-RPC envelope
    if "result" in data:
        result = data["result"]
    elif "error" in data:
        err = data["error"]
        raise RuntimeError(err.get("message") if isinstance(err, dict) else str(err))
    else:
        result = data

    if not isinstance(result, dict):
        return result

    # structuredContent preferred
    if result.get("structuredContent") is not None:
        return result["structuredContent"]

    # content[] text items (often JSON string)
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
            import json as _json
            try:
                return _json.loads(joined)
            except Exception:
                return {"text": joined}
    return result


def mcp_call_tool(mcp_url: str, name: str, arguments: dict, timeout: float = 120.0) -> Any:
    """POST JSON-RPC tools/call to Streamable HTTP MCP endpoint."""
    import httpx
    import json as _json

    endpoint = _mcp_endpoint(mcp_url)
    payload = {
        "jsonrpc": "2.0",
        "id": _next_mcp_id(),
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": arguments or {},
        },
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.post(endpoint, json=payload, headers=headers)
        ctype = (r.headers.get("content-type") or "").lower()
        body = r.text
        if r.status_code >= 400:
            raise RuntimeError(f"MCP HTTP {r.status_code}: {body[:400]}")

        if "text/event-stream" in ctype:
            # Take last data: JSON line from SSE
            last = None
            for line in body.splitlines():
                if line.startswith("data:"):
                    raw = line[5:].strip()
                    if raw and raw != "[DONE]":
                        try:
                            last = _json.loads(raw)
                        except Exception:
                            pass
            if last is None:
                raise RuntimeError(f"MCP SSE empty: {body[:300]}")
            return _parse_mcp_tool_result(last)

        try:
            data = r.json()
        except Exception:
            raise RuntimeError(f"MCP non-JSON: {body[:300]}")
        return _parse_mcp_tool_result(data)


def _run_mcp_generate(job_id: str, job: dict, settings_cfg: dict) -> bool:
    """
    wangp_generate(source=settings, wait=False) then poll wangp_get_job.
    """
    mcp_url = (settings_cfg.get("wan2gp_mcp_url") or "").strip()
    if not mcp_url:
        return False

    defaults = {
        "model_type": settings_cfg.get("default_model_type") or "ltx2_22B_distilled",
        "resolution": settings_cfg.get("default_resolution") or "1280x704",
        "num_inference_steps": settings_cfg.get("default_steps") or 8,
        "force_fps": settings_cfg.get("default_fps") or "24",
    }
    source = _map_job_to_settings(job, defaults)
    logger.info("MCP wangp_generate source keys=%s", list(source.keys()))

    try:
        out = mcp_call_tool(
            mcp_url,
            "wangp_generate",
            {"source": source, "wait": False, "event_limit": 20},
            timeout=60.0,
        )
    except Exception as e:
        logger.exception("wangp_generate failed")
        db.update_job(job_id, {"error": f"wangp_generate: {e}"[:800]})
        return False

    # Expect job_id in response
    remote_id = None
    if isinstance(out, dict):
        remote_id = out.get("job_id") or out.get("id")
        # sometimes nested
        if not remote_id and isinstance(out.get("job"), dict):
            remote_id = out["job"].get("id") or out["job"].get("job_id")
    if not remote_id:
        # Maybe waited and returned result directly
        if isinstance(out, dict) and (out.get("generated_files") or out.get("success") is not None):
            return _apply_mcp_result(job_id, out)
        db.update_job(job_id, {"error": f"wangp_generate no job_id: {str(out)[:400]}"})
        return False

    db.update_job(job_id, {
        "progress": 20,
        "status": "processing",
        "params": {**(job.get("params") or {}), "mcp_job_id": remote_id},
    })

    # Poll
    import time
    deadline = time.time() + float(settings_cfg.get("mcp_timeout_s") or 3600)
    while time.time() < deadline:
        try:
            st = mcp_call_tool(
                mcp_url,
                "wangp_get_job",
                {"job_id": remote_id, "event_limit": 20},
                timeout=30.0,
            )
        except Exception as e:
            logger.warning("wangp_get_job: %s", e)
            time.sleep(2)
            continue

        if not isinstance(st, dict):
            time.sleep(2)
            continue

        # progress
        prog = st.get("progress")
        if prog is None and isinstance(st.get("status"), dict):
            prog = st["status"].get("progress")
        if isinstance(prog, (int, float)):
            pct = int(prog * 100) if prog <= 1 else int(prog)
            db.update_job(job_id, {"progress": max(20, min(95, pct))})

        done = st.get("done") or st.get("finished") or st.get("complete")
        status = str(st.get("status") or st.get("state") or "").lower()
        result = st.get("result") or st.get("generation_result")

        if done or status in ("completed", "success", "failed", "error", "cancelled", "canceled"):
            if result is None:
                result = st
            ok = _apply_mcp_result(job_id, result if isinstance(result, dict) else st)
            return ok

        time.sleep(2.0)

    db.update_job(job_id, {"error": "MCP job timed out"})
    return False


def _apply_mcp_result(job_id: str, result: dict) -> bool:
    success = result.get("success")
    files = result.get("generated_files") or result.get("files") or []
    gallery = result.get("gallery_items") or []
    errors = result.get("errors") or []

    if success is False or (not files and not gallery and errors):
        msgs = []
        for err in errors:
            if isinstance(err, dict):
                msgs.append(err.get("message") or str(err))
            else:
                msgs.append(str(err))
        msg = "; ".join(msgs) or "MCP generation failed"
        db.update_job(job_id, {"error": msg[:800]})
        return False

    path = None
    if files:
        first = files[0]
        path = first.get("path") if isinstance(first, dict) else str(first)
    elif gallery:
        g0 = gallery[0]
        if isinstance(g0, dict):
            path = g0.get("path") or g0.get("url") or g0.get("file")
        else:
            path = str(g0)

    if not path:
        # try nested
        path = result.get("path") or result.get("output")
    if not path:
        db.update_job(job_id, {"error": f"No output files in MCP result: {str(result)[:400]}"})
        return False

    # Local path → copy into static; URL → use as-is
    if str(path).startswith("http://") or str(path).startswith("https://"):
        url = str(path)
    else:
        url = _copy_result_into_static(path, job_id)

    db.update_job(job_id, {
        "status": "completed",
        "progress": 100,
        "result_url": url,
        "preview_url": url,
        "error": None,
    })
    logger.info("Job %s completed via MCP → %s", job_id, url)
    return True


async def process_job(job_id: str) -> None:
    job = db.get_job(job_id)
    if not job:
        return

    db.update_job(job_id, {"status": "processing", "progress": 5, "error": None})
    settings_cfg = db.get_settings()
    root = (settings_cfg.get("wan2gp_root") or "").strip()
    mcp_url = (settings_cfg.get("wan2gp_mcp_url") or "").strip()
    enabled = bool(settings_cfg.get("wan2gp_enabled", False))
    allow_mock = bool(settings_cfg.get("allow_mock_fallback", True))

    try:
        if enabled and mcp_url:
            ok = await asyncio.to_thread(_run_mcp_generate, job_id, job, settings_cfg)
            if ok:
                return
            if not allow_mock:
                db.update_job(job_id, {
                    "status": "failed",
                    "progress": 0,
                    "error": "MCP wangp_generate failed (mock disabled)",
                })
                return
            logger.warning("MCP generate failed for %s – trying local API / mock", job_id)

        if enabled and root:
            ok = await asyncio.to_thread(_run_wangp_api, job_id, job, settings_cfg)
            if ok:
                return
            if not allow_mock:
                db.update_job(job_id, {
                    "status": "failed",
                    "progress": 0,
                    "error": "WanGP API call failed (mock fallback disabled)",
                })
                return
            logger.warning("WanGP API failed for %s – mock fallback", job_id)

        if enabled and not mcp_url and not root:
            err = "Wan2GP enabled but neither wan2gp_mcp_url nor wan2gp_root is set."
            if not allow_mock:
                db.update_job(job_id, {"status": "failed", "progress": 0, "error": err})
                return
            logger.warning(err)

        await _mock_generation(job_id, job)
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        db.update_job(job_id, {
            "status": "failed",
            "progress": 0,
            "error": str(e)[:800],
        })


def _run_wangp_api(job_id: str, job: dict, settings_cfg: dict) -> bool:
    """Official API.md: session.submit_task(settings) -> job.result()."""
    root = settings_cfg.get("wan2gp_root", "").strip()
    cli_raw = settings_cfg.get("wan2gp_cli_args") or "--attention sdpa --profile 4"
    cli_args = [a for a in str(cli_raw).split() if a]

    defaults = {
        "model_type": settings_cfg.get("default_model_type") or "ltx2_22B_distilled",
        "resolution": settings_cfg.get("default_resolution") or "1280x704",
        "num_inference_steps": settings_cfg.get("default_steps") or 8,
        "force_fps": settings_cfg.get("default_fps") or "24",
    }
    settings = _map_job_to_settings(job, defaults)
    logger.info("WanGP submit_task: %s", {k: v for k, v in settings.items() if k != "_api"})

    try:
        session = _get_session(root, cli_args)
    except Exception as e:
        logger.exception("Failed to init WanGP session")
        db.update_job(job_id, {"error": f"init failed: {e}"[:500]})
        return False

    db.update_job(job_id, {"progress": 15, "status": "processing"})

    try:
        wjob = session.submit_task(settings)
        try:
            for event in wjob.events.iter(timeout=0.5):
                if event.kind == "progress":
                    p = event.data
                    ratio = float(getattr(p, "progress", 0) or 0)
                    if ratio <= 1.0:
                        ratio *= 100
                    pct = max(15, min(95, int(ratio)))
                    db.update_job(job_id, {"progress": pct, "status": "processing"})
                if getattr(wjob, "done", False):
                    break
        except Exception as ev_err:
            logger.debug("event iter: %s", ev_err)

        result = wjob.result()
        if not getattr(result, "success", False):
            errors = getattr(result, "errors", None) or []
            msgs = [getattr(err, "message", None) or str(err) for err in errors]
            msg = "; ".join(msgs) or "WanGP generation failed"
            logger.warning("WanGP result failed: %s", msg)
            db.update_job(job_id, {"error": msg[:800]})
            return False

        files = list(getattr(result, "generated_files", None) or [])
        if not files:
            db.update_job(job_id, {"error": "WanGP returned no generated_files"})
            return False

        first = files[0]
        if hasattr(first, "path"):
            first = first.path
        url = _copy_result_into_static(str(first), job_id)

        db.update_job(job_id, {
            "status": "completed",
            "progress": 100,
            "result_url": url,
            "preview_url": url,
            "error": None,
        })
        logger.info("Job %s completed via WanGP API → %s", job_id, url)
        return True

    except Exception as e:
        logger.exception("WanGP submit_task error")
        db.update_job(job_id, {"error": f"submit_task: {e}"[:800]})
        return False


async def _mock_generation(job_id: str, job: dict) -> None:
    steps = [15, 30, 50, 70, 85, 95, 100]
    for p in steps:
        await asyncio.sleep(1.0)
        current = db.get_job(job_id)
        if not current or current["status"] == "failed":
            return
        db.update_job(job_id, {"progress": p})

    jtype = job.get("job_type", "t2v")
    if jtype in ("t2i", "i2i"):
        result = "https://placehold.co/1024x1024/6366f1/ffffff?text=Generated+Image+(mock)"
    elif jtype == "ia2v":
        result = "https://placehold.co/1280x720/8b5cf6/ffffff?text=Image%2BAudio+Video+(mock)"
    elif jtype == "v2v":
        result = "https://placehold.co/1280x720/22d3ee/ffffff?text=Video+Video+(mock)"
    else:
        result = "https://placehold.co/1280x720/6366f1/ffffff?text=Generated+Video+(mock)"

    db.update_job(job_id, {
        "status": "completed",
        "progress": 100,
        "result_url": result,
        "preview_url": result,
        "error": "mock (set wan2gp_root to your WanGP folder for real generation)",
    })


async def test_wan2gp_connection(url: str = "", root: str = "", mcp_url: str = "") -> dict:
    root = (root or "").strip()
    url = (url or "").rstrip("/")
    mcp_url = (mcp_url or "").strip()

    if mcp_url:
        try:
            # tools/list or a lightweight generate discovery
            out = await asyncio.to_thread(
                mcp_call_tool, mcp_url, "wangp_list_models", {"limit": 1}, 15.0
            )
            return {
                "ok": True,
                "message": f"MCP OK at {_mcp_endpoint(mcp_url)} (wangp_list_models worked)",
                "version": "mcp",
                "sample": str(out)[:200],
            }
        except Exception as e:
            # try initialize-style fallback: just POST invalid to see reachability
            try:
                import httpx
                async with httpx.AsyncClient(timeout=8.0) as client:
                    r = await client.post(
                        _mcp_endpoint(mcp_url),
                        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                        headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
                    )
                    if r.status_code < 500:
                        return {
                            "ok": True,
                            "message": f"MCP endpoint reachable (HTTP {r.status_code}). tools/list may need session. Detail: {e}",
                            "version": "mcp-http",
                        }
            except Exception as e2:
                return {"ok": False, "message": f"MCP failed: {e}; reachability: {e2}", "version": None}
            return {"ok": False, "message": str(e)[:300], "version": None}


    if root:
        try:
            p = Path(root)
            if not p.is_dir():
                return {"ok": False, "message": f"Path not found: {root}", "version": None}
            if not (p / "shared" / "api.py").exists():
                return {
                    "ok": False,
                    "message": f"shared/api.py not found under {root}",
                    "version": None,
                }
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            import shared.api  # noqa: F401
            return {
                "ok": True,
                "message": "WanGP install found. Real generate uses session.submit_task.",
                "version": "python-api",
            }
        except Exception as e:
            return {"ok": False, "message": str(e)[:300], "version": None}

    if url:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=8.0) as client:
                r = await client.get(f"{url}/config")
                if r.status_code == 200:
                    return {
                        "ok": True,
                        "message": "Gradio reachable – set WanGP install PATH for real submit_task generation.",
                        "version": "gradio",
                    }
                r2 = await client.get(url)
                if r2.status_code < 500:
                    return {
                        "ok": True,
                        "message": "Server reachable – set wan2gp_root for real generation.",
                        "version": None,
                    }
                return {"ok": False, "message": f"HTTP {r2.status_code}", "version": None}
        except Exception as e:
            return {"ok": False, "message": str(e)[:200], "version": None}

    return {"ok": False, "message": "Provide WanGP install path (recommended) or Gradio URL", "version": None}
