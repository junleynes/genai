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
from urllib.parse import quote

import httpx

from . import db

logger = logging.getLogger("genai.generation")

# Deployed-code fingerprint — GET /api/version must show this
BACKEND_ID = "mcp-streamable-http-v2-no-mock"
BACKEND_BUILT = "2026-08-23-mcp-inline-media-transfer"

UPLOAD_DIR = Path(__file__).parent.parent / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(__file__).parent.parent / "static" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_mcp_req_id = 0
_mcp_sessions: dict[str, str] = {}  # endpoint -> session id


def _duration_to_frames(seconds: float, fps: int = 24) -> int:
    n = max(1, int(round(float(seconds) * fps)))
    return max(5, (n // 4) * 4 + 1)


def _supports_reference_images(model_type: str, name: str = "", family: str = "") -> bool:
    """
    Does this model consume image_refs?

    Reference images are a per-model feature, not a per-job-type one. VACE
    does reference-to-video; the LTX-2.3 MSR finetune packs 2-5 references
    into a pseudo-video (plain LTX-2 Distilled does NOT — MSR is a separate
    finetune); Qwen Image Edit Plus and Krea 2 Identity Edit do multi-
    reference image editing; Phantom/Animate/Lynx take identity references.

    Everything else accepts image_refs and ignores it, so sending refs
    there produces output that silently disregards them.
    """
    blob = f"{model_type} {name} {family}".lower()
    return any(k in blob for k in (
        "vace",          # reference-to-video + control
        "msr",           # LTX-2.3 Multiple Subject Reference finetune
        "phantom",       # reference conditioning
        "animate",       # Wan 2.2 Animate: reference images + pose
        "lynx",          # identity reference
        "standin",       # identity reference
        "bernini",       # generates from multiple reference images
        "identity",      # Krea 2 Identity Edit (up to 2 refs)
        "edit",          # Qwen Image Edit Plus: multi-reference editing
        "recam",
    ))


def _is_control_capable(model_type: str, name: str = "", family: str = "") -> bool:
    """
    Can this model consume a guide video (pose / depth / canny)?

    Two different mechanisms qualify:
      - VACE and friends implement guide conditioning in the architecture.
      - LTX-2 implements it through IC LoRAs: WanGP restores the LTX 2.3
        LoRA workflows, pose/depth/canny control among them, from the
        shared loras/ltx2 folder. Control still needs the matching IC LoRA
        activated, but the model itself is a valid choice for a guided job.

    Models outside both groups (plain Wan t2v/i2v, Hunyuan, Flux) accept
    the guide parameters and silently ignore them, which looks exactly
    like pose control being broken.
    """
    blob = f"{model_type} {name} {family}".lower()
    if any(k in blob for k in (
        "vace",      # VACE + VACE Lynx: the main control family
        "control",   # generic controlnet-style finetunes
        "fun",       # Wan-Fun control variants
        "animate",   # Wan 2.2 Animate (character animation from a driving video)
        "scail",     # Scail / Scail-2 character animators
        "recam",     # ReCamMaster: re-shoot along a guide
        "phantom",   # Phantom: reference/guide conditioning
        "ltx2",      # LTX-2 family: pose/depth/canny via IC LoRAs
        "ltx-2",
        "ltx_2",
    )):
        return True
    return False


def _needs_control_lora(model_type: str, name: str = "", family: str = "") -> bool:
    """
    LTX-2 guide control is supplied by an IC LoRA rather than being built
    into the checkpoint, so a guided LTX-2 job with no LoRA activated will
    generate but ignore the guide. Used to warn rather than to block.
    """
    blob = f"{model_type} {name} {family}".lower()
    return any(k in blob for k in ("ltx2", "ltx-2", "ltx_2"))


def _pick_control_model(mcp_url: str, requested: str | None = None) -> Optional[str]:
    """
    Choose a guide-capable model for pose/control-driven jobs. Returns
    None if the catalogue has none, so the caller can fail with a clear
    message instead of silently generating unguided video.
    """
    try:
        models = list_models_for_job_type(mcp_url, "p2v", limit=200)
    except Exception:
        logger.exception("could not list models to resolve a control-capable model")
        return None

    capable = [
        m for m in models
        if _is_control_capable(m.get("model_type", ""), m.get("name", ""), m.get("family", ""))
    ]
    if not capable:
        return None

    # Honour an explicit request when it is genuinely capable.
    if requested:
        for m in capable:
            if m.get("model_type") == requested:
                return requested

    # Prefer VACE proper, then other control families. LTX-2 ranks lower
    # only because it additionally needs an IC LoRA activated to work.
    def rank(m: dict) -> int:
        blob = f"{m.get('model_type','')} {m.get('name','')} {m.get('family','')}".lower()
        if "vace" in blob:
            return 4
        if "animate" in blob or "scail" in blob:
            return 3
        if "ltx2" in blob or "ltx-2" in blob or "ltx_2" in blob:
            return 2
        return 1

    best = sorted(capable, key=rank, reverse=True)[0]
    return best.get("model_type")


def _normalize_lora_list(raw: Any) -> list[dict]:
    """Accept the various shapes a LoRA-listing tool might return."""
    items: Any = raw
    if isinstance(raw, dict):
        items = (
            raw.get("loras") or raw.get("items") or raw.get("data")
            or raw.get("result") or raw.get("files") or raw.get("names")
        )
        if items is None:
            text = raw.get("text")
            if isinstance(text, str) and text.strip().startswith(("[", "{")):
                import json as _json
                try:
                    return _normalize_lora_list(_json.loads(text))
                except Exception:
                    return []
            return []
    if isinstance(items, str):
        import json as _json
        try:
            return _normalize_lora_list(_json.loads(items))
        except Exception:
            # newline/comma separated plain list
            parts = [p.strip() for p in items.replace("\n", ",").split(",") if p.strip()]
            return [{"filename": p, "name": p} for p in parts]
    if not isinstance(items, list):
        return []

    out: list[dict] = []
    for it in items:
        if isinstance(it, str):
            out.append({"filename": it, "name": it})
            continue
        if not isinstance(it, dict):
            continue
        fn = (
            it.get("filename") or it.get("file") or it.get("path")
            or it.get("id") or it.get("name") or ""
        )
        if not fn:
            continue
        fn = str(fn).replace("\\", "/").split("/")[-1]
        out.append({
            "filename": fn,
            "name": str(it.get("name") or fn),
            "model_type": str(it.get("model_type") or it.get("model") or ""),
        })
    return out


def list_loras_for_model(mcp_url: str, model_type: str = "") -> tuple[list[dict], bool]:
    """
    LoRAs available for a model. WanGP keeps LoRAs in model-specific
    subdirectories under loras/, so the set is not global — a LoRA built
    for one architecture will not load on another.

    Tool naming varies between WanGP builds (and older builds may expose
    no LoRA tool at all), so discover rather than hardcode. Returns
    (loras, supported); supported=False means this install exposes no
    LoRA listing tool and the UI should fall back to free-text entry.
    """
    candidates = _find_tools(
        mcp_url,
        include=("lora", "loras"),
        exclude=("download", "delete", "remove", "upload", "apply", "activate"),
    )
    if not candidates:
        logger.info("No LoRA-listing tool on this MCP server; free-text entry only")
        return [], False

    arg_sets: list[dict] = []
    if model_type:
        arg_sets.extend([{"model_type": model_type}, {"model": model_type}])
    arg_sets.append({})

    for tool in candidates:
        tname = str(tool.get("name") or "")
        params = set(_tool_param_names(tool))
        for args in arg_sets:
            # Don't send an argument the tool doesn't declare.
            if args and not (set(args) & params):
                continue
            try:
                raw = mcp_call_tool(mcp_url, tname, args, timeout=45.0)
            except Exception as e:
                logger.debug("LoRA tool %s(%s) failed: %s", tname, args, e)
                continue
            loras = _normalize_lora_list(raw)
            if loras:
                # If the tool ignored our model filter and tagged results,
                # narrow client-side so we never offer an incompatible LoRA.
                if model_type and any(l.get("model_type") for l in loras):
                    filtered = [
                        l for l in loras
                        if not l.get("model_type") or l["model_type"] == model_type
                    ]
                    if filtered:
                        loras = filtered
                logger.info(
                    "Found %d LoRA(s) via %s for model_type=%s",
                    len(loras), tname, model_type or "(any)",
                )
                return loras, True

    logger.info("LoRA tool(s) present but returned nothing for model_type=%s", model_type)
    return [], True


def _parse_loras(raw: str) -> tuple[list[str], str]:
    """
    Parse a "filename:weight, filename2:weight2" string (commas and/or
    newlines as separators) into WanGP's expected pair: a list of LoRA
    filenames for `activated_loras`, and a space-separated string of
    matching weights for `loras_multipliers`. Weight defaults to 1.0 when
    omitted. Blank/whitespace-only entries are ignored.
    """
    names: list[str] = []
    weights: list[float] = []
    if not raw:
        return names, ""
    parts = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
    for part in parts:
        if ":" in part:
            name, w = part.rsplit(":", 1)
            name = name.strip()
            try:
                weight = float(w.strip())
            except ValueError:
                weight = 1.0
        else:
            name, weight = part, 1.0
        if not name:
            continue
        names.append(name)
        weights.append(weight)
    multipliers = " ".join(str(w) for w in weights)
    return names, multipliers


def _map_job_to_settings(job: dict, defaults: dict) -> dict:
    params = job.get("params") or {}
    jtype = job.get("job_type", "t2v")
    prompt = job.get("prompt") or ""

    model = params.get("model_type") or params.get("model")
    if model in (None, "", "auto"):
        # Auto resolves per job type first, then per output medium — an image
        # job must not inherit the video default (which would hand WanGP a
        # model that can't do stills), and job types with special needs
        # (p2v=VACE, ia2v=audio-capable) need their own answer so Easy mode
        # can hide the model picker entirely.
        per_type = defaults.get(f"model_{jtype}")
        if per_type:
            model = per_type
            logger.info("Job %s: auto model -> %s (per-job-type default for %s)",
                        job.get("id"), model, jtype)
        else:
            if jtype in ("t2i", "i2i"):
                model = (
                    defaults.get("image_model_type")
                    or defaults.get("model_type")
                    or "flux_dev"
                )
            else:
                model = defaults.get("model_type") or "ltx2_22B_distilled"
            logger.info("Job %s: auto model -> %s (%s)", job.get("id"), model, jtype)

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
    if jtype == "v2v" and video_path:
        settings["video_source"] = str(video_path)
        settings["image_prompt_type"] = "V"
    # End frame. Only i2v/v2v offer it, and WanGP only honours it when "E"
    # is present in image_prompt_type — setting image_end alone leaves the
    # end frame silently ignored, the same failure mode as an unguided pose.
    if image_end and jtype in ("i2v", "v2v"):
        settings["image_end"] = str(image_end)
        letters = settings.get("image_prompt_type") or ""
        if "E" not in letters:
            settings["image_prompt_type"] = letters + "E"
    elif image_end:
        logger.info(
            "Job %s: ignoring end image — %s does not support one",
            job.get("id"), jtype,
        )
    if jtype == "ia2v" and audio_path:
        settings["audio_guide"] = str(audio_path)
        settings["audio_prompt_type"] = "A"
    # ── Reference images ──────────────────────────────────────────────────
    # Both VACE and LTX-2.3 MSR take an ordered list, and both use the same
    # convention: background/setting first, then subjects and objects. MSR
    # accepts 2-5; VACE has no fixed cap. Falls back to the single legacy
    # image_path so older jobs (and Reuse) keep working.
    ref_paths: list[str] = []
    for p in (params.get("reference_image_paths") or []):
        rp = _resolve_local_media(p) or p
        if rp:
            ref_paths.append(str(rp))
    if not ref_paths and image_path:
        ref_paths = [str(image_path)]

    # ── Pose / control-guided generation ──────────────────────────────────
    # A driving video supplies motion; the model follows its pose, depth or
    # edges rather than the guide's appearance. Available as its own job type
    # and as an add-on to t2v / i2v (reference image + driving motion).
    control_path = (
        job.get("control_video_path") or params.get("control_video_path")
        or params.get("control_video_url") or job.get("control_video_url")
    )
    control_type = (params.get("control_type") or "pose").lower()

    if control_path and jtype in ("p2v", "t2v", "i2v"):
        settings["video_guide"] = str(control_path)
        letter = _control_letter(control_type)
        settings["video_prompt_type"] = f"V{letter}"
        # Reference images still drive identity/appearance when supplied.
        if ref_paths:
            settings["image_refs"] = ref_paths
            settings["video_prompt_type"] = f"V{letter}I"
            # image_start would pin frame one to the reference and fight the
            # guide's first pose, so it is deliberately not set here.
            settings.pop("image_start", None)
            settings.pop("image_prompt_type", None)
        strength = params.get("control_strength")
        if strength is not None:
            try:
                settings["control_net_weight"] = float(strength)
            except (TypeError, ValueError):
                pass
        # Optional window: apply the guide to part of the clip only
        for key, target in (("control_start", "video_guide_start"),
                            ("control_end", "video_guide_end")):
            if params.get(key) is not None:
                settings[target] = params[key]
        logger.info(
            "Job %s: %s-guided generation (video_prompt_type=%s, %d ref image(s))",
            job.get("id"), control_type, settings["video_prompt_type"], len(ref_paths),
        )
    elif len(ref_paths) > 1 and jtype in ("t2v", "i2v", "t2i", "i2i"):
        # Reference-to-video with no driving video: LTX-2.3 MSR packs 2-5
        # reference images (background first, then subjects/objects) into a
        # pseudo-video sequence. VACE reference-to-video works the same way.
        # A start frame would compete with the references for frame one, so
        # drop it in favour of the reference set.
        settings["image_refs"] = ref_paths
        settings.pop("image_start", None)
        settings.pop("image_prompt_type", None)
        if msr_ref_len := params.get("msr_reference_video_length"):
            settings["MSR_reference_video_length"] = msr_ref_len
        logger.info(
            "Job %s: multi-reference generation with %d images",
            job.get("id"), len(ref_paths),
        )

    # ── LoRAs ─────────────────────────────────────────────────────────────
    # Applies to every job type, not just guided ones. When the user picked
    # none, fall back to the per-job-type default so Easy mode (where the
    # LoRA picker is hidden) still gets the right ones automatically.
    loras_raw = params.get("loras") or defaults.get(f"loras_{jtype}") or ""
    if loras_raw:
        lora_names, lora_multipliers = _parse_loras(str(loras_raw))
        if lora_names:
            settings["activated_loras"] = lora_names
            settings["loras_multipliers"] = lora_multipliers
            logger.info(
                "Job %s: activating LoRAs %s (multipliers=%s)%s",
                job.get("id"), lora_names, lora_multipliers,
                "" if params.get("loras") else " [per-job-type default]",
            )

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
    dest = _unique_result_name(job_id, ext)
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
    # Collect binary content blocks (image/audio/embedded resource) BEFORE
    # falling back to text-only parsing. MCP servers commonly return generated
    # media as {"type":"image","data":"<b64>","mimeType":"image/png"} — the old
    # code dropped these on the floor.
    media_blocks = _collect_media_blocks(result.get("content"))

    if result.get("structuredContent") is not None:
        sc = result["structuredContent"]
        if media_blocks and isinstance(sc, dict):
            sc = dict(sc)
            sc.setdefault("_mcp_media", media_blocks)
        return sc

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
            parsed: Any
            try:
                parsed = json.loads(joined)
            except Exception:
                parsed = {"text": joined}
            if media_blocks and isinstance(parsed, dict):
                parsed = dict(parsed)
                parsed.setdefault("_mcp_media", media_blocks)
            return parsed

    if media_blocks:
        out = dict(result)
        out["_mcp_media"] = media_blocks
        return out
    return result


def _collect_media_blocks(content: Any) -> list[dict]:
    """Pull base64 media out of MCP content blocks, at any nesting depth."""
    found: list[dict] = []

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        ntype = str(node.get("type") or "").lower()
        if ntype in ("image", "audio", "video") and isinstance(node.get("data"), str):
            found.append({
                "data": node["data"],
                "mime": node.get("mimeType") or node.get("mime_type") or "",
            })
            return
        if ntype == "resource" or "resource" in node:
            res = node.get("resource") if isinstance(node.get("resource"), dict) else node
            blob = res.get("blob") if isinstance(res, dict) else None
            if isinstance(blob, str) and len(blob) > 64:
                found.append({
                    "data": blob,
                    "mime": res.get("mimeType") or res.get("mime_type") or "",
                    "uri": res.get("uri") or "",
                })
                return
        for value in node.values():
            if isinstance(value, (dict, list)):
                walk(value)

    walk(content)
    return found


def _ext_from_mime(mime: str, fallback: str = ".png") -> str:
    m = (mime or "").lower()
    table = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "video/mp4": ".mp4",
        "video/webm": ".webm",
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav",
    }
    for key, ext in table.items():
        if key in m:
            return ext
    return fallback


def _media_blocks_to_local(job_id: str, obj: Any) -> str | None:
    """Decode any MCP media blocks found anywhere in a payload into static/results."""
    import base64

    blocks: list[dict] = []
    if isinstance(obj, dict) and isinstance(obj.get("_mcp_media"), list):
        blocks.extend([b for b in obj["_mcp_media"] if isinstance(b, dict)])
    blocks.extend(_collect_media_blocks(obj))

    for block in blocks:
        raw = block.get("data")
        if not isinstance(raw, str) or len(raw) < 64:
            continue
        if raw.strip().startswith("data:") and "," in raw:
            raw = raw.split(",", 1)[1]
        try:
            data = base64.b64decode(raw, validate=False)
        except Exception:
            continue
        if len(data) < 64:
            continue
        ext = _ext_from_mime(str(block.get("mime") or ""))
        uri = str(block.get("uri") or "")
        if uri and "." in uri.rsplit("/", 1)[-1]:
            ext = "." + uri.rsplit(".", 1)[-1].lower()[:8]
        return _save_bytes_result(job_id, data, ext)
    return None


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
    """
    Make a local input file reachable by WanGP.

    Preferred: wangp_create_gallery_upload (PUT bytes, get a gallery id).
    Builds without that tool use wan2gp_input_dir + wan2gp_input_remote_prefix:
    a shared folder both machines can see, so we copy the file in and hand
    WanGP the path as it sees it.
    """
    path = Path(local_path)
    if not path.is_file():
        return str(local_path)

    if not mcp_has_tool(mcp_url, "wangp_create_gallery_upload"):
        return _stage_input_without_gallery(path)

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


def _stage_input_without_gallery(path: Path) -> str:
    """
    Expose a local input file to WanGP when the server has no upload tool.

    Two ways to get the file to WanGP:
      1. HTTP upload to a scripts/inputs_server.py running on the WanGP
         host (wan2gp_input_http_base) — works across machines with no
         shared filesystem. Preferred when configured.
      2. A folder both machines can see (wan2gp_input_dir / mounted
         share) — we copy the file in, then hand WanGP the path as it
         sees it. Passing a URL instead does not work — WanGP treats the
         value as a filesystem path and joins it onto its own working
         directory, producing e.g. C:\\AI-Tools\\Wan2GP\\http:\\host\\file.jpg
         and failing with Errno 22.
    """
    settings = db.get_settings()

    http_base = (settings.get("wan2gp_input_http_base") or "").strip().rstrip("/")
    if http_base:
        return _stage_input_via_http(path, http_base, settings)

    shared_dir = (settings.get("wan2gp_input_dir") or "").strip()
    remote_prefix = (settings.get("wan2gp_input_remote_prefix") or "").strip()

    if not shared_dir:
        raise RuntimeError(
            "This WanGP build has no wangp_create_gallery_upload tool, so input "
            "media must go through either an inputs_server.py upload URL or a "
            "shared folder. Set 'Shared input folder — upload URL' (recommended "
            "if genai and WanGP are on different machines), or 'Shared input "
            "folder' + 'Shared input folder — WanGP-side path', in "
            f"Admin → Wan2GP Server. (file={path.name})"
        )

    dest_dir = Path(shared_dir)
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"shared input folder {shared_dir} is not writable: {e}")

    # A dropped SMB mount silently becomes an ordinary empty local directory,
    # which would hand WanGP a path to a file it cannot see. Check we can
    # actually write, and warn loudly if the mount looks wrong.
    try:
        dest = dest_dir / path.name
        if not (dest.exists() and dest.stat().st_size == path.stat().st_size):
            shutil.copy2(path, dest)
        if not dest.is_file() or dest.stat().st_size != path.stat().st_size:
            raise RuntimeError("copy verification failed")
    except Exception as e:
        raise RuntimeError(
            f"could not stage input into {shared_dir}: {e} "
            "— check the share is mounted and writable by the GenAI service account"
        )

    if remote_prefix:
        sep = "\\" if ("\\" in remote_prefix or ":" in remote_prefix[:3]) else "/"
        remote = remote_prefix.rstrip("/\\") + sep + path.name
        logger.info("Staged input for WanGP at %s", remote)
        return remote

    logger.warning(
        "No WanGP-side path configured; sending %s, which only works if WanGP "
        "sees the identical path.", dest
    )
    return str(dest)


def _stage_input_via_http(path: Path, http_base: str, settings: dict) -> str:
    """
    Upload a local input file to a companion scripts/inputs_server.py
    running on the WanGP host, and return the absolute path it reports
    back — the path WanGP itself will see. Used when genai and WanGP are
    on different machines with no mounted shared folder between them.
    """
    token = (settings.get("wan2gp_input_http_token") or "").strip()
    headers = {}
    if token:
        headers["X-Auth-Token"] = token

    data = path.read_bytes()
    url = f"{http_base}/{quote(path.name)}"
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.put(url, content=data, headers=headers)
    except Exception as e:
        raise RuntimeError(f"could not reach input upload server at {http_base}: {e}")

    if resp.status_code >= 400:
        raise RuntimeError(
            f"input upload to {http_base} failed: HTTP {resp.status_code} {resp.text[:300]}"
        )
    try:
        body = resp.json()
    except Exception:
        raise RuntimeError(f"input upload to {http_base}: bad response {resp.text[:300]}")

    remote_path = body.get("path")
    if not remote_path:
        raise RuntimeError(f"input upload to {http_base}: no path in response {body}")
    logger.info("Staged input for WanGP via HTTP at %s", remote_path)
    return remote_path


def _resolve_local_media(p: str | None) -> Optional[str]:
    if not p:
        return None
    path = Path(p)
    if path.is_file():
        return str(path)
    rel = str(p).lstrip("/")
    root = Path(__file__).parent.parent
    for candidate in (
        root / rel,
        UPLOAD_DIR / Path(p).name,
        UPLOAD_DIR / "jobs" / Path(p).name,
        root / "static" / "uploads" / "jobs" / Path(p).name,
        # personal library: uploaded imports and reused generations
        root / "static" / "library" / Path(p).name,
        RESULTS_DIR / Path(p).name,
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def _prepare_mcp_source(mcp_url: str, job: dict, settings_cfg: dict) -> dict:
    defaults = {
        "model_type": settings_cfg.get("default_model_type") or "ltx2_22B_distilled",
        "image_model_type": settings_cfg.get("default_image_model_type") or "flux_dev",
        "resolution": settings_cfg.get("default_resolution") or "1280x704",
        "num_inference_steps": settings_cfg.get("default_steps") or 8,
        "force_fps": settings_cfg.get("default_fps") or "24",
    }
    # Per-job-type model/LoRA defaults, so Easy mode can hide both pickers.
    for _jt in ("t2v", "i2v", "ia2v", "v2v", "p2v", "t2i", "i2i"):
        mv = (settings_cfg.get(f"default_model_{_jt}") or "").strip()
        if mv:
            defaults[f"model_{_jt}"] = mv
        lv = (settings_cfg.get(f"default_loras_{_jt}") or "").strip()
        if lv:
            defaults[f"loras_{_jt}"] = lv
    source = _map_job_to_settings(job, defaults)

    # ── Guide-capable model enforcement ───────────────────────────────────
    # A guide video only does anything on a VACE-style model. "Auto" used to
    # fall through to default_model_type (an LTX2 build), which accepts
    # video_guide/video_prompt_type and then ignores them — the job succeeds
    # but the output never follows the pose. Resolve to a capable model here,
    # or fail loudly rather than silently returning unguided video.
    if source.get("video_guide"):
        params = job.get("params") or {}
        requested = params.get("model_type") or params.get("model")
        explicit = bool(requested) and requested not in ("auto", "")
        current = str(source.get("model_type") or "")

        if not _is_control_capable(current):
            picked = _pick_control_model(mcp_url, requested if explicit else None)
            if picked and picked != current:
                if explicit:
                    raise RuntimeError(
                        f"Model '{current}' cannot follow a driving video — "
                        "pose/depth/edge control needs a VACE-style or LTX-2 "
                        f"model (e.g. '{picked}'). Pick one on the Generate "
                        "page, or set Model to Auto."
                    )
                logger.info(
                    "Job %s: '%s' is not guide-capable; using '%s' for %s control",
                    job.get("id"), current, picked,
                    (job.get("params") or {}).get("control_type") or "pose",
                )
                source["model_type"] = picked
            elif not picked:
                raise RuntimeError(
                    "No guide-capable model is available on this WanGP install, "
                    "so a driving video cannot be used. Install a VACE model "
                    "(e.g. Wan 2.1/2.2 VACE) or an LTX-2 model with its "
                    "pose/depth/canny IC LoRA."
                )

        # LTX-2 gets control from an IC LoRA, not from the checkpoint, so a
        # guided LTX-2 job with nothing activated will quietly ignore the
        # guide. Warn loudly — we can't tell which local file is the right
        # IC LoRA, so this is not something we can fix automatically.
        final_model = str(source.get("model_type") or "")
        if _needs_control_lora(final_model) and not source.get("activated_loras"):
            logger.warning(
                "Job %s: %s is an LTX-2 model and needs a pose/depth/canny IC "
                "LoRA for control, but none is activated — the driving video "
                "will be ignored. Set one in Admin -> Server & Queue "
                "(default_loras_p2v) or pick one on the Generate page.",
                job.get("id"), final_model,
            )

    # Reference images only do something on models that consume them. Sending
    # them elsewhere isn't an error, but the output silently disregards them,
    # so say so rather than letting it look like the references "didn't work".
    if source.get("image_refs"):
        rm = str(source.get("model_type") or "")
        if not _supports_reference_images(rm):
            logger.warning(
                "Job %s: %d reference image(s) supplied but '%s' does not "
                "support them — they will be ignored. Use a VACE model, the "
                "LTX-2.3 MSR finetune, or an edit/identity model.",
                job.get("id"), len(source["image_refs"]), rm,
            )

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


def _unique_result_name(job_id: str, ext: str, data: bytes | None = None) -> Path:
    """
    Unique per-file name. Results are retained in the personal library, so a
    retry that produces *different* output must not overwrite the earlier file.
    If the bytes are identical to an existing file for this job, reuse it — that
    keeps re-applying the same MCP result idempotent.
    """
    base = RESULTS_DIR / f"{job_id}{ext}"
    if not base.exists():
        return base
    if data is not None:
        for existing in sorted(RESULTS_DIR.glob(f"{job_id}*{ext}")):
            try:
                if existing.stat().st_size == len(data) and existing.read_bytes() == data:
                    return existing
            except OSError:
                continue
    stamp = time.strftime("%Y%m%d%H%M%S")
    candidate = RESULTS_DIR / f"{job_id}_{stamp}{ext}"
    n = 1
    while candidate.exists():
        candidate = RESULTS_DIR / f"{job_id}_{stamp}_{n}{ext}"
        n += 1
    return candidate


def _media_type_for(url_or_ext: str) -> str:
    low = str(url_or_ext).lower()
    if any(low.endswith(e) for e in (".mp4", ".webm", ".mov", ".mkv", ".avi")):
        return "video"
    if any(low.endswith(e) for e in (".mp3", ".wav", ".flac", ".m4a", ".ogg")):
        return "audio"
    return "image"


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
    dest = _unique_result_name(job_id, ext, data)
    if not dest.exists():
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


_mcp_tool_cache: dict[str, list[dict]] = {}


def mcp_discover_tools(mcp_url: str, force: bool = False) -> list[dict]:
    """Cached tools/list returning full tool dicts (name + inputSchema)."""
    endpoint = _mcp_endpoint(mcp_url)
    if not force and endpoint in _mcp_tool_cache:
        return _mcp_tool_cache[endpoint]
    tools: list[dict] = []
    try:
        data, _ = mcp_raw_request(mcp_url, "tools/list", {}, timeout=20.0)
        if isinstance(data, dict):
            result = data.get("result") or data
            raw = result.get("tools") if isinstance(result, dict) else None
            for tool in raw or []:
                if isinstance(tool, dict) and tool.get("name"):
                    tools.append(tool)
    except Exception as e:
        logger.warning("tools/list discovery failed: %s", e)
    _mcp_tool_cache[endpoint] = tools
    if tools:
        logger.info("MCP tools discovered: %s", ", ".join(t.get("name", "") for t in tools))
    return tools


def _tool_param_names(tool: dict) -> list[str]:
    schema = tool.get("inputSchema") or tool.get("input_schema") or {}
    props = schema.get("properties") if isinstance(schema, dict) else None
    return list(props.keys()) if isinstance(props, dict) else []


def _find_tools(mcp_url: str, include: tuple[str, ...], exclude: tuple[str, ...] = ()) -> list[dict]:
    out = []
    for tool in mcp_discover_tools(mcp_url):
        name = str(tool.get("name") or "").lower()
        if any(word in name for word in include) and not any(word in name for word in exclude):
            out.append(tool)
    return out


def mcp_register_path_and_download(mcp_url: str, remote_path: str, job_id: str) -> str:
    """
    Ask the server to register/import an on-disk output into its gallery, then
    download it by the returned id. Tool names vary between WanGP builds, so we
    discover candidates instead of hardcoding.
    """
    if not remote_path:
        raise RuntimeError("no remote path to register")

    candidates = _find_tools(
        mcp_url,
        include=("gallery", "output", "file", "media", "import", "register", "scan"),
        exclude=("upload", "delete", "remove", "clear"),
    )
    # Prefer tools that look like they take a path and produce/refresh gallery state
    def rank(tool: dict) -> int:
        name = str(tool.get("name") or "").lower()
        params = [p.lower() for p in _tool_param_names(tool)]
        score = 0
        if any(w in name for w in ("register", "import", "add", "scan", "refresh", "index")):
            score += 50
        if any(p in params for p in ("path", "file", "filename", "filepath", "file_path", "src")):
            score += 40
        if "gallery" in name:
            score += 20
        if any(w in name for w in ("get", "read", "fetch", "download")):
            score += 10
        return score

    ranked = [t for t in sorted(candidates, key=rank, reverse=True) if rank(t) >= 40]
    if not ranked:
        raise RuntimeError("no path-registration tool found on server")

    name_only = Path(str(remote_path).replace("\\", "/")).name
    errors: list[str] = []
    for tool in ranked[:6]:
        tname = str(tool.get("name"))
        params = [p.lower() for p in _tool_param_names(tool)]
        keys = [p for p in ("path", "file_path", "filepath", "file", "filename", "src")
                if p in params] or ["path"]
        for key in keys:
            for value in (remote_path, str(remote_path).replace("\\", "/"), name_only):
                try:
                    out = mcp_call_tool(mcp_url, tname, {key: value}, timeout=60.0)
                except Exception as e:
                    errors.append(f"{tname}({key}): {e}")
                    continue
                # The call may return the bytes directly...
                got = _media_blocks_to_local(job_id, out)
                if got:
                    logger.info("Registered+received media via %s", tname)
                    return got
                # ...or a gallery id we can then download.
                gid = _extract_gallery_id(out) if out is not None else None
                if not gid:
                    for cand in _walk_gallery_ids(out if isinstance(out, (dict, list)) else {}):
                        gid = cand
                        break
                if gid:
                    try:
                        url = mcp_download_gallery_item(mcp_url, str(gid), job_id)
                        logger.info("Registered %s via %s → gallery %s", name_only, tname, gid)
                        return url
                    except Exception as e:
                        errors.append(f"download {gid}: {e}")
    raise RuntimeError("path registration failed: " + ("; ".join(errors)[:300] or "no usable response"))


def mcp_read_file_resource(mcp_url: str, remote_path: str, job_id: str) -> str:
    """
    Try the MCP resources API (resources/read) to pull the file's bytes.
    Many servers expose outputs as file:// resources even when no gallery exists.
    """
    if not remote_path:
        raise RuntimeError("no remote path for resource read")

    norm = str(remote_path).replace("\\", "/")
    from urllib.parse import quote
    uris = [
        remote_path,
        norm,
        f"file:///{quote(norm.lstrip('/'), safe=':/')}",
        f"file://{quote(norm, safe=':/')}",
    ]

    # If the server advertises resources, prefer a matching advertised URI.
    try:
        data, _ = mcp_raw_request(mcp_url, "resources/list", {}, timeout=20.0)
        listed = []
        if isinstance(data, dict):
            result = data.get("result") or data
            for res in (result.get("resources") or []) if isinstance(result, dict) else []:
                if isinstance(res, dict) and res.get("uri"):
                    listed.append(str(res["uri"]))
        target = Path(norm).name.lower()
        for uri in listed:
            if target and target in uri.lower():
                uris.insert(0, uri)
    except Exception:
        pass

    errors = []
    for uri in uris:
        try:
            data, _ = mcp_raw_request(mcp_url, "resources/read", {"uri": uri}, timeout=90.0)
        except Exception as e:
            errors.append(str(e))
            continue
        if not isinstance(data, dict):
            continue
        if data.get("error"):
            errors.append(str(data["error"])[:120])
            continue
        result = data.get("result") or data
        got = _media_blocks_to_local(job_id, result)
        if got:
            logger.info("Fetched result via resources/read %s", uri)
            return got
        # contents[].blob / .text
        contents = result.get("contents") if isinstance(result, dict) else None
        for item in contents or []:
            if not isinstance(item, dict):
                continue
            blob = item.get("blob")
            if isinstance(blob, str) and len(blob) > 64:
                import base64
                try:
                    raw = base64.b64decode(blob, validate=False)
                except Exception:
                    continue
                if len(raw) > 64:
                    ext = _ext_from_mime(
                        str(item.get("mimeType") or ""),
                        Path(norm).suffix or ".png",
                    )
                    return _save_bytes_result(job_id, raw, ext)
    raise RuntimeError("resources/read failed: " + ("; ".join(errors)[:250] or "no blob returned"))


def mcp_download_via_origin_guess(mcp_url: str, remote_path: str, job_id: str) -> str:
    """
    Last resort needing zero configuration: the MCP server already serves HTTP
    (gallery uploads PUT to its origin), so probe common static routes for the
    output filename on that same origin.
    """
    if not remote_path:
        raise RuntimeError("no remote path to probe")
    origin = _mcp_origin(mcp_url)
    if not origin:
        raise RuntimeError("no MCP origin")

    from urllib.parse import quote
    name = Path(str(remote_path).replace("\\", "/")).name
    if not name:
        raise RuntimeError("no filename")
    enc = quote(name)
    routes = [
        f"/outputs/{enc}", f"/output/{enc}", f"/files/{enc}", f"/file/{enc}",
        f"/gallery/{enc}", f"/media/{enc}", f"/static/outputs/{enc}",
        f"/download/{enc}", f"/results/{enc}",
    ]
    last_err = None
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        for route in routes:
            url = origin.rstrip("/") + route
            try:
                r = client.get(url)
            except Exception as e:
                last_err = str(e)
                continue
            if r.status_code == 200 and r.content and len(r.content) > 64:
                ct = (r.headers.get("content-type") or "").lower()
                if "text/html" in ct and b"<html" in r.content[:200].lower():
                    last_err = f"HTML at {url}"
                    continue
                logger.info("Fetched result from MCP origin route %s", route)
                return _save_bytes_result(job_id, r.content, Path(name).suffix or ".bin")
            last_err = f"HTTP {r.status_code} {route}"
    raise RuntimeError(f"origin probe failed: {last_err}")


def mcp_has_tool(mcp_url: str, *names: str) -> bool:
    """True if the server advertises any of these tool names."""
    try:
        available = {str(t.get("name") or "") for t in mcp_discover_tools(mcp_url)}
    except Exception:
        return True  # discovery failed — don't block, let the call try
    if not available:
        return True  # tools/list unsupported — assume yes rather than skip
    return any(n in available for n in names)


def mcp_supports_gallery(mcp_url: str) -> bool:
    return mcp_has_tool(
        mcp_url,
        "wangp_list_gallery",
        "wangp_create_gallery_download",
        "wangp_get_gallery_selection",
    )


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
    has_gallery = mcp_supports_gallery(mcp_url) if mcp_url else False
    if mcp_url and not has_gallery:
        logger.info(
            "Job %s: server has no gallery tools — skipping gallery strategies", job_id
        )

    def _result_path_hint() -> str | None:
        paths = _walk_file_paths(result)
        if paths:
            return paths[0]
        for f in (files if isinstance(files, list) else []):
            if isinstance(f, str):
                return f
            if isinstance(f, dict) and (f.get("path") or f.get("file")):
                return f.get("path") or f.get("file")
        return None

    # 0) Inline MCP media blocks — bytes already in the response, zero extra I/O.
    try:
        url = _media_blocks_to_local(job_id, result)
        if url:
            logger.info("Job %s: result arrived inline over MCP", job_id)
    except Exception as e:
        transfer_errors.append(f"inline media: {e}")

    # 0b) Static HTTP root over WanGP outputs/ — the reliable path for builds
    # with no gallery tools, so try it before the slower fallbacks.
    if not url:
        try:
            out_base = (db.get_settings().get("wan2gp_outputs_http_base") or "").strip()
            hint = _result_path_hint()
            if out_base and hint:
                url = mcp_download_via_outputs_http(out_base, hint, job_id)
                if url:
                    logger.info("Job %s transferred via outputs HTTP", job_id)
        except Exception as e:
            transfer_errors.append(f"outputs_http: {e}")

    # 1) Explicit gallery list
    if not url and has_gallery:
        for g in gallery if isinstance(gallery, list) else []:
            gid = _extract_gallery_id(g)
            if gid and mcp_url:
                try:
                    url = mcp_download_gallery_item(mcp_url, gid, job_id)
                    break
                except Exception as e:
                    transfer_errors.append(f"gallery {gid}: {e}")

    # 2) Any gallery-like ids nested in result
    if not url and mcp_url and has_gallery:
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
            if gid and mcp_url and has_gallery:
                try:
                    url = mcp_download_gallery_item(mcp_url, str(gid), job_id)
                    break
                except Exception as e:
                    transfer_errors.append(str(e))
            # NOTE: previously this branch required gradio_url to be set, which
            # meant an http:// URL or a locally-visible/mounted path was never
            # even attempted. mcp_download_remote_file handles both by itself.
            if remote_path:
                try:
                    url = mcp_download_remote_file(
                        mcp_url, str(remote_path), job_id, gradio_url=gradio_url
                    )
                    break
                except Exception as e:
                    transfer_errors.append(str(e))

    # 4) Any media paths nested in result
    if not url and (gradio_url or "").strip():
        for path in _walk_file_paths(result):
            try:
                url = mcp_download_remote_file(mcp_url, path, job_id, gradio_url=gradio_url)
                break
            except Exception as e:
                transfer_errors.append(str(e))

    # 4c) Ask the server to register the on-disk output, then download it.
    if not url and mcp_url:
        hint = _result_path_hint()
        if hint:
            strategies = [("resources/read", mcp_read_file_resource)]
            if has_gallery:
                strategies.append(("register_path", mcp_register_path_and_download))
            strategies.append(("origin_probe", mcp_download_via_origin_guess))
            for strategy, fn in strategies:
                try:
                    url = fn(mcp_url, str(hint), job_id)
                    if url:
                        logger.info("Job %s transferred via %s", job_id, strategy)
                        break
                except Exception as e:
                    transfer_errors.append(f"{strategy}: {e}")

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

    # 6) Gallery listing sweep — only meaningful if the server has gallery tools
    # (outputs HTTP was already attempted first, at step 0b).
    if not url and mcp_url and has_gallery:
        try:
            url = mcp_fetch_result_via_gallery(mcp_url, job_id, _result_path_hint())
        except Exception as e:
            transfer_errors.append(f"list_gallery: {e}")
            logger.warning("gallery list transfer failed: %s", e)

    if not url:
        tool_names = []
        try:
            tool_names = [str(t.get("name")) for t in mcp_discover_tools(mcp_url)] if mcp_url else []
        except Exception:
            pass
        msg = (
            "Generation finished on remote WanGP, but the file could not be transferred. "
            "Tried: inline media, gallery download, resources/read, path registration, "
            "origin probe, outputs HTTP. "
            + (f"server_tools=[{', '.join(tool_names)[:300]}] " if tool_names else "")
            + ("errors=" + "; ".join(transfer_errors)[:400] if transfer_errors else "")
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

    # Auto-file the result into the owner's personal library.
    try:
        job = db.get_job(job_id) or {}
        owner = job.get("user_id")
        if owner:
            params = job.get("params") or {}
            item = db.add_library_item(
                user_id=owner,
                url=url,
                media_type=_media_type_for(url),
                job_id=job_id,
                job_type=job.get("job_type") or "",
                prompt=job.get("prompt") or "",
                title=job.get("title") or "",
                model=str(params.get("model_type") or params.get("model") or ""),
                params={
                    k: params.get(k)
                    for k in ("resolution", "steps", "seed", "guidance_scale",
                              "duration_seconds", "fps", "model", "model_type")
                    if params.get(k) is not None
                },
                source="generated",
            )
            logger.info("Job %s filed to library as %s", job_id, item.get("id"))
    except Exception as e:
        # Never fail a completed job because of library bookkeeping.
        logger.warning("Library insert failed for job %s: %s", job_id, e)

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

    gen_args: dict = {"source": source, "wait": False, "event_limit": 20}
    # If the server advertises a "return the bytes inline" flag, use it — that
    # removes the file-transfer problem entirely.
    inline_flags = {}
    for tool in mcp_discover_tools(mcp_url):
        if str(tool.get("name")) != "wangp_generate":
            continue
        for param in _tool_param_names(tool):
            if param.lower() in (
                "return_media", "include_media", "return_files", "return_bytes",
                "include_bytes", "return_images", "inline_media", "embed_media",
            ):
                inline_flags[param] = True
        break
    if inline_flags:
        logger.info("wangp_generate supports inline media: %s", list(inline_flags))
        gen_args.update(inline_flags)

    try:
        out = mcp_call_tool(mcp_url, "wangp_generate", gen_args, timeout=90.0)
    except Exception as e:
        if inline_flags:
            # Flag rejected despite being advertised — retry without it.
            logger.warning("wangp_generate with inline flags failed (%s); retrying plain", e)
            try:
                out = mcp_call_tool(
                    mcp_url,
                    "wangp_generate",
                    {"source": source, "wait": False, "event_limit": 20},
                    timeout=90.0,
                )
            except Exception as e2:
                logger.exception("wangp_generate failed")
                db.update_job(job_id, {"error": f"wangp_generate: {e2}"[:800]})
                return False
        else:
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
# WanGP/VACE encodes guide preprocessing as letters in `video_prompt_type`.
# "V" marks that a guide video is supplied; the second letter selects the
# preprocessor. Overridable per-install via the wan2gp_control_letters setting
# because the alphabet has shifted between WanGP releases.
CONTROL_LETTERS = {
    "pose": "P",     # OpenPose skeleton — motion/pose transfer
    "depth": "D",    # depth map
    "canny": "E",    # edges
    "gray": "G",     # luminance
    "flow": "F",     # optical flow
    "raw": "",       # pass the guide through unprocessed
}

CONTROL_LABELS = {
    "pose": "Pose (skeleton)",
    "depth": "Depth",
    "canny": "Edges",
    "gray": "Grayscale",
    "flow": "Motion flow",
    "raw": "Raw video",
}


def _control_letter(control_type: str) -> str:
    """Letter for a control type, honouring any per-install override."""
    try:
        override = (db.get_settings().get("wan2gp_control_letters") or "").strip()
    except Exception:
        override = ""
    if override:
        # format: "pose=P,depth=D,canny=E"
        for pair in override.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                if k.strip().lower() == control_type:
                    return v.strip()
    return CONTROL_LETTERS.get(control_type, "P")


_JOB_TYPE_FILTERS = {
    "t2v": {"main_output": "video"},
    "i2v": {"main_output": "video", "inputs": "image"},
    "ia2v": {"main_output": "video", "inputs": "image"},  # audio-capable video models
    "v2v": {"main_output": "video", "inputs": "video"},
    "t2i": {"main_output": "image"},
    "i2i": {"main_output": "image", "inputs": "image"},
    # Pose/control-driven video needs a VACE-style model that accepts a guide
    "p2v": {"main_output": "video", "inputs": "video"},
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
        elif job_type == "p2v":
            if "video" in mo:
                s += 3
            if "image" in mo and "video" not in mo:
                s -= 2
            # VACE / control-capable models are the only ones that can use a guide
            if any(k in blob for k in ("vace", "control", "pose", "fun")):
                s += 4
            if "video" in inp:
                s += 2
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

    # Hard filter on output media: an image job must never be offered a
    # video model (and vice versa). Scoring alone only pushed them down the
    # list, so they still appeared and could be picked by mistake.
    wants = "image" if job_type in ("t2i", "i2i") else "video"

    def produces(m: dict) -> Optional[str]:
        mo = [str(x).lower() for x in (m.get("main_output") or [])]
        if mo:
            if "video" in mo:
                return "video"
            if "image" in mo:
                return "image"
            return mo[0]
        # No declared output — infer from naming rather than dropping it.
        blob = f"{m.get('model_type','')} {m.get('name','')} {m.get('family','')}".lower()
        if any(k in blob for k in ("2v", "video", "vace", "wan", "ltx", "hunyuan", "mochi")):
            return "video"
        if any(k in blob for k in ("2i", "image", "flux", "qwen", "sdxl", "sd3", "chroma")):
            return "image"
        return None  # unknown: keep it rather than hide a usable model

    matching = [m for m in models if produces(m) in (wants, None)]
    if not matching:
        # Nothing matched — surface everything rather than an empty dropdown,
        # so a mislabelled catalogue doesn't make the page unusable.
        logger.warning(
            "No %s-output models found for job_type=%s; showing all %d",
            wants, job_type, len(models),
        )
        matching = list(models)

    # Pose/control jobs are a hard filter, not a preference: a non-VACE model
    # accepts the guide parameters and silently ignores them, so offering one
    # in the dropdown just produces confusingly unguided output. Only narrow
    # if we actually found capable models, so an unusual catalogue still
    # leaves the page usable.
    if job_type == "p2v":
        capable = [
            m for m in matching
            if _is_control_capable(
                m.get("model_type", ""), m.get("name", ""), m.get("family", "")
            )
        ]
        if capable:
            matching = capable
        else:
            logger.warning(
                "No guide-capable (VACE-style) models found among %d candidates; "
                "pose control will not work with any of them",
                len(matching),
            )

    ranked = sorted(matching, key=score, reverse=True)
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
