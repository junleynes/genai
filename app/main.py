"""
Opensource Generative AI – Modern frontend for Wan2GP
User/Job management • Branding • Easy & Advanced generation modes
"""
import asyncio
import logging
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from . import auth, db
from . import generation as gen_mod
from .generation import process_job, test_wan2gp_connection, BACKEND_ID, BACKEND_BUILT, mcp_call_tool, list_models_for_job_type, try_start_queued_jobs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("genai")

BASE = Path(__file__).parent.parent

app = FastAPI(title="Opensource Generative AI", version="1.0.0")


@app.on_event("startup")
async def _startup_queue_hook():
    def _kick(ids):
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        for jid in ids:
            loop.create_task(process_job(jid))
    gen_mod._on_job_finished = _kick
    logger.info("Opensource Generative AI ready")
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE / "templates"))



@app.get("/api/version")
def api_version():
    """Use this to verify the deployed code is MCP (not Gradio mock)."""
    return {
        "app": "Opensource Generative AI",
        "backend": BACKEND_ID,
        "built": BACKEND_BUILT,
        "mock": False,
        "expects_mcp_url_suffix": "/mcp/",
    }

@app.on_event("startup")
def startup():
    db.ensure_admin()
    logger.info("Opensource Generative AI ready")


# ─── Pydantic models ──────────────────────────────────────────────────────────

class RegisterIn(BaseModel):
    email: str
    password: str = Field(min_length=6)
    name: str = Field(min_length=1, max_length=80)


class LoginIn(BaseModel):
    email: str
    password: str


class JobCreateIn(BaseModel):
    job_type: str = Field(pattern="^(t2v|i2v|t2i|i2i|ia2v|v2v)$")
    mode: str = Field(pattern="^(easy|advanced)$")
    prompt: str = Field(default="", max_length=4000)
    title: str = ""
    negative_prompt: str = ""
    # Easy defaults + advanced overrides
    resolution: str = "832x480"
    steps: int = Field(default=20, ge=1, le=100)
    seed: int = -1
    guidance_scale: float = Field(default=7.5, ge=0, le=30)
    duration_seconds: float = Field(default=4.0, ge=1, le=30)
    fps: int = Field(default=16, ge=8, le=30)
    model: str = "auto"
    extra: dict[str, Any] = {}


class BrandingIn(BaseModel):
    app_name: Optional[str] = None
    tagline: Optional[str] = None
    footer: Optional[str] = None
    primary_color: Optional[str] = None
    secondary_color: Optional[str] = None
    accent_color: Optional[str] = None
    default_theme: Optional[str] = None


class ServerConfigIn(BaseModel):
    wan2gp_url: str = ""
    wan2gp_root: str = ""
    wan2gp_mcp_url: str = ""
    wan2gp_enabled: bool = False
    wan2gp_cli_args: str = "--attention sdpa --profile 4"
    default_model_type: str = "ltx2_22B_distilled"
    default_image_model_type: str = "flux_dev"
    default_resolution: str = "1280x704"
    default_steps: int = 8
    default_guidance_scale: float = 7.5
    default_quality_preset: str = "balanced"
    allow_mock_fallback: bool = False
    queue_enabled: bool = True
    max_concurrent_jobs: int = 1
    concurrent_scope: str = "overall"  # overall | per_user
    # Optional/None so a stale admin page that omits a field cannot wipe it
    # (db.update_settings skips None values).
    wan2gp_outputs_http_base: Optional[str] = None
    wan2gp_input_dir: Optional[str] = None
    wan2gp_input_remote_prefix: Optional[str] = None
    wan2gp_control_letters: Optional[str] = None


class UserUpdateIn(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


# ─── Auth API ─────────────────────────────────────────────────────────────────

@app.post("/api/auth/register")
def register(body: RegisterIn):
    try:
        user = db.create_user(
            email=body.email,
            password_hash=auth.hash_password(body.password),
            name=body.name,
            role="user",
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    token = auth.create_access_token(user["id"], user["role"])
    return {"token": token, "user": user}


@app.post("/api/auth/login")
def login(body: LoginIn):
    user = db.get_user_by_email(body.email)
    if not user or not auth.verify_password(body.password, user["password_hash"]):
        raise HTTPException(401, "Invalid email or password")
    if not user.get("is_active", True):
        raise HTTPException(403, "Account disabled")
    public = {k: v for k, v in user.items() if k != "password_hash"}
    token = auth.create_access_token(user["id"], user["role"])
    return {"token": token, "user": public}


@app.get("/api/auth/me")
def me(user: dict = Depends(auth.get_current_user)):
    return user


# ─── Settings / Branding ──────────────────────────────────────────────────────

@app.get("/api/settings")
def public_settings():
    s = db.get_settings()
    # Never expose internal flags publicly beyond what UI needs
    return {
        "app_name": s["app_name"],
        "tagline": s["tagline"],
        "footer": s["footer"],
        "logo_url": s["logo_url"],
        "favicon_url": s["favicon_url"],
        "primary_color": s["primary_color"],
        "secondary_color": s["secondary_color"],
        "accent_color": s["accent_color"],
        "default_theme": s["default_theme"],
        "wan2gp_enabled": s.get("wan2gp_enabled", False),
    }


@app.put("/api/admin/branding")
def update_branding(body: BrandingIn, admin: dict = Depends(auth.require_admin)):
    data = body.model_dump(exclude_none=True)
    return db.update_settings(data)


@app.put("/api/admin/server")
async def update_server(body: ServerConfigIn, admin: dict = Depends(auth.require_admin)):
    scope = (body.concurrent_scope or "overall").strip().lower()
    if scope not in ("overall", "per_user"):
        scope = "overall"
    max_c = int(body.max_concurrent_jobs or 1)
    if max_c < 1:
        max_c = 1
    if max_c > 32:
        max_c = 32
    return db.update_settings({
        "wan2gp_url": (body.wan2gp_url or "").strip(),
        "wan2gp_root": (body.wan2gp_root or "").strip(),
        "wan2gp_mcp_url": (body.wan2gp_mcp_url or "").strip(),
        "wan2gp_enabled": body.wan2gp_enabled,
        "wan2gp_cli_args": (body.wan2gp_cli_args or "").strip(),
        "default_model_type": (body.default_model_type or "").strip() or "ltx2_22B_distilled",
        "default_image_model_type": (body.default_image_model_type or "").strip() or "flux_dev",
        "default_resolution": (body.default_resolution or "").strip() or "1280x704",
        "default_steps": int(body.default_steps or 8),
        "default_guidance_scale": float(body.default_guidance_scale or 7.5),
        "default_quality_preset": (
            body.default_quality_preset
            if body.default_quality_preset in ("fast", "balanced", "quality", "broadcast")
            else "balanced"
        ),
        "allow_mock_fallback": bool(body.allow_mock_fallback),
        "queue_enabled": bool(body.queue_enabled),
        "max_concurrent_jobs": max_c,
        "concurrent_scope": scope,
        "wan2gp_outputs_http_base": (
            body.wan2gp_outputs_http_base.strip().rstrip("/")
            if body.wan2gp_outputs_http_base is not None else None
        ),
        "wan2gp_input_dir": (
            body.wan2gp_input_dir.strip()
            if body.wan2gp_input_dir is not None else None
        ),
        "wan2gp_input_remote_prefix": (
            body.wan2gp_input_remote_prefix.strip()
            if body.wan2gp_input_remote_prefix is not None else None
        ),
        "wan2gp_control_letters": (
            body.wan2gp_control_letters.strip()
            if body.wan2gp_control_letters is not None else None
        ),
    })


@app.post("/api/admin/server/test")
async def test_server(body: ServerConfigIn, admin: dict = Depends(auth.require_admin)):
    result = await test_wan2gp_connection(url=body.wan2gp_url or '', root=body.wan2gp_root or '', mcp_url=body.wan2gp_mcp_url or '')
    return result


@app.post("/api/admin/upload")
async def upload_asset(
    kind: str = Form(...),  # logo | favicon
    file: UploadFile = File(...),
    admin: dict = Depends(auth.require_admin),
):
    if kind not in ("logo", "favicon"):
        raise HTTPException(400, "kind must be logo or favicon")
    ext = Path(file.filename or "bin").suffix.lower() or ".png"
    if ext not in (".png", ".jpg", ".jpeg", ".webp", ".svg", ".ico"):
        raise HTTPException(400, "Unsupported file type")
    name = f"{kind}{ext}"
    dest = BASE / "static" / "uploads" / name
    content = await file.read()
    dest.write_bytes(content)
    url = f"/static/uploads/{name}"
    key = "logo_url" if kind == "logo" else "favicon_url"
    db.update_settings({key: url})
    return {"url": url}


@app.get("/api/admin/users")
def list_users(admin: dict = Depends(auth.require_admin)):
    users = db.get_users()
    return [{k: v for k, v in u.items() if k != "password_hash"} for u in users]


@app.patch("/api/admin/users/{user_id}")
def patch_user(user_id: str, body: UserUpdateIn, admin: dict = Depends(auth.require_admin)):
    updated = db.update_user(user_id, body.model_dump(exclude_none=True))
    if not updated:
        raise HTTPException(404, "User not found")
    return updated


# ─── Jobs ─────────────────────────────────────────────────────────────────────

@app.get("/api/jobs")
def list_jobs(user: dict = Depends(auth.get_current_user)):
    if user["role"] == "admin":
        jobs = db.get_jobs(limit=200)
        # attach owner email for admin view
        users = {u["id"]: u for u in db.get_users()}
        for j in jobs:
            owner = users.get(j["user_id"])
            j["owner_email"] = owner["email"] if owner else "?"
            j["owner_name"] = owner["name"] if owner else "?"
        return jobs
    return db.get_jobs(user_id=user["id"])


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, user: dict = Depends(auth.get_current_user)):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if user["role"] != "admin" and job["user_id"] != user["id"]:
        raise HTTPException(403, "Not your job")
    return job




@app.get("/api/models")
async def api_models(job_type: str = "t2v", user: dict = Depends(auth.get_current_user)):
    """List WanGP models filtered for the selected generation mode."""
    settings = db.get_settings()
    mcp_url = (settings.get("wan2gp_mcp_url") or "").strip()
    enabled = bool(settings.get("wan2gp_enabled"))
    if not mcp_url:
        return {
            "ok": False,
            "models": [],
            "message": "MCP URL is empty. Set it in Admin → Wan2GP Server (…/mcp/).",
            "job_type": job_type,
        }
    if not enabled:
        return {
            "ok": False,
            "models": [],
            "message": "Wan2GP is disabled. Enable it in Admin → Wan2GP Server.",
            "job_type": job_type,
        }
    try:
        models = await asyncio.to_thread(list_models_for_job_type, mcp_url, job_type)
        return {
            "ok": True,
            "models": models,
            "job_type": job_type,
            "count": len(models),
            "mcp": mcp_url,
        }
    except Exception as e:
        logger = __import__("logging").getLogger("genai")
        logger.exception("api/models failed")
        return {
            "ok": False,
            "models": [],
            "message": str(e)[:500],
            "job_type": job_type,
            "mcp": mcp_url,
        }


@app.post("/api/jobs/{job_id}/retry")
async def retry_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    user: dict = Depends(auth.get_current_user),
):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if user["role"] != "admin" and job["user_id"] != user["id"]:
        raise HTTPException(403, "Not your job")
    if job.get("status") not in ("failed", "completed", "cancelled", "canceled"):
        raise HTTPException(400, "Only failed, cancelled, or completed jobs can be retried")

    db.update_job(job_id, {
        "status": "queued",
        "progress": 0,
        "error": None,
        "result_url": None,
        "preview_url": None,
        "completed_at": None,
    })
    settings = db.get_settings()
    ok_start, reason = db.can_start_job(job["user_id"], settings)
    if not ok_start and reason != "queued":
        raise HTTPException(429, reason)
    if ok_start:
        background_tasks.add_task(process_job, job_id)
    return db.get_job(job_id)



@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    background_tasks: BackgroundTasks,
    user: dict = Depends(auth.get_current_user),
):
    """Cancel a queued/processing job; best-effort MCP wangp_cancel_job."""
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if user["role"] != "admin" and job["user_id"] != user["id"]:
        raise HTTPException(403, "Not your job")
    if job.get("status") not in ("queued", "processing"):
        raise HTTPException(400, "Only queued or processing jobs can be cancelled")

    # Flag local job first so background worker stops treating it as active
    updated = db.update_job(job_id, {
        "status": "cancelled",
        "progress": 0,
        "error": "Cancelled by user",
    })

    settings = db.get_settings()
    mcp_url = (settings.get("wan2gp_mcp_url") or "").strip()
    remote_id = (job.get("params") or {}).get("mcp_job_id")
    if mcp_url and remote_id and settings.get("wan2gp_enabled"):
        try:
            await asyncio.to_thread(
                mcp_call_tool,
                mcp_url,
                "wangp_cancel_job",
                {"job_id": remote_id},
                30.0,
            )
        except Exception as e:
            # Local cancel still sticks; note remote failure
            db.update_job(job_id, {
                "error": f"Cancelled locally; MCP cancel: {e}"[:800],
            })
            updated = db.get_job(job_id)

    try:
        for jid in try_start_queued_jobs():
            background_tasks.add_task(process_job, jid)
    except Exception:
        logger.exception("queue kick after cancel")
    return updated


@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    user: dict = Depends(auth.get_current_user),
    job_type: str = Form(...),
    mode: str = Form("easy"),
    prompt: str = Form(""),
    title: str = Form(""),
    negative_prompt: str = Form(""),
    resolution: str = Form("832x480"),
    steps: int = Form(20),
    seed: int = Form(-1),
    guidance_scale: float = Form(7.5),
    duration_seconds: float = Form(4.0),
    fps: int = Form(16),
    model: str = Form("auto"),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    end_image: Optional[UploadFile] = File(None),
    # Pose / control-guided generation
    control_video: Optional[UploadFile] = File(None),
    control_type: str = Form("pose"),
    control_strength: Optional[float] = Form(None),
    control_video_library_id: str = Form(""),
    # Reuse existing library media instead of uploading (ids from /api/library)
    image_library_id: str = Form(""),
    audio_library_id: str = Form(""),
    video_library_id: str = Form(""),
    end_image_library_id: str = Form(""),
):
    allowed = {"t2v", "i2v", "t2i", "i2i", "ia2v", "v2v", "p2v"}
    if job_type not in allowed:
        raise HTTPException(400, f"Invalid job_type. Allowed: {sorted(allowed)}")
    if mode not in ("easy", "advanced"):
        raise HTTPException(400, "mode must be easy or advanced")

    def from_library(item_id: str, want: str) -> Optional[str]:
        """Resolve a library id to a served URL, checking ownership and type."""
        if not item_id:
            return None
        item = db.get_library_item(item_id.strip(), user_id=user["id"])
        if not item:
            raise HTTPException(404, f"Library item not found: {item_id}")
        if want and item.get("media_type") != want:
            raise HTTPException(
                400,
                f"Library item {item_id} is {item.get('media_type')}, expected {want}",
            )
        return item.get("url")

    lib_image = from_library(image_library_id, "image")
    lib_audio = from_library(audio_library_id, "audio")
    lib_video = from_library(video_library_id, "video")
    lib_end_image = from_library(end_image_library_id, "image")
    lib_control = from_library(control_video_library_id, "video")

    if job_type == "p2v" and not control_video and not lib_control:
        raise HTTPException(400, "Pose-guided generation needs a driving video")
    if control_type and control_type not in (
        "pose", "depth", "canny", "gray", "flow", "raw"
    ):
        raise HTTPException(400, f"Unknown control type: {control_type}")

    # Require media for certain types
    if job_type in ("i2v", "i2i", "ia2v") and not image and not lib_image:
        raise HTTPException(400, "Image is required for this job type")
    if job_type == "ia2v" and not audio and not lib_audio:
        raise HTTPException(400, "Audio is required for Image+Audio → Video")
    if job_type == "v2v" and not video and not image and not lib_video and not lib_image:
        raise HTTPException(400, "Video or start image is required for Video → Video")

    upload_dir = BASE / "static" / "uploads" / "jobs"
    upload_dir.mkdir(parents=True, exist_ok=True)

    async def save_upload(uf: Optional[UploadFile], prefix: str) -> Optional[str]:
        if not uf or not uf.filename:
            return None
        ext = Path(uf.filename).suffix.lower() or ".bin"
        name = f"{prefix}_{user['id'][:8]}_{Path(uf.filename).stem[:40]}{ext}"
        # sanitize
        name = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
        dest = upload_dir / name
        content = await uf.read()
        dest.write_bytes(content)
        return f"/static/uploads/jobs/{name}"

    image_url = await save_upload(image, "img") or lib_image
    audio_url = await save_upload(audio, "aud") or lib_audio
    video_url = await save_upload(video, "vid") or lib_video
    end_image_url = await save_upload(end_image, "end") or lib_end_image
    control_video_url = await save_upload(control_video, "ctl") or lib_control

    params = {
        "resolution": resolution,
        "steps": steps,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "duration_seconds": duration_seconds,
        "fps": fps,
        "model": model,
        "model_type": model if model and model != "auto" else None,
        "negative_prompt": negative_prompt,
        "image_url": image_url,
        "audio_url": audio_url,
        "video_url": video_url,
        "end_image_url": end_image_url,
        "control_video_url": control_video_url,
        "control_type": control_type if control_video_url else None,
        "control_strength": control_strength,
    }
    if mode == "easy":
        # Ceiling is a guard against runaway values, not a quality cap —
        # the broadcast preset legitimately needs more than 25.
        params["steps"] = min(params["steps"], 40)
        params["resolution"] = params.get("resolution") or "832x480"
        if job_type in ("t2v", "i2v", "ia2v", "v2v", "p2v"):
            params["duration_seconds"] = min(float(params["duration_seconds"]), 5)

    prompt_clean = (prompt or "").strip()
    if not prompt_clean and job_type in ("t2v", "t2i"):
        raise HTTPException(400, "Prompt is required for text-based generation")

    settings = db.get_settings()
    ok_start, reason = db.can_start_job(user["id"], settings)
    if not ok_start and reason != "queued":
        raise HTTPException(429, reason)

    job = db.create_job(
        user_id=user["id"],
        job_type=job_type,
        mode=mode,
        prompt=prompt_clean or f"{job_type} generation",
        params=params,
        title=(title or "").strip(),
    )
    if ok_start:
        background_tasks.add_task(process_job, job["id"])
    else:
        # stays queued until capacity frees
        db.update_job(job["id"], {"status": "queued", "progress": 0})
    return job


# ─── Personal library ─────────────────────────────────────────────────────────

class LibraryUpdateIn(BaseModel):
    title: Optional[str] = None
    tags: Optional[list[str]] = None
    favorite: Optional[bool] = None


@app.get("/api/library")
def list_library(
    user: dict = Depends(auth.get_current_user),
    media_type: str = "",
    favorites: bool = False,
    search: str = "",
    limit: int = 500,
):
    items = db.get_library(
        user["id"],
        media_type=media_type or None,
        favorites_only=bool(favorites),
        search=search,
        limit=max(1, min(int(limit or 500), 2000)),
    )
    return {"items": items, "stats": db.library_stats(user["id"])}


@app.get("/api/library/showcase")
def library_showcase(user: dict = Depends(auth.get_current_user)):
    """Latest real result per job type, used as Create-page card previews."""
    return db.library_showcase(user["id"])


@app.get("/api/library/{item_id}")
def get_library_item(item_id: str, user: dict = Depends(auth.get_current_user)):
    item = db.get_library_item(item_id, user_id=user["id"])
    if not item:
        raise HTTPException(404, "Not found")
    return item


@app.patch("/api/library/{item_id}")
def patch_library_item(
    item_id: str,
    body: LibraryUpdateIn,
    user: dict = Depends(auth.get_current_user),
):
    updated = db.update_library_item(
        item_id, user["id"],
        {"title": body.title, "tags": body.tags, "favorite": body.favorite},
    )
    if not updated:
        raise HTTPException(404, "Not found")
    return updated


@app.delete("/api/library/{item_id}")
def remove_library_item(item_id: str, user: dict = Depends(auth.get_current_user)):
    removed = db.delete_library_item(
        item_id, user_id=None if user["role"] == "admin" else user["id"]
    )
    if not removed:
        raise HTTPException(404, "Not found or not allowed")
    # Only unlink the file when no other library row references it.
    url = removed.get("url") or ""
    if url.startswith("/static/results/") and not db.library_url_in_use(url, exclude_id=item_id):
        try:
            (BASE / url.lstrip("/")).unlink(missing_ok=True)
        except Exception:
            pass
    return {"ok": True}


@app.post("/api/library/import")
async def import_to_library(
    user: dict = Depends(auth.get_current_user),
    file: UploadFile = File(...),
    title: str = Form(""),
):
    """Upload an external file straight into the library so it can seed a job."""
    if not file or not file.filename:
        raise HTTPException(400, "No file provided")
    ext = Path(file.filename).suffix.lower() or ".bin"
    if ext not in (".png", ".jpg", ".jpeg", ".webp", ".gif",
                   ".mp4", ".webm", ".mov", ".mp3", ".wav", ".m4a"):
        raise HTTPException(400, f"Unsupported file type: {ext}")
    lib_dir = BASE / "static" / "library"
    lib_dir.mkdir(parents=True, exist_ok=True)
    stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in Path(file.filename).stem[:40])
    name = f"{user['id'][:8]}_{uuid4().hex[:8]}_{stem}{ext}"
    dest = lib_dir / name
    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty file")
    dest.write_bytes(content)
    url = f"/static/library/{name}"
    media = "video" if ext in (".mp4", ".webm", ".mov") else (
        "audio" if ext in (".mp3", ".wav", ".m4a") else "image")
    return db.add_library_item(
        user_id=user["id"], url=url, media_type=media,
        title=title or Path(file.filename).stem, source="uploaded",
    )


@app.delete("/api/jobs/{job_id}")
def remove_job(job_id: str, user: dict = Depends(auth.get_current_user)):
    ok = db.delete_job(job_id, user_id=None if user["role"] == "admin" else user["id"])
    if not ok:
        raise HTTPException(404, "Job not found or not allowed")
    return {"ok": True}


# ─── HTML pages ───────────────────────────────────────────────────────────────

def _page(request: Request, name: str, **ctx):
    s = db.get_settings()
    return templates.TemplateResponse(
        request,
        name,
        context={
            "settings": s,
            **ctx,
        },
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return _page(request, "index.html")


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    # Login is a popup on the landing page only — bounce any direct/legacy
    # link to "/login" back to "/" with a flag that auto-opens the modal.
    next_url = request.query_params.get("next", "")
    qs = f"?login=1&next={next_url}" if next_url else "?login=1"
    return RedirectResponse(url=f"/{qs}")


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    # Signup is a popup on the landing page only — same treatment as /login.
    next_url = request.query_params.get("next", "")
    qs = f"?register=1&next={next_url}" if next_url else "?register=1"
    return RedirectResponse(url=f"/{qs}")


@app.get("/generate", response_class=HTMLResponse)
def generate_page(request: Request):
    return _page(request, "generate.html")


@app.get("/library", response_class=HTMLResponse)
def library_page(request: Request):
    return _page(request, "library.html")


@app.get("/jobs", response_class=HTMLResponse)
def jobs_page(request: Request):
    return _page(request, "jobs.html")


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request):
    return _page(request, "admin.html")


@app.get("/admin/branding", response_class=HTMLResponse)
def branding_page(request: Request):
    return _page(request, "admin_branding.html")


@app.get("/admin/server", response_class=HTMLResponse)
def server_page(request: Request):
    return _page(request, "admin_server.html")


@app.get("/admin/users", response_class=HTMLResponse)
def users_page(request: Request):
    return _page(request, "admin_users.html")
