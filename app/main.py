"""
Opensource Generative AI – Modern frontend for Wan2GP
User/Job management • Branding • Easy & Advanced generation modes
"""
import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

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
from .generation import process_job, test_wan2gp_connection, BACKEND_ID, BACKEND_BUILT, mcp_call_tool, list_models_for_job_type

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("genai")

BASE = Path(__file__).parent.parent
app = FastAPI(title="Opensource Generative AI", version="1.0.0")
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
    default_resolution: str = "1280x704"
    default_steps: int = 8
    allow_mock_fallback: bool = True


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
    return db.update_settings({
        "wan2gp_url": (body.wan2gp_url or "").strip(),
        "wan2gp_root": (body.wan2gp_root or "").strip(),
        "wan2gp_mcp_url": (body.wan2gp_mcp_url or "").strip(),
        "wan2gp_enabled": body.wan2gp_enabled,
        "wan2gp_cli_args": (body.wan2gp_cli_args or "").strip(),
        "default_model_type": (body.default_model_type or "").strip() or "ltx2_22B_distilled",
        "default_resolution": (body.default_resolution or "").strip() or "1280x704",
        "default_steps": int(body.default_steps or 8),
        "allow_mock_fallback": bool(body.allow_mock_fallback),
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
    if job.get("status") not in ("failed", "completed"):
        raise HTTPException(400, "Only failed (or completed) jobs can be retried")

    db.update_job(job_id, {
        "status": "queued",
        "progress": 0,
        "error": None,
        "result_url": None,
        "preview_url": None,
        "completed_at": None,
    })
    background_tasks.add_task(process_job, job_id)
    return db.get_job(job_id)


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
):
    allowed = {"t2v", "i2v", "t2i", "i2i", "ia2v", "v2v"}
    if job_type not in allowed:
        raise HTTPException(400, f"Invalid job_type. Allowed: {sorted(allowed)}")
    if mode not in ("easy", "advanced"):
        raise HTTPException(400, "mode must be easy or advanced")

    # Require media for certain types
    if job_type in ("i2v", "i2i", "ia2v") and not image:
        raise HTTPException(400, "Image is required for this job type")
    if job_type == "ia2v" and not audio:
        raise HTTPException(400, "Audio is required for Image+Audio → Video")
    if job_type == "v2v" and not video and not image:
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

    image_url = await save_upload(image, "img")
    audio_url = await save_upload(audio, "aud")
    video_url = await save_upload(video, "vid")
    end_image_url = await save_upload(end_image, "end")

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
    }
    if mode == "easy":
        params["steps"] = min(params["steps"], 25)
        params["resolution"] = params.get("resolution") or "832x480"
        if job_type in ("t2v", "i2v", "ia2v", "v2v"):
            params["duration_seconds"] = min(float(params["duration_seconds"]), 5)

    prompt_clean = (prompt or "").strip()
    if not prompt_clean and job_type in ("t2v", "t2i"):
        raise HTTPException(400, "Prompt is required for text-based generation")

    job = db.create_job(
        user_id=user["id"],
        job_type=job_type,
        mode=mode,
        prompt=prompt_clean or f"{job_type} generation",
        params=params,
        title=(title or "").strip(),
    )
    background_tasks.add_task(process_job, job["id"])
    return job


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
    return _page(request, "login.html")


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return _page(request, "register.html")


@app.get("/generate", response_class=HTMLResponse)
def generate_page(request: Request):
    return _page(request, "generate.html")


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
