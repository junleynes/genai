"""
LTX Creative Lab – Modern frontend for Wan2GP
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

from . import auth, catalog, credits, db
from . import generation as gen_mod
from .generation import CREATIVE_LAB_TYPES, process_job, test_wan2gp_connection, BACKEND_ID, BACKEND_BUILT, mcp_call_tool, list_models_for_job_type, list_loras_for_model, try_start_queued_jobs, mcp_discover_tools, mcp_ensure_session

# ─── Job types ─────────────────────────────────────────────────────────────
# Single source of truth for every place that validates or branches on a job
# type. Creative Lab types come from generation.CREATIVE_LAB_LORA, so adding
# a new LTX IC-LoRA card there extends validation, settings and defaults here.
BASE_JOB_TYPES = ("t2v", "i2v", "ia2v", "v2v", "p2v", "t2i", "i2i", "cs", "fs", "msr")
ALL_JOB_TYPES = BASE_JOB_TYPES + CREATIVE_LAB_TYPES
IMAGE_JOB_TYPES = ("t2i", "i2i", "cs")
VIDEO_JOB_TYPES = tuple(t for t in ALL_JOB_TYPES if t not in IMAGE_JOB_TYPES)
# Creative Lab cards driven by a still image instead of a source video.
CREATIVE_IMAGE_INPUT_TYPES = ("ingredients", "cinemagraph")
CREATIVE_VIDEO_SOURCE_TYPES = tuple(
    t for t in CREATIVE_LAB_TYPES if t not in CREATIVE_IMAGE_INPUT_TYPES
)
# The UI catalog and the backend must describe the same set of tools.
assert set(catalog.TOOL_IDS) == set(ALL_JOB_TYPES), (
    "app/catalog.py TOOLS and ALL_JOB_TYPES disagree: "
    f"{sorted(set(catalog.TOOL_IDS) ^ set(ALL_JOB_TYPES))}"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("genai")

BASE = Path(__file__).parent.parent

app = FastAPI(title="LTX Creative Lab", version="1.0.0")


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

    # Recover jobs left in "processing" after a process restart, then start the queue.
    try:
        recovered = db.recover_stale_jobs(stale_minutes=0)  # all processing → queued
        if recovered.get("requeued"):
            logger.info(
                "queue.recover requeued=%s ids=%s",
                recovered["requeued"],
                recovered.get("ids"),
            )
        started = try_start_queued_jobs()
        if started:
            logger.info("queue.startup started=%s", started)
            _kick(started)
    except Exception:
        logger.exception("queue startup recovery failed")

    logger.info("LTX Creative Lab ready")
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE / "templates"))


# ─── Template helpers ─────────────────────────────────────────────────────────

def _hex_rgb(value: str) -> tuple[float, float, float]:
    h = (value or "").strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]
    except (ValueError, IndexError):
        return (0.39, 0.40, 0.95)  # indigo fallback for a malformed setting


def _luminance(rgb: tuple[float, float, float]) -> float:
    def ch(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (ch(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _contrast(a: float, b: float) -> float:
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def brand_ink(value: str) -> str:
    """Text colour that stays readable on a solid brand-colour fill. A light
    brand (lime, yellow) needs near-black text; a dark one needs white."""
    lum = _luminance(_hex_rgb(value))
    return "#0a0a0b" if _contrast(lum, 0.0) >= _contrast(lum, 1.0) else "#ffffff"


def brand_text(value: str) -> str:
    """Brand colour usable as *text* on a light surface. Light brands fail
    contrast on white, so darken them until they reach WCAG AA (4.5:1)."""
    r, g, b = _hex_rgb(value)
    for _ in range(40):
        if _contrast(_luminance((r, g, b)), 1.0) >= 4.5:
            break
        r, g, b = r * 0.93, g * 0.93, b * 0.93
    return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))


def asset(path: str) -> str:
    """Static URL with the file's mtime as a version, so replacing a file
    (e.g. a card image) is picked up by every page without a hard refresh."""
    rel = path.lstrip("/").removeprefix("static/")
    try:
        v = int((BASE / "static" / rel).stat().st_mtime)
    except OSError:
        return f"/static/{rel}"
    return f"/static/{rel}?v={v}"


templates.env.filters["brand_ink"] = brand_ink
templates.env.filters["brand_text"] = brand_text
templates.env.globals["asset"] = asset
templates.env.globals["visible_tools"] = catalog.visible_tools
templates.env.globals["all_tools"] = catalog.all_tools
templates.env.globals["categories_for"] = catalog.categories_for
templates.env.globals["TOOL_STATUSES"] = catalog.TOOL_STATUSES
templates.env.globals["TOOL_STATUS_LABELS"] = catalog.STATUS_LABELS
templates.env.globals["TOOL_CATEGORIES"] = catalog.CATEGORIES



@app.get("/api/version")
def api_version():
    """Use this to verify the deployed code is MCP (not Gradio mock)."""
    return {
        "app": "LTX Creative Lab",
        "backend": BACKEND_ID,
        "built": BACKEND_BUILT,
        "mock": False,
        "expects_mcp_url_suffix": "/mcp/",
    }

@app.on_event("startup")
def startup():
    db.ensure_admin()
    logger.info("LTX Creative Lab ready")


# ─── Pydantic models ──────────────────────────────────────────────────────────

class RegisterIn(BaseModel):
    email: str
    password: str = Field(min_length=6)
    name: str = Field(min_length=1, max_length=80)


class LoginIn(BaseModel):
    email: str
    password: str


class JobCreateIn(BaseModel):
    job_type: str = Field(pattern="^(" + "|".join(ALL_JOB_TYPES) + ")$")
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
    default_model_type: str = "ltx2_22B_distilled_1_1"
    default_image_model_type: str = "flux_dev"
    default_model_t2v: Optional[str] = None
    default_model_i2v: Optional[str] = None
    default_model_ia2v: Optional[str] = None
    default_model_v2v: Optional[str] = None
    default_model_p2v: Optional[str] = None
    default_model_t2i: Optional[str] = None
    default_model_i2i: Optional[str] = None
    default_model_cs: Optional[str] = None
    default_model_fs: Optional[str] = None
    default_model_msr: Optional[str] = None
    default_loras_t2v: Optional[str] = None
    default_loras_i2v: Optional[str] = None
    default_loras_ia2v: Optional[str] = None
    default_loras_v2v: Optional[str] = None
    default_loras_p2v: Optional[str] = None
    default_loras_t2i: Optional[str] = None
    default_loras_i2i: Optional[str] = None
    default_loras_cs: Optional[str] = None
    default_loras_fs: Optional[str] = None
    default_loras_msr: Optional[str] = None
    default_loras_ingredients: Optional[str] = None
    default_loras_outpaint: Optional[str] = None
    default_loras_cleanplate: Optional[str] = None
    default_loras_relight: Optional[str] = None
    default_loras_daynight: Optional[str] = None
    default_loras_colorize: Optional[str] = None
    default_loras_upscale: Optional[str] = None
    default_loras_foley: Optional[str] = None
    default_model_ingredients: Optional[str] = None
    default_model_outpaint: Optional[str] = None
    default_model_cleanplate: Optional[str] = None
    default_model_relight: Optional[str] = None
    default_model_daynight: Optional[str] = None
    default_model_colorize: Optional[str] = None
    default_model_upscale: Optional[str] = None
    default_model_foley: Optional[str] = None
    default_model_water: Optional[str] = None
    default_loras_water: Optional[str] = None
    default_model_deblur: Optional[str] = None
    default_loras_deblur: Optional[str] = None
    default_model_decompress: Optional[str] = None
    default_loras_decompress: Optional[str] = None
    default_model_crosseyed: Optional[str] = None
    default_loras_crosseyed: Optional[str] = None
    default_model_shave: Optional[str] = None
    default_loras_shave: Optional[str] = None
    default_model_cinemagraph: Optional[str] = None
    default_loras_cinemagraph: Optional[str] = None
    default_resolution: str = "1280x704"
    default_steps: int = 8
    default_guidance_scale: float = 7.5
    default_quality_preset: str = "balanced"
    allow_mock_fallback: bool = False
    queue_enabled: bool = True
    max_concurrent_jobs: int = 1
    concurrent_scope: str = "overall"  # overall | per_user
    mcp_timeout_s: Optional[int] = None
    stale_job_minutes: Optional[int] = None
    # Optional/None so a stale admin page that omits a field cannot wipe it
    # (db.update_settings skips None values).
    wan2gp_outputs_http_base: Optional[str] = None
    wan2gp_input_dir: Optional[str] = None
    wan2gp_input_remote_prefix: Optional[str] = None
    wan2gp_input_http_base: Optional[str] = None
    wan2gp_input_http_token: Optional[str] = None
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
    bonus = int(db.get_settings().get("credits_signup_bonus") or 0)
    if bonus > 0:
        db.adjust_credits("user", user["id"], bonus, "Sign-up bonus")
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


class ToolStatusIn(BaseModel):
    tool_status: dict[str, str]


@app.get("/api/admin/tools")
def get_tools(admin: dict = Depends(auth.require_admin)):
    return {
        "statuses": list(catalog.TOOL_STATUSES),
        "tools": catalog.all_tools(db.get_settings()),
    }


@app.put("/api/admin/tools")
def update_tools(body: ToolStatusIn, admin: dict = Depends(auth.require_admin)):
    try:
        clean = catalog.validate_statuses(body.tool_status)
    except ValueError as e:
        raise HTTPException(400, str(e))
    # Merge so a partial update doesn't reset tools it didn't mention.
    current = dict(db.get_settings().get("tool_status") or {})
    current.update(clean)
    db.update_settings({"tool_status": current})
    return {"ok": True, "tools": catalog.all_tools(db.get_settings())}


# ─── Credits ──────────────────────────────────────────────────────────────────

def _credits_exempt(user: dict, settings: dict) -> bool:
    if not settings.get("credits_enabled"):
        return True
    return user.get("role") == "admin" and bool(settings.get("credits_admins_unlimited", True))


def _charge_or_402(user: dict, settings: dict, job_type: str, params: dict, job_id: str) -> dict:
    """Price the job and reserve credits. Returns the record to store on the
    job; raises 402 when the balance can't cover it."""
    quote = credits.job_cost(settings, job_type, params)
    if _credits_exempt(user, settings) or quote["cost"] == 0:
        return {"cost": quote["cost"], "status": "free", **{k: quote[k] for k in (
            "base", "resolution_multiplier", "duration_multiplier")}}
    try:
        rec = db.reserve_credits(user["id"], quote["cost"], job_id)
    except db.InsufficientCredits as e:
        raise HTTPException(402, str(e))
    rec.update({k: quote[k] for k in ("base", "resolution_multiplier", "duration_multiplier")})
    return rec


@app.get("/api/me/credits")
def my_credits(user: dict = Depends(auth.get_current_user)):
    s = db.get_settings()
    return {"enabled": bool(s.get("credits_enabled")), "exempt": _credits_exempt(user, s),
            **db.credit_balances(user["id"])}


@app.get("/api/credits/estimate")
def credit_estimate(job_type: str, resolution: str = "832x480", duration_seconds: float = 4,
                    mode: str = "easy", user: dict = Depends(auth.get_current_user)):
    if job_type not in ALL_JOB_TYPES:
        raise HTTPException(400, "Unknown job type")
    s = db.get_settings()
    if mode == "easy" and job_type in VIDEO_JOB_TYPES:
        duration_seconds = min(float(duration_seconds), 5)   # mirrors job creation
    quote = credits.estimate(s, job_type, resolution, duration_seconds)
    bal = db.credit_balances(user["id"])
    exempt = _credits_exempt(user, s)
    return {**quote, "enabled": bool(s.get("credits_enabled")), "exempt": exempt,
            "available": bal["available"],
            "affordable": exempt or quote["cost"] <= bal["available"]}


class CreditSettingsIn(BaseModel):
    enabled: Optional[bool] = None
    admins_unlimited: Optional[bool] = None
    signup_bonus: Optional[int] = None
    tool_costs: Optional[dict[str, Any]] = None


class CreditAdjustIn(BaseModel):
    account_type: str            # user | group
    account_id: str
    delta: int
    note: str = ""


class GroupIn(BaseModel):
    name: str
    credits: int = 0


class GroupPatchIn(BaseModel):
    name: str


class UserGroupIn(BaseModel):
    group_id: Optional[str] = None


@app.get("/api/admin/credits")
def admin_credits(admin: dict = Depends(auth.require_admin)):
    s = db.get_settings()
    users = db.get_users()
    groups = db.get_groups()
    names = {u["id"]: u.get("name") or u.get("email") for u in users}
    gnames = {g["id"]: g["name"] for g in groups}
    ledger = db.get_ledger(100)
    for e in ledger:
        kind, _, aid = (e.get("account") or "").partition(":")
        e["account_type"] = kind
        e["account_name"] = names.get(aid) if kind == "user" else gnames.get(aid, "(deleted group)")
        e["user_name"] = names.get(e.get("user_id") or "")
        e["actor_name"] = names.get(e.get("actor_id") or "")
    costs = credits.tool_costs(s)
    return {
        "settings": {
            "enabled": bool(s.get("credits_enabled")),
            "admins_unlimited": bool(s.get("credits_admins_unlimited", True)),
            "signup_bonus": int(s.get("credits_signup_bonus") or 0),
        },
        "tools": [{**t, "cost": costs[t["id"]], "default": credits.DEFAULT_COSTS.get(t["id"], 1)}
                  for t in catalog.all_tools(s)],
        "resolution_tiers": [{"label": l, "multiplier": m} for _, m, l in credits.RES_TIERS],
        "duration_unit_seconds": credits.DURATION_UNIT_S,
        "users": [{"id": u["id"], "name": u.get("name"), "email": u.get("email"),
                   "role": u.get("role"), "is_active": u.get("is_active", True),
                   "credits": int(u.get("credits") or 0), "group_id": u.get("group_id")}
                  for u in users],
        "groups": groups,
        "ledger": ledger,
    }


@app.put("/api/admin/credits/settings")
def admin_credit_settings(body: CreditSettingsIn, admin: dict = Depends(auth.require_admin)):
    upd: dict[str, Any] = {}
    if body.enabled is not None:
        upd["credits_enabled"] = body.enabled
    if body.admins_unlimited is not None:
        upd["credits_admins_unlimited"] = body.admins_unlimited
    if body.signup_bonus is not None:
        if body.signup_bonus < 0:
            raise HTTPException(400, "Sign-up bonus can't be negative")
        upd["credits_signup_bonus"] = int(body.signup_bonus)
    if body.tool_costs is not None:
        try:
            clean = credits.validate_costs(body.tool_costs)
        except ValueError as e:
            raise HTTPException(400, str(e))
        merged = dict(db.get_settings().get("credit_tool_costs") or {})
        merged.update(clean)
        upd["credit_tool_costs"] = merged
    db.update_settings(upd)
    return {"ok": True}


@app.post("/api/admin/credits/adjust")
def admin_adjust_credits(body: CreditAdjustIn, admin: dict = Depends(auth.require_admin)):
    try:
        new = db.adjust_credits(body.account_type, body.account_id, body.delta,
                                body.note.strip(), actor_id=admin["id"])
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'"))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "balance": new}


@app.post("/api/admin/groups")
def admin_create_group(body: GroupIn, admin: dict = Depends(auth.require_admin)):
    try:
        return db.create_group(body.name, body.credits, actor_id=admin["id"])
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.patch("/api/admin/groups/{group_id}")
def admin_rename_group(group_id: str, body: GroupPatchIn, admin: dict = Depends(auth.require_admin)):
    try:
        return db.rename_group(group_id, body.name)
    except KeyError:
        raise HTTPException(404, "Group not found")
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/admin/groups/{group_id}")
def admin_delete_group(group_id: str, admin: dict = Depends(auth.require_admin)):
    try:
        return {"ok": True, "members_ungrouped": db.delete_group(group_id)}
    except KeyError:
        raise HTTPException(404, "Group not found")


@app.put("/api/admin/users/{user_id}/group")
def admin_set_user_group(user_id: str, body: UserGroupIn, admin: dict = Depends(auth.require_admin)):
    if body.group_id and not db.get_group(body.group_id):
        raise HTTPException(404, "Group not found")
    updated = db.update_user(user_id, {"group_id": body.group_id})
    if not updated:
        raise HTTPException(404, "User not found")
    return updated


def _require_usable_tool(job_type: str) -> None:
    """Coming-soon and disabled tools can't take jobs, however they're reached
    (hidden card, stale tab, direct API call, or a retry of an old job)."""
    status = catalog.tool_status(db.get_settings(), job_type)
    if status not in catalog.USABLE_STATUSES:
        label = catalog.STATUS_LABELS.get(status, status).lower()
        raise HTTPException(403, f"This tool is {label} and can't be used right now.")


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
        "default_model_type": (body.default_model_type or "").strip() or "ltx2_22B_distilled_1_1",
        "default_image_model_type": (body.default_image_model_type or "").strip() or "flux_dev",
        **{
            f"default_model_{_jt}": (
                (getattr(body, f"default_model_{_jt}") or "").strip()
                if getattr(body, f"default_model_{_jt}") is not None else None
            )
            for _jt in ALL_JOB_TYPES
        },
        **{
            f"default_loras_{_jt}": (
                (getattr(body, f"default_loras_{_jt}") or "").strip()
                if getattr(body, f"default_loras_{_jt}") is not None else None
            )
            for _jt in ALL_JOB_TYPES
        },
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
        "mcp_timeout_s": (
            max(60, int(body.mcp_timeout_s))
            if body.mcp_timeout_s is not None else None
        ),
        "stale_job_minutes": (
            max(1, int(body.stale_job_minutes))
            if body.stale_job_minutes is not None else None
        ),
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
        "wan2gp_input_http_base": (
            body.wan2gp_input_http_base.strip().rstrip("/")
            if body.wan2gp_input_http_base is not None else None
        ),
        "wan2gp_input_http_token": (
            body.wan2gp_input_http_token.strip()
            if body.wan2gp_input_http_token is not None else None
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


@app.get("/api/admin/health")
async def admin_health(admin: dict = Depends(auth.require_admin)):
    """Queue depth, MCP reachability, outputs/inputs HTTP probes."""
    import time
    settings = db.get_settings()
    stats = db.queue_stats()
    mcp_url = (settings.get("wan2gp_mcp_url") or "").strip()
    enabled = bool(settings.get("wan2gp_enabled"))

    mcp = {"ok": False, "message": "MCP URL not set", "latency_ms": None, "tools": 0}
    if mcp_url and enabled:
        t0 = time.time()
        try:
            sid = await asyncio.to_thread(mcp_ensure_session, mcp_url, 15.0)
            tools = await asyncio.to_thread(mcp_discover_tools, mcp_url, True)
            mcp = {
                "ok": True,
                "message": f"session={bool(sid)} tools={len(tools)}",
                "latency_ms": int((time.time() - t0) * 1000),
                "tools": len(tools),
            }
        except Exception as e:
            mcp = {
                "ok": False,
                "message": str(e)[:400],
                "latency_ms": int((time.time() - t0) * 1000),
                "tools": 0,
            }
    elif mcp_url and not enabled:
        mcp = {"ok": False, "message": "WanGP generation is disabled", "latency_ms": None, "tools": 0}

    async def _probe(url: str, label: str) -> dict:
        if not url:
            return {"ok": False, "message": f"{label} not configured", "latency_ms": None}
        t0 = time.time()
        try:
            import httpx
            async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
                r = await client.get(url if "://" in url else f"http://{url}")
            return {
                "ok": r.status_code < 500,
                "message": f"HTTP {r.status_code}",
                "latency_ms": int((time.time() - t0) * 1000),
            }
        except Exception as e:
            return {
                "ok": False,
                "message": str(e)[:300],
                "latency_ms": int((time.time() - t0) * 1000),
            }

    outputs = await _probe(
        (settings.get("wan2gp_outputs_http_base") or "").strip(),
        "Outputs HTTP",
    )
    inputs = await _probe(
        (settings.get("wan2gp_input_http_base") or "").strip(),
        "Inputs HTTP",
    )

    return {
        "ok": bool(mcp.get("ok")),
        "queue": stats,
        "mcp": mcp,
        "outputs_http": outputs,
        "inputs_http": inputs,
        "settings": {
            "wan2gp_enabled": enabled,
            "queue_enabled": bool(settings.get("queue_enabled", True)),
            "max_concurrent_jobs": int(settings.get("max_concurrent_jobs") or 1),
            "concurrent_scope": settings.get("concurrent_scope") or "overall",
            "mcp_timeout_s": int(settings.get("mcp_timeout_s") or 3600),
            "stale_job_minutes": int(settings.get("stale_job_minutes") or 30),
            "mcp_url": mcp_url or None,
        },
    }


@app.post("/api/admin/diagnose")
async def admin_diagnose(admin: dict = Depends(auth.require_admin)):
    """Run WanGP MCP diagnostics (tools, LoRAs, p2v capability) and return text report."""
    settings = db.get_settings()
    mcp_url = (settings.get("wan2gp_mcp_url") or "").strip()
    if not mcp_url:
        return {"ok": False, "report": "No MCP URL configured in Admin → Server & Queue."}

    lines: list[str] = [f"MCP URL: {mcp_url}", ""]
    ok = True
    try:
        tools = await asyncio.to_thread(mcp_discover_tools, mcp_url, True)
    except Exception as e:
        return {
            "ok": False,
            "report": f"FAILED to list tools: {e}\nCheck URL ends with /mcp/ and WanGP MCP is running.",
        }

    lines.append("=" * 60)
    lines.append("1. TOOLS")
    lines.append("=" * 60)
    if not tools:
        ok = False
        lines.append("  (none)")
    else:
        for t in tools:
            from .generation import _tool_param_names
            params = ", ".join(_tool_param_names(t)) or "-"
            lines.append(f"  {str(t.get('name', '?')):40} params: {params}")
        lines.append(f"\n  {len(tools)} tool(s) total")

    # LoRA tool probe
    lines.append("")
    lines.append("=" * 60)
    lines.append("2. LORA LISTING")
    lines.append("=" * 60)
    lora_tools = [
        str(t.get("name"))
        for t in tools
        if "lora" in str(t.get("name") or "").lower()
    ]
    if not lora_tools:
        lines.append("  No LoRA-related tool advertised.")
    else:
        lines.append(f"  Candidates: {', '.join(lora_tools)}")
        try:
            sample = await asyncio.to_thread(list_loras_for_model, mcp_url, "")
            if isinstance(sample, tuple):
                sample, supported = sample
            else:
                supported = True
            n = len(sample) if isinstance(sample, list) else 0
            lines.append(f"  list_loras_for_model returned {n} item(s) supported={supported}")
        except Exception as e:
            lines.append(f"  list_loras_for_model error: {e}")

    # p2v models
    lines.append("")
    lines.append("=" * 60)
    lines.append("3. P2V / CONTROL-CAPABLE MODELS")
    lines.append("=" * 60)
    try:
        models = await asyncio.to_thread(list_models_for_job_type, mcp_url, "p2v")
        lines.append(f"  {len(models)} model(s) offered for p2v")
        for m in (models or [])[:15]:
            if isinstance(m, dict):
                lines.append(f"    - {m.get('id') or m.get('model_type') or m.get('name') or m}")
            else:
                lines.append(f"    - {m}")
        if len(models or []) > 15:
            lines.append(f"    … +{len(models) - 15} more")
    except Exception as e:
        ok = False
        lines.append(f"  FAILED: {e}")

    return {"ok": ok, "report": "\n".join(lines)}


@app.post("/api/admin/queue/recover")
async def admin_queue_recover(
    background_tasks: BackgroundTasks,
    admin: dict = Depends(auth.require_admin),
):
    """Manually re-queue stale processing jobs and start capacity."""
    recovered = db.recover_stale_jobs()  # uses stale_job_minutes from settings
    started = try_start_queued_jobs()
    for jid in started:
        background_tasks.add_task(process_job, jid)
    return {
        "ok": True,
        "requeued": recovered.get("requeued", 0),
        "ids": recovered.get("ids") or [],
        "started": started,
        "queue": db.queue_stats(),
    }



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
    _require_usable_tool(job.get("job_type") or "")

    # A retry is a new generation, so it's priced and paid for again (the
    # failed/cancelled attempt was already refunded). Charge the job's owner.
    owner = db.get_user_by_id(job["user_id"]) or user
    new_rec = _charge_or_402(owner, db.get_settings(), job.get("job_type") or "",
                             job.get("params") or {}, job_id)
    history = list(job.get("credits_history") or [])
    if job.get("credits"):
        history.append(job["credits"])

    db.update_job(job_id, {
        "credits": new_rec,
        "credits_history": history,
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


@app.get("/api/loras")
async def api_loras(
    model_type: str = "",
    job_type: str = "",
    user: dict = Depends(auth.get_current_user),
):
    """
    LoRAs available for a given model. WanGP stores LoRAs in
    model-specific subdirectories, so the set changes with the model.

    When `model_type` is empty ("Auto"), resolve it to the per-job-type
    default model the job runner itself uses — WanGP's LoRA tool requires
    a model_type, so querying with none yields nothing instead of the
    default model's LoRAs.

    `supported` is False when this WanGP build exposes no LoRA-listing
    tool at all — the UI falls back to free-text entry in that case
    rather than showing an empty picker.
    """
    settings = db.get_settings()
    mcp_url = (settings.get("wan2gp_mcp_url") or "").strip()
    enabled = bool(settings.get("wan2gp_enabled"))
    if not mcp_url or not enabled:
        return {
            "ok": False,
            "loras": [],
            "supported": False,
            "message": "Wan2GP is not configured or is disabled.",
            "model_type": model_type,
        }

    resolved = (model_type or "").strip()
    if not resolved:
        per_type = (
            settings.get(f"default_model_{job_type}") or ""
            if job_type in ALL_JOB_TYPES
            else ""
        )
        resolved = (per_type or "").strip() or (settings.get("default_model_type") or "").strip()

    try:
        loras, supported = await asyncio.to_thread(
            list_loras_for_model, mcp_url, resolved
        )
        return {
            "ok": True,
            "loras": loras,
            "supported": supported,
            "count": len(loras),
            "model_type": resolved,
        }
    except Exception as e:
        logger.exception("api/loras failed")
        return {
            "ok": False,
            "loras": [],
            "supported": False,
            "message": str(e)[:500],
            "model_type": model_type,
        }


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
    reference_images: list[UploadFile] = File(default_factory=list),
    reference_image_library_ids: str = Form(""),
    msr_reference_video_length: Optional[int] = Form(None),
    sheet_layout: str = Form("turnaround"),
    loras: str = Form(""),
    # Reuse existing library media instead of uploading (ids from /api/library)
    image_library_id: str = Form(""),
    audio_library_id: str = Form(""),
    video_library_id: str = Form(""),
    end_image_library_id: str = Form(""),
):
    allowed = set(ALL_JOB_TYPES)
    if job_type not in allowed:
        raise HTTPException(400, f"Invalid job_type. Allowed: {sorted(allowed)}")
    _require_usable_tool(job_type)
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
    if job_type == "ingredients" and not image and not lib_image:
        raise HTTPException(400, "A reference sheet image is required for Ingredients")
    if job_type == "cinemagraph" and not image and not lib_image:
        raise HTTPException(400, "A still image is required for Cinemagraph")
    if job_type in CREATIVE_VIDEO_SOURCE_TYPES:
        if not video and not lib_video:
            raise HTTPException(400, f"A source video is required for {job_type}")
    if job_type == "fs" and not image and not lib_image:
        raise HTTPException(400, "A face reference image is required for Face Swap")
    if job_type == "fs" and not video and not lib_video:
        raise HTTPException(400, "A source video is required for Face Swap")

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

    # Ordered reference images. Both VACE and LTX-2.3 MSR use the same
    # convention: background/setting first, then subjects and objects, so the
    # order the user arranged them in is meaningful and must be preserved.
    reference_image_paths: list[str] = []
    for idx, uf in enumerate(reference_images or []):
        saved = await save_upload(uf, f"ref{idx}")
        if saved:
            reference_image_paths.append(saved)
    for lid in [s.strip() for s in (reference_image_library_ids or "").split(",") if s.strip()]:
        lib_ref = from_library(lid, "image")
        if lib_ref:
            reference_image_paths.append(lib_ref)
    # MSR tops out at 5; more would be silently dropped by WanGP.
    if len(reference_image_paths) > 5:
        raise HTTPException(
            400,
            f"Too many reference images ({len(reference_image_paths)}). "
            "Multi-subject reference supports up to 5.",
        )
    if job_type == "msr" and len(reference_image_paths) < 2:
        raise HTTPException(
            400,
            "Multi-Subject Reference needs at least 2 reference images "
            "(background/setting first, then subjects and objects).",
        )

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
        "reference_image_paths": reference_image_paths,
        "msr_reference_video_length": msr_reference_video_length,
        "sheet_layout": sheet_layout if job_type == "cs" else None,
        "loras": loras.strip(),
    }
    if mode == "easy":
        # Ceiling is a guard against runaway values, not a quality cap —
        # the broadcast preset legitimately needs more than 25.
        params["steps"] = min(params["steps"], 40)
        params["resolution"] = params.get("resolution") or "832x480"
        if job_type in VIDEO_JOB_TYPES:
            params["duration_seconds"] = min(float(params["duration_seconds"]), 5)

    prompt_clean = (prompt or "").strip()
    if not prompt_clean and job_type in ("t2v", "t2i", "cs", "msr"):
        if job_type == "cs":
            has_cs_image = bool(
                (image is not None and getattr(image, "filename", None))
                or (image_library_id or "").strip()
            )
            if not has_cs_image:
                raise HTTPException(
                    400,
                    "Describe the character, or upload a reference photo of them.",
                )
        if job_type == "msr":
            raise HTTPException(400, "Describe the scene — name each reference (e.g. \"Image 1 is the background, Image 2 is the presenter…\")")
        raise HTTPException(400, "Prompt is required for text-based generation")

    settings = db.get_settings()
    ok_start, reason = db.can_start_job(user["id"], settings)
    if not ok_start and reason != "queued":
        raise HTTPException(429, reason)

    job_id = str(uuid4())
    credit_rec = _charge_or_402(user, settings, job_type, params, job_id)
    try:
        job = db.create_job(
            user_id=user["id"],
            job_type=job_type,
            mode=mode,
            prompt=prompt_clean or f"{job_type} generation",
            params=params,
            title=(title or "").strip(),
            job_id=job_id,
            extra={"credits": credit_rec},
        )
    except Exception:
        db.refund_record(user["id"], job_id, credit_rec, "Job could not be created")
        raise
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


@app.get("/admin/credits", response_class=HTMLResponse)
def credits_page(request: Request):
    return _page(request, "admin_credits.html")


@app.get("/admin/tools", response_class=HTMLResponse)
def tools_page(request: Request):
    return _page(request, "admin_tools.html")


@app.get("/admin/users", response_class=HTMLResponse)
def users_page(request: Request):
    return _page(request, "admin_users.html")
