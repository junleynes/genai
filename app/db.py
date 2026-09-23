"""Simple JSON-based database for users, jobs, and settings."""
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

USERS_FILE = DATA_DIR / "users.json"
JOBS_FILE = DATA_DIR / "jobs.json"
SETTINGS_FILE = DATA_DIR / "settings.json"
LIBRARY_FILE = DATA_DIR / "library.json"
GROUPS_FILE = DATA_DIR / "groups.json"
LEDGER_FILE = DATA_DIR / "credit_ledger.json"

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ─── Settings / Branding ───────────────────────────────────────────────────────

DEFAULT_SETTINGS = {
    "app_name": "LTX Creative Lab",
    "tagline": "Your AI creative lab",
    "footer": "© 2026 LTX Creative Lab.",
    "logo_url": "",
    "favicon_url": "",
    "primary_color": "#d1fe17",   # electric lime (Higgsfield-style); editable in Admin → Branding
    "secondary_color": "#a3e635",
    "accent_color": "#22d3ee",
    "default_theme": "system",
    "wan2gp_url": "http://localhost:7860",
    "wan2gp_root": "",
    "wan2gp_mcp_url": "",
    "wan2gp_outputs_http_base": "",  # e.g. http://HOST:8090 serving WanGP outputs/
    # Input staging for WanGP builds with no wangp_create_gallery_upload tool
    "wan2gp_input_dir": "",            # local/mounted dir both servers can see
    "wan2gp_input_remote_prefix": "",
    # Cross-host alternative to the mounted-share fields above: point at a
    # scripts/inputs_server.py instance running on the WanGP host and genai
    # uploads input files to it over HTTP instead of needing a shared mount.
    "wan2gp_input_http_base": "",   # e.g. http://WANGP_HOST:8091
    "wan2gp_input_http_token": "",  # optional, must match inputs_server.py --token
    # Override VACE guide letters if your WanGP build uses a different alphabet,
    # e.g. "pose=P,depth=D,canny=E". Empty uses the built-in defaults.
    "wan2gp_control_letters": "",  # same dir as WanGP sees it, e.g. C:\AI-Tools\Wan2GP\inputs
    "wan2gp_enabled": False,
    "wan2gp_cli_args": "--attention sdpa --profile 4",
    # Auto-model defaults. Kept separate per output medium: a single default
    # meant an image job resolved to a video model and vice versa.
    "default_model_type": "ltx2_22B_distilled_1_1",    # video jobs
    "default_image_model_type": "flux_dev",            # image jobs (t2i / i2i)
    # Per-job-type overrides. Blank falls back to the medium defaults above.
    # These exist so Easy mode can resolve a correct model with no user
    # input: job types have genuinely different needs (p2v must be VACE,
    # ia2v must be audio-capable, i2i must be edit-capable) that a single
    # video/image split cannot express.
    # Every video-output card defaults to LTX-2.3 22B Distilled 1.1.
    "default_model_t2v": "ltx2_22B_distilled_1_1",
    "default_model_i2v": "ltx2_22B_distilled_1_1",
    "default_model_ia2v": "ltx2_22B_distilled_1_1",
    "default_model_v2v": "ltx2_22B_distilled_1_1",
    "default_model_p2v": "ltx2_22B_distilled_1_1",
    "default_model_t2i": "",
    "default_model_i2i": "",
    "default_model_cs": "",
    "default_model_fs": "ltx2_22B_distilled_1_1",
    "default_model_msr": "ltx2_22B_distilled_1_1",
    "default_model_ingredients": "ltx2_22B_distilled_1_1",
    "default_model_outpaint": "ltx2_22B_distilled_1_1",
    "default_model_cleanplate": "ltx2_22B_distilled_1_1",
    "default_model_relight": "ltx2_22B_distilled_1_1",
    "default_model_daynight": "ltx2_22B_distilled_1_1",
    "default_model_colorize": "ltx2_22B_distilled_1_1",
    "default_model_upscale": "ltx2_22B_distilled_1_1",
    "default_model_foley": "ltx2_22B_distilled_1_1",
    "default_model_water": "ltx2_22B_distilled_1_1",
    "default_model_deblur": "ltx2_22B_distilled_1_1",
    "default_model_decompress": "ltx2_22B_distilled_1_1",
    "default_model_crosseyed": "ltx2_22B_distilled_1_1",
    "default_model_shave": "ltx2_22B_distilled_1_1",
    "default_model_cinemagraph": "ltx2_22B_distilled_1_1",
    # Default LoRAs per job type, "filename:weight, ..." — applied when the
    # user supplied none, so Easy mode never has to think about LoRAs.
    # Prefer exact WanGP filenames; keywords also resolve (see generation).
    "default_loras_t2v": "",
    "default_loras_i2v": "",
    "default_loras_ia2v": "id-lora-celebvhq-ltx2.3.safetensors:1.0",
    "default_loras_v2v": "",
    # Pose-driven: LTX Union Control IC-LoRA (pose/depth/canny)
    "default_loras_p2v": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors:1.0",
    "default_loras_t2i": "",
    "default_loras_i2i": "",
    "default_loras_cs": "",
    "default_loras_fs": "head_swap_v3_rank_adaptive_fro_098.safetensors:1.0",
    "default_loras_msr": "",
    # LTX-2.3 Creative Lab IC-LoRAs
    "default_loras_ingredients": "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors:1.0",
    "default_loras_outpaint": "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors:1.0",
    "default_loras_cleanplate": "ltx-2.3-22b-ic-lora-clean-plate-1.0.safetensors:1.0",
    "default_loras_relight": "ltx-2.3-22b-ic-lora-relight-1.0.safetensors:1.0",
    "default_loras_daynight": "ltx-2.3-22b-ic-lora-day-to-night-0.9.safetensors:1.0",
    "default_loras_colorize": "ltx-2.3-22b-ic-lora-colorization-0.9.safetensors:1.0",
    "default_loras_upscale": "ltx-2.3-22b-ic-lora-pixel-spatial-upscaler-x2-0.9.safetensors:1.0",
    "default_loras_foley": "ltx-2.3-22b-lora-foley-v2a-1.0.safetensors:1.0",
    # LTX-2.3 Creative Lab — edit/restore adapters (trigger words are added
    # automatically, see generation.CREATIVE_LAB_PROMPTS)
    "default_loras_water": "ltx-2.3-22b-ic-lora-water-simulation-0.9.safetensors:1.0",
    "default_loras_deblur": "ltx-2.3-22b-ic-lora-deblur-0.9.safetensors:1.0",
    "default_loras_decompress": "ltx-2.3-22b-ic-lora-decompression-0.9.safetensors:1.0",
    "default_loras_crosseyed": "lora_weights_step_03000.safetensors:1.0",
    "default_loras_shave": "ltx-2.3-22b-ic-lora-instant-shave-0.9.safetensors:1.0",
    "default_loras_cinemagraph": "ltx-2.3-22b-lora-cinemagraph-0.9.safetensors:1.0",
    # Per-tool status set in Admin → Tools: enabled | new | coming_soon | disabled.
    # Tools not listed use catalog.DEFAULT_STATUS.
    "tool_status": {},
    # Credits (Admin → Credits). Off by default so existing installs keep working.
    "credits_enabled": False,
    "credits_admins_unlimited": True,   # admins generate without spending
    "credits_signup_bonus": 0,          # personal credits for new registrations
    "credit_tool_costs": {},            # {tool_id: base cost}; unset → credits.DEFAULT_COSTS
    "default_resolution": "1280x704",
    "default_steps": 8,
    "default_fps": "24",
    "default_guidance_scale": 7.5,
    # Quality preset preselected in Easy mode: fast | balanced | quality | broadcast
    "default_quality_preset": "balanced",
    "allow_mock_fallback": False,
    # Job queue
    "queue_enabled": True,
    "max_concurrent_jobs": 1,
    "concurrent_scope": "overall",  # overall | per_user
    # How long a generation may run before we mark it failed (seconds)
    "mcp_timeout_s": 3600,
    # On startup, processing jobs with no update for this many minutes are re-queued
    "stale_job_minutes": 30,
}



def get_settings() -> dict:
    with _lock:
        data = _load(SETTINGS_FILE, {})
        merged = {**DEFAULT_SETTINGS, **data}
        return merged


def update_settings(updates: dict) -> dict:
    with _lock:
        current = _load(SETTINGS_FILE, {})
        current.update({k: v for k, v in updates.items() if v is not None})
        _save(SETTINGS_FILE, current)
        return {**DEFAULT_SETTINGS, **current}


# ─── Users ─────────────────────────────────────────────────────────────────────

def get_users() -> list[dict]:
    with _lock:
        return _load(USERS_FILE, [])


def get_user_by_id(user_id: str) -> Optional[dict]:
    for u in get_users():
        if u["id"] == user_id:
            return u
    return None


def get_user_by_email(email: str) -> Optional[dict]:
    email = email.lower().strip()
    for u in get_users():
        if u["email"].lower() == email:
            return u
    return None


def create_user(email: str, password_hash: str, name: str, role: str = "user") -> dict:
    with _lock:
        users = _load(USERS_FILE, [])
        if any(u["email"].lower() == email.lower() for u in users):
            raise ValueError("Email already registered")
        user = {
            "id": str(uuid4()),
            "email": email.lower().strip(),
            "password_hash": password_hash,
            "name": name.strip(),
            "role": role if role in ("admin", "user") else "user",
            "created_at": _now(),
            "is_active": True,
            "credits": 0,
            "group_id": None,
        }
        users.append(user)
        _save(USERS_FILE, users)
        return {k: v for k, v in user.items() if k != "password_hash"}


def update_user(user_id: str, updates: dict) -> Optional[dict]:
    with _lock:
        users = _load(USERS_FILE, [])
        for i, u in enumerate(users):
            if u["id"] == user_id:
                for k, v in updates.items():
                    if k in ("name", "role", "is_active") and v is not None:
                        users[i][k] = v
                    elif k == "group_id":   # None is meaningful: removes the user from their group
                        users[i][k] = v or None
                _save(USERS_FILE, users)
                return {k: v for k, v in users[i].items() if k != "password_hash"}
        return None


def ensure_admin():
    """Create default admin if no users exist."""
    users = get_users()
    if not users:
        from .auth import hash_password
        create_user(
            email="admin@example.com",
            password_hash=hash_password("admin123"),
            name="Administrator",
            role="admin",
        )
        print("✓ Default admin created: admin@example.com / admin123")


# ─── Jobs ──────────────────────────────────────────────────────────────────────

def get_jobs(user_id: Optional[str] = None, limit: int = 100) -> list[dict]:
    with _lock:
        jobs = _load(JOBS_FILE, [])
        if user_id:
            jobs = [j for j in jobs if j["user_id"] == user_id]
        jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return jobs[:limit]


def get_job(job_id: str) -> Optional[dict]:
    for j in get_jobs(limit=10000):
        if j["id"] == job_id:
            return j
    return None


def create_job(
    user_id: str,
    job_type: str,  # t2v | i2v | t2i | i2i
    mode: str,      # easy | advanced
    prompt: str,
    params: dict,
    title: str = "",
    job_id: Optional[str] = None,
    extra: Optional[dict] = None,
) -> dict:
    with _lock:
        jobs = _load(JOBS_FILE, [])
        job = {
            "id": job_id or str(uuid4()),
            "user_id": user_id,
            "title": title or prompt[:60] + ("…" if len(prompt) > 60 else ""),
            "job_type": job_type,
            "mode": mode,
            "prompt": prompt,
            "negative_prompt": params.get("negative_prompt", ""),
            "params": params,
            "status": "queued",  # queued | processing | completed | failed
            "progress": 0,
            "result_url": None,
            "preview_url": None,
            "error": None,
            "created_at": _now(),
            "updated_at": _now(),
            "completed_at": None,
            **(extra or {}),
        }
        jobs.append(job)
        _save(JOBS_FILE, jobs)
        return job


def update_job(job_id: str, updates: dict) -> Optional[dict]:
    with _lock:
        jobs = _load(JOBS_FILE, [])
        for i, j in enumerate(jobs):
            if j["id"] == job_id:
                jobs[i].update(updates)
                jobs[i]["updated_at"] = _now()
                if updates.get("status") in ("completed", "failed"):
                    jobs[i]["completed_at"] = _now()
                # Every failure path (generation error, timeout, disabled
                # backend) and cancel ends here, so this is the one place a
                # reserved charge is returned.
                if updates.get("status") in REFUNDABLE_STATUSES:
                    _refund_job_locked(jobs[i], reason=f"Job {updates['status']}")
                _save(JOBS_FILE, jobs)
                return jobs[i]
        return None


def delete_job(job_id: str, user_id: Optional[str] = None) -> bool:
    with _lock:
        jobs = _load(JOBS_FILE, [])
        new_jobs = []
        found = False
        for j in jobs:
            if j["id"] == job_id:
                if user_id and j["user_id"] != user_id:
                    new_jobs.append(j)
                else:
                    found = True
                    # Deleted before it ever ran: nothing was produced.
                    if j.get("status") == "queued":
                        _refund_job_locked(j, reason="Queued job deleted")
            else:
                new_jobs.append(j)
        if found:
            _save(JOBS_FILE, new_jobs)
        return found


def count_active_jobs(user_id: Optional[str] = None) -> int:
    """Jobs currently processing (and optionally scoped to a user)."""
    with _lock:
        jobs = _load(JOBS_FILE, [])
        n = 0
        for j in jobs:
            if j.get("status") != "processing":
                continue
            if user_id and j.get("user_id") != user_id:
                continue
            n += 1
        return n


def list_queued_jobs(limit: int = 50) -> list:
    """Oldest queued jobs first."""
    with _lock:
        jobs = _load(JOBS_FILE, [])
        q = [j for j in jobs if j.get("status") == "queued"]
        q.sort(key=lambda x: x.get("created_at") or "")
        return q[:limit]



def recover_stale_jobs(stale_minutes: int | None = None) -> dict:
    """
    Jobs left in 'processing' after a process restart cannot finish — re-queue them.
    Also re-queue anything that has been processing longer than stale_minutes
    without an update (orphaned worker).
    Returns counts: {"requeued": n, "ids": [...]}.
    """
    from datetime import datetime, timezone, timedelta
    s = get_settings()
    mins = stale_minutes if stale_minutes is not None else int(s.get("stale_job_minutes") or 30)
    if mins < 1:
        mins = 1
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=mins)
    requeued = []
    with _lock:
        jobs = _load(JOBS_FILE, [])
        changed = False
        for j in jobs:
            if j.get("status") != "processing":
                continue
            # Always recover on explicit startup call (stale_minutes=0 means all processing)
            if stale_minutes == 0:
                stale = True
            else:
                ts = j.get("updated_at") or j.get("created_at") or ""
                try:
                    # support both Z and +00:00
                    ts_n = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
                    updated = datetime.fromisoformat(ts_n)
                    if updated.tzinfo is None:
                        updated = updated.replace(tzinfo=timezone.utc)
                    stale = updated < cutoff
                except Exception:
                    stale = True  # unparseable → treat as stale
            if not stale:
                continue
            j["status"] = "queued"
            j["progress"] = 0
            j["error"] = "Re-queued after server restart / stale processing state"
            j["updated_at"] = _now()
            requeued.append(j["id"])
            changed = True
        if changed:
            _save(JOBS_FILE, jobs)
    return {"requeued": len(requeued), "ids": requeued}


def queue_stats() -> dict:
    """Counts for admin health panel."""
    with _lock:
        jobs = _load(JOBS_FILE, [])
    by = {"queued": 0, "processing": 0, "completed": 0, "failed": 0, "cancelled": 0}
    for j in jobs:
        st = (j.get("status") or "").lower()
        if st in ("canceled",):
            st = "cancelled"
        if st in by:
            by[st] += 1
        else:
            by.setdefault("other", 0)
            by["other"] += 1
    by["total"] = len(jobs)
    return by


def can_start_job(user_id: str, settings: Optional[dict] = None) -> tuple:
    """
    Return (ok: bool, reason: str).
    If queue_enabled is False and capacity is full, reject new work.
    If queue_enabled is True and full, job may still be created as queued.
    """
    s = settings or get_settings()
    max_c = int(s.get("max_concurrent_jobs") or 1)
    if max_c < 1:
        max_c = 1
    scope = (s.get("concurrent_scope") or "overall").lower()
    if scope == "per_user":
        active = count_active_jobs(user_id)
    else:
        active = count_active_jobs(None)
    if active < max_c:
        return True, ""
    if s.get("queue_enabled", True):
        return False, "queued"  # may wait
    return False, f"Concurrency limit reached ({active}/{max_c}, scope={scope}). Queue is disabled."



# ─── Personal media library ───────────────────────────────────────────────────

def get_library(
    user_id: str,
    media_type: Optional[str] = None,
    favorites_only: bool = False,
    search: str = "",
    limit: int = 500,
) -> list[dict]:
    """Items owned by this user, newest first."""
    with _lock:
        items = _load(LIBRARY_FILE, [])
    out = [i for i in items if i.get("user_id") == user_id and not i.get("deleted")]
    if media_type in ("image", "video", "audio"):
        out = [i for i in out if i.get("media_type") == media_type]
    if favorites_only:
        out = [i for i in out if i.get("favorite")]
    term = (search or "").strip().lower()
    if term:
        out = [
            i for i in out
            if term in (i.get("prompt") or "").lower()
            or term in (i.get("title") or "").lower()
            or any(term in t.lower() for t in (i.get("tags") or []))
        ]
    out.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return out[:limit]


def get_library_item(item_id: str, user_id: Optional[str] = None) -> Optional[dict]:
    with _lock:
        items = _load(LIBRARY_FILE, [])
    for i in items:
        if i.get("id") == item_id and not i.get("deleted"):
            if user_id and i.get("user_id") != user_id:
                return None
            return i
    return None


def add_library_item(
    user_id: str,
    url: str,
    media_type: str,
    job_id: Optional[str] = None,
    job_type: str = "",
    prompt: str = "",
    title: str = "",
    model: str = "",
    params: Optional[dict] = None,
    source: str = "generated",
) -> dict:
    """Record a media file in the user's library. Idempotent per (job_id, url)."""
    with _lock:
        items = _load(LIBRARY_FILE, [])
        if job_id:
            for existing in items:
                if (
                    existing.get("job_id") == job_id
                    and existing.get("url") == url
                    and not existing.get("deleted")
                ):
                    return existing
        item = {
            "id": str(uuid4()),
            "user_id": user_id,
            "job_id": job_id,
            "url": url,
            "media_type": media_type,
            "job_type": job_type,          # t2v / i2v / ia2v / v2v / t2i / i2i
            "title": (title or prompt[:60] or "Untitled").strip(),
            "prompt": prompt or "",
            "model": model or "",
            "params": params or {},
            "source": source,          # generated | uploaded
            "tags": [],
            "favorite": False,
            "created_at": _now(),
        }
        items.append(item)
        _save(LIBRARY_FILE, items)
        return item


def update_library_item(item_id: str, user_id: str, updates: dict) -> Optional[dict]:
    allowed = {"title", "tags", "favorite"}
    clean = {k: v for k, v in updates.items() if k in allowed and v is not None}
    if not clean:
        return get_library_item(item_id, user_id)
    with _lock:
        items = _load(LIBRARY_FILE, [])
        for i, item in enumerate(items):
            if item.get("id") == item_id and item.get("user_id") == user_id:
                items[i].update(clean)
                items[i]["updated_at"] = _now()
                _save(LIBRARY_FILE, items)
                return items[i]
    return None


def delete_library_item(item_id: str, user_id: Optional[str] = None) -> Optional[dict]:
    """Soft-delete so a shared underlying file is never yanked from another row."""
    with _lock:
        items = _load(LIBRARY_FILE, [])
        for i, item in enumerate(items):
            if item.get("id") == item_id:
                if user_id and item.get("user_id") != user_id:
                    return None
                items[i]["deleted"] = True
                items[i]["deleted_at"] = _now()
                _save(LIBRARY_FILE, items)
                return items[i]
    return None


def library_url_in_use(url: str, exclude_id: str = "") -> bool:
    """True if any live library row still points at this file."""
    with _lock:
        items = _load(LIBRARY_FILE, [])
    return any(
        i.get("url") == url and not i.get("deleted") and i.get("id") != exclude_id
        for i in items
    )


def library_stats(user_id: str) -> dict:
    items = get_library(user_id, limit=100000)
    return {
        "total": len(items),
        "images": sum(1 for i in items if i.get("media_type") == "image"),
        "videos": sum(1 for i in items if i.get("media_type") == "video"),
        "favorites": sum(1 for i in items if i.get("favorite")),
    }


def library_showcase(user_id: str) -> dict:
    """
    Newest usable result per job type, for the Create page's preview cards.
    Favourites win over recency so people can pin what they want to see.
    """
    items = get_library(user_id, limit=2000)
    best: dict[str, dict] = {}
    for it in items:
        jt = it.get("job_type") or ""
        if not jt or it.get("source") != "generated":
            continue
        cur = best.get(jt)
        if cur is None:
            best[jt] = it
        elif it.get("favorite") and not cur.get("favorite"):
            best[jt] = it
    return {
        jt: {
            "url": v.get("url"),
            "media_type": v.get("media_type"),
            "title": v.get("title"),
            "prompt": (v.get("prompt") or "")[:120],
        }
        for jt, v in best.items()
    }


# ─── Credits: groups, balances, ledger ────────────────────────────────────────
# Balances live on the user record ("credits") and on groups (shared pool).
# A charge draws on the user's personal balance first, then their group's
# pool. Every movement is written to the ledger. All helpers ending in
# _locked assume the caller already holds _lock (it is not re-entrant).

REFUNDABLE_STATUSES = ("failed", "cancelled", "canceled")


class InsufficientCredits(Exception):
    def __init__(self, needed: int, available: int):
        super().__init__(f"Not enough credits: this needs {needed}, you have {available}.")
        self.needed = needed
        self.available = available


def _ledger_add_locked(entries: list, account: str, delta: int, balance_after: int,
                       kind: str, note: str = "", job_id: Optional[str] = None,
                       user_id: Optional[str] = None, actor_id: Optional[str] = None) -> None:
    entries.append({
        "id": str(uuid4()), "ts": _now(), "kind": kind, "account": account,
        "delta": int(delta), "balance_after": int(balance_after),
        "job_id": job_id, "user_id": user_id, "actor_id": actor_id, "note": note,
    })


def get_groups() -> list[dict]:
    with _lock:
        groups = _load(GROUPS_FILE, [])
        users = _load(USERS_FILE, [])
    for g in groups:
        g["members"] = sum(1 for u in users if u.get("group_id") == g["id"])
    return groups


def get_group(group_id: Optional[str]) -> Optional[dict]:
    if not group_id:
        return None
    return next((g for g in get_groups() if g["id"] == group_id), None)


def create_group(name: str, credits: int = 0, actor_id: Optional[str] = None) -> dict:
    name = (name or "").strip()
    if not name:
        raise ValueError("Group name is required")
    with _lock:
        groups = _load(GROUPS_FILE, [])
        if any(g["name"].lower() == name.lower() for g in groups):
            raise ValueError(f"A group named “{name}” already exists")
        g = {"id": str(uuid4()), "name": name, "credits": 0, "created_at": _now()}
        groups.append(g)
        if credits:
            if credits < 0:
                raise ValueError("Starting credits can't be negative")
            g["credits"] = int(credits)
            ledger = _load(LEDGER_FILE, [])
            _ledger_add_locked(ledger, f"group:{g['id']}", credits, g["credits"], "grant",
                               "Starting balance", actor_id=actor_id)
            _save(LEDGER_FILE, ledger)
        _save(GROUPS_FILE, groups)
        return {**g, "members": 0}


def rename_group(group_id: str, name: str) -> dict:
    name = (name or "").strip()
    if not name:
        raise ValueError("Group name is required")
    with _lock:
        groups = _load(GROUPS_FILE, [])
        if any(g["name"].lower() == name.lower() and g["id"] != group_id for g in groups):
            raise ValueError(f"A group named “{name}” already exists")
        for g in groups:
            if g["id"] == group_id:
                g["name"] = name
                _save(GROUPS_FILE, groups)
                return g
    raise KeyError("Group not found")


def delete_group(group_id: str) -> int:
    """Remove a group; its members become ungrouped. Returns members affected.
    The pool's remaining balance is discarded (recorded in the ledger)."""
    with _lock:
        groups = _load(GROUPS_FILE, [])
        g = next((x for x in groups if x["id"] == group_id), None)
        if not g:
            raise KeyError("Group not found")
        users = _load(USERS_FILE, [])
        n = 0
        for u in users:
            if u.get("group_id") == group_id:
                u["group_id"] = None
                n += 1
        if g.get("credits"):
            ledger = _load(LEDGER_FILE, [])
            _ledger_add_locked(ledger, f"group:{group_id}", -g["credits"], 0, "adjust",
                               f"Group “{g['name']}” deleted")
            _save(LEDGER_FILE, ledger)
        _save(USERS_FILE, users)
        _save(GROUPS_FILE, [x for x in groups if x["id"] != group_id])
        return n


def credit_balances(user_id: str) -> dict:
    """{'personal': n, 'group': {...} | None, 'available': n}"""
    with _lock:
        users = _load(USERS_FILE, [])
        groups = _load(GROUPS_FILE, [])
    u = next((x for x in users if x["id"] == user_id), None) or {}
    personal = int(u.get("credits") or 0)
    g = next((x for x in groups if x["id"] == u.get("group_id")), None)
    group = {"id": g["id"], "name": g["name"], "credits": int(g.get("credits") or 0)} if g else None
    return {"personal": personal, "group": group,
            "available": personal + (group["credits"] if group else 0)}


def adjust_credits(account_type: str, account_id: str, delta: int, note: str = "",
                   actor_id: Optional[str] = None) -> int:
    """Admin grant (+) or deduction (−). Returns the new balance."""
    delta = int(delta)
    if delta == 0:
        raise ValueError("Enter a non-zero amount")
    with _lock:
        path = USERS_FILE if account_type == "user" else GROUPS_FILE
        if account_type not in ("user", "group"):
            raise ValueError("account_type must be 'user' or 'group'")
        rows = _load(path, [])
        row = next((r for r in rows if r["id"] == account_id), None)
        if not row:
            raise KeyError(f"{account_type.title()} not found")
        new = int(row.get("credits") or 0) + delta
        if new < 0:
            raise ValueError(f"That would leave a negative balance ({new})")
        row["credits"] = new
        ledger = _load(LEDGER_FILE, [])
        _ledger_add_locked(ledger, f"{account_type}:{account_id}", delta, new,
                           "grant" if delta > 0 else "adjust", note,
                           user_id=account_id if account_type == "user" else None,
                           actor_id=actor_id)
        _save(path, rows)
        _save(LEDGER_FILE, ledger)
        return new


def reserve_credits(user_id: str, cost: int, job_id: str) -> dict:
    """Take `cost` from the user's personal balance, then their group pool.
    Raises InsufficientCredits. Returns the record stored on the job."""
    cost = int(cost)
    with _lock:
        users = _load(USERS_FILE, [])
        groups = _load(GROUPS_FILE, [])
        u = next((x for x in users if x["id"] == user_id), None)
        if not u:
            raise KeyError("User not found")
        g = next((x for x in groups if x["id"] == u.get("group_id")), None)
        personal = int(u.get("credits") or 0)
        pool = int(g.get("credits") or 0) if g else 0
        if personal + pool < cost:
            raise InsufficientCredits(cost, personal + pool)
        from_personal = min(personal, cost)
        from_group = cost - from_personal
        ledger = _load(LEDGER_FILE, [])
        if from_personal:
            u["credits"] = personal - from_personal
            _ledger_add_locked(ledger, f"user:{user_id}", -from_personal, u["credits"],
                               "charge", "Generation", job_id=job_id, user_id=user_id)
        if from_group:
            g["credits"] = pool - from_group
            _ledger_add_locked(ledger, f"group:{g['id']}", -from_group, g["credits"],
                               "charge", "Generation", job_id=job_id, user_id=user_id)
            _save(GROUPS_FILE, groups)
        _save(USERS_FILE, users)
        _save(LEDGER_FILE, ledger)
        return {"cost": cost, "from_personal": from_personal, "from_group": from_group,
                "group_id": g["id"] if (g and from_group) else None,
                "status": "charged", "charged_at": _now()}


def _refund_job_locked(job: dict, reason: str) -> None:
    """Return a job's reserved credits to where they came from. Idempotent:
    only a record in 'charged' state is refunded, and it's marked 'refunded'."""
    rec = job.get("credits")
    if not isinstance(rec, dict) or rec.get("status") != "charged":
        return
    users = _load(USERS_FILE, [])
    groups = _load(GROUPS_FILE, [])
    ledger = _load(LEDGER_FILE, [])
    uid = job.get("user_id")
    u = next((x for x in users if x["id"] == uid), None)
    back_personal = int(rec.get("from_personal") or 0)
    back_group = int(rec.get("from_group") or 0)
    g = next((x for x in groups if x["id"] == rec.get("group_id")), None)
    if back_group and not g:
        back_personal += back_group     # the group was deleted: refund to the user
        back_group = 0
    if back_personal and u:
        u["credits"] = int(u.get("credits") or 0) + back_personal
        _ledger_add_locked(ledger, f"user:{uid}", back_personal, u["credits"],
                           "refund", reason, job_id=job["id"], user_id=uid)
    if back_group and g:
        g["credits"] = int(g.get("credits") or 0) + back_group
        _ledger_add_locked(ledger, f"group:{g['id']}", back_group, g["credits"],
                           "refund", reason, job_id=job["id"], user_id=uid)
    _save(USERS_FILE, users)
    _save(GROUPS_FILE, groups)
    _save(LEDGER_FILE, ledger)
    rec["status"] = "refunded"
    rec["refunded_at"] = _now()


def refund_record(user_id: str, job_id: str, record: dict, reason: str) -> None:
    """Refund a reservation whose job was never stored (creation failed)."""
    with _lock:
        _refund_job_locked({"id": job_id, "user_id": user_id, "credits": record}, reason)


def refund_job(job_id: str, reason: str) -> None:
    """Refund outside update_job (e.g. job creation failed after the charge)."""
    with _lock:
        jobs = _load(JOBS_FILE, [])
        j = next((x for x in jobs if x["id"] == job_id), None)
        if j:
            _refund_job_locked(j, reason)
            _save(JOBS_FILE, jobs)


def get_ledger(limit: int = 100) -> list[dict]:
    with _lock:
        rows = _load(LEDGER_FILE, [])
    return list(reversed(rows[-limit:]))
