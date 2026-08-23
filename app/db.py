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
    "app_name": "Opensource Generative AI",
    "tagline": "Simplify AI Video & Image Generation",
    "footer": "© 2026 Opensource Generative AI. Powered by Wan2GP.",
    "logo_url": "",
    "favicon_url": "",
    "primary_color": "#6366f1",
    "secondary_color": "#8b5cf6",
    "accent_color": "#22d3ee",
    "default_theme": "system",
    "wan2gp_url": "http://localhost:7860",
    "wan2gp_root": "",
    "wan2gp_mcp_url": "",
    "wan2gp_outputs_http_base": "",  # e.g. http://HOST:8090 serving WanGP outputs/
    # Input staging for WanGP builds with no wangp_create_gallery_upload tool
    "wan2gp_input_dir": "",            # local/mounted dir both servers can see
    "wan2gp_input_remote_prefix": "",
    # Override VACE guide letters if your WanGP build uses a different alphabet,
    # e.g. "pose=P,depth=D,canny=E". Empty uses the built-in defaults.
    "wan2gp_control_letters": "",  # same dir as WanGP sees it, e.g. C:\AI-Tools\Wan2GP\inputs
    "wan2gp_enabled": False,
    "wan2gp_cli_args": "--attention sdpa --profile 4",
    "default_model_type": "ltx2_22B_distilled",
    "default_resolution": "1280x704",
    "default_steps": 8,
    "default_fps": "24",
    "allow_mock_fallback": False,
    # Job queue
    "queue_enabled": True,
    "max_concurrent_jobs": 1,
    "concurrent_scope": "overall",  # overall | per_user
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
) -> dict:
    with _lock:
        jobs = _load(JOBS_FILE, [])
        job = {
            "id": str(uuid4()),
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
