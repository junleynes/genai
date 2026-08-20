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
    "app_name": "WanForge",
    "tagline": "Simplify AI Video & Image Generation",
    "footer": "© 2026 WanForge. Powered by Wan2GP.",
    "logo_url": "",
    "favicon_url": "",
    "primary_color": "#6366f1",
    "secondary_color": "#8b5cf6",
    "accent_color": "#22d3ee",
    "default_theme": "system",
    "wan2gp_url": "http://localhost:7860",
    "wan2gp_root": "",
    "wan2gp_enabled": False,
    "wan2gp_cli_args": "--attention sdpa --profile 4",
    "default_model_type": "ltx2_22B_distilled",
    "default_resolution": "1280x704",
    "default_steps": 8,
    "default_fps": "24",
    "allow_mock_fallback": True,
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
