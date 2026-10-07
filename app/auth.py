"""JWT authentication helpers."""
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from . import db

logger = logging.getLogger(__name__)

MIN_SECRET_LEN = 32


def load_secret_key(data_dir: Path) -> str:
    """Signing key for JWTs. Never a value that lives in the repository.

    Order: GENAI_SECRET_KEY environment variable, then data/.jwt_secret, which
    is generated on first run and kept out of git. Because the file is created
    exclusively, several workers starting at once all end up with the same key.
    """
    env = os.environ.get("GENAI_SECRET_KEY", "").strip()
    if env:
        if len(env) >= MIN_SECRET_LEN:
            return env
        logger.warning("GENAI_SECRET_KEY is shorter than %d characters; ignoring it", MIN_SECRET_LEN)

    path = Path(data_dir) / ".jwt_secret"
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if len(existing) >= MIN_SECRET_LEN:
            return existing
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("Could not read %s: %s", path, e)

    new = secrets.token_urlsafe(48)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(new)
        logger.info("Generated a new JWT signing key at %s (existing sessions were signed out)", path)
        return new
    except FileExistsError:
        # Another worker won the race; use its key.
        return path.read_text(encoding="utf-8").strip()
    except OSError as e:
        # Read-only data dir: fall back to a per-process key. Logins still
        # work, but they won't survive a restart — say so loudly.
        logger.error("Cannot persist a JWT key (%s). Using a temporary one; set GENAI_SECRET_KEY.", e)
        return new


SECRET_KEY = load_secret_key(db.DATA_DIR)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 72

security = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    pw = password.encode("utf-8")[:72]
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    pw = plain.encode("utf-8")[:72]
    try:
        return bcrypt.checkpw(pw, hashed.encode("utf-8"))
    except Exception:
        return False


def create_access_token(user_id: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    payload = {"sub": user_id, "role": role, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = db.get_user_by_id(payload["sub"])
    if not user or not user.get("is_active", True):
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[dict]:
    if not credentials:
        return None
    payload = decode_token(credentials.credentials)
    if not payload:
        return None
    return db.get_user_by_id(payload["sub"])


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user
