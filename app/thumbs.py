"""Small JPEG thumbnails for library tiles.

The library grid used to load every full-size image and open a <video> per
tile. A ~480px JPEG per item makes the grid cheap; the full file is only
fetched for hover-preview and the detail drawer.

Thumbnails are cached on disk as static/thumbs/<item_id>.jpg and generated
on first request (or in the background when a result is filed), so existing
library items need no backfill.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent
STATIC = (BASE / "static").resolve()
THUMB_DIR = STATIC / "thumbs"
THUMB_WIDTH = 480
FFMPEG = shutil.which("ffmpeg")

# Bound concurrent ffmpeg/Pillow work so a first view of a big library
# can't spawn dozens of processes at once.
_slots = threading.BoundedSemaphore(3)


def source_path(url: str) -> Optional[Path]:
    """Map a /static/... URL to a file on disk, refusing anything outside static/."""
    if not url or not url.startswith("/static/"):
        return None
    p = (BASE / url.lstrip("/")).resolve()
    try:
        p.relative_to(STATIC)
    except ValueError:
        return None
    return p if p.is_file() else None


def thumb_path(item_id: str) -> Path:
    safe = "".join(c for c in item_id if c.isalnum() or c in "-_")
    return THUMB_DIR / f"{safe}.jpg"


def _from_video(src: Path, dest: Path) -> bool:
    if not FFMPEG:
        return False
    # Try 0.5s in (skips black first frames), then fall back to frame 0 for
    # clips shorter than that.
    for seek in ("0.5", "0"):
        cmd = [FFMPEG, "-v", "error", "-y", "-ss", seek, "-i", str(src),
               "-frames:v", "1", "-vf", f"scale={THUMB_WIDTH}:-2",
               "-q:v", "4", str(dest)]
        try:
            subprocess.run(cmd, check=True, timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            continue
        if dest.exists() and dest.stat().st_size > 0:
            return True
    return False


def _from_image(src: Path, dest: Path) -> bool:
    try:
        from PIL import Image, ImageOps
    except ImportError:
        logger.warning("Pillow not installed; image thumbnails disabled")
        return False
    try:
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            im.thumbnail((THUMB_WIDTH, THUMB_WIDTH * 4))
            if im.mode not in ("RGB", "L"):
                # Flatten transparency onto white rather than black.
                bg = Image.new("RGB", im.size, (255, 255, 255))
                rgba = im.convert("RGBA")
                bg.paste(rgba, mask=rgba.split()[-1])
                im = bg
            im.convert("RGB").save(dest, "JPEG", quality=78, optimize=True)
        return True
    except Exception as e:
        logger.warning("Image thumbnail failed for %s: %s", src, e)
        return False


def ensure_thumb(item_id: str, url: str, media_type: str) -> Optional[Path]:
    """Return the cached thumbnail path, generating it if needed. None if impossible."""
    if media_type not in ("image", "video"):
        return None
    dest = thumb_path(item_id)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    src = source_path(url)
    if not src:
        return None
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.jpg")
    with _slots:
        ok = _from_video(src, tmp) if media_type == "video" else _from_image(src, tmp)
    if not ok:
        tmp.unlink(missing_ok=True)
        return None
    tmp.replace(dest)  # atomic: readers never see a half-written file
    return dest


def pregenerate(item: dict) -> None:
    """Fire-and-forget thumbnail creation for a freshly filed library item."""
    def _run():
        try:
            ensure_thumb(item["id"], item.get("url", ""), item.get("media_type", ""))
        except Exception as e:
            logger.warning("thumb pregenerate failed: %s", e)
    threading.Thread(target=_run, daemon=True).start()


def delete_thumb(item_id: str) -> None:
    thumb_path(item_id).unlink(missing_ok=True)
