"""Media fitting and post-processing.

Two jobs live here because both exist to keep generated video faithful to what
the user supplied and asked for:

* fit_image   — conform a start/end frame to the exact generation size *before*
                it is staged, so the backend never has to guess (resize, crop
                or stretch) and the first frame matches the user's design.
* postprocess_video — trim/scale the result back to the requested size (the
                model may have run on a padded grid) and optionally loop it to
                a target length with an ffmpeg re-encode so seams stay clean.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

# ≤ this aspect-ratio mismatch is stretched (invisible); beyond it we crop.
STRETCH_TOLERANCE = 0.03
# Crop is preferred up to this much lost width/height; past it we pad instead.
MAX_CROP_LOSS = 0.08


def fit_image(src: str | Path, width: int, height: int, out_dir: Path,
              mode: str = "cover") -> tuple[str, dict]:
    """Conform an image to exactly width×height.

    Returns (path, info). info["action"] is one of: "none" (already exact or
    fitting is off), "resize", "cover", "contain". The original is never
    modified; fitted copies are PNG (lossless — text stays sharp) and cached
    by content signature.
    """
    src = Path(src)
    info: dict = {"action": "none"}
    if mode == "off":
        return str(src), info
    try:
        from PIL import Image, ImageOps
    except ImportError:
        logger.warning("Pillow not installed; source image fitting disabled")
        return str(src), info

    try:
        st = src.stat()
        sig = hashlib.sha1(
            f"{src.resolve()}|{st.st_mtime_ns}|{st.st_size}|{width}x{height}|{mode}".encode()
        ).hexdigest()[:16]
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"fit_{sig}.png"

        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            sw, sh = im.size
            info["src_size"] = (sw, sh)
            if (sw, sh) == (width, height):
                return str(src), info

            src_ar, dst_ar = sw / sh, width / height
            mismatch = abs(src_ar / dst_ar - 1)
            info["aspect_mismatch"] = round(mismatch, 4)

            rgb = im.convert("RGB")
            if mismatch <= STRETCH_TOLERANCE:
                action = "resize"
                out = rgb.resize((width, height), Image.LANCZOS)
            else:
                # Fraction of the source lost if we cover-crop.
                loss = 1 - (min(src_ar, dst_ar) / max(src_ar, dst_ar))
                if mode == "cover" and loss <= MAX_CROP_LOSS:
                    action = "cover"
                    info["crop_loss"] = round(loss, 4)
                    out = ImageOps.fit(rgb, (width, height), Image.LANCZOS, centering=(0.5, 0.5))
                else:
                    action = "contain"
                    scale = min(width / sw, height / sh)
                    nw, nh = max(1, round(sw * scale)), max(1, round(sh * scale))
                    resized = rgb.resize((nw, nh), Image.LANCZOS)
                    # Graphics are usually flat at the edges; pad with the colour
                    # at the top-centre so the bars don't read as black.
                    canvas = Image.new("RGB", (width, height), rgb.getpixel((sw // 2, 0)))
                    canvas.paste(resized, ((width - nw) // 2, (height - nh) // 2))
                    out = canvas
            info["action"] = action
            if not dest.exists() or dest.stat().st_size == 0:
                tmp = dest.with_suffix(".tmp.png")
                out.save(tmp, "PNG", optimize=False)
                tmp.replace(dest)
        return str(dest), info
    except Exception as e:
        logger.warning("fit_image failed for %s: %s", src, e)
        return str(src), {"action": "none", "error": str(e)}


def _probe(path: Path) -> Optional[dict]:
    if not FFPROBE:
        return None
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
             "-of", "json", str(path)],
            check=True, capture_output=True, timeout=30, text=True,
        ).stdout
        st = (json.loads(out).get("streams") or [None])[0]
        if not st:
            return None
        num, _, den = str(st.get("r_frame_rate", "0/1")).partition("/")
        fps = float(num) / float(den or 1) if float(den or 1) else 0.0
        dur = float(st["duration"]) if st.get("duration") not in (None, "N/A") else 0.0
        nb = int(st["nb_frames"]) if str(st.get("nb_frames", "")).isdigit() else round(dur * fps)
        return {"w": int(st["width"]), "h": int(st["height"]), "fps": fps, "frames": nb, "dur": dur or (nb / fps if fps else 0)}
    except Exception as e:
        logger.warning("ffprobe failed for %s: %s", path, e)
        return None


def _has_audio(path: Path) -> bool:
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=codec_type", "-of", "csv=p=0", str(path)],
            capture_output=True, timeout=30, text=True,
        ).stdout
        return "audio" in out
    except Exception:
        return False


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, timeout=900,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def postprocess_video(path: str | Path, crop_to: Optional[tuple[int, int]] = None,
                      loop_to: float = 0.0, drop_last_frame: bool = False) -> dict:
    """Trim/scale to crop_to, optionally loop to loop_to seconds. In place.

    drop_last_frame is for seamless loops where the last frame equals the
    first: without it the repeated frame holds for one extra tick at every
    seam. Returns {"changed": bool, ...details}; never raises — on any failure
    the original file is left untouched.
    """
    path = Path(path)
    res: dict = {"changed": False}
    if not FFMPEG or not FFPROBE or not path.is_file():
        return res
    meta = _probe(path)
    if not meta or meta["fps"] <= 0:
        return res

    vf: Optional[str] = None
    if crop_to and (meta["w"], meta["h"]) != tuple(crop_to):
        tw, th = crop_to
        if meta["w"] >= tw and meta["h"] >= th and meta["w"] - tw <= 32 and meta["h"] - th <= 32:
            vf = f"crop={tw}:{th}"          # lossless trim of the padding
        else:
            vf = f"scale={tw}:{th}:flags=lanczos"
        res["resized_to"] = [tw, th]

    frames = meta["frames"]
    trim_t = None
    if drop_last_frame and frames > 2:
        trim_t = (frames - 1) / meta["fps"]
        frames -= 1
        res["dropped_last_frame"] = True

    clip_dur = frames / meta["fps"]
    do_loop = loop_to and loop_to > clip_dur + 0.01
    if not (vf or trim_t or do_loop):
        return res

    audio = _has_audio(path)
    enc_v = ["-c:v", "libx264", "-preset", "medium", "-crf", "14", "-pix_fmt", "yuv420p"]
    enc_a = ["-c:a", "aac", "-b:a", "192k"] if audio else ["-an"]
    stamp = f".pp{os.getpid()}"
    step1 = path.with_name(path.stem + stamp + "a.mp4")
    step2 = path.with_name(path.stem + stamp + "b.mp4")
    try:
        cur = path
        if vf or trim_t:
            cmd = [FFMPEG, "-v", "error", "-y", "-i", str(cur)]
            if vf:
                cmd += ["-vf", vf]
            if trim_t:
                cmd += ["-t", f"{trim_t:.6f}"]
            cmd += ["-map", "0:v:0"] + (["-map", "0:a:0"] if audio else []) + enc_v + enc_a
            cmd += ["-movflags", "+faststart", str(step1)]
            _run(cmd)
            cur = step1
        if do_loop:
            repeats = math.ceil(loop_to / clip_dur) - 1
            cmd = [FFMPEG, "-v", "error", "-y", "-stream_loop", str(repeats), "-i", str(cur),
                   "-t", f"{loop_to:.6f}", "-map", "0:v:0"] + (["-map", "0:a:0"] if audio else [])
            cmd += enc_v + enc_a + ["-movflags", "+faststart", str(step2)]
            _run(cmd)
            cur = step2
            res["looped_to"] = loop_to
            res["loop_repeats"] = repeats + 1
        os.replace(cur, path)
        res["changed"] = True
    except Exception as e:
        err = getattr(e, "stderr", b"")
        logger.warning("postprocess_video failed for %s: %s %s", path, e,
                       (err.decode("utf-8", "ignore")[-300:] if err else ""))
        res["error"] = str(e)
    finally:
        for t in (step1, step2):
            try:
                if t.exists() and t != path:
                    t.unlink()
            except Exception:
                pass
    return res
