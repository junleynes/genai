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


def _edge_extend(img, width: int, height: int):
    """Centre img on a width×height canvas, replicating its edge rows/columns.

    Used when the model runs on a slightly larger grid than the size the user
    asked for (e.g. 1920x1088 for 1920x1080): the extra rows are copies of the
    border, so after the result is cropped back the user's image lines up
    exactly instead of being stretched to fill the grid.
    """
    from PIL import Image
    iw, ih = img.size
    left, top = (width - iw) // 2, (height - ih) // 2
    canvas = Image.new("RGB", (width, height))
    canvas.paste(img, (left, top))
    if top > 0:
        canvas.paste(img.crop((0, 0, iw, 1)).resize((iw, top), Image.NEAREST), (left, 0))
        canvas.paste(img.crop((0, ih - 1, iw, ih)).resize((iw, height - top - ih), Image.NEAREST), (left, top + ih))
    if left > 0:
        col_l = canvas.crop((left, 0, left + 1, height)).resize((left, height), Image.NEAREST)
        col_r = canvas.crop((left + iw - 1, 0, left + iw, height)).resize((width - left - iw, height), Image.NEAREST)
        canvas.paste(col_l, (0, 0))
        canvas.paste(col_r, (left + iw, 0))
    return canvas


def fit_image(src: str | Path, width: int, height: int, out_dir: Path,
              mode: str = "cover",
              inner_size: Optional[tuple[int, int]] = None) -> tuple[str, dict]:
    """Conform an image to exactly width×height.

    inner_size, when given and ≤32px smaller than width×height, is the size
    the user actually asked for: the image is fitted to that and then
    edge-extended to the full grid so a later crop restores it exactly.

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
            f"{src.resolve()}|{st.st_mtime_ns}|{st.st_size}|{width}x{height}|{mode}|{inner_size}".encode()
        ).hexdigest()[:16]
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"fit_{sig}.png"

        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            sw, sh = im.size
            info["src_size"] = (sw, sh)
            if (sw, sh) == (width, height):
                return str(src), info

            extend = False
            fw, fh = width, height
            if (inner_size and tuple(inner_size) != (width, height)
                    and width >= inner_size[0] and height >= inner_size[1]
                    and width - inner_size[0] <= 32 and height - inner_size[1] <= 32):
                fw, fh = inner_size
                extend = True
                info["extended_from"] = tuple(inner_size)
                if (sw, sh) == (fw, fh):
                    # Exact size already: only the edge-extension is needed.
                    info["action"] = "extend"
                    dest2 = out_dir / f"fit_{sig}.png"
                    if not dest2.exists() or dest2.stat().st_size == 0:
                        tmp = dest2.with_suffix(".tmp.png")
                        _edge_extend(im.convert("RGB"), width, height).save(tmp, "PNG")
                        tmp.replace(dest2)
                    return str(dest2), info

            src_ar, dst_ar = sw / sh, fw / fh
            mismatch = abs(src_ar / dst_ar - 1)
            info["aspect_mismatch"] = round(mismatch, 4)

            rgb = im.convert("RGB")
            if mismatch <= STRETCH_TOLERANCE:
                action = "resize"
                out = rgb.resize((fw, fh), Image.LANCZOS)
            else:
                # Fraction of the source lost if we cover-crop.
                loss = 1 - (min(src_ar, dst_ar) / max(src_ar, dst_ar))
                if mode == "cover" and loss <= MAX_CROP_LOSS:
                    action = "cover"
                    info["crop_loss"] = round(loss, 4)
                    out = ImageOps.fit(rgb, (fw, fh), Image.LANCZOS, centering=(0.5, 0.5))
                else:
                    action = "contain"
                    scale = min(fw / sw, fh / sh)
                    nw, nh = max(1, round(sw * scale)), max(1, round(sh * scale))
                    resized = rgb.resize((nw, nh), Image.LANCZOS)
                    # Graphics are usually flat at the edges; pad with the colour
                    # at the top-centre so the bars don't read as black.
                    canvas = Image.new("RGB", (fw, fh), rgb.getpixel((sw // 2, 0)))
                    canvas.paste(resized, ((fw - nw) // 2, (fh - nh) // 2))
                    out = canvas
            if extend:
                out = _edge_extend(out, width, height)
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
