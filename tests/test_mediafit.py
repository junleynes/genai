import shutil
import subprocess

import pytest
from PIL import Image

from app import mediafit

needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                  reason="ffmpeg not installed")


def _img(tmp_path, w, h, color=(10, 120, 200)):
    p = tmp_path / f"src_{w}x{h}.png"
    Image.new("RGB", (w, h), color).save(p)
    return p


def test_fit_exact_size_is_untouched(tmp_path):
    p = _img(tmp_path, 1920, 1080)
    out, info = mediafit.fit_image(p, 1920, 1080, tmp_path / "o")
    assert out == str(p) and info["action"] == "none"


def test_fit_small_mismatch_resizes(tmp_path):
    p = _img(tmp_path, 1916, 1080)       # <3% off 16:9
    out, info = mediafit.fit_image(p, 1920, 1080, tmp_path / "o")
    assert info["action"] == "resize"
    assert Image.open(out).size == (1920, 1080)


def test_fit_moderate_mismatch_crops_and_reports_loss(tmp_path):
    p = _img(tmp_path, 1916, 1018)       # ~6.6% wider than 1920x1088
    out, info = mediafit.fit_image(p, 1920, 1088, tmp_path / "o")
    assert info["action"] == "cover" and 0 < info["crop_loss"] <= mediafit.MAX_CROP_LOSS
    assert Image.open(out).size == (1920, 1088)


def test_fit_extreme_mismatch_pads_instead_of_cropping(tmp_path):
    p = _img(tmp_path, 1000, 1000)
    out, info = mediafit.fit_image(p, 1920, 1088, tmp_path / "o")
    assert info["action"] == "contain"
    assert Image.open(out).size == (1920, 1088)


def test_fit_off_and_cache(tmp_path):
    p = _img(tmp_path, 800, 600)
    out, info = mediafit.fit_image(p, 1920, 1080, tmp_path / "o", mode="off")
    assert out == str(p) and info["action"] == "none"
    a, _ = mediafit.fit_image(p, 1920, 1080, tmp_path / "o")
    b, _ = mediafit.fit_image(p, 1920, 1080, tmp_path / "o")
    assert a == b


def _make_clip(path, w, h, seconds=2, fps=24, audio=False):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
           f"testsrc=duration={seconds}:size={w}x{h}:rate={fps}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac"]
    cmd += ["-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True)


def _probe(path):
    m = mediafit._probe(path)
    return m["w"], m["h"], m["frames"], m["dur"]


@needs_ffmpeg
def test_crop_back_to_requested_size(tmp_path):
    clip = tmp_path / "c.mp4"
    _make_clip(clip, 1920, 1088, seconds=1)
    r = mediafit.postprocess_video(clip, crop_to=(1920, 1080))
    assert r["changed"] and _probe(clip)[:2] == (1920, 1080)


@needs_ffmpeg
def test_loop_reaches_target_length_with_audio(tmp_path):
    clip = tmp_path / "c.mp4"
    _make_clip(clip, 640, 360, seconds=2, audio=True)
    r = mediafit.postprocess_video(clip, loop_to=7)
    assert r["changed"] and r["loop_repeats"] == 4
    w, h, frames, dur = _probe(clip)
    assert (w, h) == (640, 360)
    assert abs(dur - 7) < 0.15
    assert mediafit._has_audio(clip)


@needs_ffmpeg
def test_drop_last_frame_for_seamless_loop(tmp_path):
    clip = tmp_path / "c.mp4"
    _make_clip(clip, 640, 360, seconds=2)       # 48 frames
    r = mediafit.postprocess_video(clip, drop_last_frame=True)
    assert r["changed"] and r["dropped_last_frame"]
    assert _probe(clip)[2] == 47


@needs_ffmpeg
def test_loop_shorter_than_clip_is_noop_and_failure_keeps_original(tmp_path):
    clip = tmp_path / "c.mp4"
    _make_clip(clip, 640, 360, seconds=2)
    before = clip.read_bytes()
    assert mediafit.postprocess_video(clip, loop_to=1)["changed"] is False
    assert clip.read_bytes() == before
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    assert mediafit.postprocess_video(bad, crop_to=(100, 100))["changed"] is False
    assert bad.read_bytes() == b"not a video"


def test_exact_size_source_is_edge_extended_to_grid(tmp_path):
    # User supplies exactly 1920x1080; model runs on 1920x1088. The image must
    # not be stretched: extra rows replicate the border so a centred crop back
    # to 1080 restores the original pixels exactly.
    src = tmp_path / "s.png"
    im = Image.new("RGB", (1920, 1080), (200, 10, 10))
    im.putpixel((0, 0), (1, 2, 3))          # marker in the top-left corner
    im.putpixel((1919, 1079), (4, 5, 6))
    im.save(src)
    out, info = mediafit.fit_image(src, 1920, 1088, tmp_path / "o", inner_size=(1920, 1080))
    assert info["action"] == "extend"
    got = Image.open(out)
    assert got.size == (1920, 1088)
    restored = got.crop((0, 4, 1920, 1084))        # what ffmpeg's centred crop keeps
    assert restored.tobytes() == im.tobytes()
    assert got.getpixel((0, 0)) == (1, 2, 3)        # border row replicated upward
    assert got.getpixel((1919, 1087)) == (4, 5, 6)


def test_inner_size_fit_then_extend(tmp_path):
    src = _img(tmp_path, 1916, 1018)
    out, info = mediafit.fit_image(src, 1920, 1088, tmp_path / "o", inner_size=(1920, 1080))
    assert Image.open(out).size == (1920, 1088)
    assert info["extended_from"] == (1920, 1080)
