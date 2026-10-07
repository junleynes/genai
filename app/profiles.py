"""Per-model-family sampler profiles.

A single global CFG / frame-count / resolution rule can't be right for every
model: distilled LTX checkpoints want CFG ~1 and ignore negative prompts,
LTX needs sizes divisible by 32 and 8k+1 frames, Wan uses 4k+1 frames. Sending
the wrong values doesn't fail — it quietly degrades output (over-driven
guidance, off-grid sizes that get cropped or stretched by the backend), which
shows up as drift and invented content.

Keep this table the single place that encodes those facts. Values are the
generally documented ones for each family; if WanGP on your host disagrees,
adjust here rather than at call sites.
"""
from __future__ import annotations

from typing import Optional

# cfg            — default guidance when the user left it on "Auto" (None = no
#                  opinion; fall back to the admin default).
# uses_negative  — whether the negative prompt has any effect at that CFG.
# res_multiple   — generation width/height must be a multiple of this.
# frame_step     — video length must be frame_step*k + 1 frames.
PROFILES: dict[str, dict] = {
    # LTX-2.x distilled checkpoints: few-step, guidance baked in.
    "ltx2_distilled": {"cfg": 1.0, "uses_negative": False, "res_multiple": 32, "frame_step": 8},
    # Non-distilled LTX-2.x: real CFG, negatives work.
    "ltx2":           {"cfg": 3.0, "uses_negative": True,  "res_multiple": 32, "frame_step": 8},
    # Wan family: leave guidance to the caller, 4k+1 frames, WanGP rounds sizes itself.
    "wan":            {"cfg": None, "uses_negative": True, "res_multiple": None, "frame_step": 4},
    "default":        {"cfg": None, "uses_negative": True, "res_multiple": None, "frame_step": 4},
}


def family_of(model_type: Optional[str]) -> str:
    m = (model_type or "").lower()
    if not m:
        return "default"
    if m.startswith("ltx") or "ltx2" in m or "ltxv" in m:
        return "ltx2_distilled" if "distilled" in m else "ltx2"
    if m.startswith("wan") or "vace" in m:
        return "wan"
    return "default"


def profile_for(model_type: Optional[str]) -> dict:
    return PROFILES[family_of(model_type)]


def resolve_cfg(model_type: Optional[str], requested, admin_default=None) -> float:
    """Explicit user value wins (including 0); otherwise profile, admin default, then 5.0."""
    if requested not in (None, ""):
        try:
            return float(requested)
        except (TypeError, ValueError):
            pass
    prof_cfg = profile_for(model_type)["cfg"]
    if prof_cfg is not None:
        return float(prof_cfg)
    if admin_default not in (None, ""):
        try:
            return float(admin_default)
        except (TypeError, ValueError):
            pass
    return 5.0


def negative_is_effective(model_type: Optional[str], cfg: float) -> bool:
    """Classifier-free guidance only uses the negative branch when CFG > 1."""
    return profile_for(model_type)["uses_negative"] and float(cfg) > 1.0


def _snap_up(v: int, mult: int) -> int:
    return ((int(v) + mult - 1) // mult) * mult


def parse_resolution(res: str) -> Optional[tuple[int, int]]:
    try:
        w, h = str(res).lower().replace("×", "x").split("x")
        w, h = int(w), int(h)
        return (w, h) if w > 0 and h > 0 else None
    except Exception:
        return None


def snap_resolution(res: str, model_type: Optional[str]) -> tuple[str, Optional[tuple[int, int]]]:
    """Return (generation_resolution, requested_wh_if_changed).

    Rounds *up* to the model's grid so nothing is cropped away before the
    caller trims back to the size the user asked for. requested_wh is None
    when no change was needed.
    """
    mult = profile_for(model_type)["res_multiple"]
    wh = parse_resolution(res)
    if not mult or not wh:
        return res, None
    gw, gh = _snap_up(wh[0], mult), _snap_up(wh[1], mult)
    if (gw, gh) == wh:
        return res, None
    return f"{gw}x{gh}", wh


def snap_frames(frames: int, model_type: Optional[str]) -> int:
    """Nearest valid frame count (step*k + 1), never below one step + 1."""
    step = profile_for(model_type)["frame_step"]
    k = max(1, round((int(frames) - 1) / step))
    return k * step + 1
