"""
Credit pricing. One formula, used by job creation, retries, the live estimate
on the Generate page, and the Admin → Credits page:

    cost = ceil(base cost of the tool × resolution multiplier × duration multiplier)

- Base cost per tool is set by an admin (0 = free); unset tools use DEFAULT_COSTS.
- Resolution multiplier comes from the output's pixel count (RES_TIERS).
- Duration multiplier applies to video/audio output only: duration ÷ 4 s,
  never below 1 (so anything up to 4 s costs the base).

Balances, charges and refunds live in db.py (they are persistence and must
share its lock); this module only prices.
"""
from __future__ import annotations

import math
from typing import Any

from .catalog import TOOL_IDS

IMAGE_OUTPUT = ("t2i", "i2i", "cs")
DURATION_UNIT_S = 4.0

DEFAULT_COSTS: dict[str, int] = {
    # image
    "t2i": 1, "i2i": 1, "cs": 2,
    # video generation
    "t2v": 5, "i2v": 5, "ia2v": 6, "v2v": 5, "p2v": 6,
    # characters
    "fs": 6, "msr": 6, "ingredients": 6,
    # edit
    "outpaint": 4, "cleanplate": 4, "relight": 4, "daynight": 4, "colorize": 4,
    "water": 4, "shave": 4, "crosseyed": 4, "cinemagraph": 3,
    # restore
    "upscale": 4, "deblur": 3, "decompress": 3,
    # audio
    "foley": 2,
}

# (max pixels, multiplier, label) — first tier that fits wins.
RES_TIERS: list[tuple[float, float, str]] = [
    (450_000, 1.0, "Up to 480p"),
    (1_000_000, 1.5, "720p"),
    (2_200_000, 2.5, "1080p"),
    (math.inf, 4.0, "Above 1080p"),
]


def tool_costs(settings: dict) -> dict[str, int]:
    saved = settings.get("credit_tool_costs") or {}
    out = {}
    for tid in TOOL_IDS:
        v = saved.get(tid)
        out[tid] = int(v) if isinstance(v, (int, float)) and v >= 0 else DEFAULT_COSTS.get(tid, 1)
    return out


def _pixels(resolution: str | None) -> int:
    try:
        w, h = str(resolution or "832x480").lower().split("x")
        return max(1, int(w)) * max(1, int(h))
    except (ValueError, AttributeError):
        return 832 * 480


def estimate(settings: dict, job_type: str, resolution: str | None,
             duration_seconds: float | None) -> dict[str, Any]:
    base = tool_costs(settings).get(job_type, 1)
    px = _pixels(resolution)
    res_mult, tier = next((m, label) for cap, m, label in RES_TIERS if px <= cap)
    if job_type in IMAGE_OUTPUT:
        dur_mult = 1.0
    else:
        try:
            d = float(duration_seconds or DURATION_UNIT_S)
        except (TypeError, ValueError):
            d = DURATION_UNIT_S
        dur_mult = max(1.0, d / DURATION_UNIT_S)
    cost = 0 if base == 0 else max(1, math.ceil(base * res_mult * dur_mult - 1e-9))
    return {
        "cost": cost, "base": base,
        "resolution_multiplier": res_mult, "resolution_tier": tier,
        "duration_multiplier": round(dur_mult, 3),
    }


def job_cost(settings: dict, job_type: str, params: dict) -> dict[str, Any]:
    return estimate(settings, job_type, params.get("resolution"), params.get("duration_seconds"))


def validate_costs(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        raise ValueError("tool_costs must be an object of {tool_id: credits}")
    clean = {}
    for tid, v in raw.items():
        if tid not in TOOL_IDS:
            raise ValueError(f"Unknown tool: {tid}")
        try:
            n = int(v)
        except (TypeError, ValueError):
            raise ValueError(f"Cost for {tid} must be a whole number")
        if n < 0 or n > 100_000:
            raise ValueError(f"Cost for {tid} must be between 0 and 100000")
        clean[tid] = n
    return clean
