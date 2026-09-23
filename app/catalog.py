"""
Tool catalog: every generation tool the UI offers, in display order, plus
each tool's admin-controlled status.

Single source of truth for the landing gallery, the Generate picker, the
Admin → Tools page, and server-side job validation. A new tool is one entry
in TOOLS (its `id` must be a backend job type) plus card art at
static/img/cards/<id>.jpg.
"""
from __future__ import annotations

from typing import Any

# enabled      — normal
# new          — normal, with a "New" badge
# coming_soon  — card shown with a "Coming soon" badge, cannot be used
# disabled     — hidden everywhere, cannot be used
TOOL_STATUSES = ("enabled", "new", "coming_soon", "disabled")
USABLE_STATUSES = ("enabled", "new")
STATUS_LABELS = {
    "enabled": "Enabled",
    "new": "New",
    "coming_soon": "Coming soon",
    "disabled": "Disabled",
}

CATEGORIES: list[dict[str, str]] = [
    {"id": "video",   "name": "Video"},
    {"id": "people",  "name": "Characters"},
    {"id": "image",   "name": "Image"},
    {"id": "edit",    "name": "Edit"},
    {"id": "restore", "name": "Restore"},
    {"id": "audio",   "name": "Audio"},
]

TOOLS: list[dict[str, str]] = [
    {"id": "t2v",  "cat": "video", "name": "Text to Video",  "label": "Text → Video",          "desc": "Describe a shot, get a cinematic clip."},
    {"id": "i2v",  "cat": "video", "name": "Image to Video", "label": "Image → Video",         "desc": "Animate a still, with an optional end frame."},
    {"id": "ia2v", "cat": "video", "name": "Image + Audio",  "label": "Image + Audio → Video", "desc": "Drive motion and lip-sync from a soundtrack."},
    {"id": "v2v",  "cat": "video", "name": "Video to Video", "label": "Video → Video",         "desc": "Restyle or transform existing footage."},
    {"id": "p2v",  "cat": "video", "name": "Pose Driven",    "label": "Pose Driven → Video",   "desc": "Transfer a performance from a driving clip."},

    {"id": "fs",          "cat": "people", "name": "Face Swap",       "label": "Face Swap → Video",               "desc": "Put a new face onto a source video."},
    {"id": "msr",         "cat": "people", "name": "Multi-Subject",   "label": "Multi-Subject Reference → Video", "desc": "Compose a scene from 2–5 reference images."},
    {"id": "ingredients", "cat": "people", "name": "Ingredients",     "label": "Ingredients → Video",             "desc": "A reference sheet drives the whole scene."},
    {"id": "cs",          "cat": "people", "name": "Character Sheet", "label": "Character Sheet",                 "desc": "Multi-view reference sheet of one character."},

    {"id": "t2i", "cat": "image", "name": "Text to Image",  "label": "Text → Image",  "desc": "Generate stills from a prompt."},
    {"id": "i2i", "cat": "image", "name": "Image to Image", "label": "Image → Image", "desc": "Re-imagine a reference image."},

    {"id": "outpaint",    "cat": "edit", "name": "In / Outpaint",    "label": "In / Outpaint → Video",    "desc": "Fill or extend the frame of a video."},
    {"id": "cleanplate",  "cat": "edit", "name": "Clean Plate",      "label": "Clean Plate → Video",      "desc": "Remove people and objects from a shot."},
    {"id": "relight",     "cat": "edit", "name": "Relight",          "label": "Relight → Video",          "desc": "Change the lighting of existing footage."},
    {"id": "daynight",    "cat": "edit", "name": "Day to Night",     "label": "Day → Night",              "desc": "Turn daytime footage into night."},
    {"id": "colorize",    "cat": "edit", "name": "Colorize",         "label": "Colorize → Video",         "desc": "Bring black-and-white video to color."},
    {"id": "water",       "cat": "edit", "name": "Water Simulation", "label": "Water Simulation → Video", "desc": "Add rain, surf, floods and splashes."},
    {"id": "shave",       "cat": "edit", "name": "Instant Shave",    "label": "Instant Shave → Video",    "desc": "Remove facial hair, keep identity."},
    {"id": "crosseyed",   "cat": "edit", "name": "Cross-Eyed",       "label": "Cross-Eyed → Video",       "desc": "Turn eyes inward in close-up portraits."},
    {"id": "cinemagraph", "cat": "edit", "name": "Cinemagraph",      "label": "Cinemagraph",              "desc": "Loop selective motion from a still."},

    {"id": "upscale",    "cat": "restore", "name": "Upscale",    "label": "Upscale → Video",    "desc": "Generative 2× detail upscale."},
    {"id": "deblur",     "cat": "restore", "name": "Deblur",     "label": "Deblur → Video",     "desc": "Restore focus to soft footage."},
    {"id": "decompress", "cat": "restore", "name": "Decompress", "label": "Decompress → Video", "desc": "Remove blocking, banding and ringing."},

    {"id": "foley", "cat": "audio", "name": "Foley", "label": "Foley → Audio", "desc": "Synced sound effects from silent video."},
]

TOOL_IDS = tuple(t["id"] for t in TOOLS)

# Status a tool has until an admin changes it.
DEFAULT_STATUS: dict[str, str] = {
    "water": "new", "deblur": "new", "decompress": "new",
    "crosseyed": "new", "shave": "new", "cinemagraph": "new",
}


def tool_statuses(settings: dict) -> dict[str, str]:
    """Effective status for every tool: saved value if valid, else default."""
    saved = settings.get("tool_status") or {}
    out = {}
    for tid in TOOL_IDS:
        s = saved.get(tid)
        out[tid] = s if s in TOOL_STATUSES else DEFAULT_STATUS.get(tid, "enabled")
    return out


def tool_status(settings: dict, job_type: str) -> str:
    return tool_statuses(settings).get(job_type, "enabled")


def is_usable(settings: dict, job_type: str) -> bool:
    return tool_status(settings, job_type) in USABLE_STATUSES


def all_tools(settings: dict) -> list[dict[str, Any]]:
    """Every tool with its status (Admin → Tools lists them all)."""
    st = tool_statuses(settings)
    return [{**t, "status": st[t["id"]]} for t in TOOLS]


def visible_tools(settings: dict) -> list[dict[str, Any]]:
    """Tools users can see: everything except disabled."""
    return [t for t in all_tools(settings) if t["status"] != "disabled"]


def categories_for(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Categories that still have at least one tool, with counts."""
    out = []
    for c in CATEGORIES:
        n = sum(1 for t in tools if t["cat"] == c["id"])
        if n:
            out.append({**c, "count": n})
    return out


def validate_statuses(raw: Any) -> dict[str, str]:
    """Check an admin-submitted {tool_id: status} map. Raises ValueError."""
    if not isinstance(raw, dict):
        raise ValueError("tool_status must be an object of {tool_id: status}")
    clean = {}
    for tid, status in raw.items():
        if tid not in TOOL_IDS:
            raise ValueError(f"Unknown tool: {tid}")
        if status not in TOOL_STATUSES:
            raise ValueError(f"Invalid status for {tid}: {status} (use {', '.join(TOOL_STATUSES)})")
        clean[tid] = status
    return clean
