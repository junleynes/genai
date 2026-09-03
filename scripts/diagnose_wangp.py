#!/usr/bin/env python3
"""
Report what a WanGP MCP server actually exposes.

Run this on the genai server when pose-driven generation or the LoRA
picker isn't behaving, and it will say which of the two is at fault:
the WanGP install (missing VACE model / no LoRA tool) or genai's
detection of it.

Usage:
    python scripts/diagnose_wangp.py                       # reads the URL from settings
    python scripts/diagnose_wangp.py http://HOST:PORT/mcp/ # or pass it explicitly

Prints, in order:
  1. every tool the MCP server advertises
  2. whether a LoRA-listing tool exists, and what it returns
  3. the p2v model list and which entries are guide-capable (VACE-style)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db  # noqa: E402
from app import generation as gen  # noqa: E402


def main():
    if len(sys.argv) > 1:
        mcp_url = sys.argv[1]
    else:
        mcp_url = (db.get_settings().get("wan2gp_mcp_url") or "").strip()
    if not mcp_url:
        print("No MCP URL. Set it in Admin -> Server & Queue, or pass it as an argument.")
        return 1
    print(f"MCP URL: {mcp_url}\n")

    # ── 1. Tools ──────────────────────────────────────────────────────────
    print("=" * 68)
    print("1. TOOLS ADVERTISED BY THE SERVER")
    print("=" * 68)
    try:
        tools = gen.mcp_discover_tools(mcp_url, force=True)
    except Exception as e:
        print(f"  FAILED to list tools: {e}")
        print("  -> genai cannot reach the MCP server at all. Check the URL")
        print("     (it must end in /mcp/ with the trailing slash) and that")
        print("     WanGP was started with its MCP server enabled.")
        return 1

    if not tools:
        print("  (none returned)\n")
        print("  -> Either genai cannot reach the MCP server, or the server")
        print("     advertises no tools. Everything below would be misleading,")
        print("     so stopping here. Check that:")
        print("       - the URL ends in /mcp/ (a trailing slash; /mcp gives a 307)")
        print("       - WanGP is running with its MCP server enabled")
        print("       - the port is reachable from this machine (try curl)")
        return 1
    for t in tools:
        params = ", ".join(gen._tool_param_names(t)) or "-"
        print(f"  {t.get('name','?'):40} params: {params}")
    print(f"\n  {len(tools)} tool(s) total")

    # ── 2. LoRA listing ───────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("2. LORA LISTING")
    print("=" * 68)
    lora_tools = gen._find_tools(
        mcp_url,
        include=("lora", "loras"),
        exclude=("download", "delete", "remove", "upload", "apply", "activate"),
    )
    if not lora_tools:
        print("  No LoRA-listing tool found.")
        print("  -> This is why the picker shows nothing. genai falls back to")
        print("     the free-text box; type filenames exactly as they appear in")
        print("     WanGP's own Loras folder for the selected model.")
        print("     Any tool with 'lora' in the name would have been used, so")
        print("     this WanGP build simply doesn't expose one over MCP.")
    else:
        print(f"  Candidate tool(s): {[t.get('name') for t in lora_tools]}")
        loras, supported = gen.list_loras_for_model(mcp_url, "")
        print(f"  supported={supported}  count={len(loras)}")
        for l in loras[:20]:
            print(f"    - {l.get('filename')}   (model_type={l.get('model_type') or '-'})")
        if not loras:
            print("  Tool exists but returned nothing for an unfiltered query.")
            print("  -> Likely no LoRAs installed, or it needs a model_type.")

    # ── 3. Models / guide capability ──────────────────────────────────────
    print("\n" + "=" * 68)
    print("3. POSE-DRIVEN (p2v) MODEL AVAILABILITY")
    print("=" * 68)
    try:
        models = gen.list_models_for_job_type(mcp_url, "p2v", limit=200)
    except Exception as e:
        print(f"  FAILED to list models: {e}")
        return 1

    capable = [
        m for m in models
        if gen._is_control_capable(
            m.get("model_type", ""), m.get("name", ""), m.get("family", "")
        )
    ]
    print(f"  p2v dropdown would show: {len(models)} model(s)")
    print(f"  guide-capable (VACE-style): {len(capable)}\n")

    if capable:
        for m in capable[:20]:
            note = "  (needs pose/depth IC LoRA)" if gen._needs_control_lora(
                m.get("model_type", ""), m.get("name", ""), m.get("family", "")
            ) else ""
            print(f"    OK  {m.get('model_type')}  ({m.get('name')}){note}")
        print("\n  -> Pose control should work. Pick one of these, or leave")
        print("     Model on Auto and genai will choose one.")
        if any(gen._needs_control_lora(m.get("model_type",""), m.get("name",""),
                                       m.get("family","")) for m in capable):
            print("\n  NOTE: LTX-2 gets pose/depth/canny control from an IC LoRA,")
            print("  not from the checkpoint. Put the control IC LoRA in WanGP's")
            print("  loras/ltx2 folder and activate it (Generate page, or set")
            print("  default_loras_p2v in Admin -> Server & Queue). Without it")
            print("  the job runs but ignores the driving video.")
    else:
        print("  NONE. This is why pose-driven generation does nothing:")
        print("  the guide is sent, and the model ignores it.")
        print("\n  Install either a VACE model (e.g. 'Vace 14B') or an LTX-2")
        print("  model plus its pose/depth/canny IC LoRA, then re-run.\n")
        print("  First 20 models currently offered for p2v:")
        for m in models[:20]:
            print(f"    --  {m.get('model_type')}  ({m.get('name')})")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
