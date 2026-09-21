# Opensource Generative AI

Modern web frontend for **Wan2GP** (WanGP): generate video and images through a clean UI, with user & job management, branding, Easy/Advanced modes, light/dark theme, and remote MCP integration.

**What it’s for:** point genai at a WanGP MCP server, then create Text→Video, Image→Video, pose-driven clips, face swaps, multi-subject scenes, character sheets, and more — without living in the raw WanGP UI. Results land in a personal library you can reuse as inputs.

## Features

- **User & role management** — register, login, JWT auth, `admin` / `user` roles
- **Jobs owned by creator** — every generation is attached to the user who posted it
- **10 generation types** with **Easy** and **Advanced** modes (see below)
- **Personal library** — completed outputs are filed automatically; favourites and search
- **Job queue** — concurrency limits, startup recovery of stuck jobs, cancel / retry
- **Admin Branding** — app name, tagline, footer, logo, favicon, colors
- **Admin Server & Queue** — MCP URL, outputs/inputs HTTP helpers, health panel, diagnostics
- **Day / Dark theme** — system preference + manual toggle

## Generation modes

| Type | ID | Output | Typical inputs |
|------|----|--------|----------------|
| Text → Video | `t2v` | video | prompt |
| Image → Video | `i2v` | video | start image (+ prompt) |
| Image + Audio → Video | `ia2v` | video | image + audio |
| Video → Video | `v2v` | video | source video |
| Pose-driven Video | `p2v` | video | subject + driving video / pose |
| Face Swap → Video | `fs` | video | identity face + source video |
| Multi-Subject Reference → Video | `msr` | video | 2–5 reference images |
| Text → Image | `t2i` | image | prompt |
| Image → Image | `i2i` | image | reference image + prompt |
| Character Sheet | `cs` | image | prompt (+ optional identity photo) |
| Ingredients → Video | `ingredients` | video | reference sheet image (LTX IC-LoRA) |
| In / Outpaint | `outpaint` | video | source video (LTX IC-LoRA) |
| Clean Plate | `cleanplate` | video | source video (LTX IC-LoRA) |
| Relight | `relight` | video | source video (LTX IC-LoRA) |
| Day → Night | `daynight` | video | source video (LTX IC-LoRA) |
| Colorize | `colorize` | video | B&W source video (LTX IC-LoRA) |
| Upscale | `upscale` | video | source video (LTX spatial upscaler IC-LoRA) |
| Foley | `foley` | audio | silent source video (LTX Foley V2A) |

Easy mode uses sensible defaults; Advanced exposes models, LoRAs, steps, guidance, control strength, and more.

### Model families that actually use references (FS / MSR / CS)

Sending identity or multi-subject references to a plain t2v/i2v model is a common silent no-op: the job succeeds, but the references are ignored. Prefer **Auto** model (genai picks a capable family) or install one of these on WanGP:

| Mode | What it needs | Preferred WanGP families (keywords) |
|------|----------------|-------------------------------------|
| **Face Swap (`fs`)** | Identity image + source video | `identity`, `exid`, `ecc`, `swap`, `face`, `animate`, `phantom`, `standin`, `lynx`, `msr`, `vace` |
| **Multi-Subject Reference (`msr`)** | 2–5 stills (background first, then subjects/objects) | **`msr`** (LTX-2.3 MSR finetune), then `vace`, `phantom`, `animate`, `lynx`, `bernini`, `identity`, `edit` |
| **Character Sheet (`cs`)** | Prompt; optional identity photo for consistency | `identity`, `edit`, `bernini`, then other reference-capable image models |

**Pose-driven (`p2v`)** needs a control-capable model (e.g. **VACE**, Wan-Fun control, Animate, or LTX-2 with the matching IC LoRA). Plain Wan t2v/i2v will accept a guide video and ignore it.

In **Admin → Server & Queue** you can set per-type default models/LoRAs and run **Diagnostics** to see which tools and p2v models the connected WanGP advertises.

## Quick start

```bash
pip install -r requirements.txt
python run.py
```

Open **http://localhost:8080**

**Default admin:** `admin@example.com` / `admin123`  
Change this password immediately on any shared or production host.

## Connect Wan2GP (MCP)

On the WanGP machine:

```bash
python wgp.py --mcp --mcp-transport streamable-http --mcp-host 0.0.0.0 --mcp-port 8080
```

In **Admin → Server & Queue**, set **MCP URL** to `http://<host>:8080/mcp/` (trailing slash matters) and enable WanGP generation. Use **Test connection** or **Run diagnostics** to verify.

## Branding

Change the visible app name, tagline, and colors under **Admin → Branding** (stored in settings; no code change needed).

## Serving WanGP outputs (video/image files)

WanGP’s MCP server exposes generation and job tools — not file transfer. genai fetches finished media over plain HTTP from a small file server pointed at WanGP’s `outputs/` folder:

```bash
python scripts/outputs_server.py --directory /path/to/WanGP/outputs --port 8090
```

Then set **Admin → Server & Queue → WanGP outputs HTTP base** to `http://<host>:8090`.

This is a drop-in replacement for `python -m http.server 8090`, but it also supports HTTP Range requests (206 Partial Content). Plain `http.server` does not, so browsers often cannot seek in `<video>` and may re-download the whole file. `outputs_server.py` serves each request on its own thread and answers Range requests properly.

## Sending inputs to WanGP (image/video-to-* jobs)

Getting an uploaded input file *to* WanGP needs either the `wangp_create_gallery_upload` MCP tool (not on every WanGP build) or a folder both machines can see.

If genai and WanGP run on different hosts without a shared mount, use the HTTP upload companion:

```bash
# On the WanGP machine:
python scripts/inputs_server.py --directory /path/to/WanGP/inputs --port 8091
```

Set **Admin → Server & Queue → Shared input folder — upload URL** to `http://<wangp-host>:8091`. genai PUTs input files over HTTP; `inputs_server.py` writes them into WanGP’s inputs folder and returns the path WanGP will use. Optional `--token` requires a shared secret in the `X-Auth-Token` header (match the admin “upload token” field).

The older **Shared input folder** / **WanGP-side path** fields still work as a fallback when no upload URL is set. “Not accessible or writable” usually means those paths only exist on one of the two machines — mount a real share or switch to the upload URL.

## Job queue & reliability

- **Queue** — when concurrent capacity is full, new jobs wait (or are rejected if the queue is disabled).
- **Startup recovery** — jobs left in `processing` after a process restart are re-queued and the queue is started again.
- **Timeout** — configurable MCP job timeout (`mcp_timeout_s`, default 3600s); timed-out jobs are marked failed.
- **Stale recovery** — Admin → **Recover stale jobs** re-queues processing jobs that have not updated within `stale_job_minutes`.
- **Health** — Admin health panel probes MCP, outputs HTTP, and inputs HTTP.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/outputs_server.py` | Range-aware HTTP server for WanGP `outputs/` |
| `scripts/inputs_server.py` | HTTP upload target for WanGP `inputs/` |
| `scripts/diagnose_wangp.py` | CLI: list MCP tools, LoRAs, p2v models (also available in Admin UI) |

```bash
python scripts/diagnose_wangp.py
python scripts/diagnose_wangp.py http://HOST:PORT/mcp/
```

## License / upstream

This project is a frontend for [Wan2GP](https://github.com/). Use and configure models according to their respective licenses and WanGP’s documentation.
