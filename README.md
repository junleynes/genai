# Opensource Generative AI

Modern web frontend for **Wan2GP** (WanGP) – user & job management, branding, Easy/Advanced generation modes, light/dark theme, remote MCP integration.

## Features

- **User & role management** – register, login, JWT auth, `admin` / `user` roles
- **Jobs owned by creator** – every generation is attached to the user who posted it
- **Easy & Advanced modes** – Text→Video, Image→Video, Image+Audio→Video, Video→Video, Text→Image, Image→Image
- **Admin Branding** – app name, tagline, footer, logo, favicon, colors (change the display name anytime)
- **Admin Server config** – connect to remote WanGP via MCP Streamable HTTP (`http://host:8080/mcp/`)
- **Day / Dark theme** – system preference + manual toggle

## Quick start

```bash
pip install -r requirements.txt
python run.py
```

Open **http://localhost:8080**

**Default admin:** `admin@example.com` / `admin123`

## Connect Wan2GP (MCP)

```bash
python wgp.py --mcp --mcp-transport streamable-http --mcp-host 0.0.0.0 --mcp-port 8080
```

In **Admin → Wan2GP Server**, set MCP URL to `http://<host>:8080/mcp/` and enable.

## Branding

Change the visible app name under **Admin → Branding** (stored in settings; no code change needed).

## Serving WanGP outputs (video/image files)

WanGP's MCP server only exposes generation/job-management tools — it has
no file-transfer capability. So genai fetches finished media over plain
HTTP, from a small file server pointed at WanGP's `outputs/` folder:

```bash
python scripts/outputs_server.py --directory /path/to/WanGP/outputs --port 8090
```

Then set **Admin → Server & Queue → WanGP outputs HTTP base** to
`http://<host>:8090`.

This is a drop-in replacement for `python -m http.server 8090`, but it
also supports HTTP Range requests (206 Partial Content). Plain
`http.server` doesn't, which means browsers can't seek/scrub `<video>`
playback and some players re-download the whole file just to jump
forward a few seconds — with several jobs/library cards open at once
this can make playback look stuck behind a single stream. `outputs_server.py`
serves each request on its own thread and answers Range requests properly,
so seeking only transfers the requested bytes.

## Sending inputs to WanGP (image/video-to-* jobs)

The reverse direction — getting an uploaded input file *to* WanGP — needs
either the `wangp_create_gallery_upload` MCP tool (not present on every
WanGP build) or a folder both machines can see. If genai and WanGP run on
different hosts with no shared/mounted folder, use the HTTP upload
companion instead of setting up SMB/NFS:

```bash
# On the WanGP machine:
python scripts/inputs_server.py --directory /path/to/WanGP/inputs --port 8091
```

Then set **Admin → Server & Queue → Shared input folder — upload URL** to
`http://<wangp-host>:8091`. genai will PUT input files to it over HTTP;
`inputs_server.py` writes them into WanGP's local inputs folder and hands
back the path WanGP itself will use — no mount required. An optional
`--token` flag can be passed to require a shared secret in the
`X-Auth-Token` header (matched by the "upload token" admin field), useful
if that port is reachable beyond a trusted LAN.

The older **Shared input folder** / **Shared input folder — WanGP-side
path** fields are still supported as a fallback for setups that do have a
real mounted share, but only take effect when no upload URL is set. If you
see a "not accessible or writable" error, it usually means those fields
are pointing at a path that only exists on one of the two machines —
either mount a real share there, or switch to the upload URL above.
