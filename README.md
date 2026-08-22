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
