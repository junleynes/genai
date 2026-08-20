# WanForge

Modern web frontend for **Wan2GP** (WanGP) – user & job management, branding, Easy/Advanced generation modes, light/dark theme.

## Features

- **User & role management** – register, login, JWT auth, `admin` / `user` roles
- **Jobs owned by creator** – every generation is attached to the user who posted it
- **Easy & Advanced modes** for Text→Video, Image→Video, Text→Image, Image→Image
- **Admin Branding** – app name, tagline, footer, logo, favicon, primary/secondary/accent colors
- **Admin Server config** – connect to a Wan2GP Gradio server (`http://host:7860`), test connection
- **Day / Dark theme** – system preference + manual toggle
- Mock generation when Wan2GP is offline; real Gradio client attempts when enabled

## Quick start

```bash
cd wan2gp-manager
pip install fastapi uvicorn python-multipart "python-jose[cryptography]" "passlib[bcrypt]" aiofiles httpx gradio_client jinja2 email-validator
python run.py
```

Open **http://localhost:8080**

**Default admin:** `admin@example.com` / `admin123`

## Connect Wan2GP

1. Run Wan2GP somewhere:
   ```bash
   git clone https://github.com/deepbeepmeep/Wan2GP
   cd Wan2GP && pip install -r requirements.txt
   python wgp.py --listen --server-port 7860
   ```
2. In WanForge → Admin → Wan2GP Server, set URL to `http://<host>:7860` and enable.
3. Use **Test connection**. Jobs will try the Gradio API; on failure they fall back to mock results so the UI stays usable.

## Project layout

```
wan2gp-manager/
├── app/
│   ├── main.py          # FastAPI routes + pages
│   ├── auth.py          # JWT + password hashing
│   ├── db.py            # JSON file store (users, jobs, settings)
│   └── generation.py    # Job processor + Wan2GP client
├── templates/           # Jinja2 HTML (Tailwind CDN)
├── static/js/app.js     # Client auth, theme, helpers
├── static/uploads/      # Logo / favicon
├── data/                # users.json, jobs.json, settings.json
└── run.py
```

## Notes

- Data is stored in `data/*.json` (no external DB required).
- Change `SECRET_KEY` in `app/auth.py` for production.
- Gradio endpoint names differ by Wan2GP version; adjust `_try_wan2gp` in `generation.py` to match your instance’s API (`client.view_api()`).
