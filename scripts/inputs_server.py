#!/usr/bin/env python3
"""
Small HTTP upload endpoint for WanGP input files.

Companion to outputs_server.py, but for the other direction: getting
image/video/audio input files FROM genai TO WanGP when the two run on
different machines with no shared/mounted folder between them, and the
WanGP build has no `wangp_create_gallery_upload` MCP tool.

Without either of those, genai has no way to hand WanGP an input file —
WanGP only understands local filesystem paths, not URLs.

This server runs ON THE WANGP HOST. genai PUTs the input file to it over
HTTP; this server writes it straight into WanGP's local inputs folder and
reports back the absolute path *as WanGP sees it*. genai then hands that
path to WanGP's generate tool, exactly as if a shared folder had been
mounted — but without needing SMB/NFS set up between the two machines.

Usage (on the WanGP machine):
    python scripts/inputs_server.py --directory C:\\AI-Tools\\Wan2GP\\inputs --port 8091

Optionally require a shared secret (recommended if this port is reachable
beyond a trusted LAN, since this endpoint writes files):
    python scripts/inputs_server.py --directory ... --port 8091 --token SOME_SECRET

Then in genai's Admin -> Server & Queue, set:
    "Shared input folder — upload URL" to http://<wangp-host>:8091
    "Shared input folder — upload token" to the same --token value (if used)
Leave the older "Shared input folder" / "WanGP-side path" fields blank —
those are for a mounted-share setup and are only used as a fallback when
no upload URL is configured.
"""
import argparse
import http.server
import json
import os
import socketserver
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


class UploadHandler(http.server.BaseHTTPRequestHandler):
    directory = "."
    token = ""

    def _json(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _check_auth(self) -> bool:
        if not self.token:
            return True
        return self.headers.get("X-Auth-Token", "") == self.token

    def do_GET(self):
        if urlsplit(self.path).path == "/health":
            self._json(200, {"ok": True, "directory": str(Path(self.directory).resolve())})
            return
        self._json(404, {"error": "not found"})

    def _handle_upload(self):
        if not self._check_auth():
            self._json(401, {"error": "unauthorized"})
            return

        # Basename only — never let the client write outside the inputs dir.
        filename = os.path.basename(unquote(urlsplit(self.path).path.lstrip("/")))
        if not filename:
            self._json(400, {"error": "missing filename in path, e.g. PUT /photo.jpg"})
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self._json(400, {"error": "missing/invalid Content-Length"})
            return

        dest_dir = Path(self.directory)
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self._json(500, {"error": f"cannot create {dest_dir}: {e}"})
            return

        dest = dest_dir / filename
        bufsize = 64 * 1024
        remaining = length
        try:
            with open(dest, "wb") as f:
                while remaining > 0:
                    chunk = self.rfile.read(min(bufsize, remaining))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
        except Exception as e:
            self._json(500, {"error": f"write failed: {e}"})
            return

        self._json(200, {
            "path": str(dest.resolve()),
            "filename": filename,
            "size": dest.stat().st_size,
        })

    def do_PUT(self):
        self._handle_upload()

    def do_POST(self):
        self._handle_upload()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--directory", "-d", required=True, help="WanGP inputs folder (local to this machine)")
    parser.add_argument("--port", "-p", type=int, default=8091)
    parser.add_argument("--bind", "-b", default="0.0.0.0")
    parser.add_argument("--token", default="", help="Optional shared secret required in X-Auth-Token header")
    args = parser.parse_args()

    UploadHandler.directory = args.directory
    UploadHandler.token = args.token
    Path(args.directory).mkdir(parents=True, exist_ok=True)

    with ThreadingHTTPServer((args.bind, args.port), UploadHandler) as httpd:
        print(f"Accepting input uploads into {Path(args.directory).resolve()} "
              f"on http://{args.bind}:{args.port} (PUT or POST to /<filename>)")
        if args.token:
            print("Auth: X-Auth-Token header required")
        else:
            print("Auth: none — restrict this to a trusted LAN")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    main()
