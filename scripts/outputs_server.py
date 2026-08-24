#!/usr/bin/env python3
"""
Threaded, Range-request-aware static file server.

Drop-in replacement for `python -m http.server` when serving WanGP's
outputs/ folder to genai (the `wan2gp_outputs_http_base` setting under
Admin -> Server & Queue).

Why not just `python -m http.server 8090`?
  - It DOES handle multiple connections concurrently (ThreadingHTTPServer
    has been the default under the -m CLI since Python 3.7), so it's not
    literally single-stream.
  - BUT it does NOT support HTTP Range requests / 206 Partial Content.
    Without that, browsers can't seek/scrub a <video>, and some players
    re-request (and the server re-sends) the entire file just to jump
    forward a few seconds. With several jobs/library cards open at once
    this saturates the connection and makes playback look like it's
    stuck behind a single stream.

This script adds:
  - Explicit ThreadingHTTPServer (each request handled in its own thread)
  - Range / 206 Partial Content support, so seeking only transfers the
    requested byte range
  - HTTP/1.1 keep-alive so a browser doesn't pay a new TCP handshake for
    every thumbnail/segment request
  - Permissive CORS header (Access-Control-Allow-Origin: *), harmless on
    a LAN and avoids surprises if genai and the outputs server ever end
    up on different hosts/ports

Usage:
    python scripts/outputs_server.py --directory /path/to/WanGP/outputs --port 8090

Then point genai's Admin -> Server & Queue "WanGP outputs HTTP base" at:
    http://<host>:8090
"""
import argparse
import functools
import http.server
import os
import re
import socketserver

RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


class RangeHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    # HTTP/1.1 + Content-Length (already sent by SimpleHTTPRequestHandler)
    # gets us keep-alive for free from BaseHTTPRequestHandler.
    protocol_version = "HTTP/1.1"

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path) or "Range" not in self.headers:
            return super().send_head()

        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        fs = os.fstat(f.fileno())
        file_len = fs.st_size

        match = RANGE_RE.search(self.headers["Range"])
        if not match:
            f.close()
            self.send_error(416, "Requested Range Not Satisfiable")
            return None

        start_s, end_s = match.groups()
        if start_s == "":
            # Suffix range, e.g. "bytes=-500" == last 500 bytes.
            length = int(end_s)
            start = max(file_len - length, 0)
            end = file_len - 1
        else:
            start = int(start_s)
            end = int(end_s) if end_s else file_len - 1

        if file_len == 0 or start >= file_len or end >= file_len or start > end:
            f.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{file_len}")
            self.end_headers()
            return None

        self.send_response(206)
        self.send_header("Content-type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{file_len}")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
        self.end_headers()

        f.seek(start)
        self._range_remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_range_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        bufsize = 64 * 1024
        try:
            while remaining > 0:
                chunk = source.read(min(bufsize, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        finally:
            self._range_remaining = None


class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--directory", "-d", default=".", help="Directory to serve (WanGP outputs/)")
    parser.add_argument("--port", "-p", type=int, default=8090)
    parser.add_argument("--bind", "-b", default="0.0.0.0")
    args = parser.parse_args()

    directory = os.path.abspath(args.directory)
    handler = functools.partial(RangeHTTPRequestHandler, directory=directory)

    with ThreadingHTTPServer((args.bind, args.port), handler) as httpd:
        print(f"Serving {directory} on http://{args.bind}:{args.port} "
              "(threaded, Range-request support enabled)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    main()
