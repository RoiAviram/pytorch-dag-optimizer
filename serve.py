"""
serve.py
--------
Minimal HTTP server for the PyTorch DAG Optimizer frontend.

Serves:
  /           → frontend/index.html
  /frontend/* → frontend/ static assets (JS, CSS)
  /output/*   → output/ JSON files (graph.json, optimized_graph.json)

Usage:
  python serve.py            # default port 8080
  python serve.py --port 5000
"""

import argparse
import http.server
import os
import urllib.parse

# ── Resolve project root (directory containing this script) ──────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))

FRONTEND_DIR = os.path.join(ROOT, "frontend")
OUTPUT_DIR   = os.path.join(ROOT, "output")


class DAGHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # noqa: N802
        print(f"  [{self.command}] {self.path}  →  {args[1]}")

    def do_GET(self):  # noqa: N802
        parsed   = urllib.parse.urlparse(self.path)
        url_path = parsed.path

        # ── Route table ──────────────────────────────────────────────────────
        if url_path in ('/', '/index.html'):
            self._serve_file(os.path.join(FRONTEND_DIR, "index.html"), "text/html")

        elif url_path.startswith("/output/"):
            rel = url_path[len("/output/"):]
            self._serve_file(os.path.join(OUTPUT_DIR, rel), "application/json")

        elif url_path.startswith("/frontend/") or url_path.startswith("/static/"):
            # Strip the leading /frontend/ or /static/
            segments = url_path.split("/", 2)
            rel = segments[2] if len(segments) >= 3 else ""
            self._serve_file(os.path.join(FRONTEND_DIR, rel), self._mime(rel))

        else:
            # Try serving directly from frontend/ (for app.js, style.css, etc.)
            rel  = url_path.lstrip("/")
            path = os.path.join(FRONTEND_DIR, rel)
            if os.path.isfile(path):
                self._serve_file(path, self._mime(rel))
            else:
                self.send_error(404, f"Not found: {self.path}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _serve_file(self, abs_path: str, content_type: str) -> None:
        if not os.path.isfile(abs_path):
            self.send_error(404, f"File not found: {abs_path}")
            return
        data = open(abs_path, "rb").read()
        self.send_response(200)
        self.send_header("Content-Type",   content_type)
        self.send_header("Content-Length", str(len(data)))
        # CORS: allow local JS fetch
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    @staticmethod
    def _mime(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        return {
            ".html": "text/html",
            ".css":  "text/css",
            ".js":   "application/javascript",
            ".json": "application/json",
            ".ico":  "image/x-icon",
            ".png":  "image/png",
            ".svg":  "image/svg+xml",
        }.get(ext, "application/octet-stream")


def main() -> None:
    parser = argparse.ArgumentParser(description="DAG Optimizer dev server")
    parser.add_argument("--port", type=int, default=8080, help="TCP port (default 8080)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default 127.0.0.1)")
    args = parser.parse_args()

    addr = (args.host, args.port)
    httpd = http.server.HTTPServer(addr, DAGHandler)

    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │     PyTorch DAG Optimizer  ·  Dev Server             │")
    print("  ├─────────────────────────────────────────────────────┤")
    print(f"  │  URL  :  http://{args.host}:{args.port:<37}│")
    print(f"  │  Serving frontend/ and output/                      │")
    print("  │  Press Ctrl+C to stop                               │")
    print("  └─────────────────────────────────────────────────────┘")
    print()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")


if __name__ == "__main__":
    main()
