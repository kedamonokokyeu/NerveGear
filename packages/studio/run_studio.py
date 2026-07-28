"""
run_studio.py — zero-install launcher for NerveGear Studio.

Serves the Studio over http://localhost:8777 (ES modules are blocked on
file://, and the webcam gesture feature needs a secure context). Prefers the
Vite build in dist/ when present, else serves the source tree directly —
index.html's importmap resolves the vendored, version-pinned Three.js, so no
network or node install is required.

The physics backend is separate: `cd ../backend && python run.py` (port 8200).
"""

from __future__ import annotations

import http.server
import os
import socketserver
import webbrowser

PORT = int(os.environ.get("STUDIO_PORT", "8777"))
HERE = os.path.dirname(os.path.abspath(__file__))
DIST = os.path.join(HERE, "dist")
ROOT = DIST if os.path.exists(os.path.join(DIST, "index.html")) else HERE


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".json": "application/json",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):  # quiet
        pass


def main() -> None:
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}"
        which = "dist build" if ROOT == DIST else "source (vendored three)"
        print(f"NerveGear Studio → {url}   [{which}]")
        print("Backend expected at http://localhost:8200  (cd ../backend && python run.py)")
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nbye")


if __name__ == "__main__":
    main()
