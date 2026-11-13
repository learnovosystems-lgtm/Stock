"""Local dashboard HTTP server — static UI + live ATI JSON APIs."""

from __future__ import annotations

import json
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ati.config import REPORTS_DIR, ROOT, ensure_dirs
from ati.ops.session import run_eod, run_intrasession_once
from ati.web.snapshot import build_snapshot, write_snapshot

DASHBOARD_DIR = ROOT / "dashboard"


class DeskHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def log_message(self, fmt: str, *args) -> None:  # quieter
        print(f"[desk] {args[0] if args else fmt}")

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api(parsed)
            return
        if parsed.path in ("", "/"):
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api(parsed, post=True)
            return
        self._json(404, {"error": "not found"})

    def _handle_api(self, parsed, post: bool = False) -> None:
        try:
            qs = parse_qs(parsed.query)
            if parsed.path == "/api/health":
                self._json(200, {"ok": True, "service": "ati-desk"})
                return
            if parsed.path == "/api/snapshot":
                fast = qs.get("fast", ["0"])[0] == "1"
                snap = build_snapshot(run_scan=not fast, limit_scan=60)
                (DASHBOARD_DIR / "data").mkdir(parents=True, exist_ok=True)
                payload = json.dumps(snap, indent=2, default=str)
                (DASHBOARD_DIR / "data" / "snapshot.json").write_text(payload)
                (REPORTS_DIR / "dashboard_snapshot.json").write_text(payload)
                self._json(200, snap)
                return
            if parsed.path == "/api/session" and post:
                mode = qs.get("mode", ["auto"])[0]
                if mode == "eod":
                    out = run_eod()
                elif mode == "intra":
                    out = run_intrasession_once()
                else:
                    from ati.ops.session import is_market_open

                    out = run_intrasession_once() if is_market_open() else run_eod()
                snap = build_snapshot(run_scan=True, limit_scan=60)
                (DASHBOARD_DIR / "data").mkdir(parents=True, exist_ok=True)
                (DASHBOARD_DIR / "data" / "snapshot.json").write_text(
                    json.dumps(snap, indent=2, default=str)
                )
                self._json(200, {"session": out, "snapshot": snap})
                return
            if parsed.path == "/api/refresh" and post:
                path = write_snapshot(limit_scan=80)
                snap = json.loads(path.read_text())
                (DASHBOARD_DIR / "data").mkdir(parents=True, exist_ok=True)
                (DASHBOARD_DIR / "data" / "snapshot.json").write_text(
                    json.dumps(snap, indent=2, default=str)
                )
                self._json(200, snap)
                return
            self._json(404, {"error": f"unknown api {parsed.path}"})
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": str(exc), "trace": traceback.format_exc()[-2000:]})


def serve(host: str = "127.0.0.1", port: int = 8787) -> None:
    ensure_dirs()
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    (DASHBOARD_DIR / "data").mkdir(parents=True, exist_ok=True)
    # Seed static snapshot if missing
    snap_path = DASHBOARD_DIR / "data" / "snapshot.json"
    if not snap_path.exists():
        write_snapshot(snap_path, run_scan=False)
    httpd = ThreadingHTTPServer((host, port), DeskHandler)
    print(f"ATI Desk listening on http://{host}:{port}")
    httpd.serve_forever()
