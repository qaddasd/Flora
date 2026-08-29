"""Dev server for the Flora web demo.

Serves web-demo/ over HTTP plus a POST /report endpoint that the in-browser
self-test (?autotest=1) uses to persist its results to autotest-report.txt,
so headless runs can be verified without scraping the DOM.

Usage:  python tools/dev_server.py [port]   (default 8765)
"""

import http.server
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent / "web-demo"
REPORT = ROOT / "autotest-report.txt"


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def translate_path(self, path):
        p = pathlib.Path(super().translate_path(path))
        if not p.exists():
            for cand in (p.with_name(p.name + ".html"), p / "index.html"):
                if cand.exists():
                    return str(cand)
        return str(p)

    def send_error(self, code, message=None, explain=None):
        if code == 404:
            try:
                body = (ROOT / "404.html").read_bytes()
            except OSError:
                return super().send_error(code, message, explain)
            self.send_response(404, message)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
            return
        super().send_error(code, message, explain)

    def do_POST(self):
        if self.path == "/report":
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            REPORT.write_bytes(body)
            self.send_response(204)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):  # quiet
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f"serving {ROOT} on http://127.0.0.1:{port}")
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
