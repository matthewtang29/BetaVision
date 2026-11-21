"""
Start the BetaVision web app:

    python -m betavision_app            # http://127.0.0.1:8780
    python -m betavision_app --open     # ... and open it in the default browser
    python -m betavision_app --window   # ... and open it in its own app window (Edge / Chrome)

run_betavision.bat does this for you on Windows.
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

sys.dont_write_bytecode = True     # same as climb_analyze.py: no __pycache__ folders


def port_in_use(host, port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def is_betavision(host, port):
    """Something is listening on the port: is it BetaVision? (Another local server, e.g. a
    `python -m http.server` left running, can use the same port.)"""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/limits", timeout=1.5) as r:
            return "max_upload_mb" in json.loads(r.read().decode("utf-8"))
    except Exception:
        return False


def find_port(host, start, tries=20):
    """From `start` on: (port, True) for the first port already running BetaVision, else
    (port, False) for the first free one, skipping ports other programs use. (None, False)
    if all are taken by other programs."""
    free = None
    for p in range(start, start + tries):
        if not port_in_use(host, p):
            if free is None:
                free = p
        elif is_betavision(host, p):
            return p, True
    return free, False


def app_browser():
    """A Chromium browser that can open a page as its own app window (no tabs or address bar)."""
    names = ["msedge", "chrome", "google-chrome", "chromium", "chromium-browser", "brave"]
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    if sys.platform == "win32":
        for base in (os.environ.get("ProgramFiles(x86)", ""), os.environ.get("ProgramFiles", ""),
                     os.environ.get("LOCALAPPDATA", "")):
            for rel in (r"Microsoft\Edge\Application\msedge.exe", r"Google\Chrome\Application\chrome.exe"):
                path = os.path.join(base, rel)
                if base and os.path.exists(path):
                    return path
    if sys.platform == "darwin":
        for path in ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                     "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"):
            if os.path.exists(path):
                return path
    return None


def show(url, window):
    browser = app_browser() if window else None
    if browser:
        try:
            subprocess.Popen([browser, f"--app={url}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except OSError:
            pass
    webbrowser.open(url)


def open_when_ready(url, host, port, window):
    for _ in range(120):               # the first start can be slow (MediaPipe import)
        if is_betavision(host, port):
            show(url, window)
            return
        time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser(description="BetaVision local web app")
    ap.add_argument("--host", default="127.0.0.1", help="default 127.0.0.1 (this computer only)")
    ap.add_argument("--port", type=int, default=8780)   # not 8765: other local tools commonly use it
    ap.add_argument("--open", action="store_true", help="open the app in your browser")
    ap.add_argument("--window", action="store_true",
                    help="open the app in its own window (Edge or Chrome app mode), else a browser tab")
    args = ap.parse_args()
    opening = args.open or args.window

    port, running = find_port(args.host, args.port)
    if port is None:
        sys.exit(f"Port {args.port} is used by another program, and so are the next 19. "
                 "Close it, or start BetaVision with --port <a free port>.")
    if running:                                 # already running (e.g. double-clicked twice): just show it
        url = f"http://{args.host}:{port}/"
        print(f"BetaVision is already running at {url}")
        if opening:
            show(url, args.window)
        return
    if port != args.port:
        print(f"Port {args.port} is used by another program, so BetaVision is using port {port} instead.")
    args.port = port
    url = f"http://{args.host}:{args.port}/"

    from . import jobs
    jobs.reset_workspace()
    print(f"Cleared {jobs.WORKSPACE}")
    import uvicorn
    from .server import allow_host, app

    allow_host(args.host)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"Warning: serving on {args.host}, so other devices on your network can open BetaVision. "
              "It has no password.")
    if opening:
        threading.Thread(target=open_when_ready, args=(url, args.host, args.port, args.window), daemon=True).start()
    print(f"\nBetaVision is running at {url}\nKeep this window open while you use it. Close it to stop.\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
