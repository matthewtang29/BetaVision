"""
Start the BetaVision web app:

    python -m betavision_app            # http://127.0.0.1:8765
    python -m betavision_app --open     # ... and open it in the default browser

run_betavision.bat does this for you on Windows.
"""

import argparse
import socket
import sys
import threading
import time
import webbrowser

sys.dont_write_bytecode = True     # same as climb_analyze.py: no __pycache__ folders


def port_in_use(host, port):
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def open_when_ready(url, host, port):
    for _ in range(120):               # the first start can be slow (MediaPipe import)
        if port_in_use(host, port):
            webbrowser.open(url)
            return
        time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser(description="BetaVision local web app")
    ap.add_argument("--host", default="127.0.0.1", help="default 127.0.0.1 (this computer only)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open the app in your browser")
    args = ap.parse_args()
    url = f"http://{args.host}:{args.port}/"

    if port_in_use(args.host, args.port):  # already running (e.g. double-clicked twice): just show it
        print(f"BetaVision is already running at {url}")
        if args.open:
            webbrowser.open(url)
        return

    from . import jobs
    jobs.reset_workspace()
    print(f"Cleared {jobs.WORKSPACE}")
    import uvicorn
    from .server import app

    if args.open:
        threading.Thread(target=open_when_ready, args=(url, args.host, args.port), daemon=True).start()
    print(f"\nBetaVision is running at {url}\nKeep this window open while you use it. Close it to stop.\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
