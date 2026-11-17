"""
eval_colour_pick.py - run the AUTO colour picker on fixed clicks and save annotated JPEGs.

Each click is run on its own (one AUTO click on a fresh picker, default slider values),
the same way the web app and the OpenCV picker call climb_holds. For every click the
script writes one JPEG with the found outlines (green), the click (magenta cross) and a
header line: hold count, whether the clicked hold is in the result, and the time taken.

    python tools/eval_colour_pick.py --videos "../climbing videos" --tag after
    python tools/eval_colour_pick.py --impl old_climb_holds.py --tag before
    python tools/eval_colour_pick.py --compare before after      # side-by-side JPEGs

--impl loads climb_holds from another file (e.g. `git show HEAD~1:climb_holds.py`), so the
same clicks can be replayed on the old code. Outputs go to eval_out/<tag>/ (git-ignored).
Videos are only read, never copied.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# (video, frame) -> [(name, x, y)] in full-resolution pixels. frame "mid" = total // 2.
CLICKS = {
    ("IMG_3247.mov", "mid"): [("yellow", 474, 245), ("pink", 245, 267), ("green_gloss", 768, 949),
                              ("navy", 875, 235), ("white", 503, 503)],
    ("IMG_5468.mov", "mid"): [("yellow_big", 501, 533), ("yellow_small", 416, 1643), ("red_chalk", 875, 640),
                              ("light_blue", 405, 341), ("green", 213, 171), ("pink", 320, 32)],
}
# Extra cases (other frames / other videos), not tuned against: see README in the eval notes.
EXTRA = ROOT / "tools" / "eval_clicks_extra.json"


def load_impl(path):
    if path is None:
        import climb_holds as ch
        return ch
    spec = importlib.util.spec_from_file_location("climb_holds_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_frame(video, which):
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = total // 2 if which == "mid" else int(which)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read()
    cap.release()
    return idx, (img if ok else None)


def run_pick(ch, img, x, y, tol, min_size):
    """One AUTO click. Returns (holds, extra-info dict, seconds for the cached refresh)."""
    h, w = img.shape[:2]
    min_area = int(ch._fd(h, w)[1] * min_size / 10000)
    if hasattr(ch, "hold_model"):                 # new code: segment the clicked hold, model its colour
        fc = ch.FrameColour(img)
        m = ch.hold_model(fc, x, y)
        t0 = time.perf_counter()
        holds = ch.combine(ch.find_holds_auto(fc, [m], tol, min_area), [], [])
        dt = time.perf_counter() - t0
        info = {"message": m.get("message", "")}
        if not m.get("reject"):
            info["model"] = {k: m[k] for k in ("kind", "hue", "hue_tol", "c_hold", "c_wall", "L", "L_wall", "seed_area")
                             if k in m}
        return holds, info, dt
    lab = ch.to_lab(img)                          # old code: sample a patch, match its colour
    r = max(3, ch._fd(h, w)[0] // 200)
    s = ch.sample_lab(lab, x, y, r)
    t0 = time.perf_counter()
    holds = ch.combine(ch.find_holds_auto(lab, [s], tol, min_area), [], [])
    dt = time.perf_counter() - t0
    return holds, {"sample": [round(v, 1) for v in s]}, dt


def annotate(ch, img, holds, x, y, title, lines):
    out = img.copy()
    hit = [hd for hd in holds if ch.inside(hd, (x, y))]
    for hd in holds:
        col = (0, 255, 0) if hd not in hit else (255, 255, 0)
        cv2.drawContours(out, [np.array(hd["contour"], np.int32)], -1, (0, 0, 0), 7, cv2.LINE_AA)
        cv2.drawContours(out, [np.array(hd["contour"], np.int32)], -1, col, 4, cv2.LINE_AA)
    cv2.drawMarker(out, (x, y), (255, 0, 255), cv2.MARKER_CROSS, 60, 5, cv2.LINE_AA)
    out = cv2.resize(out, (out.shape[1] // 2, out.shape[0] // 2), interpolation=cv2.INTER_AREA)
    y0 = 30
    for i, s in enumerate([title] + lines):
        sc = 0.75 if i == 0 else 0.6
        cv2.putText(out, s, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(out, s, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), 2, cv2.LINE_AA)
        y0 += 28
    return out


def cases(videos, only):
    allc = dict(CLICKS)
    if EXTRA.exists():
        for k, v in json.loads(EXTRA.read_text()).items():
            vid, fr = k.split("@")
            allc[(vid, fr)] = [tuple(c) for c in v]
    for (vid, fr), clicks in allc.items():
        if only and not any(o in vid for o in only):
            continue
        yield videos / vid, fr, clicks


def evaluate(args):
    ch = load_impl(args.impl)
    out_dir = ROOT / "eval_out" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for video, fr, clicks in cases(Path(args.videos), args.only):
        idx, img = read_frame(video, fr)
        if img is None:
            print(f"!! could not read {video.name} frame {fr}")
            continue
        for name, x, y in clicks:
            t0 = time.perf_counter()
            holds, extra, dt = run_pick(ch, img, x, y, args.tol, args.min_size)
            total = time.perf_counter() - t0
            hit = any(ch.inside(hd, (x, y)) for hd in holds)
            title = f"{args.tag}: {video.stem} f{idx} {name} ({x},{y})"
            lines = [f"{len(holds)} holds   clicked hold {'IN' if hit else 'MISSING'}",
                     f"first click {total:.2f}s  refresh {dt * 1000:.0f}ms"]
            if extra.get("message"):
                lines.append(extra["message"][:70])
            if extra.get("sample"):
                lines.append(f"sample Lab {extra['sample']}")
            if extra.get("model"):
                lines.append(" ".join(f"{k}={v}" for k, v in extra["model"].items())[:75])
            jpg = out_dir / f"{video.stem}_f{idx}_{name}.jpg"
            cv2.imwrite(str(jpg), annotate(ch, img, holds, x, y, title, lines), [cv2.IMWRITE_JPEG_QUALITY, 85])
            rows.append({"video": video.name, "frame": idx, "click": name, "xy": [x, y], "holds": len(holds),
                         "clicked_in": hit, "first_s": round(total, 3), "refresh_s": round(dt, 3), **extra})
            print(f"{video.stem:10s} f{idx:<5d} {name:14s} holds={len(holds):3d} clicked={'IN ' if hit else 'OUT'} "
                  f"first={total:.2f}s refresh={dt * 1000:.0f}ms {extra.get('message', '')} {extra.get('model', '')}")
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=1))


def compare(a, b):
    da, db = ROOT / "eval_out" / a, ROOT / "eval_out" / b
    out = ROOT / "eval_out" / f"compare_{a}_vs_{b}"
    out.mkdir(parents=True, exist_ok=True)
    for pa in sorted(da.glob("*.jpg")):
        pb = db / pa.name
        if pb.exists():
            ia, ib = cv2.imread(str(pa)), cv2.imread(str(pb))
            sep = np.full((ia.shape[0], 8, 3), 255, np.uint8)
            cv2.imwrite(str(out / pa.name), np.hstack([ia, sep, ib]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"side-by-side images in {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default=str(ROOT.parent / "climbing videos"))
    ap.add_argument("--impl", default=None, help="climb_holds.py file to test (default: the repo's)")
    ap.add_argument("--tag", default="after")
    ap.add_argument("--tol", type=int, default=22)
    ap.add_argument("--min-size", type=int, default=3)
    ap.add_argument("--only", nargs="*", help="only videos whose name contains one of these")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
