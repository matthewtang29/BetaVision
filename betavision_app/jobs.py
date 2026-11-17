"""
jobs.py - the web app's bridge to the existing analysis code.

One Job per uploaded video:
  1. prepare        follows the camera (climb_motion.track_camera) and, if it moved,
                    picks a few keyframes that between them show the whole wall
                    (climb_motion.pick_keyframes).
  2. build_route()  the browser's hold picker sends its clicks; they're replayed
                    through climb_holds / climb_volumes' own detection functions, on
                    every keyframe, and merged into one route in base-frame pixels.
  3. start()        converts the route to base-frame pixels (climb_analyze.route_to_base)
                    and queues climb_analyze.analyze_video() on a background worker,
                    then re-encodes the result to H.264 so it plays in Chrome / Edge.
Nothing here opens an OpenCV window.
"""

import math
import queue
import re
import shutil
import statistics
import subprocess
import sys
import threading
import traceback
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))      # climb_*.py live in the repo root
import climb_analyze as ca         # noqa: E402
import climb_holds as ch           # noqa: E402
import climb_motion as cm          # noqa: E402
import climb_notes as cn           # noqa: E402
import climb_pose as cp            # noqa: E402
import climb_volumes as cvol       # noqa: E402

WORKSPACE = ROOT / "workspace"     # uploads + outputs, wiped every time the app starts
VIDEO_TYPES = {".mp4", ".mov", ".avi"}
LIMB_NAMES = {"l_hand": "Left hand", "r_hand": "Right hand", "l_foot": "Left foot", "r_foot": "Right foot"}
ANGLE_NAMES = {"elbow": "Elbows", "knee": "Knees", "hip": "Hips"}


def reset_workspace():
    shutil.rmtree(WORKSPACE, ignore_errors=True)
    WORKSPACE.mkdir(parents=True, exist_ok=True)


class Job:
    def __init__(self, filename):
        self.id = uuid.uuid4().hex[:12]
        self.dir = WORKSPACE / self.id
        self.dir.mkdir(parents=True)
        p = Path(filename)
        self.name = re.sub(r"[^\w\-]+", "_", p.stem).strip("_") or "climb"   # safe on any filesystem
        self.input = self.dir / f"{self.name}{p.suffix.lower()}"
        self.state, self.stage, self.message, self.progress = "uploading", "", "", 0.0
        self.results = None
        self.lock = threading.Lock()   # cv2.VideoCapture isn't thread-safe
        self.cache = {}                # auto volumes, manual outlines, hold models
        self.fcs = OrderedDict()       # per-frame colour data (big: only the last few frames)
        self.cam, self.moved, self.keyframes = None, 0.0, None   # keyframes: set when the camera moved a lot

    # ---------------------------------------------------------------- open + follow the camera
    def open(self, still):
        cap = cv2.VideoCapture(str(self.input))
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if not ok:
            raise ValueError("Couldn't read that video. Try exporting it again as .mp4.")
        self.reader = ch.FrameReader(self.input)
        self.height, self.width = frame.shape[:2]
        self.still = still
        self.base = 0 if still else self.reader.total // 2          # same base frame as the CLI
        self.state = "preparing"
        threading.Thread(target=self._prepare, daemon=True).start()

    def _prepare(self):
        try:
            if not self.still:
                self.stage, self.message = "camera", "Following the camera"

                def progress(i, n):
                    self.progress = i / n if n else 0.0
                self.cam = cm.track_camera(self.input, self.base, progress=progress)
                self.moved = ca.camera_moved(self.reader, self.cam)
                if self.moved > 25:
                    self.stage, self.message, self.progress = "keyframes", "Choosing frames that show the whole wall", 0.0
                    self.keyframes = self._sharpest(cm.pick_keyframes(self.cam, self.width, self.height))
            self.state, self.stage, self.message = "picking", "", ""
        except Exception as e:
            traceback.print_exc()
            self.state, self.message = "error", f"Couldn't follow the camera ({e}). Try again with the tripod box ticked."

    def _sharpest(self, frames, reach=4):
        """A panning phone blurs some frames: swap each keyframe for the sharpest of its neighbours."""
        out = []
        for k in frames:
            near = range(max(0, k - reach), min(self.reader.total, k + reach + 1), 2)
            score = {}
            for i in near:
                f = self.frame(i)
                if f is not None:
                    g = cv2.cvtColor(cv2.resize(f, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                    score[i] = cv2.Laplacian(g, cv2.CV_32F).var()
            out.append(max(score, key=score.get) if score else k)
        return sorted(set(out))

    def info(self):
        return {"id": self.id, "name": self.name, "frames": self.reader.total, "fps": self.reader.fps,
                "width": self.width, "height": self.height, "keyframes": self.keyframes,
                "camera_moved": self.cam is not None and self.moved > 5}

    # ---------------------------------------------------------------- the picker's picture
    def frame(self, i):
        """Frame i (BGR). The last few are kept decoded: a keyframe refresh needs every
        keyframe several times, and seeking a phone video costs ~0.2 s a frame."""
        i = max(0, min(int(i), self.reader.total - 1))
        with self.lock:
            cache = self.__dict__.setdefault("decoded", OrderedDict())
            if i in cache:
                cache.move_to_end(i)
                return cache[i]
            img = self.reader.read(i)
            if img is not None:
                cache[i] = img
                while len(cache) > max(8, len(self.keyframes or []) + 3):
                    cache.popitem(last=False)
            return img

    def picture_jpeg(self, i, width=None):
        img = self.frame(i)
        if img is None:
            return None
        if width and width < img.shape[1]:          # thumbnails for the keyframe strip
            img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()

    def cached(self, key, fn):
        if key not in self.cache:
            if len(self.cache) > 400:
                self.cache.clear()
            self.cache[key] = fn()
        return self.cache[key]

    def fc(self, i):
        """climb_holds.FrameColour of frame i (Lab, hue, chroma, climber mask), kept for the
        last few frames only: each is ~60 MB at 1080p."""
        i = int(i)
        with self.lock:
            hit = self.fcs.get(i)
            if hit is not None:
                self.fcs.move_to_end(i)
                return hit
        fc = ch.FrameColour(self.frame(i))
        with self.lock:
            self.fcs[i] = fc
            while len(self.fcs) > max(8, len(self.keyframes or []) + 2):
                self.fcs.popitem(last=False)
        return fc

    def lab(self, i):
        return self.fc(i).lab

    def models(self, st, cur):
        """AUTO clicks -> hold models (one per click, cached). Clicks that aren't on a hold
        come back in `rejected` as {index, message} so the page can drop them again."""
        models, rejected = [], []
        for k, c in enumerate(st.get("auto", [])):
            f, x, y = int(c.get("frame", cur)), int(c["x"]), int(c["y"])
            m = self.cached(("model", f, x, y), lambda f=f, x=x, y=y: ch.hold_model(self.fc(f), x, y, frame=f))
            if m.get("reject"):
                rejected.append({"index": k, "message": m["message"]})
            elif any(o["frame"] == f and o.get("seed") and ch.inside({"contour": o["seed"]}, (x, y)) for o in models):
                # a second click on a hold that's already a sample (the page checks this too, but
                # only once the first click's outlines have come back)
                rejected.append({"index": k, "message": "That hold is already selected, so nothing was added."})
            else:
                models.append(m)
        return models, rejected

    # ---------------------------------------------------------------- route (same steps as the picker window)
    def build_route(self, st):
        """st = the browser's picker state:
             frame, tol, min_size        current frame and the two picker sliders
             frames [i, ...]             keyframes the route is marked on (camera moved a lot)
             auto   [{x, y, frame}]      AUTO clicks (each one's hold is outlined and its colour
                                         modelled on the frame clicked; see climb_holds.hold_model)
             manual [{x, y, frame}]      MANUAL clicks (one hold each, outlined on the frame clicked)
             deleted [{x, y, frame, contour}]  right-clicked holds' outlines (old: [x, y, frame])
             starts [[x, y, frame]], end [x, y, frame] or null
             vols_manual [{kind: "click", x, y, frame} | {kind: "poly", points, frame}], vols_deleted [[x, y, frame]]
        Returns (img, holds, vols, info, extra) like climb_holds.interactive, plus counts for the page
        and rejected_auto ([{index, message}]: AUTO clicks that weren't on a hold).
        With keyframes, holds + volumes come back in BASE-frame pixels (img = the base frame)."""
        if self.keyframes:
            return self._build_route_keyframes(st)
        frame = int(st.get("frame", 0))
        img = self.frame(frame)
        h, w = img.shape[:2]
        fc = self.fc(frame)
        lab = fc.lab
        models, rejected = self.models(st, frame)
        tol = max(1, int(st.get("tol", 22)))
        min_area = int(ch._fd(h, w)[1] * int(st.get("min_size", 3)) / 10000)
        outlines, points = split_deleted(st.get("deleted"), frame)
        auto = ch.find_holds_auto(fc, ch.models_for_frame(models, frame), tol, min_area,
                                  deleted=[c for _, c in outlines])

        manual = []
        for k, c in enumerate(st.get("manual", [])):
            x, y = int(c["x"]), int(c["y"])
            hold = self.cached(("hold", int(c["frame"]), x, y),
                               lambda c=c, x=x, y=y: ch.outline_one_hold_img(self.frame(c["frame"]), x, y)
                               or ch.outline_one_hold(self.lab(c["frame"]), x, y))
            if hold:
                manual.append(dict(hold, manual_index=k))      # lets the page undo / right-click this click
        starts = [p[:2] for p in st.get("starts", [])]
        holds = ch.combine(auto, manual, [p[:2] for p in points], starts,
                           st["end"][:2] if st.get("end") else None)
        mark_starts(holds, starts)

        vols_auto = self.cached(("vols", frame), lambda: cvol.detect_volumes(img))
        vols = [dict(v) for v in vols_auto if not any(cvol.inside_volume(v, p) for p in st.get("vols_deleted", []))]
        # painted volumes that aren't the route's colour are off-route (manual ones always kept)
        vols, skipped = ch.filter_coloured_volumes(lab, vols, holds)
        failed = []
        for k, v in enumerate(st.get("vols_manual", [])):
            vol = self._manual_volume(v)
            if vol is None:
                failed.append(k)
            else:
                vols.append(dict(vol, manual_index=k))
        vols = cvol.cut_foreign_holds(cvol.renumber(vols), img, ch.route_mask(holds, img.shape), min_area)
        info = {"frame": frame, "colour_tol": tol, "min_area_px": min_area,
                "colour_samples_lab": [m["lab"] for m in models], "colour_models": models,
                "clicks": [[c["x"], c["y"]] for c in st.get("auto", [])], "deleted_points": st.get("deleted", []),
                "volumes": vols}
        return img, holds, vols, info, {"skipped_volumes": len(skipped), "failed_volumes": failed,
                                        "rejected_auto": rejected}

    def _manual_volume(self, v):
        if v.get("kind") == "poly":
            pts = [[int(x), int(y)] for x, y in v["points"]]
            return {"contour": pts, "faces": [pts], "source": "manual"}
        x, y = int(v["x"]), int(v["y"])
        return self.cached(("vol", int(v["frame"]), x, y), lambda: cvol.volume_from_click(self.frame(v["frame"]), x, y))

    def _build_route_keyframes(self, st):
        """The route marked across several keyframes. Each keyframe is detected on its own
        (the same functions as above), moved into base-frame pixels with the tracked camera,
        and merged: a hold seen in more than one keyframe is kept from the frame that shows
        it furthest from the picture's edge (so it isn't cut off there)."""
        cur = int(st.get("frame", self.keyframes[0]))
        h, w = self.height, self.width
        to_base = lambda f: np.linalg.inv(self.cam[f])
        on = lambda c: int(c.get("frame", cur))
        pt_base = lambda p: cm.warp_points([p[:2]], to_base(int(p[2]) if len(p) > 2 else cur))[0].tolist()
        frames = set(int(f) for f in st.get("frames") or self.keyframes)
        frames |= {on(c) for key in ("auto", "manual", "vols_manual") for c in st.get(key, [])}
        frames = sorted(f for f in frames if 0 <= f < len(self.cam))

        models, rejected = self.models(st, cur)
        tol = max(1, int(st.get("tol", 22)))
        min_area = int(ch._fd(h, w)[1] * int(st.get("min_size", 3)) / 10000)
        outlines, points = split_deleted(st.get("deleted"), cur)

        def deleted_on(f):                                 # deleted outlines, moved into frame f's pixels
            return [cm.warp_points(c, self.cam[f] @ np.linalg.inv(self.cam[src])).tolist() for src, c in outlines]
        key = (tuple((on(c), int(c["x"]), int(c["y"])) for c in st.get("auto", [])), tol, min_area,
               tuple((src, len(c), tuple(c[0])) for src, c in outlines))
        with ThreadPoolExecutor(max_workers=4) as ex:      # OpenCV releases the GIL, so keyframes run side by side
            autos = list(ex.map(lambda f: self.cached(("auto", f, key), lambda: ch.find_holds_auto(
                self.fc(f), ch.models_for_frame(models, f), tol, min_area, deleted=deleted_on(f))), frames))
        clicks = [(on(c), [int(c["x"]), int(c["y"])]) for c in st.get("auto", [])]
        autos = self.cached(("agree", tuple(frames), key), lambda: agree_across_keyframes(
            frames, autos, self.cam, (h, w), clicks, lambda f: self.fc(f).person()))
        cands = [(edge_room(hd["contour"], w, h), f, hd) for f, holds_f in zip(frames, autos) for hd in holds_f]
        auto = [b for _, _, b in merge_views(cands, to_base, ca.warp_holds, ch.inside)]

        manual = []
        for k, c in enumerate(st.get("manual", [])):
            f, x, y = on(c), int(c["x"]), int(c["y"])
            hold = self.cached(("hold", f, x, y), lambda: ch.outline_one_hold_img(self.frame(f), x, y)
                               or ch.outline_one_hold(self.lab(f), x, y))
            if hold:
                manual.append(dict(ca.warp_holds([hold], to_base(f))[0], manual_index=k))
        starts = [pt_base(p) for p in st.get("starts", [])]
        holds = ch.combine(auto, manual, [pt_base(p) for p in points], starts,
                           pt_base(st["end"]) if st.get("end") else None)
        mark_starts(holds, starts)

        # volumes: detected per keyframe, painted ones filtered against the route there, then merged
        vols_deleted = [pt_base(p) for p in st.get("vols_deleted", [])]
        kept, skipped = [], []
        for f in frames:
            img = self.frame(f)
            found = [dict(v) for v in self.cached(("vols", f), lambda: cvol.detect_volumes(img))]
            found = [v for v in found if not any(cvol.inside_volume(v, p) for p in
                                                 cm.warp_points(vols_deleted, self.cam[f]).tolist())] if vols_deleted else found
            k_f, s_f = ch.filter_coloured_volumes(self.lab(f), found, ca.warp_holds(holds, self.cam[f]))
            kept += [(edge_room(v["contour"], w, h), f, v) for v in k_f]
            skipped += [(edge_room(v["contour"], w, h), f, v) for v in s_f]
        by_frame = {}                                      # frame -> its volumes, in that frame's pixels
        for f, v, _ in merge_views(kept, to_base, ca.warp_volumes, cvol.inside_volume):
            by_frame.setdefault(f, []).append(v)
        failed = []
        for k, v in enumerate(st.get("vols_manual", [])):
            vol = self._manual_volume(v)
            if vol is None:
                failed.append(k)
            else:
                by_frame.setdefault(on(v), []).append(dict(vol, manual_index=k))
        vols = []
        for f, vs in by_frame.items():                     # off-route holds are cut out on the frame they're seen in
            vs = cvol.cut_foreign_holds(vs, self.frame(f), ch.route_mask(ca.warp_holds(holds, self.cam[f]), (h, w)), min_area)
            vols += ca.warp_volumes(vs, to_base(f))
        vols = cvol.renumber(vols)
        n_skipped = sum(1 for _, f, v in skipped
                        if not any(cvol.inside_volume(o, center(ca.warp_volumes([v], to_base(f))[0]["contour"])) for o in vols))

        info = {"frame": self.base, "keyframes": frames, "colour_tol": tol, "min_area_px": min_area,
                "colour_samples_lab": [m["lab"] for m in models], "colour_models": models,
                "clicks": [[c["x"], c["y"], on(c)] for c in st.get("auto", [])],
                "deleted_points": st.get("deleted", []), "volumes": vols}
        return self.frame(self.base), holds, vols, info, {"skipped_volumes": n_skipped, "failed_volumes": failed,
                                                          "rejected_auto": rejected}

    def preview(self, st):
        _, holds, vols, _, extra = self.build_route(st)
        if self.keyframes:                                 # base-frame pixels -> the frame on screen
            Hm = self.cam[int(st.get("frame", self.keyframes[0]))]
            holds, vols = ca.warp_holds(holds, Hm), ca.warp_volumes(vols, Hm)
        return {"holds": holds, "volumes": vols, **extra}

    def route_for_analysis(self, st):
        """The marked route in BASE-frame pixels (also saved as <name>_holds.json / .png)."""
        img, holds, vols, info, _ = self.build_route(st)

        def strip(items):
            return [{k: v for k, v in d.items() if k not in ("manual_index", "start_idx")} for d in items]
        holds, info["volumes"] = strip(holds), strip(vols)
        if self.keyframes:
            to_base = np.eye(3)                            # already there
        else:
            to_base = np.linalg.inv(self.cam[info["frame"]]) if self.cam is not None else np.eye(3)
        return ca.route_to_base(self.input, self.dir, True, img, holds, info, to_base, self.base)

    # ---------------------------------------------------------------- status
    def status(self):
        return {"id": self.id, "state": self.state, "stage": self.stage, "message": self.message,
                "progress": round(self.progress, 3), "results": self.results,
                "info": self.info() if self.state == "picking" else None}


def split_deleted(entries, cur):
    """Right-click deletions: {x, y, frame, contour} (the hold's outline, stays deleted at any
    tolerance) or the old [x, y, frame] points. Returns ([(frame, outline)], [[x, y, frame]])."""
    outlines, points = [], []
    for d in entries or []:
        if isinstance(d, dict):
            if d.get("contour"):
                outlines.append((int(d.get("frame", cur)), d["contour"]))
            else:
                points.append([d["x"], d["y"], int(d.get("frame", cur))])
        else:
            points.append(list(d))
    return outlines, points


def center(contour):
    return np.mean(np.asarray(contour, float), axis=0).tolist()


def edge_room(contour, w, h):
    """How far (px) an outline stays from the edge of a w x h frame: 0 = cut off by it."""
    x, y, bw, bh = cv2.boundingRect(np.asarray(contour, np.int32))
    return min(x, y, w - x - bw, h - y - bh)


def agree_across_keyframes(frames, autos, cam, size, clicks, person):
    """Camera moved: a hold found on one keyframe should also be found on the other keyframes
    that show the same spot (clear of the picture's edge and not hidden by the climber).
    Drops the ones that fewer than half of those views found: things that move (the climber,
    a rope, a shadow) or a colour that only matched in one lighting. Clicked holds always stay."""
    if len(frames) < 2:
        return autos
    h, w = size
    margin = 0.02 * ch._fd(h, w)[0]
    out = []
    for i, f in enumerate(frames):
        kept = []
        for hd in autos[i]:
            if any(cf == f and ch.inside(hd, p) for cf, p in clicks):
                kept.append(hd)
                continue
            seen, found = 1, 1
            for j, g in enumerate(frames):
                if g == f:
                    continue
                hg = ca.warp_holds([hd], cam[g] @ np.linalg.inv(cam[f]))[0]
                if edge_room(hg["contour"], w, h) < margin:
                    continue                               # off (or cut by) frame g's edge
                m = np.zeros((h, w), np.uint8)
                cv2.fillPoly(m, [np.asarray(hg["contour"], np.int32)], 1)
                if not m.any() or person(g)[m > 0].mean() > 0.3:
                    continue                               # the climber hides it on frame g
                seen += 1
                c = center(hg["contour"])
                found += any(ch.inside(o, c) or ch.inside(hg, center(o["contour"])) for o in autos[j])
            if found >= 0.5 * seen:
                kept.append(hd)
        out.append(kept)
    return out


def merge_views(cands, to_base, warp, inside):
    """cands = [(edge_room, frame, item in that frame's pixels)] from several keyframes.
    Returns [(frame, item, item in base pixels)], one per real hold / volume: where views
    overlap, the one furthest from its frame's edge wins."""
    out = []
    for _, f, it in sorted(cands, key=lambda t: -t[0]):
        b = warp([it], to_base(f))[0]
        if not any(inside(o, center(b["contour"])) or inside(b, center(o["contour"])) or same_place(o, b)
                   for _, _, o in out):
            out.append((f, it, b))
    return out


def same_place(a, b):
    """Two outlines of one object: their convex hulls mostly overlap. (The centre test in
    merge_views misses crescent / two-lobed holds, whose centre lies outside their own outline.)"""
    ha = cv2.convexHull(np.asarray(a["contour"], np.float32))
    hb = cv2.convexHull(np.asarray(b["contour"], np.float32))
    small = min(cv2.contourArea(ha), cv2.contourArea(hb))
    if small <= 0:
        return False
    inter, _ = cv2.intersectConvexConvex(ha, hb)
    return inter > 0.5 * small


def mark_starts(holds, starts):
    """Which start clicks landed on each hold (lets the page untag one from any frame)."""
    for hd in holds:
        hd["start_idx"] = [k for k, p in enumerate(starts) if ch.inside(hd, p)]


# ---------------------------------------------------------------- results summary
def summarize(job, res, holds, vols):
    """Turn analyze_video's per-frame data into what the results panel shows."""
    fps, frames = res["fps"], res["frames"]
    min_run = max(3, int(round(0.2 * fps)))        # ignore contacts shorter than ~0.2 s (tracking jitter)
    moves = []
    for limb, label in LIMB_NAMES.items():
        run_hold, run_start = "", 0
        for i, lh in enumerate(res["limb_holds"] + [{}]):
            hid = lh.get(limb, "")
            if hid != run_hold:
                if run_hold != "" and i - run_start >= min_run:
                    moves.append({"t": round(run_start / fps, 2), "dur": round((i - run_start) / fps, 2),
                                  "limb": label, "hold": run_hold, "volume": isinstance(run_hold, str)})
                run_hold, run_start = hid, i
    moves.sort(key=lambda m: (m["t"], m["limb"]))
    roles = {hd["id"]: hd.get("role", "") for hd in holds}
    for m in moves:
        m["role"] = "" if m["volume"] else roles.get(m["hold"], "")

    angles = []
    for key, label in ANGLE_NAMES.items():
        vals = [a[k] for a in res["angles"] for k in (f"l_{key}_deg", f"r_{key}_deg")
                if k in a and not math.isnan(a[k])]
        if vals:
            angles.append({"joint": label, "min": round(min(vals)), "median": round(statistics.median(vals)),
                           "max": round(max(vals))})

    notes = []
    for n in res["notes"]:
        if n.get("hold") is None:
            to = (f"your {n['limb'].replace('l_', 'left ').replace('r_', 'right ')}" if n.get("limb")
                  else "the top of the video")
        else:
            to = f"volume {n['hold']}" if isinstance(n["hold"], str) else f"hold {n['hold']}"
        notes.append({"start": n["start"], "end": n["end"], "text": n["text"], "to": to})

    used = sorted({m["hold"] for m in moves if not m["volume"]})
    return {"frames": frames, "fps": round(fps, 2), "duration_s": round(frames / fps, 2),
            "detected": res["detected"], "detected_pct": round(100 * res["detected"] / max(frames, 1)),
            "holds_total": len(holds), "holds_used": len(used), "volumes": len(vols), "has_end": res["has_end"],
            "topped_s": round(res["topped_s"], 2) if res["topped_s"] is not None else None,
            "moves": moves, "angles": angles, "notes": notes,
            "video_url": f"/api/jobs/{job.id}/video", "download_url": f"/api/jobs/{job.id}/download",
            "csv_url": f"/api/jobs/{job.id}/csv"}


def to_h264(src, dst):
    """OpenCV writes mp4v, which browsers won't play. Re-encode with the ffmpeg
    that ships inside the imageio-ffmpeg wheel (nothing to install separately)."""
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src),
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",          # H.264 4:2:0 needs even sizes
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-an", str(dst)]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)          # no console flash on Windows
    subprocess.run(cmd, check=True, capture_output=True, creationflags=flags)


def parse_notes(text):
    """Sticky notes typed in the page, one per line (same format as a --notes file)."""
    lines = (text or "").splitlines()
    cn.load_notes(lines)           # raises ValueError with a readable message on a bad line
    return lines


# ---------------------------------------------------------------- the background worker
STAGES = {"lock": "Locking each hold onto the wall", "track": "Tracking the climber",
          "draw": "Drawing the video"}


class Worker:
    """Runs one analysis at a time in a background thread, so the page stays responsive."""

    def __init__(self):
        self.jobs = {}
        self.q = queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def new_job(self, filename):
        job = Job(filename)
        self.jobs[job.id] = job
        return job

    def start(self, job, picker_state, model, notes, use_lock):
        if picker_state:
            job.holds, job.vols = job.route_for_analysis(picker_state)
        else:
            job.holds, job.vols = [], []
        job.model, job.notes, job.use_lock = model, notes, use_lock
        job.state, job.stage, job.progress = "queued", "queued", 0.0
        job.message = "Waiting for the previous climb to finish"
        self.q.put(job)

    def _loop(self):
        while True:
            job = self.q.get()
            try:
                self._run(job)
            except Exception as e:                      # show the error on the page, keep the server alive
                traceback.print_exc()
                job.state, job.message = "error", f"Analysis failed: {e}"

    def _run(self, job):
        job.state = "running"
        if not (Path.home() / ".climb_analyzer" / f"pose_landmarker_{job.model}.task").exists():
            job.stage, job.message = "model", f"Downloading the {job.model} pose model (one time only)"
            cp.ensure_model(job.model)

        def progress(stage, done, total):
            job.stage, job.message = stage, STAGES[stage]
            job.progress = done / total if total else 0.0

        cam = job.cam if job.moved > 5 and job.holds else None    # still camera: nothing to warp (as in the CLI)
        res = ca.analyze_video(job.input, job.holds, job.vols, cam, job.model, job.dir, save_data=True,
                               show=False, use_lock=job.use_lock, note_args=job.notes,
                               progress=progress, log=lambda msg: print(f"[{job.id}] {msg}"))
        if res["frames"] == 0:
            raise ValueError("No frames could be read from the video")
        job.stage, job.message, job.progress = "encode", "Encoding the video for your browser", 1.0
        job.output = job.dir / f"{job.name}_analysis_h264.mp4"
        to_h264(res["out_video"], job.output)
        job.csv = res["out_csv"]
        job.results = summarize(job, res, job.holds, job.vols)
        job.state, job.message = "done", ""
        print(f"[{job.id}] done: {job.output}")
