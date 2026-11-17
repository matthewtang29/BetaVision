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

import json
import math
import os
import queue
import re
import shutil
import statistics
import subprocess
import sys
import threading
import traceback
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))      # climb_*.py live in the repo root
import climb_analyze as ca         # noqa: E402
import climb_compare as ccmp       # noqa: E402
import climb_holds as ch           # noqa: E402
import climb_motion as cm          # noqa: E402
import climb_notes as cn           # noqa: E402
import climb_offroute as co        # noqa: E402
import climb_pose as cp            # noqa: E402
import climb_volumes as cvol       # noqa: E402
from .cache import ByteLRU          # noqa: E402

WORKSPACE = ROOT / "workspace"     # uploads + outputs, wiped every time the app starts
VIDEO_TYPES = {".mp4", ".mov", ".avi"}
LIMB_NAMES = {"l_hand": "Left hand", "r_hand": "Right hand", "l_foot": "Left foot", "r_foot": "Right foot"}
ANGLE_NAMES = {"elbow": "Elbows", "knee": "Knees", "hip": "Hips"}
CACHE_BYTES = int(float(os.environ.get("BETAVISION_CACHE_MB", 1536)) * 1024 * 1024)   # per job: frames, colour data, outlines
MAX_KEPT_JOBS = 4         # jobs that aren't busy (picking / done / failed) kept in memory; the least recently used are freed
BUSY = ("uploading", "preparing", "starting", "queued", "running")
COMPARE_BUSY = ("queued", "running")
COMPARE_LOCK = threading.Lock()    # one comparison at a time (each tracks a whole video)


class PickerError(ValueError):
    """The picker state sent by the page isn't usable. The message is shown to the user (HTTP 400)."""


class JobBusy(Exception):
    """The job isn't in a state that allows starting (already started)."""


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
        self.lock = threading.Lock()   # cv2.VideoCapture isn't thread-safe: held while reading a frame
        self.cache = ByteLRU(CACHE_BYTES)   # decoded frames, per-frame colour data (~60 MB each at 1080p), outlines, models
        self.reader = None
        self.touched = time.monotonic()
        self.discard = False           # freed while busy: free it when it stops
        self.cam, self.moved, self.keyframes = None, 0.0, None   # keyframes: set when the camera moved a lot
        self.off_route, self.flag_off_route = [], True             # off-route wall holds (base pixels), flag touches?
        self.pose = None               # after the analysis: (px, vis, wall) per frame, for a comparison
        self.cam_used = None           # the camera homographies the analysis used (None = still camera)
        self.compare = None            # comparison with a second climb: {state, message, progress, results}
        self.compare_files = {}        # its outputs: video (H.264), check (registration JPEG)

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
        if self.discard:                                # the user started over while this ran
            self.discard = False
            self.free()

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
        """Frame i (BGR), shared between callers (don't draw on it). Decoded frames are kept
        in the job's byte-bounded cache: a keyframe refresh needs every keyframe several
        times, and seeking a phone video costs ~0.2 s a frame."""
        reader = self.reader
        if reader is None:
            raise PickerError("This climb has been closed. Upload it again.")
        i = max(0, min(int(i), reader.total - 1))
        return self.cache.get_or_compute(("frame", i), lambda: self._read(reader, i))

    def _read(self, reader, i):
        with self.lock:
            return reader.read(i)

    def picture_jpeg(self, i, width=None):
        img = self.frame(i)
        if img is None:
            return None
        if width and width < img.shape[1]:          # thumbnails for the keyframe strip
            img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()

    def cached(self, key, fn):
        """fn()'s result for key, computed once even when several requests ask at the same time."""
        return self.cache.get_or_compute(key, fn)

    def fc(self, i):
        """climb_holds.FrameColour of frame i (Lab, hue, chroma, climber mask). Big, so it
        shares the job's byte budget with everything else and the oldest go first."""
        i = int(i)

        def make():
            img = self.frame(i)
            if img is None:
                raise PickerError(f"Couldn't read frame {i} of the video.")
            return ch.FrameColour(img)
        return self.cache.get_or_compute(("fc", i), make)

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
             off_deleted [{x, y, frame, contour}]  right-clicked off-route outlines (not really off-route holds)
             cuts_deleted [[x, y, frame]]          right-clicked volume cut-outs (not really holds)
        Returns (img, holds, vols, info, extra) like climb_holds.interactive, plus counts for the page
        and rejected_auto ([{index, message}]: AUTO clicks that weren't on a hold).
        info["off_route"] = the off-route holds on the plain wall (climb_offroute), same pixels as holds.
        With keyframes, holds + volumes come back in BASE-frame pixels (img = the base frame)."""
        st = self.clean_picker(st)
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
            cf = int(c.get("frame", frame))
            hold = self.cached(("hold", cf, x, y),
                               lambda cf=cf, x=x, y=y: ch.outline_one_hold_img(self.frame(cf), x, y)
                               or ch.outline_one_hold(self.lab(cf), x, y))
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
        vols = co.drop_cuts(vols, [p[:2] for p in st.get("cuts_deleted", [])])
        off_del, _ = split_deleted(st.get("off_deleted"), frame)
        off = co.find_off_route(fc, holds, vols, models, [c for _, c in off_del], min_area,
                                cands=self.cached(("offcands", frame, min_area), lambda: co.detect(fc, min_area)))
        info = {"frame": frame, "colour_tol": tol, "min_area_px": min_area,
                "colour_samples_lab": [m["lab"] for m in models], "colour_models": models,
                "clicks": [[c["x"], c["y"]] for c in st.get("auto", [])], "deleted_points": st.get("deleted", []),
                "volumes": vols, "off_route": off}
        return img, holds, vols, info, {"skipped_volumes": len(skipped), "failed_volumes": failed,
                                        "rejected_auto": rejected}

    def _manual_volume(self, v):
        if v.get("kind") == "poly":
            pts = [[int(x), int(y)] for x, y in v["points"]]
            return {"contour": pts, "faces": [pts], "source": "manual"}
        x, y = int(v["x"]), int(v["y"])
        vf = int(v["frame"])
        return self.cached(("vol", vf, x, y), lambda: cvol.volume_from_click(self.frame(vf), x, y))

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

        def deleted_on(f, outlines=outlines):              # deleted outlines, moved into frame f's pixels
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
            hold = self.cached(("hold", f, x, y), lambda f=f, x=x, y=y: ch.outline_one_hold_img(self.frame(f), x, y)
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
            found = [dict(v) for v in self.cached(("vols", f), lambda img=img: cvol.detect_volumes(img))]
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
        vols = co.drop_cuts(cvol.renumber(vols), [pt_base(p) for p in st.get("cuts_deleted", [])])
        off = self._off_route_keyframes(frames, holds, vols, models, min_area, deleted_on,
                                        split_deleted(st.get("off_deleted"), cur)[0])
        n_skipped = sum(1 for _, f, v in skipped
                        if not any(cvol.inside_volume(o, center(ca.warp_volumes([v], to_base(f))[0]["contour"])) for o in vols))

        info = {"frame": self.base, "keyframes": frames, "colour_tol": tol, "min_area_px": min_area,
                "colour_samples_lab": [m["lab"] for m in models], "colour_models": models,
                "clicks": [[c["x"], c["y"], on(c)] for c in st.get("auto", [])],
                "deleted_points": st.get("deleted", []), "volumes": vols, "off_route": off}
        return self.frame(self.base), holds, vols, info, {"skipped_volumes": n_skipped, "failed_volumes": failed,
                                                          "rejected_auto": rejected}

    def _off_route_keyframes(self, frames, holds, vols, models, min_area, deleted_on, off_deleted):
        """Off-route wall holds across the keyframes, in base pixels: found on each keyframe,
        kept only if the other keyframes that show the spot agree (as for route holds), merged."""
        h, w = self.height, self.width
        sels = []
        for f in frames:
            fc = self.fc(f)
            cands = self.cached(("offcands", f, min_area), lambda fc=fc: co.detect(fc, min_area))
            sels.append(co.select(cands, ca.warp_holds(holds, self.cam[f]), models,
                                  deleted_on(f, off_deleted), (h, w)))
        sels = agree_across_keyframes(frames, sels, self.cam, (h, w), [], lambda f: self.fc(f).person())
        cands = [(edge_room(hd["contour"], w, h), f, hd) for f, sel in zip(frames, sels) for hd in sel]
        merged = [b for _, _, b in merge_views(cands, lambda f: np.linalg.inv(self.cam[f]), ca.warp_holds, ch.inside)]
        return co.number(co.drop_on_volumes(merged, vols))

    def preview(self, st):
        _, holds, vols, info, extra = self.build_route(st)
        off = info.get("off_route", [])
        if self.keyframes:                                 # base-frame pixels -> the frame on screen
            Hm = self.cam[int(st.get("frame", self.keyframes[0]))]
            holds, vols, off = ca.warp_holds(holds, Hm), ca.warp_volumes(vols, Hm), ca.warp_holds(off, Hm)
        return {"holds": holds, "volumes": vols, "off_route": off, **extra}

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
        self.off_route = ca.off_route_to_base(info, to_base)
        return ca.route_to_base(self.input, self.dir, True, img, holds, info, to_base, self.base)

    # ---------------------------------------------------------------- picker state from the page
    def clean_picker(self, st):
        """Check the picker state the page sent and return a copy with every number turned
        into the right type and the optional parts filled in. Raises PickerError with a
        message the user can read instead of letting a missing key or a bad number become a 500."""
        reader = self.reader
        total = reader.total if reader is not None else 0
        if self.cam is not None:
            total = min(total, len(self.cam))
        if total < 1:
            raise PickerError("This climb has no frames to mark. Upload it again.")
        return clean_picker(st, self.width, self.height, total, self.keyframes[0] if self.keyframes else 0)

    # ---------------------------------------------------------------- lifetime
    def release(self):
        """Give back the memory: caches and the open video. The files stay (results are served from them)."""
        self.cache.clear()
        with self.lock:
            reader, self.reader = self.reader, None
            if reader is not None:
                reader.cap.release()

    def busy(self):
        return self.state in BUSY or (self.compare or {}).get("state") in COMPARE_BUSY

    def free(self):
        """Release everything and delete this job's files. A job that is still working is
        marked and freed when it stops."""
        if self.busy():
            self.discard = True
            return False
        self.release()
        shutil.rmtree(self.dir, ignore_errors=True)
        return True

    # ---------------------------------------------------------------- status
    def status(self):
        return {"id": self.id, "state": self.state, "stage": self.stage, "message": self.message,
                "progress": round(self.progress, 3), "results": self.results, "compare": self.compare,
                "info": self.info() if self.state == "picking" else None}


def _num(v, what, lo=None, hi=None):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise PickerError(f"{what} should be a number (got {v!r}).")
    if (lo is not None and v < lo) or (hi is not None and v >= hi):
        raise PickerError(f"{what} is out of range (got {v:g}).")
    return v


def _frame(v, what, nframes):
    f = _num(v, what)
    if f != int(f) or not 0 <= f < nframes:
        raise PickerError(f"{what} should be a frame number from 0 to {nframes - 1} (got {v:g}).")
    return int(f)


def _point(p, what, w, h, nframes):
    """[x, y] or [x, y, frame] -> the same shape, checked."""
    if not isinstance(p, (list, tuple)) or not 2 <= len(p) <= 3:
        raise PickerError(f"{what} should be [x, y] or [x, y, frame].")
    out = [_num(p[0], f"{what} x", 0, w), _num(p[1], f"{what} y", 0, h)]
    if len(p) == 3:
        out.append(_frame(p[2], f"{what} frame", nframes))
    return out


def _items(st, key):
    v = st.get(key)
    if v is None:
        return []
    if not isinstance(v, list):
        raise PickerError(f"'{key}' should be a list.")
    return v


def clean_picker(st, w, h, nframes, default_frame=0):
    """Validate the page's picker state (shape: see Job.build_route) for a w x h video with
    nframes frames. Returns a copy with ints where ints are needed and `frame` filled in on
    clicks that leave it out (= the current frame). Raises PickerError, whose message is fit
    to show the user (HTTP 400), on anything else."""
    if not isinstance(st, dict):
        raise PickerError("The picker state should be an object.")
    cur = _frame(st["frame"], "frame", nframes) if st.get("frame") is not None else default_frame
    out = {"frame": cur,
           "tol": int(_num(st.get("tol", 22), "tol", 0, 256)),
           "min_size": int(_num(st.get("min_size", 3), "min_size", 0, 1000))}
    if st.get("frames"):
        if not isinstance(st["frames"], list):
            raise PickerError("'frames' should be a list of frame numbers.")
        out["frames"] = [_frame(f, f"frames[{k}]", nframes) for k, f in enumerate(st["frames"])]

    def frame_of(c, what):
        return _frame(c["frame"], f"{what} frame", nframes) if c.get("frame") is not None else cur

    def click(c, what):
        if not isinstance(c, dict):
            raise PickerError(f"{what} should be an object with x, y and frame.")
        for k in ("x", "y"):
            if k not in c:
                raise PickerError(f"{what} is missing '{k}'.")
        return {"x": int(_num(c["x"], f"{what} x", 0, w)), "y": int(_num(c["y"], f"{what} y", 0, h)),
                "frame": frame_of(c, what)}
    out["auto"] = [click(c, f"auto[{k}]") for k, c in enumerate(_items(st, "auto"))]
    out["manual"] = [click(c, f"manual[{k}]") for k, c in enumerate(_items(st, "manual"))]

    deleted = []                                       # {x, y, frame, contour} or the old [x, y, frame]
    for k, d in enumerate(_items(st, "deleted")):
        what = f"deleted[{k}]"
        if not isinstance(d, dict):
            deleted.append(_point(d, what, w, h, nframes))
        elif d.get("contour"):
            pts = d["contour"]
            if not isinstance(pts, list) or len(pts) < 3 or not all(isinstance(q, (list, tuple)) and len(q) == 2 for q in pts):
                raise PickerError(f"{what} contour should be a list of at least 3 [x, y] points.")
            deleted.append({"x": _num(d.get("x", 0), f"{what} x"), "y": _num(d.get("y", 0), f"{what} y"),
                            "frame": frame_of(d, what),
                            "contour": [[_num(q[0], f"{what} contour x"), _num(q[1], f"{what} contour y")] for q in pts]})
        else:
            deleted.append(click(d, what))
    out["deleted"] = deleted
    off_deleted = []                                   # right-clicked off-route outlines: {x, y, frame, contour}
    for k, d in enumerate(_items(st, "off_deleted")):
        what = f"off_deleted[{k}]"
        pts = d.get("contour") if isinstance(d, dict) else None
        if not isinstance(pts, list) or len(pts) < 3 or not all(isinstance(q, (list, tuple)) and len(q) == 2 for q in pts):
            raise PickerError(f"{what} should be an object with a contour of at least 3 [x, y] points.")
        off_deleted.append({"x": _num(d.get("x", 0), f"{what} x"), "y": _num(d.get("y", 0), f"{what} y"),
                            "frame": frame_of(d, what),
                            "contour": [[_num(q[0], f"{what} contour x"), _num(q[1], f"{what} contour y")] for q in pts]})
    out["off_deleted"] = off_deleted
    out["cuts_deleted"] = [_point(p, f"cuts_deleted[{k}]", w, h, nframes) for k, p in enumerate(_items(st, "cuts_deleted"))]

    out["starts"] = [_point(p, f"starts[{k}]", w, h, nframes) for k, p in enumerate(_items(st, "starts"))]
    out["end"] = _point(st["end"], "end", w, h, nframes) if st.get("end") else None
    out["vols_deleted"] = [_point(p, f"vols_deleted[{k}]", w, h, nframes) for k, p in enumerate(_items(st, "vols_deleted"))]
    vols = []
    for k, v in enumerate(_items(st, "vols_manual")):
        what = f"vols_manual[{k}]"
        if not isinstance(v, dict):
            raise PickerError(f"{what} should be an object.")
        if v.get("kind") == "poly":
            pts = v.get("points")
            if not isinstance(pts, list) or len(pts) < 3:
                raise PickerError(f"{what} needs at least 3 corner points.")
            vols.append({"kind": "poly", "frame": frame_of(v, what),
                         "points": [_point(q, f"{what} point", w, h, nframes)[:2] for q in pts]})
        elif v.get("kind") == "click":
            vols.append(dict(click(v, what), kind="click"))
        else:
            raise PickerError(f"{what} kind should be 'click' or 'poly'.")
    out["vols_manual"] = vols
    return out


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

    off = [{"t": e["t"], "dur": e["dur"], "limb": LIMB_NAMES.get(e["limb"], e["limb"]), "hold": e["hold"],
            "on_volume": next((hd.get("volume") for hd in res.get("off_route_holds", []) if hd["id"] == e["hold"]), None)}
           for e in res.get("off_route", [])]

    used = sorted({m["hold"] for m in moves if not m["volume"]})
    return {"frames": frames, "fps": round(fps, 2), "duration_s": round(frames / fps, 2),
            "detected": res["detected"], "detected_pct": round(100 * res["detected"] / max(frames, 1)),
            "holds_total": len(holds), "holds_used": len(used), "volumes": len(vols), "has_end": res["has_end"],
            "topped_s": round(res["topped_s"], 2) if res["topped_s"] is not None else None,
            "moves": moves, "angles": angles, "notes": notes,
            "off_route": off, "off_route_enabled": bool(res.get("off_route_enabled")),
            "off_route_holds": len(res.get("off_route_holds", [])),
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


# ---------------------------------------------------------------- comparing with a second climb
COMPARE_STAGES = {"camera": "Following the camera", "pose": "Tracking the climber",
                  "draw": "Drawing the side-by-side video"}


def start_compare(job, video_b):
    """Compare the analysed climb (A) with a second video of the same route (B) in a background
    thread. The route is the one marked for A; B's wall is registered onto A's."""
    if job.state != "done":
        raise JobBusy("Analyze this climb first, then compare.")
    if (job.compare or {}).get("state") in COMPARE_BUSY:
        raise JobBusy("A comparison for this climb is already running.")
    job.compare = {"state": "queued", "stage": "", "message": "Waiting for the previous comparison", "progress": 0.0,
                   "results": None}
    threading.Thread(target=_run_compare, args=(job, Path(video_b)), daemon=True).start()


def _run_compare(job, video_b):
    st = job.compare
    try:
        with COMPARE_LOCK:
            if job.discard:
                return
            st["state"], st["message"] = "running", "Starting"

            def progress(stage, done, total):
                st["stage"], st["message"] = stage, COMPARE_STAGES.get(stage, stage)
                st["progress"] = done / total if total else 0.0

            route_file = job.dir / f"{job.name}_holds.json"
            route = json.loads(route_file.read_text()) if route_file.exists() else {"frame": job.base, "holds": []}
            out = job.dir / "compare"
            data, cam = job.pose, job.cam_used
            if cam is None:              # the analysis didn't warp (still camera, or no holds marked)
                px, vis, _ = job.pose
                if job.cam is not None:  # but the camera was followed: wall coordinates from that
                    data, cam = (px, vis, None), job.cam
                else:
                    cam = [np.eye(3)] * len(px)
            s = ccmp.compare(job.input, video_b, route, out, job.model, data_a=data, cam_a=cam,
                             log=lambda m: print(f"[{job.id} compare] {m}"), progress=progress)
            st["stage"], st["message"], st["progress"] = "encode", "Encoding the video for your browser", 1.0
            video = out / f"{job.name}_vs_{video_b.stem}_compare_h264.mp4"
            to_h264(s["outputs"]["video"], video)
            job.compare_files = {"video": video, "check": Path(s["outputs"]["registration_image"])}
            st["results"] = summarize_compare(job, s)
            st["state"], st["message"] = "done", ""
    except ccmp.RegistrationError as e:
        st["state"], st["message"] = "error", f"Couldn't match the second video's wall to this one: {e}"
    except Exception as e:
        traceback.print_exc()
        st["state"], st["message"] = "error", f"Comparison failed: {e}"
    finally:
        if st["state"] in COMPARE_BUSY:
            st["state"] = "error"
        if job.discard and not job.busy():
            job.discard = False
            job.free()


def summarize_compare(job, s):
    """climb_compare's summary -> what the comparison panel shows."""
    seqs = [{"limb": LIMB_NAMES[q["limb"]], "same": q["same"], "a": [str(h) for h in q["a"]],
             "b": [str(h) for h in q["b"]], "differences": q["differences"]} for q in s["hold_sequences"]]
    moves = [dict(m, slow=bool(abs(m["diff_s"]) >= max(0.5, 0.4 * m["a_s"]))) for m in s["moves"]]
    # the comparison video plays one alignment step per frame: where A's time t shows up in it
    fps_a = s["a"]["fps"]
    steps_a = [i for i, _ in s["path"]]

    def video_t(t):
        k = next((k for k, i in enumerate(steps_a) if i >= t * fps_a), len(steps_a) - 1)
        return round(k / fps_a, 2)
    posture = [dict(p, video_t=video_t(p["a_from"])) for p in s["posture_sections"]]
    return {"a": s["a"], "b": s["b"], "registration": {k: v for k, v in s["registration"].items() if k != "b_to_a"},
            "sequences": seqs, "moves": moves, "posture": posture,
            "video_url": f"/api/jobs/{job.id}/compare/video", "check_url": f"/api/jobs/{job.id}/compare/check"}


# ---------------------------------------------------------------- the background worker
STAGES = {"lock": "Locking each hold onto the wall", "track": "Tracking the climber",
          "draw": "Drawing the video"}


class Worker:
    """Runs one analysis at a time in a background thread, so the page stays responsive."""

    def __init__(self):
        self.jobs = {}
        self.lock = threading.Lock()                    # guards self.jobs and the start transition
        self.q = queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def new_job(self, filename):
        job = Job(filename)
        with self.lock:
            self.jobs[job.id] = job
        self.trim()
        return job

    def get(self, job_id):
        job = self.jobs.get(job_id)
        if job is not None:
            job.touched = time.monotonic()
        return job

    def remove(self, job_id):
        """The user started over: free the job (its memory, open video and files)."""
        with self.lock:
            job = self.jobs.pop(job_id, None)
        if job is not None:
            job.free()
        return job is not None

    def trim(self):
        """Keep only the MAX_KEPT_JOBS most recently used jobs that aren't busy; free the rest."""
        with self.lock:
            idle = sorted((j for j in self.jobs.values() if not j.busy() and not j.discard),
                          key=lambda j: j.touched, reverse=True)
            drop = idle[MAX_KEPT_JOBS:]
            for j in drop:
                del self.jobs[j.id]
        for j in drop:
            j.free()

    def start(self, job, picker_state, model, notes, use_lock, flag_off_route=True):
        """picking/error -> starting -> queued. The first caller flips the state under the lock,
        so a second request (Enter plus a click) gets JobBusy instead of queueing the job twice."""
        with self.lock:
            if job.state not in ("picking", "error"):
                raise JobBusy("This climb is already being analyzed.")
            before = job.state
            job.state, job.stage, job.message, job.progress = "starting", "", "Marking the route", 0.0
        try:
            if picker_state:
                job.holds, job.vols = job.route_for_analysis(picker_state)
            else:
                job.holds, job.vols, job.off_route = [], [], []
        except BaseException:
            job.state = before
            raise
        job.model, job.notes, job.use_lock = model, notes, use_lock
        job.flag_off_route = bool(flag_off_route)
        job.cache.clear()                               # the picker's frames and colour data aren't needed any more
        job.state, job.stage, job.progress = "queued", "queued", 0.0
        job.message = "Waiting for the previous climb to finish"
        self.q.put(job)

    def _loop(self):
        while True:
            job = self.q.get()
            try:
                if not job.discard:
                    self._run(job)
            except Exception as e:                      # show the error on the page, keep the server alive
                traceback.print_exc()
                job.state, job.message = "error", f"Analysis failed: {e}"
            finally:
                if job.state in ("queued", "running"):  # never leave it looking busy
                    job.state = "error"
                if job.state == "done":
                    job.release()                       # results are files now: drop the video and caches
                if job.discard:                         # the user started over while it ran
                    job.discard = False
                    with self.lock:
                        self.jobs.pop(job.id, None)
                    job.free()
                else:
                    self.trim()

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
                               progress=progress, log=lambda msg: print(f"[{job.id}] {msg}"),
                               off_route=job.off_route, flag_off_route=job.flag_off_route and bool(job.holds))
        if res["frames"] == 0:
            raise ValueError("No frames could be read from the video")
        job.stage, job.message, job.progress = "encode", "Encoding the video for your browser", 1.0
        job.output = job.dir / f"{job.name}_analysis_h264.mp4"
        to_h264(res["out_video"], job.output)
        job.csv = res["out_csv"]
        job.pose, job.cam_used = (res["px"], res["vis"], res["wall"]), cam
        job.results = summarize(job, res, job.holds, job.vols)
        job.state, job.message = "done", ""
        print(f"[{job.id}] done: {job.output}")
