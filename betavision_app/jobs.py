"""
jobs.py - the web app's bridge to the existing analysis code.

One Job per uploaded video:
  1. prepare        follows the camera (climb_motion.track_camera) and, if it moved,
                    stitches the whole wall into one picture, exactly as the CLI does
                    before opening the picker.
  2. build_route()  the browser's hold picker sends its clicks; they're replayed
                    through climb_holds / climb_volumes' own detection functions.
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
ROUTE_LOCK = threading.Lock()      # climb_volumes.FRAME is module-global: one route preview at a time


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
        self.cache = {}                # picture labs, auto volumes, manual outlines
        self.cam, self.moved, self.pano = None, 0.0, None

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
                    self.stage, self.message, self.progress = "stitch", "Stitching the whole wall into one picture", 0.0
                    self.pano, self.T = cm.build_panorama(self.input, self.cam, self.base)
            self.state, self.stage, self.message = "picking", "", ""
        except Exception as e:
            traceback.print_exc()
            self.state, self.message = "error", f"Couldn't follow the camera ({e}). Try again with the tripod box ticked."

    def info(self):
        pano = self.pano is not None
        return {"id": self.id, "name": self.name, "frames": self.reader.total, "fps": self.reader.fps,
                "pano": pano, "frame_w": self.width, "frame_h": self.height,
                "width": self.pano.shape[1] if pano else self.width,
                "height": self.pano.shape[0] if pano else self.height,
                "camera_moved": self.cam is not None and self.moved > 5}

    # ---------------------------------------------------------------- the picker's picture
    def frame(self, i):
        with self.lock:
            return self.reader.read(max(0, min(int(i), self.reader.total - 1)))

    def picture(self, i):
        """What the picker shows: the stitched wall, or frame i."""
        return self.pano if self.pano is not None else self.frame(i)

    def picture_key(self, i):
        return "pano" if self.pano is not None else int(i)

    def picture_jpeg(self, i):
        img = self.picture(i)
        if img is None:
            return None
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()

    def cached(self, key, fn):
        if key not in self.cache:
            if len(self.cache) > 400:
                self.cache.clear()
            self.cache[key] = fn()
        return self.cache[key]

    def lab(self, i):
        return self.cached(("lab", self.picture_key(i)), lambda: ch.to_lab(self.picture(i)))

    # ---------------------------------------------------------------- route (same steps as the picker window)
    def build_route(self, st):
        """st = the browser's picker state:
             frame, tol, min_size        current frame and the two picker sliders
             auto   [{x, y, frame}]      AUTO clicks (colour samples, taken on the frame clicked)
             manual [{x, y, frame}]      MANUAL clicks (one hold each, outlined on the frame clicked)
             deleted [[x, y]], starts [[x, y]], end [x, y] or null
             vols_manual [{kind: "click", x, y, frame} | {kind: "poly", points}], vols_deleted [[x, y]]
        Returns (img, holds, vols, info, extra) like climb_holds.interactive, plus counts for the page."""
        frame = int(st.get("frame", 0))
        img = self.picture(frame)
        h, w = img.shape[:2]
        with ROUTE_LOCK:
            cvol.FRAME = (self.height, self.width) if self.pano is not None else None
            try:
                return self._build_route(st, frame, img, h, w)
            finally:
                cvol.FRAME = None

    def _build_route(self, st, frame, img, h, w):
        r = max(3, ch._fd(h, w)[0] // 200)
        lab = self.lab(frame)
        samples = [ch.sample_lab(self.lab(c["frame"]), int(c["x"]), int(c["y"]), r) for c in st.get("auto", [])]
        tol = max(1, int(st.get("tol", 22)))
        min_area = int(ch._fd(h, w)[1] * int(st.get("min_size", 3)) / 10000)
        auto = ch.find_holds_auto(lab, samples, tol, min_area)

        manual = []
        for k, c in enumerate(st.get("manual", [])):
            x, y = int(c["x"]), int(c["y"])
            hold = self.cached(("hold", self.picture_key(c["frame"]), x, y),
                               lambda c=c, x=x, y=y: ch.outline_one_hold_img(self.picture(c["frame"]), x, y)
                               or ch.outline_one_hold(self.lab(c["frame"]), x, y))
            if hold:
                manual.append(dict(hold, manual_index=k))      # lets the page undo / right-click this click
        holds = ch.combine(auto, manual, st.get("deleted", []), st.get("starts", []), st.get("end"))

        vols_auto = self.cached(("vols", self.picture_key(frame)), lambda: cvol.detect_volumes(img))
        vols = [dict(v) for v in vols_auto if not any(cvol.inside_volume(v, p) for p in st.get("vols_deleted", []))]
        # painted volumes that aren't the route's colour are off-route (manual ones always kept)
        vols, skipped = ch.filter_coloured_volumes(lab, vols, holds)
        failed = []
        for k, v in enumerate(st.get("vols_manual", [])):
            if v.get("kind") == "poly":
                pts = [[int(x), int(y)] for x, y in v["points"]]
                vol = {"contour": pts, "faces": [pts], "source": "manual"}
            else:
                x, y = int(v["x"]), int(v["y"])
                vol = self.cached(("vol", self.picture_key(v["frame"]), x, y),
                                  lambda v=v, x=x, y=y: cvol.volume_from_click(self.picture(v["frame"]), x, y))
            if vol is None:
                failed.append(k)
            else:
                vols.append(dict(vol, manual_index=k))
        vols = cvol.cut_foreign_holds(cvol.renumber(vols), img, ch.route_mask(holds, img.shape), min_area)
        info = {"frame": frame, "colour_tol": tol, "min_area_px": min_area, "colour_samples_lab": samples,
                "clicks": [[c["x"], c["y"]] for c in st.get("auto", [])], "deleted_points": st.get("deleted", []),
                "volumes": vols}
        return img, holds, vols, info, {"skipped_volumes": len(skipped), "failed_volumes": failed}

    def preview(self, st):
        _, holds, vols, _, extra = self.build_route(st)
        return {"holds": holds, "volumes": vols, **extra}

    def route_for_analysis(self, st):
        """The marked route in BASE-frame pixels (also saved as <name>_holds.json / .png)."""
        img, holds, vols, info, _ = self.build_route(st)

        def strip(items):
            return [{k: v for k, v in d.items() if k != "manual_index"} for d in items]
        holds, info["volumes"] = strip(holds), strip(vols)
        if self.pano is not None:
            to_base = np.linalg.inv(self.T)
        else:
            to_base = np.linalg.inv(self.cam[info["frame"]]) if self.cam is not None else np.eye(3)
        return ca.route_to_base(self.input, self.dir, True, img, holds, info, to_base, self.base)

    # ---------------------------------------------------------------- status
    def status(self):
        return {"id": self.id, "state": self.state, "stage": self.stage, "message": self.message,
                "progress": round(self.progress, 3), "results": self.results,
                "info": self.info() if self.state == "picking" else None}


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
