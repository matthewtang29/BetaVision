"""
climb_pose.py - Step 1 of the climbing analyzer.

Runs MediaPipe pose tracking on a climbing video and produces:
  1. <name>_pose.mp4 : the video with a skeleton and joint dots drawn on it
  2. <name>_pose.csv : one row per frame with pixel positions of every tracked joint
                       plus elbow / knee / hip angles

Usage:
    pip install mediapipe opencv-python
    python climb_pose.py my_climb.mp4
    python climb_pose.py my_climb.mp4 --model heavy --show

The first run downloads the pose model (~10-30 MB) into ~/.climb_analyzer.
"""

import argparse
import csv
import math
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

# The MediaPipe landmarks a climbing coach cares about (index -> name).
JOINTS = {
    11: "l_shoulder", 12: "r_shoulder",
    13: "l_elbow",    14: "r_elbow",
    15: "l_wrist",    16: "r_wrist",
    19: "l_hand",     20: "r_hand",      # index-finger knuckle, best "hand on hold" point
    23: "l_hip",      24: "r_hip",
    25: "l_knee",     26: "r_knee",
    27: "l_ankle",    28: "r_ankle",
    29: "l_heel",     30: "r_heel",
    31: "l_toe",      32: "r_toe",
}

BONES = [
    (11, 12), (11, 23), (12, 24), (23, 24),          # torso
    (11, 13), (13, 15), (15, 19),                    # left arm
    (12, 14), (14, 16), (16, 20),                    # right arm
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),  # left leg + foot
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),  # right leg + foot
]

# Angle at the middle joint of each triple.
ANGLES = {
    "l_elbow_deg": (11, 13, 15), "r_elbow_deg": (12, 14, 16),
    "l_knee_deg":  (23, 25, 27), "r_knee_deg":  (24, 26, 28),
    "l_hip_deg":   (11, 23, 25), "r_hip_deg":   (12, 24, 26),
}

LEFT_COLOR = (255, 160, 0)    # BGR: blue-ish for left side
RIGHT_COLOR = (0, 140, 255)   # orange for right side
CENTER_COLOR = (255, 255, 255)
MIN_VIS = 0.5                 # hide joints the model isn't confident about


def ensure_model(kind: str) -> Path:
    """Download the pose model once into ~/.climb_analyzer (kept out of your video folder)."""
    cache = Path.home() / ".climb_analyzer"
    cache.mkdir(exist_ok=True)
    path = cache / f"pose_landmarker_{kind}.task"
    if not path.exists():
        old = Path(__file__).with_name(path.name)        # tidy up a copy from an earlier version
        if old.exists():
            old.replace(path)
        else:
            print(f"Downloading {kind} pose model (one time only) ...")
            urllib.request.urlretrieve(MODEL_URLS[kind], path)
    return path


def angle(a, b, c):
    """Angle ABC in degrees, from (x, y) pixel points."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def side_color(idx):
    # MediaPipe uses odd indices for the climber's left side.
    return LEFT_COLOR if idx % 2 == 1 else RIGHT_COLOR


def draw(frame, pts, vis):
    for a, b in BONES:
        if vis[a] >= MIN_VIS and vis[b] >= MIN_VIS:
            color = side_color(a) if (a % 2) == (b % 2) else CENTER_COLOR
            cv2.line(frame, pts[a], pts[b], color, 3, cv2.LINE_AA)
    for idx in JOINTS:
        if vis[idx] >= MIN_VIS:
            cv2.circle(frame, pts[idx], 6, side_color(idx), -1, cv2.LINE_AA)
            cv2.circle(frame, pts[idx], 6, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Pose-track a climbing video.")
    ap.add_argument("video", help="input video file (mp4/mov)")
    ap.add_argument("--model", choices=MODEL_URLS, default="full",
                    help="lite = fastest, heavy = most accurate (default: full)")
    ap.add_argument("--out", default=None, help="output folder (default: next to the video)")
    ap.add_argument("--show", action="store_true", help="show a live preview window (q to quit)")
    args = ap.parse_args()

    src = Path(args.video)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        sys.exit(f"Could not open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir = Path(args.out) if args.out else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_video = out_dir / f"{src.stem}_pose.mp4"
    out_csv = out_dir / f"{src.stem}_pose.csv"

    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(ensure_model(args.model))),
        running_mode=vision.RunningMode.VIDEO,   # uses previous frames to track smoothly
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    writer = None
    header = ["frame", "time_s", "detected"]
    for name in JOINTS.values():
        header += [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_vis"]
    header += ["hip_center_x", "hip_center_y", *ANGLES]

    detected_frames = 0
    with vision.PoseLandmarker.create_from_options(options) as landmarker, \
            open(out_csv, "w", newline="") as f:
        rows = csv.writer(f)
        rows.writerow(header)
        frame_i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:  # size from the first real frame (handles rotated phone video)
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            t_ms = int(frame_i * 1000 / fps)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), t_ms)

            row = [frame_i, round(t_ms / 1000, 3)]
            if result.pose_landmarks:
                detected_frames += 1
                lm = result.pose_landmarks[0]
                pts = {i: (int(p.x * w), int(p.y * h)) for i, p in enumerate(lm)}
                vis = {i: (p.visibility or 0.0) for i, p in enumerate(lm)}
                row.append(1)
                for idx in JOINTS:
                    row += [pts[idx][0], pts[idx][1], round(lm[idx].z, 4), round(vis[idx], 3)]
                hip = ((pts[23][0] + pts[24][0]) // 2, (pts[23][1] + pts[24][1]) // 2)
                row += [hip[0], hip[1]]
                row += [round(angle(pts[a], pts[b], pts[c]), 1) for a, b, c in ANGLES.values()]
                draw(frame, pts, vis)
            else:
                row.append(0)
                row += [""] * (len(header) - len(row))
            rows.writerow(row)

            cv2.putText(frame, f"{t_ms / 1000:5.1f}s", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)

            if args.show:
                scale = 900 / max(h, 1)
                cv2.imshow("climb_pose (q to quit)", cv2.resize(frame, None, fx=scale, fy=scale))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_i += 1
            if total and frame_i % 30 == 0:
                print(f"\r  {frame_i}/{total} frames", end="", flush=True)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    pct = 100 * detected_frames / max(frame_i, 1)
    print(f"\nDone: climber detected in {detected_frames}/{frame_i} frames ({pct:.0f}%)")
    print(f"  video -> {out_video}\n  data  -> {out_csv}")


if __name__ == "__main__":
    main()
