"""
climb_analyze.py - holds + body tracking in one go.

  1. Opens the hold picker (same window as climb_holds.py): scrub to a good
     frame, mark the route's holds (M = auto/manual, right-click = delete),
     press Enter.
  2. Tracks the climber through the whole video and draws on every frame:
        holds        magenta outlines + numbers  (turn YELLOW while a hand/foot is on them)
        start holds  cyan, labelled START     end hold  red, labelled END
        volumes      white outlines (always allowed; turn YELLOW when touched).
                     Off-route holds sitting on a volume are cut out (pink) - not allowed.
        "CONGRATULATIONS!" banner once both hands are on the END hold
     If the phone pans, tilts or zooms, the camera is tracked so every outline
     stays on its hold, and the picker shows the WHOLE wall stitched into one
     picture so the full route can be marked at once (--still skips this).
                     (hold outlines are hidden from then on)
        left side    blue skeleton
        right side   orange skeleton
  3. Saves ONE file next to the video: <name>_analysis.mp4
     (add --save-data to also save the per-frame CSV and the holds JSON/PNG)

Needs climb_holds.py and climb_pose.py in the same folder.

Usage:
    python climb_analyze.py IMG_5687.MOV
    python climb_analyze.py IMG_5687.MOV --save-data                   # also save CSV + holds files
    python climb_analyze.py IMG_5687.MOV --holds IMG_5687_holds.json   # reuse saved holds, skip the picker
    python climb_analyze.py IMG_5687.MOV --model heavy --show
"""

import sys
sys.dont_write_bytecode = True     # don't leave a __pycache__ folder next to your videos

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

import climb_holds as ch
import climb_motion as cm
import climb_notes as cn
import climb_pose as cp
import climb_volumes as cvol

HOLD_COLOR = (255, 0, 255)      # BGR magenta
TOUCH_COLOR = (0, 230, 255)     # BGR yellow
ROLE_COLOR = {"start": (255, 255, 0), "end": (0, 0, 255)}   # cyan, red
ROLE_LABEL = {"start": "START", "end": "END"}
GOLD = (0, 215, 255)
# limb -> MediaPipe landmark used as the contact point
LIMBS = {"l_hand": 19, "r_hand": 20, "l_foot": 31, "r_foot": 32}


def hold_under(point, hold_cnts, reach):
    """Id of the hold this point is on (or within `reach` pixels of), else ''."""
    best_id, best_d = "", -reach
    for hid, cnt in hold_cnts:
        d = cv2.pointPolygonTest(cnt, (float(point[0]), float(point[1])), True)  # + inside, - outside
        if d >= best_d:
            best_id, best_d = hid, d
    return best_id


def draw_volumes_layer(frame, vols, touched):
    """Volumes under the holds: white outline, yellow tint while touched, off-route cut-outs pink."""
    h, w = frame.shape[:2]
    th = max(2, min(h, w) // 300)
    if any(v["id"] in touched for v in vols):
        overlay = frame.copy()
        for v in vols:
            if v["id"] in touched:
                cv2.fillPoly(overlay, [np.array(v["contour"], np.int32)], TOUCH_COLOR)
                for e in v.get("exclude", []):                       # keep cut-outs untinted
                    cv2.fillPoly(overlay, [np.array(e, np.int32)], (0, 0, 0))
        mask = np.any(overlay != frame, axis=2)
        blend = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
        frame[mask] = blend[mask]
    for v in vols:
        on = v["id"] in touched
        cnt = np.array(v["contour"], np.int32)
        cv2.polylines(frame, [cnt], True, TOUCH_COLOR if on else ch.VOL_COLOR, th + (1 if on else 0), cv2.LINE_AA)
        for e in v.get("exclude", []):
            cv2.polylines(frame, [np.array(e, np.int32)], True, ch.CUT_COLOR, max(1, th - 1), cv2.LINE_AA)


def draw_holds_layer(frame, holds, hold_cnts, touched):
    h, w = frame.shape[:2]
    th = max(2, min(h, w) // 300)
    fs = max(0.6, min(h, w) / 1100)
    if touched:                                    # translucent yellow fill on touched holds
        overlay = frame.copy()
        for hid, cnt in hold_cnts:
            if hid in touched:
                cv2.drawContours(overlay, [cnt], -1, TOUCH_COLOR, -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, dst=frame)
    for hold, (hid, cnt) in zip(holds, hold_cnts):
        on = hid in touched
        role = hold.get("role", "")
        color = TOUCH_COLOR if on else ROLE_COLOR.get(role, HOLD_COLOR)
        cv2.drawContours(frame, [cnt], -1, color, th + (2 if on or role else 0), cv2.LINE_AA)
        label = str(hid) + (f" {ROLE_LABEL[role]}" if role else "")
        x, y = hold["bbox"][0] + hold["bbox"][2] + 4, hold["bbox"][1] + 12
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 3, cv2.LINE_AA)
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, ROLE_COLOR.get(role, (255, 255, 255)) if role else (255, 255, 255), th, cv2.LINE_AA)


def draw_banner(frame, title, subtitle, alpha=1.0):
    """Big centred banner across the lower part of the frame."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    # size the title to ~85% of the frame width
    (base_w, base_h), _ = cv2.getTextSize(title, font, 1.0, 2)
    fs = min(0.85 * w / base_w, 0.06 * h / base_h)   # as wide as fits, but never taller than ~6% of the frame
    th = max(2, int(fs * 2))
    (tw, tht), _ = cv2.getTextSize(title, font, fs, th)
    sfs = fs * 0.45
    (sw, sh), _ = cv2.getTextSize(subtitle, font, sfs, max(1, th // 2))
    pad = int(tht * 0.6)
    box_h = tht + sh + pad * 3
    top = int(h * 0.78) - box_h      # lower part of the frame: the climber is up at the END hold by now
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, top), (w, top + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6 * alpha, frame, 1 - 0.6 * alpha, 0, dst=frame)
    if alpha < 1.0:
        layer = frame.copy()
    else:
        layer = frame
    cv2.putText(layer, title, ((w - tw) // 2, top + pad + tht), font, fs, (0, 0, 0), th + 4, cv2.LINE_AA)
    cv2.putText(layer, title, ((w - tw) // 2, top + pad + tht), font, fs, GOLD, th, cv2.LINE_AA)
    cv2.putText(layer, subtitle, ((w - sw) // 2, top + pad * 2 + tht + sh), font, sfs,
                (255, 255, 255), max(1, th // 2), cv2.LINE_AA)
    if alpha < 1.0:
        cv2.addWeighted(layer, alpha, frame, 1 - alpha, 0, dst=frame)


def model_path(kind):
    """Pose model lives in ~/.climb_analyzer, not next to your videos (see climb_pose.ensure_model)."""
    return cp.ensure_model(kind)


def warp_holds(holds, Hm):
    """Hold outlines from the marked frame, moved to where the camera now sees them."""
    out = []
    for hd in holds:
        c = cm.warp_points(hd["contour"], Hm).astype(np.float32)
        d = dict(hd)
        d["contour"] = c.round().astype(int).tolist()
        d["bbox"] = list(cv2.boundingRect(c))
        if "center" in hd:
            d["center"] = cm.warp_points([hd["center"]], Hm)[0].round().astype(int).tolist()
        out.append(d)
    return out


def warp_volumes(vols, Hm):
    out = []
    for v in vols:
        d = dict(v)
        d["contour"] = cm.warp_points(v["contour"], Hm).round().astype(int).tolist()
        d["exclude"] = [cm.warp_points(e, Hm).round().astype(int).tolist() for e in v.get("exclude", [])]
        out.append(d)
    return out


def camera_moved(reader, cam):
    """How far (px) the frame corners move over the video; 0 for a still camera."""
    if cam is None:
        return 0.0
    corners = [[0, 0], [reader.cap.get(3), reader.cap.get(4)]]
    return max(float(np.abs(cm.warp_points(corners, Hm) - np.array(corners)).max()) for Hm in cam)


def route_to_base(video, out_dir, save_files, img, holds, info, to_base, base):
    """Holds + volumes marked on the picker image -> BASE-frame pixels (saved if asked)."""
    vols = info.get("volumes", [])
    if save_files:
        ch.save(video, out_dir, img, holds, info)      # picture of what you marked
    holds_b, vols_b = warp_holds(holds, to_base), warp_volumes(vols, to_base)
    if save_files:                                     # data in base-frame pixels, reusable with --holds
        js = out_dir / f"{video.stem}_holds.json"
        js.write_text(json.dumps({"video": video.name, "frame": base, "holds": holds_b, "volumes": vols_b}, indent=1))
    return holds_b, vols_b


def pick_holds(video, start_frame, out_dir, save_files, cam=None, base=0):
    """Open the picker and return holds + volumes in BASE-frame pixels.
    If the camera moved, the picker shows a stitched panorama of the whole wall so the
    entire route can be marked at once; otherwise a normal frame with a Frame slider."""
    reader = ch.FrameReader(video)
    moved = camera_moved(reader, cam)
    if moved > 25:
        print("Camera moved, so stitching the whole wall into one picture ...")
        pano, T = cm.build_panorama(video, cam, base)
        fh, fw = int(reader.cap.get(4)), int(reader.cap.get(3))
        print("Mark the holds on the whole wall, then press Enter.")
        result = ch.interactive(reader, start_frame, fixed_img=pano, frame_dims=(fh, fw),
                                fixed_label="Whole wall (stitched)")
        to_base = np.linalg.inv(T)
    else:
        print("Mark the holds, then press Enter.")
        result = ch.interactive(reader, start_frame)
    if result is None:
        sys.exit("Quit without saving.")
    img, holds, info = result
    vols = info.get("volumes", [])
    if moved <= 25:
        to_base = np.linalg.inv(cam[info["frame"]]) if cam is not None else np.eye(3)
    holds_b, vols_b = route_to_base(video, out_dir, save_files, img, holds, info, to_base, base)
    print(f"Marked {len(holds)} holds and {len(vols)} volumes.")
    return holds_b, vols_b, moved


def analyze_video(video, holds, vols, cam, model="full", out_dir=None, save_data=False, show=False,
                  use_lock=True, note_args=(), notes_file=None, progress=None, log=print):
    """Everything after the route is marked: lock each hold onto the wall, track the climber
    (pass 1), pin the sticky notes, then draw and write the annotated video (pass 2).
    Callable from other code (the web app); main() uses it for the CLI.

    holds, vols  route in BASE-frame pixels (as in *_holds.json); may be empty
    cam          per-frame homographies from climb_motion.track_camera, or None (still camera)
    show         live cv2 preview window (CLI --show only)
    progress     optional callback(stage, done, total), stage = "lock" / "track" / "draw";
                 when given, the console progress counters aren't printed
    log          where status lines go (print for the CLI)

    Returns a dict: out_video, out_csv (or None), fps, frames, detected, has_end, topped_s,
    notes, and per frame `limb_holds` ({limb: hold id / volume id / ""}) and `angles`.
    """
    video = Path(video)
    out_dir = Path(out_dir) if out_dir else video.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    # fine-tune each hold's position per frame (handheld video, angled wall panels)
    lock = None
    if cam is not None and use_lock and holds:
        log("Locking each hold onto the wall ...")
        lock = cm.lock_on(video, cam, [hd["contour"] for hd in holds],
                          [hd.get("ref_frame") for hd in holds],
                          progress=(lambda k, n: progress("lock", k, n)) if progress else
                          (lambda k, n: print(f"\r  {k}/{n} holds", end="", flush=True)))
        if not progress:
            print()
    has_end = any(hd.get("role") == "end" for hd in holds)
    if not has_end:
        log("No END hold tagged, so there will be no congratulations banner.")

    # ---- last step: pose tracking with holds drawn underneath
    log("Tracking the climber ...")
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_video = out_dir / f"{video.stem}_analysis.mp4"
    out_csv = out_dir / f"{video.stem}_analysis.csv"

    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path(model))),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)

    header = ["frame", "time_s", "detected"]
    for name in cp.JOINTS.values():
        header += [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_vis"]
    header += ["hip_center_x", "hip_center_y", *cp.ANGLES]
    header += [f"{limb}_hold" for limb in LIMBS]

    notes = cn.load_notes(note_args, notes_file)
    detected, frame_i = 0, 0
    both_on_end, topped_frame = 0, None
    hold_needed = max(3, int(round(0.2 * fps)))   # both hands on the END hold for ~0.2 s = topped
    records = []                                  # per frame: pose + which hold each limb is on
    angles = []                                   # per frame: joint angles (for the web app)

    def frame_layout(fi):
        """Where the marked holds/volumes are in frame fi (they follow the camera)."""
        if cam is not None and fi < len(cam):
            src = holds
            if lock is not None and fi < len(lock):
                src = []
                for k, hd in enumerate(holds):
                    d = dict(hd)
                    dx, dy = lock[fi, k]
                    d["contour"] = [[x + dx, y + dy] for x, y in hd["contour"]]
                    d["center"] = [hd["center"][0] + dx, hd["center"][1] + dy]
                    src.append(d)
            hf, vf = warp_holds(src, cam[fi]), warp_volumes(vols, cam[fi])
        else:
            hf, vf = holds, vols
        return hf, vf, [(hd["id"], np.array(hd["contour"], np.int32).reshape(-1, 1, 2)) for hd in hf]

    # ---- pass 1: pose + contacts (nothing drawn yet, so notes can look ahead)
    with vision.PoseLandmarker.create_from_options(options) as landmarker, \
            open(out_csv if save_data else os.devnull, "w", newline="") as f:
        rows = csv.writer(f)
        rows.writerow(header)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            reach = min(h, w) * 0.02            # how close a hand/foot must be to count as "on" a hold
            holds_f, vols_f, hold_cnts = frame_layout(frame_i)
            end_cnt = next((c for hd, (hid, c) in zip(holds_f, hold_cnts) if hd.get("role") == "end"), None)

            t_ms = int(frame_i * 1000 / fps)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), t_ms)

            row = [frame_i, round(t_ms / 1000, 3)]
            touched, on_hold = set(), {limb: "" for limb in LIMBS}
            pts = vis = None
            if result.pose_landmarks:
                detected += 1
                lm = result.pose_landmarks[0]
                pts = {i: (int(p.x * w), int(p.y * h)) for i, p in enumerate(lm)}
                vis = {i: (p.visibility or 0.0) for i, p in enumerate(lm)}
                for limb, idx in LIMBS.items():
                    if vis[idx] >= cp.MIN_VIS:
                        on_hold[limb] = hold_under(pts[idx], hold_cnts, reach)
                        if on_hold[limb] == "":                 # not on a hold: on a volume? (allowed)
                            on_hold[limb] = cvol.on_volume(vols_f, pts[idx], margin=reach / 2)
                        if on_hold[limb] != "":
                            touched.add(on_hold[limb])
                row.append(1)
                for idx in cp.JOINTS:
                    row += [pts[idx][0], pts[idx][1], round(lm[idx].z, 4), round(vis[idx], 3)]
                hip = ((pts[23][0] + pts[24][0]) // 2, (pts[23][1] + pts[24][1]) // 2)
                row += [hip[0], hip[1]]
                row += [round(cp.angle(pts[a], pts[b], pts[c]), 1) for a, b, c in cp.ANGLES.values()]
                # topped? both hands on (or right next to) the END hold
                if end_cnt is not None and topped_frame is None:
                    on_end = all(vis[i] >= cp.MIN_VIS and
                                 cv2.pointPolygonTest(end_cnt, (float(pts[i][0]), float(pts[i][1])), True) >= -1.5 * reach
                                 for i in (LIMBS["l_hand"], LIMBS["r_hand"]))
                    both_on_end = both_on_end + 1 if on_end else 0
                    if both_on_end >= hold_needed:
                        topped_frame = frame_i - hold_needed + 1
            else:
                row.append(0)
                row += [""] * (len(header) - len(LIMBS) - len(row))
            row += [on_hold[limb] for limb in LIMBS]
            rows.writerow(row)
            angles.append(dict(zip(cp.ANGLES, row[-len(LIMBS) - len(cp.ANGLES):-len(LIMBS)])) if pts is not None else {})
            base_pts = {}
            if pts is not None:
                to_base = np.linalg.inv(cam[frame_i]) if cam is not None and frame_i < len(cam) else np.eye(3)
                for limb, idx in LIMBS.items():
                    if on_hold[limb] != "":
                        base_pts[limb] = tuple(cm.warp_points([pts[idx]], to_base)[0])
            records.append({"pts": pts, "vis": vis, "on_hold": on_hold, "touched": touched, "base_pts": base_pts})
            frame_i += 1
            if progress:
                progress("track", frame_i, total)
            elif total and frame_i % 30 == 0:
                print(f"\r  {frame_i}/{total} frames", end="", flush=True)
    cap.release()
    if not progress:
        print()

    # ---- notes: decide which hold each one points at
    if notes:
        log("Sticky notes:")
        cn.resolve_anchors(notes, records, fps, {hd["id"] for hd in holds})

    # ---- pass 2: draw everything and write the video
    log("Drawing the video ...")
    cap = cv2.VideoCapture(str(video))
    writer = None
    for fi, rec in enumerate(records):
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        if writer is None:
            writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        holds_f, vols_f, hold_cnts = frame_layout(fi)
        t = fi / fps
        topped = topped_frame is not None and fi >= topped_frame
        if not topped:                                # hold outlines disappear once you've topped
            draw_volumes_layer(frame, vols_f, rec["touched"])
            draw_holds_layer(frame, holds_f, hold_cnts, rec["touched"])
        if rec["pts"] is not None:
            cp.draw(frame, rec["pts"], rec["vis"])
        hip = None
        if rec["pts"] is not None:
            hip = ((rec["pts"][23][0] + rec["pts"][24][0]) // 2, (rec["pts"][23][1] + rec["pts"][24][1]) // 2)
        for n in notes:
            a = cn.note_alpha(t, n["start"], n["end"])
            if a > 0:
                if n.get("point") is not None:                 # a spot on a volume, follows the camera
                    Hm = cam[fi] if cam is not None and fi < len(cam) else np.eye(3)
                    anchor = tuple(int(v) for v in cm.warp_points([n["point"]], Hm)[0])
                elif n["hold"] is None and n.get("limb") and rec["pts"] is not None:   # follow the limb itself
                    anchor = tuple(rec["pts"][LIMBS[n["limb"]]])
                else:
                    anchor = next((tuple(hd["center"]) for hd in holds_f if hd["id"] == n["hold"]), None)
                cn.draw_note(frame, n["text"], anchor, a, avoid=hip)
        cv2.putText(frame, f"{t:5.1f}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2, cv2.LINE_AA)
        if topped:
            fade = min(1.0, (fi - topped_frame + 1) / max(1, int(0.3 * fps)))   # quick fade-in
            draw_banner(frame, "CONGRATULATIONS!", f"Topped at {topped_frame / fps:.1f}s", fade)
        writer.write(frame)
        if show:
            s = 720 / max(h, 1)
            cv2.imshow("climb_analyze (q to stop)", cv2.resize(frame, None, fx=s, fy=s))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if progress:
            progress("draw", fi + 1, len(records))
        elif fi % 30 == 0:
            print(f"\r  {fi}/{len(records)} frames", end="", flush=True)
    cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    return {"out_video": out_video, "out_csv": out_csv if save_data else None, "fps": fps,
            "frames": frame_i, "detected": detected, "has_end": has_end,
            "topped_s": topped_frame / fps if topped_frame is not None else None,
            "limb_holds": [r["on_hold"] for r in records], "angles": angles, "notes": notes}


def main():
    ap = argparse.ArgumentParser(description="Mark a route's holds, then track the climber on it.")
    ap.add_argument("video")
    ap.add_argument("--holds", default=None, help="existing *_holds.json to reuse (skips the picker)")
    ap.add_argument("--frame", type=int, default=0, help="frame the hold picker starts on (still camera)")
    ap.add_argument("--model", choices=cp.MODEL_URLS, default="full", help="lite / full / heavy")
    ap.add_argument("--out", default=None, help="output folder (default: next to the video)")
    ap.add_argument("--show", action="store_true", help="live preview while tracking (q to stop)")
    ap.add_argument("--save-data", action="store_true", help="also save the CSV and holds JSON/PNG")
    ap.add_argument("--still", action="store_true",
                    help="camera didn't move: skip camera tracking (slightly faster)")
    ap.add_argument("--no-lock", action="store_true",
                    help="skip fine-tuning each hold's position per frame (faster)")
    ap.add_argument("--note", action="append", default=[], metavar='"20-21 footswap"',
                    help='sticky note: "start-end text [@hold]" (repeatable)')
    ap.add_argument("--notes", default=None, help="text file of sticky notes, one per line")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        here = sorted(p.name for p in Path(".").iterdir()
                      if p.suffix.lower() in (".mov", ".mp4", ".m4v", ".avi"))
        msg = f"Can't find a video called '{video}' in this folder."
        msg += ("\nVideos here: " + ", ".join(here)) if here else "\nThere are no videos in this folder."
        sys.exit(msg)
    if not cv2.VideoCapture(str(video)).isOpened():
        sys.exit(f"Found '{video}' but couldn't open it as a video.")
    out_dir = Path(args.out) if args.out else video.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- step 1: follow the camera (so outlines stay on the holds if the phone moves)
    total_frames = int(cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    data = json.loads(Path(args.holds).read_text()) if args.holds else None
    base = data.get("frame", 0) if data else (min(args.frame, total_frames - 1) if args.still else total_frames // 2)
    cam = None
    if not args.still:
        print("Following the camera ...")
        cam = cm.track_camera(video, base,
                              progress=lambda i, n: print(f"\r  {i}/{n} frames", end="", flush=True))
        print()

    # ---- step 2: holds
    if data:
        holds, vols = data["holds"], data.get("volumes", [])
        print(f"Loaded {len(holds)} holds from {args.holds}")
    else:
        holds, vols, moved = pick_holds(video, args.frame, out_dir, args.save_data, cam, base)
        if cam is not None and moved <= 5:
            cam = None                                   # camera was still: no need to warp anything
    # ---- steps 3-5: lock-on, tracking, notes, drawing
    res = analyze_video(video, holds, vols, cam, args.model, out_dir, args.save_data, args.show,
                        use_lock=not args.no_lock, note_args=args.note, notes_file=args.notes)
    detected, frame_i = res["detected"], res["frames"]
    print(f"\nDone: climber detected in {detected}/{frame_i} frames ({100 * detected / max(frame_i, 1):.0f}%)")
    if res["has_end"]:
        print(f"  Topped at {res['topped_s']:.1f}s!" if res["topped_s"] is not None
              else "  Both hands never matched on the END hold.")
    print(f"  video -> {res['out_video']}")
    if args.save_data:
        print(f"  data  -> {res['out_csv']}")

if __name__ == "__main__":
    main()
