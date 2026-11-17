"""
climb_compare.py - compare two climbs of the same route (two climbers, or two attempts).

Inputs: two videos of the same route on the same wall (filmed from anywhere, the phone may
pan) and the route marked once, on the first video (its *_holds.json from climb_analyze.py
--save-data or the web app).

  1. Follow each video's camera (climb_motion.track_camera) and track each climber's pose.
     Every joint is moved into its video's BASE frame ("wall coordinates": base-frame pixels,
     the frame the route JSON is in for the first video).
  2. Register video B's wall onto video A's: SIFT features on the wall (the climbers are
     masked out) are matched between a spread of frames of both videos, moved into each
     video's base frame and fitted with one homography. If too few wall features agree the
     comparison STOPS with an error: a wrong registration would put B's climber on the wrong
     holds and every result after that would be wrong.
  3. Per frame, features that don't depend on the climber's size: hip centre on the route
     (fraction of the route's height), each hand/foot relative to the hips (in torso lengths),
     elbow/knee/hip angles (measured in wall coordinates, so the camera angle doesn't change
     them), and which hold each limb is on.
  4. Dynamic time warping (numpy, below) aligns the two climbs, so a slow and a fast climber
     are compared at the same point of the route.
  5. Outputs, next to video A (or --out):
       <A>_vs_<B>_compare.mp4    side by side, synced by the alignment. A on the left with
                                 B's skeleton as a green ghost on A's wall; B on the right.
                                 Both show the route holds (B's through the registration).
       <A>_vs_<B>_compare.json   summary: per-limb hold sequences and where they differ,
                                 time per move, sections where posture differs most, the
                                 registration's quality, the alignment path.
       <A>_vs_<B>_registration.jpg   the route drawn on video B: check this first.
     and the summary is printed.

Usage:
    python climb_compare.py A.mov B.mov --holds A_holds.json
    python climb_compare.py A.mov B.mov --holds A_holds.json --data-a A_analysis.csv   # reuse A's pose
    python climb_compare.py A.mov B.mov --holds A_holds.json --model heavy --out results/

Pose tracking and camera tracking of each video are cached in the output folder
(<name>_track_<model>.npz, <name>_cam_<base>.npy) so a second comparison is fast.
"""

import sys
sys.dont_write_bytecode = True

import argparse
import difflib
import json
import math
from pathlib import Path

import cv2
import numpy as np

import climb_motion as cm
import climb_pose as cp
import climb_volumes as cvol

LIMBS = {"l_hand": 19, "r_hand": 20, "l_foot": 31, "r_foot": 32}      # as climb_analyze.LIMBS
LIMB_NAMES = {"l_hand": "left hand", "r_hand": "right hand", "l_foot": "left foot", "r_foot": "right foot"}
JOINT_IDS = list(cp.JOINTS)                     # 16 MediaPipe indices, CSV order
JOINT_NAMES = list(cp.JOINTS.values())
J = {idx: k for k, idx in enumerate(JOINT_IDS)}  # MediaPipe index -> column in our arrays
ANGLE_NAMES = list(cp.ANGLES)
ANGLE_LABELS = {"l_elbow_deg": "left elbow", "r_elbow_deg": "right elbow", "l_knee_deg": "left knee",
                "r_knee_deg": "right knee", "l_hip_deg": "left hip", "r_hip_deg": "right hip"}

# DTW feature weights (see feature_matrix): hip position on the route carries the alignment,
# posture refines it inside a move.
W_HIP, W_LIMB, W_ANGLE, W_HOLD = 6.0, 0.5, 1.0, 0.3

# registration: thresholds at half resolution (climb_motion works there too)
REG_SCALE_W = 540
MIN_PAIR_INLIERS = 25          # one frame pair counts as a match
MIN_TOTAL_INLIERS = 60         # all agreeing pairs together
MIN_AGREEING_PAIRS = 2


class RegistrationError(RuntimeError):
    """Video B's wall couldn't be placed on video A's reliably. The message says why."""


# ====================================================================== dynamic time warping
def cost_matrix(X, Y):
    """Euclidean distance between every row of X (n, d) and every row of Y (m, d)."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    d2 = (X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2.0 * X @ Y.T
    return np.sqrt(np.clip(d2, 0, None))


def dtw(C):
    """Dynamic time warping on a cost matrix C (n, m): steps (1,1), (1,0), (0,1), each cell's
    cost counted once. Returns (path, total) with path a list of (i, j) from (0, 0) to
    (n-1, m-1), both indices never decreasing.

    Each row is filled with numpy instead of a Python loop over cells: with P the row's
    running sum, D[i, j] = P[j] + min over k <= j of (min(D[i-1, k-1], D[i-1, k]) - P[k-1]),
    which is one np.minimum.accumulate."""
    C = np.asarray(C, float)
    n, m = C.shape
    if n == 0 or m == 0:
        raise ValueError("dtw needs two non-empty sequences")
    if not np.isfinite(C).all():
        raise ValueError("dtw cost matrix has NaN or inf")
    D = np.empty((n, m))
    D[0] = np.cumsum(C[0])
    for i in range(1, n):
        prev = D[i - 1]
        a = prev.copy()
        a[1:] = np.minimum(prev[1:], prev[:-1])            # best of up and diagonal
        P = np.cumsum(C[i])
        P_before = np.concatenate(([0.0], P[:-1]))
        D[i] = P + np.minimum.accumulate(a - P_before)
    i, j = n - 1, m - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            k = int(np.argmin((D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])))   # ties: diagonal first
            if k == 0:
                i, j = i - 1, j - 1
            elif k == 1:
                i -= 1
            else:
                j -= 1
        path.append((i, j))
    return path[::-1], float(D[-1, -1])


def path_map(path, n):
    """For each index i of the first sequence (0..n-1), the median matched index of the second."""
    js = [[] for _ in range(n)]
    for i, j in path:
        js[i].append(j)
    return np.array([float(np.median(v)) for v in js])


# ====================================================================== features
def _interp_nan(a):
    """Fill NaNs along axis 0 by linear interpolation (ends: nearest value). Columns that are
    all NaN stay NaN."""
    a = np.array(a, float)
    flat = a.reshape(len(a), -1)
    t = np.arange(len(a))
    for c in range(flat.shape[1]):
        ok = np.isfinite(flat[:, c])
        if ok.any() and not ok.all():
            flat[:, c] = np.interp(t, t[ok], flat[ok, c])
    return flat.reshape(a.shape)


def _smooth(a, k):
    """Centred moving average over k frames along axis 0 (edges use what's there)."""
    if k <= 1 or len(a) < 2:
        return a
    flat = a.reshape(len(a), -1)
    ker = np.ones(k)
    num = np.apply_along_axis(lambda c: np.convolve(c, ker, "same"), 0, flat)
    den = np.convolve(np.ones(len(a)), ker, "same")[:, None]
    return (num / den).reshape(a.shape)


def angle_deg(a, b, c):
    """Angle ABC in degrees for arrays of points (..., 2); NaN where a side has zero length."""
    v1, v2 = np.asarray(a, float) - b, np.asarray(c, float) - b
    n = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip((v1 * v2).sum(-1) / n, -1, 1)
    return np.where(n > 0, np.degrees(np.arccos(cos)), np.nan)


def route_box(holds, fallback_pts=None):
    """(x0, y0, height) of the route in wall coordinates: the box around the hold centres.
    With no holds, the box around fallback_pts (e.g. both climbers' hips)."""
    pts = np.array([hd["center"] for hd in holds if "center" in hd] or
                   [np.mean(hd["contour"], axis=0) for hd in holds if hd.get("contour")], float).reshape(-1, 2)
    if len(pts) < 2 and fallback_pts is not None:
        pts = np.asarray(fallback_pts, float).reshape(-1, 2)
        pts = pts[np.isfinite(pts).all(1)]
    if len(pts) < 2:
        return 0.0, 0.0, 1.0
    x0, y0 = pts.min(0)
    return float(x0), float(y0), float(max(pts[:, 1].max() - y0, pts[:, 0].max() - x0, 1.0))


def normalise(wall, vis, box, fps=30.0, min_vis=cp.MIN_VIS):
    """Per-frame features that don't depend on the climber's size or the camera.

    wall   (n, 16, 2) joints in wall coordinates (JOINT_IDS order), NaN where not tracked
    vis    (n, 16) visibility
    box    route_box(): hip positions are given relative to the route
    Returns dict: hip (n, 2) hip centre as (x - x0) / height, (y - y0) / height;
    limbs (n, 4, 2) each hand/foot minus the hip centre, in torso lengths; angles (n, 6)
    degrees; torso (float, wall pixels, median shoulder-centre to hip-centre distance).
    Joints the model isn't confident about (vis < min_vis) count as missing and are filled
    in from the neighbouring frames, then everything is smoothed over ~0.15 s."""
    wall = np.array(wall, float)
    vis = np.asarray(vis, float)
    wall[~(vis >= min_vis)] = np.nan
    wall = _interp_nan(wall)
    hip = (wall[:, J[23]] + wall[:, J[24]]) / 2
    sho = (wall[:, J[11]] + wall[:, J[12]]) / 2
    tl = np.linalg.norm(sho - hip, axis=1)
    torso = float(np.nanmedian(tl)) if np.isfinite(tl).any() else float("nan")
    if not torso > 0:
        torso = 1.0
    x0, y0, height = box
    limbs = np.stack([(wall[:, J[idx]] - hip) / torso for idx in LIMBS.values()], axis=1)
    angles = np.stack([angle_deg(wall[:, J[a]], wall[:, J[b]], wall[:, J[c]])
                       for a, b, c in cp.ANGLES.values()], axis=1)
    angles = _interp_nan(angles)
    k = max(1, int(round(0.15 * fps)) | 1)
    return {"hip": _smooth((hip - (x0, y0)) / height, k), "limbs": _smooth(limbs, k),
            "angles": _smooth(angles, k), "torso": torso}


def feature_matrix(nf):
    """normalise() output -> (n, d) rows for the DTW cost (weighted, NaN -> 0)."""
    F = np.concatenate([W_HIP * nf["hip"], W_LIMB * nf["limbs"].reshape(len(nf["hip"]), -1),
                        W_ANGLE * nf["angles"] / 90.0], axis=1)
    return np.nan_to_num(F)


def hold_codes(limb_holds, ids):
    """Per frame, per limb: index of the hold in `ids`, -1 when on nothing."""
    pos = {h: k for k, h in enumerate(ids)}
    return np.array([[pos.get(lh.get(limb, ""), -1) for limb in LIMBS] for lh in limb_holds], int).reshape(-1, len(LIMBS))


def alignment_cost(FA, FB, codes_a=None, codes_b=None):
    """DTW cost matrix: feature distance, plus W_HOLD per limb that is on a hold in both
    climbs but not the same one."""
    C = cost_matrix(FA, FB)
    if codes_a is not None and codes_b is not None and len(codes_a) and len(codes_b):
        for k in range(codes_a.shape[1]):
            a, b = codes_a[:, k][:, None], codes_b[:, k][None, :]
            C += W_HOLD * ((a >= 0) & (b >= 0) & (a != b))
    return C


# ====================================================================== holds per limb
def hold_under(pt, hold_cnts, reach):
    best_id, best_d = "", -reach
    for hid, cnt in hold_cnts:
        d = cv2.pointPolygonTest(cnt, (float(pt[0]), float(pt[1])), True)
        if d >= best_d:
            best_id, best_d = hid, d
    return best_id


def limb_holds(wall, vis, holds, vols, reach):
    """Per frame {limb: hold id / volume id / ""} from wall-coordinate joints (the same rule
    as climb_analyze: on or within `reach` of a hold, else on a volume)."""
    cnts = [(hd["id"], np.array(hd["contour"], np.float32).reshape(-1, 1, 2)) for hd in holds]
    out = []
    for f in range(len(wall)):
        lh = {}
        for limb, idx in LIMBS.items():
            p = wall[f, J[idx]]
            h = ""
            if vis[f, J[idx]] >= cp.MIN_VIS and np.isfinite(p).all():
                h = hold_under(p, cnts, reach)
                if h == "" and vols:
                    h = cvol.on_volume(vols, (int(p[0]), int(p[1])), margin=reach / 2)
            lh[limb] = h
        out.append(lh)
    return out


def runs(limb_holds, limb, min_run):
    """Contacts of one limb: [(hold, start, end)] (frames, end exclusive), shorter than
    min_run frames dropped, consecutive repeats of the same hold merged."""
    out, cur, start = [], "", 0
    for i, lh in enumerate(list(limb_holds) + [{}]):
        h = lh.get(limb, "")
        if h != cur:
            if cur != "" and i - start >= min_run:
                if out and out[-1][0] == cur:
                    out[-1] = (cur, out[-1][1], i)
                else:
                    out.append((cur, start, i))
            cur, start = h, i
    return out


def climb_window(limb_holds, detected, fps):
    """(start, end) frames of the climb itself: from a little before a hand first reaches a
    route hold or volume to a little after one last leaves one. Without that, the frames the
    climber was tracked. Walking up to the wall and stepping away are left out, so the
    alignment starts and ends at the same moments of the route."""
    pad = int(round(0.3 * fps))
    n = len(detected)
    min_run = max(3, int(round(0.2 * fps)))
    on = [i for limb in ("l_hand", "r_hand") for _, s, e in runs(limb_holds, limb, min_run) for i in (s, e - 1)]
    if on:
        return max(0, min(on) - pad), min(n, max(on) + 1 + pad)
    det = np.flatnonzero(np.asarray(detected, bool))
    if len(det) == 0:
        return 0, n
    return int(det[0]), int(det[-1]) + 1


# ====================================================================== tracking
def _stat_key(video):
    st = Path(video).stat()
    return np.array([st.st_size, int(st.st_mtime)], np.int64)


def track_pose(video, model="full", progress=None, cache_dir=None):
    """Pose of the climber in every frame: (fps, px (n, 16, 2) frame pixels NaN when not
    found, vis (n, 16), size (w, h)). Cached in cache_dir as <stem>_track_<model>.npz."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision
    video = Path(video)
    cache = Path(cache_dir) / f"{video.stem}_track_{model}.npz" if cache_dir else None
    if cache and cache.exists():
        d = np.load(cache)
        if (d["key"] == _stat_key(video)).all():
            return float(d["fps"]), d["px"], d["vis"], tuple(int(v) for v in d["size"])
    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(cp.ensure_model(model))),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    px, vis, size, i = [], [], (0, 0), 0
    with vision.PoseLandmarker.create_from_options(options) as lm_model:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            size = (w, h)
            res = lm_model.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
                int(i * 1000 / fps))
            if res.pose_landmarks:
                lm = res.pose_landmarks[0]
                px.append([(lm[k].x * w, lm[k].y * h) for k in JOINT_IDS])
                vis.append([lm[k].visibility or 0.0 for k in JOINT_IDS])
            else:
                px.append([(np.nan, np.nan)] * len(JOINT_IDS))
                vis.append([0.0] * len(JOINT_IDS))
            i += 1
            if progress:
                progress(i, total)
    cap.release()
    px, vis = np.array(px, float).reshape(-1, len(JOINT_IDS), 2), np.array(vis, float).reshape(-1, len(JOINT_IDS))
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, px=px, vis=vis, fps=fps, size=np.array(size), key=_stat_key(video))
    return fps, px, vis, size


def load_analysis_csv(path):
    """Pose from a climb_analyze --save-data CSV: (px, vis, wall) with wall None when the CSV
    predates the wall-coordinate columns."""
    import csv
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    has_wall = bool(rows) and f"{JOINT_NAMES[0]}_wx" in rows[0]

    def num(r, k):
        v = r.get(k, "")
        return float(v) if v not in ("", None) else np.nan
    px = np.array([[(num(r, f"{n}_x"), num(r, f"{n}_y")) for n in JOINT_NAMES] for r in rows], float)
    vis = np.nan_to_num(np.array([[num(r, f"{n}_vis") for n in JOINT_NAMES] for r in rows], float))
    wall = (np.array([[(num(r, f"{n}_wx"), num(r, f"{n}_wy")) for n in JOINT_NAMES] for r in rows], float)
            if has_wall else None)
    return px.reshape(-1, len(JOINT_NAMES), 2), vis.reshape(-1, len(JOINT_NAMES)), (
        wall.reshape(-1, len(JOINT_NAMES), 2) if wall is not None else None)


def track_camera(video, base, progress=None, cache_dir=None):
    """climb_motion.track_camera, cached as <stem>_cam_<base>.npy."""
    video = Path(video)
    cache = Path(cache_dir) / f"{video.stem}_cam_{base}.npz" if cache_dir else None
    if cache and cache.exists():
        d = np.load(cache)
        if (d["key"] == _stat_key(video)).all():
            return list(d["cam"])
    cam = cm.track_camera(video, base, progress=progress)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, cam=np.array(cam), key=_stat_key(video))
    return cam


def to_wall(px, cam):
    """Frame pixels (n, k, 2) -> base-frame pixels using each frame's homography."""
    out = np.full_like(px, np.nan)
    for i in range(len(px)):
        ok = np.isfinite(px[i]).all(1)
        if ok.any():
            Hm = np.linalg.inv(cam[i]) if cam is not None and i < len(cam) else np.eye(3)
            out[i, ok] = cm.warp_points(px[i, ok], Hm)
    return out


def warp_pts(pts, Hm):
    """warp_points that keeps NaN rows NaN and the input's shape."""
    pts = np.asarray(pts, float)
    flat = pts.reshape(-1, 2)
    out = np.full_like(flat, np.nan)
    ok = np.isfinite(flat).all(1)
    if ok.any():
        out[ok] = cm.warp_points(flat[ok], Hm)
    return out.reshape(pts.shape)


# ====================================================================== registration
def _climber_mask(shape, px, scale):
    """255 = wall, 0 = around the climber (box round the tracked joints, padded)."""
    m = np.full(shape, 255, np.uint8)
    if px is None:
        return m
    p = px[np.isfinite(px).all(1)] * scale
    if len(p) < 4:
        return m
    (x0, y0), (x1, y1) = p.min(0), p.max(0)
    pad = 0.35 * max(y1 - y0, x1 - x0, 20)
    cv2.rectangle(m, (int(x0 - pad), int(y0 - 1.3 * pad)), (int(x1 + pad), int(y1 + pad)), 0, -1)
    return m


def sample_frames(video, cam, px, n=8, window=None):
    """n frames spread over the video (or window=(start, end)) for registration:
    [(grey half-size frame, half-size -> base-frame homography, wall mask)]."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lo, hi = window if window else (0, total)
    out = []
    for i in np.linspace(lo, max(lo, hi - 1), n).round().astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            continue
        s = REG_SCALE_W / f.shape[1]
        g = cv2.cvtColor(cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
        to_base = (np.linalg.inv(cam[i]) if cam is not None and i < len(cam) else np.eye(3)) @ np.diag([1 / s, 1 / s, 1])
        out.append((g, to_base, _climber_mask(g.shape, px[i] if px is not None and i < len(px) else None, s)))
    cap.release()
    return out


def register_samples(samples_a, samples_b, frame_w=1080):
    """Homography R mapping video B's base frame onto video A's base frame, from wall
    features matched between frame samples (see sample_frames). Raises RegistrationError
    when the wall can't be matched reliably. Returns (R, stats)."""
    sift = cv2.SIFT_create(3000)
    fa = [(sift.detectAndCompute(g, m), T) for g, T, m in samples_a]
    fb = [(sift.detectAndCompute(g, m), T) for g, T, m in samples_b]
    bf = cv2.BFMatcher()
    pairs = []                                      # (inliers, src B-base pts, dst A-base pts)
    for (ka, da), Ta in fa:
        if da is None or len(ka) < 20:
            continue
        for (kb, db), Tb in fb:
            if db is None or len(kb) < 20:
                continue
            good = [p[0] for p in bf.knnMatch(db, da, k=2) if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
            if len(good) < MIN_PAIR_INLIERS:
                continue
            src = np.float32([kb[g.queryIdx].pt for g in good])
            dst = np.float32([ka[g.trainIdx].pt for g in good])
            Hm, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            if Hm is None:
                continue
            inl = inl.ravel().astype(bool)
            if inl.sum() < MIN_PAIR_INLIERS or inl.mean() < 0.2:
                continue
            pairs.append((int(inl.sum()), cm.warp_points(src[inl], Tb), cm.warp_points(dst[inl], Ta)))
    stats = {"frame_pairs_tried": len(fa) * len(fb), "frame_pairs_matched": len(pairs)}
    if not pairs:
        raise RegistrationError(
            "Couldn't match the wall between the two videos: no pair of frames shares enough wall "
            "features. Are they of the same wall? Film more of the wall around the route.")
    src = np.vstack([p[1] for p in pairs]).astype(np.float32)
    dst = np.vstack([p[2] for p in pairs]).astype(np.float32)
    tol = 0.012 * frame_w                         # base-frame pixels (~13 px on 1080p)
    R, inl = cv2.findHomography(src, dst, cv2.RANSAC, tol)
    if R is None:
        raise RegistrationError("Couldn't fit one wall-to-wall transform to the matched features.")
    err = np.linalg.norm(cm.warp_points(src, R) - dst, axis=1)
    agree = [p for p in pairs if np.median(np.linalg.norm(cm.warp_points(p[1], R) - p[2], axis=1)) < tol]
    n_in = int(sum(p[0] for p in agree))
    stats.update({"frame_pairs_agreeing": len(agree), "inliers": n_in,
                  "median_error_px": round(float(np.median(err[inl.ravel().astype(bool)])), 2),
                  "inlier_fraction": round(float(inl.mean()), 3)})
    # sanity: a believable view change (no mirror, no extreme scale)
    det = np.linalg.det(R[:2, :2] / R[2, 2])
    stats["scale"] = round(float(math.sqrt(abs(det))), 3)
    if len(agree) < MIN_AGREEING_PAIRS or n_in < MIN_TOTAL_INLIERS:
        raise RegistrationError(
            f"The wall matched in only {len(agree)} frame pair(s) with {n_in} agreeing features "
            f"(need {MIN_AGREEING_PAIRS} pairs and {MIN_TOTAL_INLIERS} features). The videos may show "
            "different walls, or too little of the same wall.")
    if det <= 0 or not 0.25 < math.sqrt(det) < 4:
        raise RegistrationError(f"The wall-to-wall transform isn't believable (scale {math.sqrt(abs(det)):.2f}).")
    return R, stats


# ====================================================================== summary
def hold_label(h):
    return f"volume {h}" if isinstance(h, str) else f"hold {h}"


def sequence_diffs(seq_a, seq_b):
    """Plain-language differences between two hold sequences of one limb."""
    out = []
    sm = difflib.SequenceMatcher(a=seq_a, b=seq_b, autojunk=False)
    for op, a0, a1, b0, b1 in sm.get_opcodes():
        A = ", ".join(hold_label(h) for h in seq_a[a0:a1])
        B = ", ".join(hold_label(h) for h in seq_b[b0:b1])
        if op == "replace":
            out.append(f"A used {A} where B used {B}")
        elif op == "delete":
            out.append(f"only A used {A}")
        elif op == "insert":
            out.append(f"only B used {B}")
    return out


def moves_of(limb_holds, fps, lo, hi):
    """A climb's moves: each time a hand or foot arrives on a hold/volume it wasn't on, within
    frames [lo, hi). [(frame, limb, hold)] in time order."""
    min_run = max(3, int(round(0.2 * fps)))
    ev = []
    for limb in LIMBS:
        prev = None
        for h, s, e in runs(limb_holds[lo:hi], limb, min_run):
            if h != prev:
                ev.append((lo + s, limb, h))
            prev = h
    return sorted(ev, key=lambda e: (e[0], e[1]))


POSTURE_MIN_DEG = 25.0    # the same climber filmed from two places differs by up to ~20 deg (pose noise)


def posture_sections(ang_diff, fps, n_max=3, min_deg=POSTURE_MIN_DEG, edge_s=0.5):
    """The stretches of A's timeline where the posture differs most.
    ang_diff (n, 6) signed B - A angle difference (degrees) per A frame. A stretch counts when
    the mean over the six angles, smoothed over 0.5 s, reaches min_deg. The first and last
    edge_s are skipped: the two climbs' windows don't start and end on exactly the same frame.
    Returns [(start, end, peak score)] in window-relative frames."""
    score = _smooth(np.abs(ang_diff).mean(1), max(1, int(fps * 0.5))).copy()
    out = []
    used = np.zeros(len(score), bool)
    edge = int(round(edge_s * fps))
    used[:edge] = True
    if edge:
        used[-edge:] = True
    for _ in range(n_max):
        s = np.where(used, -1, score)
        k = int(np.argmax(s))
        if s[k] < min_deg:
            break
        a = b = k
        while a > 0 and not used[a - 1] and score[a - 1] > 0.6 * score[k]:
            a -= 1
        while b < len(score) - 1 and not used[b + 1] and score[b + 1] > 0.6 * score[k]:
            b += 1
        used[max(0, a - int(fps)):b + int(fps) + 1] = True
        out.append((a, b + 1, float(score[k])))
    return sorted(out)


def describe_posture(d_ang, d_hip):
    """d_ang (6,) mean signed angle difference B - A, d_hip (2,) mean B - A hip offset (torso
    lengths, wall x right / y down)."""
    parts = []
    for k in np.argsort(-np.abs(d_ang))[:2]:
        if abs(d_ang[k]) >= 10:
            parts.append(f"B's {ANGLE_LABELS[ANGLE_NAMES[k]]} {abs(d_ang[k]):.0f}° "
                         f"{'straighter' if d_ang[k] > 0 else 'more bent'}")
    dx, dy = d_hip
    if abs(dy) >= 0.25:
        parts.append(f"B's hips {abs(dy):.1f} torso lengths {'lower' if dy > 0 else 'higher'}")
    if abs(dx) >= 0.25:
        parts.append(f"B's hips {abs(dx):.1f} torso lengths further {'right' if dx > 0 else 'left'}")
    return "; ".join(parts) or "small differences spread over several joints"


# ====================================================================== drawing
GHOST = (80, 255, 80)


def _draw_skeleton(img, pts, vis, s, color=None, th=3):
    """pts (16, 2) frame pixels (scaled by s); color None = climb_pose's left/right colours."""
    for a, b in cp.BONES:
        ka, kb = J[a], J[b]
        if vis[ka] >= cp.MIN_VIS and vis[kb] >= cp.MIN_VIS and np.isfinite(pts[[ka, kb]]).all():
            c = color or (cp.side_color(a) if a % 2 == b % 2 else cp.CENTER_COLOR)
            cv2.line(img, tuple((pts[ka] * s).astype(int)), tuple((pts[kb] * s).astype(int)), c, th, cv2.LINE_AA)
    for k in range(len(JOINT_IDS)):
        if vis[k] >= cp.MIN_VIS and np.isfinite(pts[k]).all():
            cv2.circle(img, tuple((pts[k] * s).astype(int)), th + 1, color or cp.side_color(JOINT_IDS[k]), -1, cv2.LINE_AA)


def _draw_holds(img, holds, Hm, s, color=(255, 0, 255)):
    for hd in holds:
        c = (cm.warp_points(hd["contour"], Hm) * s).round().astype(np.int32)
        cv2.polylines(img, [c], True, color, 2, cv2.LINE_AA)


def _label(img, text, org, scale=0.7):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


class _SeqReader:
    """Frames by index, read forward (the alignment path never goes back)."""
    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        self.pos, self.frame = -1, None

    def get(self, i):
        if i < self.pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            self.pos = i - 1
        while self.pos < i - 1:
            self.cap.grab()
            self.pos += 1
        if self.pos < i:
            ok, f = self.cap.read()
            self.pos = i
            if ok:
                self.frame = f
        return self.frame

    def close(self):
        self.cap.release()


def render(out_path, A, B, R, path, holds, panel_h=960, progress=None):
    """Side-by-side video along the alignment path (absolute frame indices)."""
    ra, rb = _SeqReader(A["video"]), _SeqReader(B["video"])
    Rinv = np.linalg.inv(R)
    writer = None
    fps = A["fps"]
    for k, (i, j) in enumerate(path):
        fa, fb = ra.get(i), rb.get(j)
        if fa is None or fb is None:
            continue
        sa, sb = panel_h / fa.shape[0], panel_h / fb.shape[0]
        pa = cv2.resize(fa, None, fx=sa, fy=sa, interpolation=cv2.INTER_AREA)
        pb = cv2.resize(fb, None, fx=sb, fy=sb, interpolation=cv2.INTER_AREA)
        Ha = A["cam"][i] if A["cam"] is not None and i < len(A["cam"]) else np.eye(3)
        Hb = B["cam"][j] if B["cam"] is not None and j < len(B["cam"]) else np.eye(3)
        _draw_holds(pa, holds, Ha, sa)
        _draw_holds(pb, holds, Hb @ Rinv, sb)
        ghost = warp_pts(B["wall_a"][j], Ha)                      # B's climber on A's wall, in A's frame
        _draw_skeleton(pa, ghost, B["vis"][j], sa, GHOST, 2)
        _draw_skeleton(pa, A["px"][i], A["vis"][i], sa)
        _draw_skeleton(pb, B["px"][j], B["vis"][j], sb)
        _label(pa, f"A {A['name']}  {i / A['fps']:5.1f}s", (10, 28))
        _label(pa, "green = B on A's wall", (10, 56), 0.55)
        _label(pb, f"B {B['name']}  {j / B['fps']:5.1f}s", (10, 28))
        frame = np.hstack([pa, pb])
        if writer is None:
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        if progress:
            progress(k + 1, len(path))
    ra.close()
    rb.close()
    if writer:
        writer.release()


def registration_image(out_path, A, B, R, holds, height=960):
    """A's base frame with the route next to B's base frame with the route moved there
    through the registration: if the outlines sit on the same holds, the registration is right."""
    imgs = []
    for V, Hm in ((A, np.eye(3)), (B, np.linalg.inv(R))):
        cap = cv2.VideoCapture(str(V["video"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, V["base"])
        ok, f = cap.read()
        cap.release()
        if not ok:
            return
        s = height / f.shape[0]
        f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        _draw_holds(f, holds, Hm, s, (255, 0, 255))
        _label(f, f"{'A' if V is A else 'B'} {V['name']} frame {V['base']}", (10, 28))
        imgs.append(f)
    cv2.imwrite(str(out_path), np.hstack(imgs))


# ====================================================================== the comparison
def prepare(video, model, out_dir, base=None, data=None, cam=None, log=print, progress=None):
    """Camera + pose of one video. data: optional (px, vis, wall) from an analysis (CSV or
    analyze_video's result) whose wall coordinates are in this base frame; cam: optional camera
    homographies to that base frame (None = track the camera here)."""
    video = Path(video)
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, f0 = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Couldn't read {video}")
    base = total // 2 if base is None else int(base)
    if cam is None:
        log(f"{video.name}: following the camera ...")
        cam = track_camera(video, base, cache_dir=out_dir,
                           progress=(lambda i, n: progress("camera", i, n)) if progress else None)
    if data is not None:
        px, vis, wall = data
        log(f"{video.name}: pose from the analysis data")
    else:
        log(f"{video.name}: tracking the climber ...")
        fps, px, vis, _ = track_pose(video, model, cache_dir=out_dir,
                                     progress=(lambda i, n: progress("pose", i, n)) if progress else None)
        wall = None
    if len(cam) < len(px):                      # a still camera passed as [] / short: identity beyond
        cam = list(cam) + [np.eye(3)] * (len(px) - len(cam))
    n = len(px)
    cam = cam[:n]
    if wall is None:
        wall = to_wall(px, cam)
    return {"video": video, "name": video.stem, "fps": fps, "base": base, "cam": cam,
            "px": px, "vis": vis, "wall": wall[:n], "size": (f0.shape[1], f0.shape[0])}


def compare(video_a, video_b, route, out_dir=None, model="full", data_a=None, render_video=True,
            log=print, progress=None, cam_a=None):
    """Compare two climbs of the route in `route` (a *_holds.json dict of video A).
    Writes the outputs (see the module docstring) and returns the summary dict.
    Raises RegistrationError when video B's wall can't be placed on A's."""
    video_a, video_b = Path(video_a), Path(video_b)
    out_dir = Path(out_dir) if out_dir else video_a.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    holds, vols = route.get("holds", []), route.get("volumes", [])
    A = prepare(video_a, model, out_dir, base=route.get("frame", None), data=data_a, cam=cam_a, log=log, progress=progress)
    B = prepare(video_b, model, out_dir, log=log, progress=progress)

    log("Matching B's wall to A's ...")
    reach = 0.02 * min(A["size"])
    lh_a = limb_holds(A["wall"], A["vis"], holds, vols, reach)
    win_a = climb_window(lh_a, A["vis"].max(1) > 0, A["fps"])
    R, reg = register_samples(sample_frames(A["video"], A["cam"], A["px"], window=win_a),
                              sample_frames(B["video"], B["cam"], B["px"]), frame_w=A["size"][0])
    log(f"  registered: {reg['inliers']} wall features in {reg['frame_pairs_agreeing']} frame pairs, "
        f"median error {reg['median_error_px']} px")
    B["wall_a"] = warp_pts(B["wall"], R)
    lh_b = limb_holds(B["wall_a"], B["vis"], holds, vols, reach)
    win_b = climb_window(lh_b, B["vis"].max(1) > 0, B["fps"])

    # ---- features + alignment over the climbs themselves
    (a0, a1), (b0, b1) = win_a, win_b
    hips = np.vstack([(A["wall"][a0:a1, J[23]] + A["wall"][a0:a1, J[24]]) / 2,
                      (B["wall_a"][b0:b1, J[23]] + B["wall_a"][b0:b1, J[24]]) / 2])
    box = route_box(holds, hips)
    na = normalise(A["wall"][a0:a1], A["vis"][a0:a1], box, A["fps"])
    nb = normalise(B["wall_a"][b0:b1], B["vis"][b0:b1], box, B["fps"])
    ids = sorted({lh[l] for lh in lh_a + lh_b for l in LIMBS if lh[l] != ""}, key=str)
    C = alignment_cost(feature_matrix(na), feature_matrix(nb),
                       hold_codes(lh_a[a0:a1], ids), hold_codes(lh_b[b0:b1], ids))
    log("Aligning the two climbs ...")
    path, total = dtw(C)

    # ---- summary
    fa, fb = A["fps"], B["fps"]
    min_run_a, min_run_b = max(3, int(round(0.2 * fa))), max(3, int(round(0.2 * fb)))
    seqs = []
    for limb in LIMBS:
        sa = [h for h, _, _ in runs(lh_a[a0:a1], limb, min_run_a)]
        sb = [h for h, _, _ in runs(lh_b[b0:b1], limb, min_run_b)]
        sa = [h for k, h in enumerate(sa) if k == 0 or h != sa[k - 1]]
        sb = [h for k, h in enumerate(sb) if k == 0 or h != sb[k - 1]]
        seqs.append({"limb": limb, "a": sa, "b": sb, "same": sa == sb, "differences": sequence_diffs(sa, sb)})

    jmap = path_map(path, a1 - a0)                       # A frame (window) -> B frame (window)
    moves = []
    ev = moves_of(lh_a, fa, a0, a1)
    bounds = [a0] + [e[0] for e in ev] + [a1]
    labels = ["start"] + [f"{LIMB_NAMES[l]} to {hold_label(h)}" for _, l, h in ev]
    for k in range(len(bounds) - 1):
        s, e = bounds[k], bounds[k + 1]
        if e - s < 1:
            continue
        bs, be = jmap[s - a0], jmap[e - 1 - a0] + 1
        ta, tb = (e - s) / fa, (be - bs) / fb
        moves.append({"move": labels[k], "a_t": round(s / fa, 2), "a_s": round(ta, 2),
                      "b_t": round((b0 + bs) / fb, 2), "b_s": round(tb, 2), "diff_s": round(tb - ta, 2)})

    # posture per A frame against the B frames matched to it
    ia = np.array([i for i, _ in path])
    jb = np.array([j for _, j in path])
    d_ang = np.zeros((a1 - a0, len(ANGLE_NAMES)))
    d_hip = np.zeros((a1 - a0, 2))
    torso = na["torso"]
    for i in range(a1 - a0):
        js = jb[ia == i]
        d_ang[i] = (nb["angles"][js] - na["angles"][i]).mean(0)
        d_hip[i] = (nb["hip"][js] - na["hip"][i]).mean(0) * box[2] / torso
    # A frames that only match B's first or last frame: one video started late or ended early,
    # so there is nothing in B to compare them with
    stuck = np.zeros(a1 - a0, bool)
    for i in range(a1 - a0):
        js = jb[ia == i]
        stuck[i] = (js.max() == 0 and i > 0) or (js.min() == b1 - b0 - 1 and i < a1 - a0 - 1)
    d_ang[stuck] = 0
    sections = []
    for s, e, peak in posture_sections(d_ang, fa):
        bs, be = jmap[s], jmap[e - 1] + 1
        sections.append({"a_from": round((a0 + s) / fa, 2), "a_to": round((a0 + e) / fa, 2),
                         "b_from": round((b0 + bs) / fb, 2), "b_to": round((b0 + be) / fb, 2),
                         "mean_angle_diff_deg": round(peak, 1),
                         "angle_diff_deg": {ANGLE_NAMES[k]: round(float(v), 1) for k, v in enumerate(d_ang[s:e].mean(0))},
                         "summary": describe_posture(d_ang[s:e].mean(0), d_hip[s:e].mean(0))})

    stem = f"{A['name']}_vs_{B['name']}"
    summary = {
        "a": {"video": video_a.name, "climb_from_s": round(a0 / fa, 2), "climb_to_s": round(a1 / fa, 2),
              "climb_s": round((a1 - a0) / fa, 2), "fps": round(fa, 3), "base_frame": A["base"]},
        "b": {"video": video_b.name, "climb_from_s": round(b0 / fb, 2), "climb_to_s": round(b1 / fb, 2),
              "climb_s": round((b1 - b0) / fb, 2), "fps": round(fb, 3), "base_frame": B["base"]},
        "registration": {**reg, "b_to_a": np.round(R / R[2, 2], 6).tolist()},
        "alignment": {"mean_cost": round(total / len(path), 3), "steps": len(path)},
        "hold_sequences": seqs, "moves": moves, "posture_sections": sections,
        "path": [[a0 + i, b0 + j] for i, j in path],
        "outputs": {"json": str(out_dir / f"{stem}_compare.json"),
                    "registration_image": str(out_dir / f"{stem}_registration.jpg")},
    }
    registration_image(out_dir / f"{stem}_registration.jpg", A, B, R, holds)
    if render_video:
        log("Drawing the side-by-side video ...")
        out_video = out_dir / f"{stem}_compare.mp4"
        render(out_video, A, B, R, [(a0 + i, b0 + j) for i, j in path], holds,
               progress=(lambda k, n: progress("draw", k, n)) if progress else None)
        summary["outputs"]["video"] = str(out_video)
    summary = plain(summary)
    (out_dir / f"{stem}_compare.json").write_text(json.dumps(summary, indent=1))
    return summary


def plain(o):
    """Numpy scalars/arrays inside dicts and lists -> plain Python (for JSON)."""
    if isinstance(o, dict):
        return {k: plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return plain(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def report(s):
    """The summary as text lines."""
    a, b = s["a"], s["b"]
    out = [f"A {a['video']}: climb {a['climb_from_s']}-{a['climb_to_s']} s ({a['climb_s']} s)",
           f"B {b['video']}: climb {b['climb_from_s']}-{b['climb_to_s']} s ({b['climb_s']} s)",
           f"Registration: {s['registration']['inliers']} wall features agree over "
           f"{s['registration']['frame_pairs_agreeing']} frame pairs (median error "
           f"{s['registration']['median_error_px']} px). Check the registration image.", "",
           "Hold sequences:"]
    for q in s["hold_sequences"]:
        name = LIMB_NAMES[q["limb"]]
        if q["same"]:
            out.append(f"  {name}: same ({len(q['a'])} holds)")
        else:
            out.append(f"  {name}: " + "; ".join(q["differences"]))
    out += ["", "Time per move (A's moves; B over the same stretch):"]
    for m in s["moves"]:
        flag = "  <-" if abs(m["diff_s"]) >= max(0.5, 0.4 * m["a_s"]) else ""
        out.append(f"  {m['a_t']:6.1f}s  {m['move']:<28} A {m['a_s']:5.1f}s  B {m['b_s']:5.1f}s"
                   f"  ({m['diff_s']:+.1f}s){flag}")
    out += ["", "Where the posture differs most:"]
    for p in s["posture_sections"] or []:
        out.append(f"  A {p['a_from']}-{p['a_to']} s / B {p['b_from']}-{p['b_to']} s "
                   f"(mean {p['mean_angle_diff_deg']}°): {p['summary']}")
    if not s["posture_sections"]:
        out.append(f"  nowhere by more than {POSTURE_MIN_DEG:.0f}° on average")
    return out


def main():
    ap = argparse.ArgumentParser(description="Compare two climbs of the same route.")
    ap.add_argument("video_a", help="the video the route was marked on")
    ap.add_argument("video_b", help="another climb of the same route (any camera position)")
    ap.add_argument("--holds", required=True, help="video A's *_holds.json")
    ap.add_argument("--data-a", default=None,
                    help="video A's *_analysis.csv (climb_analyze --save-data) to reuse its pose")
    ap.add_argument("--model", choices=cp.MODEL_URLS, default="full", help="lite / full / heavy")
    ap.add_argument("--out", default=None, help="output folder (default: next to video A)")
    ap.add_argument("--no-video", action="store_true", help="skip the side-by-side video")
    args = ap.parse_args()
    for v in (args.video_a, args.video_b, args.holds):
        if not Path(v).exists():
            sys.exit(f"Can't find {v}")
    route = json.loads(Path(args.holds).read_text())
    if not route.get("holds"):
        sys.exit(f"{args.holds} has no holds. The comparison needs the marked route: it finds where each "
                 "climb starts and ends, and which hold each hand and foot is on, from it.")
    data_a = None
    if args.data_a:
        px, vis, wall = load_analysis_csv(args.data_a)
        if wall is None:
            sys.exit(f"{args.data_a} has no wall-coordinate columns (made by an older version). "
                     "Leave out --data-a and the pose is tracked again.")
        data_a = (px, vis, wall)
    try:
        s = compare(args.video_a, args.video_b, route, args.out, args.model, data_a, not args.no_video)
    except RegistrationError as e:
        sys.exit(f"\nCan't compare: video B's wall couldn't be matched to video A's.\n  {e}\n"
                 "Nothing was written.")
    print()
    print("\n".join(report(s)))
    print(f"\n  summary -> {s['outputs']['json']}")
    print(f"  check   -> {s['outputs']['registration_image']}")
    if "video" in s["outputs"]:
        print(f"  video   -> {s['outputs']['video']}")


if __name__ == "__main__":
    main()
