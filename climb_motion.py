"""
climb_motion.py - follow the camera so hold outlines stay on the holds when the
phone pans, tilts or zooms.

The holds are marked on ONE frame (the "reference" frame). For every other frame we
need the homography H that maps reference-frame pixels to that frame's pixels
(a homography is exact for a flat wall seen from a moving phone).

Two signals, combined:
  1. Frame-to-frame optical flow (Lucas-Kanade on corner features) -> a small H per
     step. Chained from the reference frame outward, this is smooth but slowly drifts.
  2. Every few frames, direct feature matching (ORB) of that frame against the
     reference frame -> an H with no drift. It's accepted only if it agrees with the
     chain (protects against bad matches), and then the chain restarts from it.
The climber moves differently from the wall, so RANSAC throws their points out.
Everything runs on a half-size copy for speed.
"""

import os

import cv2
import numpy as np

WORK_W = 540


def _gray(frame, s):
    small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def _step_homography(prev, cur):
    """H mapping prev -> cur from tracked corners (None if it can't tell)."""
    pts = cv2.goodFeaturesToTrack(prev, maxCorners=500, qualityLevel=0.01, minDistance=7, blockSize=7)
    if pts is None or len(pts) < 12:
        return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, nxt, None, winSize=(21, 21), maxLevel=3)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(back - pts, axis=2).ravel() < 1.0)
    if ok.sum() < 12:
        return None
    Hm, inl = cv2.findHomography(pts[ok], nxt[ok], cv2.RANSAC, 2.0)
    if Hm is None or inl.sum() < 10:
        return None
    return Hm


_ORB = None


def _features(g):
    global _ORB
    if _ORB is None:
        _ORB = cv2.ORB_create(3000, scaleFactor=1.2, nlevels=8)
    return _ORB.detectAndCompute(g, None)


class _RefMatcher:
    """Matches frames against one fixed frame (the reference, or a checkpoint)."""
    def __init__(self, ref_gray=None, feats=None):
        self.kp, self.des = feats if feats is not None else _features(ref_gray)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def match(self, g=None, feats=None):
        """H mapping the fixed frame -> this frame, or None."""
        if self.des is None or len(self.kp) < 20:
            return None
        kp, des = feats if feats is not None else _features(g)
        if des is None or len(kp) < 20:
            return None
        good = []
        for m in self.bf.knnMatch(self.des, des, k=2):
            if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
                good.append(m[0])
        if len(good) < 25:
            return None
        src = np.float32([self.kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        Hm, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if Hm is None or inl.sum() < 25 or inl.mean() < 0.25:
            return None
        return Hm


def _corner_shift(H1, H2, w, h):
    c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    return float(np.max(np.linalg.norm(cv2.perspectiveTransform(c, H1) - cv2.perspectiveTransform(c, H2), axis=2)))


def track_camera(video, ref_frame, anchor_every=8, progress=None):
    """Return a list with one 3x3 homography per frame (full-resolution pixels),
    mapping the reference frame onto that frame. Identity for a still camera."""
    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    s = WORK_W / W
    cap.set(cv2.CAP_PROP_POS_FRAMES, ref_frame)
    ok, ref = cap.read()
    if not ok:
        raise RuntimeError("could not read the reference frame")
    ref_g = _gray(ref, s)
    h, w = ref_g.shape
    matcher = _RefMatcher(ref_gray=ref_g)

    # pass over the whole video: step homographies + direct matches every few frames.
    # Checkpoint frames keep their features so later frames can match against them
    # when the camera has panned too far for the reference frame to still be in view.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    steps, direct, kfeats = [None], {}, {}
    prev = None
    i = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = _gray(f, s)
        if prev is not None:
            steps.append(_step_homography(prev, g))
        if i % anchor_every == 0 and i != ref_frame:
            kfeats[i] = _features(g)
            direct[i] = matcher.match(feats=kfeats[i])
        prev = g
        i += 1
        if progress and i % 30 == 0:
            progress(i, total)
    cap.release()
    n = i
    ref_frame = min(ref_frame, n - 1)
    I = np.eye(3)

    ref_feats = (matcher.kp, matcher.des)

    def sweep(order, step_of):
        """Walk away from the reference frame. At each checkpoint try, in order: a direct
        match to the reference, then a match to the previous good checkpoint. Accept
        it only if it agrees with the optical-flow chain."""
        out = {}
        cur, since = I, 0
        last_kf, last_kf_H = ref_frame, I
        last_feats = ref_feats
        for k in order:
            cur = step_of(k) @ cur
            since += 1
            if k in kfeats:
                tol = 0.05 * w + 0.004 * w * since      # chains drift a little, so allow more after a while
                cands = []
                if direct.get(k) is not None:
                    cands.append(direct[k])
                if last_kf != ref_frame:
                    M = _RefMatcher(feats=last_feats).match(feats=kfeats[k])
                    if M is not None:
                        cands.append(M @ last_kf_H)
                for A in cands:
                    if _corner_shift(A, cur, w, h) < tol:
                        cur, since = A, 0
                        break
                if since == 0 or last_kf == ref_frame:
                    last_kf, last_kf_H, last_feats = k, cur, kfeats[k]
            out[k] = cur
        return out

    def fwd(k):
        return steps[k] if steps[k] is not None else I

    def bwd(k):
        return np.linalg.inv(steps[k + 1]) if steps[k + 1] is not None else I

    Hs = [None] * n
    Hs[ref_frame] = I
    for k, Hm in sweep(range(ref_frame + 1, n), fwd).items():
        Hs[k] = Hm
    for k, Hm in sweep(range(ref_frame - 1, -1, -1), bwd).items():
        Hs[k] = Hm

    D, Di = np.diag([s, s, 1.0]), np.diag([1 / s, 1 / s, 1.0])
    return [Di @ (Hm / Hm[2, 2]) @ D for Hm in Hs]


def warp_points(pts, Hm):
    p = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, Hm).reshape(-1, 2)


def build_panorama(video, cam, base_frame, n_samples=24, max_side=4000):
    """Stitch the whole wall into one picture using the tracked camera motion, so a
    route that never fits in a single frame can be marked in one place. Each pixel is
    the median of all sampled frames that saw it, which also erases the climber.
    Returns (panorama, T) where T maps base-frame pixels -> panorama pixels."""
    cap = cv2.VideoCapture(str(video))
    n = len(cam)
    idx = sorted(set(np.linspace(0, n - 1, min(n_samples, n)).round().astype(int).tolist() + [base_frame]))
    frames = {}
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            frames[i] = f
    cap.release()
    h, w = next(iter(frames.values())).shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    to_base = {i: np.linalg.inv(cam[i]) for i in frames}          # frame i -> base frame
    allc = np.vstack([cv2.perspectiveTransform(corners, to_base[i]).reshape(-1, 2) for i in frames])
    x0, y0 = np.floor(allc.min(axis=0))
    x1, y1 = np.ceil(allc.max(axis=0))
    scale = min(1.0, max_side / max(x1 - x0, y1 - y0))
    T = np.array([[scale, 0, -x0 * scale], [0, scale, -y0 * scale], [0, 0, 1]])
    PW, PH = int((x1 - x0) * scale) + 1, int((y1 - y0) * scale) + 1

    pano = np.zeros((PH, PW, 3), np.uint8)
    strip = 160
    for sy in range(0, PH, strip):
        sh = min(strip, PH - sy)
        S = np.array([[1, 0, 0], [0, 1, -sy], [0, 0, 1]], float)
        vals, masks = [], []
        for i, f in frames.items():
            M = S @ T @ to_base[i]
            m = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (PW, sh), flags=cv2.INTER_NEAREST)
            if not m.any():
                continue
            vals.append(cv2.warpPerspective(f, M, (PW, sh), flags=cv2.INTER_LINEAR))
            masks.append(cv2.erode(m, np.ones((5, 5), np.uint8)) > 0)    # avoid ragged frame edges
        if not vals:
            continue
        V = np.stack(vals).astype(np.uint16)
        Mk = np.stack(masks)
        V[~Mk] = 1000                                   # unseen samples sort to the end
        V.sort(axis=0)
        cnt = Mk.sum(axis=0)
        mid = np.clip((cnt - 1) // 2, 0, None)
        med = np.take_along_axis(V, mid[None, :, :, None].repeat(3, axis=3), axis=0)[0]
        med[cnt == 0] = 0
        pano[sy:sy + sh] = np.clip(med, 0, 255).astype(np.uint8)
    return pano, T


def pick_keyframes(cam, w, h, max_frames=8, margin=0.12, cover=0.97, n_candidates=150):
    """Choose a few frames that between them show the whole wall, so the route can be
    marked frame by frame instead of on a stitched picture. Greedy set cover in base-frame
    pixels: each step takes the frame whose middle part (inset by `margin`, where holds
    aren't cut off by the edge) shows the most wall the frames so far haven't.
    Returns frame indices in video order."""
    n = len(cam)
    cands = sorted(set(np.linspace(0, n - 1, min(n, n_candidates)).round().astype(int).tolist()))
    mx, my = margin * w, margin * h
    full = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    inner = np.float32([[mx, my], [w - mx, my], [w - mx, h - my], [mx, h - my]]).reshape(-1, 1, 2)
    quads = {}
    for i in cands:
        to_base = np.linalg.inv(cam[i])
        q = cv2.perspectiveTransform(full, to_base).reshape(-1, 2)
        area = cv2.contourArea(q)
        if not cv2.isContourConvex(q.astype(np.float32)) or not 0.25 * w * h < area < 4 * w * h:
            continue                                    # tracking glitch: not a believable view of the wall
        quads[i] = cv2.perspectiveTransform(inner, to_base).reshape(-1, 2)
    if not quads:
        return [n // 2]
    allq = np.vstack(list(quads.values()))
    x0, y0 = allq.min(axis=0)
    s = 200 / max(1.0, float((allq.max(axis=0) - (x0, y0)).max()))     # coarse grid, ~200 cells across
    GW, GH = (np.ceil((allq.max(axis=0) - (x0, y0)) * s) + 1).astype(int)
    masks = {}
    for i, q in quads.items():
        m = np.zeros((GH, GW), np.uint8)
        cv2.fillConvexPoly(m, ((q - (x0, y0)) * s).round().astype(np.int32), 1)
        masks[i] = m.astype(bool)
    wall = np.logical_or.reduce(list(masks.values()))
    covered = np.zeros_like(wall)
    picked = []
    while len(picked) < max_frames and covered.sum() < cover * wall.sum():
        gain, best = max(((masks[i] & ~covered).sum(), i) for i in masks if i not in picked)
        if picked and gain < 0.02 * wall.sum():      # only a sliver more: not worth a frame
            break
        picked.append(best)
        covered |= masks[best]
    return sorted(picked)


def lock_on(video, cam, items, ref_frames=None, radius_frac=0.3, step=2, progress=None):
    """Fine-tune each hold's position per frame.

    One homography per frame is exact only for one flat surface. Real walls have panels at
    different angles and the phone is close, so near the frame edges the camera estimate can be
    off by a hold's width. For each hold, the patch of WALL around it (spray dots, bolt holes)
    in a reference frame (where its outline is known to be right) is matched to every other
    frame with SIFT features; RANSAC ignores features on the climber, and the fitted motion
    moves the hold.

    The video is streamed once, so memory does not grow with its length: only one greyscale frame
    is alive at a time (plus a small reference window per hold), and the holds of a frame are
    matched in parallel threads.

    items: outlines in BASE-frame pixels. ref_frames: per item, the frame its outline was traced
    on (None = the base frame). Returns offsets[frame][item] = (dx, dy) in base pixels.
    `progress(done, total)` is called per hold once the pass is finished."""
    from concurrent.futures import ThreadPoolExecutor
    n = len(cam)
    if n == 0 or not len(items):
        return np.zeros((n, len(items), 2), np.float32)
    cams = [np.asarray(c, float) for c in cam]
    cam_inv = [np.linalg.inv(c) for c in cams]
    base_i = min(range(n), key=lambda i: np.abs(cams[i] - np.eye(3)).sum())
    cen = [np.float32(it).mean(axis=0) for it in items]
    refs = [ref_frames[k] if ref_frames and ref_frames[k] is not None else base_i for k in range(len(items))]

    cap = cv2.VideoCapture(str(video))
    ok, f0 = cap.read()
    if not ok:
        cap.release()
        return np.zeros((0, len(items), 2), np.float32)
    H, W = f0.shape[:2]
    r = int(radius_frac * min(H, W))
    ones = np.full((H, W), 255, np.uint8)

    def window(gray, k, i):
        cx, cy = cen[k]
        M = np.array([[1, 0, -(cx - r)], [0, 1, -(cy - r)], [0, 0, 1]]) @ cam_inv[i]
        win = cv2.warpPerspective(gray, M, (2 * r, 2 * r), flags=cv2.INTER_LINEAR)
        val = cv2.warpPerspective(ones, M, (2 * r, 2 * r), flags=cv2.INTER_NEAREST)
        return win, (val > 0).astype(np.uint8) * 255

    def in_view(k, i):
        p = cv2.perspectiveTransform(np.float32([[cen[k]]]), cams[i])[0, 0]
        return 0 <= p[0] < W and 0 <= p[1] < H

    # pass 1: the reference window of every hold (decode only the reference frames)
    ref_at = {}
    for k, rf in enumerate(refs):
        ref_at.setdefault(rf, []).append(k)
    ref_feat = {}                                   # k -> (keypoint coords, descriptors)
    sift0 = cv2.SIFT_create(1500)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    last_ref = max(ref_at)
    for i in range(min(n, last_ref + 1)):
        if i not in ref_at:
            if not cap.grab():
                break
            continue
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        for k in ref_at[i]:
            rw, rv = window(g, k, i)
            kr, dr = sift0.detectAndCompute(rw, rv)
            if dr is not None and len(kr) >= 20:
                ref_feat[k] = (np.float32([p.pt for p in kr]), dr)
    active = sorted(ref_feat)
    raws = {k: np.full((n, 2), np.nan, np.float32) for k in active}
    for k in active:
        raws[k][refs[k]] = 0

    def match_one(g, k, i):
        w_, v_ = window(g, k, i)
        sift = cv2.SIFT_create(1500)
        k2, d2 = sift.detectAndCompute(w_, v_)
        if d2 is None or len(k2) < 10:
            return
        src_all, dr = ref_feat[k]
        pairs = cv2.BFMatcher().knnMatch(dr, d2, k=2)
        good = [m[0] for m in pairs if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        if len(good) < 10:
            return
        src = src_all[[m.queryIdx for m in good]]
        dst = np.float32([k2[m.trainIdx].pt for m in good])
        A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4)
        if A is None or inl.sum() < 12 or inl.mean() < 0.25:
            return
        shift = A @ np.array([r, r, 1.0]) - r
        if np.hypot(*shift) < 0.15 * min(H, W):
            raws[k][i] = shift

    # pass 2: stream the video; every `step`-th frame, match all holds in parallel
    n_read = 0
    if active:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        with ThreadPoolExecutor(max_workers=min(len(active), os.cpu_count() or 1)) as pool:
            for i in range(n):
                if i % step:
                    if not cap.grab():
                        break
                    n_read = i + 1
                    continue
                ok, f = cap.read()
                if not ok:
                    break
                n_read = i + 1
                todo = [k for k in active if i != refs[k] and in_view(k, i)]
                if todo:
                    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                    list(pool.map(lambda k: match_one(g, k, i), todo))
    cap.release()
    if not active:
        n_read = n
    n = min(n, n_read) if active else n
    out = np.zeros((n, len(items), 2), np.float32)
    for done, k in enumerate(range(len(items)), 1):
        if k in raws:
            raw = raws[k][:n]
            idx = np.where(~np.isnan(raw[:, 0]))[0]
            if len(idx) >= 2:
                for d in range(2):
                    col = np.interp(np.arange(n), idx, raw[idx, d])
                    med = np.array([np.median(col[max(0, t - 4):t + 5]) for t in range(n)])
                    out[:, k, d] = [np.mean(med[max(0, t - 2):t + 3]) for t in range(n)]
        if progress:
            progress(done, len(items))
    return out
