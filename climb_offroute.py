"""
climb_offroute.py - holds that are NOT on the route, and touches of them worth flagging.

  1. Detecting off-route holds on the plain wall (find_off_route): every colourful hold-like
     blob in the picker's picture that isn't one of the route's holds. It reuses the colour
     picker in climb_holds.py: a colour model is made for each hue that shows up on the
     wall (wall_models), and climb_holds.find_holds_auto outlines every blob that matches
     one, with its own size / shape / fragment / climber-mask rules. Blobs on a route hold,
     of the route's own colour, or on a volume are dropped (holds on volumes are already cut
     out of the volume by climb_volumes.cut_foreign_holds, and count as off-route too).
     White, grey and black off-route holds are NOT found (too easily confused with wall).
  2. Flagging touches (touch_events): a hand or foot counts as a POSSIBLE off-route touch
     only when it stays inside one off-route hold for a sustained time AND stays roughly
     still in wall coordinates over that time, so a limb swinging past in front of a hold
     isn't flagged. Pose landmarks are 2D and noisy, so the rules prefer missing a marginal
     touch over a false alarm.
  3. Drawing (draw_flags): the touched hold outlined in a warning colour with a small label.

Everything here is in the pixels of whatever picture it's given; climb_analyze keeps the
outlines in BASE-frame pixels (like the route's holds) so they follow the camera.
"""

import cv2
import numpy as np

import climb_holds as ch

OFF_COLOR = (0, 69, 255)          # BGR orange-red: possible off-route touch (video and picker)
LABEL = "possible off-route"

# ---- detection
WALL_CHROMA = 15                  # Lab chroma above this = a pixel with a colour (a candidate hold pixel)
HUE_BIN = 10                      # degrees per bin of the hue histogram
MIN_HUE_SHARE = 0.0004            # a hue needs at least this share of the picture's pixels to get a model
MAX_MODELS = 10
ROUTE_HUE_MARGIN = 15             # degrees: a blob this close to a route colour's hue is the route's colour
ROUTE_OVERLAP = 0.2               # a blob this much covered by a (slightly grown) route hold is that hold

# ---- touches
MIN_TOUCH_S = 0.5                 # steadiness is judged over windows this long (moves list: 0.2 s) ...
STEADY_WINDOWS = 0.25             # ... and a touch needs (this x a window's frames) + 1 steady windows at
                                  # one place, so the shortest flag is about 0.6 s (min_touch_frames)
MAX_GAP_S = 0.15                  # a limb lost / off the hold for up to this long doesn't end the touch
MAX_SPREAD = 0.03                 # x the frame's short side: how far the limb may wander while "still"


# ---------------------------------------------------------------- detection
def wall_models(fc, max_models=MAX_MODELS):
    """climb_holds colour models (no clicked hold) for each hue that is common enough on the
    wall to be a set of holds. Same fields as climb_holds.hold_model's colour models, so
    find_holds_auto treats them exactly like clicked ones."""
    step = max(1, int(np.sqrt(fc.h * fc.w / 250000)))              # subsample: ~250k pixels
    C, H = fc.C[::step, ::step], fc.H[::step, ::step]
    bg = fc.person_wide()[::step, ::step] == 0
    C, H = C[bg], H[bg]
    if not len(C):
        return []
    col = C >= WALL_CHROMA
    nb = 360 // HUE_BIN
    hist = np.bincount(((H[col] + 180) // HUE_BIN).astype(int) % nb, minlength=nb).astype(float)
    smooth = hist + 0.5 * (np.roll(hist, 1) + np.roll(hist, -1))
    need = MIN_HUE_SHARE * len(C)
    peaks = [i for i in range(nb) if smooth[i] >= need and smooth[i] >= smooth[i - 1] and smooth[i] > smooth[(i + 1) % nb]]
    peaks.sort(key=lambda i: -smooth[i])
    models = []
    for i in peaks[:max_models]:
        h0 = -180 + (i + 0.5) * HUE_BIN
        band = col & (ch._hue_diff(H, h0) <= 1.5 * HUE_BIN)
        if band.sum() < 20:
            continue
        hue = ch._circ_median(H[band])
        band = col & (ch._hue_diff(H, hue) <= 1.5 * HUE_BIN)
        if band.sum() < 20:
            continue
        dev = np.abs((H[band] - hue + 180) % 360 - 180)
        hue_tol = float(np.clip(1.5 * np.percentile(dev, 85) + 4, 8, 15))
        c_hold = float(np.percentile(C[band], 60))
        # the picture's background level of this colour (as in hold_model): a hue that covers
        # much of the picture (mats, a painted wall) needs a lot more colour to count
        along = C * np.cos(np.radians(np.minimum(ch._hue_diff(H, hue), 90)))
        c_wall = max(4.0, float(np.percentile(np.maximum(along, 0), 85)))
        if c_hold < c_wall + 2 * ch.PALE_MARGIN:
            continue
        if any(ch._hue_diff(np.array([hue]), m["hue"])[0] < 0.5 * HUE_BIN for m in models):
            continue
        models.append({"kind": "colour", "hue": round(hue, 1), "hue_tol": round(hue_tol, 1),
                       "c_hold": round(c_hold, 1), "c_wall": round(c_wall, 1), "frame": None,
                       "lab": [0.0, round(128 + c_hold * np.cos(np.radians(hue)), 1),
                               round(128 + c_hold * np.sin(np.radians(hue)), 1)]})
    return models


def detect(fc, min_area=None):
    """Every colourful hold-like blob in the picture (route or not): climb_holds hold dicts
    plus "hue" (degrees). Slow-ish (about a second at 1080p); cache it per frame."""
    models = wall_models(fc)
    if not models:
        return []
    out = []
    for hd in ch.find_holds_auto(fc, models, 22, min_area):
        m = np.zeros((fc.h, fc.w), np.uint8)
        cv2.fillPoly(m, [np.array(hd["contour"], np.int32)], 1)
        sel = (m > 0) & (fc.C >= ch.COLOURFUL)
        if sel.sum() < 10:
            continue
        hd = dict(hd, source="off", hue=round(ch._circ_median(fc.H[sel]), 1))
        out.append(hd)
    return out


def _center(c):
    return np.mean(np.asarray(c, float), axis=0)


def _touches(a, b):
    """Two outlines are the same object: either's centre is inside the other, or (crescent /
    two-lobed outlines, whose centre lies outside themselves) their convex hulls mostly overlap."""
    ca, cb = np.asarray(a, np.int32), np.asarray(b, np.int32)
    pa, pb = _center(a), _center(b)
    if (cv2.pointPolygonTest(ca, (float(pb[0]), float(pb[1])), False) >= 0 or
            cv2.pointPolygonTest(cb, (float(pa[0]), float(pa[1])), False) >= 0):
        return True
    ha, hb = cv2.convexHull(ca.astype(np.float32)), cv2.convexHull(cb.astype(np.float32))
    small = min(cv2.contourArea(ha), cv2.contourArea(hb))
    return small > 0 and cv2.intersectConvexConvex(ha, hb)[0] > 0.5 * small


def select(cands, route_holds, route_models=(), deleted=(), shape=None):
    """The candidates that are off-route: not (on) a route hold, not the route's colour and not
    deleted by the user (deleted = outlines). All in the same picture's pixels."""
    hues = [(m["hue"], m.get("hue_tol", 12)) for m in route_models
            if isinstance(m, dict) and m.get("kind") == "colour" and not m.get("reject")]
    rm = None
    if route_holds and shape is not None:
        rm = ch.route_mask(route_holds, shape)
        rm = cv2.dilate(rm, ch._box(max(2, int(ch._fd(*shape[:2])[0] * 0.006))))
    out = []
    for hd in cands:
        if any(ch._hue_diff(np.array([hd["hue"]]), h)[0] <= max(ROUTE_HUE_MARGIN, 1.5 * t) for h, t in hues):
            continue
        if any(_touches(hd["contour"], r["contour"]) for r in route_holds):
            continue
        if rm is not None:
            m = np.zeros(rm.shape, np.uint8)
            cv2.fillPoly(m, [np.array(hd["contour"], np.int32)], 1)
            if (m & rm).sum() > ROUTE_OVERLAP * max(m.sum(), 1):
                continue
        if any(_touches(hd["contour"], d) for d in deleted):
            continue
        out.append(hd)
    return out


def drop_on_volumes(off, vols):
    """Holds on a volume are handled by the volume's own cut-outs (v["exclude"])."""
    def on(hd):
        c = _center(hd["contour"])
        return any(cv2.pointPolygonTest(np.array(v["contour"], np.int32), (float(c[0]), float(c[1])), False) >= 0
                   for v in vols)
    return [hd for hd in off if not on(hd)]


def number(off):
    """Ids X1, X2 ... bottom to top (like the route's holds); only the fields worth saving."""
    off = sorted(off, key=lambda d: -_center(d["contour"])[1])
    out = []
    for i, hd in enumerate(off, 1):
        c = np.asarray(hd["contour"], np.float32)
        out.append({"id": f"X{i}", "contour": np.asarray(hd["contour"]).round().astype(int).tolist(),
                    "center": _center(hd["contour"]).round().astype(int).tolist(),
                    "bbox": list(cv2.boundingRect(c)), "hue": hd.get("hue")})
    return out


def find_off_route(fc, route_holds, vols=(), route_models=(), deleted=(), min_area=None, cands=None):
    """Off-route holds on the plain wall in fc's picture (ids X1 ...). cands: detect(fc) if
    already computed (it doesn't depend on the route, so callers cache it per frame)."""
    cands = detect(fc, min_area) if cands is None else cands
    off = select(cands, route_holds, route_models, deleted, (fc.h, fc.w))
    return number(drop_on_volumes(off, vols))


def drop_cuts(vols, points):
    """Remove the volume cut-outs (off-route holds on a volume) that contain one of the
    points the user right-clicked: that spot then counts as the volume again."""
    if not points:
        return vols
    out = []
    for v in vols:
        ex = [e for e in v.get("exclude", [])
              if not any(cv2.pointPolygonTest(np.array(e, np.int32), (float(p[0]), float(p[1])), False) >= 0
                         for p in points)]
        out.append(dict(v, exclude=ex))
    return out


def all_off_route(off, vols):
    """Wall off-route holds + the volumes' cut-outs, as one list with unique ids (the cut-outs
    are numbered after the wall ones, X<n+1> ...). Same pixels as the inputs."""
    out = [dict(hd) for hd in off]
    used = {hd["id"] for hd in out}
    k = len(out)
    for v in vols:
        for e in v.get("exclude", []):
            k += 1
            while f"X{k}" in used:
                k += 1
            used.add(f"X{k}")
            out.append({"id": f"X{k}", "contour": [list(map(int, p)) for p in e],
                        "center": _center(e).round().astype(int).tolist(), "volume": v.get("id")})
    return out


def hold_at(off_cnts, pt):
    """Id of the off-route hold whose outline contains pt (strictly: no reach slack), else ''."""
    p = (float(pt[0]), float(pt[1]))
    for hid, cnt in off_cnts:
        if cv2.pointPolygonTest(cnt, p, False) >= 0:
            return hid
    return ""


# ---------------------------------------------------------------- touches
def _spread(pts):
    """How far (px) the points stray from their median: 90th percentile, so one bad frame
    of pose jitter doesn't count."""
    p = np.asarray(pts, float)
    d = np.hypot(*(p - np.median(p, axis=0)).T)
    return float(np.percentile(d, 90))


def _min_windows(win):
    return max(2, int(STEADY_WINDOWS * win) + 1)


def min_touch_frames(fps, min_s=MIN_TOUCH_S):
    """Shortest perfectly still touch that is flagged, in frames."""
    win = max(2, int(round(min_s * fps)))
    return win + _min_windows(win) - 1


def touch_events(tracks, fps, frame_min, min_s=MIN_TOUCH_S, max_gap_s=MAX_GAP_S, max_spread=MAX_SPREAD):
    """Possible off-route touches.
    tracks     {limb: [per frame: (off-route hold id or "", (x, y) in wall coordinates or None)]}
               - "" whenever the limb isn't inside an off-route hold (on a route hold, a
               volume, bare wall, or not visible)
    fps        frames per second;  frame_min: the video frame's short side in px
    Returns [{"limb", "hold", "start", "end"}] (frames, end exclusive), sorted by start.

    A touch = the limb stays inside ONE off-route hold for at least min_s, and over some
    stretch of at least min_s its position stays within max_spread x frame_min of its median
    (90th percentile). Gaps of up to max_gap_s (limb lost, or a frame just off the hold) are
    bridged; another off-route hold ends the run."""
    win = max(2, int(round(min_s * fps)))
    gap = max(0, int(round(max_gap_s * fps)))
    thr = max_spread * frame_min
    events = []
    for limb, seq in tracks.items():
        runs, cur, last = [], None, -10 ** 9
        for i, (hid, pt) in enumerate(seq):
            if hid and pt is not None:
                if cur is not None and (hid != cur[0] or i - last > gap + 1):
                    runs.append(cur)
                    cur = None
                if cur is None:
                    cur = (hid, [])
                cur[1].append((i, pt))
                last = i
            elif cur is not None and i - last > gap:
                runs.append(cur)
                cur = None
        if cur is not None:
            runs.append(cur)
        for hid, entries in runs:
            if entries[-1][0] - entries[0][0] + 1 < win:
                continue
            frames = np.array([e[0] for e in entries])
            pts = [e[1] for e in entries]
            # steady windows: win frames whose points stay within thr of their median
            steady = []
            for s in range(len(entries)):
                e = int(np.searchsorted(frames, frames[s] + win))  # entries within win frames from s
                if frames[-1] < frames[s] + win - 1 or e - s < max(2, 0.6 * win):
                    continue                                       # runs past the end / too little seen
                if _spread(pts[s:e]) <= thr:
                    steady.append((s, e, np.median(np.asarray(pts[s:e], float), axis=0)))
            # chain overlapping steady windows that stay at ONE place (median within thr/4 of the
            # first): a limb resting there. A slow slide moves the median along, so its chains
            # stay short; it needs many steady windows in one place to count.
            groups, g = [], None
            for s, e, med in steady:
                if g is not None and s < g["end"] and np.hypot(*(med - g["anchor"])) <= thr / 4:
                    g["end"], g["n"] = max(g["end"], e), g["n"] + 1
                else:
                    g = {"start": s, "end": e, "anchor": med, "n": 1}
                    groups.append(g)
            for g in groups:
                a, b = frames[g["start"]], frames[g["end"] - 1]
                if g["n"] >= _min_windows(win) and b - a + 1 >= win:
                    events.append({"limb": limb, "hold": hid, "start": int(a), "end": int(b) + 1})
    # one touch split in two (the limb shifted on the hold for a moment): join them again
    events.sort(key=lambda e: (e["limb"], e["hold"], e["start"]))
    merged = []
    for e in events:
        p = merged[-1] if merged else None
        if p and p["limb"] == e["limb"] and p["hold"] == e["hold"] and e["start"] - p["end"] <= win:
            p["end"] = max(p["end"], e["end"])
        else:
            merged.append(dict(e))
    merged.sort(key=lambda e: (e["start"], e["limb"]))
    return merged


def flags_per_frame(events, n_frames):
    """[per frame: {hold id: [limbs]}] for drawing and the CSV."""
    out = [{} for _ in range(n_frames)]
    for e in events:
        for i in range(e["start"], min(e["end"], n_frames)):
            out[i].setdefault(e["hold"], []).append(e["limb"])
    return out


# ---------------------------------------------------------------- drawing
def draw_flags(frame, items):
    """items = [(contour in this frame's pixels, label)]: warning outline + small label."""
    h, w = frame.shape[:2]
    th = max(2, min(h, w) // 300)
    fs = max(0.5, min(h, w) / 1600)
    for cnt, label in items:
        c = np.array(cnt, np.int32).reshape(-1, 1, 2)
        overlay = frame.copy()
        cv2.drawContours(overlay, [c], -1, OFF_COLOR, -1)
        x, y, bw, bh = cv2.boundingRect(c)
        x0, y0, x1, y1 = max(0, x - 2), max(0, y - 2), min(w, x + bw + 2), min(h, y + bh + 2)
        if x1 > x0 and y1 > y0:                     # tint just around the hold (cheap)
            frame[y0:y1, x0:x1] = cv2.addWeighted(overlay[y0:y1, x0:x1], 0.3, frame[y0:y1, x0:x1], 0.7, 0)
        cv2.drawContours(frame, [c], -1, (0, 0, 0), th + 4, cv2.LINE_AA)
        cv2.drawContours(frame, [c], -1, OFF_COLOR, th + 1, cv2.LINE_AA)
        (tw, tht), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, th - 1))
        tx = int(min(max(0, x + bw // 2 - tw // 2), w - tw - 4))
        ty = int(y - 8 if y - 8 - tht - 4 > 0 else y + bh + tht + 10)
        cv2.rectangle(frame, (tx - 4, ty - tht - 4), (tx + tw + 4, ty + base + 2), OFF_COLOR, -1)
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, th - 1), cv2.LINE_AA)
