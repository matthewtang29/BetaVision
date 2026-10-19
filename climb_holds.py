"""
climb_holds.py - Step 2 of the climbing analyzer: mark every hold on one route.

How it works:
  1. A window shows your video. Drag the Frame slider to a sharp moment where
     the route's holds are visible (not blocked by the climber).
  2. Mark the holds, either way (switch with the M key):
       AUTO mode   : click one hold -> every hold of that colour is outlined.
                     Click a missed hold (shadowed, chalky) to widen the match.
       MANUAL mode : each click outlines just the hold you clicked.
     In both modes, RIGHT-click a hold (or volume) to delete it.
     VOLUMES are found automatically. Grey ones always count as allowed; a painted
     volume only counts if it's the route's colour (add one by hand with V/P to force it);
     holds on a volume that aren't the route's colour are cut out of it.
  3. Tag the route: press S then click a start hold (up to 2),
                    press E then click the end (top) hold.
  4. Press Enter to save:
       <name>_holds.png  : the frame with the holds outlined and numbered
       <name>_holds.json : each hold's centre, bounding box and outline (pixels)

Usage:
    python climb_holds.py IMG_5687.MOV
    python climb_holds.py IMG_5687.MOV --frame 120    # start the slider at frame 120
    python climb_holds.py IMG_5687.MOV --median       # blend frames to erase the climber (tripod only)
    python climb_holds.py IMG_5687.MOV --pick 540,900  # no window: auto mode with these click points

Window controls (also listed in the window itself):
    mouse wheel  scroll up/down     shift+wheel  left/right
    ctrl+wheel or + / -  zoom       arrow keys   move      0  zoom to fit
    left click   AUTO: find all holds of this colour | MANUAL: add this one hold
    right click  delete the hold under the cursor
    m            switch AUTO / MANUAL
    s            next click tags a START hold (click a tagged one again to untag; max 2)
    e            next click tags the END hold
    v            next click auto-outlines a VOLUME the auto-detect missed
    p            draw a VOLUME by clicking its corners (click the first corner, or Enter, to finish)
    z            undo          r  clear everything
    Enter        save          q / Esc  quit without saving
    Sliders: Frame (scrub video), Colour tol (how loose the auto colour match is), Min size
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import climb_volumes as cvol

WIN = "Holds  |  M = auto/manual   S = start   E = end   V/P = add volume   right-click = delete   Enter = save"
GREY_CHROMA = 14      # samples with less colour than this are white/grey/black holds
GROW = 1.7            # holds grow into connected pixels within GROW x the colour tolerance
GRABCUT_SEED = 1
GREEN, CYAN, MAGENTA, ORANGE, RED = (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 165, 255), (0, 0, 255)
VOL_COLOR, CUT_COLOR = (235, 235, 235), (180, 105, 255)     # volumes white, off-route cut-outs pink
ROLE_COLOR = {"start": CYAN, "end": RED}
ROLE_LABEL = {"start": "START", "end": "END"}


# ---------------------------------------------------------------- video
class FrameReader:
    def __init__(self, video):
        self.cap = cv2.VideoCapture(str(video))
        if not self.cap.isOpened():
            sys.exit(f"Could not open {video}")
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def read(self, i):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = self.cap.read()
        return f if ok else None


def median_wall(video, n_samples=15):
    """Blend frames to erase the moving climber. Only works if the phone was still."""
    reader = FrameReader(video)
    frames = [reader.read(i) for i in np.linspace(0, reader.total - 1, n_samples)]
    frames = [f for f in frames if f is not None]
    if not frames:
        sys.exit("Could not read any frames")
    median = np.median(np.stack(frames), axis=0).astype(np.uint8)

    def sharpness(im):
        return cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    if sharpness(median) < 0.5 * float(np.median([sharpness(f) for f in frames])):
        print("The camera moved, so frames can't be blended. Using the first frame instead.")
        return frames[0]
    print(f"Blended {len(frames)} frames to erase the climber.")
    return median


# ---------------------------------------------------------------- colour helpers
def _fd(h, w):
    """(min side, area) of ONE video frame - thresholds are relative to that, so they
    also work on a stitched panorama of the whole wall."""
    fh, fw = cvol.frame_dims(h, w)
    return min(fh, fw), fh * fw


def flatten_lighting(img):
    """Gym walls are lit unevenly (spotlights, dark corners), which makes the same
    hold colour look different across the wall. Divide out the large-scale
    brightness pattern so every part of the wall looks evenly lit."""
    h, w = img.shape[:2]
    f = img.astype(np.float32) + 1
    small = cv2.resize(f, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
    bg = cv2.GaussianBlur(small, (0, 0), _fd(h, w)[0] * 0.2 / 8)
    bg = cv2.resize(bg.mean(axis=2), (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
    return np.clip(f / bg * bg.mean(), 0, 255).astype(np.uint8)


def to_lab(img):
    """Lab colour space: a/b = the colour itself, L = brightness. After flattening
    the lighting, matching on colour this way is far less finicky than HSV boxes."""
    img = flatten_lighting(cv2.GaussianBlur(img, (5, 5), 0))
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)


def sample_lab(lab, x, y, r):
    """Colour of the hold at (x, y). Uses the most colourful half of a small patch,
    so a chalky spot or a bolt hole under the click doesn't spoil the sample."""
    h, w = lab.shape[:2]
    r = max(r, int(_fd(h, w)[0] * 0.01))
    patch = lab[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)].reshape(-1, 3)
    chroma = np.hypot(patch[:, 1] - 128, patch[:, 2] - 128)
    if np.percentile(chroma, 75) < GREY_CHROMA:      # white / grey / black hold: plain median
        return [float(v) for v in np.median(patch, axis=0)]
    keep = patch[chroma >= np.median(chroma)]
    return [float(v) for v in np.median(keep if len(keep) else patch, axis=0)]


def colour_distance(lab, sample):
    L, a, b = sample
    chroma = np.hypot(a - 128, b - 128)
    dL = lab[..., 0] - L
    dab = np.hypot(lab[..., 1] - a, lab[..., 2] - b)
    if chroma < GREY_CHROMA:            # white/grey/black: brightness matters, colour doesn't
        return np.sqrt(dab ** 2 + (0.6 * dL) ** 2)
    return np.sqrt(dab ** 2 + (0.25 * dL) ** 2)   # coloured: mostly the colour, a little brightness


def clean_mask(mask, h, w):
    k = max(3, int(round(_fd(h, w)[0] / 250)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def contour_to_hold(c, source):
    m = cv2.moments(c)
    if m["m00"] == 0:
        return None
    x, y, bw, bh = cv2.boundingRect(c)
    poly = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True).reshape(-1, 2)
    return {"center": [int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])],
            "bbox": [x, y, bw, bh], "area": int(cv2.contourArea(c)),
            "contour": poly.tolist(), "source": source}


def inside(hold, pt):
    cnt = np.array(hold["contour"], np.int32)
    return cv2.pointPolygonTest(cnt, (float(pt[0]), float(pt[1])), False) >= 0


# ---------------------------------------------------------------- AUTO: every hold of a colour
def outline_arcs(lab):
    """Hold-sized curved edge pieces. Every hold has an outline (its edge and the shadow
    under it); on a grey wall a grey hold's outline is often broken where the lit side
    matches the wall, but what's left is still a curved arc around the hold. The convex
    hull of each arc approximates the hold. Straight pieces (creases, seams) are skipped.
    Returns [(hull, median Lab inside)]."""
    h, w = lab.shape[:2]
    L = np.clip(lab[..., 0], 0, 255).astype(np.uint8)
    e = cv2.Canny(cv2.GaussianBlur(L, (5, 5), 0), 12, 35)
    e = cv2.dilate(e, np.ones((3, 3), np.uint8))
    n, lbl, st, _ = cv2.connectedComponentsWithStats(e, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if max(bw, bh) < 0.02 * _fd(h, w)[0] or max(bw, bh) > 0.15 * _fd(h, w)[0]:
            continue
        pts = np.column_stack(np.where(lbl[y:y + bh, x:x + bw] == i))[:, ::-1] + [x, y]
        hull = cv2.convexHull(pts.astype(np.int32))
        (_, _), (rw, rh), _ = cv2.minAreaRect(hull)
        if min(rw, rh) < 0.25 * max(rw, rh):
            continue                                       # straight-ish: crease or seam
        m = np.zeros((bh, bw), np.uint8)
        cv2.drawContours(m, [hull - [x, y]], -1, 1, -1)
        m &= (lbl[y:y + bh, x:x + bw] != i)                # inside the arc, not the arc itself
        if m.sum() < 20:
            continue
        out.append((hull, np.median(lab[y:y + bh, x:x + bw][m > 0], axis=0)))
    return out


def _hold_shaped(c, area, strict=True):
    """Reject long thin strips (edges of volume faces, seams, panel edges). Coloured holds
    can be oddly shaped (X / bow-tie / crescent), so the non-strict test is looser."""
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    if min(rw, rh) == 0:
        return False
    hull_area = cv2.contourArea(cv2.convexHull(c))
    if strict:
        return max(rw, rh) / min(rw, rh) < 4.0 and area / max(hull_area, 1) > 0.6
    return max(rw, rh) / min(rw, rh) < 6.0 and area / max(hull_area, 1) > 0.35


def find_holds_auto(lab, samples, tol=22, min_area=None):
    h, w = lab.shape[:2]
    if not samples:
        return []
    if min_area is None:
        min_area = int(_fd(h, w)[1] * 0.0003)
    col = [s for s in samples if np.hypot(s[1] - 128, s[2] - 128) >= GREY_CHROMA]
    mask = np.zeros((h, w), np.uint8)
    if col:
        dist = np.min([colour_distance(lab, s) for s in col], axis=0)
        strong = clean_mask((dist < tol).astype(np.uint8) * 255, h, w)
        # Grow each hold into its shaded / sun-lit parts: a looser colour match, but only
        # where it's connected to a strong match (so a similar-coloured wall isn't pulled in).
        weak = (dist < GROW * tol).astype(np.uint8)
        n, lbl = cv2.connectedComponents(weak | (strong > 0), connectivity=8)
        keep = np.zeros(n, bool)
        keep[np.unique(lbl[strong > 0])] = True
        keep[0] = False
        grown = keep[lbl].astype(np.uint8) * 255
        # don't let a hold's growth run off far beyond its strong core
        k = max(3, int(_fd(h, w)[0] * 0.03) | 1)
        grown &= cv2.dilate(strong, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        mask = clean_mask(grown, h, w)
    colour_only = mask.copy()
    # Grey / white / black holds: the wall is grey too, so colour alone would match the
    # whole wall. Find them by their outline instead (see outline_arcs).
    grey_samples = [s for s in samples if np.hypot(s[1] - 128, s[2] - 128) < GREY_CHROMA]
    if grey_samples:                       # add hold outlines whose inside matches the grey
        for c, med in outline_arcs(lab):
            if np.hypot(med[1] - 128, med[2] - 128) >= GREY_CHROMA:
                continue
            if min(float(colour_distance(med[None, None, :], s)[0, 0]) for s in grey_samples) < tol:
                cv2.drawContours(mask, [c], -1, 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    holds = []
    for c in contours:
        area = cv2.contourArea(c)
        x, y = c[0][0]
        strict = colour_only[y, x] == 0                                       # grey hold (from outlines)
        if min_area <= area <= 0.03 * _fd(h, w)[1] and _hold_shaped(c, area, strict):   # skip specks, wall patches, strips
            hold = contour_to_hold(c, "auto")
            if hold:
                holds.append(hold)
    return holds


# ---------------------------------------------------------------- MANUAL: just the clicked hold
def _grabcut_hold(img, x, y, half):
    """GrabCut the object under (x, y) inside a box of +-half px. Returns (contour, touches_box_edge)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, x - half), max(0, y - half), min(w, x + half), min(h, y + half)
    roi = np.ascontiguousarray(img[y0:y1, x0:x1])
    if roi.shape[0] < 12 or roi.shape[1] < 12:
        return None, False
    m = np.full(roi.shape[:2], cv2.GC_PR_BGD, np.uint8)
    cx, cy = x - x0, y - y0
    cv2.circle(m, (cx, cy), int(half * 0.55), cv2.GC_PR_FGD, -1)
    cv2.circle(m, (cx, cy), max(4, int(half * 0.08)), cv2.GC_FGD, -1)
    m[:3, :] = cv2.GC_BGD; m[-3:, :] = cv2.GC_BGD; m[:, :3] = cv2.GC_BGD; m[:, -3:] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.setRNGSeed(GRABCUT_SEED)            # GrabCut's colour model uses k-means: make it repeatable
    try:
        cv2.grabCut(roi, m, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None, False
    fg = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
    n, lbl = cv2.connectedComponents(fg)
    if lbl[cy, cx] == 0:
        return None, False
    fg = (lbl == lbl[cy, cx]).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cs, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None, False
    c = max(cs, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(c)
    edge = bx <= 4 or by <= 4 or bx + bw >= roi.shape[1] - 4 or by + bh >= roi.shape[0] - 4
    return c + np.array([x0, y0]), edge


def outline_one_hold_img(img, x, y):
    """Outline the single hold under a click with GrabCut (separates an object from a
    similar-coloured background far better than a colour threshold). The search box
    starts small and grows if the hold turns out to be bigger."""
    mind = _fd(*img.shape[:2])[0]
    lab = to_lab(img)
    sample = sample_lab(lab, x, y, max(3, mind // 200))
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    cands = []
    for half in (int(mind * 0.06), int(mind * 0.10), int(mind * 0.17)):
        c, edge = _grabcut_hold(img, x, y, half)
        if c is None or edge:                      # touching the box edge = spilled into the wall
            continue
        area = cv2.contourArea(c)
        if not ((mind * 0.008) ** 2 < area < 0.6 * (2 * half) ** 2):
            continue
        pts = c.reshape(-1, 2)
        edge_strength = float(grad[np.clip(pts[:, 1], 0, g.shape[0] - 1), np.clip(pts[:, 0], 0, g.shape[1] - 1)].mean())
        m = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(m, [c], -1, 1, -1)
        same = float((colour_distance(lab[m > 0][None], sample)[0] < 20).mean())
        cands.append((area, edge_strength, same, c))
    if not cands:
        return None
    # Start from the smallest outline. A bigger box only wins if its outline runs along clearly
    # stronger edges (a real hold boundary, not a line through the middle of the hold) without
    # losing most of the clicked colour (which would mean it swallowed a neighbouring hold).
    cands.sort(key=lambda t: t[0])
    _, best_edge, best_same, best = cands[0]
    for _, e, sm, c in cands[1:]:
        clearly_better = e >= 3.0 * best_edge                  # smaller one wasn't a real boundary
        if clearly_better or (e >= 1.1 * best_edge and sm >= 0.5 * best_same):
            best_edge, best_same, best = e, sm, c
    best = cv2.approxPolyDP(best, max(1.0, 0.004 * cv2.arcLength(best, True)), True)
    return contour_to_hold(best.reshape(-1, 1, 2), "manual")


def outline_one_hold(lab, x, y):
    """Outline the single hold under (x, y): look in a box around the click, split
    'like the click' from 'like the wall' by colour (Otsu threshold), keep the blob
    touching the click. Falls back to a circle if the hold blends into the wall."""
    h, w = lab.shape[:2]
    R = int(_fd(h, w)[0] * 0.08)
    x0, y0, x1, y1 = max(0, x - R), max(0, y - R), min(w, x + R + 1), min(h, y + R + 1)
    roi = lab[y0:y1, x0:x1]
    r = max(3, _fd(h, w)[0] // 200)
    sample = sample_lab(lab, x, y, r)
    dist = colour_distance(roi, sample)
    d8 = np.clip(dist * 3, 0, 255).astype(np.uint8)
    thresh, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = max(min(thresh, 40 * 3), 8 * 3)        # keep the cut-off sensible
    mask = clean_mask(((d8 < thresh) * 255).astype(np.uint8), h, w)
    n, labels = cv2.connectedComponents(mask)
    lbl = labels[y - y0, x - x0]
    fallback_r = int(_fd(h, w)[0] * 0.025)
    if lbl != 0:
        blob = (labels == lbl).astype(np.uint8) * 255
        area = blob.sum() / 255
        if blob.shape[0] * blob.shape[1] * 0.85 > area > (fallback_r ** 2):   # not leaked into the wall, not a speck
            cs, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cs, key=cv2.contourArea) + np.array([x0, y0])
            return contour_to_hold(c, "manual")
    circle = cv2.ellipse2Poly((x, y), (fallback_r, fallback_r), 0, 0, 360, 15).reshape(-1, 1, 2)
    return contour_to_hold(circle, "manual")


# ---------------------------------------------------------------- combine + draw
def combine(auto_holds, manual_holds, deleted, start_pts=(), end_pt=None):
    holds = [hd for hd in auto_holds if not any(inside(hd, p) for p in deleted)]
    holds += manual_holds
    holds = [dict(hd) for hd in holds]
    holds.sort(key=lambda d: -d["center"][1])        # number bottom-to-top
    for i, hd in enumerate(holds, 1):
        hd["id"] = i
        hd["role"] = ""
        if end_pt is not None and inside(hd, end_pt):
            hd["role"] = "end"
        elif any(inside(hd, p) for p in start_pts):
            hd["role"] = "start"
    return holds


def draw_volumes(out, vols):
    """Volumes: white outline + id. Off-route holds sitting on them: pink outline."""
    h, w = out.shape[:2]
    th = max(2, _fd(h, w)[0] // 300)
    fs = max(0.6, _fd(h, w)[0] / 1000)
    for v in vols:
        cnt = np.array(v["contour"], np.int32)
        cv2.polylines(out, [cnt], True, (40, 40, 40), th + 2, cv2.LINE_AA)
        cv2.polylines(out, [cnt], True, VOL_COLOR, th, cv2.LINE_AA)
        for e in v.get("exclude", []):
            cv2.polylines(out, [np.array(e, np.int32)], True, CUT_COLOR, th, cv2.LINE_AA)
        top = min(v["contour"], key=lambda p: p[1])
        org = (int(top[0]) - 20, int(top[1]) - 10)
        cv2.putText(out, v["id"], org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 3, cv2.LINE_AA)
        cv2.putText(out, v["id"], org, cv2.FONT_HERSHEY_SIMPLEX, fs, VOL_COLOR, th, cv2.LINE_AA)
    return out


COLOURED_VOLUME = 12   # a volume with more colour (Lab chroma) than this is "painted", not grey


def filter_coloured_volumes(lab, vols, holds):
    """Gym rule: grey / neutral volumes are always allowed, but a coloured (painted) volume
    only counts if it's the same colour as the route's holds. Returns (kept, skipped)."""
    if not vols:
        return [], []
    rm = route_mask(holds, lab.shape)
    route_ab = None
    if rm.any():
        route_ab = (float(np.median(lab[..., 1][rm > 0])), float(np.median(lab[..., 2][rm > 0])))
    kept, skipped = [], []
    for v in vols:
        m = np.zeros(lab.shape[:2], np.uint8)
        cv2.fillPoly(m, [np.array(v["contour"], np.int32)], 1)
        m = cv2.erode(m, np.ones((9, 9), np.uint8)) & (1 - rm)
        if not m.any():
            kept.append(v)
            continue
        a, b = float(np.median(lab[..., 1][m > 0])), float(np.median(lab[..., 2][m > 0]))
        neutral = np.hypot(a - 128, b - 128) < COLOURED_VOLUME
        same_as_route = route_ab is not None and np.hypot(a - route_ab[0], b - route_ab[1]) < 20
        (kept if neutral or same_as_route else skipped).append(v)
    return kept, skipped


def route_mask(holds, shape):
    m = np.zeros(shape[:2], np.uint8)
    for hd in holds:
        cv2.fillPoly(m, [np.array(hd["contour"], np.int32)], 1)
    return m


def draw_holds(img, holds, clicks=(), mode=None, header=True):
    out = img.copy()
    h, w = img.shape[:2]
    th = max(2, _fd(h, w)[0] // 300)
    fs = max(0.6, _fd(h, w)[0] / 1000)

    def text(s, org, scale):
        cv2.putText(out, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 4, cv2.LINE_AA)
        cv2.putText(out, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), th, cv2.LINE_AA)

    for hd in holds:
        cnt = np.array(hd["contour"], np.int32)
        role = hd.get("role", "")
        color = ROLE_COLOR.get(role, GREEN if hd["source"] == "auto" else ORANGE)
        cv2.drawContours(out, [cnt], -1, color, th + (2 if role else 0), cv2.LINE_AA)
        label = str(hd["id"]) + (f" {ROLE_LABEL[role]}" if role else "")
        text(label, (hd["center"][0] + 8, hd["center"][1]), fs)
    for x, y in clicks:
        cv2.drawMarker(out, (x, y), MAGENTA, cv2.MARKER_CROSS, 30, th, cv2.LINE_AA)
    if header:
        title = f"{len(holds)} holds" + (f"   [{mode}]" if mode else "")
        text(title, (20, int(50 * fs)), fs * 1.2)
    return out


# ---------------------------------------------------------------- saving
def save(video, out_dir, img, holds, info):
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{video.stem}_holds.png"
    js = out_dir / f"{video.stem}_holds.json"
    cv2.imwrite(str(png), draw_volumes(draw_holds(img, holds), info.get("volumes", [])))
    data = {"video": video.name, "image_size": [img.shape[1], img.shape[0]], **info, "holds": holds}
    js.write_text(json.dumps(data, indent=1))
    print(f"Saved {len(holds)} holds\n  image -> {png}\n  data  -> {js}")


# ---------------------------------------------------------------- the window
VIEW_W, VIEW_H, PANEL_W = 820, 720, 390         # window = picture viewport + instructions panel
ARROWS = {2490368: (0, -1), 2621440: (0, 1), 2424832: (-1, 0), 2555904: (1, 0),      # Windows
          65362: (0, -1), 65364: (0, 1), 65361: (-1, 0), 65363: (1, 0)}              # Linux / Mac


def wheel_delta(flags):
    """+ for wheel up, - for wheel down (OpenCV packs it into the top 16 bits of flags)."""
    if hasattr(cv2, "getMouseWheelDelta"):
        return cv2.getMouseWheelDelta(flags)
    d = (flags >> 16) & 0xFFFF
    return d - 0x10000 if d >= 0x8000 else d


def draw_panel(lines):
    """Instructions + status panel on the right of the window."""
    panel = np.full((VIEW_H, PANEL_W, 3), 32, np.uint8)
    y = 26
    for item in lines:
        if item is None:                      # blank line
            y += 10
            continue
        txt, style = item if isinstance(item, tuple) else (item, "body")
        if style == "head":
            y += 6
            cv2.putText(panel, txt, (14, y), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 215, 255), 1, cv2.LINE_AA)
            y += 24
        elif style.startswith("swatch"):      # coloured line sample + label
            color = {"swatch_g": GREEN, "swatch_o": ORANGE, "swatch_c": CYAN, "swatch_r": RED,
                     "swatch_w": VOL_COLOR, "swatch_p": CUT_COLOR}[style]
            cv2.line(panel, (16, y - 5), (40, y - 5), color, 3, cv2.LINE_AA)
            cv2.putText(panel, txt, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
            y += 20
        else:
            color = (255, 255, 255) if style == "strong" else (200, 200, 200)
            cv2.putText(panel, txt, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            y += 20
    return panel


def interactive(reader, start_frame, fixed_img=None, frame_dims=None, fixed_label=None):
    """Open the hold picker. With fixed_img (e.g. a stitched panorama of the whole wall)
    there's no Frame slider; frame_dims = (h, w) of one video frame keeps hold-size
    thresholds right on a picture bigger than a frame."""
    cvol.FRAME = frame_dims
    try:
        return _interactive(reader, start_frame, fixed_img, fixed_label)
    finally:
        cvol.FRAME = None


def _interactive(reader, start_frame, fixed_img=None, fixed_label=None):
    img = fixed_img if fixed_img is not None else reader.read(start_frame)
    if img is None:
        sys.exit("Could not read the video")
    h, w = img.shape[:2]
    fit = min(VIEW_W / w, VIEW_H / h)               # zoom level that shows the whole frame
    r = max(3, _fd(h, w)[0] // 200)
    S = {"img": img, "lab": to_lab(img), "frame": start_frame, "mode": "AUTO",
         "samples": [], "clicks": [], "manual": [], "deleted": [], "history": [],
         "starts": [], "end": None, "tag": None, "msg": "", "poly": [],
         "vols_auto": cvol.detect_volumes(img), "vols_manual": [], "vols_deleted": [], "vols": [],
         "auto": [], "auto_key": None, "holds": [], "dirty": True, "annotated": None,
         "zoom": fit, "ox": 0.0, "oy": 0.0}         # ox/oy = image pixel at the viewport's top-left

    # ---------- view (zoom + scroll)
    def clamp_view():
        vw, vh = VIEW_W / S["zoom"], VIEW_H / S["zoom"]       # visible size in image pixels
        S["ox"] = (w - vw) / 2 if vw >= w else min(max(S["ox"], 0), w - vw)
        S["oy"] = (h - vh) / 2 if vh >= h else min(max(S["oy"], 0), h - vh)

    def set_zoom(z, cx=VIEW_W / 2, cy=VIEW_H / 2):
        """Zoom keeping the image point under (cx, cy) in the viewport fixed."""
        ix, iy = S["ox"] + cx / S["zoom"], S["oy"] + cy / S["zoom"]
        S["zoom"] = min(max(z, fit), 4.0)
        S["ox"], S["oy"] = ix - cx / S["zoom"], iy - cy / S["zoom"]
        clamp_view()

    def scroll(dx, dy):                                 # in viewport pixels
        S["ox"] += dx / S["zoom"]
        S["oy"] += dy / S["zoom"]
        clamp_view()

    def to_image(x, y):
        if x >= VIEW_W:
            return None                                 # click landed on the panel
        ix, iy = int(S["ox"] + x / S["zoom"]), int(S["oy"] + y / S["zoom"])
        return (ix, iy) if 0 <= ix < w and 0 <= iy < h else None

    def render_view():
        z, ox, oy = S["zoom"], S["ox"], S["oy"]
        M = np.float32([[z, 0, -ox * z], [0, z, -oy * z]])        # image -> viewport
        interp = cv2.INTER_AREA if z < 1 else cv2.INTER_LINEAR
        return cv2.warpAffine(S["annotated"], M, (VIEW_W, VIEW_H), flags=interp,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(45, 45, 45))

    # ---------- marking
    def tag_hold(fx, fy):
        hit = next((hd for hd in S["holds"] if inside(hd, (fx, fy))), None)
        if hit is None and S["holds"]:                  # forgiving: nearest hold if the click was close
            dist = lambda hd: -cv2.pointPolygonTest(np.array(hd["contour"], np.int32), (float(fx), float(fy)), True)
            near = min(S["holds"], key=dist)
            if dist(near) <= 0.04 * _fd(h, w)[0]:
                hit = near
        if hit is None:
            S["msg"] = "No hold there - click inside an outline."
            return
        pt = hit["center"] if inside(hit, hit["center"]) else [fx, fy]
        if S["tag"] == "start":
            S["history"].append(("starts", list(S["starts"])))   # snapshot for undo
            same = [p for p in S["starts"] if inside(hit, p)]
            if same:                                    # clicking a start hold again untags it
                for p in same:
                    S["starts"].remove(p)
            else:
                S["starts"].append(pt)
                if len(S["starts"]) > 2:
                    S["starts"].pop(0)
        else:
            S["history"].append(("end", S["end"]))
            S["end"] = pt
        S["tag"] = None

    def finish_poly():
        if len(S["poly"]) >= 3:
            pts = [list(map(int, p)) for p in S["poly"]]
            S["vols_manual"].append({"contour": pts, "faces": [pts], "source": "manual"})
            S["history"].append("vol_manual")
            S["dirty"] = True
        S["poly"] = []
        S["tag"] = None

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = wheel_delta(flags)
            if flags & cv2.EVENT_FLAG_CTRLKEY:          # Ctrl + wheel = zoom at the cursor
                set_zoom(S["zoom"] * (1.2 if delta > 0 else 1 / 1.2), min(x, VIEW_W), y)
            elif flags & cv2.EVENT_FLAG_SHIFTKEY:       # Shift + wheel = left/right
                scroll(-120 if delta > 0 else 120, 0)
            else:                                       # wheel = up/down
                scroll(0, -120 if delta > 0 else 120)
            S["view_dirty"] = True
            return
        if event == cv2.EVENT_MOUSEHWHEEL:              # trackpad sideways swipe
            scroll(120 if wheel_delta(flags) > 0 else -120, 0)
            S["view_dirty"] = True
            return
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        p = to_image(x, y)
        if p is None:
            return
        fx, fy = p
        S["msg"] = ""
        if event == cv2.EVENT_LBUTTONDOWN and S["tag"] == "poly":
            close_px = 18 / S["zoom"]                   # ~18 screen pixels, whatever the zoom
            if len(S["poly"]) >= 3 and np.hypot(fx - S["poly"][0][0], fy - S["poly"][0][1]) < close_px:
                finish_poly()
            else:
                S["poly"].append([fx, fy])
            S["view_dirty"] = True
            return
        if event == cv2.EVENT_LBUTTONDOWN and S["tag"] == "volume":
            v = cvol.volume_from_click(S["img"], fx, fy)
            if v is None:
                S["msg"] = "Couldn't find a volume there - try its middle."
            else:
                S["vols_manual"].append(v)
                S["history"].append("vol_manual")
            S["tag"] = None
        elif event == cv2.EVENT_LBUTTONDOWN and S["tag"]:
            tag_hold(fx, fy)
        elif event == cv2.EVENT_LBUTTONDOWN:
            if S["mode"] == "AUTO":
                S["samples"].append(sample_lab(S["lab"], fx, fy, r))
                S["clicks"].append([fx, fy])
                S["history"].append("sample")
            else:
                hold = outline_one_hold_img(S["img"], fx, fy) or outline_one_hold(S["lab"], fx, fy)
                if hold:
                    S["manual"].append(hold)
                    S["history"].append("manual")
        else:                                           # right click: delete hold, else volume
            for i, hd in enumerate(S["manual"]):
                if inside(hd, (fx, fy)):
                    S["history"].append(("unmanual", S["manual"].pop(i)))
                    break
            else:
                if any(inside(hd, (fx, fy)) for hd in S["holds"]):
                    S["deleted"].append([fx, fy])
                    S["history"].append("deleted")
                else:
                    for i, v in enumerate(S["vols_manual"]):
                        if cvol.inside_volume(v, (fx, fy)):
                            S["history"].append(("unvol", S["vols_manual"].pop(i)))
                            break
                    else:
                        if any(cvol.inside_volume(v, (fx, fy)) for v in S["vols_auto"]):
                            S["vols_deleted"].append([fx, fy])
                            S["history"].append("vol_deleted")
                        else:
                            S["msg"] = "Nothing to delete there."
        S["dirty"] = True

    def undo():
        if not S["history"]:
            return
        last = S["history"].pop()
        if last == "sample":
            S["samples"].pop(); S["clicks"].pop()
        elif last == "manual":
            S["manual"].pop()
        elif last == "deleted":
            S["deleted"].pop()
        elif last[0] == "starts":
            S["starts"][:] = last[1]
        elif last[0] == "end":
            S["end"] = last[1]
        elif last == "vol_manual":
            S["vols_manual"].pop()
        elif last == "vol_deleted":
            S["vols_deleted"].pop()
        elif last[0] == "unmanual":
            S["manual"].append(last[1])
        elif last[0] == "unvol":
            S["vols_manual"].append(last[1])

    def panel_lines():
        n_start = sum(hd["role"] == "start" for hd in S["holds"])
        n_end = sum(hd["role"] == "end" for hd in S["holds"])
        if S["tag"] == "start":
            doing = "Now: click a START hold"
        elif S["tag"] == "end":
            doing = "Now: click the END hold"
        elif S["tag"] == "poly":
            doing = f"Now: click the volume's corners ({len(S['poly'])} so far)"
        elif S["tag"] == "volume":
            doing = "Now: click a VOLUME to add it"
        elif S["mode"] == "AUTO":
            doing = "Now: AUTO - click = all holds this colour"
        else:
            doing = "Now: MANUAL - click = add one hold"
        frame_txt = (f"Frame {S['frame']}  ({S['frame'] / reader.fps:.1f}s)" if fixed_img is None
                     else (fixed_label or "Frame: blended wall"))
        n_cut = sum(len(v.get("exclude", [])) for v in S["vols"])
        lines = [("HOW TO MARK THE ROUTE", "head"),
                 ("1. Drag the Frame slider to a clear frame" if fixed_img is None
                  else "1. This is your whole wall, stitched together"),
                 "2. Click one hold of your route",
                 "    -> every hold that colour is outlined",
                 "3. Right-click wrong holds or volumes",
                 "4. M = manual: click any missed holds",
                 "5. S, click a start hold (twice)",
                 "    E, click the end hold",
                 "6. Volumes are found automatically",
                 "    (always allowed). Missed one? V + click,",
                 "    or P + click its corners (exact outline)",
                 ("7. Press ENTER to save and continue", "strong"),
                 ("MOVING AROUND", "head"),
                 "Wheel scroll    Shift+wheel sideways",
                 "Ctrl+wheel or + -  zoom     0  fit",
                 "Arrow keys  move around",
                 ("OTHER KEYS", "head"),
                 "Z undo     R clear all     Q quit",
                 "Colour tol slider: up = catch more holds",
                 ("COLOURS", "head"),
                 ("auto-found hold", "swatch_g"), ("manually added hold", "swatch_o"),
                 ("start hold", "swatch_c"), ("end hold", "swatch_r"),
                 ("volume (always allowed)", "swatch_w"), ("off-route hold on a volume", "swatch_p"),
                 ("STATUS", "head"),
                 (doing, "strong"),
                 f"Holds: {len(S['holds'])}    Start: {n_start}/2    End: {'yes' if n_end else 'no'}",
                 f"Volumes: {len(S['vols'])}  (coloured, skipped: {len(S.get('vols_skipped', []))})",
                 f"Off-route holds cut out of volumes: {n_cut}",
                 f"{frame_txt}    Zoom: {S['zoom'] / fit * 100:.0f}%"]
        if S["msg"]:
            lines.append((S["msg"], "strong"))
        return lines

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    sliders = [("Colour tol", 22, 60), ("Min size", 3, 50)]   # Min size in 0.01% of the image
    if fixed_img is None:
        sliders.insert(0, ("Frame", start_frame, reader.total - 1))
    for name, val, mx in sliders:
        cv2.createTrackbar(name, WIN, val, mx, lambda v: None)
    for name, val, mx in sliders:
        cv2.setTrackbarPos(name, WIN, val)
    clamp_view()

    canvas = None
    S["view_dirty"] = True
    while True:
        if fixed_img is None:
            f_idx = cv2.getTrackbarPos("Frame", WIN)
            if f_idx != S["frame"]:
                f = reader.read(f_idx)
                if f is not None:
                    S.update(img=f, lab=to_lab(f), frame=f_idx, dirty=True,
                             vols_auto=cvol.detect_volumes(f))
        tol = max(1, cv2.getTrackbarPos("Colour tol", WIN))
        min_area = int(_fd(h, w)[1] * cv2.getTrackbarPos("Min size", WIN) / 10000)
        auto_key = (S["frame"], len(S["samples"]), tol, min_area)
        if auto_key != S["auto_key"]:
            S["auto"] = find_holds_auto(S["lab"], S["samples"], tol, min_area)
            S["auto_key"] = auto_key
            S["dirty"] = True
        if S["dirty"]:
            S["holds"] = combine(S["auto"], S["manual"], S["deleted"], S["starts"], S["end"])
            vols = [dict(v) for v in S["vols_auto"]
                    if not any(cvol.inside_volume(v, p) for p in S["vols_deleted"])]
            # painted volumes that aren't the route's colour are off-route (manual ones always kept)
            vols, S["vols_skipped"] = filter_coloured_volumes(S["lab"], vols, S["holds"])
            vols += [dict(v) for v in S["vols_manual"]]
            S["vols"] = cvol.cut_foreign_holds(cvol.renumber(vols), S["img"],
                                               route_mask(S["holds"], S["img"].shape), min_area)
            S["annotated"] = draw_volumes(draw_holds(S["img"], S["holds"], S["clicks"], header=False), S["vols"])
            S["dirty"] = False
            S["view_dirty"] = True
        if S["view_dirty"] or canvas is None:
            view = render_view()
            if S["poly"]:                               # corners clicked so far
                vp = [(int((px - S["ox"]) * S["zoom"]), int((py - S["oy"]) * S["zoom"])) for px, py in S["poly"]]
                cv2.polylines(view, [np.array(vp, np.int32)], False, VOL_COLOR, 2, cv2.LINE_AA)
                for q in vp:
                    cv2.circle(view, q, 5, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(view, q, 3, VOL_COLOR, -1, cv2.LINE_AA)
                cv2.circle(view, vp[0], 18, VOL_COLOR, 1, cv2.LINE_AA)      # click here to close
            canvas = np.hstack([view, draw_panel(panel_lines())])
            S["view_dirty"] = False
        cv2.imshow(WIN, canvas)

        key = cv2.waitKeyEx(30)
        if key == -1:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                return None
            continue
        if key in ARROWS:
            dx, dy = ARROWS[key]
            scroll(dx * 100, dy * 100)
            S["view_dirty"] = True
            continue
        key &= 0xFF
        if key in (13, 10) and S["tag"] == "poly":        # Enter finishes the volume being drawn
            finish_poly()
            S["view_dirty"] = True
            continue
        if key == 27 and S["tag"] == "poly":              # Esc cancels it
            S["poly"], S["tag"] = [], None
            S["view_dirty"] = True
            continue
        if key in (13, 10):
            cv2.destroyAllWindows()
            n_start = sum(hd["role"] == "start" for hd in S["holds"])
            n_end = sum(hd["role"] == "end" for hd in S["holds"])
            if n_start == 0 or n_end == 0:
                print(f"  Note: {n_start} start / {n_end} end holds tagged (press S / E to tag them).")
            info = {"frame": S["frame"], "colour_tol": tol, "min_area_px": min_area,
                    "colour_samples_lab": S["samples"], "clicks": S["clicks"], "deleted_points": S["deleted"],
                    "volumes": S["vols"]}
            return S["img"], S["holds"], info
        if key in (27, ord("q")):
            cv2.destroyAllWindows()
            return None
        if key in (ord("+"), ord("=")):
            set_zoom(S["zoom"] * 1.25)
        elif key in (ord("-"), ord("_")):
            set_zoom(S["zoom"] / 1.25)
        elif key == ord("0"):
            set_zoom(fit)
        elif key == ord("s"):
            S["tag"] = "start"
        elif key == ord("e"):
            S["tag"] = "end"
        elif key == ord("p"):
            S["tag"], S["poly"] = "poly", []
        elif key == ord("v"):
            S["tag"] = "volume"
        elif key == ord("m"):
            S["mode"] = "MANUAL" if S["mode"] == "AUTO" else "AUTO"
            S["tag"] = None
        elif key == ord("z"):
            undo(); S["dirty"] = True
        elif key == ord("r"):
            for k in ("samples", "clicks", "manual", "deleted", "history", "starts",
                      "vols_manual", "vols_deleted"):
                S[k].clear()
            S["end"] = None; S["tag"] = None
            S["dirty"] = True
        S["view_dirty"] = True


def main():
    ap = argparse.ArgumentParser(description="Mark all holds of one route in a climbing video.")
    ap.add_argument("video")
    ap.add_argument("--frame", type=int, default=0, help="frame to start on (default 0)")
    ap.add_argument("--median", action="store_true",
                    help="blend frames to erase the climber (only if the phone was perfectly still)")
    ap.add_argument("--pick", default=None, help="no window: auto-mode click points as 'x,y;x,y'")
    ap.add_argument("--out", default=None, help="output folder (default: next to the video)")
    args = ap.parse_args()

    video = Path(args.video)
    out_dir = Path(args.out) if args.out else video.parent

    if args.pick:
        img = median_wall(video) if args.median else FrameReader(video).read(args.frame)
        lab = to_lab(img)
        r = max(3, min(img.shape[:2]) // 200)
        clicks = [[int(v) for v in p.split(",")] for p in args.pick.split(";")]
        samples = [sample_lab(lab, x, y, r) for x, y in clicks]
        holds = combine(find_holds_auto(lab, samples), [], [])
        vols, _ = filter_coloured_volumes(lab, cvol.detect_volumes(img), holds)
        vols = cvol.cut_foreign_holds(vols, img, route_mask(holds, img.shape),
                                      int(img.shape[0] * img.shape[1] * 0.0003))
        info = {"frame": args.frame, "colour_samples_lab": samples, "clicks": clicks, "volumes": vols}
    else:
        reader = FrameReader(video)
        fixed = median_wall(video) if args.median else None
        result = interactive(reader, args.frame, fixed)
        if result is None:
            print("Quit without saving.")
            return
        img, holds, info = result

    save(video, out_dir, img, holds, info)


if __name__ == "__main__":
    main()
