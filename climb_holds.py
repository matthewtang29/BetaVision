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
import threading
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


#
# One AUTO click -> hold_model(): the clicked hold is segmented FIRST (by colour if it has
# one, by brightness if it's white / grey / black), and its colour is then measured over the
# whole hold, leaving out chalk, highlights and shadow. So it doesn't matter where on the hold
# you click, or which hold of the colour. find_holds_auto() then outlines every blob that
# matches a model: same hue (within the hold's own hue spread) and clearly more colourful than
# the wall around the clicked hold (or, for grey holds, clearly brighter / darker than it).
# The clicked hold itself is always in the result; the climber never is.

COLOURFUL = 12            # Lab chroma above this counts as "has a colour" when looking near a click
SEED_HUE = 24             # degrees: hue band used to segment the clicked hold
GREY_CONTRAST = 20        # a grey hold must differ from the wall around it by this much (L)
PALE_MARGIN = 4           # a hold not more colourful than the wall (by this much) is matched as grey
PERSON_OVERLAP = 0.7      # a blob more than this much covered by the climber isn't a hold
FRAGMENT_RING = 0.3       # a blob whose surroundings are this much its own colour is a piece of something bigger
MAX_AREA_FRAC = 0.03      # usual size cap for a found hold (fraction of one frame) ...
SEED_AREA_MULT = 2.5      # ... raised to this x the largest clicked hold


class FrameColour:
    """Everything the colour picker needs about one picture, computed once and reused on
    every refresh: flattened Lab, chroma, hue and (lazily) where the climber is."""

    def __init__(self, img=None, lab=None):
        self.img = img
        self.lab = to_lab(img) if lab is None else lab
        self.h, self.w = self.lab.shape[:2]
        self.mind = _fd(self.h, self.w)[0]
        self.L = self.lab[..., 0]
        a, b = self.lab[..., 1] - 128, self.lab[..., 2] - 128
        self.C = np.hypot(a, b)
        self.H = np.degrees(np.arctan2(b, a))          # hue angle, -180..180
        self._person = None
        self._Lbg = None

    def wall_L(self):
        """Brightness of the wall around each pixel (median over ~10% of the frame): grey
        holds are matched by how much brighter / darker they are than the wall around them,
        since the lighting flattening makes absolute brightness unreliable."""
        if self._Lbg is None:
            f = 4
            small = cv2.resize(np.clip(self.L, 0, 255).astype(np.uint8), (max(1, self.w // f), max(1, self.h // f)),
                               interpolation=cv2.INTER_AREA)
            bg = cv2.medianBlur(small, max(3, int(self.mind * 0.1 / f) | 1))
            self._Lbg = cv2.resize(bg, (self.w, self.h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        return self._Lbg

    def person_wide(self):
        """The climber's mask grown by a few px (its edge is soft)."""
        if getattr(self, "_person_wide", None) is None:
            self._person_wide = cv2.dilate(self.person(), _box(max(1, int(self.mind * 0.004))))
        return self._person_wide

    def person(self):
        """1 where the climber is (MediaPipe), 0 elsewhere. All 0 without a picture / model."""
        if self._person is None:
            self._person = (person_mask(self.img) if self.img is not None
                            else np.zeros((self.h, self.w), np.uint8))
        return self._person


_PERSON = {"setup": None, "tried": False, "lock": threading.Lock()}


def person_mask(img):
    """Where the climber is, from MediaPipe's pose segmentation (the same pose model the
    analysis uses, downloaded once). Returns zeros if MediaPipe can't run."""
    h, w = img.shape[:2]
    with _PERSON["lock"]:                    # MediaPipe isn't thread-safe: one frame at a time
        if not _PERSON["tried"]:
            _PERSON["tried"] = True
            try:
                import mediapipe as mp
                from mediapipe.tasks import python as mp_tasks
                from mediapipe.tasks.python import vision
                import climb_pose
                cache = Path.home() / ".climb_analyzer"
                model = next((cache / f"pose_landmarker_{k}.task" for k in ("full", "heavy", "lite")
                              if (cache / f"pose_landmarker_{k}.task").exists()), None)
                model = model or climb_pose.ensure_model("full")
                opts = vision.PoseLandmarkerOptions(
                    base_options=mp_tasks.BaseOptions(model_asset_path=str(model)),
                    running_mode=vision.RunningMode.IMAGE, num_poses=2, output_segmentation_masks=True,
                    min_pose_detection_confidence=0.3)
                _PERSON["setup"] = (vision, opts, mp)
            except Exception as e:           # no MediaPipe / no model / offline: just don't mask
                print(f"  (Couldn't load the pose model to hide the climber from the picker: {e})")
        if _PERSON["setup"] is None:
            return np.zeros((h, w), np.uint8)
        vision, opts, mp = _PERSON["setup"]
        try:
            # a landmarker per frame (closed by `with`): once per picker frame, ~0.15 s; one kept
            # open for the whole program errors when Python tears it down at exit
            with vision.PoseLandmarker.create_from_options(opts) as lm:
                res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                         data=np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
        except Exception as e:
            print(f"  (Pose segmentation failed: {e})")
            return np.zeros((h, w), np.uint8)
    out = np.zeros((h, w), np.uint8)
    for sm in res.segmentation_masks or []:
        m = sm.numpy_view()
        m = m[..., 0] if m.ndim == 3 else m
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h))
        out |= (m > 0.5).astype(np.uint8)
    return out


def _hue_diff(H, h0):
    d = np.abs(H - h0)
    return np.minimum(d, 360 - d)


def _circ_median(H):
    """Median of hue angles (degrees), robust to the -180/180 wrap."""
    r = np.radians(H)
    h0 = np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean()))
    dev = (H - h0 + 180) % 360 - 180
    return float((h0 + np.median(dev) + 180) % 360 - 180)


def _otsu(vals, lo, hi):
    """Otsu threshold of a 1-D array of chroma / brightness values, clipped to [lo, hi]."""
    if len(vals) < 20:
        return lo
    v = np.clip(vals, 0, 255).astype(np.uint8).reshape(-1, 1)
    t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(min(max(t, lo), hi))


def _disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _box(r):
    """Square kernel: OpenCV dilates with it many times faster than with a disk."""
    return np.ones((2 * r + 1, 2 * r + 1), np.uint8)


def _fill(mask):
    """Fill holes (chalk, bolt holes, highlights inside a hold)."""
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, cs, -1, 1, -1)
    return out


def _component_at(cand, cx, cy, reach):
    """The blob of `cand` under (cx, cy), or the biggest one within `reach` px of it."""
    n, lbl = cv2.connectedComponents(cand, connectivity=8)
    l = lbl[cy, cx]
    if l == 0:
        y0, y1, x0, x1 = max(0, cy - reach), cy + reach + 1, max(0, cx - reach), cx + reach + 1
        near = lbl[y0:y1, x0:x1]
        near = near[near > 0]
        if not len(near):
            return None
        l = np.bincount(near).argmax()
    return _fill((lbl == l).astype(np.uint8))


def _grow_seed(fc, x, y, cand_fn, hull_reach=0.0):
    """Segment the hold under (x, y) inside a box that grows until the blob fits in it.
    cand_fn(slices) -> uint8 0/1 candidate pixels in that box. Returns a full-size mask, or
    None if nothing is under the click or the blob runs off as far as we look (bare wall,
    a wall panel, a volume face). hull_reach > 0: a click up to that far (x the frame's
    short side) outside the blob still counts if it's inside the blob's convex hull - a
    chalk cap or highlight on the hold's edge - and the hull becomes the seed."""
    m = fc.mind
    k = _disk(max(2, int(m * 0.006)))                   # bridges chalk / highlights / bolt holes
    reach = max(2, int(m * max(0.012, hull_reach)))
    for frac in (0.1, 0.2, 0.35):
        R = int(m * frac)
        y0, y1, x0, x1 = max(0, y - R), min(fc.h, y + R + 1), max(0, x - R), min(fc.w, x + R + 1)
        cand = cv2.morphologyEx(cand_fn((slice(y0, y1), slice(x0, x1))), cv2.MORPH_CLOSE, k)
        comp = _component_at(cand, x - x0, y - y0, reach)
        if comp is None:
            return None
        if not comp[y - y0, x - x0]:
            cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull = cv2.convexHull(max(cs, key=cv2.contourArea))
            if not hull_reach or cv2.pointPolygonTest(hull, (float(x - x0), float(y - y0)), False) < 0:
                return None
            cv2.fillPoly(comp, [hull], 1)
        rows, cols = np.any(comp, 1), np.any(comp, 0)
        runs_off = ((rows[0] and y0 > 0) or (rows[-1] and y1 < fc.h) or
                    (cols[0] and x0 > 0) or (cols[-1] and x1 < fc.w))
        if not runs_off:
            full = np.zeros((fc.h, fc.w), np.uint8)
            full[y0:y1, x0:x1] = comp
            return full
    return None


def _seed_colour(fc, x, y):
    """Clicked hold has a colour: find it from the colourful pixels nearest the click (so a
    click on chalk or a highlight still reads the hold's colour), then segment it."""
    m = fc.mind
    # (radius, share of colourful pixels needed, hull reach): first look right at the click;
    # failing that, a bigger area with only a little colour - a click on a chalk cap or a
    # highlight that covers much of the hold - accepted only if the click is inside the hold
    for r, share, hull_reach in ((0.012, 0.15, 0), (0.025, 0.15, 0), (0.045, 0.15, 0.03), (0.06, 0.05, 0.04)):
        R = max(4, int(m * r))
        y0, x0 = max(0, y - R), max(0, x - R)
        C, H = fc.C[y0:y + R + 1, x0:x + R + 1], fc.H[y0:y + R + 1, x0:x + R + 1]
        yy, xx = np.mgrid[y0:y0 + C.shape[0], x0:x0 + C.shape[1]]
        d2 = (yy - y) ** 2 + (xx - x) ** 2
        col = (d2 <= R * R) & (C >= COLOURFUL)
        if col.sum() < max(12, share * (d2 <= R * R).sum()):
            continue
        near = np.argsort(d2[col])[:max(12, int(col.sum()) // 3)]
        hue, c_click = _circ_median(H[col][near]), float(np.median(C[col][near]))
        # hold vs wall pixels of this hue: Otsu first; if that lets a tinted wall in (the
        # blob runs off), retry with a cut-off close to the hold's own chroma
        for hi in (0.6, 0.8):
            def cand(sl, hi=hi):
                C, dh = fc.C[sl], _hue_diff(fc.H[sl], hue)
                band = dh <= SEED_HUE
                thr = _otsu(C[band], 8, max(8, hi * c_click)) if hi < 0.7 else max(8, hi * c_click)
                return (band & (C >= thr)).astype(np.uint8)
            seed = _grow_seed(fc, x, y, cand, hull_reach)
            if seed is not None:
                return seed, hue
        if r >= 0.045:
            break
    return None


def _seed_grey(fc, x, y):
    """Clicked hold has no colour (white / grey / black): segment it by brightness against
    the wall around it. None if it's as bright as the wall (= probably bare wall)."""
    m = fc.mind
    r = max(3, int(m * 0.01))
    L0 = float(np.median(fc.L[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]))

    def cand(sl):
        L = fc.L[sl]
        Lw = float(np.median(L))
        if abs(L0 - Lw) < GREY_CONTRAST:
            return np.zeros(L.shape, np.uint8)
        T = (L0 + Lw) / 2
        return (((L >= T) if L0 > Lw else (L <= T)) & (fc.C[sl] < max(COLOURFUL * 1.5, 18))).astype(np.uint8)
    seed = _grow_seed(fc, x, y, cand)
    if seed is None:
        return None
    # a hold is a compact blob; a ragged patch of chalk dust / lighter wall is not
    cs, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    return seed if cv2.contourArea(c) >= 0.6 * cv2.contourArea(cv2.convexHull(c)) else None


def hold_model(fc, x, y, frame=None):
    """Colour model of the hold under an AUTO click (JSON-friendly dict). Has a "reject"
    key with a message for the user when the click isn't on a hold."""
    x, y = int(min(max(x, 0), fc.w - 1)), int(min(max(y, 0), fc.h - 1))
    base = {"click": [x, y], "frame": frame}
    if fc.person()[y, x]:
        return dict(base, reject="person", message="That's the climber, not a hold. Click a hold.")
    s = _seed_colour(fc, x, y)
    kind = "colour"
    if s is not None:
        seed, hue = s
    else:
        seed, kind = _seed_grey(fc, x, y), "grey"
    if seed is None:
        return dict(base, reject="wall",
                    message="No hold found there (it looks like bare wall). Click on a hold, "
                            "or use manual mode for a hold that blends into the wall.")
    seed &= 1 - fc.person()                           # a hand on the hold isn't part of it
    if kind == "colour":
        # now that the hold's own hue is known (from the whole hold, not the click), cut off
        # any neighbouring hold of a nearby hue the first pass took in; chalk / highlight
        # pixels (no colour) stay.
        hue, hue_tol, _ = _colour_stats(fc, seed, hue)
        ok = (_hue_diff(fc.H, hue) <= max(hue_tol, 10)) | (fc.C < COLOURFUL)
        seed = _component_at((seed & ok).astype(np.uint8), x, y, max(2, int(fc.mind * 0.012)))
        if seed is None:
            return dict(base, reject="wall", message="No hold found there. Click on a hold.")
    area = int(seed.sum())
    if area < 20:
        return dict(base, reject="wall", message="No hold found there. Click on a hold.")
    # smooth the outline (marbled / chalky holds leave notches); a nearly convex hold gets its
    # convex hull, which also fills a notch where chalk or a highlight sat on its edge
    seed = _fill(cv2.morphologyEx(seed, cv2.MORPH_CLOSE, _disk(max(2, int(0.06 * np.sqrt(area))))))
    cs, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    if cv2.contourArea(c) >= 0.8 * cv2.contourArea(hull):
        seed = np.zeros_like(seed)
        cv2.fillPoly(seed, [hull], 1)
        seed &= 1 - fc.person()
    area = int(seed.sum())
    # the wall around it: a ring that starts a little way out (the hold's blurred edge isn't wall)
    k_in, k = max(3, int(fc.mind * 0.008)), max(12, int(0.4 * np.sqrt(area)))
    around = (cv2.dilate(seed, _disk(k_in + k)) > 0) & (cv2.dilate(seed, _disk(k_in)) == 0) & (fc.person() == 0)
    if kind == "colour":
        hue, hue_tol, clean = _colour_stats(fc, seed, hue)
        Cc, Lc = fc.C[clean], fc.L[clean]
        # how colourful the wall is in this hold's colour direction: chroma projected on the
        # hold's hue, upper quartile over the ring (robust to a neighbouring hold in the ring)
        along = fc.C[around] * np.cos(np.radians(_hue_diff(fc.H[around], hue)))
        c_wall = float(np.percentile(np.maximum(along, 0), 80)) if len(along) >= 30 else 4.0
        # ... and anywhere in the picture (mats, wood panels, a painted wall): the hold's
        # colour may only occur on a small part of it, so its 85th percentile is "background"
        step = max(1, int(np.sqrt(fc.h * fc.w / 250000)))          # subsample: ~250k pixels
        bg = (fc.person()[::step, ::step] == 0) & (seed[::step, ::step] == 0)
        along_all = fc.C[::step, ::step][bg] * np.cos(np.radians(_hue_diff(fc.H[::step, ::step][bg], hue)))
        c_wall = max(c_wall, float(np.percentile(np.maximum(along_all, 0), 85)))
        L10, L50, L90 = (float(v) for v in np.percentile(Lc, [10, 50, 90]))
        half = max(40.0, 0.75 * (L90 - L10))         # shading range on the clicked hold, padded
        L_lo, L_hi = L50 - half, L50 + half
        c_hold = float(np.percentile(Cc, 30))
        if c_hold < max(c_wall, 4.0) + PALE_MARGIN:
            # barely more colourful than the wall (cream, off-white, a washed-out pastel):
            # colour can't separate it, so treat it like a white / grey hold
            kind = "grey"
        else:
            model = dict(base, kind="colour", hue=round(hue, 1), hue_tol=round(hue_tol, 1),
                         c_hold=round(c_hold, 1), c_wall=round(max(c_wall, 4.0), 1),
                         c_hi=round(float(np.percentile(Cc, 95)), 1),
                         L_lo=round(L_lo, 1), L_hi=round(L_hi, 1))
    if kind == "grey":
        clean = seed > 0
        L0 = float(np.median(fc.L[clean]))
        Lw = float(np.median(fc.L[around])) if around.any() else float(np.median(fc.wall_L()[clean]))
        model = dict(base, kind="grey", L=round(L0, 1), L_wall=round(Lw, 1),
                     c_max=round(float(np.percentile(fc.C[clean], 90)), 1))
    model["lab"] = [round(float(np.median(fc.lab[..., i][clean])), 1) for i in range(3)]
    cs, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, max(1.0, 0.003 * cv2.arcLength(c, True)), True)
    model["seed"] = c.reshape(-1, 2).tolist()
    model["seed_area"] = area
    return model


def _colour_stats(fc, seed, hue):
    """Hue, hue tolerance and the 'clean' pixels (no chalk, highlight, shadow or other colour)
    of a segmented coloured hold."""
    clean = (seed > 0) & (_hue_diff(fc.H, hue) <= SEED_HUE) & (fc.C >= COLOURFUL * 0.75)
    if clean.sum() < 0.15 * max(seed.sum(), 1):
        clean = seed > 0
    hue = _circ_median(fc.H[clean])
    dev = np.abs((fc.H[clean] - hue + 180) % 360 - 180)
    hue_tol = float(np.clip(1.5 * np.percentile(dev, 85) + 4, 7, 20))
    return hue, hue_tol, clean


def _legacy_model(fc, s):
    """An old-style [L, a, b] sample (no clicked hold): match on its colour alone."""
    L, a, b = s
    if np.hypot(a - 128, b - 128) < GREY_CHROMA:
        return {"kind": "grey", "L": L, "L_wall": float(np.median(fc.L)), "c_max": GREY_CHROMA, "lab": list(s)}
    return {"kind": "colour", "hue": float(np.degrees(np.arctan2(b - 128, a - 128))), "hue_tol": 12.0,
            "c_hold": float(np.hypot(a - 128, b - 128)) * 0.8, "c_wall": 6.0, "lab": list(s)}


def models_for_frame(models, frame):
    """Models to use on `frame`: a model's clicked-hold outline only applies on the frame it
    was clicked on; elsewhere just its colour is used."""
    return [m if m.get("frame") == frame else {k: v for k, v in m.items() if k != "seed"} for m in models]


def _tight(tol):
    """Tolerance slider -> 0..1: where between the wall's colour (0) and the hold's own (1)
    the cut-off sits. Default 22 -> 0.55; a looser slider moves it towards the wall."""
    return float(np.clip(0.55 * 22 / max(tol, 1), 0.08, 0.95))


def match_model(fc, m, tol):
    """uint8 0/1 mask of pixels matching one model at this tolerance (cached on the frame)."""
    key = (tol, m["kind"], tuple(m.get("lab", ())), m.get("hue"), m.get("c_wall"), m.get("L_wall"))
    cache = fc.__dict__.setdefault("_match", {})
    if key not in cache:
        if len(cache) > 32:
            cache.clear()
        cache[key] = _match_model(fc, m, tol)
    return cache[key]


def _match_model(fc, m, tol):
    s = tol / 22
    t = _tight(tol)
    if m["kind"] == "colour":
        dh = _hue_diff(fc.H, m["hue"])
        ht = min(m["hue_tol"] * s, 45)
        c_min = max(6.0, m["c_wall"] + (m["c_hold"] - m["c_wall"]) * t)
        strong = (dh <= ht) & (fc.C >= c_min)
        if "c_hi" in m:              # a pastel hold isn't the saturated one of the same hue
            strong &= fc.C <= (m["c_hi"] + max(10.0, 0.5 * m["c_hi"])) * max(1.0, s)
        if "L_lo" in m:              # navy isn't light blue: a loose brightness band as well
            mid, half = (m["L_lo"] + m["L_hi"]) / 2, (m["L_hi"] - m["L_lo"]) / 2 * max(1.0, s)
            strong &= np.abs(fc.L - mid) <= half
        strong = strong.astype(np.uint8)
        # grow into the hold's shaded / washed-out parts, only where connected to a strong match
        weak = (dh <= 1.5 * ht) & (fc.C >= max(5.0, m["c_wall"] + (c_min - m["c_wall"]) * 0.5))
    else:
        # grey / white / black: how much brighter (darker) than the wall around it, plus a
        # loose band around the hold's own brightness and little colour
        L0, Lw = m["L"], m["L_wall"]
        d0 = L0 - Lw
        rel = (fc.L - fc.wall_L()) * (1 if d0 >= 0 else -1)
        c_ok = fc.C <= m["c_max"] + 4 * max(1.0, s)
        near = np.abs(fc.L - L0) <= max(30.0, 0.8 * abs(d0)) * max(1.0, s)
        strong = ((rel >= abs(d0) * t) & near & c_ok).astype(np.uint8)
        weak = (rel >= abs(d0) * t * 0.6) & near & c_ok
    n, lbl = cv2.connectedComponents((weak | (strong > 0)).astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lbl[strong > 0])] = True
    keep[0] = False
    grown = keep[lbl].astype(np.uint8)
    grown &= cv2.dilate(strong, _box(max(2, int(fc.mind * 0.015))))
    return grown


def find_holds_auto(lab, samples, tol=22, min_area=None, deleted=()):
    """Every hold matching one of the AUTO models.
    lab      FrameColour (or a to_lab() array - then no climber masking)
    samples  hold_model() dicts (old [L, a, b] samples still work, colour only)
    deleted  outlines (lists of [x, y]) the user deleted: those pixels are cut out, so a
             deleted hold stays deleted at any tolerance and never takes another hold with it."""
    fc = lab if isinstance(lab, FrameColour) else FrameColour(lab=lab)
    h, w = fc.h, fc.w
    models = [s if isinstance(s, dict) else _legacy_model(fc, s) for s in samples]
    models = [m for m in models if not m.get("reject")]
    if not models:
        return []
    frame_area = _fd(h, w)[1]
    if min_area is None:
        min_area = int(frame_area * 0.0003)
    col, grey, seeds = (np.zeros((h, w), np.uint8) for _ in range(3))
    seed_areas = []
    for m in models:
        mm = match_model(fc, m, tol)
        if m["kind"] == "colour":
            col |= mm
        else:
            grey |= mm
        if m.get("seed"):
            cv2.fillPoly(seeds, [np.array(m["seed"], np.int32)], 1)
            seed_areas.append(m.get("seed_area", 0))
    k_open = _disk(max(1, int(fc.mind / 500)))
    k_close = _disk(max(2, int(fc.mind * 0.006)))          # merges chalk / highlight fragments
    match = cv2.morphologyEx(cv2.morphologyEx(col | grey, cv2.MORPH_OPEN, k_open), cv2.MORPH_CLOSE, k_close)
    if fc.img is not None:                                   # the climber is never a hold
        match &= 1 - fc.person_wide()
    # the clicked holds: always in, and their blob may only grow a little past the clicked outline
    if seeds.any():
        n, lbl = cv2.connectedComponents(match | seeds, connectivity=8)
        seeded = np.zeros(n, bool)
        seeded[np.unique(lbl[seeds > 0])] = True
        seeded[0] = False
        near_seed = cv2.dilate(seeds, _box(max(3, int(0.15 * np.sqrt(max(seed_areas))))))
        sl = seeded[lbl]
        match[sl] = ((match & near_seed) | seeds)[sl]
    del_mask = np.zeros((h, w), np.uint8)
    for d in deleted:
        cv2.fillPoly(del_mask, [np.array(d, np.int32)], 1)
    if del_mask.any():
        del_mask = cv2.dilate(del_mask, _box(max(2, int(fc.mind * 0.006))))
        match &= 1 - del_mask
        seeds &= 1 - del_mask

    n, lbl, stats, _ = cv2.connectedComponentsWithStats(match, connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA]
    # Per-blob statistics below are ratios, so they're counted on every 2nd pixel (4x faster).
    q = 2 if fc.mind >= 600 else 1
    lq, mq = lbl[::q, ::q], match[::q, ::q]
    Cq, Hq = fc.C[::q, ::q], fc.H[::q, ::q]
    area_q = np.maximum(np.bincount(lq.ravel(), minlength=n), 1)
    # how colourful each blob's surroundings are (a ring a few px out): a highlight or chalk
    # patch on a coloured hold, or a shaded corner of a different-shade hold, is surrounded by
    # colour; a real hold is surrounded by wall.
    r1 = max(1, int(fc.mind * 0.005) // q)
    near = cv2.dilate(mq, _box(r1))
    lbl_far = cv2.dilate(lq.astype(np.float32), _box(3 * r1)).astype(np.int32)
    ring = (lbl_far > 0) & (near == 0)
    rl = lbl_far[ring]
    n_ring = np.bincount(rl, minlength=n)
    # ... counting only colour close to the blob's own hue (a neighbouring hold of another
    # colour doesn't make a blob a fragment), or any colour at all for grey blobs
    on = lq > 0
    lo = lq[on]
    a_, b_ = fc.lab[::q, ::q, 1][on] - 128, fc.lab[::q, ::q, 2][on] - 128
    blob_hue = np.degrees(np.arctan2(np.bincount(lo, b_, n), np.bincount(lo, a_, n)))
    # "colourful" = clearly above the wall's chroma for these colours (min 8)
    c_walls = [m["c_wall"] for m in models if m["kind"] == "colour"]
    c_ring = max(8.0, min(c_walls, default=8.0) + 4)
    Cr = Cq[ring]
    same = (Cr >= c_ring) & (_hue_diff(Hq[ring], blob_hue[rl]) <= 45)
    n_ring_col = np.bincount(rl[same], minlength=n)
    n_ring_any = np.bincount(rl[Cr >= 1.5 * COLOURFUL], minlength=n)
    n_seed = np.bincount(lbl[seeds > 0], minlength=n)            # full res: a tiny seed must count
    n_person = (np.bincount(lq[fc.person()[::q, ::q] > 0], minlength=n) if fc.img is not None
                else np.zeros(n, int))
    n_grey = (np.bincount(lq[(grey[::q, ::q] > 0) & (col[::q, ::q] == 0)], minlength=n) if grey.any()
              else np.zeros(n, int))
    # grey blobs: a real white / black hold stands out from the wall about as much as the
    # clicked one does over its whole area; chalk smudges and lighter wall patches only just
    # clear the threshold. Mean contrast per blob, against the weakest grey model.
    greys = [m for m in models if m["kind"] == "grey"]
    if greys:
        d0 = min((m["L"] - m["L_wall"] for m in greys), key=abs)
        rel = (fc.L[::q, ::q] - fc.wall_L()[::q, ::q])[on] * (1 if d0 >= 0 else -1)
        mean_rel = np.bincount(lo, rel, n) / area_q
        need_rel = abs(d0) * min(0.9, _tight(tol) + 0.2)
    max_area = max(MAX_AREA_FRAC * frame_area, SEED_AREA_MULT * max(seed_areas, default=0))
    holds = []
    for i in range(1, n):
        is_seed = n_seed[i] > 0
        if not is_seed:
            if area[i] < min_area * 0.5 or area[i] > max_area * 1.2 or n_person[i] > PERSON_OVERLAP * area_q[i]:
                continue
            grey_blob = n_grey[i] > 0.5 * area_q[i]     # a highlight / chalk patch on a coloured hold
            if (n_ring_any if grey_blob else n_ring_col)[i] > FRAGMENT_RING * max(n_ring[i], 1):
                continue
            if grey_blob and greys and mean_rel[i] < need_rel:
                continue
        x, y, bw, bh = stats[i, :4]
        sub = (lbl[y:y + bh, x:x + bw] == i).astype(np.uint8)
        cs, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea) + np.array([x, y])
        a = cv2.contourArea(c)                               # holes filled
        strict = n_grey[i] > 0.5 * area_q[i]                 # grey matches: wall patches look alike
        if not is_seed and not (min_area <= a <= max_area and _hold_shaped(c, a, strict)):
            continue
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
    fc0 = FrameColour(img)
    S = {"img": img, "fc": fc0, "lab": fc0.lab, "frame": start_frame, "mode": "AUTO",
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
                if any(inside(hd, (fx, fy)) for hd in S["holds"]):
                    S["msg"] = "That hold is already selected - nothing added."
                else:
                    model = hold_model(S["fc"], fx, fy, frame=S["frame"])
                    if model.get("reject"):
                        S["msg"] = model["message"]
                    else:
                        S["samples"].append(model)
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
                hits = [hd for hd in S["holds"] if inside(hd, (fx, fy))]
                if hits:                                # remember its OUTLINE, not just the point
                    hit = min(hits, key=lambda hd: hd["area"])
                    S["deleted"].append({"pt": [fx, fy], "contour": hit["contour"]})
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
                    fcf = FrameColour(f)
                    S.update(img=f, fc=fcf, lab=fcf.lab, frame=f_idx, dirty=True,
                             vols_auto=cvol.detect_volumes(f))
        tol = max(1, cv2.getTrackbarPos("Colour tol", WIN))
        min_area = int(_fd(h, w)[1] * cv2.getTrackbarPos("Min size", WIN) / 10000)
        auto_key = (S["frame"], len(S["samples"]), len(S["deleted"]), tol, min_area)
        if auto_key != S["auto_key"]:
            S["auto"] = find_holds_auto(S["fc"], models_for_frame(S["samples"], S["frame"]), tol, min_area,
                                        deleted=[d["contour"] for d in S["deleted"]])
            S["auto_key"] = auto_key
            S["dirty"] = True
        if S["dirty"]:
            S["holds"] = combine(S["auto"], S["manual"], [], S["starts"], S["end"])
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
                    "colour_samples_lab": [m["lab"] for m in S["samples"]], "colour_models": S["samples"],
                    "clicks": S["clicks"], "deleted_points": [d["pt"] for d in S["deleted"]],
                    "deleted_outlines": [d["contour"] for d in S["deleted"]],
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
        fc = FrameColour(img)
        lab = fc.lab
        clicks = [[int(v) for v in p.split(",")] for p in args.pick.split(";")]
        models = []
        for x, y in clicks:                  # same rules as clicking in the window
            if any(inside(hd, (x, y)) for hd in find_holds_auto(fc, models)):
                print(f"  click {x},{y}: that hold is already selected - nothing added.")
                continue
            m = hold_model(fc, x, y, frame=args.frame)
            if m.get("reject"):
                print(f"  click {x},{y}: {m['message']}")
            else:
                models.append(m)
        holds = combine(find_holds_auto(fc, models), [], [])
        samples = [m["lab"] for m in models]
        vols, _ = filter_coloured_volumes(lab, cvol.detect_volumes(img), holds)
        vols = cvol.cut_foreign_holds(vols, img, route_mask(holds, img.shape),
                                      int(img.shape[0] * img.shape[1] * 0.0003))
        info = {"frame": args.frame, "colour_samples_lab": samples, "colour_models": models, "clicks": clicks,
                "volumes": vols}
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
