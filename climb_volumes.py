"""
climb_volumes.py - find climbing VOLUMES (the big grey geometric shapes bolted to
the wall) in a frame. Volumes are always allowed for every route, so the
analyzer treats them as "on route".

How it works (on a half-size copy of the frame, for speed):
  1. Even out the lighting, smooth texture while keeping edges sharp (bilateral filter).
  2. Volumes are grey, so coloured pixels (holds) are set aside.
  3. Find edges, then split the grey area into regions bounded by those edges.
     The plain wall is the big region(s) at the wall's usual brightness.
  4. A volume FACE is a region that
       - is a polygon with few corners (3-7) and straight sides  -> "sharp corners"
       - is solid (convex-ish), mid-sized,
       - and is lighter/darker than the wall (faces are angled, so they catch light differently).
  5. Faces that touch each other are merged into one volume; its outline is the
     convex hull of its faces.
  6. Holds sitting on a volume that are NOT the route's colour are cut out of the
     volume, so touching them still counts as off-route.
"""

import cv2
import numpy as np

WORK_W = 540            # processing width for one video frame (pixels)

# Size of ONE video frame (h, w). Thresholds are relative to a frame, so they still work
# on a stitched panorama of the whole wall (which is much bigger than a frame).
FRAME = None


def frame_dims(H, W):
    return FRAME if FRAME else (H, W)
GREY_CHROMA = 20        # Lab chroma below this = grey (wall / volume / white-grey-black holds)


def _flatten(img, ref_min=None):
    h, w = img.shape[:2]
    f = img.astype(np.float32) + 1
    small = cv2.resize(f, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
    bg = cv2.GaussianBlur(small, (0, 0), (ref_min or min(h, w)) * 0.2 / 8)
    bg = cv2.resize(bg.mean(axis=2), (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
    return np.clip(f / bg * bg.mean(), 0, 255).astype(np.uint8)


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _face_candidates(img_small, fh, fw):
    """Return (faces, debug) where faces are dicts with a filled mask contour.
    fh, fw = size of one video frame at this scale (thresholds are relative to it)."""
    h, w = img_small.shape[:2]
    A = fh * fw
    flat = _flatten(img_small, min(fh, fw))
    sm = cv2.bilateralFilter(flat, 9, 35, 7)
    lab = cv2.cvtColor(sm, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    chroma = np.hypot(lab[..., 1].astype(np.float32) - 128, lab[..., 2].astype(np.float32) - 128)
    colourful = (chroma > GREY_CHROMA).astype(np.uint8)
    colourful = cv2.dilate(colourful, np.ones((5, 5), np.uint8))

    edges = cv2.Canny(cv2.GaussianBlur(L, (3, 3), 0), 18, 45)
    # small specks (bolt holes, texture) shouldn't count as edges between regions
    n, lbl, st, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 25
    edges = (keep[lbl] * 255).astype(np.uint8)
    seams = _find_seams(edges, fh, fw)
    lines = _straight_grey_lines(edges, chroma, fh, fw, seams)
    # draw straight segments back in (slightly extended) to close small gaps in creases
    bar = edges.copy()
    for (x1, y1), (x2, y2) in lines:
        d = np.hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / d, (y2 - y1) / d
        cv2.line(bar, (int(x1 - 6 * ux), int(y1 - 6 * uy)), (int(x2 + 6 * ux), int(y2 + 6 * uy)), 255, 1)
    barrier = cv2.dilate(bar, np.ones((3, 3), np.uint8))

    line_mask = np.zeros((h, w), np.uint8)              # where the straight volume edges are
    for (x1, y1), (x2, y2) in lines:
        cv2.line(line_mask, (int(x1), int(y1)), (int(x2), int(y2)), 1, 7)

    free = ((barrier == 0) & (colourful == 0)).astype(np.uint8)
    n, lbl, st, _ = cv2.connectedComponentsWithStats(free, connectivity=4)

    grey_px = (colourful == 0)
    wall_L = float(np.median(L[grey_px])) if grey_px.any() else float(np.median(L))

    faces = []
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA]
        if not (0.0008 * A <= area <= 0.15 * A):
            continue
        comp = (lbl == i).astype(np.uint8)
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=cv2.contourArea)
        filled_area = cv2.contourArea(c)
        if filled_area <= 0:
            continue
        # Judge the face by its outer shape: holds bolted onto a face cut notches
        # out of it, so count hold (colourful) pixels inside the hull as face too.
        hull = cv2.convexHull(c)
        hull_area = max(cv2.contourArea(hull), 1)
        hm = np.zeros((h, w), np.uint8)
        cv2.drawContours(hm, [hull], -1, 1, -1)
        covered = ((comp | colourful) & hm).sum() / hull_area
        peri = cv2.arcLength(hull, True)
        poly = cv2.approxPolyDP(hull, 0.025 * peri, True)
        nv = len(poly)
        solidity = covered
        fit = abs(cv2.contourArea(poly) - hull_area) / hull_area
        dL = abs(float(np.median(L[comp > 0])) - wall_L)
        # "sharp edges": how much of the face's outline runs along straight edges
        per = cv2.approxPolyDP(hull, 1, True).reshape(-1, 2).astype(np.float32)
        samples = []
        for i_ in range(len(per)):
            p0, p1 = per[i_], per[(i_ + 1) % len(per)]
            n_ = max(2, int(np.hypot(*(p1 - p0)) / 3))
            samples += [p0 + (p1 - p0) * t for t in np.linspace(0, 1, n_, endpoint=False)]
        samples = np.clip(np.array(samples).round().astype(int), [0, 0], [w - 1, h - 1])
        support = float(line_mask[samples[:, 1], samples[:, 0]].mean()) if len(samples) else 0.0
        # touching the frame edge -> probably a cut-off piece of wall
        x, y, bw, bh = cv2.boundingRect(c)
        on_border = (x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
                     or (img_small[y:y + bh, x:x + bw] == 0).all(axis=2).mean() > 0.02)   # touches unseen (black) area
        faces.append({"contour": hull, "poly": poly, "nv": nv, "solidity": solidity, "fit": fit,
                      "dL": dL, "support": support, "area": filled_area, "on_border": on_border})
    return faces, {"wall_L": wall_L, "edges": edges, "lab": lab, "chroma": chroma,
                   "colourful": colourful, "lines": lines, "seams": seams, "A": A}


def _is_face(f, A):
    if not (3 <= f["nv"] <= 7 and f["fit"] < 0.2 and not f["on_border"] and f["area"] <= 0.12 * A):
        return False
    # either a different shade from the wall, or outlined by straight edges on most sides
    shaded = f["dL"] > 7 and f["solidity"] > 0.88
    outlined = f["support"] > 0.7 and f["solidity"] > 0.7
    return shaded or outlined


def _find_seams(edges, h, w):
    """Wall panel seams: (near) horizontal/vertical straight lines running much
    further than any volume edge could. Returned as (rho, theta) lines."""
    found = cv2.HoughLines(edges, 1, np.pi / 360, threshold=int(0.45 * min(h, w)))
    seams = []
    if found is None:
        return seams
    for rho, theta in np.asarray(found).reshape(-1, 2):
        deg = np.degrees(theta) % 180
        if min(deg, abs(deg - 90), 180 - deg) < 5:
            seams.append((float(rho), float(theta)))
    return seams


def _on_seam(a, b, seams, tol=4.0):
    for rho, theta in seams:
        c, s_ = np.cos(theta), np.sin(theta)
        if abs(a[0] * c + a[1] * s_ - rho) < tol and abs(b[0] * c + b[1] * s_ - rho) < tol:
            return True
    return False


def _straight_grey_lines(edges, chroma, h, w, seams=()):
    """Long straight edges with grey on BOTH sides (volume creases / outlines).
    Hold outlines are curved or have colour on one side; wall panel seams are
    long, perfectly horizontal/vertical lines across the frame - all excluded."""
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=18, minLineLength=28, maxLineGap=5)
    if segs is None:
        return []
    out = []
    for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        ang = abs(np.degrees(np.arctan2(dy, dx))) % 180
        axis = min(ang, abs(ang - 90), abs(180 - ang)) < 2.5
        if (axis and length > 0.25 * min(h, w)) or _on_seam((x1, y1), (x2, y2), seams):
            continue                                   # panel seam
        nx, ny = -dy / length, dx / length              # unit normal
        ih, iw = chroma.shape[:2]
        ok = True
        for t in (0.25, 0.5, 0.75):
            mx, my = x1 + t * dx, y1 + t * dy
            for sgn in (1, -1):
                px, py = int(mx + sgn * 5 * nx), int(my + sgn * 5 * ny)
                if 0 <= px < iw and 0 <= py < ih and chroma[py, px] > GREY_CHROMA:
                    ok = False
        if ok:
            out.append(((float(x1), float(y1)), (float(x2), float(y2))))
    return out


def _grow_with_lines(pts, lines, seed_area):
    """Extend a volume's outline along straight edges that start on it."""
    pts = np.asarray(pts, np.float32)
    seed_r = np.sqrt(seed_area / np.pi)
    center = pts.mean(axis=0)
    used = set()
    for _ in range(5):
        hull = cv2.convexHull(pts)
        changed = False
        for k, (a, b) in enumerate(lines):
            if k in used:
                continue
            da = cv2.pointPolygonTest(hull, a, True)
            db = cv2.pointPolygonTest(hull, b, True)
            for near, far, dn, dfar in ((a, b, da, db), (b, a, db, da)):
                if dn > -6 and dfar < -6:               # starts on the outline, leaves it
                    if np.hypot(far[0] - center[0], far[1] - center[1]) > 3.0 * seed_r:
                        continue                         # wandered too far: not this volume
                    pts = np.vstack([pts, np.float32([far])])
                    used.add(k)
                    changed = True
                    break
                if dn > -6 and dfar >= -6:
                    used.add(k)                          # already inside
        if not changed:
            break
    return pts


def detect_volumes(img):
    """Find volumes in a full-size BGR frame. Returns a list of volume dicts:
       {"id", "contour" (outline, full-res pixels), "faces": [face outlines]}"""
    H, W = img.shape[:2]
    FH, FW = frame_dims(H, W)
    s = WORK_W / FW
    small = cv2.resize(img, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    A = FH * FW * s * s
    cands, dbg = _face_candidates(small, FH * s, FW * s)
    faces = [f for f in cands if _is_face(f, A)]
    if not faces:
        return []
    # A clean polygon touching a confirmed face is another face of the same volume,
    # even if it happens to be lit exactly like the wall.
    def _touch_mask(f):
        m = np.zeros((h, w), np.uint8)
        cv2.drawContours(m, [f["contour"]], -1, 1, -1)
        return cv2.dilate(m, np.ones((9, 9), np.uint8))
    confirmed = np.zeros((h, w), np.uint8)
    for f in faces:
        confirmed |= _touch_mask(f)
    for f in cands:
        if any(f is g for g in faces):
            continue
        if (3 <= f["nv"] <= 6 and f["solidity"] > 0.95 and f["fit"] < 0.1 and not f["on_border"]
                and f["area"] <= 0.04 * A):
            m = np.zeros((h, w), np.uint8)
            cv2.drawContours(m, [f["contour"]], -1, 1, -1)
            if (m & confirmed).any():
                faces.append(f)

    # merge faces that touch (they share an edge, separated only by the thin edge line)
    masks = []
    for f in faces:
        m = np.zeros((h, w), np.uint8)
        cv2.drawContours(m, [f["contour"]], -1, 1, -1)
        masks.append(cv2.dilate(m, np.ones((15, 15), np.uint8)))   # a hold between faces shouldn't split a volume
    uf = _UF(len(faces))
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            if (masks[i] & masks[j]).any():
                uf.union(i, j)
    groups = {}
    for i in range(len(faces)):
        groups.setdefault(uf.find(i), []).append(i)

    lines = dbg["lines"]
    vols = []
    for idx in groups.values():
        pts = np.vstack([faces[i]["contour"].reshape(-1, 2) for i in idx])
        seed_area = cv2.contourArea(cv2.convexHull(pts))
        pts = _grow_with_lines(pts, lines, seed_area)
        hull = cv2.convexHull(pts.astype(np.float32)).astype(np.int32)
        hull_area = cv2.contourArea(hull)
        if hull_area < 0.004 * A:                     # a lone small face: more likely a hold
            continue
        outline = cv2.approxPolyDP(hull, 0.01 * cv2.arcLength(hull, True), True)
        vols.append({
            "contour": (outline.reshape(-1, 2) / s).round().astype(int).tolist(),
            "faces": [(faces[i]["poly"].reshape(-1, 2) / s).round().astype(int).tolist() for i in idx],
        })
    vols.sort(key=lambda v: -np.mean([p[1] for p in v["contour"]]))   # bottom to top
    for i, v in enumerate(vols, 1):
        v["id"] = f"V{i}"
    return vols


def cut_foreign_holds(vols, img, route_mask, min_area):
    """Holds sitting on a volume that are NOT on the route (not in route_mask) are
    cut out of the volume, so touching them still counts as off-route.
    Finds them as blobs that are either coloured, or clearly lighter/darker than
    the volume face around them (catches white/black holds on grey volumes).
    Adds v["exclude"] = [outlines] to each volume."""
    H, W = img.shape[:2]
    FMIN = min(frame_dims(H, W))
    lab = cv2.cvtColor(cv2.GaussianBlur(_flatten(img, FMIN), (5, 5), 0), cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    k = max(3, int(round(FMIN / 250)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    for v in vols:
        hull = np.array(v["contour"], np.int32)
        vm = np.zeros((H, W), np.uint8)
        cv2.fillPoly(vm, [hull], 1)
        x, y, bw, bh = cv2.boundingRect(hull)
        pad = int(0.03 * FMIN)
        x0, y0, x1, y1 = max(0, x - pad), max(0, y - pad), min(W, x + bw + pad), min(H, y + bh + pad)
        sub = (slice(y0, y1), slice(x0, x1))
        Ls = L[sub].astype(np.float32)
        local = cv2.medianBlur(L[sub], 2 * int(0.03 * FMIN) + 1).astype(np.float32)
        # Volumes can be painted (blue, wood...), so "a different hold" means a colour that
        # differs from THIS volume's own colour, not just "not grey".
        a_ = lab[sub][..., 1].astype(np.float32)
        b_ = lab[sub][..., 2].astype(np.float32)
        inside = (vm[sub] > 0) & (route_mask[sub] == 0)
        va, vb = (float(np.median(a_[inside])), float(np.median(b_[inside]))) if inside.any() else (128.0, 128.0)
        off_colour = np.hypot(a_ - va, b_ - vb) > max(GREY_CHROMA, 18)
        allowed = cv2.dilate(vm[sub], kernel)
        # route holds (and a small ring around them) can never be cut out
        allowed &= 1 - cv2.dilate((route_mask[sub] > 0).astype(np.uint8), kernel, iterations=3)
        band_px = max(3, int(0.025 * FMIN))
        rim = vm[sub] - cv2.erode(vm[sub], np.ones((2 * band_px + 1, 2 * band_px + 1), np.uint8))
        coloured = off_colour
        excl = []
        # Look for off-colour blobs and light/dark blobs SEPARATELY, so a coloured hold can't get
        # glued to the shadow line along a volume's edge (which would then look like a strip).
        for cue in (off_colour, np.abs(Ls - local) > 28):
            blob = cue.astype(np.uint8) & allowed
            blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, kernel)         # drop thin crease lines
            blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel, iterations=2)
            n, lbl, st, _ = cv2.connectedComponentsWithStats(blob)
            for i in range(1, n):
                if st[i, cv2.CC_STAT_AREA] < min_area:
                    continue
                comp = lbl == i
                # a grey wedge in the volume's rim is wall peeking into the outline's corner, not a hold
                if coloured[comp].mean() < 0.3 and rim[comp].mean() > 0.5:
                    continue
                cs, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                c = max(cs, key=cv2.contourArea)
                # holds are compact blobs; a thin ring/strip is shading around something else
                if cv2.contourArea(c) / max(cv2.contourArea(cv2.convexHull(c)), 1) < 0.6:
                    continue
                (_, _), (rw, rh), _ = cv2.minAreaRect(c)
                if min(rw, rh) == 0 or max(rw, rh) / min(rw, rh) > 3.0:   # a shadow strip along an edge
                    continue
                if cv2.contourArea(c) > 0.15 * cv2.contourArea(hull):
                    continue                                               # a whole face of a two-tone volume, not a hold
                c = c + np.array([x0, y0])
                m = cv2.moments(c)
                cxy = (m["m10"] / max(m["m00"], 1), m["m01"] / max(m["m00"], 1))
                if any(cv2.pointPolygonTest(np.array(e, np.int32), cxy, False) >= 0 for e in excl):
                    continue                                               # already found by the other cue
                excl.append(c.reshape(-1, 2).tolist())
        v["exclude"] = excl
    return vols


def on_volume(vols, pt, margin=0.0):
    """Id of the volume under pt (not inside a cut-out off-route hold), else ''."""
    p = (float(pt[0]), float(pt[1]))
    for v in vols:
        if cv2.pointPolygonTest(np.array(v["contour"], np.int32), p, True) >= -margin:
            if any(cv2.pointPolygonTest(np.array(e, np.int32), p, False) >= 0 for e in v.get("exclude", [])):
                return ""
            return v["id"]
    return ""


def volume_from_click(img, x, y):
    """Outline the volume under a click (for volumes the auto-detect missed):
    takes the face under the click plus clean polygon faces touching it, then
    follows straight edges outward. Returns a volume dict or None."""
    H, W = img.shape[:2]
    FH, FW = frame_dims(H, W)
    s = WORK_W / FW
    small = cv2.resize(img, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    A = FH * FW * s * s
    cands, dbg = _face_candidates(small, FH * s, FW * s)
    p = (float(x * s), float(y * s))
    seed = [f for f in cands if not f["on_border"] and f["area"] <= 0.12 * A
            and cv2.pointPolygonTest(f["contour"], p, False) >= 0]
    if not seed:
        return None
    faces = [min(seed, key=lambda f: f["area"])]
    for _ in range(2):                                   # add touching polygon faces
        conf = np.zeros((h, w), np.uint8)
        for f in faces:
            cv2.drawContours(conf, [f["contour"]], -1, 1, -1)
        conf = cv2.dilate(conf, np.ones((9, 9), np.uint8))
        for f in cands:
            if any(f is g for g in faces):
                continue
            if 3 <= f["nv"] <= 6 and f["solidity"] > 0.9 and f["fit"] < 0.15 and not f["on_border"] \
                    and f["area"] <= 0.04 * A:
                m = np.zeros((h, w), np.uint8)
                cv2.drawContours(m, [f["contour"]], -1, 1, -1)
                if (m & conf).any():
                    faces.append(f)
    pts = np.vstack([f["contour"].reshape(-1, 2) for f in faces])
    pts = _grow_with_lines(pts, dbg["lines"], cv2.contourArea(cv2.convexHull(pts)))
    hull = cv2.convexHull(pts.astype(np.float32)).astype(np.int32)
    outline = cv2.approxPolyDP(hull, 0.01 * cv2.arcLength(hull, True), True)
    return {"contour": (outline.reshape(-1, 2) / s).round().astype(int).tolist(),
            "faces": [(f["poly"].reshape(-1, 2) / s).round().astype(int).tolist() for f in faces],
            "source": "manual"}


def renumber(vols):
    vols.sort(key=lambda v: -np.mean([p[1] for p in v["contour"]]))   # bottom to top
    for i, v in enumerate(vols, 1):
        v["id"] = f"V{i}"
    return vols


def inside_volume(v, pt):
    return cv2.pointPolygonTest(np.array(v["contour"], np.int32), (float(pt[0]), float(pt[1])), False) >= 0
