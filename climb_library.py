"""
climb_library.py - a library of recorded climbs.

After each analysis BetaVision can keep one small JSON "climb record" (no video, no frames):
the route as marked, the sequence of hold contacts, possible off-route touches, whether and
when the climber topped out, and joint-angle ranges. Records live in a folder that survives
restarts, so later features (beta suggestions) have something to learn from.

  default folder   ~/.climb_analyzer/library        (override: --library, or BETAVISION_LIBRARY)
  turn it off      climb_analyze.py --no-record     web app: untick "Save to my library"
                   (or BETAVISION_RECORD=0 to make "off" the web app's default)
  read it          python climb_library.py list
                   python climb_library.py route "Blue wall V3"

A record is one <id>.json file with "schema_version". Records of the same named route share a
route id (derived from the name), so unnamed climbs are stored with route id null and never
grouped. Holds are stored in base-frame pixels AND relative to the route's start-to-end extent
("norm"), so records of one route filmed from different spots can still be lined up.

Nothing here opens a window or imports the pose model, so it is cheap to import and test.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

SCHEMA_VERSION = 1
MAX_RECORD_BYTES = 300_000          # a record is a few tens of KB; this is the hard ceiling
MAX_NAME = 80
LIMB_NAMES = {"l_hand": "Left hand", "r_hand": "Right hand", "l_foot": "Left foot", "r_foot": "Right foot"}
ANGLE_NAMES = {"elbow": "Elbows", "knee": "Knees", "hip": "Hips"}
SAME_HOLD = 0.06                    # holds of different records closer than this (route extents) are one hold


class RecordError(ValueError):
    """A file isn't a usable climb record."""


class UnsupportedSchema(RecordError):
    """The record was written by a newer BetaVision than this one."""


# ---------------------------------------------------------------- where the library is
def library_dir(path=None):
    """The folder records are kept in: path, else $BETAVISION_LIBRARY, else ~/.climb_analyzer/library."""
    if path:
        return Path(path).expanduser()
    env = os.environ.get("BETAVISION_LIBRARY")
    return Path(env).expanduser() if env else Path.home() / ".climb_analyzer" / "library"


def recording_default():
    """False when BETAVISION_RECORD is 0 / false / no / off (the web app's default for its checkbox)."""
    return os.environ.get("BETAVISION_RECORD", "1").strip().lower() not in ("0", "false", "no", "off")


# ---------------------------------------------------------------- route names and ids
def clean_name(name):
    """A route name as typed -> tidy text ("" if none). Raises ValueError if it is too long."""
    name = " ".join(str(name or "").split())
    if len(name) > MAX_NAME:
        raise ValueError(f"Route names can be at most {MAX_NAME} characters.")
    return name


def route_id(name):
    """Shared id of a named route (same for any capitalisation / spacing); None when unnamed."""
    name = clean_name(name)
    if not name:
        return None
    return "r-" + hashlib.sha1(name.casefold().encode("utf-8")).hexdigest()[:10]


# ---------------------------------------------------------------- what an analysis produced
def contact_moves(limb_holds, fps, holds=()):
    """Per-frame {limb: hold or volume id or ""} -> hold contacts of at least ~0.2 s, as
    [{t, dur, limb (l_hand ...), hold, volume (True for a volume id), role}] by start time."""
    min_run = max(3, int(round(0.2 * fps)))        # shorter contacts are tracking jitter
    moves = []
    for limb in LIMB_NAMES:
        run_hold, run_start = "", 0
        for i, lh in enumerate(list(limb_holds) + [{}]):
            hid = lh.get(limb, "")
            if hid != run_hold:
                if run_hold != "" and i - run_start >= min_run:
                    moves.append({"t": round(run_start / fps, 2), "dur": round((i - run_start) / fps, 2),
                                  "limb": limb, "hold": run_hold, "volume": isinstance(run_hold, str)})
                run_hold, run_start = hid, i
    moves.sort(key=lambda m: (m["t"], m["limb"]))
    roles = {hd["id"]: hd.get("role", "") for hd in holds}
    for m in moves:
        m["role"] = "" if m["volume"] else roles.get(m["hold"], "")
    return moves


def angle_ranges(angles):
    """Per-frame angle dicts ({"l_elbow_deg": ...}) -> [{joint, min, median, max}] over both sides."""
    out = []
    for key, label in ANGLE_NAMES.items():
        vals = [a[k] for a in angles for k in (f"l_{key}_deg", f"r_{key}_deg")
                if k in a and not math.isnan(a[k])]
        if vals:
            out.append({"joint": label, "min": round(min(vals)), "median": round(statistics.median(vals)),
                        "max": round(max(vals))})
    return out


def off_route_touches(res):
    """analyze_video's off_route events -> [{t, dur, limb (l_hand ...), hold, on_volume}]."""
    vol = {hd["id"]: hd.get("volume") for hd in res.get("off_route_holds", [])}
    return [{"t": e["t"], "dur": e["dur"], "limb": e["limb"], "hold": e["hold"], "on_volume": vol.get(e["hold"])}
            for e in res.get("off_route", [])]


# ---------------------------------------------------------------- building a record
def _simplify(contour, max_points):
    """Fewer outline points (Douglas-Peucker) so a record stays small. Returns [[x, y], ...]."""
    pts = np.asarray(contour, np.float32).reshape(-1, 1, 2)
    if len(pts) <= 4:
        return [[int(round(x)), int(round(y))] for x, y in pts.reshape(-1, 2)]
    peri = cv2.arcLength(pts, True)
    eps = max(1.0, 0.004 * peri)
    out = cv2.approxPolyDP(pts, eps, True)
    while len(out) > max_points:
        eps *= 1.4
        out = cv2.approxPolyDP(pts, eps, True)
    return [[int(round(x)), int(round(y))] for x, y in out.reshape(-1, 2)]


def _centre(item):
    c = item.get("center")
    if c is None:
        c = np.mean(np.asarray(item["contour"], float), axis=0)
    return [float(c[0]), float(c[1])]


def normalisation(holds):
    """How to express base-frame pixels relative to the route: origin and scale (pixels per
    route extent). Start-to-end when the route has both tags (origin = the start holds' middle,
    scale = distance to the end hold), else the holds' bounding box (origin = bottom middle).
    None when there are no holds."""
    if not holds:
        return None
    cs = [_centre(h) for h in holds]
    starts = [_centre(h) for h in holds if h.get("role") == "start"]
    ends = [_centre(h) for h in holds if h.get("role") == "end"]
    if starts and ends:
        s = np.mean(starts, axis=0)
        e = np.mean(ends, axis=0)
        dist = float(np.hypot(*(e - s)))
        if dist >= 1:
            return {"basis": "start_end", "origin": [round(float(s[0]), 1), round(float(s[1]), 1)],
                    "scale": round(dist, 1)}
    xs, ys = [c[0] for c in cs], [c[1] for c in cs]
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    return {"basis": "bbox", "origin": [round((min(xs) + max(xs)) / 2, 1), round(max(ys), 1)],
            "scale": round(float(scale), 1)}


def _to_norm(p, norm):
    return [round((p[0] - norm["origin"][0]) / norm["scale"], 4), round((p[1] - norm["origin"][1]) / norm["scale"], 4)]


def route_record(holds, vols):
    """The route as stored: holds / volumes with simplified outlines (base-frame pixels), centres,
    and the same centres relative to the route ("nc": x, y in route extents; y grows downwards)."""
    norm = normalisation(holds)
    out_h = []
    for h in holds:
        c = _centre(h)
        d = {"id": h["id"], "role": h.get("role", ""), "center": [round(c[0], 1), round(c[1], 1)],
             "contour": _simplify(h["contour"], 48)}
        if norm:
            d["nc"] = _to_norm(c, norm)
            d["nsize"] = round(math.sqrt(max(cv2.contourArea(np.asarray(h["contour"], np.float32)), 0)) / norm["scale"], 4)
        out_h.append(d)
    out_v = []
    for v in vols:
        c = _centre(v)
        d = {"id": v["id"], "center": [round(c[0], 1), round(c[1], 1)], "contour": _simplify(v["contour"], 64)}
        if norm:
            d["nc"] = _to_norm(c, norm)
        out_v.append(d)
    return {"norm": norm, "holds": out_h, "volumes": out_v}


def build_record(res, holds, vols, video_name, base_frame=0, frame_size=None, route_name="",
                 video_file_date=None, now=None):
    """One climb record (a plain dict, JSON-ready) from analyze_video's result `res` and the
    route (holds, vols in base-frame pixels). Raises ValueError for a route name that is too long."""
    now = now or datetime.now(timezone.utc)
    name = clean_name(route_name)
    fps = res["fps"]
    topped = res.get("topped_s")
    rec = {
        "schema_version": SCHEMA_VERSION,
        "id": now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6],
        "created": now.isoformat(timespec="seconds"),
        "route": {"id": route_id(name), "name": name or None},
        "video": {"name": str(video_name), "file_date": video_file_date, "frames": int(res["frames"]),
                  "fps": round(float(fps), 2), "duration_s": round(res["frames"] / fps, 2),
                  "base_frame": int(base_frame), "frame_size": list(frame_size) if frame_size else None},
        "wall": route_record(holds, vols),
        "moves": contact_moves(res["limb_holds"], fps, holds),
        "off_route": {"checked": bool(res.get("off_route_enabled")), "touches": off_route_touches(res)},
        "topped": {"has_end": bool(res.get("has_end")), "topped": topped is not None,
                   "t": round(float(topped), 2) if topped is not None else None},
        "angles": angle_ranges(res["angles"]),
    }
    return rec


# ---------------------------------------------------------------- writing
def _json(rec):
    return json.dumps(rec, separators=(",", ":"), default=_np_default)


def _np_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"can't store {type(o).__name__}")


def save_record(rec, library=None):
    """Write the record into the library folder (created if needed) and return its path. The file
    appears all at once (written to a temp name, then renamed). If the record somehow passes
    MAX_RECORD_BYTES, the outlines are dropped (centres stay) so it never grows without limit."""
    text = _json(rec)
    if len(text.encode("utf-8")) > MAX_RECORD_BYTES:
        slim = json.loads(text)
        for h in slim["wall"]["holds"] + slim["wall"]["volumes"]:
            h.pop("contour", None)
        slim["wall"]["outlines_dropped"] = True
        text = _json(slim)
        if len(text.encode("utf-8")) > MAX_RECORD_BYTES:
            raise RecordError("This record is too large to store.")
    folder = library_dir(library)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{rec['id']}.json"
    tmp = folder / f".{rec['id']}.tmp"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def record_analysis(res, holds, vols, video_name, library=None, **kw):
    """build_record + save_record. Returns (path, record), or (None, None) when no route was
    marked (a body-only analysis has nothing to learn from)."""
    if not holds:
        return None, None
    rec = build_record(res, holds, vols, video_name, **kw)
    return save_record(rec, library), rec


# ---------------------------------------------------------------- reading
def _migrate(rec):
    """Bring an older record up to SCHEMA_VERSION (one step per version; none exist yet)."""
    while rec["schema_version"] < SCHEMA_VERSION:
        step = MIGRATIONS.get(rec["schema_version"])
        if step is None:
            raise RecordError(f"no way to read schema version {rec['schema_version']}")
        rec = step(rec)
    return rec


MIGRATIONS = {}                     # {old version: function(record) -> record of the next version}


def load_record(path):
    """Read one record file. Raises RecordError (UnsupportedSchema if it is from a newer
    BetaVision) with a message that says why; never returns half a record."""
    path = Path(path)
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise RecordError(f"{path.name}: can't read it ({e})")
    v = rec.get("schema_version") if isinstance(rec, dict) else None
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise RecordError(f"{path.name}: not a climb record (no schema_version)")
    if v > SCHEMA_VERSION:
        raise UnsupportedSchema(f"{path.name}: written by a newer BetaVision (schema {v}, this reads up to "
                                f"{SCHEMA_VERSION}). Update BetaVision to use it.")
    rec = _migrate(rec)
    for key in ("id", "route", "video", "wall", "moves", "topped"):
        if key not in rec:
            raise RecordError(f"{path.name}: missing '{key}'")
    return rec


def read_library(library=None):
    """All readable records, oldest first, and a list of problems for the files that weren't
    ([] when all fine). A missing folder is an empty library."""
    folder = library_dir(library)
    records, problems = [], []
    if folder.is_dir():
        for p in sorted(folder.glob("*.json")):
            try:
                records.append(load_record(p))
            except RecordError as e:
                problems.append(str(e))
    records.sort(key=lambda r: (r.get("created", ""), r["id"]))
    return records, problems


def summary(rec):
    """One line's worth of a record."""
    top = rec["topped"]
    return {"id": rec["id"], "created": rec["created"], "video": rec["video"]["name"],
            "route": rec["route"]["name"], "route_id": rec["route"]["id"],
            "duration_s": rec["video"]["duration_s"], "moves": len(rec["moves"]),
            "topped": top["topped"], "topped_s": top["t"], "off_route": len(rec["off_route"]["touches"])}


def routes(records):
    """Named routes in the library: [{id, name, climbs, topped}], most climbs first, plus the
    number of unnamed records as the second return value."""
    by = {}
    unnamed = 0
    for r in records:
        rid = r["route"]["id"]
        if rid is None:
            unnamed += 1
            continue
        d = by.setdefault(rid, {"id": rid, "name": r["route"]["name"], "climbs": 0, "topped": 0})
        d["name"] = r["route"]["name"]              # records are oldest first: the latest spelling wins
        d["climbs"] += 1
        d["topped"] += bool(r["topped"]["topped"])
    return sorted(by.values(), key=lambda d: (-d["climbs"], d["name"].casefold())), unnamed


# ---------------------------------------------------------------- the one "suggestion": what climbers did most
def _canonical_holds(records):
    """Name the route's holds the same way across records. Records of one route may number the
    holds differently (the picker numbers bottom to top, and a missed hold shifts the rest), so
    holds are matched by where they are relative to the route (nc). Returns
    ({(record id, hold id, is_volume): label}, {label: {"nc": [x, y], "volume": bool}}).
    Labels H1, H2 ... / V1, V2 ... go bottom to top."""
    clusters = []                                   # {"volume", "pts": [nc, ...], "members": [key, ...]}
    for r in records:
        for items, vol in ((r["wall"]["holds"], False), (r["wall"]["volumes"], True)):
            for it in items:
                nc = it.get("nc")
                if nc is None:
                    continue
                best, bd = None, SAME_HOLD
                for c in clusters:
                    if c["volume"] != vol:
                        continue
                    cx, cy = np.mean(c["pts"], axis=0)
                    d = math.hypot(nc[0] - cx, nc[1] - cy)
                    if d < bd:
                        best, bd = c, d
                if best is None:
                    best = {"volume": vol, "pts": [], "members": []}
                    clusters.append(best)
                best["pts"].append(nc)
                best["members"].append((r["id"], it["id"], vol))
    for c in clusters:
        c["nc"] = [round(float(v), 3) for v in np.mean(c["pts"], axis=0)]
    labels, info = {}, {}
    for vol, prefix in ((False, "H"), (True, "V")):
        group = sorted((c for c in clusters if c["volume"] == vol), key=lambda c: -c["nc"][1])   # bottom first
        for k, c in enumerate(group, 1):
            label = f"{prefix}{k}"
            info[label] = {"nc": c["nc"], "volume": vol}
            for m in c["members"]:
                labels[m] = label
    return labels, info


def _limb_sequence(rec, limb, labels):
    seq = []
    for m in sorted(rec["moves"], key=lambda m: m["t"]):
        if m["limb"] != limb:
            continue
        label = labels.get((rec["id"], m["hold"], bool(m["volume"])), f"?{m['hold']}")
        if not seq or seq[-1] != label:             # staying on the same hold after a pause isn't a new move
            seq.append(label)
    return seq


def _climb_time(rec):
    """Seconds from the first hold contact to topping out (walking up to the wall doesn't count)."""
    top = rec["topped"]["t"]
    first = min((m["t"] for m in rec["moves"]), default=0.0)
    return round(max(top - first, 0.0), 2)


def route_report(records, name):
    """What the library knows about the route called `name`:
         records    how many climbs are recorded for it
         enough     False until there are two or more (then nothing is suggested)
         sequences  per limb: the hold sequence most climbers used, how many climbs used exactly
                    that sequence, and out of how many that moved that limb
         holds      what H1.. / V1.. mean: where each is relative to the route (nc)
         fastest    the quickest climb that topped out (first contact to top), or None
    Raises KeyError if no record has that name."""
    rid = route_id(name)
    mine = [r for r in records if rid is not None and r["route"]["id"] == rid]
    if not mine:
        raise KeyError(name)
    out = {"route": {"id": rid, "name": mine[-1]["route"]["name"]}, "records": len(mine),
           "topped": sum(bool(r["topped"]["topped"]) for r in mine), "enough": len(mine) >= 2,
           "sequences": {}, "holds": {}, "fastest": None}
    if not out["enough"]:
        return out
    labels, info = _canonical_holds(mine)
    out["holds"] = info
    for limb in LIMB_NAMES:
        seqs = [tuple(s) for s in (_limb_sequence(r, limb, labels) for r in mine) if s]
        if not seqs:
            continue
        # most common; ties go to the sequence seen first (oldest record)
        counts = Counter(seqs)
        best = max(counts, key=lambda s: (counts[s], -seqs.index(s)))
        out["sequences"][limb] = {"sequence": list(best), "count": counts[best], "of": len(seqs)}
    topped = [r for r in mine if r["topped"]["topped"] and r["topped"]["t"] is not None]
    if topped:
        r = min(topped, key=lambda r: (_climb_time(r), r["created"]))
        out["fastest"] = {"id": r["id"], "video": r["video"]["name"], "created": r["created"],
                          "climb_time_s": _climb_time(r), "topped_at_s": r["topped"]["t"]}
    return out


# ---------------------------------------------------------------- command line: read the library
def _print_list(args):
    records, problems = read_library(args.library)
    print(f"Library: {library_dir(args.library)}  ({len(records)} climbs)")
    for r in records:
        s = summary(r)
        top = f"topped at {s['topped_s']:.1f}s" if s["topped"] else "not topped"
        print(f"  {s['created'][:16].replace('T', ' ')}  {s['route'] or '(unnamed)':<24} {s['video']:<28} "
              f"{s['moves']:>3} moves  {top}" + (f"  {s['off_route']} off-route" if s["off_route"] else ""))
    named, unnamed = routes(records)
    if named:
        print("Routes: " + ", ".join(f"{d['name']} ({d['climbs']})" for d in named)
              + (f"; {unnamed} unnamed" if unnamed else ""))
    for p in problems:
        print("  skipped:", p)


def _print_route(args):
    records, problems = read_library(args.library)
    try:
        rep = route_report(records, args.name)
    except KeyError:
        names = ", ".join(d["name"] for d in routes(records)[0]) or "none yet"
        sys.exit(f"No recorded climbs of a route called '{args.name}'. Routes in the library: {names}")
    print(f"{rep['route']['name']}: {rep['records']} climbs, {rep['topped']} topped")
    if not rep["enough"]:
        print("  Record at least two climbs of this route to see the most common sequences.")
        return
    for limb, s in rep["sequences"].items():
        print(f"  {LIMB_NAMES[limb]:<11} {' -> '.join(s['sequence'])}   ({s['count']} of {s['of']} climbs)")
    f = rep["fastest"]
    print(f"  Fastest topped climb: {f['climb_time_s']:.1f}s from first contact ({f['video']}, {f['created'][:10]})"
          if f else "  No climb of this route topped out yet.")
    print("  Holds (H = hold, V = volume; x, y in route extents from the start, y grows downwards):")
    for label, d in rep["holds"].items():
        print(f"    {label}: ({d['nc'][0]:+.2f}, {d['nc'][1]:+.2f})")
    for p in problems:
        print("  skipped:", p)


def main():
    ap = argparse.ArgumentParser(description="Read the library of recorded climbs.")
    ap.add_argument("--library", default=None, help="library folder (default: ~/.climb_analyzer/library)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list recorded climbs and routes").set_defaults(fn=_print_list)
    p = sub.add_parser("route", help="most common hold sequences and the fastest topped climb of a named route")
    p.add_argument("name")
    p.set_defaults(fn=_print_route)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
