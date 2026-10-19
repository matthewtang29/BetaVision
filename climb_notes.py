"""
climb_notes.py - sticky-note annotations for the analysis video.

Write notes in a text file (one per line) or pass them with --note:

    20-21  footswap
    0:33-0:35  drop knee @5          <- "@5" pins the note to hold 5
    38.5-40  big move to the jug @hand

  time range   start-end in seconds (20-21, 20.5-22) or m:ss (0:20-0:21)
  text         what to show on the note
  @N           optional: pin to hold N (the numbers shown in the hold picker)
  @V2          optional: pin to volume V2
  @lfoot @rfoot @lhand @rhand   optional: that limb's hold - or the limb itself if it's
               not on a hold (e.g. a flag). Writing "left foot" / "right hand" in the
               text does the same.
  @hand/@foot  optional: pin to whichever hold your hands / feet are using then

Without an @, the note pins to the hold or volume being used in that time window
(feet if the text mentions a foot/heel/toe, hands if it mentions a hand/match,
otherwise any limb). The note follows its hold when the camera moves.
"""

import re

import cv2
import numpy as np

NOTE_BG = (130, 236, 255)       # sticky-note yellow (BGR)
NOTE_EDGE = (60, 175, 220)
NOTE_TEXT = (35, 35, 35)
LEADER = (40, 40, 40)

FOOT_WORDS = ("foot", "feet", "heel", "toe", "smear", "flag")
HAND_WORDS = ("hand", "match", "crimp", "pinch", "sloper", "jug", "cross", "dyno", "reach",
              "grip", "establish", "start", "gaston", "undercling", "sidepull")


def _sec(t):
    t = t.strip()
    if ":" in t:
        m, s = t.split(":")
        return int(m) * 60 + float(s)
    return float(t)


def parse_note(line):
    """'20-21 footswap @3' -> dict, or None for blank/comment lines."""
    line = line.split("#", 1)[0].strip()
    if not line:
        return None
    m = re.match(r"^\s*([\d:.]+)\s*-\s*([\d:.]+)\s+(.*)$", line)
    if not m:
        raise ValueError(f"Can't read note '{line}'. Use: start-end text   e.g.  20-21 footswap")
    start, end, rest = _sec(m.group(1)), _sec(m.group(2)), m.group(3).strip()
    anchor = None
    a = re.search(r"(?:^|\s)@(\S+)\s*$", rest)
    if a:
        anchor = a.group(1).lower()
        rest = rest[:a.start()].strip()
    if end < start:
        start, end = end, start
    return {"start": start, "end": end, "text": rest, "anchor": anchor}


def load_notes(note_args=(), notes_file=None):
    notes = []
    lines = list(note_args)
    if notes_file:
        with open(notes_file, encoding="utf-8") as f:
            lines += f.read().splitlines()
    for ln in lines:
        n = parse_note(ln)
        if n:
            notes.append(n)
    return notes


SIDE_ANCHORS = {"lfoot": "l_foot", "rfoot": "r_foot", "lhand": "l_hand", "rhand": "r_hand",
                "leftfoot": "l_foot", "rightfoot": "r_foot", "lefthand": "l_hand", "righthand": "r_hand"}


def _limbs_for(note):
    if note["anchor"] in SIDE_ANCHORS:
        return (SIDE_ANCHORS[note["anchor"]],)
    t = note["text"].lower()
    for side, s in (("left", "l"), ("right", "r")):            # "flag left foot", "right hand cross"
        for part, p in (("foot", "foot"), ("heel", "foot"), ("toe", "foot"), ("hand", "hand")):
            if f"{side} {part}" in t:
                return (f"{s}_{p}",)
    if note["anchor"] in ("foot", "feet"):
        return ("l_foot", "r_foot")
    if note["anchor"] in ("hand", "hands"):
        return ("l_hand", "r_hand")
    t = note["text"].lower()
    if any(wd in t for wd in FOOT_WORDS):
        return ("l_foot", "r_foot")
    if any(wd in t for wd in HAND_WORDS):
        return ("l_hand", "r_hand")
    return ("l_hand", "r_hand", "l_foot", "r_foot")


def resolve_anchors(notes, records, fps, hold_ids):
    """Decide what each note points at: a hold or a volume (whichever the relevant limbs use
    most in the note's time window). records[i]["on_hold"] = {limb: hold id / volume id / ''};
    records[i]["base_pts"] = {limb: (x, y)} = that limb's position in the marked frame.
    Sets n["hold"] (hold id, volume id or None) and n["point"] (where to aim, marked-frame px)."""
    for n in notes:
        n["hold"], n["point"] = None, None
        if n["anchor"] and n["anchor"].isdigit():
            hid = int(n["anchor"])
            n["hold"] = hid if hid in hold_ids else None
            if n["hold"] is None:
                print(f"  note '{n['text']}': there's no hold {hid}, so it isn't pinned to a hold")
            continue
        if n["anchor"] and n["anchor"].upper().startswith("V"):
            n["hold"] = n["anchor"].upper()
        limbs = _limbs_for(n)
        for pad in (0.0, 0.5, 1.5):                     # widen the window if nothing was touched
            a = max(0, int((n["start"] - pad) * fps))
            b = min(len(records), int((n["end"] + pad) * fps) + 1)
            counts, where = {}, {}
            for r in records[a:b]:
                for limb in limbs:
                    hid = r["on_hold"].get(limb, "")
                    if hid == "" or (n["hold"] is not None and hid != n["hold"]):
                        continue
                    counts[hid] = counts.get(hid, 0) + 1
                    bp = r.get("base_pts", {}).get(limb)
                    if bp is not None:
                        where.setdefault(hid, []).append(bp)
            if counts:
                n["hold"] = max(counts, key=counts.get)
                if isinstance(n["hold"], str) and where.get(n["hold"]):   # volume: aim at the contact spot
                    n["point"] = tuple(np.median(np.array(where[n["hold"]]), axis=0))
                break
        n["limb"] = limbs[0] if len(limbs) == 1 else None
        if n["hold"] is None and n["limb"]:
            desc = f"your {n['limb'].replace('l_', 'left ').replace('r_', 'right ')} (follows it)"
        elif n["hold"] is None:
            desc = "nothing (shown at the top)"
        elif isinstance(n["hold"], str):
            desc = f"volume {n['hold']}"
        else:
            desc = f"hold {n['hold']}"
        print(f"  note '{n['text']}' {n['start']:g}-{n['end']:g}s -> {desc}")
    return notes


def _note_patch(text, fs, th):
    """Render the sticky note on its own small image (+ alpha) so it can be tilted."""
    font = cv2.FONT_HERSHEY_DUPLEX
    words, lines, cur = text.split(), [], ""
    max_chars = 18                                      # wrap long notes
    for wd in words:
        if len(cur) + len(wd) + 1 > max_chars and cur:
            lines.append(cur)
            cur = wd
        else:
            cur = (cur + " " + wd).strip()
    lines.append(cur)
    sizes = [cv2.getTextSize(ln, font, fs, th)[0] for ln in lines]
    lh = max(s[1] for s in sizes)
    pad = int(lh * 0.8)
    W = max(s[0] for s in sizes) + 2 * pad
    H = len(lines) * lh + (len(lines) - 1) * int(lh * 0.6) + 2 * pad
    W, H = max(W, int(H * 1.2)), H
    S = int(lh * 0.35)                                  # shadow offset
    img = np.zeros((H + S * 2, W + S * 2, 3), np.uint8)
    a = np.zeros(img.shape[:2], np.float32)
    # soft shadow
    cv2.rectangle(a, (S, S), (W + S, H + S), 0.35, -1)
    a = cv2.GaussianBlur(a, (0, 0), S * 0.6)
    # note body with a folded corner
    fold = int(min(W, H) * 0.22)
    body = np.array([[0, 0], [W, 0], [W, H - fold], [W - fold, H], [0, H]], np.int32)
    cv2.fillPoly(img, [body], NOTE_BG)
    cv2.fillPoly(a, [body], 1.0)
    cv2.fillPoly(img, [np.array([[W, H - fold], [W - fold, H - fold], [W - fold, H]], np.int32)], NOTE_EDGE)
    cv2.rectangle(img, (0, 0), (W, max(2, int(lh * 0.35))), NOTE_EDGE, -1)    # sticky strip
    y = pad + lh
    for ln, sz in zip(lines, sizes):
        cv2.putText(img, ln, ((W - sz[0]) // 2, y), font, fs, NOTE_TEXT, th, cv2.LINE_AA)
        y += lh + int(lh * 0.6)
    return img, a


def draw_note(frame, text, anchor, alpha=1.0, avoid=None, tilt=-3.0):
    """Draw a sticky note near `anchor` (x, y) with a pointer to it. `avoid` (x, y), e.g. the
    climber's hips, makes the note go on the other side. anchor=None: top-centre, no pointer."""
    if alpha <= 0:
        return
    h, w = frame.shape[:2]
    fs = min(h, w) / 750
    th = max(2, int(fs * 2))
    patch, a = _note_patch(text, fs, th)
    M = cv2.getRotationMatrix2D((patch.shape[1] / 2, patch.shape[0] / 2), tilt, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    pw, ph = int(patch.shape[1] * cos + patch.shape[0] * sin), int(patch.shape[1] * sin + patch.shape[0] * cos)
    M[0, 2] += pw / 2 - patch.shape[1] / 2
    M[1, 2] += ph / 2 - patch.shape[0] / 2
    patch = cv2.warpAffine(patch, M, (pw, ph))
    a = cv2.warpAffine(a, M, (pw, ph)) * alpha

    if anchor is None:
        x0, y0 = (w - pw) // 2, int(h * 0.08)
    else:
        ax, ay = anchor
        gap = int(w * 0.10)
        left_room, right_room = ax, w - ax
        go_right = right_room > left_room
        if avoid is not None and abs(avoid[0] - ax) < w * 0.35:
            go_right = avoid[0] < ax                       # put the note away from the climber
        x0 = ax + gap if go_right else ax - gap - pw
        y0 = ay - ph - int(h * 0.03)
        if y0 < 8:                                          # no room above the hold: go below it
            y0 = ay + int(h * 0.03)
        x0 = int(min(max(x0, 8), w - pw - 8))
        y0 = int(min(max(y0, 8), h - ph - 8))
        # pointer from the note's nearest edge to the hold
        nx = x0 if go_right else x0 + pw
        ny = min(max(ay, y0 + ph // 3), y0 + ph - ph // 4)
        overlay = frame.copy()
        cv2.line(overlay, (int(nx), int(ny)), (int(ax), int(ay)), LEADER, th + 2, cv2.LINE_AA)
        cv2.line(overlay, (int(nx), int(ny)), (int(ax), int(ay)), NOTE_BG, th, cv2.LINE_AA)
        cv2.circle(overlay, (int(ax), int(ay)), th * 3, LEADER, -1, cv2.LINE_AA)
        cv2.circle(overlay, (int(ax), int(ay)), th * 2, NOTE_BG, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)

    # paste with alpha (clipped to the frame)
    x1, y1 = min(w, x0 + pw), min(h, y0 + ph)
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0c, y0c = max(0, x0), max(0, y0)
    if x1 <= x0c or y1 <= y0c:
        return
    roi = frame[y0c:y1, x0c:x1].astype(np.float32)
    p = patch[sy0:sy0 + (y1 - y0c), sx0:sx0 + (x1 - x0c)].astype(np.float32)
    al = a[sy0:sy0 + (y1 - y0c), sx0:sx0 + (x1 - x0c)][..., None]
    # the shadow is drawn only in alpha (patch is black there), so it darkens; the body is opaque
    frame[y0c:y1, x0c:x1] = (roi * (1 - al) + p * al).astype(np.uint8)


def note_alpha(t, start, end, fade=0.2):
    if t < start - fade or t > end + fade:
        return 0.0
    if t < start:
        return (t - (start - fade)) / fade
    if t > end:
        return ((end + fade) - t) / fade
    return 1.0
