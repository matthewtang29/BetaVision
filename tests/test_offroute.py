"""Off-route holds and the 'possible off-route touch' rules (climb_offroute), on synthetic data."""
import time

import cv2
import numpy as np
import pytest

import climb_holds as ch
import climb_offroute as co

FPS = 30
FMIN = 1080                  # short side of the video frame: max wander = 0.03 x 1080 = ~32 px


def still(n, at=(500, 500), noise=4.0, seed=0):
    """n frames of a limb resting at `at`, with pose jitter."""
    rng = np.random.default_rng(seed)
    return [(at[0] + rng.normal(0, noise), at[1] + rng.normal(0, noise)) for _ in range(n)]


def on(hold, pts):
    return [(hold, p) for p in pts]


def off(n):
    return [("", None)] * n


def events(seq, limb="r_foot", **kw):
    return co.touch_events({limb: seq}, FPS, FMIN, **kw)


# ---------------------------------------------------------------- dwell
def test_resting_limb_is_flagged_with_its_time_span():
    ev = events(off(10) + on("X3", still(30)) + off(10))
    assert ev == [{"limb": "r_foot", "hold": "X3", "start": 10, "end": 40}]


def test_short_touch_is_not_flagged():
    assert events(off(5) + on("X3", still(12)) + off(5)) == []          # 0.4 s: too short


def test_dwell_threshold_is_the_minimum_time():
    n = co.min_touch_frames(FPS)
    assert 0.5 * FPS <= n <= 0.7 * FPS                     # about 0.6 s at 30 fps
    assert len(events(on("X1", still(n)))) == 1
    assert events(on("X1", still(n - 1))) == []


def test_short_gaps_are_bridged_long_ones_are_not():
    a, b = still(10, seed=1), still(10, seed=2)
    assert len(events(on("X1", a) + off(3) + on("X1", b))) == 1            # 0.1 s lost: one touch of 23 frames
    assert events(on("X1", a) + off(10) + on("X1", b)) == []               # two separate 0.33 s touches


def test_another_off_route_hold_ends_the_run():
    assert events(on("X1", still(10)) + on("X2", still(10))) == []


def test_touches_of_each_limb_are_separate():
    ev = co.touch_events({"l_hand": on("X1", still(20)), "r_hand": off(5) + on("X2", still(20))}, FPS, FMIN)
    assert [(e["limb"], e["hold"], e["start"]) for e in ev] == [("l_hand", "X1", 0), ("r_hand", "X2", 5)]


# ---------------------------------------------------------------- stationarity
def test_limb_moving_across_a_hold_is_not_flagged():
    # a foot swinging through a big hold: 40 px per frame for 20 frames (0.67 s, long enough)
    sweep = [(300 + 40 * i, 600 + 5 * i) for i in range(20)]
    assert events(on("X2", sweep)) == []


def test_slow_drift_beyond_the_limit_is_not_flagged():
    drift = [(500 + 4 * i, 500) for i in range(30)]       # 120 px over 1 s
    assert events(on("X2", drift)) == []


def test_only_the_still_part_of_a_run_is_flagged():
    move = [(200 + 30 * i, 500) for i in range(12)]       # arrives fast ...
    rest = still(24, at=(560, 500), seed=3)               # ... then rests 0.8 s
    ev = events(on("X4", move + rest))
    assert len(ev) == 1
    assert 9 <= ev[0]["start"] <= 13 and ev[0]["end"] == 36


def test_one_bad_pose_frame_does_not_spoil_a_touch():
    pts = still(30, seed=4)
    pts[15] = (pts[15][0] + 150, pts[15][1] - 120)       # landmark jumps for one frame
    assert len(events(on("X1", pts))) == 1


def test_noisy_but_still_limb_is_flagged():
    assert len(events(on("X1", still(30, noise=10, seed=5)))) == 1


def test_stationarity_scales_with_frame_size():
    pts = still(30, noise=10, seed=6)
    assert len(co.touch_events({"l_foot": on("X1", pts)}, FPS, 1080)) == 1
    assert co.touch_events({"l_foot": on("X1", pts)}, FPS, 240) == []      # same jitter on a tiny video = moving


def test_a_shift_on_the_hold_does_not_split_one_touch_in_two():
    a = still(30, at=(500, 500), seed=7)
    shift = [(500 + 15 * i, 500) for i in range(1, 6)]           # hand re-grips 75 px over
    b = still(30, at=(575, 500), seed=8)
    ev = events(on("X1", a + shift + b))
    assert len(ev) == 1 and ev[0]["start"] == 0 and ev[0]["end"] == 65


def test_flags_per_frame():
    ev = [{"limb": "l_foot", "hold": "X1", "start": 2, "end": 4}, {"limb": "r_hand", "hold": "X1", "start": 3, "end": 5}]
    fl = co.flags_per_frame(ev, 6)
    assert fl[1] == {} and fl[2] == {"X1": ["l_foot"]} and fl[3] == {"X1": ["l_foot", "r_hand"]} and fl[5] == {}


# ---------------------------------------------------------------- off-route holds
def _box(x, y, r=20):
    return [[x - r, y - r], [x + r, y - r], [x + r, y + r], [x - r, y + r]]


def test_cutouts_join_the_wall_holds_with_unique_ids():
    wall = co.number([{"contour": _box(100, 300)}, {"contour": _box(100, 100)}])
    assert [h["id"] for h in wall] == ["X1", "X2"] and wall[0]["center"] == [100, 300]   # bottom to top
    vols = [{"id": "V1", "contour": _box(400, 400, 100), "exclude": [_box(400, 400, 10)]}]
    allh = co.all_off_route(wall, vols)
    assert [h["id"] for h in allh] == ["X1", "X2", "X3"] and allh[2]["volume"] == "V1"
    cnts = [(h["id"], np.array(h["contour"], np.int32).reshape(-1, 1, 2)) for h in allh]
    assert co.hold_at(cnts, (402, 398)) == "X3" and co.hold_at(cnts, (250, 250)) == ""


def test_right_clicked_cutouts_are_dropped():
    vols = [{"id": "V1", "contour": _box(400, 400, 100), "exclude": [_box(380, 380, 10), _box(430, 430, 10)]}]
    out = co.drop_cuts(vols, [[431, 428]])
    assert out[0]["exclude"] == [_box(380, 380, 10)] and len(vols[0]["exclude"]) == 2   # input untouched


def test_select_drops_route_holds_route_colour_deleted_and_volumes():
    cands = [{"contour": _box(100, 100), "hue": -80.0},     # blue: off-route
             {"contour": _box(300, 100), "hue": 40.0},      # red, the route's colour
             {"contour": _box(500, 100), "hue": 140.0},     # green, but it IS a marked route hold
             {"contour": _box(700, 100), "hue": 100.0},     # yellow, deleted by the user
             {"contour": _box(900, 100), "hue": 100.0}]     # yellow, on a volume
    route = [{"contour": _box(502, 101, 18)}]
    models = [{"kind": "colour", "hue": 38.0, "hue_tol": 10}]
    sel = co.select(cands, route, models, deleted=[_box(700, 100, 25)], shape=(400, 1100))
    assert [c["contour"][0][0] for c in sel] == [80, 880]
    left = co.drop_on_volumes(sel, [{"contour": _box(900, 100, 60)}])
    assert [c["contour"][0][0] for c in left] == [80]


def test_deleting_a_two_lobed_outline_removes_it():
    # the outline X5 from a real clip: its centre lies outside its own shape
    c = [[562, 769], [565, 799], [593, 807], [639, 843], [605, 853], [562, 849], [553, 872], [624, 894],
         [670, 880], [682, 830], [657, 837], [650, 829], [678, 799]]
    moved = [[x + 3, y - 2] for x, y in c]                           # the same hold, re-detected
    assert co.select([{"contour": moved, "hue": 140.0}], [], [], deleted=[c], shape=(1280, 720)) == []


def test_find_off_route_on_a_synthetic_wall():
    """Red route holds and two blue holds on a grey wall: the blue ones are off-route."""
    img = np.full((480, 640, 3), 125, np.uint8)
    for x, y in [(100, 100), (300, 120), (500, 300), (200, 380)]:
        cv2.circle(img, (x, y), 16, (40, 50, 210), -1)        # red (BGR)
    for x, y in [(420, 120), (120, 250)]:
        cv2.circle(img, (x, y), 16, (210, 90, 30), -1)        # blue
    fc = ch.FrameColour(lab=ch.to_lab(img))                   # no picture: no climber mask needed
    model = ch.hold_model(fc, 100, 100, frame=0)
    route = ch.combine(ch.find_holds_auto(fc, [model]), [], [])
    assert len(route) == 4
    off_holds = co.find_off_route(fc, route, [], [model])
    centres = sorted(tuple(h["center"]) for h in off_holds)
    assert len(centres) == 2
    assert all(np.hypot(cx - x, cy - y) < 6 for (cx, cy), (x, y) in zip(centres, sorted([(420, 120), (120, 250)])))
    gone = co.find_off_route(fc, route, [], [model], deleted=[off_holds[0]["contour"]])
    assert len(gone) == 1


# ---------------------------------------------------------------- web app pieces
def test_picker_state_keeps_off_route_deletions(jobs_mod):
    st = jobs_mod.clean_picker({"frame": 1, "off_deleted": [{"x": 5, "y": 6, "contour": [[1, 1], [9, 1], [9, 9]]}],
                                "cuts_deleted": [[3, 4, 2]]}, 100, 100, 10)
    assert st["off_deleted"][0]["frame"] == 1 and st["cuts_deleted"] == [[3, 4, 2]]
    with pytest.raises(jobs_mod.PickerError):
        jobs_mod.clean_picker({"off_deleted": [{"x": 5, "y": 6}]}, 100, 100, 10)
    old = jobs_mod.clean_picker({"frame": 0}, 100, 100, 10)                 # pages that don't send them
    assert old["off_deleted"] == [] and old["cuts_deleted"] == []


def test_summary_lists_off_route_touches(jobs_mod):
    class J:
        id = "abc"
    res = {"fps": 30.0, "frames": 90, "detected": 90, "has_end": False, "topped_s": None,
           "limb_holds": [{} for _ in range(90)], "angles": [], "notes": [], "off_route_enabled": True,
           "off_route": [{"limb": "l_foot", "hold": "X4", "start": 30, "end": 60, "t": 1.0, "dur": 1.0}],
           "off_route_holds": [{"id": "X4", "contour": _box(1, 1)}, {"id": "X5", "contour": _box(9, 9), "volume": "V1"}]}
    r = jobs_mod.summarize(J(), res, [], [])
    assert r["off_route"] == [{"t": 1.0, "dur": 1.0, "limb": "Left foot", "hold": "X4", "on_volume": None}]
    assert r["off_route_enabled"] and r["off_route_holds"] == 2
    r = jobs_mod.summarize(J(), {k: v for k, v in res.items() if not k.startswith("off_route")}, [], [])
    assert r["off_route"] == [] and not r["off_route_enabled"]              # older results dicts still work


def test_page_checkbox_reaches_the_analysis(jobs_mod, video, monkeypatch):
    """off_route in the start request (the page's checkbox) turns flagging on / off for the job."""
    from fastapi.testclient import TestClient
    from betavision_app import server

    class Q:
        def put(self, job):
            pass
    monkeypatch.setattr(server.worker, "q", Q())
    server.worker.jobs.clear()
    with TestClient(server.app) as c:
        for flag in (False, True):
            r = c.post("/api/upload", files={"file": ("wall.avi", video.read_bytes())}, data={"still": "true"})
            jid = r.json()["id"]
            job = server.worker.jobs[jid]
            for _ in range(300):
                if job.state == "picking":
                    break
                time.sleep(0.05)
            job.route_for_analysis = lambda picker: ([], [])
            r = c.post(f"/api/jobs/{jid}/start", json={"picker": {"frame": 0}, "off_route": flag})
            assert r.status_code == 200, r.text
            assert job.flag_off_route is flag
    server.worker.jobs.clear()
