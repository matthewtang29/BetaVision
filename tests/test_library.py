"""Climb records: building, writing, reading, schema versions, and the read side (synthetic data)."""
import json
import math
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import climb_library as lib


def square(cx, cy, r=10):
    return [[cx - r, cy - r], [cx + r, cy - r], [cx + r, cy + r], [cx - r, cy + r]]


def make_route(shift=(0, 0), ids=None):
    """Four holds on a 1000-px-tall wall: start (bottom), two middle, end (top). `ids` renumbers them."""
    spots = [(100, 900, "start"), (200, 600, ""), (120, 400, ""), (150, 100, "end")]
    ids = ids or [1, 2, 3, 4]
    holds = [{"id": i, "role": role, "center": [x + shift[0], y + shift[1]],
              "contour": square(x + shift[0], y + shift[1])} for i, (x, y, role) in zip(ids, spots)]
    vols = [{"id": "V1", "contour": square(300 + shift[0], 500 + shift[1], 40)}]
    return holds, vols


def make_res(seq, fps=10, topped_s=None, frames=None, off=()):
    """analyze_video-like result. seq = [(limb, hold, first frame, last frame)]."""
    n = frames or max(e for _, _, _, e in seq) + 2
    limb_holds = [{} for _ in range(n)]
    for limb, hold, a, b in seq:
        for i in range(a, b + 1):
            limb_holds[i][limb] = hold
    return {"fps": fps, "frames": n, "limb_holds": limb_holds, "has_end": True, "topped_s": topped_s,
            "angles": [{"l_elbow_deg": 90.0, "r_elbow_deg": 150.0, "l_knee_deg": float("nan"), "r_knee_deg": 120.0}] * 3 + [{}],
            "off_route": list(off), "off_route_holds": [{"id": "O1", "volume": "V1"}], "off_route_enabled": True}


def record(ids=None, seq=None, topped_s=None, name="Blue V3", when=None, shift=(0, 0)):
    holds, vols = make_route(shift, ids)
    ids = ids or [1, 2, 3, 4]
    seq = seq or [("l_hand", ids[0], 0, 5), ("r_hand", ids[1], 6, 12), ("l_hand", ids[2], 13, 20), ("r_hand", ids[3], 21, 30)]
    res = make_res(seq, topped_s=topped_s)
    return lib.build_record(res, holds, vols, "clip.mp4", base_frame=7, frame_size=(1080, 1920), route_name=name,
                            now=when or datetime(2026, 1, 1, tzinfo=timezone.utc))


def day(n):
    return datetime(2026, 1, n, tzinfo=timezone.utc)


# ---------------------------------------------------------------- building
def test_record_has_everything_asked_for():
    rec = record(topped_s=3.0)
    assert rec["schema_version"] == lib.SCHEMA_VERSION
    assert rec["route"] == {"id": lib.route_id("Blue V3"), "name": "Blue V3"}
    assert rec["video"]["name"] == "clip.mp4" and rec["created"].startswith("2026-01-01")
    h = rec["wall"]["holds"]
    assert [x["role"] for x in h] == ["start", "", "", "end"]
    assert h[0]["center"] == [100.0, 900.0] and len(h[0]["contour"]) == 4
    assert rec["wall"]["volumes"][0]["id"] == "V1"
    # start-to-end extent: origin = start hold, scale = distance to the end hold
    norm = rec["wall"]["norm"]
    assert norm["basis"] == "start_end" and norm["origin"] == [100.0, 900.0]
    assert norm["scale"] == pytest.approx(math.hypot(50, 800), abs=0.1)
    assert h[0]["nc"] == [0.0, 0.0]
    assert math.hypot(*h[3]["nc"]) == pytest.approx(1.0, abs=0.01)
    assert h[3]["nc"][1] < 0                      # the end is above the start (y grows downwards)
    assert rec["topped"] == {"has_end": True, "topped": True, "t": 3.0}
    assert {a["joint"] for a in rec["angles"]} == {"Elbows", "Knees"}
    assert rec["off_route"]["checked"] is True


def test_moves_follow_the_summary_rules():
    rec = record()
    assert [(m["limb"], m["hold"], m["t"], m["dur"]) for m in rec["moves"]] == [
        ("l_hand", 1, 0.0, 0.6), ("r_hand", 2, 0.6, 0.7), ("l_hand", 3, 1.3, 0.8), ("r_hand", 4, 2.1, 1.0)]
    assert rec["moves"][0]["role"] == "start" and rec["moves"][3]["role"] == "end"
    # a 1-frame blip is jitter; volumes are kept and flagged
    res = make_res([("l_hand", 1, 0, 0), ("r_foot", "V1", 2, 9)])
    moves = lib.contact_moves(res["limb_holds"], 10, make_route()[0])
    assert [(m["limb"], m["hold"], m["volume"]) for m in moves] == [("r_foot", "V1", True)]


def test_off_route_touches_are_kept():
    res = make_res([("l_hand", 1, 0, 5)], off=[{"limb": "r_foot", "hold": "O1", "start": 10, "end": 20, "t": 1.0, "dur": 1.0}])
    rec = lib.build_record(res, *make_route(), "c.mp4")
    assert rec["off_route"]["touches"] == [{"t": 1.0, "dur": 1.0, "limb": "r_foot", "hold": "O1", "on_volume": "V1"}]


def test_no_start_or_end_falls_back_to_the_bounding_box():
    holds, _ = make_route()
    for h in holds:
        h["role"] = ""
    norm = lib.normalisation(holds)
    assert norm["basis"] == "bbox" and norm["scale"] == 800.0 and lib.normalisation([]) is None


def test_route_ids_ignore_case_and_spacing_and_unnamed_has_none():
    assert lib.route_id("Blue  V3 ") == lib.route_id("blue v3") != lib.route_id("blue v4")
    assert lib.route_id("") is None and lib.route_id(None) is None
    assert record(name="")["route"] == {"id": None, "name": None}
    with pytest.raises(ValueError):
        lib.clean_name("x" * 81)


def test_record_stays_small():
    holds = [{"id": i, "role": "", "center": [i * 10, i * 5],
              "contour": [[500 + 80 * math.cos(a / 40), 500 + 80 * math.sin(a / 40)] for a in range(252)]}
             for i in range(1, 41)]
    rec = lib.build_record(make_res([("l_hand", 1, 0, 9)], frames=18000), holds, [], "long.mp4")
    assert len(json.dumps(rec)) < 100_000
    assert all(len(h["contour"]) <= 48 for h in rec["wall"]["holds"])


# ---------------------------------------------------------------- writing + reading
def test_save_then_load_roundtrip_in_the_library_folder(tmp_path):
    rec = record(topped_s=3.0)
    path = lib.save_record(rec)                              # BETAVISION_LIBRARY = a temp folder (conftest)
    assert path.parent == tmp_path / "test-library" and path.name == rec["id"] + ".json"
    assert lib.load_record(path) == json.loads(json.dumps(rec))
    assert [p.name for p in path.parent.iterdir()] == [path.name]      # no temp file left behind


def test_library_folder_can_be_given_and_is_created(tmp_path):
    p = lib.save_record(record(), tmp_path / "a" / "b")
    assert p.exists() and lib.library_dir(tmp_path / "a" / "b") == tmp_path / "a" / "b"
    assert lib.library_dir() == tmp_path / "test-library"


def test_default_library_is_under_the_user_folder(monkeypatch):
    monkeypatch.delenv("BETAVISION_LIBRARY")
    assert lib.library_dir().parts[-2:] == (".climb_analyzer", "library")


def test_two_saves_never_overwrite_each_other():
    assert lib.save_record(record()) != lib.save_record(record())
    assert len(lib.read_library()[0]) == 2


def test_missing_library_reads_as_empty(tmp_path):
    assert lib.read_library(tmp_path / "nothing here") == ([], [])


def test_newer_schema_is_refused_with_a_clear_message(tmp_path):
    rec = record()
    rec["schema_version"] = lib.SCHEMA_VERSION + 1
    p = tmp_path / "new.json"
    p.write_text(json.dumps(rec))
    with pytest.raises(lib.UnsupportedSchema, match="newer BetaVision"):
        lib.load_record(p)


@pytest.mark.parametrize("text", ["not json", "[]", "{}", '{"schema_version": "1"}', '{"schema_version": 0}',
                                  '{"schema_version": true}', '{"schema_version": 1}'])
def test_bad_files_are_record_errors(tmp_path, text):
    p = tmp_path / "x.json"
    p.write_text(text)
    with pytest.raises(lib.RecordError):
        lib.load_record(p)


def test_one_bad_file_does_not_hide_the_good_ones():
    lib.save_record(record())
    folder = lib.library_dir()
    (folder / "broken.json").write_text("{nope")
    newer = record()
    newer["schema_version"] = 99
    (folder / "future.json").write_text(json.dumps(newer))
    (folder / "notes.txt").write_text("ignored")
    records, problems = lib.read_library()
    assert len(records) == 1 and len(problems) == 2
    assert any("newer BetaVision" in p for p in problems)


def test_older_schema_is_migrated(tmp_path, monkeypatch):
    monkeypatch.setattr(lib, "SCHEMA_VERSION", 2)
    monkeypatch.setitem(lib.MIGRATIONS, 1, lambda r: dict(r, schema_version=2, migrated=True))
    old = record()
    old["schema_version"] = 1
    p = tmp_path / "old.json"
    p.write_text(json.dumps(old))
    got = lib.load_record(p)
    assert got["schema_version"] == 2 and got["migrated"] is True
    monkeypatch.setattr(lib, "SCHEMA_VERSION", 3)          # no step 2 -> 3: a clear error, not a crash
    with pytest.raises(lib.RecordError, match="no way to read"):
        lib.load_record(p)


def test_oversized_record_drops_outlines_instead_of_growing(monkeypatch):
    rec = lib.build_record(make_res([("l_hand", 1, 0, 9)]), *make_route(), "c.mp4")
    full = len(json.dumps(rec, separators=(",", ":")))
    monkeypatch.setattr(lib, "MAX_RECORD_BYTES", full - 50)
    path = lib.save_record(rec)
    saved = json.loads(path.read_text())
    assert saved["wall"]["outlines_dropped"] and "contour" not in saved["wall"]["holds"][0]
    rec["pad"] = "x" * 5_000
    with pytest.raises(lib.RecordError, match="too large"):
        lib.save_record(rec)


def test_no_route_marked_saves_nothing():
    assert lib.record_analysis(make_res([("l_hand", 1, 0, 9)]), [], [], "c.mp4") == (None, None)
    assert lib.read_library()[0] == []


def test_summary_and_routes():
    lib.save_record(record(name="Blue V3", topped_s=3.0))
    lib.save_record(record(name="blue  v3"))
    lib.save_record(record(name=""))
    records, _ = lib.read_library()
    named, unnamed = lib.routes(records)
    assert unnamed == 1 and len(named) == 1 and named[0]["climbs"] == 2 and named[0]["topped"] == 1
    assert {lib.summary(r)["topped"] for r in records} == {True, False}


# ---------------------------------------------------------------- read side
def test_report_needs_two_records():
    rep = lib.route_report([record()], "blue v3")
    assert rep["enough"] is False and rep["sequences"] == {} and rep["fastest"] is None
    with pytest.raises(KeyError):
        lib.route_report([record()], "other route")
    with pytest.raises(KeyError):
        lib.route_report([record(name="")], "")


def test_most_common_sequence_matches_holds_by_position_not_number():
    # climb c numbered its holds differently (the picker missed one) and was filmed from a bit off
    a, b = record(when=day(1)), record(when=day(2))
    c = record(ids=[10, 20, 30, 40], when=day(3), shift=(5, -3),
               seq=[("l_hand", 10, 0, 5), ("r_hand", 30, 6, 12), ("l_hand", 20, 13, 20), ("r_hand", 40, 21, 30)])
    rep = lib.route_report([a, b, c], "Blue V3")
    assert rep["enough"] and rep["records"] == 3
    assert rep["sequences"]["l_hand"] == {"sequence": ["H1", "H3"], "count": 2, "of": 3}
    assert rep["sequences"]["r_hand"]["sequence"] == ["H2", "H4"] and rep["sequences"]["r_hand"]["count"] == 2
    assert "l_foot" not in rep["sequences"]
    assert set(rep["holds"]) == {"H1", "H2", "H3", "H4", "V1"}
    assert rep["holds"]["H1"]["nc"][1] > rep["holds"]["H4"]["nc"][1]      # numbered bottom to top


def test_repeated_contact_with_one_hold_is_one_step():
    seq = [("l_hand", 1, 0, 5), ("l_hand", 1, 9, 15), ("l_hand", 2, 20, 30)]
    rep = lib.route_report([record(seq=seq, when=day(1)), record(seq=seq, when=day(2))], "Blue V3")
    assert rep["sequences"]["l_hand"] == {"sequence": ["H1", "H2"], "count": 2, "of": 2}


def test_tie_goes_to_the_older_record():
    a = record(when=day(1), seq=[("l_hand", 1, 0, 5), ("l_hand", 2, 9, 15)])
    b = record(when=day(2), seq=[("l_hand", 1, 0, 5), ("l_hand", 3, 9, 15)])
    rep = lib.route_report([a, b], "Blue V3")
    assert rep["sequences"]["l_hand"] == {"sequence": ["H1", "H2"], "count": 1, "of": 2}


def test_fastest_topped_attempt_counts_from_first_contact():
    slow = record(when=day(1), topped_s=8.0)                                  # first contact 0.0 -> 8.0 s
    fast = record(when=day(2), topped_s=6.0, seq=[("l_hand", 1, 20, 25), ("r_hand", 4, 40, 50)])   # 2.0 -> 6.0 s
    fell = record(when=day(3), topped_s=None)
    rep = lib.route_report([slow, fast, fell], "blue v3")
    assert rep["topped"] == 2 and rep["fastest"]["id"] == fast["id"] and rep["fastest"]["climb_time_s"] == 4.0
    assert lib.route_report([fell, record(when=day(4))], "Blue V3")["fastest"] is None


def test_only_the_named_route_is_used():
    recs = [record(when=day(1)), record(when=day(2)), record(name="Other", when=day(3)), record(name="", when=day(4))]
    assert lib.route_report(recs, "Blue V3")["records"] == 2


# ---------------------------------------------------------------- command line + web app
def test_cli_reads_the_library(capsys, monkeypatch):
    for d in (1, 2):
        lib.save_record(record(when=day(d), topped_s=5.0))
    for args in (["list"], ["route", "Blue V3"]):
        monkeypatch.setattr(sys, "argv", ["climb_library.py", *args])
        lib.main()
    out = capsys.readouterr().out
    assert "2 climbs" in out and "Blue V3 (2)" in out and "Fastest topped climb" in out and "Left hand" in out


def fake_job(**kw):
    holds, vols = make_route()
    base = dict(record=True, route_name="Blue V3", holds=holds, vols=vols, base=7, width=1080, height=1920,
                input=SimpleNamespace(name="clip.mp4"))
    return SimpleNamespace(**{**base, **kw})


def test_web_save_respects_the_opt_out_and_never_raises(jobs_mod, monkeypatch):
    res = make_res([("l_hand", 1, 0, 9)], topped_s=1.0)
    out = jobs_mod.save_to_library(fake_job(record=False), res)
    assert out["saved"] is False and lib.read_library()[0] == []
    out = jobs_mod.save_to_library(fake_job(holds=[]), res)
    assert out["saved"] is False and "no route" in out["message"]
    out = jobs_mod.save_to_library(fake_job(), res)
    assert out["saved"] is True and out["route"] == "Blue V3" and out["climbs"] == 1
    assert jobs_mod.save_to_library(fake_job(), res)["climbs"] == 2

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(lib, "save_record", boom)
    out = jobs_mod.save_to_library(fake_job(), res)
    assert out["saved"] is False and "disk full" in out["message"]


def test_web_summary_still_has_labelled_moves(jobs_mod):
    res = make_res([("l_hand", 1, 0, 9), ("r_foot", "V1", 2, 9)])
    res.update(detected=10, notes=[])
    holds, vols = make_route()
    s = jobs_mod.summarize(SimpleNamespace(id="abc"), res, holds, vols)
    assert [(m["limb"], m["hold"]) for m in s["moves"]] == [("Left hand", 1), ("Right foot", "V1")]
    assert s["angles"][0]["joint"] == "Elbows" and s["off_route"] == []


def test_library_endpoints(jobs_mod):
    from fastapi.testclient import TestClient
    from betavision_app import server
    c = TestClient(server.app)
    r = c.get("/api/library").json()
    assert r["enabled"] is True and r["climbs"] == 0 and r["routes"] == []
    for d in (1, 2):
        lib.save_record(record(when=day(d)))
    r = c.get("/api/library").json()
    assert r["climbs"] == 2 and r["routes"][0]["name"] == "Blue V3"
    assert c.get("/api/library/route", params={"name": "blue v3"}).json()["enough"] is True
    assert c.get("/api/library/route", params={"name": "nope"}).status_code == 404


def test_recording_default_can_be_turned_off_by_environment(monkeypatch):
    assert lib.recording_default() is True
    monkeypatch.setenv("BETAVISION_RECORD", "0")
    assert lib.recording_default() is False


def test_cli_records_after_analysis(video, tmp_path, capsys):
    import climb_analyze as ca
    holds, vols = make_route()
    res = make_res([("l_hand", 1, 0, 9)], topped_s=1.0)
    ca.record_climb(res, holds, vols, video, 3, "Blue V3", tmp_path / "lib2")
    files = list((tmp_path / "lib2").glob("*.json"))
    assert len(files) == 1
    rec = lib.load_record(files[0])
    assert rec["video"]["name"] == "wall.avi" and rec["video"]["frame_size"] == [320, 240] and rec["video"]["file_date"]
    assert "saved" in capsys.readouterr().out
    ca.record_climb(res, [], [], video, 3, "", tmp_path / "lib2")          # no route: nothing more written
    assert len(list((tmp_path / "lib2").glob("*.json"))) == 1
