import threading
import time

import pytest
from fastapi.testclient import TestClient


class FakeQueue:
    def __init__(self):
        self.items = []

    def put(self, job):
        self.items.append(job)


# ---------------------------------------------------------------- job caches
def test_frame_and_colour_share_one_byte_budget(open_job):
    job = open_job()
    job.cache.max_bytes = 1_500_000             # room for only a couple of 320x240 frames' colour data
    for i in range(job.reader.total):
        job.fc(i)
        job.frame(i)
        assert job.cache.nbytes() <= job.cache.max_bytes
    assert len(job.cache) >= 1


def test_concurrent_requests_compute_a_frame_once(open_job, monkeypatch):
    job = open_job()
    reads = []
    real = job.reader.read

    def slow_read(i):
        reads.append(i)
        time.sleep(0.1)
        return real(i)
    monkeypatch.setattr(job.reader, "read", slow_read)
    ts = [threading.Thread(target=job.frame, args=(3,)) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert reads == [3]


def test_preview_from_many_threads(open_job):
    job = open_job()
    job.cache.max_bytes = 2_000_000
    errors = []

    def go(k):
        try:
            for i in range(5):
                f = (k + i) % 12
                job.preview({"frame": f, "tol": 20 + k, "auto": [{"x": 60, "y": 60, "frame": f}]})
        except Exception as e:                  # noqa: BLE001
            errors.append(repr(e))
    ts = [threading.Thread(target=go, args=(k,)) for k in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors, errors[:1]


# ---------------------------------------------------------------- double start
def _slow_route(job, delay=0.3):
    calls = []

    def route(picker):
        calls.append(1)
        time.sleep(delay)
        return [], []
    job.route_for_analysis = route
    return calls


def test_start_twice_queues_once(jobs_mod, open_job):
    w = jobs_mod.Worker()
    w.q = FakeQueue()                           # don't run an analysis
    job = open_job(w)
    calls = _slow_route(job)
    outcomes, go = [], threading.Barrier(2)

    def press():
        go.wait()
        try:
            w.start(job, {"frame": 0}, "full", [], True)
            outcomes.append("started")
        except jobs_mod.JobBusy:
            outcomes.append("busy")
    ts = [threading.Thread(target=press) for _ in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(outcomes) == ["busy", "started"]
    assert len(w.q.items) == 1 and len(calls) == 1 and job.state == "queued"


def test_failed_start_goes_back_to_picking(jobs_mod, open_job):
    w = jobs_mod.Worker()
    job = open_job(w)

    def boom(p):
        raise RuntimeError("boom")
    job.route_for_analysis = boom
    with pytest.raises(RuntimeError):
        w.start(job, {"frame": 0}, "full", [], True)
    assert job.state == "picking"


@pytest.fixture
def client(jobs_mod, video, monkeypatch):
    from betavision_app import server
    monkeypatch.setattr(server.worker, "q", FakeQueue())
    server.worker.jobs.clear()
    with TestClient(server.app) as c:
        yield c, server
    server.worker.jobs.clear()


def upload(c, video):
    r = c.post("/api/upload", files={"file": ("wall.avi", video.read_bytes())}, data={"still": "true"})
    assert r.status_code == 200, r.text
    jid = r.json()["id"]
    s = None
    for _ in range(300):
        s = c.get(f"/api/jobs/{jid}").json()
        if s["state"] == "picking":
            return jid
        time.sleep(0.05)
    raise AssertionError(s)


def test_two_start_requests_http(client, video):
    c, server = client
    jid = upload(c, video)
    job = server.worker.jobs[jid]
    calls = _slow_route(job)
    codes, go = [], threading.Barrier(2)

    def press():
        go.wait()
        codes.append(c.post(f"/api/jobs/{jid}/start", json={"picker": {"frame": 0, "tol": 22}}).status_code)
    ts = [threading.Thread(target=press) for _ in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(codes) == [200, 409]
    assert len(calls) == 1


# ---------------------------------------------------------------- picker validation
BAD = [
    ({"frame": 0, "auto": [{"y": 5, "frame": 0}]}, "missing 'x'"),
    ({"frame": 0, "auto": [{"x": 99999, "y": 5, "frame": 0}]}, "auto[0] x"),
    ({"frame": 0, "auto": [{"x": 5, "y": -3, "frame": 0}]}, "auto[0] y"),
    ({"frame": 9999}, "frame"),
    ({"frame": 0, "manual": [{"x": 5, "y": 5, "frame": 77}]}, "manual[0] frame"),
    ({"frame": 0, "tol": "high"}, "tol"),
    ({"frame": 0, "auto": "nope"}, "'auto'"),
    ({"frame": 0, "starts": [[5]]}, "starts[0]"),
    ({"frame": 0, "deleted": [{"x": 5, "frame": 0}]}, "deleted[0]"),
    ({"frame": 0, "vols_manual": [{"kind": "poly", "points": [[1, 1]], "frame": 0}]}, "3 corner"),
    ({"frame": 0, "vols_manual": [{"kind": "blob"}]}, "kind"),
]


@pytest.mark.parametrize("state,word", BAD)
def test_bad_picker_state_is_a_readable_400(client, video, state, word):
    c, _ = client
    jid = upload(c, video)
    r = c.post(f"/api/jobs/{jid}/holds", json=state)
    assert r.status_code == 400, r.text
    assert word in r.json()["detail"]
    r = c.post(f"/api/jobs/{jid}/start", json={"picker": state})
    assert r.status_code == 400
    assert c.get(f"/api/jobs/{jid}").json()["state"] == "picking"     # a rejected start leaves it startable


def test_non_object_body_is_400(client, video):
    c, _ = client
    jid = upload(c, video)
    assert c.post(f"/api/jobs/{jid}/holds", json=[1, 2]).status_code == 400


def test_both_delete_shapes_still_work(client, video):
    c, _ = client
    jid = upload(c, video)
    square = [[40, 40], [80, 40], [80, 80], [40, 80]]
    for deleted in ([{"x": 60, "y": 60, "frame": 0, "contour": square}],     # outline (current page)
                    [{"x": 60, "y": 60, "frame": 0}],                       # point as an object
                    [[60, 60, 0], [60, 60]]):                               # old [x, y, frame]
        r = c.post(f"/api/jobs/{jid}/holds", json={"frame": 0, "tol": 22, "deleted": deleted,
                                                   "auto": [{"x": 160, "y": 100, "frame": 0}]})
        assert r.status_code == 200, r.text
        assert "holds" in r.json() and "rejected_auto" in r.json()


def test_missing_optional_frame_defaults_to_current(client, video):
    c, _ = client
    jid = upload(c, video)
    r = c.post(f"/api/jobs/{jid}/holds", json={"frame": 2, "auto": [{"x": 160, "y": 100}], "manual": [{"x": 60, "y": 60}]})
    assert r.status_code == 200, r.text


# ---------------------------------------------------------------- freeing jobs
def test_delete_frees_job_and_files(client, video):
    c, server = client
    jid = upload(c, video)
    job = server.worker.jobs[jid]
    assert job.dir.exists()
    assert c.delete(f"/api/jobs/{jid}").status_code == 200
    assert jid not in server.worker.jobs and not job.dir.exists() and job.reader is None and len(job.cache) == 0
    assert c.get(f"/api/jobs/{jid}").status_code == 404
    assert c.delete(f"/api/jobs/{jid}").status_code == 200       # already gone is fine


def test_finished_and_idle_jobs_are_capped(jobs_mod, open_job):
    w = jobs_mod.Worker()
    made = [open_job(w) for _ in range(jobs_mod.MAX_KEPT_JOBS + 3)]
    for j in made:
        j.state = "done"
        j.touched = time.monotonic()
        time.sleep(0.001)
    w.trim()
    assert len(w.jobs) == jobs_mod.MAX_KEPT_JOBS
    assert all(j.id in w.jobs for j in made[-jobs_mod.MAX_KEPT_JOBS:])
    assert all(not j.dir.exists() for j in made[:3])


def test_busy_job_is_freed_when_it_stops(jobs_mod, open_job):
    w = jobs_mod.Worker()
    job = open_job(w)
    job.state = "running"
    assert w.remove(job.id) and job.dir.exists() and job.discard      # can't pull files from under a running analysis
    job.state = "done"
    assert job.free() and not job.dir.exists()
