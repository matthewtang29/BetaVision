"""The local web app's API: upload checks, picker validation, and who may call it."""

import time

import pytest
from fastapi.testclient import TestClient

from betavision_app import jobs, server

HOST = {"host": "127.0.0.1:8765"}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "WORKSPACE", tmp_path / "workspace")
    jobs.reset_workspace()
    server.worker.jobs.clear()
    with TestClient(server.app, base_url="http://127.0.0.1:8765") as c:
        yield c


def upload(client, path, name=None, still=True, **kw):
    with open(path, "rb") as f:
        return client.post("/api/upload", files={"file": (name or path.name, f, "video/mp4")},
                           data={"still": "true" if still else "false"}, **kw)


def wait_for(client, job_id, states, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        s = client.get(f"/api/jobs/{job_id}").json()
        if s["state"] in states:
            return s
        time.sleep(0.1)
    raise AssertionError(f"job stuck in {s['state']}")


def test_page_and_install_files(client):
    page = client.get("/")
    assert page.status_code == 200 and "manifest.webmanifest" in page.text
    assert client.get("/manifest.webmanifest").status_code == 200
    assert client.get("/sw.js").headers["cache-control"] == "no-cache"


def test_unknown_page_gets_the_404_page(client):
    r = client.get("/nope")
    assert r.status_code == 404 and "Off route" in r.text


def test_rejects_wrong_file_type(client, tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hello")
    r = upload(client, p)
    assert r.status_code == 400 and ".mp4" in r.json()["detail"]


def test_rejects_unreadable_video_and_cleans_up(client, tmp_path):
    p = tmp_path / "fake.mp4"
    p.write_bytes(b"not a video" * 100)
    r = upload(client, p)
    assert r.status_code == 400
    assert not server.worker.jobs                        # forgotten
    assert not any(jobs.WORKSPACE.iterdir())             # and deleted from disk


def test_rejects_oversized_upload_before_reading_it(client, wall_video, monkeypatch):
    monkeypatch.setattr(jobs, "MAX_UPLOAD_MB", 0)
    r = upload(client, wall_video)
    assert r.status_code == 413 and "GB" in r.json()["detail"]
    assert not any(jobs.WORKSPACE.iterdir())


def test_rejects_too_long_video(client, wall_video, monkeypatch):
    monkeypatch.setattr(jobs, "MAX_MINUTES", 0.01)       # the test clip is 2 s
    r = upload(client, wall_video)
    assert r.status_code == 400 and "minutes" in r.json()["detail"]
    assert not any(jobs.WORKSPACE.iterdir())


def test_other_websites_cant_use_the_api(client, wall_video):
    r = upload(client, wall_video, headers={"origin": "https://evil.example"})
    assert r.status_code == 403
    r = client.get("/api/limits", headers={"host": "attacker.example:8765"})   # DNS rebinding
    assert r.status_code == 400
    assert client.get("/api/limits", headers={"origin": "http://127.0.0.1:8765"}).status_code == 200


def test_picker_flow_and_bad_input(client, wall_video):
    r = upload(client, wall_video)
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    s = wait_for(client, job_id, {"picking", "error"})
    assert s["state"] == "picking", s["message"]
    assert s["info"]["frames"] == 24

    frame = client.get(f"/api/jobs/{job_id}/frame/0?w=120")
    assert frame.status_code == 200 and frame.headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/jobs/{job_id}/frame/0?w=-1").status_code == 422

    good = {"frame": 0, "auto": [{"x": 230, "y": 190, "frame": 0}]}
    r = client.post(f"/api/jobs/{job_id}/holds", json=good)
    assert r.status_code == 200, r.text
    assert len(r.json()["holds"]) >= 3                    # the red holds are found by colour

    for bad in ({"auto": [{"x": "left"}]}, {"tol": 9999}, {"surprise": 1}, {"starts": [[1]]},
                {"vols_manual": [{"kind": "poly"}]}):
        r = client.post(f"/api/jobs/{job_id}/holds", json={"frame": 0, **bad})
        assert r.status_code in (400, 422), (bad, r.status_code)
    r = client.post(f"/api/jobs/{job_id}/holds", json={"frame": 500})
    assert r.status_code == 400 and "24 frames" in r.json()["detail"]

    assert client.post(f"/api/jobs/{job_id}/start", json={"model": "huge"}).status_code == 422
    assert client.delete(f"/api/jobs/{job_id}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert not any(jobs.WORKSPACE.iterdir())


def test_picker_cache_stays_within_budget(wall_video, tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "WORKSPACE", tmp_path / "ws")
    monkeypatch.setattr(jobs, "CACHE_MB", 2)              # a 480x360 colour picture is ~2 MB
    jobs.reset_workspace()
    job = jobs.Job("wall.mp4")
    job.input.write_bytes(wall_video.read_bytes())
    job.open(still=True)
    for i in range(10):
        job.lab(i)
    assert job.cache_bytes <= 2 * 1024 * 1024 and len(job.cache) == 1
    job.release()
    assert job.cache_bytes == 0 and job.frame(0) is None
