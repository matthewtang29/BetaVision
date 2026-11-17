import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.dont_write_bytecode = True


@pytest.fixture
def video(tmp_path):
    """A tiny synthetic clip (still camera): a grey wall with a few coloured 'holds'."""
    path = tmp_path / "wall.avi"
    w, h, n = 320, 240, 12
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (w, h))
    for i in range(n):
        f = np.full((h, w, 3), 120, np.uint8)
        for k, (x, y) in enumerate([(60, 60), (160, 100), (240, 180), (100, 190)]):
            cv2.circle(f, (x, y), 18, (30, 60, 220), -1)
        cv2.circle(f, (20 + 20 * i, 30), 8, (200, 200, 50), -1)
        out.write(f)
    out.release()
    return path


@pytest.fixture
def jobs_mod(tmp_path, monkeypatch):
    from betavision_app import jobs
    monkeypatch.setattr(jobs, "WORKSPACE", tmp_path / "workspace")
    (tmp_path / "workspace").mkdir()
    return jobs


def wait_for(job, state="picking", timeout=30):
    end = time.time() + timeout
    while job.state != state and time.time() < end:
        time.sleep(0.05)
    assert job.state == state, f"job is {job.state!r}: {job.message}"


@pytest.fixture
def open_job(jobs_mod, video):
    """A Job opened on the synthetic clip, ready for the picker."""
    def make(worker=None):
        job = (worker.new_job("wall.avi") if worker else jobs_mod.Job("wall.avi"))
        job.input.write_bytes(video.read_bytes())
        job.open(True)
        wait_for(job)
        return job
    return make
