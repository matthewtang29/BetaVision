"""Shared test helpers: a small synthetic climbing-wall video (no real footage needed)."""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True


def make_wall_video(path, frames=24, size=(480, 360), fps=12):
    """A beige wall with red 'holds' and a moving dark blob, written with OpenCV."""
    w, h = size
    rng = np.random.default_rng(0)
    wall = np.full((h, w, 3), (200, 210, 215), np.uint8)
    wall = cv2.add(wall, rng.integers(0, 20, (h, w, 3), dtype=np.uint8))       # a bit of texture
    for x, y in [(80, 300), (150, 240), (230, 190), (300, 130), (380, 70)]:
        cv2.ellipse(wall, (x, y), (16, 12), 20, 0, 360, (40, 40, 200), -1)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(frames):
        f = wall.copy()
        cv2.circle(f, (100 + 10 * i, 280 - 8 * i), 18, (30, 30, 30), -1)
        out.write(f)
    out.release()
    return path


@pytest.fixture(scope="session")
def wall_video(tmp_path_factory):
    return make_wall_video(tmp_path_factory.mktemp("video") / "wall.mp4")


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


@pytest.fixture(autouse=True)
def library_in_tmp(tmp_path, monkeypatch):
    """No test may write to (or read) the real ~/.climb_analyzer/library."""
    monkeypatch.setenv("BETAVISION_LIBRARY", str(tmp_path / "test-library"))


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
