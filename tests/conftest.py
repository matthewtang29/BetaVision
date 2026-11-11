"""Shared test helpers: a small synthetic climbing-wall video (no real footage needed)."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


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
