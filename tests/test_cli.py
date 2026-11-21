"""The command-line tool still starts, and the pose model download can't leave a broken file."""

import subprocess
import sys
import urllib.request

import pytest

import climb_pose as cp
from conftest import ROOT


@pytest.mark.parametrize("script", ["climb_analyze.py", "climb_pose.py", "climb_holds.py"])
def test_cli_help(script):
    r = subprocess.run([sys.executable, str(ROOT / script), "--help"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "usage" in r.stdout.lower()


def test_cli_missing_video_message(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "climb_analyze.py"), "nope.mp4"],
                       capture_output=True, text=True, cwd=tmp_path, timeout=120)
    assert r.returncode != 0 and "Can't find a video" in r.stderr


def test_interrupted_model_download_leaves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(cp.Path, "home", lambda: tmp_path)

    def cut_off(url, path):
        open(path, "wb").write(b"half a model")
        raise ConnectionResetError("connection reset")
    monkeypatch.setattr(urllib.request, "urlretrieve", cut_off)
    with pytest.raises(RuntimeError, match="internet"):
        cp.ensure_model("lite")
    assert not any((tmp_path / ".climb_analyzer").iterdir())
