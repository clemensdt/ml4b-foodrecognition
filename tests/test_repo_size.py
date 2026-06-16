"""Repository-size guards for dataset/model hygiene."""
from pathlib import Path
import subprocess

import pytest


def _tracked_files() -> list[Path]:
    try:
        out = subprocess.check_output(["git", "ls-files", "-z"])
    except Exception as exc:  # pragma: no cover - depends on running inside git
        pytest.skip(f"git ls-files unavailable: {exc}")
    return [Path(p.decode()) for p in out.split(b"\0") if p]


def test_tracked_payload_stays_under_500_mib():
    files = _tracked_files()
    total = sum(p.stat().st_size for p in files if p.exists())
    assert total < 500 * 1024 * 1024


def test_no_large_model_weights_are_tracked():
    tracked = {str(p) for p in _tracked_files()}
    assert not any(p.startswith("models/") for p in tracked)
    assert not any(p.startswith("data/ECUSTFD/") for p in tracked)
    assert not any(p.endswith((".pt", ".pth", ".onnx", ".ckpt")) for p in tracked)
