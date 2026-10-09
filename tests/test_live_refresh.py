"""The renderer-side loader following a manifest that a writer keeps extending (live mode)."""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_loader_follows_a_growing_manifest_written_by_the_real_exporter(tmp_path):
    from preprocessing.frame_exporter import FrameExporter
    fl = _load(ROOT / "touchdesigner" / "frame_loader.py", "frame_loader_live")
    ex = FrameExporter(tmp_path)
    xyz = np.zeros((128, 512, 3))
    ok = np.ones((128, 512), bool)

    def add(i):
        ex.write_frame(i, i, 100.0 + 0.1 * i, xyz + i, ok, np.zeros((128, 512, 4)), np.zeros((128, 512, 4)),
                       np.zeros((4, 4, 3)), np.zeros((4, 4), bool), np.zeros((4, 4, 3), np.uint8), [], False, np.eye(4), {})
        ex.write_manifest({"sequence": -1, "live": True})

    add(0)
    add(1)
    assert fl.set_root(tmp_path) == 2 and fl.refresh() is False                    # nothing new yet
    add(2)
    add(3)
    assert fl.refresh() is True and fl.n_frames() == 4
    assert np.isclose(fl.timestamps()[-1], 100.3) and fl.get("lidar_xyz", 3)[0, 0, 0] == 3.0     # newest frame is readable
    assert fl.refresh() is False


def test_loader_keeps_the_old_manifest_if_the_new_one_is_unreadable(tmp_path):
    from preprocessing.frame_exporter import FrameExporter
    fl = _load(ROOT / "touchdesigner" / "frame_loader.py", "frame_loader_live2")
    ex = FrameExporter(tmp_path)
    ex.write_manifest({"sequence": -1})                                          # zero frames
    fl.set_root(tmp_path)
    (tmp_path / "manifest.json").write_text('{"n_frames": 3, "fra')                # a torn write
    assert fl.refresh() is False and fl.n_frames() == 0
    (tmp_path / "manifest.json").unlink()
    assert fl.refresh() is False                                                  # missing file is not an error either
    assert json.loads('{"ok": 1}')["ok"] == 1
