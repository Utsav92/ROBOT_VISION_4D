import json
from pathlib import Path

import numpy as np

from preprocessing import lidar_loader as LL
from preprocessing.coda_paths import CodaPaths, longest_consecutive_run
from preprocessing.frame_exporter import FrameExporter
from preprocessing.sensor_fusion import voxel_keep_mask


def test_scan_reader_shape_and_extra_fields(tmp_path):
    n = LL.RINGS * LL.COLUMNS
    a = np.random.default_rng(0).random((n, 4), dtype=np.float32)
    a.tofile(tmp_path / "s4.bin")
    assert LL.read_scan(tmp_path / "s4.bin").shape == (128, 1024, 4)
    np.concatenate([a, np.zeros((n, 2), np.float32)], 1).tofile(tmp_path / "s6.bin")   # extra time/ring fields
    assert np.allclose(LL.read_scan(tmp_path / "s6.bin"), a.reshape(128, 1024, 4))
    np.zeros(100, np.float32).tofile(tmp_path / "bad.bin")
    try:
        LL.read_scan(tmp_path / "bad.bin"); assert False
    except ValueError:
        pass


def test_invalid_measurements_are_masked():
    scan = np.ones((128, 1024, 4), np.float32) * 5
    scan[0, 0] = np.nan
    scan[0, 1, :3] = 0                                   # no-return
    scan[0, 2, :3] = [500, 0, 0]                         # beyond range
    ok = LL.valid_mask(scan, 0.5, 80.0)
    assert not ok[0, :3].any() and ok[0, 3:].all()


def test_intensity_normalisation_is_bounded():
    inten = np.linspace(0, 1000, 100, dtype=np.float32)
    out = LL.normalize_intensity(inten, np.ones(100, bool))
    assert out.min() >= 0 and out.max() <= 1


def test_exporter_roundtrip_and_manifest(tmp_path):
    ex = FrameExporter(tmp_path)
    xyz = np.random.default_rng(0).random((128, 512, 3))
    ok = np.ones((128, 512), bool); ok[3, 3] = False
    ex.write_frame(0, 5000, 1673884185.5, xyz, ok, np.zeros((128, 512, 4)), np.zeros((128, 512, 4)), np.zeros((20, 30, 3)),
                   np.zeros((20, 30), bool), np.zeros((20, 30, 3), np.uint8), [{"label": "CAR 001"}], True,
                   np.eye(4), {"comparable_points": 7})
    ex.write_manifest({"sequence": 0})
    lx = np.load(tmp_path / "lidar_xyz" / "00000.npy")
    assert lx.dtype == np.float32 and lx.shape == (128, 512, 4)
    assert lx[3, 3, 3] == 0 and (lx[3, 3, :3] == 0).all() and lx[0, 0, 3] == 1
    assert np.load(tmp_path / "stereo_rgb" / "00000.npy")[..., 3].min() == 255
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["frames"][0]["timestamp"] == 1673884185.5 and m["n_frames"] == 1       # real timestamp preserved
    assert json.loads((tmp_path / "boxes" / "00000.json").read_text())["annotated"] is True


def test_voxel_filter_keeps_one_point_per_voxel():
    pts = np.zeros((4, 4, 3)); pts[..., 0] = np.repeat(np.arange(4)[:, None] * 0.01, 4, 1)
    ok = np.ones((4, 4), bool)
    keep = voxel_keep_mask(pts, ok, 1.0)
    assert keep.sum() == 1 and voxel_keep_mask(pts, ok, 0.0).sum() == 16


def test_frame_discovery_and_run_selection(tmp_path):
    root = tmp_path
    for cam in ("cam0", "cam1"):
        d = root / "2d_rect" / cam / "0"; d.mkdir(parents=True)
        for f in (10, 11, 12, 20, 21):
            (d / f"2d_rect_{cam}_0_{f}.jpg").write_bytes(b"x")
    d = root / "3d_comp" / "os1" / "0"; d.mkdir(parents=True)
    for f in (10, 11, 12, 20, 21, 22):
        (d / f"3d_comp_os1_0_{f}.bin").write_bytes(b"x")
    b = root / "3d_bbox" / "os1" / "0"; b.mkdir(parents=True)
    (b / "3d_bbox_os1_0_11.json").write_text("{}")
    p = CodaPaths(root, 0)
    assert p.available_frames() == [10, 11, 12, 20, 21]          # 22 lacks images -> excluded
    assert p.annotated_frames() == [11] and p.bbox(11) is not None and p.bbox(10) is None
    assert longest_consecutive_run([10, 11, 12, 20, 21], 10) == [10, 11, 12]
    assert longest_consecutive_run([10, 11, 12, 20, 21], 2) == [10, 11]


def test_exporter_writes_precomputed_echo_layer(tmp_path):
    ex = FrameExporter(tmp_path)
    xyz = np.random.default_rng(1).random((128, 512, 3))
    ok = np.ones((128, 512), bool)
    dyn = np.zeros((128, 512, 4), np.float32)
    dyn[10, 20, 0] = 1; dyn[11, 21, 0] = 1
    ok[11, 21] = False                                      # moving but invalid -> must not appear
    ex.write_frame(0, 1, 1.0, xyz, ok, np.zeros((128, 512, 4)), dyn, np.zeros((4, 4, 3)), np.zeros((4, 4), bool),
                   np.zeros((4, 4, 3), np.uint8), [], False, np.eye(4), {})
    full, echo = np.load(tmp_path / "lidar_xyz" / "00000.npy"), np.load(tmp_path / "lidar_echo" / "00000.npy")
    assert echo[..., 3].sum() == 1 and echo[10, 20, 3] == 1 and echo[11, 21, 3] == 0
    assert np.array_equal(echo[..., :3], full[..., :3]) and full[..., 3].sum() == 128 * 512 - 1   # positions untouched


def _loader(tmp_path, n=10, mb=1):
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location("frame_loader_t", Path(__file__).resolve().parents[1] / "touchdesigner" / "frame_loader.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    (tmp_path / "lidar_xyz").mkdir()
    for i in range(n):
        np.save(tmp_path / "lidar_xyz" / f"{i:05d}.npy", np.full((mb * 65536, 4), i, dtype=np.float32))   # mb MiB each
    (tmp_path / "manifest.json").write_text(json.dumps({"n_frames": n, "frames": [{"timestamp": float(i), "stats": {}} for i in range(n)]}))
    m.set_root(tmp_path)
    return m


def test_frame_loader_cache_is_bounded_by_bytes_and_returns_correct_frames(tmp_path):
    m = _loader(tmp_path)
    m.MAX_BYTES = int(3.5 * 1024 * 1024)                   # room for three 1 MiB frames
    for i in range(10):
        assert m.get("lidar_xyz", i)[0, 0] == i
    assert len(m._cache) == 3 and m._bytes <= m.MAX_BYTES
    assert [k[1] for k in m._cache] == [7, 8, 9]            # oldest evicted first
    assert m.get("lidar_xyz", 8)[0, 0] == 8 and list(m._cache)[-1] == ("lidar_xyz", 8)      # hit refreshes recency
    assert m.get("lidar_xyz", 999)[0, 0] == 9 and m.get("lidar_xyz", -5)[0, 0] == 0          # clamped to the sequence


def test_prefetch_warms_next_frames_and_never_runs_twice_at_once(tmp_path):
    import time
    m = _loader(tmp_path, n=8)
    m.prefetch(0, kinds=("lidar_xyz",), ahead=3)
    m._prefetching = True                                   # a second request while busy is ignored, not queued
    m.prefetch(4, kinds=("lidar_xyz",), ahead=3)
    for _ in range(100):
        if ("lidar_xyz", 3) in m._cache and not m._prefetching:
            break
        time.sleep(0.02)
    assert all(("lidar_xyz", k) in m._cache for k in (1, 2, 3)) and ("lidar_xyz", 5) not in m._cache
