"""Bridge logic that needs no ROS. The rclpy node itself is NOT covered (no ROS 2 on this machine)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sensor_bridge", ROOT / "optional_ros2" / "sensor_bridge.py")
sb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sb)


def _cloud(scan, step=20, ring=True, big=False, intensity_name="intensity", itype=7):
    n = scan.shape[0] * scan.shape[1]
    e = ">" if big else "<"
    names, fm, off = ["x", "y", "z", intensity_name], [e + "f4"] * 3 + [e + sb._PF[itype]], [0, 4, 8, 12]
    if ring:
        names.append("ring"); fm.append(e + "u2"); off.append(16)
    rec = np.zeros(n, dtype=np.dtype({"names": names, "formats": fm, "offsets": off, "itemsize": step}))
    flat = scan.reshape(-1, 4)
    for k, nm in enumerate(["x", "y", "z", intensity_name]):
        rec[nm] = flat[:, k]
    fields = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7), (intensity_name, 12, itype)] + ([("ring", 16, 4)] if ring else [])
    return rec.tobytes(), fields, step


def _scan(seed=0):
    return np.random.default_rng(seed).uniform(-20, 20, (128, 1024, 4)).astype(np.float32)


def test_pointcloud2_decoding_with_padding_extra_fields_and_endianness():
    scan = _scan()
    for kw in ({}, {"step": 32}, {"ring": False}, {"big": True}):
        data, fields, step = _cloud(scan, **kw)
        out = sb.pointcloud2_to_scan(data, fields, step, 128, 1024, is_bigendian=kw.get("big", False))
        assert out.shape == (128, 1024, 4) and out.dtype == np.float32 and np.array_equal(out, scan)


def test_pointcloud2_reflectivity_integer_field_and_errors():
    scan = _scan(1)
    scan[..., 3] = np.random.default_rng(2).integers(0, 255, (128, 1024))
    data, fields, step = _cloud(scan, intensity_name="reflectivity", itype=4)
    assert np.array_equal(sb.pointcloud2_to_scan(data, fields, step, 128, 1024)[..., 3], scan[..., 3])
    with pytest.raises(ValueError, match="organised"):
        sb.pointcloud2_to_scan(data, fields, step, 1, 131072)
    with pytest.raises(ValueError, match="'z'"):
        sb.pointcloud2_to_scan(data, [f for f in fields if f[0] != "z"], step, 128, 1024)
    no_int = [f for f in fields if f[0] != "reflectivity"]
    assert np.all(sb.pointcloud2_to_scan(data, no_int, step, 128, 1024)[..., 3] == 0)          # no intensity field -> zeros


def test_image_decoding_stride_and_encodings():
    rng = np.random.default_rng(3)
    bgr = rng.integers(0, 255, (6, 5, 3), dtype=np.uint8)
    padded = np.concatenate([bgr.reshape(6, 15), np.zeros((6, 9), np.uint8)], axis=1)           # step 24 > 15 bytes of pixels
    assert np.array_equal(sb.image_to_bgr(padded.tobytes(), "bgr8", 6, 5, 24), bgr)
    assert np.array_equal(sb.image_to_bgr(bgr[..., ::-1].tobytes(), "rgb8", 6, 5, 15), bgr)
    bgra = np.concatenate([bgr, np.full((6, 5, 1), 255, np.uint8)], axis=2)
    assert np.array_equal(sb.image_to_bgr(bgra.tobytes(), "bgra8", 6, 5, 20), bgr)
    rgba = bgra[..., [2, 1, 0, 3]]
    assert np.array_equal(sb.image_to_bgr(rgba.tobytes(), "rgba8", 6, 5, 20), bgr)
    g = rng.integers(0, 255, (6, 5), dtype=np.uint8)
    assert np.array_equal(sb.image_to_bgr(g.tobytes(), "mono8", 6, 5, 5), np.repeat(g[..., None], 3, axis=2))
    with pytest.raises(ValueError):
        sb.image_to_bgr(b"", "16UC1", 1, 1, 2)


def test_odometry_pose_roundtrip_and_composition():
    from preprocessing.ego_motion import pose_row_to_matrix
    from scipy.spatial.transform import Rotation as R
    q = R.from_euler("xyz", [0.1, -0.2, 0.9]).as_quat()
    T = sb.odom_to_T((1.0, -2.0, 0.5), q)
    assert np.allclose(pose_row_to_matrix(np.array(sb.T_to_row(12.5, T))), T, atol=1e-12)
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-12) and np.allclose(T[:3, 3], [1, -2, 0.5])


def test_pose_buffer_interpolates_and_ignores_out_of_order():
    pb = sb.PoseBuffer()
    for t, x in ((0.0, 0.0), (1.0, 2.0), (0.5, 99.0), (2.0, 4.0)):                              # 0.5 arrives late: ignored
        T = np.eye(4)
        T[0, 3] = x
        pb.push(t, T)
    assert len(pb.rows) == 3 and pb.latest() == 2.0
    assert np.isclose(pb.at(0.5)[0, 3], 1.0) and np.isclose(pb.at(1.5)[0, 3], 3.0)
    with pytest.raises(ValueError):
        pb.at(3.0)


def test_stamp_sync_matches_nearest_within_tolerance_and_waits_for_late_streams():
    s = sb.StampSync(("left", "right"), max_dt=0.05)
    s.push("lidar", 1.0, "scan")
    s.push("left", 1.003, "L")
    assert s.pop_ready() == []                                  # right has not arrived yet: hold the scan
    s.push("right", 1.020, "R")
    out = s.pop_ready()
    assert len(out) == 1 and out[0]["left"] == "L" and out[0]["right"] == "R" and out[0]["lidar"] == "scan"


def test_stamp_sync_rejects_far_matches_and_drops_stale_scans():
    s = sb.StampSync(("left", "right"), max_dt=0.05, keep_s=1.0)
    s.push("lidar", 1.0, "a")
    s.push("left", 1.0, "L")
    s.push("right", 1.2, "R")                                   # right is 200 ms away
    assert s.pop_ready() == [] and s.dropped == 1
    s.push("lidar", 5.0, "b")
    s.push("left", 5.0, "L2")                                   # right never delivers
    assert s.pop_ready() == []
    s.push("left", 7.0, "L3")                                   # time moves on: give up on 'b'
    assert s.pop_ready() == [] and s.dropped == 2 and len(s.buf["lidar"]) == 0


def _backlog():
    s = sb.StampSync(("left", "right"), max_dt=0.05)
    for k in range(3):
        t = 1.0 + 0.1 * k
        s.push("lidar", t, f"scan{k}")
        s.push("left", t, f"L{k}")
        s.push("right", t, f"R{k}")
    return s


def test_stamp_sync_latest_only_discards_backlog():
    s = _backlog()
    out = s.pop_ready(latest_only=True)
    assert [o["lidar"] for o in out] == ["scan2"] and s.dropped == 2
    assert [o["lidar"] for o in _backlog().pop_ready(latest_only=False)] == ["scan0", "scan1", "scan2"]


def test_rectifier_with_identity_geometry_is_a_no_op(calib):
    cam = calib.cam0
    cam.dist = np.zeros(5)
    cam.R_rect = np.eye(3)
    cam.K = cam.P[:, :3].copy()
    img = np.random.default_rng(4).integers(0, 255, (cam.height, cam.width, 3), dtype=np.uint8)
    assert np.array_equal(sb.Rectifier(cam)(img), img)


def test_live_session_writes_frames_and_a_readable_manifest(calib, cfg, tmp_path):
    cfg = json.loads(json.dumps(cfg))
    cfg["stereo"]["stride"] = 8
    sess = sb.LiveSession(cfg, calib, tmp_path, max_frames=2, log=lambda *a: None)
    h, w = calib.cam0.height, calib.cam0.width
    left = np.random.default_rng(5).integers(0, 255, (h, w, 3), dtype=np.uint8)
    scan = _scan()
    T = np.eye(4)
    T[0, 3] = 1.5
    sess.ingest(left, np.roll(left, -12, axis=1), scan, T, 100.0)
    sess.ingest(left, np.roll(left, -12, axis=1), scan, T, 100.1)
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert m["live"] is True and m["n_frames"] == 2 and [f["timestamp"] for f in m["frames"]] == [100.0, 100.1]
    assert not (tmp_path / "manifest.json.tmp").exists()                                          # atomic replace left nothing behind
    from preprocessing.coords import T_lidar_to_td
    assert np.allclose(np.load(tmp_path / "pose" / "00000.npy"), T_lidar_to_td(T @ calib.T_base_os1))   # world<-os1 = world<-base @ base<-os1
    assert np.load(tmp_path / "lidar_xyz" / "00001.npy").shape[:2] == (128, 512)
    with pytest.raises(RuntimeError, match="max_frames"):
        sess.ingest(left, left, scan, T, 100.2)


@pytest.mark.skipif(not (ROOT / "data/coda/calibrations/0").exists() or not (ROOT / "output/processed/seq0/manifest.json").exists(),
                    reason="needs the downloaded CODa subset and its offline export")
def test_live_path_reproduces_the_offline_export_on_real_data():
    rep = sb.selftest(config=str(ROOT / "config.yaml"), root=str(ROOT / "data/coda"), out=str(ROOT / "output/_live_selftest"),
                      frames=(400, 401))
    assert rep["frames_ingested"] == 2 and rep["compared_to_offline"] and rep["matches_offline"], rep
