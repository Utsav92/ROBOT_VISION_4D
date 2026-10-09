import cv2
import numpy as np

from preprocessing.stereo_depth import (depth_to_rect_points, disparity_to_depth, sgbm_disparity)


def test_lidar_to_camera_roundtrip(calib):
    rng = np.random.default_rng(1)
    p = rng.uniform(-20, 20, (500, 3))
    assert np.allclose(calib.rect0_to_os1(calib.os1_to_rect0(p)), p, atol=1e-9)


def test_reprojection_of_known_point(calib):
    # a point straight ahead on the rectified optical axis offset by the principal point projects to (cx, cy)
    uv, z = calib.project_rect0(np.array([[0.0, 0.0, 10.0]]))
    assert np.allclose(uv[0], [calib.cam0.cx, calib.cam0.cy]) and z[0] == 10.0
    uv, _ = calib.project_rect0(np.array([[1.0, -0.5, 10.0]]))
    assert np.allclose(uv[0], [calib.cam0.cx + calib.cam0.fx / 10, calib.cam0.cy - calib.cam0.fy / 20])


def test_points_behind_camera_are_invalid(calib):
    uv, z = calib.project_rect0(np.array([[0.0, 0.0, -3.0]]))
    assert np.isnan(uv).all()


def test_disparity_to_depth_with_offset(calib):
    fx, B, off = calib.cam0.fx, calib.baseline_m, calib.disparity_offset_px
    Z = np.array([[2.0, 10.0, 30.0, 100.0]], dtype=np.float32)
    disp = (fx * B / Z + off).astype(np.float32)
    depth, ok = disparity_to_depth(disp, np.ones_like(disp, bool), fx, B, off, 0.5, 40.0)
    assert np.allclose(depth[0, :3], Z[0, :3], rtol=1e-4)
    assert ok.tolist() == [[True, True, True, False]]                    # 100 m is beyond max_depth


def test_nonpositive_disparity_rejected(calib):
    disp = np.array([[0.0, -3.0, 1.0]], dtype=np.float32)
    depth, ok = disparity_to_depth(disp, np.ones_like(disp, bool), 700.0, 0.2, 5.0, 0.5, 40.0)
    assert not ok.any() and (depth == 0).all()


def test_backprojection_roundtrip(calib):
    H, W = 64, 96
    depth = np.full((H, W), 8.0, np.float32)
    pos, ok, (vs, us) = depth_to_rect_points(depth, np.ones((H, W), bool), calib.cam0.P, stride=4)
    uv, _ = calib.project_rect0(pos.reshape(-1, 3))
    u, v = np.meshgrid(us, vs)
    assert np.allclose(uv[:, 0], u.ravel(), atol=1e-6) and np.allclose(uv[:, 1], v.ravel(), atol=1e-6)


def test_sgbm_recovers_synthetic_disparity():
    rng = np.random.default_rng(0)
    H, W, d_true = 160, 320, 24
    base = cv2.GaussianBlur(rng.integers(0, 255, (H, W + d_true), dtype=np.uint8), (3, 3), 0)
    left = base[:, :W]                              # a feature at x_left appears at x_left - d in the right image
    right = base[:, d_true:]
    disp, valid = sgbm_disparity(left, right, {"num_disparities": 64, "block_size": 7})
    core = (slice(20, H - 20), slice(d_true + 30, W - 20))
    frac = np.mean(np.abs(disp[core] - d_true) < 1.0)
    assert frac > 0.9, f"only {frac:.2f} of pixels within 1px of the true disparity"
