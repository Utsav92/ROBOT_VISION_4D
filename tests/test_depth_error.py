import numpy as np

from preprocessing.depth_comparison import (GOOD, LARGE, MODERATE, UNAVAILABLE, classify,
                                            compare_lidar_to_stereo, visibility_mask)


def test_threshold_classes_and_boundaries(cfg):
    e = np.array([0.0, 0.19, 0.20, 0.35, 0.50, 0.51, np.nan], dtype=np.float32)
    c = classify(e, np.full(7, 10.0), cfg["depth_error"])
    assert c.tolist() == [GOOD, GOOD, MODERATE, MODERATE, MODERATE, LARGE, UNAVAILABLE]


def test_depth_scaled_thresholds_grow_with_range(cfg):
    d = dict(cfg["depth_error"], scale_with_depth=True)
    e = np.array([0.6, 0.6], dtype=np.float32)
    near, far = classify(e, np.array([2.0, 40.0]), d, fx=730.0, baseline=0.2)
    assert near == LARGE and far != LARGE


def test_visibility_removes_points_behind_nearer_return():
    u = np.array([50, 51]); v = np.array([40, 40]); z = np.array([5.0, 20.0])
    vis = visibility_mask(u, v, z, (100, 100), 7, 0.3, 0.03)
    assert vis.tolist() == [True, False]                       # the 20 m return is hidden behind the 5 m one


def _scene(calib, stereo_bias):
    """A fronto-parallel wall 12 m ahead, scanned densely, with stereo depth = 12 + bias."""
    H, W = calib.cam0.height, calib.cam0.width
    ys, xs = np.mgrid[-1.5:1.5:0.1, -3.0:3.0:0.1]
    rect = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, 12.0)], 1)
    lidar = calib.rect0_to_os1(rect)
    depth = np.full((H, W), 12.0 + stereo_bias, np.float32)
    return lidar, depth, np.ones((H, W), bool)


def test_agreement_gives_zero_error_and_all_good(calib, cfg):
    lidar, depth, ok = _scene(calib, 0.0)
    r = compare_lidar_to_stereo(lidar, depth, ok, calib, cfg["depth_error"])
    assert r.stats["comparable_points"] == len(lidar) and r.stats["valid_comparison_pct"] == 100.0
    assert np.nanmax(r.abs_err) < 1e-3 and (r.cls == GOOD).all()


def test_known_bias_is_measured_and_flagged(calib, cfg):
    lidar, depth, ok = _scene(calib, 0.7)
    r = compare_lidar_to_stereo(lidar, depth, ok, calib, cfg["depth_error"])
    assert np.isclose(r.stats["median_abs_diff_m"], 0.7, atol=1e-3)
    assert np.isclose(np.nanmax(r.rel_err), 0.7 / 12.0, atol=1e-3)
    assert (r.cls == LARGE).all() and r.stats["large_disagreement_points"] == len(lidar)


def test_invalid_stereo_and_invalid_lidar_are_unavailable(calib, cfg):
    lidar, depth, ok = _scene(calib, 0.0)
    ok[:] = False
    lidar[:5] = np.nan
    lidar[5:8] = 0.0
    r = compare_lidar_to_stereo(lidar, depth, ok, calib, cfg["depth_error"])
    assert r.stats["comparable_points"] == 0 and (r.cls == UNAVAILABLE).all()
    assert np.isnan(r.abs_err).all()


def test_points_behind_camera_never_compared(calib, cfg):
    rect = np.array([[0.0, 0.0, -5.0], [0.0, 0.0, 10.0]])
    lidar = calib.rect0_to_os1(rect)
    depth = np.full((calib.cam0.height, calib.cam0.width), 10.0, np.float32)
    r = compare_lidar_to_stereo(lidar, depth, np.ones_like(depth, bool), calib, cfg["depth_error"])
    assert r.cls[0] == UNAVAILABLE and r.cls[1] == GOOD
