"""Stereo-vs-LiDAR depth disagreement.

E = |Z_stereo - Z_lidar|,  E_rel = E / max(Z_lidar, eps)   (both Z measured along the rectified cam0 axis)

Classes (uint8): 0 unavailable (orange), 1 good (green), 2 moderate (yellow), 3 large (red).
Thresholds are initial VISUALISATION thresholds, not verified sensor-accuracy specs. Disagreement does not
by itself prove the stereo algorithm wrong: calibration, occlusion, reflectivity, sync error and moving
objects all contribute.

Visibility: a LiDAR return is only compared if no markedly nearer LiDAR return projects within a small pixel
window (a min-depth z-buffer), which removes points seen "through" nearer surfaces.
"""
from dataclasses import dataclass

import cv2
import numpy as np

UNAVAILABLE, GOOD, MODERATE, LARGE = 0, 1, 2, 3


@dataclass
class ComparisonResult:
    cls: np.ndarray        # (N,) uint8
    abs_err: np.ndarray    # (N,) float32, NaN where unavailable
    rel_err: np.ndarray    # (N,) float32, NaN where unavailable
    z_lidar: np.ndarray    # (N,) float32 rect-frame depth (NaN if behind camera)
    z_stereo: np.ndarray   # (N,) float32, NaN if no stereo sample
    stats: dict


def classify(abs_err, z_lidar, cfg, fx=None, baseline=None):
    good, large = cfg["good_below_m"], cfg["large_above_m"]
    if cfg.get("scale_with_depth") and fx and baseline:
        extra = cfg.get("disparity_sigma_px", 0.5) * z_lidar ** 2 / (fx * baseline)
        good, large = good + extra, large + extra
    cls = np.full(abs_err.shape, UNAVAILABLE, dtype=np.uint8)
    fin = np.isfinite(abs_err)
    cls[fin & (abs_err < good)] = GOOD
    cls[fin & (abs_err >= good) & (abs_err <= large)] = MODERATE
    cls[fin & (abs_err > large)] = LARGE
    return cls


def visibility_mask(u, v, z, shape, window, tol_abs, tol_rel):
    """True for points not occluded by a nearer return within `window` px."""
    H, W = shape
    zbuf = np.full((H, W), np.inf, dtype=np.float32)
    np.minimum.at(zbuf, (v, u), z.astype(np.float32))
    k = int(window) | 1
    local_min = cv2.erode(zbuf, np.ones((k, k), np.uint8))
    return z <= local_min[v, u] + np.maximum(tol_abs, tol_rel * z)


def compare_lidar_to_stereo(lidar_os1, depth_stereo, stereo_valid, calib, cfg):
    """lidar_os1: (N,3) in the LiDAR frame (invalid points as NaN or 0). depth_stereo/valid: HxW on cam0."""
    N = len(lidar_os1)
    H, W = depth_stereo.shape
    nan = np.full(N, np.nan, dtype=np.float32)
    res = ComparisonResult(np.zeros(N, np.uint8), nan.copy(), nan.copy(), nan.copy(), nan.copy(), {})

    finite = np.isfinite(lidar_os1).all(1) & (np.abs(lidar_os1).sum(1) > 0)
    p_rect = np.full((N, 3), np.nan)
    p_rect[finite] = calib.os1_to_rect0(lidar_os1[finite])
    uv, z = calib.project_rect0(p_rect)
    res.z_lidar[:] = np.where(z > 0, z, np.nan)

    ui, vi = np.round(uv[:, 0]), np.round(uv[:, 1])
    inview = finite & (z > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    idx = np.flatnonzero(inview)
    u, v = ui[idx].astype(int), vi[idx].astype(int)

    vis = visibility_mask(u, v, z[idx], (H, W), cfg["occlusion_window_px"],
                          cfg["occlusion_tol_m"], cfg["occlusion_tol_rel"])
    zs = depth_stereo[v, u]
    have = vis & stereo_valid[v, u] & (zs > 0)
    cmp_idx = idx[have]
    res.z_stereo[idx] = np.where(stereo_valid[v, u] & (zs > 0), zs, np.nan)

    e = np.abs(zs[have] - z[cmp_idx]).astype(np.float32)
    res.abs_err[cmp_idx] = e
    res.rel_err[cmp_idx] = e / np.maximum(z[cmp_idx], cfg.get("epsilon_m", 0.05))
    res.cls = classify(res.abs_err, np.nan_to_num(res.z_lidar), cfg, calib.cam0.fx, calib.baseline_m)

    n_in = int(inview.sum())
    res.stats = {
        "lidar_points_in_view": n_in,
        "comparable_points": int(len(cmp_idx)),
        "valid_comparison_pct": float(100.0 * len(cmp_idx) / n_in) if n_in else 0.0,
        "mean_abs_diff_m": float(e.mean()) if len(e) else float("nan"),
        "median_abs_diff_m": float(np.median(e)) if len(e) else float("nan"),
        "p90_abs_diff_m": float(np.percentile(e, 90)) if len(e) else float("nan"),
        "large_disagreement_points": int((res.cls == LARGE).sum()),
    }
    return res
