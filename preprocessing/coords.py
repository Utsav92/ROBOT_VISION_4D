"""Coordinate conventions.

CODa LiDAR / robot base : x forward, y left,  z up      (right-handed)   [documented]
CODa camera (and rect)  : x right,   y down,  z forward (right-handed)   [documented]
TouchDesigner world     : x right,   y up,    z toward viewer (right-handed; cameras look down -z)

M_LIDAR_TO_TD maps a CODa LiDAR-convention vector into TouchDesigner axes:
    forward (+x) -> -z_td,  left (+y) -> -x_td,  up (+z) -> +y_td
Its determinant is +1, so it is a pure rotation (no mirroring).
"""
import numpy as np

M_LIDAR_TO_TD = np.array([[0.0, -1.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [-1.0, 0.0, 0.0]])


def lidar_to_td(points):
    """(...,3) points/vectors in LiDAR convention -> TouchDesigner axes."""
    return np.asarray(points, dtype=np.float64) @ M_LIDAR_TO_TD.T


def T_lidar_to_td(T):
    """Re-express a 4x4 pose that lives in LiDAR-convention axes in TouchDesigner axes."""
    M4 = np.eye(4)
    M4[:3, :3] = M_LIDAR_TO_TD
    return M4 @ T @ M4.T


def homogeneous(points):
    pts = np.asarray(points, dtype=np.float64)
    return np.concatenate([pts, np.ones(pts.shape[:-1] + (1,))], axis=-1)


def apply_T(T, points):
    """Apply a 4x4 rigid transform to (...,3) points."""
    pts = np.asarray(points, dtype=np.float64)
    return pts @ T[:3, :3].T + T[:3, 3]
