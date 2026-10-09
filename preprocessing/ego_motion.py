"""Robot poses and ego-motion.

Pose file rows (documented): `ts x y z qw qx qy qz` - the rigid transform of the sensor frame into the
trajectory origin. We read that as T_world_os1 (maps os1 coordinates to world). THIS DIRECTION IS AN
ASSUMPTION until `check_pose_convention` is run on real consecutive scans (it scores both directions).

LiDAR deskewing: CODa's 3d_raw .bin has no per-point time, so we cannot deskew ourselves. Use the
dataset's own ego-compensated `3d_comp` scans (lidar_source: 3d_comp) for motion compensation.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from .coords import apply_T


def load_poses(path):
    poses = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if poses.shape[1] != 8:
        raise ValueError(f"{path}: expected 8 columns (ts x y z qw qx qy qz), got {poses.shape[1]}")
    if np.any(np.diff(poses[:, 0]) < 0):
        raise ValueError(f"{path}: pose timestamps are not monotonic")
    return poses


def pose_row_to_matrix(row):
    T = np.eye(4)
    T[:3, 3] = row[1:4]
    qw, qx, qy, qz = row[4:8]
    T[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()   # scipy order is x,y,z,w
    return T


def interpolate_pose(poses, t, max_extrapolation_s=0.0):
    """4x4 pose at time t: linear translation, slerp rotation. Raises outside the pose time span."""
    ts = poses[:, 0]
    if t < ts[0] - max_extrapolation_s or t > ts[-1] + max_extrapolation_s:
        raise ValueError(f"t={t:.6f} outside pose span [{ts[0]:.6f}, {ts[-1]:.6f}]")
    t = float(np.clip(t, ts[0], ts[-1]))
    i = int(np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 2))
    t0, t1 = ts[i], ts[i + 1]
    a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    q = poses[[i, i + 1], 4:8][:, [1, 2, 3, 0]]            # -> x,y,z,w
    rot = Slerp([0.0, 1.0], R.from_quat(q))([a])[0]
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = (1 - a) * poses[i, 1:4] + a * poses[i + 1, 1:4]
    return T


def relative_transform(T_world_a, T_world_b):
    """Maps points in frame b into frame a."""
    return np.linalg.inv(T_world_a) @ T_world_b


def to_world(points_os1, T_world_os1):
    return apply_T(T_world_os1, points_os1)


def check_pose_convention(scan_a, scan_b, T_a, T_b, max_pts=20000, seed=0):
    """Score both pose interpretations on two nearby real scans (os1-frame (N,3) clouds).

    Mean nearest-neighbour distance between scan_a and scan_b after mapping both into the world.
    The correct interpretation should score markedly lower. Returns (score_T, score_T_inverse).
    """
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)

    def sub(p):
        p = p[np.isfinite(p).all(1) & (np.abs(p).sum(1) > 0)]
        return p[rng.choice(len(p), min(len(p), max_pts), replace=False)]

    a, b = sub(scan_a), sub(scan_b)
    out = []
    for fa, fb in ((T_a, T_b), (np.linalg.inv(T_a), np.linalg.inv(T_b))):
        d, _ = cKDTree(apply_T(fa, a)).query(apply_T(fb, b), k=1)
        out.append(float(np.mean(np.minimum(d, 5.0))))
    return tuple(out)
