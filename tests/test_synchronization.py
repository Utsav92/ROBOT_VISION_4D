import numpy as np
import pytest

from preprocessing.ego_motion import interpolate_pose, load_poses, pose_row_to_matrix, check_pose_convention
from preprocessing.synchronize import (estimate_rate, frame_at_time, load_timestamps, match_streams,
                                       nearest_indices)


def test_nearest_indices_signed_dt():
    idx, dt = nearest_indices([0.04, 0.26], [0.0, 0.1, 0.2, 0.3])
    assert idx.tolist() == [0, 3] or idx.tolist() == [0, 2]
    assert np.allclose(dt, np.array([0.0, 0.1, 0.2, 0.3])[idx] - np.array([0.04, 0.26]))


def test_match_streams_rejects_far_and_non_mutual():
    a = np.array([0.0, 0.1, 0.2, 0.9])
    b = np.array([0.001, 0.102, 0.5])
    ia, ib = match_streams(a, b, 0.02)
    assert ia.tolist() == [0, 1] and ib.tolist() == [0, 1]          # 0.2 and 0.9 have no partner within 20 ms


def test_rate_estimate():
    r = estimate_rate(np.arange(0, 10, 0.1) + 1673884185.0)
    assert abs(r["mean_hz"] - 10.0) < 1e-6 and r["n"] == 100


def test_timestamps_must_increase(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("1.0\n2.0\n2.0\n")
    with pytest.raises(ValueError):
        load_timestamps(p)


def test_frame_at_time_playback():
    ts = np.array([0.0, 0.1, 0.25, 0.3])
    assert [frame_at_time(ts, t) for t in (-1, 0.0, 0.12, 0.26, 5)] == [0, 0, 1, 2, 3]


def _poses():
    # constant yaw rate 90 deg/s about z and 1 m/s forward-ish translation
    from scipy.spatial.transform import Rotation as R
    rows = []
    for t in (0.0, 1.0, 2.0):
        q = R.from_euler("z", 90 * t, degrees=True).as_quat()      # x y z w
        rows.append([t, t, 2 * t, 0.0, q[3], q[0], q[1], q[2]])
    return np.array(rows)


def test_pose_interpolation_midpoint():
    T = interpolate_pose(_poses(), 0.5)
    assert np.allclose(T[:3, 3], [0.5, 1.0, 0.0])
    yaw = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
    assert abs(yaw - 45.0) < 1e-6


def test_pose_interpolation_matches_exact_rows():
    P = _poses()
    assert np.allclose(interpolate_pose(P, 1.0), pose_row_to_matrix(P[1]))


def test_pose_outside_span_raises():
    with pytest.raises(ValueError):
        interpolate_pose(_poses(), 3.0)


def test_pose_convention_check_prefers_correct_direction():
    # a rigid wall-like cloud seen from two poses; correct T_world_os1 must align them
    rng = np.random.default_rng(3)
    world = rng.uniform(-10, 10, (4000, 3))
    from scipy.spatial.transform import Rotation as R
    def pose(yaw, t):
        T = np.eye(4); T[:3, :3] = R.from_euler("z", yaw, degrees=True).as_matrix(); T[:3, 3] = t; return T
    Ta, Tb = pose(10, [1, 0, 0]), pose(25, [3, 1, 0])
    obs = lambda T: (world - T[:3, 3]) @ T[:3, :3]                 # world -> sensor frame
    s_T, s_inv = check_pose_convention(obs(Ta), obs(Tb), Ta, Tb)
    assert s_T < 1e-6 < s_inv
