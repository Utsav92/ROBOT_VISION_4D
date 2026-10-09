"""Pure-numpy camera/world-reference logic (no TouchDesigner needed)."""
import importlib.util
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

TD = Path(__file__).resolve().parents[1] / "touchdesigner"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TD / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cc = _load("camera_controller")
wr = _load("world_reference")


def _fwd(W):
    return -W[:3, 2]


def test_look_at_points_camera_at_target_with_orthonormal_basis():
    W = cc.look_at([3, 4, 5], [0, 1, 0])
    assert np.allclose(W[:3, :3] @ W[:3, :3].T, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(W[:3, :3]), 1)
    assert np.allclose(_fwd(W), (np.array([0, 1, 0]) - [3, 4, 5]) / np.linalg.norm([3, 3, 5]))
    assert np.allclose(W[:3, 3], [3, 4, 5])


def test_look_straight_down_has_valid_basis():
    W = cc.pose_birdseye([1, 0, 2], height=45)
    assert np.allclose(_fwd(W), [0, -1, 0], atol=1e-9) and np.isclose(np.linalg.det(W[:3, :3]), 1)
    assert np.allclose(W[:3, 1], [0, 0, -1], atol=1e-9)           # screen-up is world -z
    assert np.allclose(W[:3, 3], [1, 45, 2])


def test_quaternion_roundtrip_and_slerp_match_scipy():
    rng = np.random.default_rng(1)
    for _ in range(25):
        A, B = R.random(random_state=rng.integers(1e9)), R.random(random_state=rng.integers(1e9))
        assert np.allclose(cc.quat_to_mat(cc.mat_to_quat(A.as_matrix())), A.as_matrix(), atol=1e-9)
        a = rng.random()
        ours = cc.quat_to_mat(cc.slerp(cc.mat_to_quat(A.as_matrix()), cc.mat_to_quat(B.as_matrix()), a))
        ref = Slerp([0, 1], R.concatenate([A, B]))([a])[0].as_matrix()
        assert np.allclose(ours, ref, atol=1e-9)


def test_blend_endpoints_and_midpoint():
    Wa = cc.look_at([0, 0, 0], [0, 0, -1])
    Wb = cc.look_at([10, 4, 0], [10, 4, -5])
    assert np.allclose(cc.blend_pose(Wa, Wb, 0), Wa, atol=1e-9) and np.allclose(cc.blend_pose(Wa, Wb, 1), Wb, atol=1e-9)
    assert np.allclose(cc.blend_pose(Wa, Wb, 0.5)[:3, 3], [5, 2, 0])


def test_ease_is_monotone_zero_slope_at_ends():
    xs = np.linspace(0, 1, 101)
    ys = np.array([cc.ease(x) for x in xs])
    assert ys[0] == 0 and ys[-1] == 1 and np.all(np.diff(ys) >= 0) and (ys[1] - ys[0]) < 1e-3


def test_robot_pov_is_the_sensor_pose():
    P = np.eye(4); P[:3, :3] = R.from_euler("y", 40, degrees=True).as_matrix(); P[:3, 3] = [5, 1, -7]
    assert np.allclose(cc.pose_robot_pov(P), P)
    assert np.allclose(_fwd(P), R.from_euler("y", 40, degrees=True).apply([0, 0, -1]))      # forward is local -z


def test_chase_is_behind_in_heading_and_above():
    P = np.eye(4); P[:3, :3] = R.from_euler("y", 90, degrees=True).as_matrix(); P[:3, 3] = [0, 1, 0]
    heading = R.from_euler("y", 90, degrees=True).apply([0, 0, -1])                         # robot forward in world
    W = cc.pose_chase(P, back=7.0, height=3.5, look_ahead=4.0)
    behind = W[:3, 3] - P[:3, 3]
    assert np.isclose(np.dot(behind[[0, 2]], heading[[0, 2]]), -7.0) and np.isclose(behind[1], 3.5)
    assert np.dot(_fwd(W)[[0, 2]], heading[[0, 2]]) > 0                                      # camera looks the robot's way


def test_chase_ignores_robot_pitch():
    P = np.eye(4); P[:3, :3] = R.from_euler("x", 20, degrees=True).as_matrix()
    assert np.isclose(cc.pose_chase(P)[1, 3], 3.5)


def test_orbit_keeps_radius_height_and_looks_at_target():
    tgt = np.array([4.0, 0.0, -9.0])
    for t in (0.0, 7.5, 21.0):
        W = cc.pose_orbit(tgt, t, radius=12, height=5, period_s=30)
        d = W[:3, 3] - tgt
        assert np.isclose(np.hypot(d[0], d[2]), 12) and np.isclose(d[1], 5)
        assert np.dot(_fwd(W), (tgt + [0, 1, 0] - W[:3, 3])) > 0
    a, b = cc.pose_orbit(tgt, 0, period_s=30)[:3, 3], cc.pose_orbit(tgt, 30, period_s=30)[:3, 3]
    assert np.allclose(a, b)                                                                 # full period closes the loop


def test_cinematic_target_visits_robot_then_objects_with_eased_handover():
    robot, objs = [0, 0, 0], [[10, 0, 0], [0, 0, 10]]
    f = lambda t: cc.cinematic_target(robot, objs, t, dwell_s=8, handover_s=2)
    assert np.allclose(f(0), robot) and np.allclose(f(5.9), robot)
    assert np.allclose(f(8.0), objs[0]) and np.allclose(f(14), objs[0]) and np.allclose(f(16.0), objs[1])
    mid = f(7.0)                                                                             # halfway through hand-over
    assert np.allclose(mid, [5, 0, 0], atol=1e-9)
    xs = [f(t)[0] for t in np.linspace(5.9, 8.1, 40)]
    assert np.all(np.diff(xs) >= -1e-12)                                                     # no overshoot / jump
    assert np.allclose(cc.cinematic_target(robot, [], 123.0), robot)                         # no moving objects: stay on robot


def test_free_cam_matrix_roundtrip_and_movement():
    W = cc.look_at([2, 3, 4], [2, 3, -6])
    fc = cc.FreeCam(); fc.from_matrix(W)
    assert np.allclose(fc.matrix()[:3, :3], W[:3, :3], atol=1e-9) and np.allclose(fc.pos, [2, 3, 4])
    fc.step(1.0, fwd=1.0, speed=5.0)
    assert np.allclose(fc.pos, [2, 3, -1], atol=1e-9)                                       # moved 5 m along the view direction
    fc.step(1.0, strafe=1.0, speed=2.0)
    assert np.allclose(fc.pos, [4, 3, -1], atol=1e-9)                                       # +x is right when facing -z
    fc.step(1.0, vert=1.0, speed=3.0)
    assert np.isclose(fc.pos[1], 6.0)
    fc.step(1.0, dyaw=math.pi / 2)
    assert np.allclose(fc.forward(), [-1, 0, 0], atol=1e-9)                                  # yaw left turns toward -x
    fc.step(0, dpitch=10.0)
    assert fc.pitch <= 1.5                                                                    # pitch clamped (no flip)


def test_free_cam_from_pitched_view():
    W = cc.look_at([0, 0, 0], [0, 5, -5])
    fc = cc.FreeCam(); fc.from_matrix(W)
    assert np.isclose(fc.pitch, math.radians(45)) and np.isclose(fc.yaw, 0)


def test_fov_from_real_intrinsics():
    assert math.isclose(cc.fov_from_intrinsics(769.33, 1224), math.degrees(2 * math.atan(612 / 769.33)), abs_tol=1e-9)
    assert math.isclose(cc.fov_from_intrinsics(769.33, 1224), 77.0, abs_tol=0.05)


def test_grid_is_snapped_and_fits_extent():
    g = wr.grid_lines((12.3, -7.9), half_extent=20, spacing=5, ground_y=-0.6)
    assert g.shape == (2 * 9, 2, 3) and np.allclose(g[:, :, 1], -0.6)
    xs = np.unique(g[::2, 0, 0])
    assert np.allclose(xs % 5, 0) and xs.min() == 10 - 20 and xs.max() == 10 + 20          # snapped to the 5 m lattice
    g2 = wr.grid_lines((11.0, -7.9), 20, 5, 0)                                                # same rounding cell (10): unchanged
    assert np.allclose(np.unique(g2[::2, 0, 0]), xs)


def test_ground_height_uses_low_percentile_of_nearby_points():
    rng = np.random.default_rng(0)
    ground = np.c_[rng.uniform(-5, 5, 500), rng.normal(-0.6, 0.01, 500), rng.uniform(-5, 5, 500)]
    wall = np.c_[rng.uniform(-5, 5, 500), rng.uniform(0, 3, 500), np.full(500, 4.0)]
    far = np.c_[np.full(50, 100.0), np.full(50, -50.0), np.full(50, 100.0)]
    pts = np.vstack([ground, wall, far]); valid = np.ones(len(pts))
    assert abs(wr.ground_height(pts, valid, (0, 0)) + 0.6) < 0.05
