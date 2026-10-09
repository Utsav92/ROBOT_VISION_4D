import numpy as np

from preprocessing.calibration import load_calibration, rigid_inverse
from preprocessing.coords import M_LIDAR_TO_TD, T_lidar_to_td, lidar_to_td


def test_loads_documented_example_values(calib):
    assert calib.cam0.width == 1224 and calib.cam0.height == 1024
    assert np.isclose(calib.cam0.fx, 730.93414758424547)
    assert np.isclose(calib.cam0.cx, 606.62505340576172)
    assert calib.cam0.R_rect.shape == (3, 3) and calib.T_cam0_os1.shape == (4, 4)
    assert np.allclose(calib.T_cam1_cam0[:3, 3], [-0.197673440251558, 0.00128769601558891, 0.00253652872125049])


def test_baseline_and_offset_come_from_projection_matrices(calib):
    assert np.isclose(calib.baseline_m, 0.2, atol=1e-6)               # -Tx/fx of the synthetic right camera
    assert np.isclose(calib.disparity_offset_px, 606.62505340576172 - 600.0)


def test_baseline_unit_mismatch_in_real_coda_files_is_corrected(calib):
    # Real CODa seq 0: P1[0,3] = -0.151076 with fx = 769.33 -> naive 0.000196 m; cam0_to_cam1 |T| = 0.1964 m.
    calib.cam0.P[0, 0] = calib.cam1.P[0, 0] = 769.3315414491426
    calib.cam1.P[0, 3] = -0.15107602233212056
    calib.T_cam1_cam0[:3, 3] = [-0.19637310339252165, 0, 0]
    assert np.isclose(calib.baseline_m, 0.19637310339252165, rtol=1e-6)
    assert "x1000" in calib.baseline_source


def test_inconsistent_baseline_raises(calib):
    import pytest
    calib.T_cam1_cam0[:3, 3] = [-5.0, 0, 0]
    with pytest.raises(ValueError):
        calib.baseline_m


def test_missing_file_is_reported(tmp_path):
    import pytest
    (tmp_path / "x").mkdir()
    with pytest.raises(FileNotFoundError):
        load_calibration(tmp_path / "x")


def test_documented_extrinsic_is_rigid(calib):
    R = calib.T_cam0_os1[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-4)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-4)


def test_rigid_inverse(calib):
    T = calib.T_cam0_os1
    assert np.allclose(rigid_inverse(T) @ T, np.eye(4), atol=1e-9)


def test_handedness_and_axis_conversion():
    assert np.isclose(np.linalg.det(M_LIDAR_TO_TD), 1.0)                       # rotation, no mirror
    assert np.allclose(lidar_to_td([1, 0, 0]), [0, 0, -1])                     # forward -> -z
    assert np.allclose(lidar_to_td([0, 1, 0]), [-1, 0, 0])                     # left -> -x
    assert np.allclose(lidar_to_td([0, 0, 1]), [0, 1, 0])                      # up -> +y
    a, b = np.array([1.0, 2, 3]), np.array([-2.0, 0.5, 4])
    assert np.allclose(np.cross(lidar_to_td(a), lidar_to_td(b)), lidar_to_td(np.cross(a, b)))  # cross product preserved


def test_pose_conjugation_matches_point_transform():
    T = np.eye(4)
    T[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    T[:3, 3] = [1, 2, 3]
    p = np.array([0.5, -0.25, 2.0])
    direct = lidar_to_td(T[:3, :3] @ p + T[:3, 3])
    Td = T_lidar_to_td(T)
    assert np.allclose(Td[:3, :3] @ lidar_to_td(p) + Td[:3, 3], direct)
