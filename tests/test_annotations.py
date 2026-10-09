import numpy as np
from scipy.spatial.transform import Rotation as R

from preprocessing.coords import lidar_to_td
from preprocessing.object_annotations import (EDGES, boxes_to_frame, boxes_to_json, box_corners, parse_boxes,
                                              parse_instance)

SAMPLE = {"3dbbox": [
    {"classId": "Pedestrian", "instanceId": "Pedestrian:12", "labelAttributes": {"isOccluded": "None"},
     "cX": 5.0, "cY": -2.0, "cZ": 0.9, "h": 1.8, "l": 0.6, "w": 0.5, "r": 0.0, "p": 0.0, "y": 0.7},
    {"classId": "Bike Rack", "instanceId": "", "labelAttributes": {},
     "cX": 1.0, "cY": 1.0, "cZ": 0.5, "h": 1.0, "l": 2.0, "w": 0.4, "r": 0.1, "p": -0.2, "y": 2.0}]}


def test_dimensions_preserved_under_rotation():
    for b in parse_boxes(SAMPLE):
        c = b.corners
        assert np.isclose(np.linalg.norm(c[1] - c[2]), b.lwh[0])      # edge 1-2 spans length (x)
        assert np.isclose(np.linalg.norm(c[0] - c[1]), b.lwh[1])      # edge 0-1 spans width (y)
        assert np.isclose(np.linalg.norm(c[0] - c[4]), b.lwh[2])      # vertical edge spans height (z)
        assert np.allclose(c.mean(0), b.center)


def test_yaw_rotates_box_about_z_not_axis_aligned():
    c = box_corners([0, 0, 0], [2.0, 1.0, 1.0], [0, 0, np.pi / 2])
    # a 90 deg yaw swaps the x/y extents
    assert np.isclose(np.ptp(c[:, 0]), 1.0) and np.isclose(np.ptp(c[:, 1]), 2.0)
    c45 = box_corners([0, 0, 0], [2.0, 1.0, 1.0], [0, 0, np.pi / 4])
    assert np.isclose(np.ptp(c45[:, 0]), (2.0 + 1.0) / np.sqrt(2))    # genuinely oblique: (l+w)/sqrt2


def test_euler_convention_is_extrinsic_xyz_as_in_devkit():
    r, p, y = 0.3, -0.2, 1.1
    expect = R.from_euler("xyz", [r, p, y]).as_matrix()
    Rz = R.from_euler("z", y).as_matrix(); Ry = R.from_euler("y", p).as_matrix(); Rx = R.from_euler("x", r).as_matrix()
    assert np.allclose(expect, Rz @ Ry @ Rx)
    c = box_corners([0, 0, 0], [2.0, 1.0, 1.0], [r, p, y])
    local = box_corners([0, 0, 0], [2.0, 1.0, 1.0], [0, 0, 0])
    assert np.allclose(c, local @ expect.T)


def test_twelve_unique_edges_of_unit_length_structure():
    assert len(EDGES) == 12 and len({tuple(sorted(e)) for e in EDGES}) == 12
    b = parse_boxes(SAMPLE)[0]
    lens = sorted(round(float(np.linalg.norm(b.corners[i] - b.corners[j])), 6) for i, j in EDGES)
    assert lens == sorted([0.6] * 4 + [0.5] * 4 + [1.8] * 4)


def test_identity_is_marked_unavailable_not_invented():
    ped, rack = parse_boxes(SAMPLE)
    assert ped.track_number == 12 and ped.label == "PEDESTRIAN 012" and ped.id_available
    assert rack.track_number is None and not rack.id_available and rack.label.endswith("ID?")
    assert parse_instance("Informational Sign:1") == 1 and parse_instance("") is None


def test_boxes_rigidly_follow_transform():
    b = parse_boxes(SAMPLE)
    T = np.eye(4); T[:3, :3] = R.from_euler("z", 0.5).as_matrix(); T[:3, 3] = [10, -4, 1]
    moved = boxes_to_frame(b, T)
    for a, m in zip(b, moved):
        assert np.isclose(np.linalg.norm(a.corners[0] - a.corners[6]), np.linalg.norm(m.corners[0] - m.corners[6]))
        assert np.allclose(m.corners.mean(0), T[:3, :3] @ a.center + T[:3, 3])
    js = boxes_to_json(moved)
    assert js[0]["label"] == "PEDESTRIAN 012" and len(js[0]["corners"]) == 8
