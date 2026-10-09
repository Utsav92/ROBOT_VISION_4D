import numpy as np

from preprocessing.dynamic_points import dynamic_mask, moving_instances, points_in_box
from preprocessing.object_annotations import parse_boxes


def test_points_in_oriented_box_respects_rotation():
    c, lwh = [0, 0, 0], [4.0, 1.0, 2.0]
    p = np.array([[1.8, 0.0, 0.0], [0.0, 1.8, 0.0], [1.2, 1.2, 0.0]])
    assert points_in_box(p, c, lwh, [0, 0, 0], margin=0).tolist() == [True, False, False]
    # rotate 90 deg about z: the long axis now points along y
    assert points_in_box(p, c, lwh, [0, 0, np.pi / 2], margin=0).tolist() == [False, True, False]
    # 45 deg: the diagonal point (1.2, 1.2) lies on the long axis
    assert points_in_box(p, c, lwh, [0, 0, np.pi / 4], margin=0)[2]


def test_moving_vs_static_instances_by_world_displacement():
    c = {"Pedestrian:1": [[0, 0, 0], [0.6, 0, 0], [1.4, 0.2, 0]], "Bollard:1": [[5, 5, 0]] * 3,
         "Bike:2": [[0, 0, 0], [0.3, 0, 0]], "Car:1": [[1, 1, 1]]}
    assert moving_instances(c, 1.0) == {"Pedestrian:1"}


def _boxes():
    return parse_boxes({"3dbbox": [
        {"classId": "Pedestrian", "instanceId": "Pedestrian:1", "labelAttributes": {}, "cX": 0, "cY": 0, "cZ": 1,
         "h": 2, "l": 1, "w": 1, "r": 0, "p": 0, "y": 0},
        {"classId": "Bollard", "instanceId": "Bollard:1", "labelAttributes": {}, "cX": 10, "cY": 0, "cZ": 1,
         "h": 2, "l": 1, "w": 1, "r": 0, "p": 0, "y": 0},
        {"classId": "Pedestrian", "instanceId": "", "labelAttributes": {}, "cX": -10, "cY": 0, "cZ": 1,
         "h": 2, "l": 1, "w": 1, "r": 0, "p": 0, "y": 0}]})


def test_dynamic_mask_only_moving_boxes_and_class_fallback_for_unidentified():
    pts = np.array([[0, 0, 1.0], [10, 0, 1.0], [-10, 0, 1.0], [50, 0, 0], [np.nan, 0, 0]])
    mask, ids = dynamic_mask(pts, _boxes(), {"Pedestrian:1"})
    assert mask.tolist() == [True, False, True, False, False]      # bollard static; id-less pedestrian via class fallback
    assert ids[0] == 1.0 and ids[2] == -2.0 and ids[1] == -1.0
