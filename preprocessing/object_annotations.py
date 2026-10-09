"""CODa 3D bounding boxes.

JSON (documented): {"3dbbox": [{"classId", "instanceId": "Class:N", "labelAttributes": {"isOccluded"},
cX,cY,cZ, l,w,h, r,p,y}]}. Boxes are in the ego-LiDAR frame, 9-DoF, NOT axis-aligned.
Orientation follows the devkit: scipy R.from_euler("xyz", [r, p, y]) (extrinsic -> R = Rz(y) Ry(p) Rx(r)).
Corner order follows the devkit's get_3dbbox_corners: 0-3 bottom ring, 4-7 top ring.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from .coords import apply_T

# 12 edges of the box wireframe over the corner ordering above.
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]

_X = np.array([-1, -1, 1, 1])
_Y = np.array([1, -1, -1, 1])


@dataclass
class Box:
    class_id: str
    instance_id: str          # full "Class:N" string, or "" when absent
    track_number: object      # int, or None when the dataset supplies no identity
    occlusion: str
    center: np.ndarray
    lwh: np.ndarray
    rpy: np.ndarray
    corners: np.ndarray = field(default=None)

    @property
    def id_available(self):
        return self.track_number is not None

    @property
    def label(self):
        n = f" {self.track_number:03d}" if self.id_available else " ID?"
        return f"{self.class_id.upper()}{n}"


def box_corners(center, lwh, rpy):
    l, w, h = lwh
    local = np.zeros((8, 3))
    for c in range(8):
        i = c % 4
        local[c] = (_X[i] * l / 2, _Y[i] * w / 2, (-1 if c < 4 else 1) * h / 2)
    Rm = R.from_euler("xyz", list(rpy), degrees=False).as_matrix()
    return local @ Rm.T + np.asarray(center)


def parse_instance(instance_id):
    if instance_id and ":" in instance_id:
        tail = instance_id.rsplit(":", 1)[1]
        if tail.isdigit():
            return int(tail)
    return None


def parse_boxes(d):
    boxes = []
    for a in d.get("3dbbox", []):
        center = np.array([a["cX"], a["cY"], a["cZ"]], dtype=np.float64)
        lwh = np.array([a["l"], a["w"], a["h"]], dtype=np.float64)
        rpy = np.array([a["r"], a["p"], a["y"]], dtype=np.float64)
        inst = a.get("instanceId", "") or ""
        boxes.append(Box(a["classId"], inst, parse_instance(inst),
                         a.get("labelAttributes", {}).get("isOccluded", "Unknown"),
                         center, lwh, rpy, box_corners(center, lwh, rpy)))
    return boxes


def load_boxes(path):
    return parse_boxes(json.loads(Path(path).read_text()))


def boxes_to_frame(boxes, T):
    """Return copies with corners/center mapped by 4x4 T (e.g. LiDAR->world). Orientation is carried by corners."""
    out = []
    for b in boxes:
        nb = Box(b.class_id, b.instance_id, b.track_number, b.occlusion,
                 apply_T(T, b.center[None])[0], b.lwh, b.rpy, apply_T(T, b.corners))
        out.append(nb)
    return out


def boxes_to_json(boxes):
    return [{"class": b.class_id, "instance": b.instance_id, "track": b.track_number,
             "id_available": b.id_available, "label": b.label, "occlusion": b.occlusion,
             "center": b.center.tolist(), "lwh": b.lwh.tolist(), "corners": b.corners.tolist()}
            for b in boxes]
