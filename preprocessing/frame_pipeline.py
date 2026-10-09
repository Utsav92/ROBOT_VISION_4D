"""The per-frame perception pipeline, independent of where the data comes from.

Both the recorded-dataset path (process_sequence.py) and the live ROS 2 path (optional_ros2/sensor_bridge.py) call
`process_frame`, so the rendering side never knows which mode produced a frame.

Inputs : rectified left/right BGR images, one organised LiDAR scan (128,1024,4), the world<-os1 pose, calibration.
Outputs: everything FrameExporter.write_frame needs, in TouchDesigner axes and the common world frame.
"""
from dataclasses import dataclass, field

import numpy as np

from . import lidar_loader as LL
from .coords import T_lidar_to_td, apply_T, lidar_to_td
from .depth_comparison import compare_lidar_to_stereo
from .dynamic_points import dynamic_mask
from .object_annotations import boxes_to_frame, boxes_to_json
from .sensor_fusion import sample_colors, stereo_to_frame, voxel_keep_mask
from .stereo_depth import depth_to_rect_points, disparity_to_depth, smooth_disparity


@dataclass
class FrameResult:
    lidar_xyz: np.ndarray        # (128, C, 3) world, TD axes
    lidar_ok: np.ndarray         # (128, C) bool
    lidar_attr: np.ndarray       # (128, C, 4) intensity01, class, abs err, rel err
    lidar_dyn: np.ndarray        # (128, C, 4) moving flag, track id, 0, 0
    stereo_xyz: np.ndarray       # (h, w, 3) world, TD axes
    stereo_ok: np.ndarray        # (h, w) bool
    stereo_rgb: np.ndarray       # (h, w, 3) uint8
    boxes_json: list = field(default_factory=list)
    annotated: bool = False
    pose_td: np.ndarray = None   # (4,4)
    stats: dict = field(default_factory=dict)


def _box_to_td(b):
    b.center = lidar_to_td(b.center[None])[0]
    b.corners = lidar_to_td(b.corners)
    return b


def process_frame(left_bgr, right_bgr, scan, T_w_os1, calib, cfg, disparity_fn, os1_boxes=None, moving=frozenset()):
    st, dc, lc = cfg["stereo"], cfg["depth_error"], cfg["lidar"]

    disp, ok = disparity_fn(left_bgr, right_bgr)
    if st["smooth"]:
        disp, ok = smooth_disparity(disp, ok)
    depth, dok = disparity_to_depth(disp, ok, calib.cam0.fx, calib.baseline_m, calib.disparity_offset_px,
                                    st["min_depth_m"], st["max_depth_m"])

    lok = LL.valid_mask(scan, lc["min_range_m"], lc["max_range_m"])
    lidar_flat = np.where(lok[..., None], scan[..., :3], np.nan).reshape(-1, 3)
    cmp = compare_lidar_to_stereo(lidar_flat, depth, dok, calib, dc)

    pts_rect, sok, (vs, us) = depth_to_rect_points(depth, dok, calib.cam0.P, st["stride"])
    sok = voxel_keep_mask(pts_rect, sok, st["voxel_size_m"])
    pts_os1 = stereo_to_frame(pts_rect, calib)
    pts_world = apply_T(T_w_os1, pts_os1.reshape(-1, 3)).reshape(pts_os1.shape)
    stereo_td = lidar_to_td(pts_world)
    rgb = sample_colors(left_bgr, vs, us)

    cs = lc["column_stride"]
    w_xyz = lidar_to_td(apply_T(T_w_os1, np.nan_to_num(scan[..., :3]).reshape(-1, 3))).reshape(scan.shape[:2] + (3,))
    inten = LL.normalize_intensity(scan[..., 3], lok, lc["intensity_percentiles"])
    attr = np.stack([inten, cmp.cls.reshape(lok.shape).astype(np.float32),
                     np.nan_to_num(cmp.abs_err.reshape(lok.shape), nan=-1.0),
                     np.nan_to_num(cmp.rel_err.reshape(lok.shape), nan=-1.0)], axis=-1)

    boxes_json, annotated = [], False
    dyn = np.zeros(scan.shape[:2] + (4,), dtype=np.float32)
    dyn[..., 1] = -1.0
    if os1_boxes is not None:
        annotated = True
        dm, did = dynamic_mask(lidar_flat, os1_boxes, moving)
        dyn[..., 0] = dm.reshape(lok.shape)
        dyn[..., 1] = did.reshape(lok.shape)
        boxes_json = boxes_to_json([_box_to_td(b) for b in boxes_to_frame(os1_boxes, T_w_os1)])

    return FrameResult(
        lidar_xyz=LL.decimate_columns(w_xyz, cs), lidar_ok=LL.decimate_columns(lok, cs),
        lidar_attr=LL.decimate_columns(attr, cs), lidar_dyn=LL.decimate_columns(dyn, cs),
        stereo_xyz=stereo_td, stereo_ok=sok, stereo_rgb=rgb, boxes_json=boxes_json, annotated=annotated,
        pose_td=T_lidar_to_td(T_w_os1), stats=cmp.stats)


def write_result(exporter, i, frame_id, timestamp, r):
    """Persist a FrameResult through FrameExporter (shared by the dataset and live paths)."""
    exporter.write_frame(i, frame_id, timestamp, r.lidar_xyz, r.lidar_ok, r.lidar_attr, r.lidar_dyn, r.stereo_xyz,
                         r.stereo_ok, r.stereo_rgb, r.boxes_json, r.annotated, r.pose_td, r.stats)
