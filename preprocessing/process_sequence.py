"""Offline pipeline: CODa sequence -> TouchDesigner-ready frames.

    python -m preprocessing.process_sequence --config config.yaml [--sequence 0] [--frames 200]

Per frame: stereo disparity -> metric depth -> rect-cam0 points; LiDAR scan; stereo-vs-LiDAR comparison;
3D boxes; everything mapped into one world frame via the robot pose, then into TouchDesigner axes.
Frames without a matching pose / annotation are reported, never invented.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import lidar_loader as LL
from .calibration import load_calibration
from .coda_paths import CodaPaths, longest_consecutive_run
from .coords import T_lidar_to_td, lidar_to_td, apply_T, M_LIDAR_TO_TD
from .dynamic_points import moving_instances
from .frame_pipeline import process_frame, write_result
from .ego_motion import load_poses, interpolate_pose
from .frame_exporter import FrameExporter
from .object_annotations import load_boxes, boxes_to_frame, boxes_to_json
from .sensor_fusion import sample_colors, stereo_to_frame, voxel_keep_mask
from .stereo_depth import (depth_to_rect_points, disparity_to_depth, make_disparity_fn, smooth_disparity)
from .synchronize import estimate_rate, load_timestamps


def process(cfg, sequence=None, n_frames=None, start=None, log=print):
    ds = cfg["dataset"]
    seq = ds["sequence"] if sequence is None else sequence
    paths = CodaPaths(Path(ds["root"]), seq, ds["image_source"], ds["lidar_source"], ds["pose_source"])
    calib = load_calibration(paths.calib_dir, cfg["calibration"]["apply_rectification_rotation"])

    ts = load_timestamps(paths.timestamps_file)
    poses = load_poses(paths.pose_file())
    avail = paths.available_frames()
    if not avail:
        raise FileNotFoundError(f"No frames with rect images + LiDAR under {paths.root} for sequence {seq}")
    ann = set(paths.annotated_frames())
    want = n_frames or ds["frame_count"]
    pool = avail if start is None else [f for f in avail if f >= start]
    if start is None and ds["frame_start"] is None and ann:
        annotated_avail = [f for f in avail if f in ann]
        pool = annotated_avail or avail
    elif ds["frame_start"] is not None and start is None:
        pool = [f for f in avail if f >= ds["frame_start"]]
    frames = longest_consecutive_run(pool, want)
    log(f"[process] sequence {seq}: {len(frames)} frames ({frames[0]}..{frames[-1]}), "
        f"{sum(f in ann for f in frames)} annotated")

    disparity_fn = make_disparity_fn(cfg["stereo"], log)
    out_dir = Path(cfg["export"]["out_root"]) / f"seq{seq}"
    exporter = FrameExporter(out_dir)

    # pass 1: which annotated instances really move (world-frame centre displacement over the window)
    centres = {}
    for f in frames:
        bp = paths.bbox(f)
        if bp is None:
            continue
        T = interpolate_pose(poses, float(ts[f]), max_extrapolation_s=cfg["sync"]["max_dt_s"])
        for b in load_boxes(bp):
            if b.id_available:
                centres.setdefault(b.instance_id, []).append(apply_T(T, b.center[None])[0])
    moving = moving_instances(centres, cfg["temporal"]["moving_min_disp_m"])
    log(f"[process] {len(moving)} of {len(centres)} identified instances move >= {cfg['temporal']['moving_min_disp_m']} m")

    for i, f in enumerate(frames):
        t = float(ts[f])
        T_w_os1 = interpolate_pose(poses, t, max_extrapolation_s=cfg["sync"]["max_dt_s"])
        left = cv2.imread(str(paths.image("cam0", f)))
        right = cv2.imread(str(paths.image("cam1", f)))
        if left is None or right is None:
            raise IOError(f"frame {f}: could not read rectified images")
        bp = paths.bbox(f)
        res = process_frame(left, right, LL.read_scan(paths.lidar(f)), T_w_os1, calib, cfg, disparity_fn,
                            load_boxes(bp) if bp is not None else None, moving)
        write_result(exporter, i, f, t, res)
        if i % 10 == 0:
            log(f"[process] {i + 1}/{len(frames)} frame {f}  cmp={res.stats['comparable_points']} "
                f"median_err={res.stats['median_abs_diff_m']:.3f}m")

    sel_ts = ts[frames]
    exporter.write_manifest({
        "sequence": int(seq), "coordinate_frame": "world (trajectory origin), TouchDesigner axes",
        "axes": "x right, y up, z toward viewer", "M_lidar_to_td": M_LIDAR_TO_TD.tolist(),
        "measured_rate": estimate_rate(sel_ts), "calibration": {
            "baseline_m": calib.baseline_m, "disparity_offset_px": calib.disparity_offset_px,
            "fx": calib.cam0.fx, "apply_rectification_rotation": calib.apply_rect_rotation},
        "config": cfg, "moving_instances": sorted(moving),
        "assumptions": ["os1->cam0 extrinsic targets the unrectified cam0 frame (verify via calibration_overlay)",
                        "pose rows are T_world_os1 (verify via inspect_dataset --check-poses)"],
    })
    log(f"[process] wrote {len(frames)} frames to {out_dir}")
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--sequence", type=int)
    ap.add_argument("--frames", type=int)
    ap.add_argument("--start", type=int)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    process(cfg, a.sequence, a.frames, a.start)


if __name__ == "__main__":
    sys.exit(main())
