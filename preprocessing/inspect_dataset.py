"""Stage 1: inspect a CODa tree and report what is actually on disk (never assumes filenames).

    python -m preprocessing.inspect_dataset --root data/coda --sequence 0 [--check-poses]
Writes output/dataset_report.json.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

from . import lidar_loader as LL
from .calibration import _load_yaml, load_calibration
from .coda_paths import CodaPaths
from .ego_motion import check_pose_convention, interpolate_pose, load_poses
from .object_annotations import load_boxes
from .synchronize import estimate_rate, load_timestamps


def inspect(root, seq, check_poses=False, out="output/dataset_report.json", log=print):
    p = CodaPaths(root, seq)
    rep = {"root": str(p.root), "sequence": seq, "problems": []}

    def need(path, what):
        ok = Path(path).exists()
        rep[what] = {"path": str(path), "exists": ok}
        if not ok:
            rep["problems"].append(f"missing {what}: {path}")
        return ok

    if need(p.calib_dir, "calibrations"):
        rep["calibration_files"] = {}
        for f in sorted(p.calib_dir.glob("*.yaml")):
            d = _load_yaml(f)
            rep["calibration_files"][f.name] = sorted(d.keys())
        try:
            c = load_calibration(p.calib_dir)
            rep["stereo"] = {"baseline_m": c.baseline_m, "baseline_source": c.baseline_source,
                             "disparity_offset_px": c.disparity_offset_px,
                             "fx": c.cam0.fx, "size": [c.cam0.width, c.cam0.height]}
            rep["stereo"]["cam0_to_cam1_translation_norm_m"] = float(np.linalg.norm(c.T_cam1_cam0[:3, 3]))
        except Exception as e:
            rep["problems"].append(f"calibration load failed: {e}")
    if need(p.timestamps_file, "timestamps"):
        rep["sensor_rate"] = estimate_rate(load_timestamps(p.timestamps_file))
    if need(p.pose_file(), "poses"):
        pz = load_poses(p.pose_file())
        rep["pose_rate"] = estimate_rate(pz[:, 0])
        rep["pose_file_used"] = str(p.pose_file())

    frames = p.available_frames()
    rep["frames_with_images_and_lidar"] = len(frames)
    ann = p.annotated_frames()
    rep["annotated_frames"] = len(ann)
    if frames:
        scan = LL.read_scan(p.lidar(frames[0]))
        ok = LL.valid_mask(scan)
        rep["lidar_scan"] = {"shape": list(scan.shape), "valid_fraction": float(ok.mean()),
                             "intensity_range": [float(scan[..., 3][ok].min()), float(scan[..., 3][ok].max())]}
    if ann:
        cls = collections.Counter()
        no_id = 0
        for f in ann[:50]:
            for b in load_boxes(p.bbox(f)):
                cls[b.class_id] += 1
                no_id += (not b.id_available)
        rep["annotation_classes_first50"] = dict(cls.most_common())
        rep["boxes_without_instance_id_first50"] = no_id

    if check_poses and len(frames) > 3:
        ts = load_timestamps(p.timestamps_file)
        poses = load_poses(p.pose_file())
        a, b = frames[0], frames[min(3, len(frames) - 1)]
        sa = LL.read_scan(p.lidar(a))[..., :3].reshape(-1, 3)
        sb = LL.read_scan(p.lidar(b))[..., :3].reshape(-1, 3)
        s_T, s_inv = check_pose_convention(sa, sb, interpolate_pose(poses, ts[a]), interpolate_pose(poses, ts[b]))
        rep["pose_convention"] = {"mean_nn_dist_if_T_world_os1": s_T, "mean_nn_dist_if_inverse": s_inv,
                                  "verdict": "T_world_os1" if s_T < s_inv else "INVERSE (flip assumption in ego_motion)"}

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, indent=1, default=float))
    log(json.dumps(rep, indent=1, default=float))
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/coda")
    ap.add_argument("--sequence", type=int, default=0)
    ap.add_argument("--check-poses", action="store_true")
    a = ap.parse_args(argv)
    rep = inspect(a.root, a.sequence, a.check_poses)
    return 1 if rep["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
