"""Calibration visualisation: project LiDAR onto the rectified left image, colour = range.

    python -m preprocessing.calibration_overlay --root data/coda --sequence 0 --frame 5000 [--no-rect-rotation]

Run once with and once without --no-rect-rotation; the version where range-coloured points sit on the
correct object edges (poles, building corners, people) is the right rectification convention.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from . import lidar_loader as LL
from .calibration import load_calibration
from .coda_paths import CodaPaths


def make_overlay(root, seq, frame, rect_rotation=True, lidar_source="3d_comp", max_range=40.0):
    p = CodaPaths(root, seq, lidar_source=lidar_source)
    calib = load_calibration(p.calib_dir, rect_rotation)
    img = cv2.imread(str(p.image("cam0", frame)))
    if img is None:
        raise IOError(f"cannot read {p.image('cam0', frame)}")
    scan = LL.read_scan(p.lidar(frame))
    pts = scan[..., :3].reshape(-1, 3)
    ok = LL.valid_mask(scan).reshape(-1)
    uv, z = calib.project_rect0(calib.os1_to_rect0(pts[ok]))
    H, W = img.shape[:2]
    keep = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    uv, z = uv[keep], z[keep]
    col = cv2.applyColorMap((255 * np.clip(z / max_range, 0, 1)).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)
    out = img.copy()
    for (u, v), c in zip(uv.astype(int), col.reshape(-1, 3)):
        cv2.circle(out, (u, v), 2, tuple(int(x) for x in c), -1)
    return out, int(keep.sum())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/coda")
    ap.add_argument("--sequence", type=int, default=0)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--no-rect-rotation", action="store_true")
    ap.add_argument("--lidar-source", default="3d_comp")
    ap.add_argument("--out", default="output/screenshots")
    a = ap.parse_args(argv)
    img, n = make_overlay(a.root, a.sequence, a.frame, not a.no_rect_rotation, a.lidar_source)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tag = "norect" if a.no_rect_rotation else "rect"
    dst = Path(a.out) / f"calib_overlay_seq{a.sequence}_f{a.frame}_{tag}.png"
    cv2.imwrite(str(dst), img)
    print(f"{n} LiDAR points in view -> {dst}")


if __name__ == "__main__":
    main()
