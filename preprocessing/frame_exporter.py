"""Write processed frames as plain float32/uint8 .npy images that TouchDesigner Script TOPs load directly.

Per frame i (all positions in WORLD space, TouchDesigner axes, metres):
  lidar_xyz/{i}.npy   (128, C, 4) float32  x y z valid(1/0)
  lidar_attr/{i}.npy  (128, C, 4) float32  intensity01, error_class(0..3), abs_err_m, rel_err
  lidar_dyn/{i}.npy   (128, C, 4) float32  moving-object flag(1/0), track id (-1 none, -2 class-fallback), 0, 0
  lidar_echo/{i}.npy  (128, C, 4) float32  lidar_xyz with valid = valid AND moving (precomputed for Motion Echo)
  stereo_xyz/{i}.npy  (h, w, 4)  float32   x y z valid(1/0)
  stereo_rgb/{i}.npy  (h, w, 4)  uint8     r g b 255
  boxes/{i}.json                            corners/labels in world TD axes (+ annotated flag)
  pose/{i}.npy        (4, 4) float64        robot (os1) pose in world, TD axes
manifest.json holds per-frame timestamps (real sensor times), measured rate, stats and conventions.
"""
import json
import os
from pathlib import Path

import numpy as np


class FrameExporter:
    KINDS = ("lidar_xyz", "lidar_attr", "lidar_dyn", "lidar_echo", "stereo_xyz", "stereo_rgb", "boxes", "pose")

    def __init__(self, out_dir):
        self.out = Path(out_dir)
        for k in self.KINDS:
            (self.out / k).mkdir(parents=True, exist_ok=True)
        self.frames = []

    @staticmethod
    def pack_xyz(xyz, ok):
        out = np.zeros(xyz.shape[:-1] + (4,), dtype=np.float32)
        out[..., :3] = np.where(ok[..., None], xyz, 0.0)
        out[..., 3] = ok.astype(np.float32)
        return out

    @staticmethod
    def echo_from(packed_xyz, lidar_dyn):
        """Motion-echo layer: the same positions, but only moving-object points stay valid."""
        out = np.array(packed_xyz, dtype=np.float32, copy=True)
        out[..., 3] = out[..., 3] * (np.asarray(lidar_dyn)[..., 0] > 0.5)
        return out

    def write_frame(self, i, frame_id, timestamp, lidar_xyz, lidar_ok, lidar_attr, lidar_dyn,
                    stereo_xyz, stereo_ok, stereo_rgb, boxes_json, annotated, pose_td, stats):
        np.save(self.out / "lidar_xyz" / f"{i:05d}.npy", self.pack_xyz(lidar_xyz, lidar_ok))
        np.save(self.out / "lidar_attr" / f"{i:05d}.npy", lidar_attr.astype(np.float32))
        np.save(self.out / "lidar_dyn" / f"{i:05d}.npy", lidar_dyn.astype(np.float32))
        np.save(self.out / "lidar_echo" / f"{i:05d}.npy", self.echo_from(self.pack_xyz(lidar_xyz, lidar_ok), lidar_dyn))
        np.save(self.out / "stereo_xyz" / f"{i:05d}.npy", self.pack_xyz(stereo_xyz, stereo_ok))
        rgba = np.concatenate([stereo_rgb, np.full(stereo_rgb.shape[:2] + (1,), 255, np.uint8)], axis=-1)
        np.save(self.out / "stereo_rgb" / f"{i:05d}.npy", rgba)
        np.save(self.out / "pose" / f"{i:05d}.npy", pose_td.astype(np.float64))
        (self.out / "boxes" / f"{i:05d}.json").write_text(
            json.dumps({"annotated": bool(annotated), "boxes": boxes_json}))
        self.frames.append({"index": i, "frame_id": int(frame_id), "timestamp": float(timestamp), "stats": stats})

    def write_manifest(self, meta):
        meta = dict(meta)
        meta["frames"] = self.frames
        meta["n_frames"] = len(self.frames)
        tmp = self.out / "manifest.json.tmp"                 # write-then-replace: a live reader never sees a half file
        tmp.write_text(json.dumps(meta, indent=1))
        os.replace(tmp, self.out / "manifest.json")
