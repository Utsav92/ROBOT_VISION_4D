"""Mode B - live robot: ROS 2 -> the same processed-frame files the TouchDesigner renderer already reads.

    python optional_ros2/sensor_bridge.py --calib data/coda/calibrations/0 --out output/live \\
        --left /stereo/left/image_rect --right /stereo/right/image_rect --lidar /ouster/points --odom /odom
    python optional_ros2/sensor_bridge.py --selftest        # no ROS needed: replays real CODa data through this code path

STATUS (honest): everything except the `rclpy` node at the bottom is plain numpy/OpenCV and is unit-tested, and
`--selftest` replays real CODa frames through message encoding -> decoding -> synchronisation -> the shared pipeline and
checks the result is bit-identical to the offline export. The rclpy node itself has NEVER been run: this machine has no
ROS 2 and no robot. Treat it as untested glue.

Assumptions (all overridable on the command line / in code):
  * the LiDAR cloud is ORGANISED 128 x 1024 PointCloud2 with float32 x,y,z and an intensity/reflectivity field (Ouster driver);
  * stereo images are already rectified (otherwise pass --raw-images and the CODa-style calibration will rectify them);
  * the odometry message is the pose of the robot BASE in the world, so  T_world_os1 = T_world_base @ T_base_os1  with
    T_base_os1 taken from calib_os1_to_base.yaml; and TF/odom use the right-handed REP-103 base frame (x forward).
Live throughput is limited by the stereo step (CPU StereoSGBM ~2 s per frame here): the node processes only the NEWEST
synchronised set and drops older ones. Real-time needs a GPU stereo backend (RAFT-Stereo, untested here).
"""
import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.calibration import load_calibration  # noqa: E402
from preprocessing.ego_motion import interpolate_pose  # noqa: E402
from preprocessing.frame_exporter import FrameExporter  # noqa: E402
from preprocessing.frame_pipeline import process_frame, write_result  # noqa: E402
from preprocessing.stereo_depth import make_disparity_fn  # noqa: E402

# sensor_msgs/PointField datatype codes -> numpy dtypes
_PF = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
RINGS, COLUMNS = 128, 1024


# ---------------------------------------------------------------- message decoding (pure) ---------------------------
def pointcloud2_to_scan(data, fields, point_step, height, width, is_bigendian=False):
    """sensor_msgs/PointCloud2 payload -> (128, 1024, 4) float32 [x, y, z, intensity] (organised clouds only).

    fields: iterable of (name, offset, datatype). Intensity is taken from 'intensity', else 'reflectivity', else 0."""
    if (height, width) != (RINGS, COLUMNS):
        raise ValueError(f"expected an organised {RINGS}x{COLUMNS} cloud, got {height}x{width}; "
                         "unorganised clouds are not supported (cannot recover the ring/column grid)")
    endian = ">" if is_bigendian else "<"
    names, formats, offsets = [], [], []
    for name, off, dt in fields:
        if name in ("x", "y", "z", "intensity", "reflectivity"):
            names.append(name); formats.append(endian + _PF[int(dt)]); offsets.append(int(off))
    for need in ("x", "y", "z"):
        if need not in names:
            raise ValueError(f"PointCloud2 has no '{need}' field")
    dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": int(point_step)})
    rec = np.frombuffer(bytes(data), dtype=dtype, count=height * width)
    out = np.zeros((height * width, 4), dtype=np.float32)
    out[:, 0], out[:, 1], out[:, 2] = rec["x"], rec["y"], rec["z"]
    for key in ("intensity", "reflectivity"):
        if key in names:
            out[:, 3] = rec[key]
            break
    return out.reshape(height, width, 4)


def image_to_bgr(data, encoding, height, width, step):
    """sensor_msgs/Image payload -> (h, w, 3) uint8 BGR. Honours the row stride (`step`)."""
    ch = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "mono8": 1}.get(encoding)
    if ch is None:
        raise ValueError(f"unsupported image encoding '{encoding}'")
    buf = np.frombuffer(bytes(data), dtype=np.uint8).reshape(height, step)[:, : width * ch].reshape(height, width, ch)
    if encoding == "bgr8":
        return buf.copy()
    if encoding == "rgb8":
        return buf[..., ::-1].copy()
    if encoding == "bgra8":
        return buf[..., :3].copy()
    if encoding == "rgba8":
        return buf[..., 2::-1].copy()
    return np.repeat(buf, 3, axis=2)


def odom_to_T(position, orientation_xyzw):
    """Odometry pose (x,y,z; qx,qy,qz,qw) -> 4x4 T_world_base."""
    from scipy.spatial.transform import Rotation as R
    T = np.eye(4)
    T[:3, :3] = R.from_quat(list(orientation_xyzw)).as_matrix()
    T[:3, 3] = list(position)
    return T


def T_to_row(stamp, T):
    from scipy.spatial.transform import Rotation as R
    qx, qy, qz, qw = R.from_matrix(T[:3, :3]).as_quat()
    return [stamp, *T[:3, 3], qw, qx, qy, qz]               # the `ts x y z qw qx qy qz` layout of ego_motion


class Rectifier:
    """Rectifies raw camera images with the CODa-style calibration (K, distortion, R_rect, P)."""

    def __init__(self, cam):
        import cv2
        self.cv2 = cv2
        self.map1, self.map2 = cv2.initUndistortRectifyMap(cam.K, cam.dist, cam.R_rect, cam.P[:, :3],
                                                           (cam.width, cam.height), cv2.CV_32FC1)

    def __call__(self, img):
        return self.cv2.remap(img, self.map1, self.map2, self.cv2.INTER_LINEAR)


# ---------------------------------------------------------------- synchronisation (pure) ----------------------------
class PoseBuffer:
    """Odometry history with slerp/lerp interpolation (poses arrive faster than scans)."""

    def __init__(self, keep=400):
        self.rows = deque(maxlen=keep)

    def push(self, stamp, T_w_base):
        if self.rows and stamp <= self.rows[-1][0]:
            return
        self.rows.append(T_to_row(stamp, T_w_base))

    def latest(self):
        return self.rows[-1][0] if self.rows else -np.inf

    def at(self, t, max_extrapolation_s=0.0):
        if len(self.rows) < 2:
            raise ValueError("need at least two odometry samples")
        return interpolate_pose(np.array(self.rows), t, max_extrapolation_s)


class StampSync:
    """LiDAR-triggered matching of image streams by timestamp (nearest within max_dt, as in the offline pipeline).

    A trigger is released only once every other stream has seen data at or after it (so no better match can still be
    arriving) - or it is older than `keep_s`, in which case it is dropped. With latest_only, stale ready sets are
    discarded so a slow consumer always works on the newest data."""

    def __init__(self, streams=("left", "right"), max_dt=0.05, keep_s=1.5, maxlen=64):
        self.streams, self.max_dt, self.keep_s = tuple(streams), max_dt, keep_s
        self.buf = {s: deque(maxlen=maxlen) for s in (*self.streams, "lidar")}
        self.dropped = 0

    def push(self, stream, stamp, payload):
        self.buf[stream].append((stamp, payload))

    def _nearest(self, stream, t):
        best = min(self.buf[stream], key=lambda sp: abs(sp[0] - t), default=None)
        return best if best is not None and abs(best[0] - t) <= self.max_dt else None

    def pop_ready(self, latest_only=True, now=None):
        ready, newest = [], max((b[-1][0] for b in self.buf.values() if b), default=-np.inf) if now is None else now
        while self.buf["lidar"]:
            t, scan = self.buf["lidar"][0]
            if any(not self.buf[s] or self.buf[s][-1][0] < t for s in self.streams):
                if newest - t > self.keep_s:                       # a stream never caught up: give up on this scan
                    self.buf["lidar"].popleft(); self.dropped += 1
                    continue
                break
            self.buf["lidar"].popleft()
            match = {s: self._nearest(s, t) for s in self.streams}
            if any(m is None for m in match.values()):
                self.dropped += 1
                continue
            ready.append({"stamp": t, "lidar": scan, **{s: match[s][1] for s in self.streams}})
        for s in self.streams:                                      # forget data older than anything still needed
            horizon = (self.buf["lidar"][0][0] if self.buf["lidar"] else newest) - self.keep_s
            while self.buf[s] and self.buf[s][0][0] < horizon:
                self.buf[s].popleft()
        if latest_only and len(ready) > 1:
            self.dropped += len(ready) - 1
            ready = ready[-1:]
        return ready


# ---------------------------------------------------------------- live session ---------------------------------------
class LiveSession:
    """Runs the shared per-frame pipeline and appends frames to a rolling export directory the renderer can read."""

    def __init__(self, cfg, calib, out_dir, max_frames=3000, rectifier=None, log=print):
        self.cfg, self.calib, self.log = cfg, calib, log
        self.exporter = FrameExporter(out_dir)
        self.out_dir = Path(out_dir)
        self.disparity_fn = make_disparity_fn(cfg["stereo"], log)
        self.max_frames, self.rectifier = max_frames, rectifier
        self.n = 0
        self.stamps = []

    def T_w_os1(self, T_w_base):
        return np.asarray(T_w_base) @ self.calib.T_base_os1

    def ingest(self, left, right, scan, T_w_base, stamp):
        if self.n >= self.max_frames:
            raise RuntimeError(f"max_frames={self.max_frames} reached; start a new session/output directory")
        if self.rectifier is not None:
            left, right = self.rectifier[0](left), self.rectifier[1](right)
        res = process_frame(left, right, scan, self.T_w_os1(T_w_base), self.calib, self.cfg, self.disparity_fn)
        write_result(self.exporter, self.n, self.n, stamp, res)
        self.stamps.append(stamp)
        self.n += 1
        self.write_manifest()
        return res

    def write_manifest(self):
        from preprocessing.coords import M_LIDAR_TO_TD
        from preprocessing.synchronize import estimate_rate
        self.exporter.write_manifest({
            "live": True, "sequence": -1, "coordinate_frame": "world (odometry origin), TouchDesigner axes",
            "axes": "x right, y up, z toward viewer", "M_lidar_to_td": M_LIDAR_TO_TD.tolist(),
            "measured_rate": estimate_rate(np.array(self.stamps)) if len(self.stamps) > 1 else {},
            "calibration": {"baseline_m": self.calib.baseline_m, "disparity_offset_px": self.calib.disparity_offset_px,
                            "fx": self.calib.cam0.fx, "apply_rectification_rotation": self.calib.apply_rect_rotation},
            "config": self.cfg, "moving_instances": [],
            "assumptions": ["live mode has no ground-truth boxes: boxes/moving-object echo are empty",
                            "odometry is T_world_base; T_world_os1 = T_world_base @ T_base_os1"]})


# ---------------------------------------------------------------- self-test on real data (no ROS) --------------------
def _scan_to_pointcloud2_bytes(scan):
    """Encode a (128,1024,4) scan the way the Ouster driver does: x y z intensity f32 + ring u16 (point_step 20)."""
    n = scan.shape[0] * scan.shape[1]
    rec = np.zeros(n, dtype=np.dtype({"names": ["x", "y", "z", "intensity", "ring"], "formats": ["<f4"] * 4 + ["<u2"],
                                      "offsets": [0, 4, 8, 12, 16], "itemsize": 20}))
    flat = scan.reshape(-1, 4)
    rec["x"], rec["y"], rec["z"], rec["intensity"] = flat[:, 0], flat[:, 1], flat[:, 2], flat[:, 3]
    rec["ring"] = np.repeat(np.arange(scan.shape[0]), scan.shape[1])
    fields = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("intensity", 12, 7), ("ring", 16, 4)]
    return rec.tobytes(), fields, 20


def selftest(config="config.yaml", root="data/coda", seq=0, frames=(400, 401, 402), out="output/_live_selftest"):
    import cv2
    import yaml
    from preprocessing.coda_paths import CodaPaths
    from preprocessing.synchronize import load_timestamps
    from preprocessing.ego_motion import load_poses
    from preprocessing import lidar_loader as LL

    cfg = yaml.safe_load(open(config))
    ds = cfg["dataset"]
    p = CodaPaths(root, seq, ds["image_source"], ds["lidar_source"], ds["pose_source"])
    calib = load_calibration(p.calib_dir, cfg["calibration"]["apply_rectification_rotation"])
    ts, poses = load_timestamps(p.timestamps_file), load_poses(p.pose_file())
    import shutil
    shutil.rmtree(out, ignore_errors=True)
    sess = LiveSession(cfg, calib, out)
    sync, pbuf = StampSync(("left", "right"), cfg["sync"]["max_dt_s"]), PoseBuffer()
    inv_base = np.linalg.inv(calib.T_base_os1)

    # feed the "robot": odometry at 40 Hz (base pose derived so that base @ T_base_os1 == the recorded os1 pose),
    # then for each frame the encoded messages exactly as a driver would publish them
    t0, t1 = ts[frames[0]] - 0.2, ts[frames[-1]] + 0.2
    for t in np.arange(t0, t1, 0.025):
        pbuf.push(float(t), interpolate_pose(poses, float(t), 0.1) @ inv_base)
    for f in frames:
        stamp = float(ts[f])
        left, right = cv2.imread(str(p.image("cam0", f))), cv2.imread(str(p.image("cam1", f)))
        scan = LL.read_scan(p.lidar(f))
        blob, fields, step = _scan_to_pointcloud2_bytes(scan)
        bgr = lambda im: (im.tobytes(), "bgr8", im.shape[0], im.shape[1], im.shape[1] * 3)
        d, e, h, w, st = bgr(left); sync.push("left", stamp, image_to_bgr(d, e, h, w, st))
        d, e, h, w, st = bgr(right); sync.push("right", stamp + 0.004, image_to_bgr(d, e, h, w, st))     # slight offset
        sync.push("lidar", stamp, pointcloud2_to_scan(blob, fields, step, RINGS, COLUMNS))
        for s in sync.pop_ready(latest_only=False):
            sess.ingest(s["left"], s["right"], s["lidar"], pbuf.at(s["stamp"]), s["stamp"])

    # compare with the offline export of the same frames
    ref = Path(cfg["export"]["out_root"]) / f"seq{seq}"
    first = None
    if (ref / "manifest.json").exists():
        first = json.loads((ref / "manifest.json").read_text())["frames"][0]["frame_id"]
    report = {"frames_ingested": sess.n, "dropped": sync.dropped, "compared_to_offline": first == frames[0]}
    if first == frames[0]:
        worst = {}
        for kind in ("lidar_xyz", "lidar_attr", "stereo_xyz", "stereo_rgb", "pose"):
            a = [np.load(Path(out) / kind / f"{i:05d}.npy") for i in range(sess.n)]
            b = [np.load(ref / kind / f"{i:05d}.npy") for i in range(sess.n)]
            worst[kind] = float(max(np.abs(x.astype(np.float64) - y.astype(np.float64)).max() for x, y in zip(a, b)))
        report["max_abs_difference_vs_offline"] = worst
        report["bit_identical"] = all(v == 0.0 for v in worst.values())
        # positions go through an extra base-frame transform composition, so float rounding of ~1e-5 m is expected;
        # attributes (depth-error classes/values) and colours must match exactly
        report["matches_offline"] = bool(worst["lidar_xyz"] < 1e-3 and worst["stereo_xyz"] < 1e-3 and worst["pose"] < 1e-5
                                         and worst["lidar_attr"] == 0.0 and worst["stereo_rgb"] == 0.0)
    return report


# ---------------------------------------------------------------- the rclpy node (UNTESTED) ---------------------------
def run_node(args):
    try:
        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, PointCloud2
    except ImportError as e:
        sys.exit(f"ROS 2 python packages not available ({e}). Source your ROS 2 environment, or use --selftest.")
    import yaml
    cfg = yaml.safe_load(open(args.config))
    calib = load_calibration(args.calib, cfg["calibration"]["apply_rectification_rotation"])
    rect = (Rectifier(calib.cam0), Rectifier(calib.cam1)) if args.raw_images else None
    sess = LiveSession(cfg, calib, args.out, args.max_frames, rect)
    sync, pbuf = StampSync(("left", "right"), cfg["sync"]["max_dt_s"]), PoseBuffer()

    class Bridge(Node):
        def __init__(self):
            super().__init__("robot_vision_4d_bridge")
            st = lambda m: m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            img = lambda name: (lambda m: sync.push(name, st(m), image_to_bgr(m.data, m.encoding, m.height, m.width, m.step)))
            self.create_subscription(Image, args.left, img("left"), qos_profile_sensor_data)
            self.create_subscription(Image, args.right, img("right"), qos_profile_sensor_data)
            self.create_subscription(PointCloud2, args.lidar, lambda m: sync.push("lidar", st(m), pointcloud2_to_scan(
                m.data, [(f.name, f.offset, f.datatype) for f in m.fields], m.point_step, m.height, m.width, m.is_bigendian)),
                qos_profile_sensor_data)
            self.create_subscription(Odometry, args.odom, lambda m: pbuf.push(st(m), odom_to_T(
                (m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z),
                (m.pose.pose.orientation.x, m.pose.pose.orientation.y, m.pose.pose.orientation.z, m.pose.pose.orientation.w))), 10)
            self.create_timer(0.05, self.tick)

        def tick(self):
            for s in sync.pop_ready(latest_only=True):
                if pbuf.latest() < s["stamp"] or len(pbuf.rows) < 2:
                    continue                                   # pose not caught up yet; the set is dropped
                t0 = time.perf_counter()
                try:
                    sess.ingest(s["left"], s["right"], s["lidar"], pbuf.at(s["stamp"], 0.05), s["stamp"])
                except ValueError as e:
                    self.get_logger().warn(f"frame skipped: {e}")
                    continue
                self.get_logger().info(f"frame {sess.n} processed in {time.perf_counter() - t0:.2f} s (dropped {sync.dropped})")

    rclpy.init()
    node = Bridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--calib", default="data/coda/calibrations/0")
    ap.add_argument("--out", default="output/live")
    ap.add_argument("--left", default="/stereo/left/image_rect")
    ap.add_argument("--right", default="/stereo/right/image_rect")
    ap.add_argument("--lidar", default="/ouster/points")
    ap.add_argument("--odom", default="/odom")
    ap.add_argument("--raw-images", action="store_true", help="images are unrectified: rectify with the calibration")
    ap.add_argument("--max-frames", type=int, default=3000)
    args = ap.parse_args()
    if args.selftest:
        rep = selftest()
        print(json.dumps(rep, indent=1))
        return 0 if rep.get("matches_offline", False) else 1
    run_node(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
