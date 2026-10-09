"""Camera system (Text-DAT module in CAMERA_SYSTEM).

One render camera ('main_camera') is driven every frame by five pose generators; switching mode blends the pose
(slerp rotation, lerp position and fov) over `cam_blend_s` seconds with smootherstep easing.

  1 Robot POV      the recorded sensor pose itself (robot-local TD axes: forward -z, up +y), real-camera fov
  2 Chase          behind and above the robot, in its heading, looking slightly ahead of it
  3 Free-fly       mouse look, W/A/S/D move, Q/E vertical, Shift fast (starts from the current view: no jump)
  4 Bird's-eye     straight down over the robot, world-fixed up (-z is screen-up)
  5 Cinematic orbit auto orbit around the robot and then each MOVING object, eased hand-over, driven by the data
                   clock (so it is reproducible when scrubbing or exporting video)

All poses are 4x4 column-vector world matrices in TouchDesigner axes; a camera looks down its local -z.
Everything above the "TouchDesigner glue" line is pure numpy and unit-tested.
"""
import math

import numpy as np

UP = np.array([0.0, 1.0, 0.0])


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def ease(x):
    """Smootherstep 0..1."""
    x = min(max(float(x), 0.0), 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)


def look_at(eye, target, up=UP):
    eye, target = np.asarray(eye, dtype=np.float64), np.asarray(target, dtype=np.float64)
    z = _norm(eye - target)                       # camera looks along -z
    up = np.asarray(up, dtype=np.float64)
    if abs(np.dot(_norm(up), z)) > 0.999:         # looking along the up vector: pick another reference
        up = np.array([0.0, 0.0, -1.0]) if abs(z[1]) > 0.5 else UP
    x = _norm(np.cross(up, z))
    y = np.cross(z, x)
    W = np.eye(4)
    W[:3, 0], W[:3, 1], W[:3, 2], W[:3, 3] = x, y, z, eye
    return W


# ---- quaternion helpers (no scipy inside TouchDesigner) ----------------------------------------------------------
def mat_to_quat(R):
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.array(q)                                # x y z w
    return q / np.linalg.norm(q)


def quat_to_mat(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def slerp(q0, q1, a):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(min(d, 1.0))
    return (math.sin((1 - a) * th) * q0 + math.sin(a * th) * q1) / math.sin(th)


def blend_pose(Wa, Wb, a):
    """a=0 -> Wa, a=1 -> Wb: slerp rotation, lerp translation."""
    W = np.eye(4)
    W[:3, :3] = quat_to_mat(slerp(mat_to_quat(Wa[:3, :3]), mat_to_quat(Wb[:3, :3]), a))
    W[:3, 3] = (1 - a) * Wa[:3, 3] + a * Wb[:3, 3]
    return W


# ---- the five pose generators --------------------------------------------------------------------------------------
def pose_robot_pov(pose_td):
    return np.array(pose_td, dtype=np.float64)


def pose_chase(pose_td, back=7.0, height=3.5, look_ahead=4.0):
    R, p = pose_td[:3, :3], pose_td[:3, 3]
    forward = _norm(R @ np.array([0.0, 0.0, -1.0]))
    forward[1] = 0.0                                              # chase in the ground plane (ignore robot pitch)
    forward = _norm(forward)
    eye = p - forward * back + np.array([0.0, height, 0.0])
    return look_at(eye, p + forward * look_ahead + np.array([0.0, 0.5, 0.0]))


def pose_birdseye(p, height=45.0):
    p = np.asarray(p, dtype=np.float64)
    return look_at(p + np.array([0.0, height, 0.0]), p, up=np.array([0.0, 0.0, -1.0]))


def cinematic_target(robot_pos, object_positions, t, dwell_s=8.0, handover_s=2.0):
    """Eased hand-over between [robot, *moving objects] every dwell_s seconds of data time."""
    targets = [np.asarray(robot_pos, dtype=np.float64)] + [np.asarray(o, dtype=np.float64) for o in object_positions]
    n = len(targets)
    k = int(math.floor(t / dwell_s)) % n
    local = t - math.floor(t / dwell_s) * dwell_s
    a = ease((local - (dwell_s - handover_s)) / handover_s) if local > dwell_s - handover_s else 0.0
    return (1 - a) * targets[k] + a * targets[(k + 1) % n]


def pose_orbit(target, t, radius=12.0, height=5.0, period_s=30.0):
    az = 2 * math.pi * (t / period_s)
    target = np.asarray(target, dtype=np.float64)
    eye = target + np.array([radius * math.sin(az), height, radius * math.cos(az)])
    return look_at(eye, target + np.array([0.0, 1.0, 0.0]))


class FreeCam:
    """Yaw about +y (0 faces -z) and pitch; forward = (-sin yaw cos p, sin p, -cos yaw cos p)."""

    def __init__(self):
        self.pos, self.yaw, self.pitch = np.zeros(3), 0.0, 0.0

    def from_matrix(self, W):
        f = _norm(-np.asarray(W)[:3, 2])
        self.pos = np.array(W[:3, 3], dtype=np.float64)
        self.pitch = math.asin(max(-1.0, min(1.0, f[1])))
        self.yaw = math.atan2(-f[0], -f[2])

    def forward(self):
        cp = math.cos(self.pitch)
        return np.array([-math.sin(self.yaw) * cp, math.sin(self.pitch), -math.cos(self.yaw) * cp])

    def step(self, dt, fwd=0.0, strafe=0.0, vert=0.0, dyaw=0.0, dpitch=0.0, speed=6.0):
        self.yaw += dyaw
        self.pitch = max(-1.5, min(1.5, self.pitch + dpitch))
        f = self.forward()
        right = _norm(np.cross(f, UP))
        self.pos = self.pos + (f * fwd + right * strafe + UP * vert) * speed * dt

    def matrix(self):
        return look_at(self.pos, self.pos + self.forward())


def fov_from_intrinsics(fx, width_px):
    """Horizontal field of view of the real camera, degrees."""
    return math.degrees(2 * math.atan(width_px / (2.0 * fx)))


# ======================================== TouchDesigner glue ========================================================
BASE = "/project1/ROBOT_VISION_4D"
_st = {"mode": None, "W": None, "fov": 55.0, "from_W": None, "from_fov": 55.0, "t0": 0.0, "last": None,
       "free": FreeCam(), "inject": None, "override": None, "mouse": None, "wheel": None, "speed_mul": 1.0}
POV_FOV_KEY = "pov_fov"


def set_override(W, fov):
    """Scripted camera (video export): bypasses the mode logic and blending; the pose is applied exactly as given."""
    _st["override"] = (np.asarray(W, dtype=np.float64), float(fov))


def clear_override():
    _st["override"] = None


def _apply(cam, td, W, fov):
    m = td.tdu.Matrix(*[float(x) for x in W.T.flatten()])
    s, r, t = m.decompose()
    cam.par.tx, cam.par.ty, cam.par.tz = t
    cam.par.rx, cam.par.ry, cam.par.rz = r
    cam.par.fov = fov


def inject(fwd=0.0, strafe=0.0, vert=0.0, dyaw=0.0, dpitch=0.0):
    """Synthetic free-fly input for the next update (used by tests; real input comes from the Keyboard/Mouse CHOPs)."""
    _st["inject"] = dict(fwd=fwd, strafe=strafe, vert=vert, dyaw=dyaw, dpitch=dpitch)


def _free_input(base, mc):
    """Free-fly input. Keyboard In CHOP channels are k-prefixed (kw, ka, ks, kd, kq, ke, kshift); the Mouse In CHOP
    provides tx/ty, lbutton and the wheel position (scaled into a speed multiplier)."""
    if _st["inject"] is not None:
        d, _st["inject"] = _st["inject"], None
        return d
    d = dict(fwd=0.0, strafe=0.0, vert=0.0, dyaw=0.0, dpitch=0.0)
    kb = base.op("CAMERA_SYSTEM/keyboard")
    mouse = base.op("CAMERA_SYSTEM/mouse")
    if kb is not None:
        def key(n):
            c = kb["k" + n]
            return float(c.eval()) if c is not None else 0.0
        d["fwd"], d["strafe"], d["vert"] = key("w") - key("s"), key("d") - key("a"), key("e") - key("q")
        d["fast"] = key("shift") > 0.5
    if mouse is not None and mouse["tx"] is not None:
        x, y = float(mouse["tx"].eval()), float(mouse["ty"].eval())
        down = mouse["lbutton"] is not None and float(mouse["lbutton"].eval()) > 0.5
        if down and _st["mouse"] is not None:
            d["dyaw"], d["dpitch"] = -(x - _st["mouse"][0]) * 3.0, (y - _st["mouse"][1]) * 3.0
        _st["mouse"] = (x, y) if down else None
        if mouse["wheel"] is not None:
            w = float(mouse["wheel"].eval())
            if _st["wheel"] is not None:
                _st["speed_mul"] = min(max(_st["speed_mul"] * 1.25 ** (w - _st["wheel"]), 0.1), 20.0)
            _st["wheel"] = w
    return d


def _targets_now(base, fl, frame, robot_pos):
    boxes = fl.get("boxes", frame)["boxes"]
    moving = set(fl.manifest().get("moving_instances", []))
    return [np.array(b["center"]) for b in boxes if b.get("instance") in moving]


def update():
    """Call once per frame (Execute DAT onFrameStart): computes the target pose, blends mode changes, drives main_camera."""
    import time
    import td
    base = td.op(BASE)
    fl = base.op("DATA_INPUT/frame_loader").module
    mc = base.op("UI/main_controls")
    ctl = base.op("DATA_INPUT/frame_control")
    cam = base.op("CAMERA_SYSTEM/main_camera")
    g = lambda n: float(mc[n].eval())
    frame, t_data = int(ctl["frame"].eval()), float(ctl["time"].eval())
    pose = np.asarray(fl.get("pose", frame))
    mode = int(round(g("camera_mode")))
    now = time.perf_counter()
    dt = 0.0 if _st["last"] is None else min(now - _st["last"], 0.25)
    _st["last"] = now

    if _st["override"] is not None:
        W, fov = _st["override"]
        _st.update(W=W, fov=fov, from_W=None)
        _apply(cam, td, W, fov)
        return mode
    pov_fov = float(base.fetch(POV_FOV_KEY, 76.0)) if hasattr(base, "fetch") else 76.0
    if mode == 1:
        W, fov = pose_robot_pov(pose), pov_fov
    elif mode == 2:
        W, fov = pose_chase(pose, g("chase_back"), g("chase_up")), g("cam_fov")
    elif mode == 4:
        W, fov = pose_birdseye(pose[:3, 3], g("bird_height")), g("cam_fov")
    elif mode == 5:
        tgt = cinematic_target(pose[:3, 3], _targets_now(base, fl, frame, pose[:3, 3]), t_data)
        W, fov = pose_orbit(tgt, t_data, g("orbit_radius"), g("orbit_height"), g("orbit_period")), g("cam_fov")
    else:  # 3 free-fly
        mode = 3
        free = _st["free"]
        if _st["mode"] != 3:
            free.from_matrix(_st["W"] if _st["W"] is not None else pose_chase(pose))
        d = _free_input(base, mc)
        free.step(dt, d["fwd"], d["strafe"], d["vert"], d["dyaw"], d["dpitch"],
                  g("camera_speed") * _st["speed_mul"] * (3.0 if d.get("fast") else 1.0))
        W, fov = free.matrix(), g("cam_fov")

    if _st["W"] is None:
        _st["mode"], _st["W"], _st["fov"] = mode, W, fov
    if mode != _st["mode"]:                                   # start a blend from wherever the camera is now
        _st.update(mode=mode, from_W=_st["W"], from_fov=_st["fov"], t0=now)
    blend_s = max(g("cam_blend_s"), 1e-3)
    a = ease((now - _st["t0"]) / blend_s) if _st["from_W"] is not None else 1.0
    if a >= 1.0:
        _st["from_W"] = None
    if _st["from_W"] is not None:
        W, fov = blend_pose(_st["from_W"], W, a), (1 - a) * _st["from_fov"] + a * fov
    _st["W"], _st["fov"] = W, fov

    _apply(cam, td, W, fov)
    return mode
