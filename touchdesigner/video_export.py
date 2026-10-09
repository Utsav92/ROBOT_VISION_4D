"""Deterministic frame-by-frame renderer for the cinematic video (Text-DAT module in VIDEO).

Every frame is rendered from scratch for a given video time t: recorded-data time, control values, camera and overlay
text all come from video_director (pure functions of t), so re-running gives identical frames. Controls are driven
through the same panel widgets a user would click. Frames are saved as PNG; encoding to H.264 is done outside by ffmpeg.
"""
import os
import time

import numpy as np

BASE = "/project1/ROBOT_VISION_4D"
TOGGLES = ("show_stereo", "show_lidar", "show_error", "show_boxes", "show_labels", "show_stats", "show_grid", "show_trajectory")
_ctx = {}


def _env():
    import td
    base = td.op(BASE)
    return dict(
        base=base, panel=base.op("UI/panel"), mc=base.op("UI/main_controls"),
        fl=base.op("DATA_INPUT/frame_loader").module, tl=base.op("DATA_INPUT/timeline").module,
        cc=base.op("CAMERA_SYSTEM/camera_controller").module, vd=base.op("VIDEO/video_director").module,
        out=base.op("OUTPUT/final_composite"), render=base.op("OUTPUT/render3d"), title=base.op("OUTPUT/title_overlay"))


def _set_const(env, name, value):
    mc = env["mc"]
    names = [c.name for c in mc.chans()]
    mc.seq.const[names.index(name)].par.value = value


def _apply_settings(env, s):
    panel = env["panel"]
    for k in TOGGLES:
        w = panel.op("t_" + k)
        w.par.value0 = int(s[k])          # set the widget value directly: click(0) is not reliable within one call
    for k in ("density", "lidar_opacity", "history_s"):
        panel.op("s_" + k).par.value0 = float(s[k])
    panel.op("r_temporal%d" % (int(s["temporal_mode"]) + 1)).click(1, left=True)
    for k in ("box_range", "fade"):
        _set_const(env, k, float(s[k]))


def _ensure_ctx(env):
    """A file-synced Text DAT can reload between bridge calls and wipe module globals, so rebuild lazily."""
    if "director" not in _ctx:
        _ctx["director"] = env["vd"].Director(env["cc"])
    if "moving" not in _ctx:
        _ctx["moving"] = set(env["fl"].manifest().get("moving_instances", []))


def begin(aa="aa8"):
    """Prepare: pause the timeline, cinematic style, scripted camera, high-quality antialiasing."""
    env = _env()
    env["tl"].pause()
    env["panel"].op("r_style2").click(1, left=True)                 # CINEMATIC
    try:
        env["render"].par.antialias = aa
    except Exception:
        pass
    _ctx["director"] = env["vd"].Director(env["cc"])
    moving = set(env["fl"].manifest().get("moving_instances", []))
    _ctx["moving"] = moving
    return {"frames": env["fl"].n_frames(), "data_duration_s": float(env["tl"].duration()), "moving": sorted(moving)}


def end():
    env = _env()
    env["cc"].clear_override()
    _set_const(env, "fade", 1.0)
    _set_const(env, "title_alpha", 0.0)
    env["render"].par.antialias = "aa4"
    return "restored"


def render_frame(n, out_dir, fps=None):
    env = _env()
    _ensure_ctx(env)
    vd, fl, tl, cc = env["vd"], env["fl"], env["tl"], env["cc"]
    fps = fps or vd.FPS
    t = n / fps
    s = vd.settings(t)
    tl.set_time(vd.data_time(t, data_end=float(tl.duration())))
    frame = tl.state()["frame"]
    _apply_settings(env, s)

    pose = np.asarray(fl.get("pose", frame))
    peds = [np.array(b["center"]) for b in fl.get("boxes", frame)["boxes"] if b.get("instance") in _ctx.get("moving", ())]
    W, fov = _ctx["director"].camera(t, (pose, peds))
    cc.set_override(W, fov)
    cc.update()

    title, sub, alpha = vd.overlay(t)
    env["title"].par.text = title + "\n" + sub
    env["base"].op("OUTPUT/title_shadow").par.text = title + "\n" + sub
    _set_const(env, "title_alpha", alpha)
    base = env["base"]
    for k in range(1, 65):                                           # history layers recook for the new frame
        h = base.op("TEMPORAL_4D/hist_pos_%d" % k)
        if h is None:
            break
        h.cook(force=True)
    base.op("OBJECT_TRACKING/object_labels").cook(force=True)
    base.op("OBJECT_TRACKING/box_geometry/box_lines").cook(force=True)
    base.op("DEPTH_DIAGNOSTICS/error_statistics").cook(force=True)
    env["render"].cook(force=True)
    env["out"].cook(force=True)
    os.makedirs(out_dir, exist_ok=True)
    env["out"].save(os.path.join(out_dir, "frame_%04d.png" % n))
    return frame


def render_range(a, b, out_dir):
    t0 = time.perf_counter()
    frames = [render_frame(n, out_dir) for n in range(a, b)]
    return "rendered %d..%d in %.1f s (%.0f ms/frame), data frames %d..%d" % (
        a, b - 1, time.perf_counter() - t0, (time.perf_counter() - t0) / max(b - a, 1) * 1000, frames[0], frames[-1])
