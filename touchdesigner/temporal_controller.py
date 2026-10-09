"""History layers for the 4D / temporal modes (Text-DAT module in TEMPORAL_4D).

All ages use MEASURED sensor timestamps: layer k shows the latest frame at or before (t_now - age_k), so a 3 s history
is 3 s of real robot time even if the sensor rate drifts. Temporal modes (spec section 8):
  0 off | 1 Motion Echo (moving-object points only) | 2 Point Trails (whole LiDAR cloud) | 6 Time Explosion
  (history displaced along an ARTISTIC time axis; measured positions are NOT being shown in that mode)
Mode 4 (scrubber) is the timeline itself; Mode 5 (freeze) is the timeline FREEZE combined with any trail mode.
Mode 3 (skeleton) is intentionally absent: no pose estimator is available, and no landmarks are invented.
"""
import numpy as np

MODE_NAMES = {0: "OFF", 1: "MOTION ECHO", 2: "POINT TRAILS", 6: "TIME EXPLOSION (artistic time axis)"}


def layer_plan(timestamps, frame, history_s, n_layers, decay):
    """-> list of (layer k, source frame, age seconds, alpha, age01) for layers 1..n_layers; alpha 0 if no history."""
    ts = np.asarray(timestamps, dtype=np.float64)
    t_now = ts[int(frame)]
    plan = []
    for k in range(1, n_layers + 1):
        age = history_s * k / n_layers
        t_src = t_now - age
        if t_src < ts[0]:
            plan.append((k, 0, age, 0.0, k / n_layers))              # before the recording starts: nothing to show
            continue
        src = int(np.clip(np.searchsorted(ts, t_src, side="right") - 1, 0, len(ts) - 1))
        age01 = age / history_s
        plan.append((k, src, float(t_now - ts[src]), float((1.0 - age01 + 1.0 / n_layers) ** decay
                                                              if age01 < 1.0 else (1.0 / n_layers) ** decay), age01))
    return plan


def layer_alpha(plan, k, mode, strength=0.9):
    if mode == 0:
        return 0.0
    return strength * plan[k - 1][3]


def layer_source_frame(plan, k):
    return plan[k - 1][1]


def history_array(arr_xyz, arr_dyn, mode):
    """Valid flag (alpha channel of the position texture) for a history layer.
    Echo mode keeps only moving-object points; the others keep every valid point."""
    out = np.array(arr_xyz, dtype=np.float32, copy=True)
    if mode == 1:
        out[..., 3] = out[..., 3] * (arr_dyn[..., 0] > 0.5)
    return out


def time_offset(age_s, mode, m_per_s, axis=(0.0, 1.0, 0.0)):
    """ARTISTIC displacement for Time Explosion only (default: up the world y axis). Zero in all other modes."""
    if mode != 6:
        return (0.0, 0.0, 0.0)
    return tuple(float(a * age_s * m_per_s) for a in axis)


# ---- expression / callback entry points (arguments are passed explicitly so TouchDesigner tracks dependencies) ----
_plan_cache = {}


def _plan(frame, history_s, decay):
    import td
    comp = td.op("/project1/ROBOT_VISION_4D/TEMPORAL_4D")
    key = (int(frame), round(float(history_s), 4), round(float(decay), 4))
    if key not in _plan_cache:
        fl = td.op("/project1/ROBOT_VISION_4D/DATA_INPUT/frame_loader").module
        if len(_plan_cache) > 64:
            _plan_cache.clear()
        _plan_cache[key] = layer_plan(fl.timestamps(), key[0], key[1], int(comp.fetch("n_layers")), key[2])
    return _plan_cache[key]


def source_frame(k, frame, history_s, decay):
    return layer_source_frame(_plan(frame, history_s, decay), k)


def alpha_k(k, frame, mode, history_s, decay, strength):
    return layer_alpha(_plan(frame, history_s, decay), k, int(round(mode)), strength)


def age01_k(k, frame, history_s, decay):
    return _plan(frame, history_s, decay)[k - 1][4]


def offset_k(k, frame, mode, history_s, decay, m_per_s, axis_index):
    age_s = _plan(frame, history_s, decay)[k - 1][2]
    return time_offset(age_s, int(round(mode)), m_per_s)[int(axis_index)]
