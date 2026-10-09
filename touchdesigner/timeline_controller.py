"""Playback clock for recorded CODa sequences (Text-DAT module).

Playback follows the MEASURED sensor timestamps (not frame-index arithmetic), so the replay runs at the true
speed of the recorded robot regardless of the TouchDesigner cook rate, and reverse/scrub stay time-correct.
Sensor rate (10 Hz) and the visualisation rate (e.g. 30/60 fps) are independent: the frame shown is the latest
sensor frame at or before the playback time.
"""
import time

import numpy as np

_state = {"t": 0.0, "playing": False, "direction": 1, "speed": 1.0, "frozen": False, "frame": 0, "loop": True}
_ts = None          # timestamps relative to the first frame (s)
_ctl = None         # Constant CHOP receiving frame / time / playing / speed
_last = None
_on_frame = None    # optional callable(frame) fired when the displayed frame changes


def attach(timestamps, control_chop, on_frame=None):
    global _ts, _ctl, _on_frame, _last
    ts = np.asarray(timestamps, dtype=np.float64)
    _ts = ts - ts[0]
    _ctl, _on_frame, _last = control_chop, on_frame, None
    _state.update(t=0.0, frame=0)
    _publish(force=True)


def _ensure():
    """Module globals are wiped if the file-synced Text DAT reloads; re-attach from the component tree."""
    if _ts is None:
        import td
        di = td.op("/project1/ROBOT_VISION_4D/DATA_INPUT")
        attach(di.op("frame_loader").module.timestamps(), di.op("frame_control"))


def duration():
    _ensure()
    return float(_ts[-1])


def _frame_for(t):
    return int(np.clip(np.searchsorted(_ts, t, side="right") - 1, 0, len(_ts) - 1))


def _publish(force=False):
    f = _frame_for(_state["t"])
    changed = f != _state["frame"]
    _state["frame"] = f
    if _ctl is not None:
        _ctl.par.value0 = f
        _ctl.par.value1 = _state["t"]
        _ctl.par.value2 = 1.0 if _state["playing"] else 0.0
        _ctl.par.value3 = _state["speed"] * _state["direction"]
    if changed and _state["playing"]:
        try:
            import td
            td.op("/project1/ROBOT_VISION_4D/DATA_INPUT/frame_loader").module.prefetch(f, direction=_state["direction"])
        except Exception:
            pass
    if (changed or force) and _on_frame:
        _on_frame(f)


def tick():
    """Call once per TouchDesigner frame (Execute DAT onFrameStart)."""
    global _last
    _ensure()
    _state["ticks"] = _state.get("ticks", 0) + 1                  # frame counter for measuring the real frame rate
    try:                                       # playback speed slider lives in UI/main_controls
        import td
        _state["speed"] = max(0.0, float(td.op("/project1/ROBOT_VISION_4D/UI/main_controls")["playback_speed"].eval()))
    except Exception:
        pass
    now = time.perf_counter()
    dt = 0.0 if _last is None else min(now - _last, 0.25)
    _last = now
    if not _state["playing"] or _state["frozen"]:
        return
    t = _state["t"] + dt * _state["speed"] * _state["direction"]
    end = duration()
    if t > end or t < 0:
        if _state["loop"]:
            t = t % end if end > 0 else 0.0
        else:
            t = min(max(t, 0.0), end)
            _state["playing"] = False
    _state["t"] = t
    _publish()


def play():     _ensure(); _state.update(playing=True, direction=1)          # FREEZE is an independent toggle
def reverse():  _ensure(); _state.update(playing=True, direction=-1)
def pause():    _state["playing"] = False; _publish()
def freeze(on=True): _state["frozen"] = bool(on); _publish()
def set_speed(s): _state["speed"] = max(0.0, float(s))
def reset():    _state.update(t=0.0, playing=False, direction=1, frozen=False); _publish(force=True)


def scrub_to_frame(i):
    _ensure()
    i = int(np.clip(i, 0, len(_ts) - 1))
    _state["t"] = float(_ts[i])
    _publish(force=True)


def refresh_live(follow=True):
    """Live mode: pick up frames the bridge has appended, extend the clock, and (optionally) jump to the newest one.
    Call periodically (e.g. from an Execute DAT every ~0.5 s). Untested against a running bridge."""
    import td
    global _ts
    _ensure()
    fl = td.op("/project1/ROBOT_VISION_4D/DATA_INPUT/frame_loader").module
    grew = fl.refresh()
    if grew:
        ts = fl.timestamps()
        _ts = ts - ts[0]
        if follow:
            _state["t"] = float(_ts[-1])
        _publish(force=True)
    return grew


def set_time(t):
    """Jump to recorded-data time t seconds (relative to the first frame) and pause. Used by the video exporter."""
    _ensure()
    _state["playing"] = False
    _state["t"] = float(min(max(t, 0.0), duration()))
    _publish(force=True)


def step(n=1):
    """Step n sensor frames (negative = backward) and pause."""
    _state["playing"] = False
    scrub_to_frame(_state["frame"] + n)


def state():
    return dict(_state)
