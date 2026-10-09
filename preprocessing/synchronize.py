"""Timestamp handling. All matching uses real sensor timestamps, never frame-index arithmetic."""
import numpy as np


def load_timestamps(path):
    """CODa `timestamps/{seq}.txt`: one timestamp per line, line index == frame number."""
    ts = np.loadtxt(path, dtype=np.float64, ndmin=1)
    if np.any(np.diff(ts) <= 0):
        raise ValueError(f"{path}: timestamps are not strictly increasing")
    return ts


def nearest_indices(query, reference):
    """For each query time return (index into reference, signed dt = reference[idx] - query)."""
    query = np.asarray(query, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    j = np.searchsorted(reference, query)
    j0 = np.clip(j - 1, 0, len(reference) - 1)
    j1 = np.clip(j, 0, len(reference) - 1)
    pick_prev = np.abs(reference[j0] - query) <= np.abs(reference[j1] - query)
    idx = np.where(pick_prev, j0, j1)
    return idx, reference[idx] - query


def match_streams(ts_a, ts_b, max_dt):
    """Mutual-nearest matching between two streams. Returns index arrays (ia, ib) of accepted pairs."""
    ia_to_b, dt_ab = nearest_indices(ts_a, ts_b)
    ib_to_a, _ = nearest_indices(ts_b, ts_a)
    ia = np.arange(len(ts_a))
    ok = (np.abs(dt_ab) <= max_dt) & (ib_to_a[ia_to_b] == ia)
    return ia[ok], ia_to_b[ok]


def estimate_rate(ts):
    """Measured sampling statistics from timestamps."""
    dt = np.diff(np.asarray(ts, dtype=np.float64))
    return {
        "n": int(len(ts)),
        "duration_s": float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0,
        "mean_hz": float(1.0 / dt.mean()) if len(dt) else float("nan"),
        "median_dt_s": float(np.median(dt)) if len(dt) else float("nan"),
        "std_dt_s": float(dt.std()) if len(dt) else float("nan"),
        "max_gap_s": float(dt.max()) if len(dt) else float("nan"),
    }


def frame_at_time(timestamps, t):
    """Largest frame index whose timestamp <= t (clamped). Used for real-time playback."""
    return int(np.clip(np.searchsorted(timestamps, t, side="right") - 1, 0, len(timestamps) - 1))
