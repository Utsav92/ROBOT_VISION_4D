"""TouchDesigner-side frame access (loaded as a module from a file-linked Text DAT).

Reads the .npy/.json frames written by preprocessing/frame_exporter.py. Keeps a small LRU cache and prefetches
upcoming frames on a worker thread so Script TOP cooks never wait on disk during playback.
"""
import json
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

_root = None
_manifest = None
_cache = OrderedDict()
_lock = threading.Lock()
MAX_BYTES = 450 * 1024 * 1024          # LRU budget by bytes (laptop-friendly); entries are evicted oldest-first
_bytes = 0
_prefetching = False

_EXT = {"boxes": "json"}


DATA_COMP = "/project1/ROBOT_VISION_4D/DATA_INPUT"


def set_root(path):
    """Point the loader at output/processed/seqN and read its manifest."""
    global _root, _manifest
    _root = Path(path)
    _manifest = json.loads((_root / "manifest.json").read_text())
    global _bytes
    with _lock:
        _cache.clear()
        _bytes = 0
    return _manifest["n_frames"]


def refresh():
    """Live mode: re-read the manifest the ROS 2 bridge keeps appending to. Returns True if new frames appeared.
    A half-written manifest (the bridge replaces it atomically, but be defensive) keeps the previous one."""
    global _manifest
    _ensure()
    try:
        m = json.loads((_root / "manifest.json").read_text())
    except (OSError, ValueError):
        return False
    if m.get("n_frames", 0) == _manifest["n_frames"]:
        return False
    _manifest = m
    return True


def _ensure():
    """A file-synced Text DAT re-executes its module on reload, wiping module globals. The data directory is
    therefore kept in the DATA_INPUT component's storage and re-read lazily."""
    if _manifest is None:
        import td
        set_root(td.op(DATA_COMP).fetch("data_dir"))


def manifest():
    _ensure()
    return _manifest


def n_frames():
    _ensure()
    return _manifest["n_frames"]


def timestamps():
    _ensure()
    return np.array([f["timestamp"] for f in _manifest["frames"]], dtype=np.float64)


def frame_stats(i):
    _ensure()
    return _manifest["frames"][_clamp(i)]["stats"]


def _clamp(i):
    return int(min(max(int(i), 0), n_frames() - 1))


def _load(kind, i):
    p = _root / kind / f"{i:05d}.{_EXT.get(kind, 'npy')}"
    if kind == "boxes":
        return json.loads(p.read_text())
    return np.load(p)


def _size(v):
    return v.nbytes if hasattr(v, "nbytes") else 4096


def get(kind, i):
    global _bytes
    _ensure()
    i = _clamp(i)
    key = (kind, i)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    val = _load(kind, i)
    with _lock:
        if key not in _cache:
            _cache[key] = val
            _bytes += _size(val)
        while _bytes > MAX_BYTES and len(_cache) > 1:
            _, old = _cache.popitem(last=False)
            _bytes -= _size(old)
    return val


def prefetch(i, kinds=("lidar_xyz", "lidar_echo", "lidar_attr", "stereo_xyz", "stereo_rgb", "boxes", "pose"), ahead=4, direction=1):
    """Warm the cache for the next few frames in the playback direction on a worker thread (one at a time)."""
    global _prefetching
    if _prefetching:
        return

    def work():
        global _prefetching
        try:
            for k in range(1, ahead + 1):
                for kind in kinds:
                    get(kind, i + direction * k)
        except Exception:
            pass
        finally:
            _prefetching = False

    _prefetching = True
    threading.Thread(target=work, daemon=True).start()
