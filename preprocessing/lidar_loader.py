"""Ouster OS1-128 scan loading.

Documented format: float32 binary, x y z intensity, reshapeable to 128 x 1024 x 4, LiDAR convention
(+x forward, +y left, +z up), metres. The devkit's reader reshapes with `-1` columns, so files may carry
extra fields (e.g. time); we detect the width from the file size and keep the first four.
"""
from pathlib import Path

import numpy as np

RINGS, COLUMNS = 128, 1024


def read_scan(path):
    """-> (128, 1024, 4) float32 [x, y, z, intensity]."""
    raw = np.fromfile(Path(path), dtype=np.float32)
    n = RINGS * COLUMNS
    if raw.size % n != 0:
        raise ValueError(f"{path}: {raw.size} floats is not a multiple of {n} (128x1024 scan)")
    width = raw.size // n
    if width < 4:
        raise ValueError(f"{path}: only {width} fields per point; need x y z intensity")
    return raw.reshape(RINGS, COLUMNS, width)[:, :, :4].copy()


def valid_mask(scan, min_range=0.5, max_range=80.0):
    xyz = scan[..., :3]
    r = np.linalg.norm(xyz, axis=-1)
    return np.isfinite(xyz).all(-1) & (r >= min_range) & (r <= max_range)


def normalize_intensity(intensity, valid, percentiles=(1, 99)):
    """Robust 0..1 normalisation (the dataset does not document an intensity range)."""
    vals = intensity[valid]
    if vals.size == 0:
        return np.zeros_like(intensity)
    lo, hi = np.percentile(vals, percentiles)
    if hi <= lo:
        return np.zeros_like(intensity)
    return np.clip((intensity - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def decimate_columns(arr, stride):
    return arr[:, ::stride]
