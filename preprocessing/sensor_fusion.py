"""Fuse the stereo cloud and LiDAR into a common frame."""
import numpy as np


def voxel_keep_mask(points, ok, voxel):
    """Keep one point per voxel by invalidating the rest (preserves the HxW grid the GPU samples)."""
    if voxel <= 0:
        return ok
    flat = points.reshape(-1, 3)
    okf = ok.reshape(-1).copy()
    idx = np.flatnonzero(okf)
    if idx.size == 0:
        return ok
    keys = np.floor(flat[idx] / voxel).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    keep = np.zeros_like(okf)
    keep[idx[first]] = True
    return keep.reshape(ok.shape)


def sample_colors(bgr, vs, us):
    """RGB (h,w,3) uint8 from a BGR image at the stride grid."""
    return bgr[np.ix_(vs, us)][..., ::-1].copy()


def stereo_to_frame(points_rect, calib, T_target_os1=None):
    """(h,w,3) rect-cam0 points -> LiDAR frame, optionally on to another frame via 4x4 T_target_os1."""
    shp = points_rect.shape
    p = calib.rect0_to_os1(points_rect.reshape(-1, 3))
    if T_target_os1 is not None:
        p = p @ T_target_os1[:3, :3].T + T_target_os1[:3, 3]
    return p.reshape(shp)
