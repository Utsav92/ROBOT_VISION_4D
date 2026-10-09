"""World reference geometry (Text-DAT module in WORLD_REFERENCE): coordinate grid and robot trajectory.

Pure numpy producers + thin Script SOP fillers. The grid is snapped to the spacing so it does not swim as the robot
moves; its fade with distance comes from the Line MAT's near/far colour, not from per-vertex data.
"""
import numpy as np


def grid_lines(center_xz, half_extent=60.0, spacing=5.0, ground_y=0.0):
    """Line segments (M,2,3) of a ground grid centred (snapped) on center_xz = (x, z)."""
    cx = np.round(center_xz[0] / spacing) * spacing
    cz = np.round(center_xz[1] / spacing) * spacing
    ticks = np.arange(-half_extent, half_extent + 1e-6, spacing)
    segs = []
    for t in ticks:
        segs.append([[cx + t, ground_y, cz - half_extent], [cx + t, ground_y, cz + half_extent]])
        segs.append([[cx - half_extent, ground_y, cz + t], [cx + half_extent, ground_y, cz + t]])
    return np.array(segs, dtype=np.float64)


def ground_height(points_xyz, valid, center_xz, radius=10.0, percentile=5.0):
    """Estimate the ground y (TD up axis) near the robot from LiDAR points: a low percentile of nearby heights."""
    p = points_xyz[valid > 0.5]
    near = p[np.hypot(p[:, 0] - center_xz[0], p[:, 2] - center_xz[1]) < radius]
    return float(np.percentile(near[:, 1], percentile)) if len(near) else float(p[:, 1].min())


def fill_polylines(sop, segments):
    """Rebuild a Script SOP as open 2-point polylines from segments (M,2,3)."""
    sop.clear()
    for a, b in segments:
        poly = sop.appendPoly(2, closed=False, addPoints=True)
        for k, q in enumerate((a, b)):
            pt = poly[k].point
            pt.x, pt.y, pt.z = float(q[0]), float(q[1]), float(q[2])
    return len(segments)


def fill_path(sop, positions):
    """One open polyline through all positions (N,3)."""
    sop.clear()
    if len(positions) < 2:
        return 0
    poly = sop.appendPoly(len(positions), closed=False, addPoints=True)
    for k, q in enumerate(positions):
        pt = poly[k].point
        pt.x, pt.y, pt.z = float(q[0]), float(q[1]), float(q[2])
    return len(positions)
