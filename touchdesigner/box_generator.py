"""3D box wireframes and labels (Text-DAT module inside OBJECT_TRACKING).

Boxes arrive from preprocessing as 8 world-space corners in TouchDesigner axes (oriented, not axis-aligned),
so no rotation convention is applied here: the 12 edges are drawn straight between the corners.
"""
import numpy as np

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def select(boxes, robot_pos, max_range):
    """Boxes whose centre is within max_range metres of the robot."""
    out = []
    for b in boxes:
        c = np.asarray(b["center"], dtype=np.float64)
        if np.linalg.norm(c - robot_pos) <= max_range:
            out.append(b)
    return out


def fill_sop(sop, boxes):
    """Rebuild a Script SOP as open 2-point polylines (12 per box)."""
    sop.clear()
    for b in boxes:
        c = np.asarray(b["corners"], dtype=np.float64)
        for i, j in EDGES:
            poly = sop.appendPoly(2, closed=False, addPoints=True)
            for k, idx in enumerate((i, j)):
                p = poly[k].point
                p.x, p.y, p.z = float(c[idx, 0]), float(c[idx, 1]), float(c[idx, 2])
    return len(boxes)


def project(points, cam_world, fov_deg, width, height):
    """World points (N,3) -> pixel (u, v_from_bottom, in_front).

    cam_world: flat row-major list of the Camera COMP's worldTransform[r, c] as read in TouchDesigner 2025
    (column-vector layout: translation in the LAST COLUMN, so world = W @ local). The camera looks down its local
    -z axis and `fov` is the HORIZONTAL field of view (verified: projection[0][0] = 1/tan(fov/2),
    projection[1][1] = that * aspect)."""
    W = np.asarray(cam_world, dtype=np.float64).reshape(4, 4)
    h = (np.linalg.inv(W) @ np.concatenate([points, np.ones((len(points), 1))], axis=1).T).T
    z = -h[:, 2]
    t = np.tan(np.radians(fov_deg) / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_ndc = h[:, 0] / z / t
        y_ndc = h[:, 1] / z / (t * height / width)
    return (x_ndc + 1) * 0.5 * width, (y_ndc + 1) * 0.5 * height, z > 0.1


def label_image(width, height, boxes, cam_world, fov_deg, color=(96, 255, 36)):
    """RGBA uint8 overlay (row 0 = bottom, as Script TOPs expect) with a text label above each visible box."""
    import cv2
    img = np.zeros((height, width, 4), dtype=np.uint8)
    if not boxes:
        return img
    tops = np.array([np.asarray(b["corners"])[4:8].mean(axis=0) for b in boxes])      # centre of top face
    u, v, ok = project(tops, cam_world, fov_deg, width, height)
    top_down = np.zeros((height, width, 4), dtype=np.uint8)
    for b, uu, vv, o in zip(boxes, u, v, ok):
        if not o or not (0 <= uu < width and 0 <= vv < height):
            continue
        txt = b["label"] + ("" if b.get("id_available", True) else " (no id)")
        org = (int(uu) - 4 * len(txt), int(height - vv) - 6)
        cv2.putText(top_down, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(top_down, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (color[0], color[1], color[2], 255), 1, cv2.LINE_AA)
    return np.ascontiguousarray(top_down[::-1])
