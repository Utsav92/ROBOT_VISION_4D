"""Which LiDAR points belong to MOVING objects (for motion echoes / temporal trails).

Motion is measured, not guessed from the class name: an annotated instance is "moving" if its box centre, mapped
into the world frame with the recorded robot pose, travels at least `min_disp_m` over the processed window. Because the
robot's own motion is removed by the pose, parked objects seen from a moving robot stay "static". Boxes that carry no
instance id cannot be tracked; only those fall back to a class list (and are flagged).
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

FALLBACK_DYNAMIC_CLASSES = ("Pedestrian", "Bike", "Scooter", "Skateboarder", "Car", "Service Vehicle",
                            "Utility Vehicle", "Delivery Truck", "Pickup Truck", "Cart")


def points_in_box(points, center, lwh, rpy, margin=0.1):
    """Boolean mask of (N,3) points inside an oriented box (R = Rz Ry Rx as in object_annotations)."""
    Rm = R.from_euler("xyz", list(rpy), degrees=False).as_matrix()
    local = (np.asarray(points, dtype=np.float64) - np.asarray(center)) @ Rm      # R^T (p - c) for row vectors
    return np.all(np.abs(local) <= (np.asarray(lwh) / 2.0 + margin), axis=1)


def moving_instances(world_centers_by_instance, min_disp_m=1.0):
    """{instance: [world centre, ...]} -> set of instances whose centre travelled >= min_disp_m."""
    moving = set()
    for inst, centres in world_centers_by_instance.items():
        c = np.asarray(centres, dtype=np.float64)
        if len(c) >= 2 and np.linalg.norm(np.ptp(c, axis=0)) >= min_disp_m:
            moving.add(inst)
    return moving


def dynamic_mask(points_os1, boxes, moving, fallback_classes=FALLBACK_DYNAMIC_CLASSES, margin=0.1):
    """(N,3) LiDAR-frame points + the frame's Box list -> (mask bool (N,), track id float (N,), -1 where none)."""
    mask = np.zeros(len(points_os1), dtype=bool)
    ids = np.full(len(points_os1), -1.0, dtype=np.float32)
    finite = np.isfinite(points_os1).all(1)
    for b in boxes:
        is_dyn = (b.instance_id in moving) if b.id_available else (b.class_id in fallback_classes)
        if not is_dyn:
            continue
        inside = points_in_box(np.where(finite[:, None], points_os1, 1e9), b.center, b.lwh, b.rpy, margin) & finite
        mask |= inside
        ids[inside] = float(b.track_number) if b.id_available else -2.0
    return mask, ids
