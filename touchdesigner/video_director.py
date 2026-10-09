"""Cinematic video director (Text-DAT module in VIDEO). Pure numpy: given a video time t it returns everything needed to
render that frame deterministically: the recorded-data time, the control values, the on-screen text and the camera.

Seven scenes, 30 s (spec section 14). The recorded sequence is only ~16.6 s long, so video time and data time differ:
the opening scenes replay slowly, scene 5 slows time further, and the data never runs past its end. The video therefore
uses recorded data from 2.0 s to 16.4 s; nothing is invented.

Honesty rules baked into the text overlays: the 3D boxes are CODa's human ground-truth annotations (not detections),
and the Time-Explosion mode is not used here, so every drawn position is a measured one.
"""
import numpy as np

SCALE = 1.5                              # the spec's 30 s storyboard stretched to 45 s (every scene keeps its proportions)
DURATION = 30.0 * SCALE
FPS = 30
# (base video t, recorded-data t) knots on the 30 s storyboard; per-scene data speed (before stretching):
# S1 0, S2 0.2x, S3 0.5x, S4 0.75x, S5 0.3x (time slows), S6 1x, S7 0.25x. Stretching the video slows each of them by SCALE.
_BASE_KNOTS = [(0.0, 2.0), (3.0, 2.0), (6.0, 2.6), (10.0, 4.6), (14.0, 7.6), (20.0, 9.4), (26.0, 15.4), (30.0, 16.4)]
DATA_KNOTS = [(t * SCALE, d) for t, d in _BASE_KNOTS]
_BASE_SCENES = [
    (0.0, 3.0, "ROBOT VISION 4D", "SENSOR AWAKENING  |  RGB STEREO RECONSTRUCTION"),
    (3.0, 6.0, "RGB + LIDAR FUSION", "OUSTER OS1-128 LIDAR + STEREO CAMERAS"),
    (6.0, 10.0, "SPATIAL PERCEPTION", "3D BOXES = CODa GROUND-TRUTH ANNOTATIONS"),
    (10.0, 14.0, "METRIC DEPTH", "STEREO vs LIDAR AGREEMENT"),
    (14.0, 20.0, "TEMPORAL RECONSTRUCTION", "WORLD-SPACE HISTORY  |  TIME SLOWED"),
    (20.0, 26.0, "SPATIAL PERCEPTION", "VIRTUAL FLY-THROUGH"),
    (26.0, 30.0, "ROBOT VISION 4D", "FULL PERCEPTION  |  UT AUSTIN CODa"),
]
SCENES = [(a * SCALE, b * SCALE, title, sub) for a, b, title, sub in _BASE_SCENES]
BLEND_S = 0.8                                            # camera hand-over between scenes (seconds)
BLEND_BY_SCENE = {5: 1.4}                                # scene 5 -> 6 swaps subject: longer hand-over


def ease(x):
    x = min(max(float(x), 0.0), 1.0)
    return x * x * x * (x * (6 * x - 15) + 10)


def scene_index(t):
    t = min(max(t, 0.0), DURATION - 1e-9)
    for i, (a, b, *_rest) in enumerate(SCENES):
        if a <= t < b:
            return i
    return len(SCENES) - 1


def data_time(t, data_end=None):
    ts, ds = zip(*DATA_KNOTS)
    d = float(np.interp(t, ts, ds))
    return min(d, data_end) if data_end is not None else d


def fade(t):
    """Fade from black over 1 s, to black over 0.6 s."""
    return float(min(1.0, max(t, 0.0) / 1.0) * min(1.0, max(DURATION - t, 0.0) / 0.6))


def overlay(t):
    """(title, subtitle, alpha 0..1): fades in over 0.5 s at each scene start and out over 0.4 s at its end."""
    i = scene_index(t)
    a, b, title, sub = SCENES[i]
    alpha = min(1.0, (t - a) / 0.5) * min(1.0, (b - t) / 0.4)
    return title, sub, float(max(0.0, min(1.0, alpha)))


def settings(t):
    """Control values for time t. Keys match the panel: toggles are 0/1, temporal_mode in {0,1,2}."""
    i = scene_index(t)
    a, b = SCENES[i][0], SCENES[i][1]
    u = min(max((t - a) / (b - a), 0.0), 1.0)                        # progress through the current scene, 0..1
    s = dict(density=1.0, lidar_opacity=1.0, show_stereo=1, show_lidar=1, show_error=0, show_boxes=0, show_labels=0,
             show_stats=0, show_grid=0, show_trajectory=0, box_range=40.0, temporal_mode=0, history_s=3.0,
             fade=fade(t))
    if i == 0:                                                       # points scatter in out of the dark
        s.update(density=0.02 + 0.98 * ease(u), show_lidar=0)
    elif i == 1:                                                     # LiDAR activates over the first two thirds
        s.update(lidar_opacity=ease(u / (2.0 / 3.0)))
    elif i == 2:                                                     # boxes reveal outward over the first three quarters
        s.update(show_boxes=1, show_labels=1, show_grid=1, show_trajectory=1, box_range=2.0 + 38.0 * ease(u / 0.75))
    elif i == 3:                                                     # depth intelligence
        s.update(show_boxes=1, show_labels=1, show_grid=1, show_trajectory=1, show_error=1, show_stats=1)
    elif i == 4:                                                     # 4D: echo (first half), then point trails
        s.update(show_boxes=1, show_labels=0, show_grid=1, show_trajectory=1)
        if u < 0.5:
            s.update(temporal_mode=1, history_s=0.5 + 2.5 * ease(u / 0.5))
        else:
            s.update(temporal_mode=2, history_s=1.5)
    elif i == 5:                                                     # clean fly-through
        s.update(show_boxes=1, show_labels=1, show_grid=1, show_trajectory=1)
    else:                                                            # pull back to the full picture
        s.update(show_boxes=1, show_labels=0, show_grid=1, show_trajectory=1)
    return s


# ---------------------------------------------------------------------------------------------------------------
class Director:
    """Camera script. `cc` is the camera_controller module (look_at, blend_pose); ctx is (robot_pose 4x4 in world TD
    axes, list of moving-pedestrian centres in world)."""

    def __init__(self, cc):
        self.cc = cc

    @staticmethod
    def _w(pose, local):
        return pose[:3, 3] + pose[:3, :3] @ np.asarray(local, dtype=np.float64)

    @staticmethod
    def _lerp(a, b, u):
        return (1 - u) * np.asarray(a, dtype=np.float64) + u * np.asarray(b, dtype=np.float64)

    @staticmethod
    def _catmull(P, u):
        P = np.asarray(P, dtype=np.float64)
        n = len(P) - 1
        x = min(max(u, 0.0), 1.0) * n
        k = min(int(x), n - 1)
        f = x - k
        p0, p1, p2, p3 = P[max(k - 1, 0)], P[k], P[k + 1], P[min(k + 2, n)]
        return 0.5 * ((2 * p1) + (-p0 + p2) * f + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f + (-p0 + 3 * p1 - 3 * p2 + p3) * f ** 3)

    def _scene_pose(self, i, u, ctx):
        """(eye, target, fov) for scene i at eased progress u in [0,1]."""
        pose, peds = ctx
        w, lerp = self._w, self._lerp
        if i == 0:
            return w(pose, lerp((0, 2.8, 9.0), (0, 2.4, 7.0), u)), w(pose, (0, 1.0, -6.0)), 55.0 - 5.0 * u
        if i == 1:
            return w(pose, lerp((0, 2.4, 7.0), (3.0, 2.6, 5.0), u)), w(pose, (0, 1.0, -6.0)), 50.0
        if i == 2:
            return w(pose, lerp((3.0, 2.6, 5.0), (7.0, 3.5, -1.0), u)), w(pose, lerp((0, 1.0, -6.0), (-1.0, 1.2, -7.0), u)), 50.0
        if i == 3:
            return w(pose, lerp((7.0, 3.5, -1.0), (-3.0, 2.0, -2.0), u)), w(pose, lerp((-1.0, 1.2, -7.0), (0, 0.8, -8.0), u)), 52.0
        if i == 4:
            c = np.mean(peds, axis=0) if len(peds) else w(pose, (0, 0.0, -5.0))
            az = np.radians(40.0 + 70.0 * u)
            eye = c + np.array([5.5 * np.sin(az), 1.8, 5.5 * np.cos(az)])
            return eye, c + np.array([0.0, 1.0, 0.0]), 50.0
        if i == 5:
            path = [(3.2, 1.8, -8.0), (0.5, 1.3, -4.0), (-2.5, 1.5, -9.0), (0.5, 2.2, -13.0), (0.0, 3.5, -16.0)]
            pts = [w(pose, p) for p in path]
            eye = self._catmull(pts, u)
            # aim along the path tangent (central difference, clamped at the ends): always well defined, never
            # degenerate even where a look-ahead point would land on the camera itself
            tangent = self._catmull(pts, min(u + 0.03, 1.0)) - self._catmull(pts, max(u - 0.03, 0.0))
            tangent = tangent / max(np.linalg.norm(tangent), 1e-9)
            return eye, eye + 6.0 * tangent + np.array([0.0, 0.2, 0.0]), 60.0
        return w(pose, lerp((0, 3.5, -16.0), (0, 26.0, 10.0), u)), w(pose, lerp((0, 3.5, -23.0), (0, 0.0, -6.0), u)), 60.0 - 5.0 * u

    def _raw(self, t, i, ctx):
        a, b = SCENES[i][0], SCENES[i][1]
        eye, target, fov = self._scene_pose(i, ease((t - a) / (b - a)), ctx)
        return self.cc.look_at(eye, target), fov

    def camera(self, t, ctx):
        """-> (4x4 world matrix, horizontal fov deg). Blends from the previous scene's end pose for BLEND_S seconds."""
        i = scene_index(t)
        W, fov = self._raw(t, i, ctx)
        a = SCENES[i][0]
        bs = BLEND_BY_SCENE.get(i, BLEND_S)
        if i > 0 and t - a < bs:
            Wp, fp = self._raw(SCENES[i - 1][1], i - 1, ctx)               # previous scene's final pose, current robot
            k = ease((t - a) / bs)
            W, fov = self.cc.blend_pose(Wp, W, k), (1 - k) * fp + k * fov
        return W, float(fov)
