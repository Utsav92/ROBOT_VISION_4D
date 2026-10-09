"""Tests are written against the spec's 30 s storyboard times (B(...)) multiplied by vd.SCALE, so they hold for any length."""
import importlib.util
from pathlib import Path

import numpy as np

TD = Path(__file__).resolve().parents[1] / "touchdesigner"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TD / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


vd = _load("video_director")
cc = _load("camera_controller")
B = lambda base_t: base_t * vd.SCALE                      # a time on the 30 s spec storyboard -> this video's time
N = int(round(vd.DURATION * vd.FPS))                      # number of video frames


def _ctx(x=0.0):
    P = np.eye(4)
    P[:3, 3] = [x, 0.0, 0.0]
    return P, [np.array([x - 3.0, 0.9, -5.0]), np.array([x - 2.0, 0.9, -6.0])]


def test_video_is_45_seconds_and_scenes_tile_it_exactly():
    assert vd.DURATION == 45.0 and N == 1350
    assert vd.SCENES[0][0] == 0 and vd.SCENES[-1][1] == vd.DURATION
    assert all(a[1] == b[0] for a, b in zip(vd.SCENES, vd.SCENES[1:]))
    assert np.allclose([s[1] - s[0] for s in vd.SCENES], [B(x) for x in (3, 3, 4, 4, 6, 6, 4)])   # spec proportions kept
    probe = [B(x) for x in (0, 2.99, 3, 9.99, 10, 14, 20, 26, 29.99, 30)] + [999]
    assert [vd.scene_index(t) for t in probe] == [0, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6]


def test_data_time_is_monotone_and_never_leaves_the_recording():
    d = np.array([vd.data_time(t) for t in np.linspace(0, vd.DURATION, N + 1)])
    assert np.all(np.diff(d) >= 0) and d.min() >= 2.0 and d.max() <= 16.4 + 1e-9
    assert vd.data_time(B(2.0)) == 2.0                                           # scene 1 holds a still moment
    slopes = [(vd.data_time(b) - vd.data_time(a)) / (b - a) for a, b, *_ in vd.SCENES]
    assert np.allclose(slopes, np.array([0, 0.2, 0.5, 0.75, 0.3, 1.0, 0.25]) / vd.SCALE)   # stretching slows every scene
    assert vd.data_time(vd.DURATION, data_end=16.0) == 16.0                      # clamped to a shorter recording


def test_fade_and_overlay_alpha_bounds():
    assert vd.fade(0) == 0 and vd.fade(vd.DURATION) == 0 and vd.fade(vd.DURATION / 2) == 1 and 0 < vd.fade(0.5) < 1
    for t in np.linspace(0, vd.DURATION, 451):
        title, sub, a = vd.overlay(t)
        assert 0 <= a <= 1 and title and sub
    assert vd.overlay(B(3.0))[2] == 0 and vd.overlay(B(3.0) + 1.0)[2] == 1       # fades through each scene change


def test_overlay_wording_is_honest_about_what_is_drawn():
    all_text = " ".join(s[2] + " " + s[3] for s in vd.SCENES)
    for needed in ("ROBOT VISION 4D", "RGB + LIDAR FUSION", "METRIC DEPTH", "SPATIAL PERCEPTION", "TEMPORAL RECONSTRUCTION"):
        assert needed in all_text
    assert "GROUND-TRUTH" in vd.SCENES[2][3] and "DETECT" not in all_text.upper()   # boxes are annotations, not detections


def test_settings_follow_the_storyboard():
    s = lambda base_t: vd.settings(B(base_t))
    assert s(0.0)["density"] < 0.1 and s(0.0)["show_lidar"] == 0 and s(2.99)["density"] > 0.99
    assert s(3.0)["lidar_opacity"] == 0 and s(5.99)["lidar_opacity"] > 0.99 and s(4.0)["show_lidar"] == 1
    assert s(6.0)["show_boxes"] == 1 and s(6.0)["box_range"] == 2.0 and s(9.99)["box_range"] > 39.9
    assert s(11)["show_error"] == 1 and s(11)["show_stats"] == 1 and s(15)["show_error"] == 0 and s(15)["show_stats"] == 0
    assert s(15)["temporal_mode"] == 1 and s(18)["temporal_mode"] == 2 and s(22)["temporal_mode"] == 0
    assert 0.5 <= s(14)["history_s"] <= s(16.9)["history_s"] <= 3.0
    assert s(16.99)["temporal_mode"] == 1 and s(17.01)["temporal_mode"] == 2          # echo -> trails switches at mid-scene
    for t in np.linspace(0, vd.DURATION, 451):
        v = vd.settings(t)
        assert 0.02 <= v["density"] <= 1 and 0 <= v["lidar_opacity"] <= 1 and 0 <= v["fade"] <= 1 and 2 <= v["box_range"] <= 40
        assert v["temporal_mode"] in (0, 1, 2)                                     # never the artistic Time Explosion


def test_camera_is_continuous_across_every_scene_boundary():
    d = vd.Director(cc)
    for b in (B(x) for x in (3.0, 6.0, 10.0, 14.0, 20.0, 26.0)):
        before, after = d.camera(b - 1e-4, _ctx())[0], d.camera(b + 1e-4, _ctx())[0]
        assert np.linalg.norm(before[:3, 3] - after[:3, 3]) < 0.05, b
        assert np.allclose(before[:3, :3], after[:3, :3], atol=0.02), b


def test_camera_speed_is_bounded_and_poses_are_valid():
    d = vd.Director(cc)
    prev, worst = None, 0.0
    for n in range(0, N + 1):
        t = n / vd.FPS
        W, fov = d.camera(t, _ctx(x=vd.data_time(t)))                              # robot drives forward with data time
        assert np.allclose(W[:3, :3] @ W[:3, :3].T, np.eye(3), atol=1e-9) and 40 <= fov <= 62 and np.isfinite(W).all()
        if prev is not None:
            worst = max(worst, np.linalg.norm(W[:3, 3] - prev) * vd.FPS)           # m/s
        prev = W[:3, 3]
    assert worst < 18.0 / vd.SCALE + 1.0, f"camera whips at {worst:.1f} m/s"        # stretching makes every move slower


def test_each_scene_aims_at_its_own_subject_once_the_handover_is_done():
    d = vd.Director(cc)
    ctx = _ctx()
    checked = 0
    for t in np.linspace(0, vd.DURATION - 0.1, 300):
        i = vd.scene_index(t)
        if t - vd.SCENES[i][0] < vd.BLEND_BY_SCENE.get(i, vd.BLEND_S) + 0.05:
            continue
        W, _ = d.camera(t, ctx)
        a, b = vd.SCENES[i][0], vd.SCENES[i][1]
        eye, target, _fov = d._scene_pose(i, vd.ease((t - a) / (b - a)), ctx)
        want = (target - eye) / np.linalg.norm(target - eye)
        assert np.dot(-W[:3, 2], want) > 0.999, (t, i)
        assert np.linalg.norm(W[:3, 3] - eye) < 1e-9
        checked += 1
    assert checked > 150


def test_no_scene_ever_aims_at_the_sky_or_ground_degenerately():
    d = vd.Director(cc)
    for t in np.linspace(0, vd.DURATION - 0.01, 900):
        W, _ = d.camera(t, _ctx())
        assert -W[1, 2] < 0.97 or vd.scene_index(t) == 6, t                          # only the wide pull-back looks steeply down
        assert W[1, 2] < 0.999                                                       # never straight UP


def test_scene5_orbits_the_pedestrians_at_constant_radius():
    d = vd.Director(cc)
    ctx = _ctx()
    c = np.mean(ctx[1], axis=0)
    radii = [np.hypot(*(d.camera(t, ctx)[0][[0, 2], 3] - c[[0, 2]])) for t in np.linspace(B(15.0), B(19.9), 20)]
    assert np.allclose(radii, 5.5, atol=1e-6)
    no_peds = (ctx[0], [])
    W, _ = d.camera(B(17.0), no_peds)                                              # no moving objects: falls back, no crash
    assert np.isfinite(W).all()
