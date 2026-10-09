"""Pure-numpy logic from the TouchDesigner-side modules (they import without TouchDesigner)."""
import importlib.util
from pathlib import Path

import numpy as np

TD = Path(__file__).resolve().parents[1] / "touchdesigner"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TD / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


bg = _load("box_generator")
ui = _load("ui_controller")


def _cam(pos):
    W = np.eye(4)
    W[:3, 3] = pos            # column-vector layout, camera looking down local -z (TouchDesigner default)
    return W.flatten().tolist()


def test_point_straight_ahead_projects_to_image_centre():
    u, v, ok = bg.project(np.array([[0.0, 0.0, -10.0]]), _cam([0, 0, 0]), 45.0, 1280, 720)
    assert ok[0] and np.isclose(u[0], 640) and np.isclose(v[0], 360)


def test_horizontal_fov_maps_edge_to_image_edge():
    d = 10.0
    x_edge = d * np.tan(np.radians(45.0) / 2)                    # on the horizontal half-FOV
    u, v, ok = bg.project(np.array([[x_edge, 0, -d]]), _cam([0, 0, 0]), 45.0, 1280, 720)
    assert np.isclose(u[0], 1280)
    y_edge = x_edge * 720 / 1280                                 # vertical half-extent is scaled by aspect
    u, v, ok = bg.project(np.array([[0, y_edge, -d]]), _cam([0, 0, 0]), 45.0, 1280, 720)
    assert np.isclose(v[0], 720)


def test_camera_translation_and_behind_camera():
    u, v, ok = bg.project(np.array([[5.0, 2.0, 3.0]]), _cam([5, 2, 13]), 60.0, 800, 600)
    assert ok[0] and np.isclose(u[0], 400) and np.isclose(v[0], 300)    # directly ahead of the moved camera
    _, _, ok = bg.project(np.array([[0.0, 0.0, 10.0]]), _cam([0, 0, 0]), 60.0, 800, 600)
    assert not ok[0]                                                    # behind the camera


def test_box_selection_by_range():
    boxes = [{"center": [0, 0, 0]}, {"center": [30, 0, 0]}, {"center": [50, 0, 0]}]
    assert len(bg.select(boxes, np.zeros(3), 40.0)) == 2


def test_layer_flags_from_toggles():
    assert ui.layer_flags(1, 0, 0) == {"stereo": True, "lidar": False, "lidar_error": False}
    assert ui.layer_flags(0, 1, 0) == {"stereo": False, "lidar": True, "lidar_error": False}
    assert ui.layer_flags(1, 1, 0) == {"stereo": True, "lidar": True, "lidar_error": False}
    assert ui.layer_flags(1, 1, 1) == {"stereo": True, "lidar": False, "lidar_error": True}
    assert ui.layer_flags(1, 0, 1) == {"stereo": True, "lidar": False, "lidar_error": False}   # error colouring needs LiDAR


def test_view_presets_reproduce_the_four_spec_modes():
    names = {m: ui.view_name(*ui.VIEW_PRESETS[m]) for m in (1, 2, 3, 4)}
    assert names == {1: "STEREO", 2: "LIDAR", 3: "STEREO + LIDAR", 4: "STEREO + LIDAR + DEPTH ERROR"}
    assert ui.view_name(0, 0, 0) == "NOTHING"


def test_radio_index_picks_first_pressed_or_default():
    assert ui.radio_index([0, 1, 0]) == 2 and ui.radio_index([0, 0, 0]) == 1 and ui.radio_index([0, 0], default=0) == 0
    assert ui.radio_index([1, 1, 0]) == 1


def test_scientific_style_has_no_post_effects_and_cinematic_does():
    sci = ui.post_params(0, 1, 0.8)
    assert sci["uGlow"] == 0.0 and sci["uVignette"] == 0.0 and sci["uHaze"] == 0.0
    cin = ui.post_params(1, 1, 0.8)
    assert cin["uGlow"] == 0.8 and cin["uVignette"] > 0 and cin["uHaze"] > 0
    assert ui.post_params(1, 0, 0.8)["uGlow"] == 0.0                         # glow checkbox off


def test_stats_text_handles_missing_values():
    s = ui.stats_text({"comparable_points": 12, "median_abs_diff_m": float("nan")}, 3, "STEREO + LIDAR", 0.2, 0.5)
    assert "comparable points        : 12" in s and "n/a" in s and "frame 3" in s and "view: STEREO + LIDAR" in s


tc = _load("temporal_controller")


def _ts(n=100, rate=10.0, jitter=0.0):
    rng = np.random.default_rng(0)
    return 1000.0 + np.arange(n) / rate + rng.normal(0, jitter, n).cumsum() * 0


def test_layer_ages_follow_measured_timestamps_not_frame_indices():
    ts = np.array([0.0, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])       # a 0.3 s dropout between frames 2 and 3
    plan = tc.layer_plan(ts, frame=9, history_s=0.4, n_layers=4, decay=1.0)
    # layer k targets t = 1.1 - 0.1k  -> frames at/before 1.0, 0.9, 0.8, 0.7
    assert [p[1] for p in plan] == [8, 7, 6, 5]
    plan = tc.layer_plan(ts, frame=5, history_s=0.66, n_layers=3, decay=1.0)
    # t_now = 0.7: targets 0.48, 0.26, 0.04 (off frame boundaries) -> latest frames at/before: 2, 2 (dropout), 0
    assert [p[1] for p in plan] == [2, 2, 0]


def test_no_history_before_recording_start_has_zero_alpha():
    ts = _ts()
    plan = tc.layer_plan(ts, frame=5, history_s=3.0, n_layers=8, decay=1.5)
    assert plan[0][3] > 0 and plan[-1][3] == 0.0 and plan[-1][1] == 0


def test_alpha_decays_monotonically_with_age_and_is_bounded():
    ts = _ts()
    plan = tc.layer_plan(ts, frame=90, history_s=3.0, n_layers=8, decay=1.5)
    alphas = [p[3] for p in plan]
    assert all(0.0 < a <= 1.0 for a in alphas) and alphas == sorted(alphas, reverse=True)
    assert tc.layer_alpha(plan, 1, 0) == 0.0 and tc.layer_alpha(plan, 1, 2) > tc.layer_alpha(plan, 8, 2)


def test_echo_mode_keeps_only_moving_points_and_leaves_input_untouched():
    xyz = np.ones((2, 3, 4), dtype=np.float32)
    dyn = np.zeros((2, 3, 4), dtype=np.float32); dyn[0, 1, 0] = 1
    out = tc.history_array(xyz, dyn, 1)
    assert out[..., 3].sum() == 1 and out[0, 1, 3] == 1 and xyz[..., 3].sum() == 6
    assert tc.history_array(xyz, dyn, 2)[..., 3].sum() == 6


def test_time_offset_is_artistic_and_only_in_explosion_mode():
    assert tc.time_offset(2.0, 1, 2.0) == (0.0, 0.0, 0.0) and tc.time_offset(2.0, 2, 2.0) == (0.0, 0.0, 0.0)
    assert tc.time_offset(2.0, 6, 2.0) == (0.0, 4.0, 0.0)
