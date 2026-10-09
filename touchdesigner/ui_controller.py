"""Control-panel logic (Text-DAT module in UI): layer flags from the toggles, radio groups, render style, and the
sensor-statistics text. Everything except panel_text() is pure python and unit-tested outside TouchDesigner.

Render style (spec section 13): 0 SCIENTIFIC = the sensor data with no artistic effects (no glow/vignette/haze),
1 CINEMATIC = restrained glow on LiDAR/box highlights, soft vignette and a faint haze. Measured positions and colours
are identical in both; only post-processing differs.
"""

# view presets set the three layer toggles (stereo, lidar, depth-error colouring)
VIEW_PRESETS = {1: (1, 0, 0), 2: (0, 1, 0), 3: (1, 1, 0), 4: (1, 1, 1)}


def layer_flags(stereo, lidar, error):
    """Which clouds to draw. Depth disagreement recolours the LiDAR, so it needs the LiDAR layer."""
    s, l, e = float(stereo) > 0.5, float(lidar) > 0.5, float(error) > 0.5
    return {"stereo": s, "lidar": l and not e, "lidar_error": l and e}


def view_name(stereo, lidar, error):
    s, l, e = float(stereo) > 0.5, float(lidar) > 0.5, float(error) > 0.5
    parts = (["STEREO"] if s else []) + (["LIDAR"] if l else [])
    if l and e:
        parts.append("DEPTH ERROR")
    return " + ".join(parts) if parts else "NOTHING"


def radio_index(values, default=1):
    """1-based index of the first truthy value in a radio group, else `default`."""
    for i, v in enumerate(values, start=1):
        if float(v) > 0.5:
            return i
    return default


def post_params(style, glow_on, glow_intensity):
    """Uniforms for the glow/atmosphere TOP. Scientific style (0) returns all zeros: the input passes through unchanged."""
    cinematic = float(style) > 0.5
    return {"uGlow": float(glow_intensity) if (cinematic and float(glow_on) > 0.5) else 0.0,
            "uRadius": 3.0,
            "uVignette": 0.35 if cinematic else 0.0,
            "uHaze": 1.0 if cinematic else 0.0}


TEMPORAL_NAMES = {0: "OFF", 1: "MOTION ECHO", 2: "POINT TRAILS", 6: "TIME EXPLOSION (artistic axis, NOT measured)"}


def panel_text(frame, stereo, lidar, error, good, large, tmode=0, hist=0.0, style=1):
    """Expression entry point for the Text TOP: looks up the current frame's stats. Arguments are passed in
    the expression (not read here) so TouchDesigner tracks the dependency and recooks when they change."""
    import td
    fl = td.op("/project1/ROBOT_VISION_4D/DATA_INPUT/frame_loader").module
    txt = stats_text(fl.frame_stats(int(frame)), frame, view_name(stereo, lidar, error), good, large)
    t = int(round(tmode))
    if t:
        txt += f"\ntemporal: {TEMPORAL_NAMES.get(t, '?')}, history {hist:.1f} s (measured sensor time)"
    txt += "\nstyle: " + ("CINEMATIC" if float(style) > 0.5 else "SCIENTIFIC (no artistic effects)")
    return txt


def stats_text(stats, frame, view, good, large):
    """Panel text from the per-frame comparison stats written by preprocessing. Not a verdict on stereo quality:
    disagreement also comes from calibration, occlusion, reflectivity, sync error and moving objects."""
    def f(key, fmt):
        v = stats.get(key)
        return "n/a" if v is None or v != v else fmt % v
    return (
        f"SENSOR STATISTICS   frame {int(frame)}\n"
        f"view: {view}\n"
        f"LiDAR pts in camera view : {stats.get('lidar_points_in_view', 0)}\n"
        f"comparable points        : {stats.get('comparable_points', 0)}\n"
        f"valid comparisons        : {f('valid_comparison_pct', '%.1f %%')}\n"
        f"mean |dZ|                : {f('mean_abs_diff_m', '%.3f m')}\n"
        f"median |dZ|              : {f('median_abs_diff_m', '%.3f m')}\n"
        f"90th pct |dZ|            : {f('p90_abs_diff_m', '%.3f m')}\n"
        f"large-disagreement pts   : {stats.get('large_disagreement_points', 0)}  (counted at export thr. 0.50 m)\n"
        f"colour thresholds (live) : good < {good:.2f} m, large > {large:.2f} m  (visual only)\n"
        f"disagreement != stereo error (occlusion, sync, calibration, motion)"
    )
