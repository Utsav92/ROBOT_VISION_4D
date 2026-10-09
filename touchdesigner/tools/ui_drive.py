"""Drives the control panel through its real widget callbacks. One bridge call performs the action; a LATER call reads
the result, because control channels and Panel Execute callbacks update on TouchDesigner's next frame.
    python ui_drive.py
"""
import time

from tdrun import run

PRE = """
rv = op('/project1/ROBOT_VISION_4D'); mc = rv.op('UI/main_controls'); panel = rv.op('UI/panel')
tl = rv.op('DATA_INPUT/timeline').module
g = lambda n: round(float(mc[n].eval()), 3)
"""
results = []


def td(code):
    out = run(PRE + code)
    d = out.get("data") or {}
    if out.get("error") or d.get("stderr"):
        return "ERROR " + str(out.get("error") or d.get("stderr"))[:200]
    return d.get("result")


def step(name, action, read, expect):
    td(action)
    time.sleep(0.4)                                    # let TouchDesigner run a few frames
    got = td(read)
    ok = got == expect
    results.append(ok)
    print(("PASS " if ok else "FAIL ") + name + " -> " + str(got) + ("" if ok else "   (expected %s)" % expect))


click = lambda k: f"panel.op('{k}').click(1, left=True)"
setv = lambda k, v: f"panel.op('{k}').click({v}, left=True)"   # explicit state: bare click() does not toggle across frames

td("tl.reset(); [panel.op(k).click(v, left=True) for k, v in (('r_camera2', 1), ('r_temporal2', 1), ('r_style2', 1), ('t_glow_on', 1), ('t_show_boxes', 1), ('t_show_grid', 1), ('t_show_trajectory', 1), ('t_show_error', 0), ('t_freeze', 0))]; panel.op('p_view3').click()")
time.sleep(0.4)
step("DEPTH ERROR toggle on", setv("t_show_error", 1), "result = [g('show_error'), rv.op('LIDAR_RENDERER/lidar_error_geometry').par.render.eval(), rv.op('LIDAR_RENDERER/lidar_geometry').par.render.eval()]", [1.0, True, False])
step("DEPTH ERROR toggle off", setv("t_show_error", 0), "result = [g('show_error'), rv.op('LIDAR_RENDERER/lidar_error_geometry').par.render.eval(), rv.op('LIDAR_RENDERER/lidar_geometry').par.render.eval()]", [0.0, False, True])
step("preset STEREO", "panel.op('p_view1').click()", "result = [g('show_stereo'), g('show_lidar'), g('show_error')]", [1.0, 0.0, 0.0])
step("preset LIDAR", "panel.op('p_view2').click()", "result = [g('show_stereo'), g('show_lidar'), g('show_error')]", [0.0, 1.0, 0.0])
step("preset BOTH + ERROR", "panel.op('p_view4').click()", "result = [g('show_stereo'), g('show_lidar'), g('show_error')]", [1.0, 1.0, 1.0])
step("preset BOTH", "panel.op('p_view3').click()", "result = [g('show_stereo'), g('show_lidar'), g('show_error')]", [1.0, 1.0, 0.0])
for i, name in enumerate(["ROBOT", "CHASE", "FREE", "BIRD", "ORBIT"], start=1):
    step(f"camera {name}", click(f"r_camera{i}"), f"result = [g('camera_mode'), [int(panel.op('r_camera%d' % k).par.value0.eval()) for k in range(1, 6)]]", [float(i), [1 if k == i else 0 for k in range(1, 6)]])
click_tmode = {1: 0.0, 2: 1.0, 3: 2.0, 4: 6.0}
for i, name in enumerate(["OFF", "ECHO", "TRAILS", "EXPLOSION"], start=1):
    step(f"temporal {name}", click(f"r_temporal{i}"), "result = g('temporal_mode')", click_tmode[i])
step("style SCIENTIFIC", click("r_style1"), "result = [g('style_mode'), rv.op('OUTPUT/glow_and_atmosphere').seq.vec[0].par.valuex.eval(), rv.op('OUTPUT/glow_and_atmosphere').seq.vec[2].par.valuex.eval()]", [0.0, 0.0, 0.0])
step("style CINEMATIC", click("r_style2"), "result = [g('style_mode'), round(rv.op('OUTPUT/glow_and_atmosphere').seq.vec[0].par.valuex.eval(), 2), rv.op('OUTPUT/glow_and_atmosphere').seq.vec[2].par.valuex.eval() > 0]", [1.0, 0.6, True])
step("glow checkbox off", setv("t_glow_on", 0), "result = rv.op('OUTPUT/glow_and_atmosphere').seq.vec[0].par.valuex.eval()", 0.0)
step("glow checkbox on", setv("t_glow_on", 1), "result = round(rv.op('OUTPUT/glow_and_atmosphere').seq.vec[0].par.valuex.eval(), 2)", 0.6)
step("BOXES off", setv("t_show_boxes", 0), "result = [g('show_boxes'), rv.op('OBJECT_TRACKING/box_geometry').par.render.eval()]", [0.0, False])
step("BOXES on", setv("t_show_boxes", 1), "result = [g('show_boxes'), rv.op('OBJECT_TRACKING/box_geometry').par.render.eval()]", [1.0, True])
step("GRID off", setv("t_show_grid", 0), "result = [g('show_grid'), rv.op('WORLD_REFERENCE/grid_geometry').par.render.eval()]", [0.0, False])
step("GRID on", setv("t_show_grid", 1), "result = [g('show_grid'), rv.op('WORLD_REFERENCE/grid_geometry').par.render.eval()]", [1.0, True])
step("TRAJECTORY off", setv("t_show_trajectory", 0), "result = [g('show_trajectory'), rv.op('WORLD_REFERENCE/trajectory_geometry').par.render.eval()]", [0.0, False])
step("TRAJECTORY on", setv("t_show_trajectory", 1), "result = [g('show_trajectory'), rv.op('WORLD_REFERENCE/trajectory_geometry').par.render.eval()]", [1.0, True])
for key, val, read in (("stereo_size", 5.5, "rv.op('STEREO_RENDERER/stereo_shader').seq.vec[0].par.valuex.eval()"),
                       ("density", 0.4, "rv.op('STEREO_RENDERER/stereo_shader').seq.vec[6].par.valuex.eval()"),
                       ("lidar_size", 6.0, "rv.op('LIDAR_RENDERER/lidar_shader').seq.vec[0].par.valuex.eval()"),
                       ("lidar_opacity", 0.5, "rv.op('LIDAR_RENDERER/lidar_shader').seq.vec[5].par.valuex.eval()"),
                       ("max_range", 25.0, "rv.op('STEREO_RENDERER/stereo_shader').seq.vec[2].par.valuex.eval()"),
                       ("good_below", 0.35, "rv.op('LIDAR_RENDERER/lidar_error_shader').seq.vec[7].par.valuex.eval()"),
                       ("large_above", 1.2, "rv.op('LIDAR_RENDERER/lidar_error_shader').seq.vec[8].par.valuex.eval()"),
                       ("history_s", 4.0, "g('history_s')"), ("trail_decay", 2.5, "g('trail_decay')"),
                       ("orbit_radius", 20.0, "g('orbit_radius')"), ("camera_speed", 12.0, "g('camera_speed')"),
                       ("glow_intensity", 1.0, "g('glow_intensity')")):
    step(f"slider {key}", f"panel.op('s_{key}').par.value0 = {val}", f"result = round(float({read}), 3)", round(val, 3))
step("slider reset", "[setattr(panel.op('s_%s' % k).par, 'value0', v) for k, v in (('stereo_size', 3.0), ('density', 1.0), ('lidar_size', 3.5), ('lidar_opacity', 1.0), ('max_range', 60.0), ('good_below', 0.2), ('large_above', 0.5), ('history_s', 3.0), ('trail_decay', 1.5), ('orbit_radius', 12.0), ('camera_speed', 6.0), ('glow_intensity', 0.6))]", "result = [g('density'), g('stereo_size')]", [1.0, 3.0])

# timeline buttons: need real elapsed frames between action and read
td("panel.op('b_reset').click()")
time.sleep(0.8)
td("panel.op('b_play').click()")
time.sleep(1.5)
t1 = td("result = [tl.state()['playing'], tl.state()['frame'], round(tl.state()['t'], 2)]")
ok = bool(t1) and t1[0] is True and 10 <= t1[1] <= 20
results.append(ok); print(("PASS " if ok else "FAIL ") + "PLAY advances with the sensor clock (~1.5 s -> ~15 frames) -> " + str(t1))
td("panel.op('b_pause').click()")
time.sleep(0.8)
a = td("result = tl.state()['frame']")
time.sleep(0.6)
b = td("result = tl.state()['frame']")
ok = a == b
results.append(ok); print(("PASS " if ok else "FAIL ") + f"PAUSE holds the frame -> {a}, {b}")
td("panel.op('b_step_fwd').click()")
time.sleep(0.8)
td("panel.op('b_step_fwd').click()")
time.sleep(0.8)
c = td("result = tl.state()['frame']")
ok = c == a + 2
results.append(ok); print(("PASS " if ok else "FAIL ") + f"STEP+ x2 -> {a} -> {c}")
td("panel.op('b_step_back').click()")
time.sleep(0.8)
d = td("result = tl.state()['frame']")
ok = d == c - 1
results.append(ok); print(("PASS " if ok else "FAIL ") + f"STEP- -> {c} -> {d}")
td("panel.op('b_reverse').click()")
time.sleep(0.8)
e = td("result = [tl.state()['direction'], tl.state()['frame']]")
ok = e is not None and e[0] == -1 and e[1] < d
results.append(ok); print(("PASS " if ok else "FAIL ") + f"REVERSE runs backwards -> {e} (from {d})")
td("panel.op('b_pause').click()")
time.sleep(0.8)
td("panel.op('t_freeze').click(1, left=True)")
time.sleep(0.8)
f = td("result = tl.state()['frozen']")
td("panel.op('b_play').click()")
time.sleep(0.8)
f2 = td("result = tl.state()['frame']")
f1 = td("result = tl.state()['frame']")
ok = f is True and f1 == f2
results.append(ok); print(("PASS " if ok else "FAIL ") + f"FREEZE TIME holds the frame while playing -> frozen={f}, frame {f2} -> {f1}")
td("panel.op('t_freeze').click(0, left=True)")
time.sleep(0.8)
td("panel.op('b_pause').click()")
time.sleep(0.8)
g2 = td("result = tl.state()['frozen']")
ok = g2 is False
results.append(ok); print(("PASS " if ok else "FAIL ") + f"FREEZE toggled off -> frozen={g2}")
td("panel.op('s_playback_speed').par.value0 = 2.0")
time.sleep(0.8)
sp = td("tl.tick(); result = tl.state()['speed']")
ok = sp == 2.0
results.append(ok); print(("PASS " if ok else "FAIL ") + f"playback speed slider -> timeline speed {sp}")
td("panel.op('s_playback_speed').par.value0 = 1.0; panel.op('b_reset').click()")
time.sleep(0.8)
r = td("result = [tl.state()['frame'], tl.state()['playing']]")
ok = r == [0, False]
results.append(ok); print(("PASS " if ok else "FAIL ") + f"RESET -> {r}")
td("[panel.op('r_camera2').click(1, left=True), panel.op('r_temporal2').click(1, left=True), panel.op('r_style2').click(1, left=True)]")
print(f"\n{sum(results)}/{len(results)} widget checks passed")
