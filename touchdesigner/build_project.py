"""ROBOT VISION 4D - TouchDesigner project builder (Stages 3-4: calibrated static scene + animated replay).

Run INSIDE TouchDesigner (Textport, or via an HTTP exec bridge):
    RV4D_DIR = r'C:\\Users\\Utsav\\ROBOT_VISION_4D'; RV4D_SEQ = 0
    exec(open(RV4D_DIR + r'\\touchdesigner\\build_project.py', encoding='utf-8').read())

Builds /project1/ROBOT_VISION_4D with: DATA_INPUT, STEREO_RENDERER, LIDAR_RENDERER, CAMERA_SYSTEM, OUTPUT.
Not built yet (later stages): OBJECT_TRACKING, DEPTH_DIAGNOSTICS UI, TEMPORAL_4D, UI, cinematic cameras.
Prints a BUILD REPORT; anything that did not apply is listed as a WARNING rather than silently skipped.
"""
import json
import math
import os

import numpy as np

PROJECT_DIR = globals().get("RV4D_DIR", r"C:\Users\Utsav\ROBOT_VISION_4D")
SEQ = globals().get("RV4D_SEQ", 0)
DATA_DIR = os.path.join(PROJECT_DIR, "output", "processed", f"seq{SEQ}")
TD_DIR = os.path.join(PROJECT_DIR, "touchdesigner")
SHADER_DIR = os.path.join(PROJECT_DIR, "shaders")
REPORT, WARN = [], []


def say(msg): REPORT.append(msg); print(msg)
def warn(msg): WARN.append(msg); print("WARNING:", msg)


# ---------------------------------------------------------------------------------------------------
def fresh(parent, optype, name):
    old = parent.op(name)
    if old is not None:
        old.destroy()
    return parent.create(optype, name)


def text_dat(parent, name, path=None, text=None):
    d = fresh(parent, textDAT, name)
    if path:
        d.par.file = path
        d.par.loadonstart = True
        d.par.syncfile = True
        d.par.loadonstartpulse.pulse()
    if text is not None:
        d.text = text
    return d


def set_vec(mat, i, name, *vals):
    """Set GLSL uniform i on the Vectors page via the sequence API, falling back to raw par names."""
    try:
        if mat.seq.vec.numBlocks <= i:
            mat.seq.vec.numBlocks = i + 1
        blk = mat.seq.vec[i]
        blk.par.name = name
        for k, v in zip("xyzw", vals):
            getattr(blk.par, "value" + k).val = v
        return blk
    except Exception:
        try:
            setattr(mat.par, f"vec{i}name", name)
            for k, v in zip("xyzw", vals):
                getattr(mat.par, f"vec{i}value{k}").val = v
        except Exception as e:
            warn(f"{mat.name}: could not set uniform {name}: {e}")


def vec_expr(mat, i, comps):
    """Bind uniform i components to python expressions."""
    for k, expr in zip("xyzw", comps):
        try:
            getattr(mat.seq.vec[i].par, "value" + k).expr = expr
        except Exception:
            try:
                getattr(mat.par, f"vec{i}value{k}").expr = expr
            except Exception as e:
                warn(f"{mat.name}: cannot bind expression {expr}: {e}")


def set_sampler(mat, i, name, top):
    try:
        if mat.seq.sampler.numBlocks <= i:
            mat.seq.sampler.numBlocks = i + 1
        blk = mat.seq.sampler[i]
        blk.par.name = name
        blk.par.top = top
    except Exception:
        try:
            setattr(mat.par, f"sampler{i}name", name)
            setattr(mat.par, f"sampler{i}top", top)
        except Exception as e:
            warn(f"{mat.name}: could not bind sampler {name}: {e}")


def make_mat(parent, name, vert, frag):
    m = fresh(parent, glslMAT, name)
    vd = text_dat(parent, name + "_vert", os.path.join(SHADER_DIR, vert))
    pd = text_dat(parent, name + "_frag", os.path.join(SHADER_DIR, frag))
    m.par.vdat = vd
    m.par.pdat = pd
    return m


def script_top(parent, name, kind, fmt):
    cb = text_dat(parent, name + "_callbacks", text=SCRIPT_TOP_CB.replace("__KIND__", kind))
    t = fresh(parent, scriptTOP, name)
    t.par.callbacks = cb
    try:
        t.par.format = fmt
    except Exception as e:
        warn(f"{name}: could not set pixel format {fmt}: {e}")
    return t


def grid_points(parent, name, cols, rows, size=4000.0):
    """Point cloud geometry: a Grid SOP converted to particles (point primitives). The Grid SOP has no 'points'
    surface type; its default rows/cols output is polylines, which the vertex shader would stretch into streaks."""
    geo = fresh(parent, geometryCOMP, name)
    for c in list(geo.children):
        c.destroy()
    g = geo.create(gridSOP, "grid1")
    g.par.rows, g.par.cols = rows, cols
    g.par.sizex = g.par.sizey = size
    g.par.texture = "rowcol"                       # generates texture coordinates 0..1 for the shader lookup
    conv = geo.create(convertSOP, "to_points")
    conv.inputConnectors[0].connect(g)
    conv.par.totype = "part"
    try:
        conv.par.prtype = PARTICLE_TYPE
    except Exception as e:
        warn(f"{name}.to_points: prtype {PARTICLE_TYPE}: {e}")
    conv.display = True
    conv.render = True
    g.display = False
    g.render = False
    return geo, conv


PARTICLE_TYPE = globals().get("RV4D_PRTYPE", "pointsprites")


SCRIPT_TOP_CB = '''
def onSetupParameters(scriptOp):
    return

def onPulse(par):
    return

def onCook(scriptOp):
    base = scriptOp.parent().parent()
    fl = base.op("DATA_INPUT/frame_loader").module
    ctl = base.op("DATA_INPUT/frame_control")
    i = int(ctl["frame"].eval()) if ctl["frame"] is not None else 0
    arr = fl.get("__KIND__", i)
    scriptOp.par.outputresolution = "custom"
    scriptOp.par.resolutionw = arr.shape[1]
    scriptOp.par.resolutionh = arr.shape[0]
    scriptOp.copyNumpyArray(arr)
    return
'''

TICK_CB = '''
def onFrameStart(frame):
    op("timeline").module.tick()
    try:
        op("/project1/ROBOT_VISION_4D/CAMERA_SYSTEM/camera_controller").module.update()
    except Exception as e:
        print("camera update error:", e)
    t = op("/project1/ROBOT_VISION_4D/OBJECT_TRACKING/object_labels")
    if t is not None:
        t.cook(force=True)
    return
'''

CHANGE_CB = '''
def onValueChange(channel, sampleIndex, val, prev):
    base = parent().parent() if False else op("/project1/ROBOT_VISION_4D")
    for p in ("STEREO_RENDERER/stereo_positions", "STEREO_RENDERER/stereo_colors",
              "LIDAR_RENDERER/lidar_positions", "LIDAR_RENDERER/lidar_attributes",
              "OBJECT_TRACKING/box_geometry/box_lines", "OBJECT_TRACKING/object_labels",
              "WORLD_REFERENCE/grid_geometry/grid_lines"):
        t = base.op(p)
        if t is not None:
            t.cook(force=True)
    for k in range(1, 65):
        h = base.op("TEMPORAL_4D/hist_pos_%d" % k)
        if h is None:
            break
        h.cook(force=True)
    # the stats Text TOP does not recook on a CHOP value change by itself: re-assign its expression to dirty it
    st = base.op("DEPTH_DIAGNOSTICS/error_statistics")
    if st is not None:
        e = st.par.text.expr
        st.par.text.expr = e
        st.cook(force=True)
    return
'''

BOX_SOP_CB = '''
def onSetupParameters(scriptOp):
    return

def onCook(scriptOp):
    base = op("/project1/ROBOT_VISION_4D")
    fl = base.op("DATA_INPUT/frame_loader").module
    bg = base.op("OBJECT_TRACKING/box_generator").module
    i = int(base.op("DATA_INPUT/frame_control")["frame"].eval())
    rng = base.op("UI/main_controls")["box_range"].eval()
    boxes = bg.select(fl.get("boxes", i)["boxes"], fl.get("pose", i)[:3, 3], rng)
    bg.fill_sop(scriptOp, boxes)
    return
'''

LABEL_TOP_CB = '''
import numpy as np

def onSetupParameters(scriptOp):
    return

def onPulse(par):
    return

def onCook(scriptOp):
    base = op("/project1/ROBOT_VISION_4D")
    fl = base.op("DATA_INPUT/frame_loader").module
    bg = base.op("OBJECT_TRACKING/box_generator").module
    r = base.op("OUTPUT/render3d")
    w, h = max(int(r.width), 16), max(int(r.height), 16)
    i = int(base.op("DATA_INPUT/frame_control")["frame"].eval())
    mc = base.op("UI/main_controls")
    scriptOp.par.outputresolution = "custom"
    scriptOp.par.resolutionw = w
    scriptOp.par.resolutionh = h
    if mc["show_labels"].eval() < 0.5 or mc["show_boxes"].eval() < 0.5:
        if scriptOp.fetch("key", None) != "off":
            scriptOp.store("key", "off")
            scriptOp.copyNumpyArray(np.zeros((h, w, 4), dtype=np.uint8))
        return
    cam = base.op("CAMERA_SYSTEM/main_camera")
    m = cam.worldTransform
    cw = [m[row, col] for row in range(4) for col in range(4)]
    key = (i, w, h, round(cam.par.fov.eval(), 3), mc["box_range"].eval(), tuple(round(x, 3) for x in cw))
    if scriptOp.fetch("key", None) == key:          # nothing moved since the last overlay: keep the texture as is
        return
    scriptOp.store("key", key)
    boxes = bg.select(fl.get("boxes", i)["boxes"], fl.get("pose", i)[:3, 3], mc["box_range"].eval())
    scriptOp.copyNumpyArray(bg.label_image(w, h, boxes, cw, cam.par.fov.eval()))
    return
'''

HIST_TOP_CB = '''
def onSetupParameters(scriptOp):
    return

def onPulse(par):
    return

def onCook(scriptOp):
    base = op("/project1/ROBOT_VISION_4D")
    fl = base.op("DATA_INPUT/frame_loader").module
    tc = base.op("TEMPORAL_4D/temporal_controller").module
    mc = base.op("UI/main_controls")
    frame = int(base.op("DATA_INPUT/frame_control")["frame"].eval())
    mode = int(round(mc["temporal_mode"].eval()))
    src = tc.source_frame(__K__, frame, mc["history_s"].eval(), mc["trail_decay"].eval())
    arr = fl.get("lidar_echo" if mode == 1 else "lidar_xyz", src)      # echo = moving points, precomputed offline
    scriptOp.par.outputresolution = "custom"
    scriptOp.par.resolutionw = arr.shape[1]
    scriptOp.par.resolutionh = arr.shape[0]
    scriptOp.copyNumpyArray(arr)
    return
'''

CONTROLS_CB = '''
def onValueChange(channel, sampleIndex, val, prev):
    if channel.name not in ("temporal_mode", "history_s", "trail_decay"):
        return
    base = op("/project1/ROBOT_VISION_4D")
    for k in range(1, 65):
        t = base.op("TEMPORAL_4D/hist_pos_%d" % k)
        if t is None:
            break
        t.cook(force=True)
    return
'''

GRID_SOP_CB = '''
def onSetupParameters(scriptOp):
    return

def onCook(scriptOp):
    base = op("/project1/ROBOT_VISION_4D")
    wr = base.op("WORLD_REFERENCE/world_reference").module
    fl = base.op("DATA_INPUT/frame_loader").module
    i = int(base.op("DATA_INPUT/frame_control")["frame"].eval())
    p = fl.get("pose", i)[:3, 3]
    wr.fill_polylines(scriptOp, wr.grid_lines((p[0], p[2]), 60.0, 5.0, float(base.fetch("ground_y", 0.0))))
    return
'''

TRAJ_SOP_CB = '''
def onSetupParameters(scriptOp):
    return

def onCook(scriptOp):
    base = op("/project1/ROBOT_VISION_4D")
    wr = base.op("WORLD_REFERENCE/world_reference").module
    traj = base.fetch("trajectory")
    wr.fill_path(scriptOp, traj + [0.0, float(base.fetch("ground_y", 0.0)) - float(traj[:, 1].mean()) + 0.05, 0.0])
    return
'''

PANEL_STATE_CB = '''
def onValueChange(panelValue, prev):
    if panelValue.owner.name == "t_freeze":
        op("/project1/ROBOT_VISION_4D/DATA_INPUT/timeline").module.freeze(bool(panelValue.val))
    return
'''

PANEL_CB = '''
BASE = "/project1/ROBOT_VISION_4D"
PRESETS = {"p_view1": (1, 0, 0), "p_view2": (0, 1, 0), "p_view3": (1, 1, 0), "p_view4": (1, 1, 1)}

def onOffToOn(panelValue):
    n = panelValue.owner.name
    base = op(BASE)
    tl = base.op("DATA_INPUT/timeline").module
    panel = base.op("UI/panel")
    if n == "b_play":
        tl.play()
    elif n == "b_pause":
        tl.pause()
    elif n == "b_reverse":
        tl.reverse()
    elif n == "b_step_back":
        tl.step(-1)
    elif n == "b_step_fwd":
        tl.step(1)
    elif n == "b_reset":
        tl.reset()
    elif n in PRESETS:
        for key, v in zip(("t_show_stereo", "t_show_lidar", "t_show_error"), PRESETS[n]):
            panel.op(key).par.value0 = v
    return

'''


# ---------------------------------------------------------------------------------------------------
def build():
    man_path = os.path.join(DATA_DIR, "manifest.json")
    if not os.path.exists(man_path):
        raise FileNotFoundError(f"{man_path} missing - run preprocessing.process_sequence first")
    manifest = json.load(open(man_path))
    T_CFG = manifest["config"]["temporal"]
    N_LAYERS = int(T_CFG["history_layers"])
    n = manifest["n_frames"]
    sx = np.load(os.path.join(DATA_DIR, "stereo_xyz", "00000.npy"), mmap_mode="r")
    lx = np.load(os.path.join(DATA_DIR, "lidar_xyz", "00000.npy"), mmap_mode="r")
    pose0 = np.load(os.path.join(DATA_DIR, "pose", "00000.npy"))
    sh, sw = sx.shape[:2]
    lh, lw = lx.shape[:2]
    say(f"data: {n} frames, stereo grid {sw}x{sh} ({sw * sh} pts), lidar grid {lw}x{lh} ({lw * lh} pts)")

    proj = op("/project1") or root.create(baseCOMP, "project1")
    rv = fresh(proj, baseCOMP, "ROBOT_VISION_4D")

    # ---- DATA_INPUT -------------------------------------------------------------------------------
    di = fresh(rv, baseCOMP, "DATA_INPUT")
    fl = text_dat(di, "frame_loader", os.path.join(TD_DIR, "frame_loader.py"))
    tl = text_dat(di, "timeline", os.path.join(TD_DIR, "timeline_controller.py"))
    ctl = fresh(di, constantCHOP, "frame_control")
    for i, nm in enumerate(("frame", "time", "playing", "speed")):
        setattr(ctl.par, f"name{i}", nm)
    di.store("data_dir", DATA_DIR)
    fl.module.set_root(DATA_DIR)
    tl.module.attach(fl.module.timestamps(), ctl)
    ex = fresh(di, executeDAT, "frame_tick")
    ex.text = TICK_CB
    ex.par.framestart = True
    ce = fresh(di, chopexecuteDAT, "frame_changed")
    ce.text = CHANGE_CB
    ce.par.chops = ctl
    ce.par.valuechange = True

    # ---- UI: control values (widgets come in the UI stage) -----------------------------------------
    ui = fresh(rv, baseCOMP, "UI")
    text_dat(ui, "ui_controller", os.path.join(TD_DIR, "ui_controller.py"))
    mc = fresh(ui, constantCHOP, "main_controls")
    controls = [("show_stereo", 1), ("show_lidar", 1), ("show_error", 0), ("glow_on", 1), ("glow_intensity", 0.6),
                ("style_mode", 1), ("density", 1.0), ("playback_speed", 1.0), ("show_boxes", 1), ("show_labels", 1), ("show_stats", 1), ("box_range", 40.0),
                ("stereo_size", 3.0), ("lidar_size", 3.5), ("lidar_opacity", 1.0), ("max_range", 60.0),
                ("good_below", 0.20), ("large_above", 0.50), ("stereo_exposure", 1.25),
                ("temporal_mode", 1), ("history_s", T_CFG["history_s"]), ("trail_decay", T_CFG["decay_exponent"]),
                ("time_scale", T_CFG["time_axis_m_per_s"]), ("trail_strength", 0.9), ("trail_size", 2.5),
                ("camera_mode", 2), ("cam_blend_s", 0.8), ("chase_back", 7.0), ("chase_up", 3.5), ("bird_height", 45.0),
                ("orbit_radius", 12.0), ("orbit_height", 5.0), ("orbit_period", 30.0), ("camera_speed", 6.0),
                ("cam_fov", 55.0), ("show_grid", 1), ("show_trajectory", 1),
                ("fade", 1.0), ("title_alpha", 0.0)]
    try:
        if mc.seq.const.numBlocks < len(controls):
            mc.seq.const.numBlocks = len(controls)
        for i, (nm, v) in enumerate(controls):
            mc.seq.const[i].par.name = nm
            mc.seq.const[i].par.value = v
    except Exception:
        for i, (nm, v) in enumerate(controls):
            setattr(mc.par, f"name{i}", nm)
            setattr(mc.par, f"value{i}", v)
    if mc["show_stereo"] is None:
        warn("main_controls: channels were not created")
    MC = "op('/project1/ROBOT_VISION_4D/UI/main_controls')['%s'].eval()"
    UIM = "op('/project1/ROBOT_VISION_4D/UI/ui_controller').module"
    # ---- UI panel: widgets are the source of truth; main_controls channels bind to them by expression ----
    panel = fresh(ui, containerCOMP, "panel")
    PW, PH = 430, 712
    panel.par.w, panel.par.h = PW, PH
    for nm, val in (("bgcolorr", 0.055), ("bgcolorg", 0.065), ("bgcolorb", 0.085), ("bgalpha", 1.0)):
        try:
            setattr(panel.par, nm, val)
        except Exception as e:
            warn(f"panel.{nm}: {e}")
    W = {}                                   # key -> widget op
    cursor = {"y": PH - 4}

    def place(op_, x, y, w, h):
        op_.par.x, op_.par.y, op_.par.w, op_.par.h = x, y, w, h

    def label(text, x, y, w, h=18, size=13, rgb=(0.62, 0.70, 0.78), name=None):
        t = panel.create(textCOMP, name or "lbl")
        if name:
            t.name = name
        try:
            t.par.text = text
            t.par.fontsizex = size
            t.par.fontcolorr, t.par.fontcolorg, t.par.fontcolorb = rgb
            t.par.alignx = "left"
        except Exception as e:
            warn(f"label '{text}': {e}")
        place(t, x, y, w, h)
        return t

    def style_button(b, on_rgb):
        for nm, val in (("bgcolorr", 0.13), ("bgcolorg", 0.15), ("bgcolorb", 0.19),
                        ("colorr", on_rgb[0]), ("colorg", on_rgb[1]), ("colorb", on_rgb[2])):
            try:
                setattr(b.par, nm, val)
            except Exception:
                pass

    def button(key, text, x, y, w, kind, default=0, on_rgb=(1.0, 0.46, 0.0)):
        b = panel.create(buttonCOMP, key)
        b.name = key
        b.par.buttontype = kind
        b.par.label = text
        place(b, x, y, w, 24)
        style_button(b, on_rgb)
        if kind != "momentary":
            b.par.value0 = default
        W[key] = b
        return b

    def header(text):
        cursor["y"] -= 18
        label(text, 12, cursor["y"], 300, 16, 12, (0.36, 0.78, 0.55))

    def row(step=26):
        cursor["y"] -= step
        return cursor["y"]

    def radio(prefix, items, default):
        y = row()
        wdt = (PW - 24) / len(items)
        keys = []
        for i, it in enumerate(items):
            k = f"{prefix}{i + 1}"
            b = button(k, it, 12 + i * wdt, y, wdt - 4, "radiodown", 1 if i + 1 == default else 0)
            b.par.buttongroup = prefix              # buttons sharing a group name are mutually exclusive
            keys.append(k)
        return keys

    label("ROBOT VISION 4D", 12, cursor["y"] - 24, 300, 24, 20, (0.95, 0.97, 1.0), name="title")
    cursor["y"] -= 26
    header("RENDER STYLE")
    style_keys = radio("r_style", ["SCIENTIFIC", "CINEMATIC"], 2)

    header("LAYERS")
    toggles = [("show_stereo", "RGB STEREO", 1), ("show_lidar", "LIDAR", 1), ("show_error", "DEPTH ERROR", 0),
               ("show_boxes", "3D BOXES", 1), ("show_labels", "LABELS", 1), ("show_grid", "GRID", 1),
               ("show_trajectory", "TRAJECTORY", 1), ("glow_on", "GLOW", 1), ("show_stats", "STATISTICS", 1)]
    for i, (key, text, dflt) in enumerate(toggles):
        if i % 3 == 0:
            ty = row()
        button("t_" + key, text, 12 + (i % 3) * 136, ty, 132, "toggledown", dflt, (0.14, 1.0, 0.38) if key != "show_lidar" else (1.0, 0.46, 0.0))

    header("VIEW PRESETS")
    ty = row()
    for i, (text) in enumerate(["STEREO", "LIDAR", "BOTH", "BOTH + ERROR"]):
        button(f"p_view{i + 1}", text, 12 + i * 102, ty, 98, "momentary")

    header("TEMPORAL 4D")
    temporal_keys = radio("r_temporal", ["OFF", "ECHO", "TRAILS", "EXPLOSION"], 2)
    header("CAMERA")
    camera_keys = radio("r_camera", ["ROBOT", "CHASE", "FREE", "BIRD", "ORBIT"], 2)

    header("TIMELINE")
    ty = row()
    for i, (key, text) in enumerate([("b_play", "PLAY"), ("b_pause", "PAUSE"), ("b_reverse", "REVERSE"), ("b_step_back", "STEP -"),
                                     ("b_step_fwd", "STEP +"), ("b_reset", "RESET")]):
        button(key, text, 12 + i * 66, ty, 62, "momentary")
    ty = row()
    button("t_freeze", "FREEZE TIME", 12, ty, 130, "toggledown", 0, (0.35, 0.8, 1.0))

    header("CONTROLS")
    sliders = [("density", "point density", 0.05, 1.0, 1.0), ("stereo_size", "stereo point size", 1.0, 8.0, 3.0),
               ("lidar_size", "LiDAR point size", 1.0, 8.0, 3.5), ("lidar_opacity", "LiDAR opacity", 0.0, 1.0, 1.0),
               ("max_range", "max visible range (m)", 5.0, 120.0, 60.0), ("good_below", "good below (m)", 0.05, 1.0, 0.20),
               ("large_above", "large above (m)", 0.1, 3.0, 0.50), ("history_s", "history (s)", 0.5, 5.0, T_CFG["history_s"]),
               ("trail_decay", "trail decay", 0.5, 4.0, T_CFG["decay_exponent"]), ("orbit_radius", "orbit radius (m)", 3.0, 40.0, 12.0),
               ("camera_speed", "camera speed (m/s)", 1.0, 30.0, 6.0), ("playback_speed", "playback speed", 0.1, 3.0, 1.0),
               ("glow_intensity", "glow intensity", 0.0, 1.5, 0.6)]
    for key, text, lo, hi, dflt in sliders:
        y = row(22)
        label(text, 12, y, 170, 20, 12, (0.72, 0.78, 0.85), name="l_" + key)
        sl = panel.create(sliderCOMP, "s_" + key)
        sl.name = "s_" + key
        place(sl, 186, y + 1, 180, 18)
        try:
            sl.par.valuerange0l, sl.par.valuerange0h = lo, hi
            sl.par.value0 = dflt
        except Exception as e:
            warn(f"slider {key}: {e}")
        W["s_" + key] = sl
        vt = label("", 372, y, 52, 20, 12, (1.0, 0.8, 0.45), name="v_" + key)
        try:
            vt.par.text.expr = f"'%.2f' % op('{sl.path}').par.value0.eval()"
        except Exception as e:
            warn(f"value label {key}: {e}")

    # bind main_controls channels to the widgets (the widgets stay the source of truth)
    cnames = [c.name for c in mc.chans()]

    def bind(ctrl, expr):
        try:
            mc.seq.const[cnames.index(ctrl)].par.value.expr = expr
        except Exception as e:
            warn(f"bind {ctrl}: {e}")

    val = lambda key: f"op('{W[key].path}').par.value0.eval()"
    for key, *_ in toggles:
        bind(key, f"float({val('t_' + key)})")
    for key, *_ in sliders:
        bind(key, f"float({val('s_' + key)})")
    group = lambda keys: f"[{', '.join(val(k) for k in keys)}]"
    bind("style_mode", f"{UIM}.radio_index({group(style_keys)}) - 1")
    bind("temporal_mode", f"[0, 1, 2, 6][{UIM}.radio_index({group(temporal_keys)}) - 1]")
    bind("camera_mode", f"{UIM}.radio_index({group(camera_keys)}, 2)")

    pex = fresh(ui, panelexecuteDAT, "panel_events")
    pex.text = PANEL_CB
    try:
        pex.par.panels = " ".join(W[k].path for k in W if k.startswith(("b_", "p_view")))
        pex.par.offtoon = True
        pex.par.valuechange = True
    except Exception as e:
        warn(f"panel_events: {e}")

    # Panel Execute watches ONE panel value: 'select' (mouse-down) for momentary buttons above; toggles need 'state'
    pst = fresh(ui, panelexecuteDAT, "panel_state_events")
    pst.text = PANEL_STATE_CB
    try:
        pst.par.panels = W["t_freeze"].path
        pst.par.panelvalue = "state"
        pst.par.valuechange = True
    except Exception as e:
        warn(f"panel_state_events: {e}")

    win = fresh(ui, windowCOMP, "control_window")
    try:
        win.par.winop = panel
        win.par.winw, win.par.winh = PW, PH
        win.par.winoffsetx, win.par.winoffsety = 936, 0
    except Exception as e:
        warn(f"control_window: {e}")
    prev = fresh(ui, opviewerTOP, "panel_preview")
    try:
        prev.par.opviewer = panel
        prev.par.outputresolution = "custom"
        prev.par.resolutionw, prev.par.resolutionh = PW, PH
    except Exception as e:
        warn(f"panel_preview: {e}")

    UIM = "op('/project1/ROBOT_VISION_4D/UI/ui_controller').module"

    # ---- STEREO_RENDERER --------------------------------------------------------------------------
    sr = fresh(rv, baseCOMP, "STEREO_RENDERER")
    script_top(sr, "stereo_positions", "stereo_xyz", "rgba32float")
    script_top(sr, "stereo_colors", "stereo_rgb", "rgba8fixed")
    smat = make_mat(sr, "stereo_shader", "stereo_points.vert", "stereo_points.frag")
    geo, _ = grid_points(sr, "stereo_geometry", sw, sh)
    geo.par.material = smat

    # ---- LIDAR_RENDERER ---------------------------------------------------------------------------
    lr = fresh(rv, baseCOMP, "LIDAR_RENDERER")
    script_top(lr, "lidar_positions", "lidar_xyz", "rgba32float")
    script_top(lr, "lidar_attributes", "lidar_attr", "rgba32float")
    lmat = make_mat(lr, "lidar_shader", "lidar_points.vert", "lidar_points.frag")
    emat = make_mat(lr, "lidar_error_shader", "lidar_points.vert", "depth_error.frag")
    lgeo, _ = grid_points(lr, "lidar_geometry", lw, lh)
    lgeo.par.material = lmat
    egeo, _ = grid_points(lr, "lidar_error_geometry", lw, lh)
    egeo.par.material = emat

    # ---- OBJECT_TRACKING: oriented 3D box wireframes + labels -------------------------------------
    ot = fresh(rv, baseCOMP, "OBJECT_TRACKING")
    text_dat(ot, "box_generator", os.path.join(TD_DIR, "box_generator.py"))
    bgeo = fresh(ot, geometryCOMP, "box_geometry")
    for c in list(bgeo.children):
        c.destroy()
    bsop_cb = text_dat(bgeo, "box_lines_callbacks", text=BOX_SOP_CB)
    bsop = bgeo.create(scriptSOP, "box_lines")
    bsop.par.callbacks = bsop_cb
    bsop.display = True
    bsop.render = True
    bmat = fresh(ot, lineMAT, "box_material")
    for nm, val in (("widthnear", 2.0), ("widthfar", 2.0), ("linenearcolorr", 0.141), ("linenearcolorg", 1.0),
                    ("linenearcolorb", 0.376), ("linenearalpha", 1.0), ("linefarcolorr", 0.141),
                    ("linefarcolorg", 1.0), ("linefarcolorb", 0.376), ("linefaralpha", 1.0)):
        try:
            setattr(bmat.par, nm, val)
        except Exception as e:
            warn(f"box_material.{nm}: {e}")
    bgeo.par.material = bmat
    script_top(ot, "object_labels", "labels", "rgba8fixed")
    ot.op("object_labels_callbacks").text = LABEL_TOP_CB

    # ---- CAMERA_SYSTEM: ONE render camera driven by five pose generators ----------------------------
    cs = fresh(rv, baseCOMP, "CAMERA_SYSTEM")
    text_dat(cs, "camera_controller", os.path.join(TD_DIR, "camera_controller.py"))
    cam = fresh(cs, cameraCOMP, "main_camera")
    try:
        cam.par.near, cam.par.far = 0.1, 400.0
    except Exception:
        pass
    kb = fresh(cs, keyboardinCHOP, "keyboard")
    kb.par.keys = "w a s d q e shift"
    ms = fresh(cs, mouseinCHOP, "mouse")
    for nm, val in (("lbuttonname", "lbutton"), ("wheel", "wheel"), ("wheelinc", 1.0)):
        try:
            setattr(ms.par, nm, val)
        except Exception as e:
            warn(f"mouse.{nm}: {e}")
    try:
        rep = json.load(open(os.path.join(PROJECT_DIR, "output", "dataset_report.json")))
        img_w = float(rep["stereo"]["size"][0])
    except Exception:
        img_w = 1224.0
    pov_fov = math.degrees(2 * math.atan(img_w / (2.0 * manifest["calibration"]["fx"])))
    rv.store("pov_fov", pov_fov)
    say(f"robot-POV fov {pov_fov:.1f} deg (from fx={manifest['calibration']['fx']:.1f}, width {img_w:.0f})")
    camx = "op('/project1/ROBOT_VISION_4D/CAMERA_SYSTEM/main_camera').par.t%s"
    scal = lambda mat, i, name, key: (set_vec(mat, i, name, 0.0), vec_expr(mat, i, [MC % key]))
    # stereo MAT: 0 size, 1 refdist, 2 maxrange, 3 campos, 4 opacity, 5 exposure
    set_vec(smat, 1, "uRefDist", 10.0)
    set_vec(smat, 3, "uCamPos", 0, 0, 0)
    vec_expr(smat, 3, [camx % "x", camx % "y", camx % "z"])
    scal(smat, 0, "uPointSize", "stereo_size")
    scal(smat, 2, "uMaxRange", "max_range")
    set_vec(smat, 4, "uOpacity", 1.0)
    scal(smat, 5, "uExposure", "stereo_exposure")
    scal(smat, 6, "uDensity", "density")
    set_sampler(smat, 0, "sPos", sr.op("stereo_positions"))
    set_sampler(smat, 1, "sColor", sr.op("stereo_colors"))
    # LiDAR MATs (shared vertex shader): 0 size .. 4 intensity, 5 opacity, 6 glow, 7 good, 8 large
    for mat in (lmat, emat):
        set_vec(mat, 1, "uRefDist", 10.0)
        set_vec(mat, 3, "uCamPos", 0, 0, 0)
        vec_expr(mat, 3, [camx % "x", camx % "y", camx % "z"])
        scal(mat, 0, "uPointSize", "lidar_size")
        scal(mat, 2, "uMaxRange", "max_range")
        set_vec(mat, 4, "uIntensityAmt", 0.0)             # 3d_comp intensity is all zero in sequence 0
        scal(mat, 5, "uOpacity", "lidar_opacity")
        set_vec(mat, 6, "uGlow", 0.5)
        scal(mat, 7, "uGoodBelow", "good_below")
        scal(mat, 8, "uLargeAbove", "large_above")
        scal(mat, 9, "uDensity", "density")
        set_sampler(mat, 0, "sLidarPos", lr.op("lidar_positions"))
        set_sampler(mat, 1, "sLidarAttr", lr.op("lidar_attributes"))

    # layer visibility from the view mode / toggles (Geometry COMP 'render' toggle)
    geo.par.render.expr = f"{UIM}.layer_flags({MC % 'show_stereo'}, {MC % 'show_lidar'}, {MC % 'show_error'})['stereo']"
    lgeo.par.render.expr = f"{UIM}.layer_flags({MC % 'show_stereo'}, {MC % 'show_lidar'}, {MC % 'show_error'})['lidar']"
    egeo.par.render.expr = f"{UIM}.layer_flags({MC % 'show_stereo'}, {MC % 'show_lidar'}, {MC % 'show_error'})['lidar_error']"
    bgeo.par.render.expr = f"{MC % 'show_boxes'} > 0.5"

    # ---- TEMPORAL_4D: world-space history layers (measured-time ages) -----------------------------
    t4 = fresh(rv, baseCOMP, "TEMPORAL_4D")
    t4.store("n_layers", N_LAYERS)
    text_dat(t4, "temporal_controller", os.path.join(TD_DIR, "temporal_controller.py"))
    TCM = "op('/project1/ROBOT_VISION_4D/TEMPORAL_4D/temporal_controller').module"
    FR = "op('/project1/ROBOT_VISION_4D/DATA_INPUT/frame_control')['frame'].eval()"
    TMODE, THIST, TDEC = MC % "temporal_mode", MC % "history_s", MC % "trail_decay"
    hist_geos = []
    for k in range(1, N_LAYERS + 1):
        script_top(t4, f"hist_pos_{k}", "lidar_xyz", "rgba32float")
        t4.op(f"hist_pos_{k}_callbacks").text = HIST_TOP_CB.replace("__K__", str(k))
        tm = make_mat(t4, f"trail_shader_{k}", "temporal_trails.vert", "temporal_trails.frag")
        tm.par.blending = True
        tm.par.srcblend = "one"
        tm.par.destblend = "omsa"
        tm.par.depthwriting = False
        set_vec(tm, 1, "uRefDist", 10.0)
        set_vec(tm, 3, "uCamPos", 0, 0, 0)
        vec_expr(tm, 3, [camx % "x", camx % "y", camx % "z"])
        scal(tm, 0, "uPointSize", "trail_size")
        scal(tm, 2, "uMaxRange", "max_range")
        set_vec(tm, 4, "uAlpha", 0.0)
        vec_expr(tm, 4, [f"{TCM}.alpha_k({k}, {FR}, {TMODE}, {THIST}, {TDEC}, {MC % 'trail_strength'})"])
        set_vec(tm, 5, "uAge01", 0.0)
        vec_expr(tm, 5, [f"{TCM}.age01_k({k}, {FR}, {THIST}, {TDEC})"])
        set_vec(tm, 6, "uTimeOffset", 0, 0, 0)
        vec_expr(tm, 6, [f"{TCM}.offset_k({k}, {FR}, {TMODE}, {THIST}, {TDEC}, {MC % 'time_scale'}, {a})" for a in range(3)])
        set_vec(tm, 7, "uGlow", 0.5)
        set_sampler(tm, 0, "sHistPos", t4.op(f"hist_pos_{k}"))
        hg, _ = grid_points(t4, f"hist_geometry_{k}", lw, lh)
        hg.par.material = tm
        hg.par.drawpriority = 5
        hg.par.render.expr = f"{TMODE} > 0.5"
        hist_geos.append(hg)
    tce = fresh(t4, chopexecuteDAT, "controls_changed")
    tce.text = CONTROLS_CB
    tce.par.chops = mc
    tce.par.valuechange = True

    # ---- WORLD_REFERENCE: coordinate grid + robot trajectory ----------------------------------------
    wrc = fresh(rv, baseCOMP, "WORLD_REFERENCE")
    wr_dat = text_dat(wrc, "world_reference", os.path.join(TD_DIR, "world_reference.py"))
    poses = np.array([np.load(os.path.join(DATA_DIR, "pose", f"{i:05d}.npy")) for i in range(n)])
    traj = poses[:, :3, 3].copy()
    lxyz0 = np.load(os.path.join(DATA_DIR, "lidar_xyz", "00000.npy"))
    ground_y = wr_dat.module.ground_height(lxyz0.reshape(-1, 4)[:, :3], lxyz0.reshape(-1, 4)[:, 3], (traj[0, 0], traj[0, 2]))
    rv.store("ground_y", ground_y)
    rv.store("trajectory", traj)
    say(f"ground y estimate {ground_y:.2f} m; trajectory {len(traj)} poses, length "
        f"{float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum()):.1f} m")
    for gname, cb, sopname, cols in (("grid_geometry", GRID_SOP_CB, "grid_lines", (0.25, 0.55, 0.35, 0.55)),
                                     ("trajectory_geometry", TRAJ_SOP_CB, "trajectory_path", (0.75, 1.0, 1.0, 1.0))):
        gg = fresh(wrc, geometryCOMP, gname)
        for c in list(gg.children):
            c.destroy()
        cbd = text_dat(gg, sopname + "_callbacks", text=cb)
        sop = gg.create(scriptSOP, sopname)
        sop.par.callbacks = cbd
        sop.display = True
        sop.render = True
        lm = fresh(wrc, lineMAT, gname.replace("geometry", "material"))
        width = 1.0 if gname == "grid_geometry" else 3.0
        for nm, val in (("widthnear", width), ("widthfar", width),
                        ("linenearcolorr", cols[0]), ("linenearcolorg", cols[1]), ("linenearcolorb", cols[2]),
                        ("linenearalpha", cols[3]), ("linefarcolorr", cols[0]), ("linefarcolorg", cols[1]),
                        ("linefarcolorb", cols[2]), ("linefaralpha", cols[3] if gname != "grid_geometry" else 0.0),
                        ("distancenear", 5.0), ("distancefar", 70.0)):
            try:
                setattr(lm.par, nm, val)
            except Exception as e:
                warn(f"{lm.name}.{nm}: {e}")
        gg.par.material = lm
        gg.par.render.expr = MC % ("show_grid" if gname == "grid_geometry" else "show_trajectory") + " > 0.5"
        ref_geos = globals().setdefault("_ref_geos", [])
        ref_geos.append(gg)

    # ---- DEPTH_DIAGNOSTICS: statistics panel ------------------------------------------------------
    dd = fresh(rv, baseCOMP, "DEPTH_DIAGNOSTICS")
    stat = fresh(dd, textTOP, "error_statistics")
    stat.par.text.expr = (f"{UIM}.panel_text(op('/project1/ROBOT_VISION_4D/DATA_INPUT/frame_control')['frame'].eval(), "
                          f"{MC % 'show_stereo'}, {MC % 'show_lidar'}, {MC % 'show_error'}, {MC % 'good_below'}, {MC % 'large_above'}, "
                          f"{MC % 'temporal_mode'}, {MC % 'history_s'}, {MC % 'style_mode'})")
    for nm, val in (("fontsizex", 15), ("alignx", "left"), ("aligny", "top"), ("bgalpha", 0.0),
                    ("fontcolorr", 0.85), ("fontcolorg", 0.95), ("fontcolorb", 0.9)):
        try:
            setattr(stat.par, nm, val)
        except Exception as e:
            warn(f"error_statistics.{nm}: {e}")
    try:
        stat.par.font = "Consolas"
    except Exception:
        pass
    stat.par.outputresolution = "custom"
    stat.par.resolutionw, stat.par.resolutionh = 1280, 720
    statlvl = fresh(dd, levelTOP, "stats_visibility")
    statlvl.inputConnectors[0].connect(stat)
    statlvl.par.opacity.expr = f"1 if {MC % 'show_stats'} > 0.5 else 0"

    # ---- OUTPUT -----------------------------------------------------------------------------------
    out = fresh(rv, baseCOMP, "OUTPUT")
    r = fresh(out, renderTOP, "render3d")
    r.par.camera = cam
    r.par.geometry = " ".join([geo.path, lgeo.path, egeo.path, bgeo.path] + [g.path for g in hist_geos]
                              + [g.path for g in globals().pop("_ref_geos", [])])
    r.par.lights = ""
    r.par.resolutionw, r.par.resolutionh = 1920, 1080
    try:
        r.par.outputresolution = "custom"
    except Exception:
        pass
    r.par.bgcolorr, r.par.bgcolorg, r.par.bgcolorb, r.par.bgcolora = 0.012, 0.016, 0.028, 1.0
    fx = fresh(out, glslTOP, "glow_and_atmosphere")
    fxd = text_dat(out, "glow_and_atmosphere_pixel", os.path.join(SHADER_DIR, "point_glow.frag"))
    try:
        fx.par.pixeldat = fxd
    except Exception as e:
        warn(f"glow_and_atmosphere: pixel DAT not set ({e})")
    fx.inputConnectors[0].connect(r)
    PP = f"{UIM}.post_params({MC % 'style_mode'}, {MC % 'glow_on'}, {MC % 'glow_intensity'})"
    for i, nm in enumerate(("uGlow", "uRadius", "uVignette", "uHaze")):
        set_vec(fx, i, nm, 0.0)
        vec_expr(fx, i, [f"{PP}['{nm}']"])
    comp = fresh(out, compositeTOP, "overlay_composite")
    comp.par.operand = "over"
    # cross-COMP TOP inputs silently fail, so pull the overlay TOPs in with Select TOPs
    sel_l = fresh(out, selectTOP, "sel_labels")
    sel_l.par.top = ot.op("object_labels").path
    sel_s = fresh(out, selectTOP, "sel_stats")
    sel_s.par.top = statlvl.path
    # Composite "over": the FIRST input is the top layer, so order is stats, labels, then the 3D render
    sel_t = fresh(out, selectTOP, "sel_title")
    sel_h = fresh(out, selectTOP, "sel_title_shadow")
    comp.inputConnectors[0].connect(sel_t)            # title text on top, then its shadow, stats, labels, the 3D render
    comp.inputConnectors[1].connect(sel_h)
    comp.inputConnectors[2].connect(sel_s)
    comp.inputConnectors[3].connect(sel_l)
    comp.inputConnectors[4].connect(fx)
    # on-screen title text (video only: title_alpha is 0 in interactive use) and fade to/from black
    ttl = fresh(out, textTOP, "title_overlay")
    for nm, val in (("fontsizex", 28), ("alignx", "left"), ("aligny", "bottom"), ("bgalpha", 0.0),
                    ("fontcolorr", 0.93), ("fontcolorg", 0.97), ("fontcolorb", 1.0)):
        try:
            setattr(ttl.par, nm, val)
        except Exception as e:
            warn(f"title_overlay.{nm}: {e}")
    try:
        ttl.par.font = "Consolas"
    except Exception:
        pass
    ttl.par.outputresolution = "custom"
    ttl.par.resolutionw, ttl.par.resolutionh = 1280, 720
    try:                                                      # margin from the bottom-left corner (fraction of frame)
        ttl.par.positionunit = "pixels"
        ttl.par.positionx, ttl.par.positiony = 44, 38
    except Exception as e:
        warn(f"title_overlay position: {e}")
    ttl.par.text = ""
    tlv = fresh(out, levelTOP, "title_visibility")
    tlv.inputConnectors[0].connect(ttl)
    tlv.par.opacity.expr = MC % "title_alpha"
    shd = fresh(out, textTOP, "title_shadow")                 # dark offset copy underneath keeps the text legible on bright points
    for nm, val in (("fontsizex", 28), ("alignx", "left"), ("aligny", "bottom"), ("bgalpha", 0.0),
                    ("fontcolorr", 0.0), ("fontcolorg", 0.0), ("fontcolorb", 0.0)):
        try:
            setattr(shd.par, nm, val)
        except Exception:
            pass
    try:
        shd.par.font = "Consolas"
        shd.par.positionunit = "pixels"
        shd.par.positionx, shd.par.positiony = 47, 35
    except Exception:
        pass
    shd.par.outputresolution = "custom"
    shd.par.resolutionw, shd.par.resolutionh = 1280, 720
    shd.par.text = ""
    shv = fresh(out, levelTOP, "title_shadow_visibility")
    shv.inputConnectors[0].connect(shd)
    shv.par.opacity.expr = MC % "title_alpha"
    fdl = fresh(out, levelTOP, "fade_to_black")
    fdl.inputConnectors[0].connect(comp)
    fdl.par.brightness1.expr = MC % "fade"
    sel_t.par.top = tlv.path
    sel_h.par.top = shv.path
    nul = fresh(out, nullTOP, "final_composite")
    nul.inputConnectors[0].connect(fdl)
    nul.viewer = True
    ow = fresh(out, windowCOMP, "output_window")             # 1280x720 preview window; open with ow.par.winopen.pulse()
    try:
        ow.par.winop = nul
        ow.par.winw, ow.par.winh = 930, 523                    # fits beside the 430 px control panel on a 1366x768 screen
        ow.par.winoffsetx, ow.par.winoffsety = 0, 0
    except Exception as e:
        warn(f"output_window: {e}")

    # ---- VIDEO: scene script + frame-by-frame renderer, and a (not used for encoding) Movie File Out TOP ------
    vc = fresh(rv, baseCOMP, "VIDEO")
    text_dat(vc, "video_director", os.path.join(TD_DIR, "video_director.py"))
    text_dat(vc, "video_export", os.path.join(TD_DIR, "video_export.py"))
    mo = fresh(out, moviefileoutTOP, "movie_output")
    try:
        mo.inputConnectors[0].connect(nul)
        mo.par.file = os.path.join(PROJECT_DIR, "output", "videos", "realtime_capture.mp4")
        mo.par.record = False
    except Exception as e:
        warn(f"movie_output: {e}")

    # ---- verification -----------------------------------------------------------------------------
    for opn in (smat, lmat, emat, bmat, bsop, ot.op("object_labels"), stat):
        errs = opn.errors(recurse=False) if hasattr(opn, "errors") else ""
        if errs:
            warn(f"{opn.path} errors: {errs}")
    try:
        cs.op("camera_controller").module.update()
    except Exception as e:
        warn(f"initial camera update: {e}")
    say(f"BUILD OK: {rv.path}  ({len(WARN)} warnings)")
    return rv

build()
