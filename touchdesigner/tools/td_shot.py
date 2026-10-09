# Runs inside TD. Expects SHOT (dict) prepended by shot.py: name, frame, settings (control channel -> value).
import time
rv = op('/project1/ROBOT_VISION_4D')
mc = rv.op('UI/main_controls')
names = [c.name for c in mc.chans()]
settings = dict(SHOT.get('settings', {}))
settings.setdefault('cam_blend_s', 0.001)            # shots want the final pose, not a mid-transition frame
panel = rv.op('UI/panel')
RADIO = {'camera_mode': ('r_camera', lambda v: int(v)), 'style_mode': ('r_style', lambda v: int(v) + 1),
         'temporal_mode': ('r_temporal', lambda v: {0: 1, 1: 2, 2: 3, 6: 4}[int(v)])}
for k, v in settings.items():                         # widgets are the source of truth for bound controls
    if k in RADIO:
        prefix, f = RADIO[k]
        panel.op('%s%d' % (prefix, f(v))).click(1, left=True)
    elif panel.op('t_' + k) is not None:
        panel.op('t_' + k).par.value0 = 1 if float(v) > 0.5 else 0
    elif panel.op('s_' + k) is not None:
        panel.op('s_' + k).par.value0 = v
    else:
        mc.seq.const[names.index(k)].par.value = v
tl = rv.op('DATA_INPUT/timeline').module
tl.scrub_to_frame(SHOT.get('frame', 0))
cc = rv.op('CAMERA_SYSTEM/camera_controller').module
if SHOT.get('inject'):
    cc.inject(**SHOT['inject'])
cc.update()
time.sleep(0.06)
cc.update()
for k in range(1, 65):
    t = rv.op('TEMPORAL_4D/hist_pos_%d' % k)
    if t is None:
        break
    t.cook(force=True)
r = rv.op('OUTPUT/final_composite')
rv.op('OUTPUT/render3d').cook(force=True)
r.cook(force=True)
path = r'C:\Users\Utsav\ROBOT_VISION_4D\output\screenshots\%s.png' % SHOT['name']
r.save(path)
cam = rv.op('CAMERA_SYSTEM/main_camera')
result = 'saved %s | mode=%s frame=%d cam t=(%.1f, %.1f, %.1f) fov=%.1f' % (
    path, mc['camera_mode'].eval(), SHOT.get('frame', 0), cam.par.tx, cam.par.ty, cam.par.tz, cam.par.fov)
