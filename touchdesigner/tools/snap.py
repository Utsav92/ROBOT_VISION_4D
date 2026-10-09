import os
name = globals().get('SNAP_NAME', 'snap')
rv = op('/project1/ROBOT_VISION_4D')
r = rv.op('OUTPUT/final_composite')
r.cook(force=True)
path = r'C:\Users\Utsav\ROBOT_VISION_4D\output\screenshots\%s.png' % name
r.save(path)
result = {'saved': path, 'res': '%dx%d' % (r.width, r.height), 'errors': r.errors() or 'none',
          'cam': [round(rv.op('CAMERA_SYSTEM/orbit_camera').par.t + 0, 2) if False else 0]}
