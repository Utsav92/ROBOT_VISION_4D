rv = op('/project1/ROBOT_VISION_4D')
bad = {}
for c in rv.findChildren(depth=4):
    try:
        e = c.errors(recurse=False)
    except Exception:
        e = ''
    if e:
        bad[c.path.replace('/project1/ROBOT_VISION_4D/', '')] = e[:300]
mc = rv.op('UI/main_controls')
info = {'channels': [ch.name for ch in mc.chans()],
        'stereo.render': rv.op('STEREO_RENDERER/stereo_geometry').par.render.eval(),
        'lidar.render': rv.op('LIDAR_RENDERER/lidar_geometry').par.render.eval(),
        'error.render': rv.op('LIDAR_RENDERER/lidar_error_geometry').par.render.eval(),
        'box.render': rv.op('OBJECT_TRACKING/box_geometry').par.render.eval(),
        'box_points': rv.op('OBJECT_TRACKING/box_geometry/box_lines').numPoints,
        'box_prims': rv.op('OBJECT_TRACKING/box_geometry/box_lines').numPrims,
        'stats_text': rv.op('DEPTH_DIAGNOSTICS/error_statistics').par.text.eval()[:200]}
result = {'errors': bad or 'none', 'info': info}
