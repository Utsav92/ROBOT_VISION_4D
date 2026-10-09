import time
rv = op('/project1/ROBOT_VISION_4D')
fl = rv.op('DATA_INPUT/frame_loader').module
tl = rv.op('DATA_INPUT/timeline').module
tl.pause()
names = (['STEREO_RENDERER/stereo_positions', 'STEREO_RENDERER/stereo_colors', 'LIDAR_RENDERER/lidar_positions',
          'LIDAR_RENDERER/lidar_attributes', 'OBJECT_TRACKING/box_geometry/box_lines', 'OBJECT_TRACKING/object_labels',
          'WORLD_REFERENCE/grid_geometry/grid_lines', 'DEPTH_DIAGNOSTICS/error_statistics'] +
         ['TEMPORAL_4D/hist_pos_%d' % k for k in (1, 4, 8)])
L = []
def timed(label, fn):
    t0 = time.perf_counter(); fn(); return (time.perf_counter() - t0) * 1000.0

for phase, f in (('cold cache', 70), ('warm cache (same frame again)', 70)):
    if phase.startswith('cold'):
        fl._cache.clear()
    tl.scrub_to_frame(f)                       # publishes the new frame number; forced cooks run via the callbacks
    rows = []
    for n in names:
        o = rv.op(n)
        rows.append('%s %.1f' % (n.split('/')[-1], timed(n, lambda: o.cook(force=True))))
    rows.append('render3d %.1f' % timed('r', lambda: rv.op('OUTPUT/render3d').cook(force=True)))
    rows.append('final %.1f' % timed('f', lambda: rv.op('OUTPUT/final_composite').cook(force=True)))
    L.append(phase + ': ' + ', '.join(rows))
# disk vs python cost of the loader alone
fl._cache.clear()
L.append('loader cold: lidar_xyz %.1f ms, lidar_dyn %.1f, stereo_xyz %.1f, stereo_rgb %.1f, boxes %.1f' % tuple(
    timed(k, lambda k=k: fl.get(k, 71)) for k in ('lidar_xyz', 'lidar_dyn', 'stereo_xyz', 'stereo_rgb', 'boxes')))
L.append('loader warm: lidar_xyz %.2f ms, stereo_xyz %.2f' % (timed('a', lambda: fl.get('lidar_xyz', 71)), timed('b', lambda: fl.get('stereo_xyz', 71))))
L.append('cache entries=%d MAX=%d' % (len(fl._cache), fl.MAX_BYTES))
result = chr(10).join(L)
