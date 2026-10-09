"""usage: python fps.py  -> measures TouchDesigner's real per-frame tick rate under the current scene load."""
import time
from tdrun import run
get = lambda: ((run("result = [op('/project1/ROBOT_VISION_4D/DATA_INPUT/timeline').module.state().get('ticks', 0)]").get('data') or {}).get('result') or [0])[0]
a, ta = get(), time.perf_counter()
time.sleep(5.0)
b, tb = get(), time.perf_counter()
print('ticks %d -> %d in %.2f s = %.1f frames/s' % (a, b, tb - ta, (b - a) / (tb - ta)))
