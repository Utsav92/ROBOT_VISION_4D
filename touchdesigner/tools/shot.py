"""usage: python shot.py name frame '{"camera_mode": 4, "view_mode": 2}' ['{"fwd": 1}']  -> output/screenshots/name.png"""
import json
import subprocess
import sys
from pathlib import Path

here = Path(__file__).parent
name, frame, settings = sys.argv[1], int(sys.argv[2]), json.loads(sys.argv[3])
inject = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None
shot = {"name": name, "frame": frame, "settings": settings, "inject": inject}
code = "SHOT = " + repr(shot) + "\n" + (here / "td_shot.py").read_text(encoding="utf-8")
tmp = here / "_shot_run.py"
tmp.write_text(code, encoding="utf-8")
print(subprocess.run([sys.executable, str(here / "tdrun.py"), str(tmp)], capture_output=True, text=True).stdout)
tmp.unlink()
