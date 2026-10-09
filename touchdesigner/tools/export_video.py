"""Render the cinematic video through the live TouchDesigner bridge, then encode H.264 with ffmpeg.

    python export_video.py --preview 45 120 240 360 510 660 840      # render just those frames to output/videos/preview
    python export_video.py --full                                    # all 1350 frames (45 s @ 30 fps) + encode + verify

Output is limited to 1280x720 by the non-commercial TouchDesigner licence (see README).
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tdrun import run

ROOT = Path(__file__).resolve().parents[2]
EXPORT = "op('/project1/ROBOT_VISION_4D/VIDEO/video_export').module"


def td(code, timeout=600):
    out = run(code, timeout=timeout)
    d = out.get("data") or {}
    if out.get("error") or d.get("stderr"):
        raise RuntimeError(f"TouchDesigner error: {out.get('error') or d.get('stderr')}")
    return d.get("result")


def ffmpeg_exe():
    p = shutil.which("ffmpeg")
    if p:
        return p
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", type=int, nargs="*")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--chunk", type=int, default=45)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=45.0)
    a = ap.parse_args()
    vids = ROOT / "output" / "videos"
    frames_dir = vids / ("preview" if a.preview else "frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    if not a.preview:
        for old in frames_dir.glob("frame_*.png"):
            old.unlink()
    d = str(frames_dir).replace("\\", "/")

    print("begin:", td(f"result = str({EXPORT}.begin())"))
    t0 = time.perf_counter()
    try:
        if a.preview:
            for n in a.preview:
                print(td(f"result = str({EXPORT}.render_range({n}, {n + 1}, r'{d}'))"))
        else:
            total = int(round(a.seconds * a.fps))
            for s in range(0, total, a.chunk):
                e = min(s + a.chunk, total)
                print(td(f"result = {EXPORT}.render_range({s}, {e}, r'{d}')"), f"| {e}/{total}", flush=True)
    finally:
        print("end:", td(f"result = {EXPORT}.end()"))
    print(f"render wall time {time.perf_counter() - t0:.0f} s")
    if a.preview:
        return 0

    out = vids / "robot_vision_4d.mp4"
    cmd = [ffmpeg_exe(), "-y", "-framerate", str(a.fps), "-i", str(frames_dir / "frame_%04d.png"),
           "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    print("encoding:", " ".join(cmd[:8]), "...")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1500:])
        return 1
    probe = subprocess.run([shutil.which("ffprobe") or ffmpeg_exe().replace("ffmpeg", "ffprobe"), "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames,pix_fmt:format=duration,size",
                            "-of", "json", str(out)], capture_output=True, text=True)
    print(probe.stdout if probe.returncode == 0 else "(ffprobe unavailable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
