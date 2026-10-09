"""Download a SUBSET of one CODa sequence via HTTP range requests (no need to fetch the whole 17 GB zip).

    python -m preprocessing.fetch_coda_subset --sequence 0 --frames 200 --plan      # list + size, no download
    python -m preprocessing.fetch_coda_subset --sequence 0 --frames 200 --max-gb 3   # download

Source (from the devkit's download_split.py):
    https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/sequences/{seq}.zip
Picks the longest run of consecutive ANNOTATED frames that also have rectified images + a LiDAR scan,
then pulls calibrations, timestamps, poses and only those frames. Retries on timeouts; skips files already
present with the right size, so it is safe to re-run. Dataset licence: CC BY-NC-SA 4.0 (non-commercial).
"""
import argparse
import re
import sys
import time
from pathlib import Path

from remotezip import RemoteZip

BASE = "https://web.corral.tacc.utexas.edu/texasrobotics/web_CODa/sequences/{seq}.zip"
TOP_DIRS = {"2d_raw", "2d_rect", "3d_bbox", "3d_comp", "3d_raw", "3d_semantic", "calibrations",
            "metadata", "poses", "timestamps"}
FRAME_RE = re.compile(r"_(\d+)\.\w+$")


def rel_path(name):
    parts = name.split("/")
    for i, p in enumerate(parts):
        if p in TOP_DIRS:
            return "/".join(parts[i:])
    return None


def open_zip(url, tries=6, timeout=60):
    err = None
    for k in range(tries):
        try:
            return RemoteZip(url, timeout=timeout)
        except Exception as e:  # network flakiness
            err = e
            wait = min(60, 5 * 2 ** k)
            print(f"[fetch] connect failed ({type(e).__name__}); retry {k + 1}/{tries} in {wait}s")
            time.sleep(wait)
    raise ConnectionError(f"could not open {url}: {err}")


def plan(infos, seq, want_frames, include_raw=False, image_source="2d_rect", lidar_source="3d_comp"):
    seqs = str(seq)
    by_rel = {}
    for i in infos:
        r = rel_path(i.filename)
        if r and not i.is_dir():
            by_rel[r] = i

    def frames_of(prefix):
        out = {}
        for r in by_rel:
            if r.startswith(prefix):
                m = FRAME_RE.search(r)
                if m:
                    out[int(m.group(1))] = r
        return out

    cam0 = frames_of(f"{image_source}/cam0/{seqs}/")
    cam1 = frames_of(f"{image_source}/cam1/{seqs}/")
    lid = frames_of(f"{lidar_source}/os1/{seqs}/")
    box = {**frames_of("3d_bbox/os1/"), **frames_of(f"3d_bbox/os1/{seqs}/")}
    box = {f: r for f, r in box.items() if f"_{seqs}_" in r}
    common = sorted(set(cam0) & set(cam1) & set(lid))
    pool = [f for f in common if f in box] or common
    best, cur = [], []
    for f in pool:
        cur = cur + [f] if cur and f == cur[-1] + 1 else [f]
        if len(cur) > len(best):
            best = cur
    chosen = best[:want_frames]
    names = [r for r in by_rel if r.startswith(f"calibrations/{seqs}/") or r == f"timestamps/{seqs}.txt"
             or r in (f"poses/dense_global/{seqs}.txt", f"poses/dense/{seqs}.txt")
             or r.startswith("metadata/") and re.search(rf"(^|[^0-9]){seqs}([^0-9]|$)", r)]
    for f in chosen:
        names += [cam0[f], cam1[f], lid[f]] + ([box[f]] if f in box else [])
        if include_raw:
            names += [r for r in by_rel if r.startswith(f"3d_raw/os1/{seqs}/") and FRAME_RE.search(r)
                      and int(FRAME_RE.search(r).group(1)) == f]
    return chosen, [(n, by_rel[n]) for n in dict.fromkeys(names)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", type=int, default=0)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--dest", default="data/coda")
    ap.add_argument("--max-gb", type=float, default=3.0)
    ap.add_argument("--plan", action="store_true", help="only list what would be downloaded")
    ap.add_argument("--include-raw", action="store_true", help="also fetch uncompensated 3d_raw scans")
    a = ap.parse_args(argv)

    url = BASE.format(seq=a.sequence)
    with open_zip(url) as z:
        chosen, items = plan(z.infolist(), a.sequence, a.frames, a.include_raw)
        total = sum(i.compress_size for _, i in items)
        print(f"[fetch] {len(items)} files, {total / 1e9:.2f} GB compressed; frames "
              f"{chosen[0] if chosen else '-'}..{chosen[-1] if chosen else '-'} ({len(chosen)})")
        if a.plan:
            return 0
        if not chosen:
            print("[fetch] no frame with rect images + LiDAR found; check the zip layout with --plan")
            return 2
        if total / 1e9 > a.max_gb:
            print(f"[fetch] refusing: {total / 1e9:.2f} GB exceeds --max-gb {a.max_gb}")
            return 2
        dest = Path(a.dest)
        for k, (rel, info) in enumerate(items):
            out = dest / rel
            if out.exists() and out.stat().st_size == info.file_size:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(5):
                try:
                    out.write_bytes(z.read(info.filename))
                    break
                except Exception as e:
                    print(f"[fetch] {rel}: {type(e).__name__}; retry {attempt + 1}/5")
                    time.sleep(3 * (attempt + 1))
            else:
                print(f"[fetch] giving up on {rel}; re-run to resume")
                return 1
            if k % 50 == 0:
                print(f"[fetch] {k + 1}/{len(items)}")
    print(f"[fetch] done -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
