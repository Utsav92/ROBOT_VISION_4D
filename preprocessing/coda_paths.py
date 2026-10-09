"""Path resolution for a CODa tree, following the DATA_REPORT layout.

The report's tree lists 3d_bbox/os1/ without a sequence folder while the metadata example uses
3d_bbox/os1/{seq}/, so annotation lookup accepts both. Nothing is assumed to exist: callers get None.
"""
import re
from pathlib import Path


class CodaPaths:
    def __init__(self, root, sequence, image_source="2d_rect", lidar_source="3d_comp", pose_source="dense_global"):
        self.root = Path(root)
        self.seq = int(sequence)
        self.image_source = image_source
        self.lidar_source = lidar_source
        self.pose_source = pose_source

    @property
    def calib_dir(self): return self.root / "calibrations" / str(self.seq)
    @property
    def timestamps_file(self): return self.root / "timestamps" / f"{self.seq}.txt"

    def pose_file(self):
        for kind in (self.pose_source, "dense"):
            p = self.root / "poses" / kind / f"{self.seq}.txt"
            if p.exists():
                return p
        return self.root / "poses" / self.pose_source / f"{self.seq}.txt"

    def image(self, cam, frame):
        """The DATA_REPORT says rectified images are .jpg; the downloaded sequence 0 has .png. Accept both."""
        d = self.root / self.image_source / cam / str(self.seq)
        stem = f"{self.image_source}_{cam}_{self.seq}_{frame}"
        for ext in ("png", "jpg"):
            if (d / f"{stem}.{ext}").exists():
                return d / f"{stem}.{ext}"
        return d / f"{stem}.png"

    def lidar(self, frame):
        return self.root / self.lidar_source / "os1" / str(self.seq) / f"{self.lidar_source}_os1_{self.seq}_{frame}.bin"

    def bbox(self, frame):
        name = f"3d_bbox_os1_{self.seq}_{frame}.json"
        for p in (self.root / "3d_bbox" / "os1" / str(self.seq) / name, self.root / "3d_bbox" / "os1" / name):
            if p.exists():
                return p
        return None

    def annotated_frames(self):
        out = set()
        for d in (self.root / "3d_bbox" / "os1" / str(self.seq), self.root / "3d_bbox" / "os1"):
            if d.is_dir():
                for f in d.glob(f"3d_bbox_os1_{self.seq}_*.json"):
                    m = re.search(r"_(\d+)\.json$", f.name)
                    if m:
                        out.add(int(m.group(1)))
        return sorted(out)

    def available_frames(self):
        """Frames for which BOTH rectified images and a LiDAR scan exist."""
        def nums(d, prefix, ext):
            if not d.is_dir():
                return set()
            return {int(m.group(1)) for f in d.iterdir()
                    if (m := re.match(rf"{re.escape(prefix)}_{self.seq}_(\d+)\.{ext}$", f.name))}
        s = self.image_source
        a = nums(self.root / s / "cam0" / str(self.seq), f"{s}_cam0", "(?:png|jpg)")
        b = nums(self.root / s / "cam1" / str(self.seq), f"{s}_cam1", "(?:png|jpg)")
        c = nums(self.root / self.lidar_source / "os1" / str(self.seq), f"{self.lidar_source}_os1", "bin")
        return sorted(a & b & c)


def longest_consecutive_run(frames, want):
    """Pick `want` consecutive integer frames from the longest run available (or the whole run if shorter)."""
    frames = sorted(frames)
    best, cur = [], []
    for f in frames:
        cur = cur + [f] if cur and f == cur[-1] + 1 else [f]
        if len(cur) > len(best):
            best = cur
    return best[:want]
