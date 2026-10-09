"""CODa calibration loading and projection geometry.

File formats are taken from the CODa DATA_REPORT ("Calibration Files (With Examples)"):
  calib_cam{0,1}_intrinsics.yaml : camera_matrix, distortion_coefficients, rectification_matrix,
                                   projection_matrix (3x4), image_width/height
  calib_cam0_to_cam1.yaml        : extrinsic_matrix: {R (3x3), T (3)}
  calib_os1_to_{base,cam0,cam1,vnav}.yaml : extrinsic_matrix (4x4)

Naming: T_<dst>_<src> is the 4x4 that maps points expressed in <src> to <dst>.
`calib_os1_to_cam0.yaml` is therefore T_cam0_os1.

ASSUMPTION (verify with calibration_overlay.py): the os1->cam0 extrinsic lands in the *unrectified*
cam0 frame, so the rectification rotation R_rect must be applied before the rectified projection P.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


def _load_yaml(path):
    text = Path(path).read_text()
    # OpenCV-written files begin with "%YAML:1.0" / "---" which PyYAML rejects.
    lines = [ln for ln in text.splitlines() if not ln.startswith("%") and ln.strip() != "---"]
    return yaml.safe_load("\n".join(lines))


def _mat(node):
    return np.asarray(node["data"], dtype=np.float64).reshape(int(node["rows"]), int(node["cols"]))


@dataclass
class CameraModel:
    name: str
    width: int
    height: int
    K: np.ndarray       # 3x3 raw intrinsics
    dist: np.ndarray    # distortion (k1,k2,p1,p2,k3)
    R_rect: np.ndarray  # 3x3 rectification rotation
    P: np.ndarray       # 3x4 rectified projection

    @property
    def fx(self): return float(self.P[0, 0])
    @property
    def fy(self): return float(self.P[1, 1])
    @property
    def cx(self): return float(self.P[0, 2])
    @property
    def cy(self): return float(self.P[1, 2])


def load_camera(path):
    d = _load_yaml(path)
    return CameraModel(
        name=str(d.get("camera_name", Path(path).stem)),
        width=int(d["image_width"]), height=int(d["image_height"]),
        K=_mat(d["camera_matrix"]),
        dist=np.asarray(d["distortion_coefficients"]["data"], dtype=np.float64).ravel(),
        R_rect=_mat(d["rectification_matrix"]),
        P=_mat(d["projection_matrix"]),
    )


def load_extrinsic_4x4(path):
    d = _load_yaml(path)
    T = _mat(d["extrinsic_matrix"])
    if T.shape != (4, 4):
        raise ValueError(f"{path}: expected 4x4 extrinsic, got {T.shape}")
    return T


def load_cam_to_cam(path):
    """calib_cam0_to_cam1.yaml -> 4x4 T_cam1_cam0."""
    e = _load_yaml(path)["extrinsic_matrix"]
    T = np.eye(4)
    T[:3, :3] = _mat(e["R"])
    T[:3, 3] = np.asarray(e["T"], dtype=np.float64).ravel()
    return T


def rigid_inverse(T):
    """Exact inverse of a 4x4 transform. Deliberately not the transpose shortcut: calibration files are
    rounded (the documented example is orthonormal only to ~1e-5), and transposing would not round-trip."""
    return np.linalg.inv(T)


@dataclass
class CodaCalibration:
    cam0: CameraModel
    cam1: CameraModel
    T_cam1_cam0: np.ndarray
    T_cam0_os1: np.ndarray
    T_cam1_os1: np.ndarray
    T_base_os1: np.ndarray
    apply_rect_rotation: bool = True

    # ---- stereo geometry (rectified, from the projection matrices, never assumed) -------------
    def _baseline(self):
        """P1[0,3] = -fx * baseline in standard ROS files. CODa sequence 0 stores it 1000x too small
        (P1[0,3] = -0.151 -> 0.000196 m), so cross-check against |T| of calib_cam0_to_cam1 (0.196 m)."""
        b_proj = float(-self.cam1.P[0, 3] / self.cam1.P[0, 0])
        b_ext = float(np.linalg.norm(self.T_cam1_cam0[:3, 3]))
        ratio = b_ext / b_proj if b_proj else float("inf")
        if abs(ratio - 1.0) < 0.05:
            return b_proj, "projection_matrix"
        if abs(ratio / 1000.0 - 1.0) < 0.05:
            return b_proj * 1000.0, "projection_matrix x1000 (file stores Tx scaled by 1/1000; confirmed by cam0_to_cam1)"
        raise ValueError(f"stereo baseline inconsistent: projection gives {b_proj:.6f} m, "
                         f"cam0_to_cam1 gives {b_ext:.6f} m")

    @property
    def baseline_m(self):
        return self._baseline()[0]

    @property
    def baseline_source(self):
        return self._baseline()[1]

    @property
    def disparity_offset_px(self):
        """cx_left - cx_right. Z = fx*B / (d - offset)."""
        return float(self.cam0.P[0, 2] - self.cam1.P[0, 2])

    # ---- point transforms ----------------------------------------------------------------------
    def os1_to_rect0(self, pts):
        """LiDAR-frame points (N,3) -> rectified cam0 frame."""
        p = np.asarray(pts, dtype=np.float64) @ self.T_cam0_os1[:3, :3].T + self.T_cam0_os1[:3, 3]
        if self.apply_rect_rotation:
            p = p @ self.cam0.R_rect.T
        return p

    def rect0_to_os1(self, pts):
        p = np.asarray(pts, dtype=np.float64)
        if self.apply_rect_rotation:
            p = p @ np.linalg.inv(self.cam0.R_rect).T
        Ti = rigid_inverse(self.T_cam0_os1)
        return p @ Ti[:3, :3].T + Ti[:3, 3]

    def project_rect0(self, p_rect):
        """Rectified cam0 points (N,3) -> (uv (N,2), z (N,)). Points with z<=0 get NaN uv."""
        p = np.asarray(p_rect, dtype=np.float64)
        z = p[:, 2]
        P = self.cam0.P
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (P[0, 0] * p[:, 0] + P[0, 2] * z + P[0, 3]) / z
            v = (P[1, 1] * p[:, 1] + P[1, 2] * z + P[1, 3]) / z
        uv = np.stack([u, v], axis=1)
        uv[z <= 0] = np.nan
        return uv, z


def load_calibration(calib_dir, apply_rect_rotation=True):
    d = Path(calib_dir)
    need = ["calib_cam0_intrinsics.yaml", "calib_cam1_intrinsics.yaml", "calib_cam0_to_cam1.yaml",
            "calib_os1_to_cam0.yaml", "calib_os1_to_cam1.yaml", "calib_os1_to_base.yaml"]
    missing = [n for n in need if not (d / n).exists()]
    if missing:
        raise FileNotFoundError(f"{d}: missing calibration files {missing}")
    return CodaCalibration(
        cam0=load_camera(d / "calib_cam0_intrinsics.yaml"),
        cam1=load_camera(d / "calib_cam1_intrinsics.yaml"),
        T_cam1_cam0=load_cam_to_cam(d / "calib_cam0_to_cam1.yaml"),
        T_cam0_os1=load_extrinsic_4x4(d / "calib_os1_to_cam0.yaml"),
        T_cam1_os1=load_extrinsic_4x4(d / "calib_os1_to_cam1.yaml"),
        T_base_os1=load_extrinsic_4x4(d / "calib_os1_to_base.yaml"),
        apply_rect_rotation=apply_rect_rotation,
    )
