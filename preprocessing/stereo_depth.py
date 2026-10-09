"""Stereo depth from a rectified pair.

Backends:
  sgbm : OpenCV StereoSGBM (default, CPU, tested on synthetic pairs).
  raft : RAFT-Stereo (PyTorch). UNTESTED here: this machine has no CUDA GPU and no checkpoint.
         Falls back to sgbm automatically when torch/CUDA/checkpoint/repo is missing.

Geometry for a rectified pair with projection offsets (disparity d = u_left - u_right):
    Z = fx * B / (d - (cx_left - cx_right))
    X = (u - cx) * Z / fx          Y = (v - cy) * Z / fy
"""
import os
import sys

import cv2
import numpy as np


def sgbm_disparity(left_bgr, right_bgr, cfg):
    """-> (disparity float32 in px, valid bool). Invalid = failed left-right consistency / no match."""
    nd = int(np.ceil(cfg.get("num_disparities", 160) / 16) * 16)
    bs = int(cfg.get("block_size", 5)) | 1
    gl = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY) if left_bgr.ndim == 3 else left_bgr
    gr = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY) if right_bgr.ndim == 3 else right_bgr
    m = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=nd, blockSize=bs,
        P1=8 * bs * bs, P2=32 * bs * bs,
        disp12MaxDiff=int(cfg.get("disp12_max_diff", 1)),
        uniquenessRatio=int(cfg.get("uniqueness_ratio", 10)),
        speckleWindowSize=100, speckleRange=2, preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = m.compute(gl, gr).astype(np.float32) / 16.0
    return disp, disp > 0.0


def smooth_disparity(disp, valid, sigma_disp=1.0, d=5, min_support=0.3):
    """Masked bilateral smoothing of disparity: edge-preserving in disparity, ignores invalid pixels."""
    v = valid.astype(np.float32)
    num = cv2.bilateralFilter(disp * v, d, sigma_disp, 3.0)
    den = cv2.bilateralFilter(v, d, sigma_disp, 3.0)
    out = np.where(den > min_support, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)
    return out, valid & (den > min_support)


def disparity_to_depth(disp, valid, fx, baseline_m, disparity_offset_px, min_depth, max_depth):
    """-> (depth float32, valid bool). Rejects non-positive effective disparity and out-of-range depth."""
    d_eff = disp - disparity_offset_px
    ok = valid & (d_eff > 1e-3)
    depth = np.zeros_like(disp, dtype=np.float32)
    depth[ok] = fx * baseline_m / d_eff[ok]
    ok &= (depth >= min_depth) & (depth <= max_depth)
    depth[~ok] = 0.0
    return depth, ok


def depth_to_rect_points(depth, valid, P, stride=1):
    """Back-project every `stride`-th pixel -> (pos (h,w,3) in rect cam0 frame, ok (h,w), (v_idx, u_idx))."""
    H, W = depth.shape
    vs = np.arange(0, H, stride)
    us = np.arange(0, W, stride)
    u, v = np.meshgrid(us, vs)
    z = depth[np.ix_(vs, us)].astype(np.float64)
    ok = valid[np.ix_(vs, us)]
    x = (u - P[0, 2]) * z / P[0, 0]
    y = (v - P[1, 2]) * z / P[1, 1]
    return np.stack([x, y, z], axis=-1), ok, (vs, us)


class RaftStereoBackend:
    """Thin wrapper over the public RAFT-Stereo code (princeton-vl/RAFT-Stereo). UNTESTED here."""

    def __init__(self, repo, checkpoint, iters=32):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(checkpoint)
        sys.path.insert(0, os.path.join(repo, "core"))
        from argparse import Namespace
        from raft_stereo import RAFTStereo
        args = Namespace(hidden_dims=[128] * 3, corr_implementation="reg", shared_backbone=False,
                         corr_levels=4, corr_radius=4, n_downsample=2, context_norm="batch",
                         slow_fast_gru=False, n_gru_layers=3, mixed_precision=False)
        model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
        model.load_state_dict(torch.load(checkpoint, map_location="cuda"))
        self.model = model.module.cuda().eval()
        self.iters = iters
        self.torch = torch

    def disparity(self, left_bgr, right_bgr):
        t = self.torch

        def prep(im):
            x = t.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None].cuda()
            pad_h, pad_w = (-x.shape[2]) % 32, (-x.shape[3]) % 32
            return t.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="replicate"), pad_h, pad_w
        l, ph, pw = prep(left_bgr)
        r, _, _ = prep(right_bgr)
        with t.no_grad():
            _, flow = self.model(l, r, iters=self.iters, test_mode=True)
        disp = -flow[0, 0].cpu().numpy()
        H, W = left_bgr.shape[:2]
        disp = disp[:H, :W].astype(np.float32)
        return disp, disp > 0.0


def make_disparity_fn(cfg, log=print):
    """Return fn(left_bgr, right_bgr)->(disp, valid) honouring cfg['backend'] with graceful fallback."""
    if cfg.get("backend", "sgbm") == "raft":
        r = cfg.get("raft", {})
        try:
            be = RaftStereoBackend(r["repo"], r["checkpoint"], r.get("iters", 32))
            log("[stereo] RAFT-Stereo backend active")
            return be.disparity
        except Exception as e:  # missing torch/CUDA/checkpoint/repo
            log(f"[stereo] RAFT-Stereo unavailable ({type(e).__name__}: {e}); falling back to StereoSGBM")
    return lambda l, r: sgbm_disparity(l, r, cfg)
