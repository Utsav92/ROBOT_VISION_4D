"""Test fixtures.

IMPORTANT: these tests exercise the MATH with synthetic geometry. They do not stand in for CODa data and
prove nothing about dataset-specific conventions (those are verified by inspect_dataset / calibration_overlay
on real frames). The calibration numbers below are the example values printed in the CODa DATA_REPORT.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CAM0_YAML = """image_width: 1224
image_height: 1024
camera_name: narrow_stereo/left
camera_matrix:
  rows: 3
  cols: 3
  data: [730.271578753826, 0, 610.90462936767, 0, 729.707285068689, 537.715474717007, 0, 0, 1]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [-0.0559502131995934, 0.123761456061624, 0.00114530935813615, -0.00367111451580028, -0.0636070725936968]
rectification_matrix:
  rows: 3
  cols: 3
  data: [0.99977534231419884, -0.015206958507487951, 0.014765273904612077, 0.015024146363298435, 0.99981006328490463, 0.012414200751121740, -0.014951251672715080, -0.012189576169272843, 0.99981391984020351]
projection_matrix:
  rows: 3
  cols: 4
  data: [730.93414758424547, 0., 606.62505340576172, 0., 0., 730.93414758424547, 531.60715866088867, 0., 0., 0., 1., 0.]
"""

# Synthetic right camera: same intrinsics, Tx = -fx * 0.2 (a 0.2 m baseline), different cx to exercise the offset.
CAM1_YAML = CAM0_YAML.replace("606.62505340576172, 0.,", "600.0, -146.186829,", 1) \
    if False else CAM0_YAML.replace(
        "data: [730.93414758424547, 0., 606.62505340576172, 0., 0., 730.93414758424547, 531.60715866088867, 0., 0., 0., 1., 0.]",
        "data: [730.93414758424547, 0., 600.0, -146.186829, 0., 730.93414758424547, 531.60715866088867, 0., 0., 0., 1., 0.]")

CAM0_TO_CAM1 = """extrinsic_matrix:
  R:
   rows: 3
   cols: 3
   data: [ 0.999607939093204, -0.00728019495660303, 0.0270363988583955,
       0.00661202516157305, 0.999672512844786, 0.024721411485882,
       -0.0272075214802657, -0.0245329538373475, 0.999328717165135 ]
  T: [ -0.197673440251558, 0.00128769601558891, 0.00253652872125049 ]
"""

OS1_TO_CAM = """extrinsic_matrix:
  rows: 4
  cols: 4
  data: [
    -0.0050614, -0.9999872, 0.0000000, 0.03,
  -0.1556502,  0.0007878, -0.9878119, -0.05,
   0.9877993, -0.0049997, -0.1556522, 0,
    0, 0, 0, 1 ]
"""


@pytest.fixture
def calib_dir(tmp_path):
    d = tmp_path / "calibrations" / "0"
    d.mkdir(parents=True)
    (d / "calib_cam0_intrinsics.yaml").write_text(CAM0_YAML)
    (d / "calib_cam1_intrinsics.yaml").write_text(CAM1_YAML)
    (d / "calib_cam0_to_cam1.yaml").write_text(CAM0_TO_CAM1)
    for n in ("cam0", "cam1", "base"):
        (d / f"calib_os1_to_{n}.yaml").write_text(OS1_TO_CAM)
    return d


@pytest.fixture
def calib(calib_dir):
    from preprocessing.calibration import load_calibration
    return load_calibration(calib_dir)


@pytest.fixture
def cfg():
    import yaml
    return yaml.safe_load(open(Path(__file__).resolve().parents[1] / "config.yaml"))
