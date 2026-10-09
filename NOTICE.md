# Notices and attribution

## Dataset: UT Austin Campus Object Dataset (CODa)

All sensor imagery, point clouds, poses and annotations visualised by this project, and therefore every image and video in
this repository and its releases, are derived from **CODa**:

> A. Zhang, C. Eranki, C. Zhang, R. Hong, P. Kalyani, L. Kalyanaraman, A. Gamare, A. Bagad, M. Esteva, J. Biswas.
> *Towards Robust Robot 3D Perception in Urban Environments: The UT Campus Object Dataset.* 2023.
> Dataset: https://doi.org/10.18738/T8/BBOQMV - Project: https://amrl.cs.utexas.edu/coda - Devkit: https://github.com/ut-amrl/coda-devkit

CODa is released under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International** licence (CC BY-NC-SA 4.0)
plus the dataset terms published with it. Consequently:

* the rendered videos and images in this repository (`docs/images/`, the release assets) are derivative works and are shared under
  **CC BY-NC-SA 4.0**: attribute CODa and this project, **no commercial use**, share alike;
* the CODa data itself is **not** included here. Download it from the official source, or use
  `preprocessing/fetch_coda_subset.py` (which pulls a subset from the official server);
* the 3D boxes shown are CODa's human ground-truth annotations, not detections made by this project.

## Source code

No open-source licence has been chosen for the code in this repository yet. Until the author adds one, the default copyright
rules apply (viewing and forking on GitHub is permitted by GitHub's terms; reuse beyond that needs the author's permission).

## Third-party components not redistributed here

* TouchDesigner (Derivative) - a licensed application; a non-commercial licence limits output to 1280x720.
* The TouchDesigner MCP web-server component (`mcp_webserver_base.tox` plus its `modules/` folder) used to drive TouchDesigner from
  outside is not included; obtain it from the TouchDesigner MCP plugin and place it next to the tools as described in the README.
* The CODa devkit (reference only), RAFT-Stereo (optional, untested), ROS 2 (optional, untested).
