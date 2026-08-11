# Coverage contract

The v5 visual essay maps the original LSS paper, pinned official implementation and one fixed nuScenes sample to twelve narrative scenes and three interactive labs. Visual teaching order may merge adjacent source topics; no technical claim is removed.

| # | Narrative scene | Paper / implementation coverage | Evidence |
|---:|---|---|---|
| 1 | The problem LSS solves | Paper §1 and Fig. 4; input/output contract and full forward map | PAPER + CHECKPOINT |
| 2 | One nuScenes sample | `SegmentationData`; image/calibration/target packet and LiDAR exclusion | OFFICIAL CODE + REAL SAMPLE |
| 3 | The camera rig is geometry | calibrated sensor, ego pose, quaternion, K/R/t provenance and cross-time transforms | REAL SAMPLE |
| 4 | Images become features | resize/crop, post-transform, EfficientNet, shared CamEncode and 8×22 anchors | OFFICIAL CODE + CHECKPOINT |
| 5 | A pixel becomes a ray | `create_frustum`, pinhole unprojection and monocular ambiguity | OFFICIAL CODE |
| 6 | Lift into latent depth | Paper §3.1; full 41D allocation × 64D context outer product | CHECKPOINT |
| 7 | Every camera meets in ego | exact `get_geometry`, outward optical axes and camera permutation invariance | REAL SAMPLE + OFFICIAL CODE |
| 8 | Splat onto the ground | half-open bounds, floor, rank, sort, QuickCumsum and exact sum pooling | OFFICIAL CODE + TEACHING |
| 9 | Reason in bird’s-eye view | `[B,C,Z,X,Y]`, z collapse, `[B,64,200,200]` and `BevEncode` | CHECKPOINT + OFFICIAL CODE |
| 10 | Learn, infer, stop | BEV target, BCEWithLogits, gradient path, sigmoid/threshold and excluded post-processing | OFFICIAL CODE + PAPER |
| 11 | The truth lab | linked camera/LiDAR/ego/BEV evidence, GT, probability and errors | REAL SAMPLE + CHECKPOINT |
| 12 | From evidence to action | paper results, dropout, calibration noise, limitations and Shoot Eq. 2 | PAPER + TEACHING |

## Interactive lab mapping

| Lab | Scenes | Browser contract |
|---|---|---|
| Calibrated geometry | 3–7 | real images, K, feature anchor, depth allocation, camera frustums and exact ego transform |
| Linked BEV evidence | 8–11 | probability, threshold, GT, errors, LiDAR occupancy, contributors and raw-grid inspection |
| Robustness and action | 12 | cached camera-drop/yaw outputs, trajectory costs and Boltzmann probabilities |

## Deliberate boundaries

- No BEVDet, BEVDepth, temporal successor, browser inference, training service, upload or backend.
- LiDAR is a single-sweep reference visualization, never an LSS input.
- The checkpoint is vehicle semantic segmentation; box decoding, NMS and tracking are not invented.
- Per-frame TP/FP/FN/IoU diagnoses this sample only; dataset metrics are labeled PAPER.
- Shoot remains an equation reconstruction because the official repository released no planning checkpoint.
