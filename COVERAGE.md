# Coverage contract

The v10 course maps the original LSS paper, the BEVDepth paper, both official implementations and one fixed nuScenes sample to ten advanced tensor-first scenes. Every scene exposes one conceptual problem and one explicit input → operation → output transition.

| # | Narrative scene | Paper / implementation coverage | Evidence |
|---:|---|---|---|
| 1 | The view must change, not just the features | motivation, input/output topology and the LSS→BEVDepth research question | LSS + BEVDEPTH PAPERS |
| 2 | Depth is a distribution over locations | 41-bin softmax, one-hot/uniform/multimodal meanings and latent-vs-explicit supervision | LSS PAPER + CHECKPOINT |
| 3 | Context is the payload, not the position | 105-channel split, 64D context and BEVDepth's 27D camera-aware conditioning | LSS + BEVDEPTH CODE |
| 4 | Lift is broadcast multiplication | unsqueeze, broadcast multiplication and `[B,6,41,8,22,64]` frustum features | LSS CODE + BEVDepth Eq. 1 |
| 5 | Undo the image before inverting the camera | optical center, network/raw image planes, inverse augmentation, K⁻¹ and direction ratios | LSS CODE |
| 6 | Depth scales the ray into camera meters | `[du,dv,d]`, pinhole unprojection and camera-axis convention | LSS CODE |
| 7 | Extrinsics make six camera tensors commensurable | real optical centers/outward axes, `R·p+t`, flattening and camera-aware contrast | REAL SAMPLE + BOTH CODEBASES |
| 8 | Pooling decides what a BEV cell remembers | floor/rank/sort/QuickCumsum; interactive sum/mean/max/bilinear semantics; efficient voxel pooling | LSS + BEVDEPTH CODE |
| 9 | The decisive difference is the depth gradient | BevEncode, indirect LSS task gradient, LiDAR projection, min pooling, binning, one-hot, masked BCE and `Ldet+3Ldepth` | LSS + BEVDEPTH CODE |
| 10 | Audit outputs in their own coordinate system | LSS sigmoid/threshold, orientation contract, GT/LiDAR audit and BEVDepth detection-head boundary | CHECKPOINT + REAL SAMPLE + BEVDEPTH PAPER |

## Practice-pause mapping

| Pause | Scenes | Browser contract |
|---|---|---|
| Calibrated geometry | 2–7 | real image, feature anchor, depth/context tensors, image plane, K⁻¹, camera XYZ, real outward frustums and exact ego transform |
| Pooling comparison | 8 | live sum/mean/max/bilinear collision rule and a candidate crossing a cell boundary |
| Linked BEV evidence | 10 | probability, threshold, GT, errors, LiDAR occupancy, contributors and raw-grid inspection |

## Deliberate boundaries

- BEVDepth coverage is limited to the paper's direct conceptual extensions of the LSS view transformer; no BEVDet/BEVStereo/temporal expansion, browser inference, training service, upload or backend.
- LiDAR is a single-sweep reference visualization, never an LSS input.
- LiDAR is described as BEVDepth training supervision only where explicitly labeled; BEVDepth inference remains camera-only.
- The checkpoint is vehicle semantic segmentation; box decoding, NMS and tracking are not invented.
- Per-frame TP/FP/FN/IoU diagnoses this sample only; dataset metrics are labeled PAPER.
- The public ten-scene lesson ends at view transformation, task supervision, inference and truth auditing. LSS planning/Shoot is outside this edition's teaching path rather than compressed into an unrelated final control.
