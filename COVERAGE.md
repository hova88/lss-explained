# Coverage contract

The v6 course maps the original LSS paper, pinned official implementation and one fixed nuScenes sample to twelve content-first scenes and three optional laboratories. Guided scenes use spatial-whiteboard drawings; calibrated WebGL appears only when a reader explicitly opens a lab.

| # | Narrative scene | Paper / implementation coverage | Evidence |
|---:|---|---|---|
| 1 | First, see the whole machine | motivation, exact input/output contract, Lift–Splat–Shoot overview and LiDAR boundary | PAPER + CHECKPOINT |
| 2 | Perspective is not a common ground | perspective ambiguity, why image stitching fails and why depth stays latent | PAPER |
| 3 | One sample, four different roles | model input, calibration, raster target and reference-only LiDAR | REAL SAMPLE + OFFICIAL CODE |
| 4 | Preserve geometry while compressing appearance | resize/crop, post-transform, shared CamEncode, 8×22 anchors and 105-channel split | OFFICIAL CODE + CHECKPOINT |
| 5 | Undo the image, then open a ray | raw/network pixels, K⁻¹, camera axes, 41 metric depth candidates | OFFICIAL CODE |
| 6 | Lift without a depth label | 41D softmax allocation, 64D context, outer product and latent-depth semantics | CHECKPOINT + PAPER |
| 7 | The complete coordinate chain | quaternion provenance, homogeneous transforms and exact `get_geometry` order | REAL SAMPLE + OFFICIAL CODE |
| 8 | Six cameras become one unordered set | real outward axes/frustums, permutation invariance and ego equivariance | PAPER + REAL SAMPLE |
| 9 | Turn irregular 3D evidence into BEV cells | bounds, floor quantization, rank, sort, QuickCumsum and z collapse | OFFICIAL CODE |
| 10 | Geometry places evidence; learning interprets it | `[B,64,200,200]`, `BevEncode`, BCE and the gradient path into latent depth | OFFICIAL CODE + PAPER |
| 11 | Read the output in the right coordinate system | logits, sigmoid, threshold, excluded post-processing and linked truth audit | CHECKPOINT + REAL SAMPLE |
| 12 | Results, robustness, Shoot—and the boundary | reported baselines/IoU, dropout, noise, limitations and Shoot Eq. 2 | PAPER + TEACHING |

## Interactive lab mapping

| Lab | Scenes | Browser contract |
|---|---|---|
| Calibrated geometry | 3–8 | real images, K, feature anchor, depth allocation, outward camera frustums and exact ego transform |
| Linked BEV evidence | 9–11 | probability, threshold, GT, errors, LiDAR occupancy, contributors and raw-grid inspection |
| Robustness and action | 12 | cached camera-drop/yaw outputs, trajectory costs and Boltzmann probabilities |

## Deliberate boundaries

- No BEVDet, BEVDepth, temporal successor, browser inference, training service, upload or backend.
- LiDAR is a single-sweep reference visualization, never an LSS input.
- The checkpoint is vehicle semantic segmentation; box decoding, NMS and tracking are not invented.
- Per-frame TP/FP/FN/IoU diagnoses this sample only; dataset metrics are labeled PAPER.
- Shoot remains an equation reconstruction because the official repository released no planning checkpoint.
