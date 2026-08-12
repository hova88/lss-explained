# Coverage contract

The v8 course maps the original LSS paper, pinned official implementation and one fixed nuScenes sample to twelve full-screen geometry scenes and three evidence drawers. One rotatable, zoomable spatial-ink renderer carries the entire explanation; each scene changes one spatial fact while preserving the shared coordinate world.

| # | Narrative scene | Paper / implementation coverage | Evidence |
|---:|---|---|---|
| 1 | See the whole machine first | motivation, exact input/output contract, Lift–Splat–Shoot overview and LiDAR boundary | PAPER + CHECKPOINT |
| 2 | A pixel is not a place | perspective ambiguity, why image stitching fails and why depth stays latent | PAPER |
| 3 | One sample, four different roles | model input, calibration, raster target and reference-only LiDAR | REAL SAMPLE + OFFICIAL CODE |
| 4 | Compress the image, keep its geometry | resize/crop, post-transform, shared CamEncode, 8×22 anchors and 105-channel split | OFFICIAL CODE + CHECKPOINT |
| 5 | Undo the image, open a ray | raw/network pixels, K⁻¹, camera axes, 41 metric depth candidates | OFFICIAL CODE |
| 6 | Lift without choosing one depth | 41D softmax allocation, 64D context, outer product and latent-depth semantics | CHECKPOINT + PAPER |
| 7 | The coordinate chain, in order | quaternion provenance, homogeneous transforms and exact `get_geometry` order | REAL SAMPLE + OFFICIAL CODE |
| 8 | Six cameras meet in ego | real outward axes/frustums, permutation invariance and ego equivariance | PAPER + REAL SAMPLE |
| 9 | Many points become one BEV grid | bounds, floor quantization, rank, sort, QuickCumsum and z collapse | OFFICIAL CODE |
| 10 | Supervise the final map | `[B,64,200,200]`, `BevEncode`, BCE and the gradient path into latent depth | OFFICIAL CODE + PAPER |
| 11 | A BEV heatmap is not a camera image | logits, sigmoid, threshold, excluded post-processing and linked truth audit | CHECKPOINT + REAL SAMPLE |
| 12 | What LSS proves—and what it does not | reported baselines/IoU, dropout, noise, limitations and Shoot Eq. 2 | PAPER + TEACHING |

## Practice-pause mapping

| Pause | Scenes | Browser contract |
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
