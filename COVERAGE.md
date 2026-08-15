# Coverage contract

The v9 course maps the original LSS paper, pinned official implementation and one fixed nuScenes sample to twelve tensor-first spatial scenes and three evidence drawers. Every scene exposes one conceptual problem and one explicit input → operation → output transition.

| # | Narrative scene | Paper / implementation coverage | Evidence |
|---:|---|---|---|
| 1 | How do six images become one BEV? | motivation, exact input/output contract and the complete Lift–transform–Splat path | PAPER + CHECKPOINT |
| 2 | Images become perspective features | 1600×900 resize/crop, normalization, post-transform bookkeeping, camera folding, shared CamEncode, 8×22 anchors and the 105-channel head | OFFICIAL CODE |
| 3 | One pixel means one ray | perspective ambiguity, post-transform inversion, K⁻¹ and why depth is missing | PAPER + OFFICIAL CODE |
| 4 | DepthNet keeps 41 hypotheses | depth logits, `softmax(dim=D)`, metric bins and latent allocation | CHECKPOINT + PAPER |
| 5 | Context carries the thing itself | 105-channel split and the separate meanings of 41D depth and 64D context | OFFICIAL CODE |
| 6 | Outer product performs the Lift | unsqueeze, broadcast multiplication and `[B,6,41,8,22,64]` frustum features | OFFICIAL CODE |
| 7 | Intrinsic turns direction plus depth into XYZ | exact post-transform undo, `[du,dv,d]`, K⁻¹ and camera-coordinate axes | REAL SAMPLE + OFFICIAL CODE |
| 8 | Six cameras meet in one coordinate system | `R·p+t`, outward optical axes, candidate flattening and ego-frame fusion | PAPER + REAL SAMPLE |
| 9 | Voxel pooling rebuilds a BEV tensor | bounds, floor quantization, rank, sort, QuickCumsum, scatter and z collapse | OFFICIAL CODE |
| 10 | Collapse height, then learn spatial context | `[B,64,200,200]`, `BevEncode`, logits, BCE and the gradient path into depth | OFFICIAL CODE + PAPER |
| 11 | Read one real frame end to end | sigmoid, threshold, orientation contract and GT/LiDAR/checkpoint audit | CHECKPOINT + REAL SAMPLE |
| 12 | One sentence should survive | core recap, reported IoU/robustness, limitations and Shoot Eq. 2 | PAPER + TEACHING |

## Practice-pause mapping

| Pause | Scenes | Browser contract |
|---|---|---|
| Calibrated geometry | 2–8 | real images, feature anchor, depth/context tensors, outer product, K⁻¹, outward camera frustums and exact ego transform |
| Linked BEV evidence | 9–11 | probability, threshold, GT, errors, LiDAR occupancy, contributors and raw-grid inspection |
| Robustness and action | 12 | cached camera-drop/yaw outputs, trajectory costs and Boltzmann probabilities |

## Deliberate boundaries

- No BEVDet, BEVDepth, temporal successor, browser inference, training service, upload or backend.
- LiDAR is a single-sweep reference visualization, never an LSS input.
- The checkpoint is vehicle semantic segmentation; box decoding, NMS and tracking are not invented.
- Per-frame TP/FP/FN/IoU diagnoses this sample only; dataset metrics are labeled PAPER.
- Shoot remains an equation reconstruction because the official repository released no planning checkpoint.
