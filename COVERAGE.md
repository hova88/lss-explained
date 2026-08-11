# Coverage contract

The v4 course maps the original LSS paper, the pinned official implementation, and one fixed nuScenes sample to 17 progressive chapters. Every chapter alternates between a staged explanation and an interactive lab around the same central scene. Secondary tutorials influenced teaching order only; technical claims are anchored to the paper or source.

| # | Act / interactive chapter | Paper / implementation coverage | Evidence |
|---:|---|---|---|
| 1 | Mission · What is LSS trying to do? | Paper §1 and Fig. 4; input/output contract; three intended symmetries | PAPER + CHECKPOINT |
| 2 | Mission · Walk the whole path once | `LSS.forward`, `get_voxels`; train versus inference paths | OFFICIAL CODE |
| 3 | Inputs · What is inside a training sample? | `SegmentationData.__getitem__`; image/calibration/target tensors; LiDAR exclusion | OFFICIAL CODE + REAL SAMPLE |
| 4 | Inputs · Matrices are not magic numbers | nuScenes calibrated sensor and ego pose; quaternion, K, R and t provenance | REAL SAMPLE |
| 5 | Inputs · Transform image and coordinates together | `data.py::img_transform`; resize/crop and `post_rot/post_trans` | REAL SAMPLE + OFFICIAL CODE |
| 6 | Image · Each camera first understands its own image | EfficientNet-B0, `Up`, shared `CamEncode`, 8×22 feature grid and 105-channel split | CHECKPOINT + OFFICIAL CODE |
| 7 | Image · A pixel is not a 3D point | `create_frustum`, exact `linspace` anchors, pinhole ray and monocular ambiguity | REAL SAMPLE + OFFICIAL CODE |
| 8 | Image · Distribute semantics, do not guess one depth | Paper §3.1; `get_depth_feat`; full 41D depth and 64D context outer product | CHECKPOINT |
| 9 | Geometry · Execute `get_geometry` one operation at a time | Exact official operation order and 18 golden geometry points | REAL SAMPLE + OFFICIAL CODE |
| 10 | Geometry · Six frustums enter one ego frame | Shared encoder, per-camera K/R/t, outward optical axes, permutation-invariant sum | REAL SAMPLE + PAPER |
| 11 | BEV · Splat sparse candidates into a regular grid | Paper §3.2/§4.2; half-open bounds, floor, rank, sort, QuickCumsum and gradient grouping | OFFICIAL CODE + TEACHING |
| 12 | BEV · Geometry ends, 2D reasoning begins | `[B,C,Z,X,Y]`, z collapse, `[B,64,200,200]`, ResNet-18 `BevEncode` | CHECKPOINT + OFFICIAL CODE |
| 13 | Learn · How does the final target teach latent depth? | Vehicle box rasterization, BCEWithLogits, end-to-end gradient flow, paper/code `pos_weight` distinction | PAPER + OFFICIAL CODE |
| 14 | Learn · What exactly happens at inference and after? | logits → sigmoid → threshold; official `preds>0`; explicit absence of box decode/NMS/tracking | OFFICIAL CODE + CHECKPOINT |
| 15 | Proof · How do we know the BEV is not mirrored? | Camera/LiDAR/ego/BEV linked brushing; logits, GT, TP/FP/FN and unified screen mapping | REAL SAMPLE + CHECKPOINT |
| 16 | Proof · What do the paper experiments establish? | Paper §4–§5.5 and Tables 1–5; baselines, dropout, calibration noise, new rigs, oracle depth | PAPER + single-frame CHECKPOINT probes |
| 17 | Proof · From BEV representation to action | Paper §3.3/§5.6; cost map, Boltzmann distribution, 1K templates and limitations | TEACHING + PAPER |

## Official function mapping

| Function / class | Chapters | Browser contract |
|---|---|---|
| `SegmentationData`, `img_transform` | 3–5, 13 | input tensor ledger, image transforms and GT raster |
| `create_frustum` | 7, 9 | `feature_anchors` and `geometry_gold` |
| `CamEncode`, `get_cam_feats`, `get_depth_feat` | 6–8 | full `[6,41,8,22]` depth and `[6,64,8,22]` context |
| `get_geometry` | 5, 7, 9–10 | exact post-transform, K, cam2ego and golden-point checks |
| `voxel_pooling`, `QuickCumsum` | 11, 13 | contributor counts and pure-algorithm equivalence tests |
| `BevEncode`, `LSS.forward` | 2, 12–16 | checkpoint logits, perturbation variants and aligned diagnostics |
| `SimpleLoss`, `get_batch_iou` | 13–16 | supervision and logit-threshold semantics |

## Deliberate boundaries

- No BEVDet, BEVDepth, temporal successor, browser inference, training service, upload, or backend.
- LiDAR is a single-sweep reference/oracle visualization, never an LSS input.
- The pinned public checkpoint is vehicle semantic segmentation; 3D box decode, NMS and tracking are not invented.
- Per-frame TP/FP/FN/IoU diagnoses this sample only; dataset numbers are labeled PAPER.
- Shoot remains a deterministic equation reconstruction because the official repository has no public planning checkpoint.
