# Coverage contract

The 15-chapter bilingual experience maps the original LSS paper and pinned official implementation to explicit evidence classes.

| # | Interactive chapter | Paper / implementation coverage | Evidence |
|---:|---|---|---|
| 1 | Why BEV | Paper §1 and §3; three intended symmetries; Lift–Splat–Shoot decomposition | PAPER |
| 2 | Frame ledger | Camera, LiDAR, ego, global, network-image, feature-grid and tensor conventions | REAL SAMPLE |
| 3 | Where matrices come from | nuScenes calibrated sensor and ego pose; quaternion → rotation → homogeneous transform | REAL SAMPLE |
| 4 | How transforms compose | Active column-vector transforms, inverse rigid transform, multiplication order | REAL SAMPLE |
| 5 | Image preprocessing | `data.py::img_transform`; resize/crop and `post_rot/post_trans` | REAL SAMPLE |
| 6 | Pixel to ray | Pinhole projection, `K⁻¹`, camera axes, monocular ambiguity | REAL SAMPLE |
| 7 | Feature-grid correspondence | `models.py::create_frustum`; exact 8×22 `linspace` anchors | CHECKPOINT + SOURCE |
| 8 | Lift | Paper §3.1; `CamEncode.get_depth_feat`; 41D α and 64D context outer product | CHECKPOINT |
| 9 | Exact `get_geometry` | Official operation order and 18 exported golden geometry points | REAL SAMPLE + SOURCE |
| 10 | Six cameras into ego | Shared encoder, per-camera K/R/t, permutation-invariant sum | REAL SAMPLE + PAPER |
| 11 | Splat and QuickCumsum | Paper §3.2/§4.2; bounds, rank, sort, prefix sum, backward grouping | SOURCE + TEACHING |
| 12 | BEV tensor and encoder | `[B,C,Z,X,Y]`, z collapse, `BevEncode`, ResNet-18 fusion | CHECKPOINT + SOURCE |
| 13 | Real-sample truth lab | Camera/LiDAR/ego/BEV linked brushing; logits, GT, TP/FP/FN | REAL SAMPLE + CHECKPOINT |
| 14 | Training, results and robustness | Paper §4–§5.5, Tables 1–5, dropout, noise, new rigs, oracle depth | PAPER + single-frame CHECKPOINT probes |
| 15 | Shoot and limitations | Paper §3.3/§5.6; cost map, Boltzmann distribution, 1K templates, limitations | TEACHING + PAPER |

## Official function mapping

| Function | Chapters | Browser contract |
|---|---|---|
| `create_frustum` | 7, 9 | `feature_anchors` and `geometry_gold` |
| `get_cam_feats` / `CamEncode.get_depth_feat` | 8 | full `[6,41,8,22]` depth and `[6,64,8,22]` context |
| `get_geometry` | 5–10 | exact post-transform, K, cam2ego and golden-point checks |
| `voxel_pooling` / `QuickCumsum` | 11 | contributor counts and pure-algorithm equivalence tests |
| `BevEncode` | 12–14 | checkpoint logits, ten perturbation variants and aligned diagnostics |

## Deliberate boundaries

- No BEVDet, BEVDepth, temporal successor, browser inference, training service, upload, or backend.
- LiDAR is a single-sweep reference/oracle visualization, never an LSS input.
- Per-frame TP/FP/FN/IoU is a diagnostic for this sample only; dataset numbers are labeled PAPER.
- Shoot remains a deterministic equation reconstruction because the official repository has no planning checkpoint.
