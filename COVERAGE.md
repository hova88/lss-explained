# Coverage contract

This document maps each interactive chapter to the original LSS paper, the pinned implementation, and the on-screen evidence class.

| Chapter | Concepts | Primary source | On-screen evidence |
|---|---|---|---|
| 1. An arbitrary camera rig | three symmetries, six-camera input, ego frame | Paper §1, §3; `data.py` camera list | REAL SAMPLE calibration |
| 2. A pixel is not a point | pinhole ray, `K⁻¹`, resize/crop post-transform | Paper §3.1; `tools.py::img_transform`; `models.py::get_geometry` | REAL SAMPLE image/matrices |
| 3. Lift with latent depth | categorical depth, 64D context, outer product | Paper §3.1; `CamEncode.get_depth_feat` | CHECKPOINT-DERIVED depth/context |
| 4. Move into the ego frame | undo augmentation, unprojection, `R,t` | Paper §3.1; `LiftSplatShoot.get_geometry` | REAL SAMPLE calibration + teaching highlight |
| 5. Splat by pillar pooling | bounds, voxel index, rank, sort, QuickCumsum | Paper §3.2, §4.2; `voxel_pooling`, `QuickCumsum` | Algorithmically exact teaching probe |
| 6. Encode the BEV | tensor shapes, ResNet-18 multi-scale fusion, vehicle logits | Paper §4.1; `BevEncode` | CHECKPOINT-DERIVED raster |
| 7. What supervision teaches | segmentation loss, baselines, IoU, parameters, rate | Paper §4.1, Table 1, §5.2 | PAPER metrics |
| 8. Robustness and arbitrary rigs | camera dropout, extrinsic noise, new rigs | Paper §5.3–5.4 | PAPER claims + CHECKPOINT-DERIVED single-frame probes |
| 9. Shoot trajectories | cost-volume sampling, Boltzmann distribution, 1K templates | Paper §3.3, §5.6 | TEACHING reconstruction only |
| 10. Return to the scene | checkpoint vs GT, oracle-depth comparison, limitations | Paper Table 5, §5.5–6 | CHECKPOINT-DERIVED + annotation-derived + PAPER |

## Deliberate exclusions

- No BEVDet, BEVDepth, temporal fusion, transformer successor, or later LSS variant.
- No browser inference, training service, user upload, or backend API.
- No claimed single-frame IoU. The demo frame is qualitative; benchmark numbers are quoted only as paper results.
- No planning output is represented as an official checkpoint result.
