# LSS Explained — from pixels to BEV

An English-only, source-audited advanced visual lesson about **Lift, Splat, Shoot** and the depth-learning correction introduced by **BEVDepth**. Eight restrained spatial-ink scenes first establish the global camera-to-BEV contract, then follow one tensor from a real image pixel to metric BEV.

**Live:** https://hova88.github.io/lss-explained/

This is the camera-to-BEV companion to [pointpillars-explained](https://github.com/hova88/pointpillars-explained). Version 12 opens with the complete input → Lift → geometry → Splat → BEV output contract. It then lets the reader click one real nuScenes pixel, inspect its checkpoint depth distribution and context vector, and preserve that sample identity through the full image→ray→camera→ego chain, collision reductions, and the different gradient paths in LSS and BEVDepth. The pinned 34,688-point LiDAR scan is reference evidence for LSS and a visual reconstruction of BEVDepth's training-time depth-target process; it is never passed into the pinned LSS checkpoint.

## Course path

```text
N images + K,R,t
→ resize / crop + post-transform
→ shared CamEncode
→ 41-bin latent depth × 64D context
→ exact get_geometry
→ pillar sum + QuickCumsum
→ [B,64,200,200] BEV tensor
→ BevEncode + vehicle logits
→ LSS: final BEV BCE only
→ BEVDepth: detection loss + 3 × sparse depth BCE
→ camera-only inference
```

The public checkpoint audited by this site is the official LSS vehicle-segmentation model. BEVDepth discussion is grounded in its paper and official code; the site does not present LSS mask results as BEVDepth detection results.

## Evidence boundary

- **REAL SAMPLE** — calibration, images, a reference LiDAR scan and annotations from sample `ca9a282c9e77460f8360f564131a8af5`.
- **CHECKPOINT** — tensors and rasters exported by strictly loading the pinned official checkpoint.
- **LSS PAPER / LSS CODE** — claims and behavior of the original ECCV 2020 system.
- **BEVDEPTH PAPER / BEVDEPTH CODE** — its explicit depth supervision, camera-aware DepthNet and efficient pooling; no BEVDepth checkpoint output is fabricated.
- **TEACHING** — equation-driven reconstruction where no trained artifact was released.

No weights or NVIDIA source are vendored. See [EVIDENCE.md](EVIDENCE.md), [COVERAGE.md](COVERAGE.md), [NOTICE.md](NOTICE.md) and the [source notes](articles/lift-splat-shoot-source-notes.md).

## Run locally

```bash
pnpm install
pnpm dev
```

Verification:

```bash
pnpm check
pnpm lint
pnpm build:pages
```

## Interaction

- Move through eight progressively dependent scenes; The Contract supplies the global map before the ray-level details begin.
- Click the real CAM_FRONT image or use the near-truck, far-vehicle and sky presets to compare the corresponding checkpoint depth and context tensors.
- Read the persistent tensor ledger from left to right: input shape, named operation, output shape.
- Step through network pixel, raw pixel, camera ray, camera point and ego point without changing the selected candidate.
- Compare sum, mean, max and bilinear splatting, then move a candidate through a cell boundary.
- Use the geometry and truth evidence drawers for real images, calibration, checkpoint output, GT and LiDAR.
- The bottom timeline, arrow keys and autoplay move between scenes; Escape closes the contents drawer.
- The UI honors `prefers-reduced-motion`.

## Sources

- [Lift, Splat, Shoot paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590188.pdf)
- [Pinned official implementation](https://github.com/nv-tlabs/lift-splat-shoot/tree/2903467c91ee9c12f0917a12c22ab1f04e607ae0)
- [BEVDepth paper](https://arxiv.org/abs/2206.10092)
- [BEVDepth official implementation](https://github.com/Megvii-BaseDetection/BEVDepth)
- [Pinned OpenMMLab nuScenes demo](https://github.com/open-mmlab/mmdetection3d/tree/fe25f7a51d36e3702f961e198894580d83c4387b/demo/data/nuscenes)

The site code is MIT licensed. Dataset and model artifacts remain subject to their own terms; see [NOTICE.md](NOTICE.md).
