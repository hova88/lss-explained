# LSS Explained — from pixels to BEV

An English-only, source-audited visual essay about the original ECCV 2020 **Lift, Splat, Shoot** architecture. Twelve hand-drawn narrative scenes and three interactive labs trace one real nuScenes sample through image preprocessing, latent-depth Lift, exact camera-to-ego geometry, pillar pooling, BEV reasoning, supervision, inference, evidence and trajectory scoring.

**Live:** https://hova88.github.io/lss-explained/

This is the camera-to-BEV companion to [pointpillars-explained](https://github.com/hova88/pointpillars-explained). Version 6 is a content-first spatial whiteboard: the narrative begins with motivation and exact I/O, then introduces only the geometry and learning machinery needed for the next step. Deterministic ink drawings guide the course; calibrated 3D is optional inside the labs. The pinned 34,688-point LiDAR scan is a reference overlay only and is never passed into LSS inference.

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
→ BCE supervision / sigmoid + threshold
```

The public checkpoint performs BEV vehicle semantic segmentation. It does not decode 3D boxes, run NMS, track objects or estimate velocity. Shoot is presented as a paper-equation teaching reconstruction because the official repository did not publish its planning checkpoint.

## Evidence boundary

- **REAL SAMPLE** — calibration, images, a reference LiDAR scan and annotations from sample `ca9a282c9e77460f8360f564131a8af5`.
- **CHECKPOINT** — tensors and rasters exported by strictly loading the pinned official checkpoint.
- **PAPER** — metrics and experimental claims reported in the ECCV 2020 paper.
- **OFFICIAL CODE** — behavior of the pinned NVIDIA implementation.
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

- Scroll through twelve progressively dependent scenes; each states what enters, what changes and what leaves.
- Guided scenes use a hand-drawn spatial whiteboard. Open calibrated 3D only when you want to orbit, zoom or inspect real geometry.
- Click cameras, depth bins, LiDAR points, GT objects, BEV cells and trajectory templates to follow one shared evidence thread.
- Use the three lab sections for geometry, BEV truth and robustness/trajectory experiments.
- Arrow keys move between scenes; Escape returns to the guided story and closes inspection notes.
- The UI honors `prefers-reduced-motion`.

## Sources

- [Lift, Splat, Shoot paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590188.pdf)
- [Pinned official implementation](https://github.com/nv-tlabs/lift-splat-shoot/tree/2903467c91ee9c12f0917a12c22ab1f04e607ae0)
- [Pinned OpenMMLab nuScenes demo](https://github.com/open-mmlab/mmdetection3d/tree/fe25f7a51d36e3702f961e198894580d83c4387b/demo/data/nuscenes)

The site code is MIT licensed. Dataset and model artifacts remain subject to their own terms; see [NOTICE.md](NOTICE.md).
