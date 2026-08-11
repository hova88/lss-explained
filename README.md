# Lift-Splat-Shoot — from pixels to BEV

An interactive, bilingual, source-audited course on the original ECCV 2020 **Lift, Splat, Shoot** architecture. Seventeen progressive chapters first establish the task, inputs, outputs, training and inference path, then trace one real nuScenes sample through preprocessing, CamEncode, latent depth, camera-to-ego geometry, pillar sum pooling, BEV encoding, supervision, post-processing and the paper's trajectory-scoring equation.

**Live:** https://hova88.github.io/lss-explained/

This is the camera-to-BEV companion to [pointpillars-explained](https://github.com/hova88/pointpillars-explained). Version 3 reorganizes the experience into seven acts—Mission, Inputs, Image, Geometry, BEV, Learn and Proof—and reveals each lesson in three layers instead of showing every fact at once. A persistent transform graph and synchronized camera/3D/BEV evidence use the sample's 34,688-point LiDAR scan. LiDAR is a reference overlay only and is never passed into LSS inference.

## Course flow

```text
N images + K,R,t
→ resize / crop + post-transform
→ shared CamEncode
→ 41-bin latent depth × 64D context
→ exact get_geometry
→ pillar sum + QuickCumsum
→ [B,64,200,200] BEV tensor
→ BevEncode + vehicle logits
→ BCE supervision (training) / sigmoid + threshold (inference)
```

The original public checkpoint is a BEV vehicle semantic-segmentation model. It does not decode 3D boxes, run NMS or perform tracking. Shoot is taught separately from the paper equation because the official repository does not publish its planning checkpoint.

## Evidence boundary

The experience labels three kinds of claims:

- **REAL SAMPLE** — calibration, images, LiDAR reference, annotations, and artifacts derived from the fixed nuScenes sample `ca9a282c9e77460f8360f564131a8af5`.
- **CHECKPOINT** — tensors or rasters produced by a strict load of the pinned official checkpoint.
- **PAPER** — metrics and experimental claims transcribed from the ECCV 2020 paper.
- **TEACHING** — explanatory geometry or equation-driven reconstructions. Chapter 17 is teaching-only because the public official repository does not ship a planning checkpoint.

The committed browser assets were produced by strictly loading the official `model525000.pt`; no weights or NVIDIA source are vendored. See [EVIDENCE.md](EVIDENCE.md), [COVERAGE.md](COVERAGE.md), and [NOTICE.md](NOTICE.md).

## Run locally

```bash
pnpm install
pnpm dev
```

Then open `http://localhost:3000`. Verification:

```bash
pnpm check
pnpm build:pages
```

## Reproduce model-derived assets

The exporter requires Python 3.12 and an isolated environment. Its dependency set is deliberately small because the pinned official model definition is imported from `.cache/`, while nuScenes metadata comes from the fixed OpenMMLab demo pickle.

```bash
bash scripts/fetch-pinned-inputs.sh
python3 -m venv .model-venv
.model-venv/bin/pip install numpy==1.26.4 Pillow==11.3.0 torch==2.7.1 torchvision==0.22.1 efficientnet_pytorch==0.7.0
.model-venv/bin/python scripts/export_official_artifacts.py
pnpm verify:assets
```

Inputs are downloaded into ignored `.cache/` storage. The exporter writes only derived, auditable static assets to `public/data/`.

## Controls

- Drag to orbit; wheel or pinch to zoom.
- Click cameras, pixels, exact 8×22 feature anchors, depth bins, LiDAR points, GT boxes, BEV cells, and trajectories to inspect them.
- Use the 中文/EN switch without losing the current chapter or selection.
- In the truth lab, switch probability/threshold/GT/error layers and adjust LiDAR color, point size, confidence threshold, and opacity.
- Arrow keys move between chapters; Space toggles autoplay; Escape closes the inspector.
- The UI honors `prefers-reduced-motion`.

## Sources

- [Lift, Splat, Shoot paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590188.pdf)
- [Pinned official implementation](https://github.com/nv-tlabs/lift-splat-shoot/tree/2903467c91ee9c12f0917a12c22ab1f04e607ae0)
- [Pinned OpenMMLab nuScenes demo](https://github.com/open-mmlab/mmdetection3d/tree/fe25f7a51d36e3702f961e198894580d83c4387b/demo/data/nuscenes)
- [Akash Prakash's implementation walkthrough](https://akashprakas.github.io/akashBlog/posts/2025-11-15-LiftSplatShoot.html) and [UCLA Deep Vision's BEV overview](https://ucladeepvision.github.io/CS163-Projects-2024Fall/2024/12/13/team21-BEVSeg.html) informed the progressive teaching order; every technical claim was checked against the paper and pinned source.

The site code is MIT licensed. Dataset and model artifacts remain subject to their own terms; see [NOTICE.md](NOTICE.md).
