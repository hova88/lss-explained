# Lift-Splat-Shoot Geometry Lab

An interactive, source-audited explanation of the original ECCV 2020 **Lift, Splat, Shoot** architecture. Six real nuScenes cameras from one timestamp are followed through latent depth, camera-to-ego geometry, pillar sum pooling, BEV encoding, robustness probes, and the paper's trajectory-scoring equation.

**Live:** https://hova88.github.io/lss-explained/

This is the camera-to-BEV companion to [pointpillars-explained](https://github.com/hova88/pointpillars-explained). It keeps the same paper-like full-screen grammar, chapter timeline, 3D camera, autoplay, keyboard/touch controls, and click-to-inspect cards.

## Evidence boundary

The experience labels three kinds of claims:

- **REAL SAMPLE** — calibration, images, annotations, and artifacts derived from the fixed nuScenes sample `ca9a282c9e77460f8360f564131a8af5`.
- **PAPER** — metrics and experimental claims transcribed from the ECCV 2020 paper.
- **TEACHING** — explanatory geometry or equation-driven reconstructions. Chapter 9 is teaching-only because the public official repository does not ship a planning checkpoint.

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
- Click cameras, rays, depth bins, BEV cells, and trajectories to inspect them.
- Arrow keys move between chapters; Space toggles autoplay; Escape closes the inspector.
- The UI honors `prefers-reduced-motion`.

## Sources

- [Lift, Splat, Shoot paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590188.pdf)
- [Pinned official implementation](https://github.com/nv-tlabs/lift-splat-shoot/tree/2903467c91ee9c12f0917a12c22ab1f04e607ae0)
- [Pinned OpenMMLab nuScenes demo](https://github.com/open-mmlab/mmdetection3d/tree/fe25f7a51d36e3702f961e198894580d83c4387b/demo/data/nuscenes)

The site code is MIT licensed. Dataset and model artifacts remain subject to their own terms; see [NOTICE.md](NOTICE.md).
