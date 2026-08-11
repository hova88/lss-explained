# Lift-Splat-Shoot source notes

These notes accompany the interactive essay at **LSS Explained**. They separate four kinds of evidence that are easy to blur together when reading a visual explanation:

- **PAPER** — claims, equations and experiments reported in the ECCV 2020 paper.
- **OFFICIAL CODE** — behavior visible in the pinned NVIDIA implementation.
- **CHECKPOINT** — tensors exported from the official `model525000.pt` checkpoint for the pinned nuScenes sample.
- **REAL SAMPLE** — nuScenes images, calibration, boxes and a single LiDAR scan used only for spatial verification.
- **TEACHING** — a reconstruction that explains an equation when the official repository did not release a corresponding trained artifact.

## 1. The task

Lift-Splat-Shoot consumes a set of calibrated camera images and predicts a raster in the vehicle’s ego frame. The public checkpoint used by this site performs vehicle semantic segmentation over a 100 m × 100 m region at 0.5 m per cell, producing raw logits shaped `[B,1,200,200]`.

The camera set is not a stitched panorama. Every image travels with its own intrinsics `K`, camera-to-ego rotation `R`, translation `t`, and image post-transform. The model is designed so camera order does not matter when images and calibration are permuted together.

## 2. Image preprocessing and CamEncode

For the pinned validation frame, each 1600 × 900 camera image is resized and cropped to 352 × 128. `post_rot` and `post_trans` remember how a pixel moved. ImageNet normalization changes RGB values but not pixel geometry.

All cameras share EfficientNet-B0 weights. The image encoder merges batch and camera dimensions, extracts multiscale features, and emits 105 channels on an 8 × 22 grid: 41 depth logits plus 64 context channels.

The 8 × 22 frustum anchors come from `linspace(0,351,22)` and `linspace(0,127,8)`. They should not be replaced with assumed 16-pixel block centers.

## 3. Lift

At one feature anchor, the first 41 channels are softmax-normalized over depth. The remaining 64 values form a context vector. Their outer product creates a 41 × 64 lifted feature:

```text
f[d,c] = softmax(depth_logits)[d] × context[c]
```

Original LSS has no ground-truth depth loss. These weights are a latent, task-oriented allocation learned through the final BEV objective. They may be broad or multimodal and should not automatically be interpreted as calibrated physical depth probabilities.

## 4. Exact geometry

The official geometry sequence is:

```text
network anchor
→ undo post transform
→ form [d·u, d·v, d]
→ multiply by K⁻¹
→ rotate camera→ego
→ add camera→ego translation
```

With column vectors:

```text
p_ego = R_cam→ego K⁻¹ [d·A⁻¹(u′−a), d]ᵀ + t_cam→ego
```

The camera optical axis is the third column of the camera-to-ego rotation matrix. The interactive frustums are built from the four image corners unprojected with the real intrinsics; they are not decorative cones.

LiDAR-to-camera deserves a separate warning. The pinned direct matrix includes the ego-pose chain between the LiDAR timestamp and each camera timestamp. In general it is not equal to the static shortcut `inverse(cam2ego) × lidar2ego`.

## 5. Splat

Six cameras, 41 depth bins and an 8 × 22 feature grid produce 43,296 candidate points before bounds filtering. Ego coordinates are quantized into the half-open grid `[-50,50)` with 0.5 m cells.

Voxel indices are encoded as ranks and sorted. `QuickCumsum` uses prefix sums, retains group-ending entries and differences neighboring group totals. Its result exactly matches naïve sum pooling while avoiding an explicit sparse tensor operation.

The pooled tensor is first arranged as `[B,C,Z,X,Y]`. The published configuration has one vertical voxel, so collapsing `Z` produces `[B,64,200,200]`.

## 6. BEV learning and inference

`BevEncode` is a ResNet-18-style multiscale network. Geometry decides where evidence lands; the BEV network supplies neighborhood context and outputs task logits.

Training applies `BCEWithLogitsLoss` only at the final BEV output. Gradients pass backward through the BEV encoder, sum pooling, outer product, latent depth softmax and image encoder. The paper describes object positive weight 1.0, while the public training script defaults to 2.13; the site keeps that discrepancy explicit.

Official IoU evaluation uses `logits > 0`, which is exactly equivalent to `sigmoid(logit) > 0.5`. The segmentation model stops there. It has no 3D box decoder, NMS, tracker or velocity estimator.

## 7. The truth lab

The six images and BEV heatmap should not resemble one another. Perspective pixels are distributed over depth, transformed into ego coordinates and summed with evidence from other cameras. Two-dimensional camera silhouettes are not preserved.

The pinned 34,688-point LiDAR scan is therefore shown as an independent ruler. LiDAR is never passed into the checkpoint. A selected point keeps one ID while the site shows its LiDAR coordinate, ego coordinate, camera projection and BEV cell.

The display contract is:

```text
ego +x = vehicle forward = screen up
ego +y = vehicle left    = screen left
ego +z = up
ix = floor((x + 50) / 0.5)
iy = floor((y + 50) / 0.5)
```

## 8. Robustness and Shoot

The paper evaluates camera dropout, calibration noise and unseen camera arrangements. Robustness improves when matching perturbations are included during training; explicit geometry does not make the model automatically immune to broken calibration.

For vehicle segmentation on nuScenes, the paper reports IoU 24.25 for the CNN baseline, 26.83 for Frozen Encoder, 30.05 for OFT and 32.07 for Lift-Splat.

Shoot is a separate planning task. A learned BEV cost field scores 1,000 trajectory templates and produces a Boltzmann distribution over them. The official repository did not release a planning checkpoint, so the site labels its trajectory experiment as **TEACHING**, not checkpoint output.

## Visual interaction references

The linked BEV laboratory borrows a functional idea from Waymo's rider-facing visualization: establish trust with a stable ego symbol, restrained metric context, nearby objects and one legible motion cue instead of exposing every internal signal at once. See Google Design's [Taming the Road](https://design.google/library/trusting-driverless-cars/) and Waymo's note on [in-car displays](https://waymo.com/blog/2021/04/waymos-fully-autonomous-ride-hailing-service-has-new-features/). This essay keeps its own field-notebook palette and never implies that its LSS tensors are Waymo production outputs.

## Primary references

- Jonah Philion and Sanja Fidler, *Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D*, ECCV 2020.
- NVIDIA official implementation pinned to commit `2903467c91ee9c12f0917a12c22ab1f04e607ae0`.
- OpenMMLab nuScenes demo data pinned to commit `fe25f7a51d36e3702f961e198894580d83c4387b`.
- See `EVIDENCE.md`, `COVERAGE.md` and `NOTICE.md` in the repository for hashes, licensing boundaries and the full audit trail.
