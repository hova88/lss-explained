# Lift-Splat-Shoot source notes

This document accompanies the twelve-scene visual essay at **LSS Explained**. It separates paper claims, official-code behavior, real-sample evidence, checkpoint exports and teaching reconstructions.

## Pipeline

```text
calibrated images
→ shared CamEncode
→ 41-bin latent-depth Lift
→ exact camera-to-ego geometry
→ pillar sum pooling
→ BEV encoder
→ task logits
```

The public checkpoint predicts vehicle segmentation logits shaped `[B,1,200,200]` over a 100 m × 100 m ego-centric region at 0.5 m per cell.

## Exact geometry

```text
p_ego = R_cam→ego K⁻¹ [d·A⁻¹(u′−a), d]ᵀ + t_cam→ego
```

The site uses column vectors. It first undoes image augmentation, forms the depth-scaled homogeneous pixel, applies `K⁻¹`, rotates into ego axes and adds the real camera center. Frustums are computed from the pinned intrinsics and image corners.

## Lift and supervision

```text
f[d,c] = softmax(depth_logits)[d] × context[c]
```

Original LSS has no depth ground-truth loss. Final BEV `BCEWithLogitsLoss` shapes both context and the latent depth allocation through backpropagation.

## Splat and BEV

Six cameras produce `6 × 41 × 8 × 22 = 43,296` candidates before filtering. Points are quantized into the half-open `[-50,50)` grid, sorted by voxel rank and exactly summed with `QuickCumsum`. A single vertical voxel collapses the result to `[B,64,200,200]` before the BEV encoder emits raw vehicle logits.

## LiDAR boundary

The pinned 34,688-point LiDAR scan is an independent spatial ruler. It never enters checkpoint inference. One selected point keeps the same ID across LiDAR coordinates, ego coordinates, camera projection and BEV index.

## Evidence boundary

- **PAPER** — ECCV 2020 claims and experiments.
- **OFFICIAL CODE** — pinned NVIDIA source behavior.
- **CHECKPOINT** — exported tensors from `model525000.pt`.
- **REAL SAMPLE** — pinned nuScenes images, calibration, boxes and LiDAR.
- **TEACHING** — equation-level reconstruction where no trained artifact was released.

## Visual interaction references

The BEV lab adopts the calm visual hierarchy of Waymo's rider displays: a stable ego glyph, metric context, nearby objects and one legible motion cue. References: [Google Design — Taming the Road](https://design.google/library/trusting-driverless-cars/) and [Waymo — New In-Car Displays](https://waymo.com/blog/2021/04/waymos-fully-autonomous-ride-hailing-service-has-new-features/). The data shown here remains the pinned nuScenes/LSS evidence described above.

For the complete audit trail, source hashes and licensing boundaries, open `EVIDENCE.md`, `COVERAGE.md` and `NOTICE.md` in the public repository.
