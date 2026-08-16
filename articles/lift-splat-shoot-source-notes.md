# LSS and BEVDepth source notes

These notes accompany **LSS Explained v12**. The lesson begins with the complete camera-to-BEV contract, follows one selected ray through the remaining seven scenes, then uses BEVDepth to expose the part of the original LSS design that is easiest to misunderstand: how the depth branch is trained.

The site keeps five evidence classes separate:

- **LSS PAPER / LSS CODE** — claims or behavior in the ECCV 2020 paper and pinned NVIDIA implementation.
- **BEVDEPTH PAPER / BEVDEPTH CODE** — the explicit depth-learning extension reported by BEVDepth and implemented in its official repository.
- **CHECKPOINT** — tensors exported from the official LSS `model525000.pt` checkpoint for the pinned nuScenes sample.
- **REAL SAMPLE** — nuScenes images, calibration, boxes and one LiDAR scan used to audit spatial alignment.
- **TEACHING** — a deterministic diagram or comparison, never presented as trained-model output.

## 1. The problem and the tensor path

LSS consumes calibrated camera images and predicts a raster in the vehicle's ego frame. In the pinned configuration the path is:

```text
[B,6,3,128,352] camera images
→ [B,6,105,8,22] CamEncode output
→ split into [B,6,41,8,22] depth logits and [B,6,64,8,22] context
→ [B,6,41,8,22,64] lifted frustum features
→ 43,296 metric candidates per batch item
→ [B,64,200,200] pooled BEV
→ [B,1,200,200] LSS vehicle logits
```

This is not panorama stitching. Each image keeps its own intrinsics, camera-to-ego pose and image post-transform. Geometry makes the six tensors commensurable before a BEV CNN reasons over their neighborhoods.

## 2. Depth distribution and context payload

CamEncode shares EfficientNet-B0 weights across cameras and emits 105 channels at every 8 × 22 anchor. The first 41 channels are softmax-normalized depth logits; the remaining 64 values are context features.

The distinction matters. Depth answers **where along this ray should evidence be placed?** Context answers **what evidence should be carried there?** The opening interaction reads both tensors from the nearest real 8 × 22 checkpoint anchor for the clicked CAM_FRONT pixel, so the near-truck, far-vehicle and sky examples are actual feature-cell comparisons. Lift is their broadcast outer product:

```text
F3d[b,n,d,h,w,c] = softmax(D)[b,n,d,h,w] × C[b,n,c,h,w]
```

The operation copies one context vector to all 41 depth hypotheses and gates each copy by its depth weight. It does not first select a single point. In original LSS, these weights are latent allocations optimized only through the final task loss; they are not directly supervised metric-depth probabilities.

The 8 × 22 anchors are the official frustum samples: `linspace(0,351,22)` and `linspace(0,127,8)`. They are not assumed 16-pixel block centers.

## 3. From network pixel to camera ray

The site separates four objects that informal diagrams often collapse:

1. the optical center `O = [0,0,0]` in camera coordinates;
2. the augmented network anchor `p' = [u',v',1]`;
3. the raw homogeneous image point `p = [u,v,1]`;
4. the metric camera point `p_cam = [x,y,z]`.

The official sequence first undoes image augmentation:

```text
[u,v] = A⁻¹([u',v'] - a)
```

Then the inverse intrinsic matrix converts the raw pixel into a direction ratio:

```text
r = K⁻¹[u,v,1]ᵀ = [(u-cx)/fx, (v-cy)/fy, 1]ᵀ
```

Finally a sampled metric depth scales that ray:

```text
p_cam(d) = d r = K⁻¹[du,dv,d]ᵀ
```

For nuScenes camera coordinates, `+x` points image-right, `+y` image-down and `+z` along the optical axis. The interactive diagram uses the real intrinsic matrix and labels every intermediate tensor rather than treating a pixel as an already-metric 3D point.

## 4. Camera coordinates into ego coordinates

For each camera, nuScenes calibration provides a rigid transform with rotation `R_cam→ego` and optical-center translation `t_cam→ego`:

```text
p_ego = R_cam→ego p_cam + t_cam→ego
```

With column vectors, transforms compose right-to-left. The optical axis in ego coordinates is the third column of `R_cam→ego`; the site constructs each frustum from the four image corners and that real axis. This prevents the common visual error of drawing a camera cone toward the vehicle.

The complete LSS geometry is therefore:

```text
p_ego(d) = R_cam→ego K⁻¹[d·A⁻¹(p'−a), d]ᵀ + t_cam→ego
```

The direct LiDAR-to-camera matrices in the fixed demo also include the ego-pose chain between different sensor timestamps. In general they are not the static shortcut `inverse(cam2ego) × lidar2ego`.

## 5. Splat and collision semantics

Six cameras × 41 depth bins × 8 × 22 anchors produce 43,296 candidates before bounds filtering. Metric ego coordinates are quantized into the half-open 0.5 m grid `[-50,50)`:

```text
ix = floor((x + 50) / 0.5)
iy = floor((y + 50) / 0.5)
```

Multiple candidates can collide in the same BEV cell. Original LSS uses **sum pooling**. It encodes voxel indices as ranks, sorts candidates, computes prefix sums, retains group ends, then differences adjacent group totals. `QuickCumsum` is an efficient implementation of exactly the same sum reduction, not a different aggregation rule.

The lesson lets the reader compare sum, mean, max and bilinear splatting. Mean changes density semantics, max discards all but the strongest feature, and bilinear splatting distributes a candidate across four neighboring cells. These alternatives are labeled teaching comparisons; the pinned LSS result uses hard-cell sum pooling.

## 6. BEV encoding in LSS

The pooled tensor is arranged as `[B,C,Z,X,Y]`. The published configuration has one vertical voxel, so collapsing `Z` yields `[B,64,200,200]`. A ResNet-18-style `BevEncode` network supplies spatial context and produces task logits.

For the official LSS vehicle-segmentation task, training applies `BCEWithLogitsLoss` at the final BEV output. Gradients travel backward through `BevEncode`, sum pooling, Lift, the latent depth softmax and the image encoder. The paper describes positive weight 1.0 while the public training script defaults to 2.13; this discrepancy remains explicit in the lesson.

Official evaluation uses `logit > 0`, equivalent to `sigmoid(logit) > 0.5`. That model has no 3D box decoder, NMS, tracker or velocity estimator.

## 7. What BEVDepth changes

BEVDepth retains the same conceptual Lift equation `F3d = F2d ⊗ Dpred`, but changes how the depth branch receives learning signal.

Its training data preparation projects LiDAR points into each camera using the calibrated ego-to-camera transform and intrinsics, rejects points behind the camera or outside the image, downsamples sparse depth by taking the minimum nonzero depth in each block, discretizes that depth into a bin, and creates a one-hot target. Binary cross entropy is evaluated only at pixels with a valid foreground depth target.

The official experiment code combines losses as:

```text
L = L_detection + 3 L_depth
```

Thus LSS asks the final task to discover a useful latent depth allocation, whereas BEVDepth adds a direct metric-depth gradient. LiDAR is used to construct training targets, not as an inference input; inference remains camera-only.

BEVDepth also conditions its DepthNet with a 27-dimensional camera parameter vector containing intrinsics and augmentation/extrinsic terms. Squeeze-and-excitation gates the context and depth branches with that camera-aware signal. Its larger depth receptive field and efficient voxel-pooling implementation are additional engineering changes; they should not be confused with the conceptual change in supervision.

The public checkpoint audited by this website is LSS vehicle segmentation. The BEVDepth comparison is sourced from its paper and code and is never presented as a BEVDepth checkpoint result.

## 8. Truth-lab orientation and evidence

Camera images and a BEV heatmap should not resemble each other pixel-for-pixel. Lift distributes perspective evidence over depth, rigid transforms move it into ego coordinates, Splat merges cameras, and the BEV encoder adds spatial context. Correct correspondence is object- and coordinate-based, not silhouette-based.

The fixed 34,688-point LiDAR scan is an independent ruler for LSS. A selected point keeps one ID across its LiDAR coordinate, ego coordinate, camera projection and BEV cell. It is marked **REAL SAMPLE / REFERENCE LIDAR**, never an LSS input.

The display contract is:

```text
ego +x = vehicle forward = screen up
ego +y = vehicle left    = screen left
ego +z = up
screenX = 1 - (iy + 0.5) / 200
screenY = 1 - (ix + 0.5) / 200
```

## Primary references

- Jonah Philion and Sanja Fidler, *Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by Implicitly Unprojecting to 3D*, ECCV 2020.
- Yinhao Li et al., *BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection*, AAAI 2023.
- NVIDIA LSS implementation pinned to `2903467c91ee9c12f0917a12c22ab1f04e607ae0`.
- Official BEVDepth implementation inspected at `d78c7b58b10b9ada940462ba83ab24d99cae5833`.
- OpenMMLab nuScenes demo data pinned to `fe25f7a51d36e3702f961e198894580d83c4387b`.
- See `EVIDENCE.md`, `COVERAGE.md` and `NOTICE.md` for hashes, licensing boundaries and audit details.
