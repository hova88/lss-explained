export type Evidence = "PAPER" | "OFFICIAL CODE" | "REAL SAMPLE" | "CHECKPOINT" | "TEACHING";
export type IllustrationKind = "overview" | "sample" | "rig" | "features" | "ray" | "lift" | "geometry" | "splat" | "bev" | "learning" | "truth" | "shoot";
export type StageView = "illustration" | "rig3d" | "bev" | "linked";
export type LabId = "geometry" | "bev" | "robustness";

export type NarrativeScene = {
  id: string;
  act: string;
  title: string;
  question: string;
  reveal: string;
  beats: [string, string, string];
  formula: string;
  evidence: Evidence;
  source: string;
  illustration: IllustrationKind;
  stageView: StageView;
  lab?: LabId;
};

export const SCENES: NarrativeScene[] = [
  {
    id: "the-problem", act: "I · ORIENT", title: "The problem LSS solves",
    question: "How can six incompatible perspective views become one space a planner can use?",
    reveal: "Lift-Splat-Shoot turns calibrated camera images into an ego-centric bird’s-eye-view grid.",
    beats: [
      "The input is N images plus each camera’s intrinsics K and camera-to-ego pose R,t — never a stitched panorama.",
      "The public checkpoint predicts one vehicle logit for every 0.5 m BEV cell in a 100 m × 100 m region.",
      "LiDAR never enters the network. It appears here later only as an independent spatial ruler."
    ],
    formula: "{Iₙ, Kₙ, Rₙ, tₙ}ₙ₌₁ᴺ → ŷBEV ∈ ℝᴮˣ¹ˣ²⁰⁰ˣ²⁰⁰",
    evidence: "PAPER", source: "Paper Fig. 4 · official forward()", illustration: "overview", stageView: "illustration"
  },
  {
    id: "one-sample", act: "II · PREPARE", title: "One nuScenes sample",
    question: "What exactly is loaded, transformed, supervised and kept out of the model?",
    reveal: "One sample is a synchronized packet of images, calibration records and a rasterized BEV target.",
    beats: [
      "SegmentationData returns images, rotations, translations, intrinsics, post-transforms and the binary target mask.",
      "Vehicle boxes are moved into the LiDAR-time ego frame, then their bottom faces are filled into a 200 × 200 target.",
      "Training randomly uses five of six cameras; validation uses all six."
    ],
    formula: "sample → (images, calibration, post-transform, BEV target)",
    evidence: "OFFICIAL CODE", source: "data.py · get_image_data() · get_binimg()", illustration: "sample", stageView: "illustration"
  },
  {
    id: "camera-rig", act: "II · PREPARE", title: "The camera rig is geometry",
    question: "Where do K, R, t and ego pose come from — and which way does every lens face?",
    reveal: "nuScenes calibration turns each optical frame into a precisely placed outward-facing sensor in ego space.",
    beats: [
      "K describes the pinhole camera; calibrated_sensor quaternion and translation define camera→ego.",
      "With column vectors, A→B→C composes as Tᴮ→ᶜTᴬ→ᴮ; the first operation sits on the right.",
      "Cross-time LiDAR→camera traverses two ego poses and cannot be replaced by static extrinsics."
    ],
    formula: "pᶜ = Tᴮ→ᶜ Tᴬ→ᴮ pᴬ · T⁻¹=[Rᵀ,−Rᵀt]",
    evidence: "REAL SAMPLE", source: "nuScenes calibrated_sensor · ego_pose", illustration: "rig", stageView: "rig3d", lab: "geometry"
  },
  {
    id: "image-features", act: "III · LIFT", title: "Images become features",
    question: "How does a 1600 × 900 photograph become an 8 × 22 field of learned evidence?",
    reveal: "Image geometry is recorded while a shared EfficientNet turns each camera into depth logits plus context.",
    beats: [
      "The pinned frame resizes to 352 × 198, crops to 352 × 128, then ImageNet-normalizes RGB values.",
      "post_rot and post_trans preserve the raw→network pixel transform so geometry can undo it later.",
      "A 1×1 head emits 41 depth logits and 64 context channels at every one of the 8 × 22 anchors."
    ],
    formula: "[B,N,3,128,352] → [B·N,105,8,22]",
    evidence: "CHECKPOINT", source: "tools.py · img_transform() · models.py · CamEncode", illustration: "features", stageView: "illustration", lab: "geometry"
  },
  {
    id: "pixel-ray", act: "III · LIFT", title: "A pixel becomes a ray",
    question: "Why does K⁻¹ recover a direction, but not a location?",
    reveal: "A monocular pixel fixes one ray from the optical center; depth remains deliberately unresolved.",
    beats: [
      "Camera coordinates use +x right, +y down and +z optical-forward; LSS depth d is zcam.",
      "The official frustum enumerates 41 hypotheses from 4 to 44 m at each feature anchor.",
      "The 8 × 22 anchors are linspace samples over the network image, not naïve 16-pixel block centers."
    ],
    formula: "p_cam(d)=K⁻¹[d·u,d·v,d]ᵀ",
    evidence: "OFFICIAL CODE", source: "models.py · create_frustum()", illustration: "ray", stageView: "rig3d", lab: "geometry"
  },
  {
    id: "lift", act: "III · LIFT", title: "Lift into latent depth",
    question: "How does LSS build 3D features without a depth sensor or depth labels?",
    reveal: "A softmax depth allocation spreads the same context vector along 41 metric candidates.",
    beats: [
      "Every lifted point carries f[d,c]=α[d]·context[c]; depth changes weight, not semantic content.",
      "The 41 weights sum to one, but they may be multimodal and are not guaranteed physical depth probabilities.",
      "The original model has no depth loss: final BEV supervision shapes this latent allocation indirectly."
    ],
    formula: "f(d,c) = softmax(ℓdepth)[d] · context[c]",
    evidence: "CHECKPOINT", source: "models.py · get_depth_feat()", illustration: "lift", stageView: "rig3d", lab: "geometry"
  },
  {
    id: "ego-geometry", act: "IV · GEOMETRY", title: "Every camera meets in ego",
    question: "How does one [u′,v′,d] anchor land at a metric location shared by all six cameras?",
    reveal: "Undo the image transform, unproject with K⁻¹, rotate the basis, then move the point to the real optical center.",
    beats: [
      "A⁻¹(u′−a) first returns network coordinates to the raw frame in which K is defined.",
      "K⁻¹[du,dv,d]ᵀ produces a camera-frame point; R changes basis and t positions the camera in ego.",
      "Permuting images together with their calibration cannot change sum pooling — camera order disappears."
    ],
    formula: "pₑ = Rcam→ego K⁻¹[d·A⁻¹(u′−a), d]ᵀ + tcam→ego",
    evidence: "REAL SAMPLE", source: "models.py · get_geometry() · 18 golden samples", illustration: "geometry", stageView: "rig3d", lab: "geometry"
  },
  {
    id: "splat", act: "V · SPLAT", title: "Splat onto the ground",
    question: "How do 43,296 frustum candidates become a regular tensor a CNN can process?",
    reveal: "Metric points are quantized, grouped by rank and summed exactly inside BEV pillars.",
    beats: [
      "The grid is the half-open interval [−50,50) at 0.5 m resolution; out-of-range points are discarded.",
      "[ix,iy,iz,b] is encoded as a rank so samples from one voxel become adjacent after sorting.",
      "QuickCumsum uses prefix sums and group-end differences; its output exactly equals naïve sum pooling."
    ],
    formula: "index=floor((pₑ−(b−Δ/2))/Δ) · BEVcell=Σ features",
    evidence: "OFFICIAL CODE", source: "models.py · voxel_pooling() · tools.py · QuickCumsum", illustration: "splat", stageView: "bev", lab: "bev"
  },
  {
    id: "bev-reasoning", act: "V · SPLAT", title: "Reason in bird’s-eye view",
    question: "Why is a BEV CNN still needed after geometry has put evidence in the right cells?",
    reveal: "Splat establishes location; BevEncode adds neighborhood context and emits task logits.",
    beats: [
      "Pooling writes [B,C,Z,X,Y]; one 20 m-tall z voxel collapses to [B,64,200,200].",
      "A ResNet-18-style trunk builds multiscale context and fuses an upsampled deep layer with an early layer.",
      "The final 1×1 convolution emits raw [B,1,200,200] logits — sigmoid has not happened yet."
    ],
    formula: "[B,64,200,200] → BEV encoder → [B,1,200,200]",
    evidence: "OFFICIAL CODE", source: "models.py · BevEncode", illustration: "bev", stageView: "bev", lab: "bev"
  },
  {
    id: "learn-infer", act: "VI · LEARN", title: "Learn, infer, stop",
    question: "How does a final BEV label teach latent depth, and where does original LSS end?",
    reveal: "BCE gradients travel back through BEV reasoning, pooling and Lift; inference ends at a thresholded vehicle mask.",
    beats: [
      "An incorrect cell penalizes every frustum point that contributed to it, adjusting both context and depth allocation.",
      "Official IoU uses logits>0, exactly equivalent to sigmoid(logit)>0.5.",
      "Original LSS segmentation has no box decoder, NMS, tracker or velocity head."
    ],
    formula: "L=BCEWithLogits(ŷ,y) → ∂L/∂α(d) · mask=σ(logit)≥τ",
    evidence: "OFFICIAL CODE", source: "train.py · SimpleLoss · tools.py · get_batch_iou()", illustration: "learning", stageView: "bev", lab: "bev"
  },
  {
    id: "truth-lab", act: "VII · PROOF", title: "The truth lab",
    question: "A camera image will never resemble a BEV heatmap — so how do we know the map is not mirrored?",
    reveal: "Link the same LiDAR point, GT box and BEV cell across image, 3D and metric ego coordinates.",
    beats: [
      "Perspective silhouettes are not preserved after depth allocation and multi-camera aggregation.",
      "All 34,688 LiDAR points are reference evidence only; none participate in checkpoint inference.",
      "Probability, GT, occupancy, contributors and screen coordinates share one ego→BEV mapping."
    ],
    formula: "LiDAR id ↔ camera uv ↔ ego xyz ↔ BEV [ix,iy]",
    evidence: "REAL SAMPLE", source: "Pinned nuScenes sample · OpenMMLab transforms", illustration: "truth", stageView: "linked", lab: "bev"
  },
  {
    id: "evidence-action", act: "VII · PROOF", title: "From evidence to action",
    question: "What do the paper’s experiments prove — and what does Shoot add beyond segmentation?",
    reveal: "LSS is robust because geometry is explicit, not because failure disappears; Shoot turns a learned BEV cost into a trajectory distribution.",
    beats: [
      "nuScenes vehicle IoU rises from CNN 24.25 and OFT 30.05 to Lift-Splat 32.07.",
      "Camera dropout and extrinsic-noise training improve matching failure cases; night and long range remain difficult.",
      "Shoot scores 1K five-second trajectory templates, but the official repository released no planning checkpoint."
    ],
    formula: "p(τᵢ|o)=exp(−cost(τᵢ))/Σⱼexp(−cost(τⱼ))",
    evidence: "PAPER", source: "Paper Tables 1–5 · Eq. 2 · Sec. 3.3 and 5.6", illustration: "shoot", stageView: "linked", lab: "robustness"
  }
];

export function sceneIndexFromHash(hash: string) {
  const id = hash.replace(/^#/, "");
  const index = SCENES.findIndex((scene) => scene.id === id);
  return index < 0 ? 0 : index;
}
