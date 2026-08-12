export type Evidence = "PAPER" | "OFFICIAL CODE" | "REAL SAMPLE" | "CHECKPOINT" | "TEACHING";
export type IllustrationKind = "overview" | "sample" | "rig" | "features" | "ray" | "lift" | "geometry" | "splat" | "learning" | "truth" | "shoot";
export type LabId = "geometry" | "bev" | "robustness";

export type NarrativeStep = { label: string; text: string };

export type NarrativeScene = {
  id: string;
  act: string;
  title: string;
  question: string;
  reveal: string;
  explanation: string;
  steps: [NarrativeStep, NarrativeStep, NarrativeStep];
  formula: string;
  handoff: string;
  evidence: Evidence;
  source: string;
  illustration: IllustrationKind;
  lab?: LabId;
};

export const SCENES: NarrativeScene[] = [
  {
    id: "system-view", act: "I · THE JOB", title: "See the whole machine first",
    question: "What does LSS turn into what?",
    reveal: "Six calibrated camera views become one metric map around the ego vehicle.",
    explanation: "Camera images describe appearance in perspective. Driving decisions need shared ground coordinates. LSS bridges that gap with an explicit pipeline: Lift image evidence into possible 3D locations, Splat it into a BEV grid, then reason on that grid.",
    steps: [
      { label: "INPUT", text: "RGB images plus each camera’s intrinsics and camera→ego pose." },
      { label: "TRANSFORM", text: "Image features → depth candidates → ego-space pillars." },
      { label: "OUTPUT", text: "One vehicle logit for every 0.5 m BEV cell." }
    ],
    formula: "{images, K, R, t} → LIFT → SPLAT → BEV logits [B,1,200,200]",
    handoff: "Hold onto one idea: LSS creates a common metric workspace.",
    evidence: "PAPER", source: "Paper Fig. 1, Fig. 4 · official forward()", illustration: "overview"
  },
  {
    id: "perspective-gap", act: "I · THE JOB", title: "A pixel is not a place",
    question: "Why not simply stitch the six photographs?",
    reveal: "A pixel fixes a viewing direction, but not a metric location.",
    explanation: "The same two-pixel shift may mean centimeters nearby or meters far away. Different cameras also have different optical centers. Until depth and calibration are applied, their pixels cannot agree on where an object sits around the car.",
    steps: [
      { label: "KNOWN", text: "K⁻¹ converts a pixel into a camera ray." },
      { label: "UNKNOWN", text: "Depth d chooses one position along that ray." },
      { label: "LSS MOVE", text: "Keep many depths, then let BEV supervision weight them." }
    ],
    formula: "pixel + K⁻¹ ⇒ ray;   ray + depth d ⇒ 3D point",
    handoff: "Before solving depth, separate what the sample uses from what only audits it.",
    evidence: "PAPER", source: "Paper Sec. 3.1", illustration: "sample"
  },
  {
    id: "data-contract", act: "II · THE EVIDENCE", title: "One sample, four roles",
    question: "Which data actually enters LSS?",
    reveal: "Images are input; calibration is geometry; boxes are supervision; LiDAR is reference only.",
    explanation: "This pinned nuScenes frame contains six JPEGs, sensor calibration, ego poses and vehicle annotations. The LiDAR scan is shown only to check spatial alignment. It never enters the released camera-only checkpoint.",
    steps: [
      { label: "SEE", text: "Six images carry appearance." },
      { label: "PLACE", text: "K, R and t place image evidence in ego coordinates." },
      { label: "TEACH / CHECK", text: "Vehicle boxes form GT; 34,688 LiDAR points verify orientation." }
    ],
    formula: "model input = images + calibration   |   target = vehicle BEV mask",
    handoff: "Now follow only the model input, beginning with image preparation.",
    evidence: "REAL SAMPLE", source: "nuScenes sample · data.py SegmentationData", illustration: "rig", lab: "geometry"
  },
  {
    id: "image-encoding", act: "III · PREPARE", title: "Compress the image, keep its geometry",
    question: "How does 1600×900 become an 8×22 feature field?",
    reveal: "Resize and crop are recorded, not forgotten.",
    explanation: "The image becomes 352×128 before a shared EfficientNet encoder produces an 8×22 feature grid. post_rot and post_trans remember the pixel transform so geometry can undo it later.",
    steps: [
      { label: "RESIZE / CROP", text: "1600×900 → 352×198 → 352×128." },
      { label: "ENCODE", text: "The same camera encoder processes every view." },
      { label: "SPLIT", text: "Each anchor emits 41 depth logits and 64 context values." }
    ],
    formula: "[B,N,3,128,352] → [B,N,105,8,22] = 41 depth + 64 context",
    handoff: "Choose one feature anchor. Its next representation is a ray.",
    evidence: "OFFICIAL CODE", source: "img_transform() · CamEncode · create_frustum()", illustration: "features", lab: "geometry"
  },
  {
    id: "pixel-ray", act: "IV · LIFT", title: "Undo the image, open a ray",
    question: "What exactly does K⁻¹ recover?",
    reveal: "It recovers direction in camera coordinates—not depth.",
    explanation: "First undo resize and crop. Then K⁻¹ converts the restored pixel into a camera-frame direction, where +x is right, +y is down and +z is forward. LSS tests 41 z-depths from 4 m to 44 m on this ray.",
    steps: [
      { label: "RESTORE", text: "A⁻¹(u′−a) returns the network anchor to raw-image coordinates." },
      { label: "UNPROJECT", text: "K⁻¹ turns [du,dv,d] into a camera-frame point." },
      { label: "ENUMERATE", text: "Repeat for d = 4,5,…,44 m." }
    ],
    formula: "p_cam(d) = K⁻¹ [d·u, d·v, d]ᵀ",
    handoff: "The ray supplies possible places; the network supplies their weights.",
    evidence: "OFFICIAL CODE", source: "create_frustum() · get_geometry()", illustration: "ray", lab: "geometry"
  },
  {
    id: "lift", act: "IV · LIFT", title: "Lift without choosing one depth",
    question: "How can LSS learn 3D without depth labels?",
    reveal: "It distributes one context feature across all 41 depth candidates.",
    explanation: "A softmax turns the depth logits into weights α(d). The same 64D context vector c is scaled at each depth, producing α(d)c. BEV segmentation loss—not LiDAR depth—learns which allocations are useful.",
    steps: [
      { label: "WEIGH", text: "softmax makes 41 non-negative weights summing to one." },
      { label: "COPY", text: "The 64D context vector is shared along the ray." },
      { label: "LIFT", text: "Their outer product forms a [41,64] frustum feature." }
    ],
    formula: "f[d,c] = softmax(depth logits)[d] · context[c]",
    handoff: "Lift creates 3D candidates—but each camera still speaks its own frame.",
    evidence: "CHECKPOINT", source: "get_depth_feat() · exported real depth allocation", illustration: "lift", lab: "geometry"
  },
  {
    id: "exact-geometry", act: "V · SHARE A FRAME", title: "The coordinate chain, in order",
    question: "How does one lifted point reach the ego frame?",
    reveal: "Undo image transform → unproject → rotate → translate.",
    explanation: "With column vectors, the first operation appears on the right. K⁻¹ creates p_cam; R changes its basis into ego axes; t then places the camera’s optical center in ego meters. Changing this order changes the point.",
    steps: [
      { label: "IMAGE → CAMERA", text: "Undo A,a; then apply K⁻¹ at depth d." },
      { label: "CAMERA → EGO", text: "Apply the selected camera’s R and then +t." },
      { label: "SANITY", text: "Check RᵀR≈I, det(R)≈1 and pixel round trips." }
    ],
    formula: "p_ego = Rcam→ego · K⁻¹[d·A⁻¹(u′−a), d]ᵀ + tcam→ego",
    handoff: "Once points are in ego coordinates, the camera name no longer matters.",
    evidence: "REAL SAMPLE", source: "get_geometry() · 18 exported golden points", illustration: "geometry", lab: "geometry"
  },
  {
    id: "camera-fusion", act: "V · SHARE A FRAME", title: "Six cameras meet in ego",
    question: "Where does multi-camera fusion actually happen?",
    reveal: "Not on image planes. It happens when ego-space candidates land in the same pillar.",
    explanation: "Each camera is encoded independently with shared weights and transformed by its own calibration. Permuting cameras together with their calibration leaves the final sum unchanged: fusion depends on metric location, not camera order.",
    steps: [
      { label: "SEPARATE", text: "Run the shared encoder per camera." },
      { label: "CALIBRATE", text: "Use that camera’s K, post-transform, R and t." },
      { label: "MEET", text: "Candidates share evidence only after entering ego space." }
    ],
    formula: "pool({Tcam→ego · Lift(image)}₁…N) is invariant to camera order",
    handoff: "The points agree on a frame. Splat turns them into a dense tensor.",
    evidence: "PAPER", source: "Paper Sec. 3.2, 5.5 · real camera calibration", illustration: "rig", lab: "geometry"
  },
  {
    id: "splat", act: "VI · SPLAT", title: "Many points become one BEV grid",
    question: "How are irregular candidates pooled without losing collisions?",
    reveal: "Filter, quantize, group by voxel rank, then sum.",
    explanation: "The ground range [−50,50) is divided into 0.5 m cells. Each valid point receives [ix,iy,iz]. Sorting equal ranks together lets QuickCumsum add every 64D feature that lands in the same pillar.",
    steps: [
      { label: "FILTER", text: "Remove candidates outside the half-open metric bounds." },
      { label: "GROUP", text: "floor maps positions to voxel indices; rank identifies collisions." },
      { label: "SUM", text: "QuickCumsum equals ordinary sum pooling, only faster." }
    ],
    formula: "index = floor((p − origin) / Δ),   Fcell = Σ points-in-cell fᵢ",
    handoff: "Geometry is finished. A BEV CNN now interprets the pooled evidence.",
    evidence: "OFFICIAL CODE", source: "voxel_pooling() · QuickCumsum", illustration: "splat", lab: "bev"
  },
  {
    id: "learn-in-bev", act: "VII · LEARN", title: "Supervise the final map",
    question: "How does a BEV mask teach latent depth?",
    reveal: "BCE loss flows backward through the BEV encoder, pooling and Lift.",
    explanation: "After vertical collapse, BevEncode adds spatial context and emits one raw vehicle logit per cell. BCEWithLogits compares it with the rasterized vehicle mask. The gradient reaches every depth weight and context feature that contributed to a wrong cell.",
    steps: [
      { label: "REASON", text: "[B,64,200,200] → multiscale BEV CNN." },
      { label: "COMPARE", text: "Predicted logits versus binary vehicle occupancy." },
      { label: "BACKPROP", text: "The BEV error indirectly shapes α(d); no depth target is used." }
    ],
    formula: "L = BCEWithLogits(BEV logits, vehicle mask)",
    handoff: "Training ends at a task-specific BEV representation. Now read it correctly.",
    evidence: "OFFICIAL CODE", source: "BevEncode · SimpleLoss", illustration: "learning", lab: "bev"
  },
  {
    id: "truth-lab", act: "VIII · READ THE RESULT", title: "A BEV heatmap is not a camera image",
    question: "How do we know the output is aligned rather than mirrored?",
    reveal: "Audit it in meters with GT, LiDAR and fixed ego axes—not by matching silhouettes.",
    explanation: "Lift deliberately spreads image evidence along depth, then rotates and mixes it across cameras. The correct match is metric and object-level. In this frame, threshold 0.5 gives TP 282, FP 122, FN 120 and IoU 0.538; this is a diagnostic, not the paper’s validation score.",
    steps: [
      { label: "DECODE", text: "logit → sigmoid probability → optional threshold." },
      { label: "ORIENT", text: "ego +x is screen up; ego +y is screen left." },
      { label: "VERIFY", text: "GT boxes and reference LiDAR expose flips and offsets." }
    ],
    formula: "[x,y]ego ↔ [⌊(x+50)/.5⌋, ⌊(y+50)/.5⌋]BEV",
    handoff: "One real frame is understood. Finish by separating evidence from ambition.",
    evidence: "CHECKPOINT", source: "model525000.pt · nuScenes GT · reference LiDAR", illustration: "truth", lab: "bev"
  },
  {
    id: "evidence-action", act: "IX · THE BOUNDARY", title: "What LSS proves—and what it does not",
    question: "Which claims belong to the paper, checkpoint and teaching lab?",
    reveal: "Explicit geometry works well, but it does not remove uncertainty or failure modes.",
    explanation: "The paper reports vehicle IoU 32.07 for Lift-Splat versus 30.05 OFT and 24.25 CNN. It also tests missing cameras, calibration noise and new rigs. Shoot maps BEV costs to trajectory probabilities, but no planning checkpoint was released here.",
    steps: [
      { label: "PROVEN", text: "Strong camera-BEV segmentation and useful rig inductive biases." },
      { label: "FRAGILE", text: "Night, distance, occlusion, calibration error and missing views still matter." },
      { label: "TEACHING", text: "The Shoot trajectories reconstruct the paper equation only." }
    ],
    formula: "p(trajectoryᵢ) ∝ exp(−cost(trajectoryᵢ) / temperature)",
    handoff: "The main idea survives every detail: lift uncertain evidence, splat it into shared metric space.",
    evidence: "PAPER", source: "Paper Tables 1–5 · Eq. 2 · Sec. 5", illustration: "shoot", lab: "robustness"
  }
];

export function sceneIndexFromHash(hash: string) {
  const id = hash.replace(/^#/, "");
  const index = SCENES.findIndex((scene) => scene.id === id);
  return index < 0 ? 0 : index;
}
