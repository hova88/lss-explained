export type Evidence = "PAPER" | "OFFICIAL CODE" | "REAL SAMPLE" | "CHECKPOINT" | "TEACHING";
export type IllustrationKind = "overview" | "sample" | "rig" | "features" | "ray" | "lift" | "geometry" | "splat" | "bev" | "learning" | "truth" | "shoot";
export type StageView = "illustration" | "rig3d" | "bev" | "linked";
export type LabId = "geometry" | "bev" | "robustness";

export type NarrativeStep = { label: string; text: string };

export type NarrativeScene = {
  id: string;
  act: string;
  title: string;
  question: string;
  reveal: string;
  explanation: [string, string];
  steps: [NarrativeStep, NarrativeStep, NarrativeStep];
  formula: string;
  handoff: string;
  evidence: Evidence;
  source: string;
  illustration: IllustrationKind;
  stageView: StageView;
  lab?: LabId;
};

export const SCENES: NarrativeScene[] = [
  {
    id: "system-view", act: "I · FRAME THE TASK", title: "First, see the whole machine",
    question: "What job is LSS hired to do, and what is the exact contract between input and output?",
    reveal: "LSS converts several calibrated perspective images into one metric, ego-centric bird’s-eye-view representation.",
    explanation: [
      "A planner reasons in meters around the vehicle: which ground cell is occupied, where a route may pass, and how evidence from different directions agrees. A camera image is rich in appearance but poor at expressing that shared metric layout.",
      "The model therefore has three explicit stages. Lift attaches learned depth hypotheses to image features. Splat moves and sums those hypotheses into ego-space cells. Shoot is an optional planning use of the resulting BEV; it is not part of the released segmentation checkpoint."
    ],
    steps: [
      { label: "INPUT", text: "N RGB images, one intrinsic matrix K per camera, and camera→ego rotation R and translation t." },
      { label: "CORE", text: "Shared image encoding → latent-depth Lift → exact geometry → pillar sum pooling → BEV CNN." },
      { label: "OUTPUT", text: "For the released model, one raw vehicle logit for every 0.5 m cell in a 100 m × 100 m ego grid." }
    ],
    formula: "{Iₙ,Kₙ,Rₙ,tₙ}ₙ₌₁ᴺ → Lift → Splat → BEV logits [B,1,200,200]",
    handoff: "The first unresolved question is why calibrated images cannot simply be stitched together.",
    evidence: "PAPER", source: "Paper Fig. 1 and Fig. 4 · models.py forward()", illustration: "overview", stageView: "illustration"
  },
  {
    id: "perspective-gap", act: "I · FRAME THE TASK", title: "Perspective is not a common ground",
    question: "Why does a useful BEV require more than arranging six photographs around a car?",
    reveal: "The same image displacement can describe a nearby small motion or a distant large motion; pixels from different cameras do not share a metric coordinate system.",
    explanation: [
      "A pixel tells us a viewing direction. It does not tell us where along that direction the visible surface lies. Without depth, two cameras cannot know which of their pixels refer to the same patch of road or vehicle.",
      "LSS does not force a single depth estimate before the task is understood. It preserves multiple depth candidates, lets BEV supervision decide how to weight them, and only then aggregates all cameras in ego space."
    ],
    steps: [
      { label: "CAMERA PLANE", text: "Each view has its own origin, focal length, principal point and perspective distortion." },
      { label: "MISSING VARIABLE", text: "K⁻¹ turns a pixel into a ray, but metric depth d is still unknown." },
      { label: "DESIGN CHOICE", text: "Represent the uncertainty explicitly, then pool evidence where candidate 3D points land." }
    ],
    formula: "pixel [u,v] + calibration K⁻¹ ⇒ ray; ray + depth d ⇒ camera-frame point",
    handoff: "Before lifting anything, we need an exact ledger of what one training sample contains.",
    evidence: "PAPER", source: "Paper Sec. 3.1 · arbitrary camera rigs", illustration: "sample", stageView: "illustration"
  },
  {
    id: "data-contract", act: "II · PREPARE EVIDENCE", title: "One sample, four different roles",
    question: "Which data enters the network, which data defines geometry, and which data exists only for teaching or supervision?",
    reveal: "Images carry appearance, calibration carries geometry, vehicle annotations create the training target, and LiDAR remains an external reference in this essay.",
    explanation: [
      "For the pinned nuScenes timestamp, the loader reads six JPEGs and their calibrated sensor records. It also reads ego poses so measurements taken at slightly different sensor timestamps can be related correctly.",
      "The official segmentation loader rasterizes vehicle boxes into a binary 200 × 200 target. The 34,688 LiDAR points shown later never enter checkpoint inference; they let us audit orientation, object position and camera projection."
    ],
    steps: [
      { label: "MODEL INPUT", text: "Six normalized images plus rots, trans, intrins, post_rots and post_trans." },
      { label: "SUPERVISION", text: "nuScenes vehicle boxes moved into the reference ego frame and filled into a BEV mask." },
      { label: "REFERENCE ONLY", text: "One LiDAR scan and 3D boxes for linked visual verification, never fused into LSS." }
    ],
    formula: "sample → (images, calibration, post-transform, BEV target)  +  reference LiDAR",
    handoff: "Now follow only the model input: first the images are made batchable without losing geometry.",
    evidence: "REAL SAMPLE", source: "data.py · SegmentationData · pinned nuScenes sample", illustration: "rig", stageView: "rig3d", lab: "geometry"
  },
  {
    id: "image-encoding", act: "II · PREPARE EVIDENCE", title: "Preserve geometry while compressing appearance",
    question: "How does a 1600 × 900 photograph become an 8 × 22 feature field without forgetting where its pixels came from?",
    reveal: "The image is resized and cropped for the network, while post_rot and post_trans record the exact raw→network pixel transform.",
    explanation: [
      "In the pinned configuration, the raw image is resized to 352 × 198 and vertically cropped to 352 × 128. RGB values are normalized and passed through the same EfficientNet encoder for every camera.",
      "The feature grid is not an arbitrary 16-pixel tiling. create_frustum uses 22 linspace anchors from 0 to 351 and 8 anchors from 0 to 127, so geometry later samples the precise network-image coordinates represented by each feature cell."
    ],
    steps: [
      { label: "AUGMENT", text: "Apply resize/crop; accumulate its 2D linear part A and translation a." },
      { label: "ENCODE", text: "Shared CamEncode emits 105 channels at each of 8 × 22 locations." },
      { label: "SPLIT", text: "The first 41 channels are depth logits; the remaining 64 are the context feature." }
    ],
    formula: "[B,N,3,128,352] → CamEncode → [B,N,105,8,22] = 41 depth + 64 context",
    handoff: "One feature anchor is still attached to an image plane. The next step turns it into a 3D direction.",
    evidence: "OFFICIAL CODE", source: "tools.py img_transform() · models.py CamEncode/create_frustum", illustration: "features", stageView: "illustration", lab: "geometry"
  },
  {
    id: "pixel-ray", act: "III · LIFT", title: "Undo the image, then open a ray",
    question: "What does K⁻¹ actually do, and why is the result still not a 3D location?",
    reveal: "After undoing resize and crop, K⁻¹ converts a raw pixel into a direction in the camera coordinate frame.",
    explanation: [
      "The camera frame uses +x right, +y down and +z forward. For pixel [u,v], subtracting the principal point and dividing by focal length gives x/z and y/z. Multiplying by a chosen z-depth d yields a metric camera point.",
      "Because one RGB observation does not supply d, LSS instantiates 41 candidates from 4 m through 44 m. They all lie on the same ray and differ only in metric distance from the optical center."
    ],
    steps: [
      { label: "UNDO A,a", text: "Map network anchor [u′,v′] back to the raw pixel coordinate in which K was calibrated." },
      { label: "UNPROJECT", text: "Apply K⁻¹ to [d·u,d·v,d]ᵀ; this is a point in camera meters, not ego meters." },
      { label: "ENUMERATE", text: "Repeat for d ∈ {4,5,…,44}; geometry is known, confidence is not." }
    ],
    formula: "[u,v]ᵀ=A⁻¹([u′,v′]ᵀ−a),  p_cam(d)=K⁻¹[d·u,d·v,d]ᵀ",
    handoff: "Geometry supplies 41 possible places. The network must now decide how much feature evidence each place receives.",
    evidence: "OFFICIAL CODE", source: "models.py create_frustum() · get_geometry()", illustration: "ray", stageView: "rig3d", lab: "geometry"
  },
  {
    id: "lift", act: "III · LIFT", title: "Lift without a depth label",
    question: "How can a model create a 3D feature volume when no LiDAR depth supervises the camera encoder?",
    reveal: "A softmax allocates one 64D context vector across the 41 depth candidates using an outer product.",
    explanation: [
      "At one anchor, α(d)=softmax(depth_logits) sums to one. The same context vector c is copied along the ray and scaled: α(d)c. Across all anchors this produces a frustum-shaped feature volume per camera.",
      "The allocation is latent and task-oriented. A broad or multimodal α can still help vehicle segmentation, and its values must not automatically be interpreted as calibrated physical depth probabilities."
    ],
    steps: [
      { label: "DEPTH HEAD", text: "41 logits become non-negative weights α(d) that sum to one." },
      { label: "CONTEXT HEAD", text: "64 channels describe the semantic evidence at that feature anchor." },
      { label: "OUTER PRODUCT", text: "α[:,None] · c[None,:] creates 41 candidate features without choosing one hard depth." }
    ],
    formula: "f[d,c] = softmax(ℓdepth)[d] · context[c]   →   [41,64] per anchor",
    handoff: "The volume is still expressed in six different camera frames. All candidates must meet in one ego frame.",
    evidence: "CHECKPOINT", source: "models.py get_depth_feat() · exported [6,41,8,22] depth", illustration: "lift", stageView: "rig3d", lab: "geometry"
  },
  {
    id: "exact-geometry", act: "IV · SHARE ONE FRAME", title: "The complete coordinate chain",
    question: "In what order must image and rigid transforms be applied for one lifted candidate to land correctly?",
    reveal: "Undo image augmentation, unproject with K⁻¹, rotate camera axes into ego axes, then translate by the real optical center.",
    explanation: [
      "With column vectors, the operation applied first appears on the right. R changes the basis of the camera-frame vector; t is added afterward because it locates the camera origin in ego coordinates. Swapping these operations changes the point.",
      "The matrix values come from nuScenes calibrated_sensor quaternion and translation records. A quaternion becomes an orthonormal rotation matrix; the inverse rigid transform is [Rᵀ,−Rᵀt]."
    ],
    steps: [
      { label: "IMAGE → CAMERA", text: "A⁻¹ and a undo preprocessing; K⁻¹ turns the restored pixel and d into p_cam." },
      { label: "CAMERA → EGO", text: "p_ego=R_cam→ego p_cam+t_cam→ego, using the selected camera’s calibration." },
      { label: "CHECK", text: "RᵀR≈I, det(R)≈1, T⁻¹T≈I and camera→pixel round trips expose transpose or sign errors." }
    ],
    formula: "pₑ = Rcam→ego K⁻¹[d·A⁻¹(u′−a), d]ᵀ + tcam→ego",
    handoff: "Once every camera speaks ego coordinates, camera identity is no longer needed for aggregation.",
    evidence: "REAL SAMPLE", source: "models.py get_geometry() · 18 exported golden points", illustration: "geometry", stageView: "rig3d", lab: "geometry"
  },
  {
    id: "camera-fusion", act: "IV · SHARE ONE FRAME", title: "Six cameras become one unordered set",
    question: "How do different viewpoints contribute to the same metric place without stitching their image planes?",
    reveal: "Each camera independently emits ego-frame candidates; points that fall into the same BEV pillar are later summed.",
    explanation: [
      "The real frustums are determined by K⁻¹, the four image corners and cam2ego. Their optical axes are the third columns of the cam2ego rotations, so every lens points away from the vehicle rather than toward it.",
      "If images and their calibration are permuted together, the final sum is unchanged. If the whole rig and scene undergo the same rigid ego transform, the representation moves with them. These are the useful inductive biases behind an arbitrary camera rig."
    ],
    steps: [
      { label: "INDEPENDENT", text: "The shared image encoder processes each view with identical weights." },
      { label: "CALIBRATED", text: "Each feature candidate uses its own K, post-transform and camera→ego pose." },
      { label: "ORDER-FREE", text: "After coordinates are expressed in ego, sum pooling depends on location and feature value, not camera order." }
    ],
    formula: "pool({Tcamₙ→ego · Lift(Iₙ)}ₙ₌₁ᴺ) = pool(any permutation of the calibrated set)",
    handoff: "The candidates now share a frame but remain an irregular point set. Splat makes them a dense tensor.",
    evidence: "PAPER", source: "Paper Sec. 3.2 and 5.5 · real six-camera calibration", illustration: "rig", stageView: "rig3d", lab: "geometry"
  },
  {
    id: "splat", act: "V · SPLAT", title: "Turn irregular 3D evidence into BEV cells",
    question: "How do 43,296 lifted candidates become a regular tensor without losing contributions that collide?",
    reveal: "Splat filters the metric range, quantizes points into voxels, sorts by voxel rank and sums every feature in each pillar.",
    explanation: [
      "The x and y grid covers the half-open interval [−50,50) at 0.5 m resolution. The published configuration uses one vertical voxel from −10 m to 10 m, so height is pooled into a pillar rather than preserved as many z layers.",
      "QuickCumsum is an efficient grouping trick, not a different aggregation rule. Prefix sums followed by group-end selection and neighboring differences produce exactly the same result as a naïve sum per voxel."
    ],
    steps: [
      { label: "FILTER", text: "Discard candidates outside the half-open x/y/z bounds; boundary points at +50 m do not enter." },
      { label: "GROUP", text: "floor((p−origin)/Δ) gives [ix,iy,iz]; a rank makes equal voxel indices adjacent after sorting." },
      { label: "SUM", text: "Add all 64D features with the same rank, then place the group result in [B,C,Z,X,Y]." }
    ],
    formula: "voxel(p)=floor((p−(bound_min−Δ/2))/Δ),   Fvoxel=Σᵢ fᵢ",
    handoff: "Geometry has ended. The tensor is spatially meaningful, but local evidence still needs BEV context.",
    evidence: "OFFICIAL CODE", source: "models.py voxel_pooling() · tools.py QuickCumsum", illustration: "splat", stageView: "bev", lab: "bev"
  },
  {
    id: "learn-in-bev", act: "VI · REASON AND LEARN", title: "Geometry places evidence; learning interprets it",
    question: "What does the BEV encoder add, and how can a final mask supervise latent depth all the way upstream?",
    reveal: "After z collapse, a multiscale BEV CNN turns pooled features into logits; BCE gradients pass backward through every differentiable stage of Lift and Splat.",
    explanation: [
      "The pooled tensor becomes [B,64,200,200]. A ResNet-18-style trunk builds a wide receptive field, upsamples a deep feature map, fuses it with an earlier map and emits one raw vehicle logit per cell.",
      "Training compares those logits with the rasterized vehicle mask using BCEWithLogitsLoss. When a cell is wrong, gradients reach the context feature and every depth weight that contributed to that cell; there is no separate depth target in original LSS."
    ],
    steps: [
      { label: "ENCODE", text: "[B,64,200,200] → multiscale BevEncode → [B,1,200,200] raw logits." },
      { label: "SUPERVISE", text: "Binary BEV vehicle occupancy is the only target used by this released segmentation task." },
      { label: "BACKPROP", text: "BCE → BEV CNN → sum pooling → α(d)c → image encoder; latent depth is learned indirectly." }
    ],
    formula: "L=BCEWithLogits(ŷBEV,yBEV),   ∂L/∂α(d) flows through Σvoxel α(d)c",
    handoff: "Training explains how the representation forms. Inference must still be interpreted without inventing extra heads.",
    evidence: "OFFICIAL CODE", source: "models.py BevEncode · train.py SimpleLoss", illustration: "learning", stageView: "bev", lab: "bev"
  },
  {
    id: "truth-lab", act: "VII · VERIFY INFERENCE", title: "Read the output in the right coordinate system",
    question: "Why should a BEV heatmap not resemble a camera image, and how can we prove it is not mirrored or transposed?",
    reveal: "Inference ends at logits, sigmoid probabilities and an optional threshold; linked LiDAR, GT and ego coordinates audit the display mapping.",
    explanation: [
      "A camera silhouette is destroyed on purpose: each feature spreads along depth, rotates into ego, mixes with other cameras and is interpreted by a BEV CNN. Correct correspondence is object-level and metric, not pixel-shape similarity.",
      "At threshold 0.5 the pinned frame yields TP 282, FP 122, FN 120 and IoU 0.538. This is a single-frame diagnostic, not the paper’s validation-set score. The model itself has no box decoder, NMS, tracker or velocity head."
    ],
    steps: [
      { label: "INFER", text: "logit → sigmoid probability; official IoU uses logit>0, equivalent to probability>0.5." },
      { label: "MAP", text: "ego +x is screen up, ego +y is screen left; ix indexes forward and iy indexes left." },
      { label: "AUDIT", text: "Select one LiDAR point, GT box or BEV cell and retain the same identity across image, 3D and BEV." }
    ],
    formula: "ego [x,y] ↔ BEV [⌊(x+50)/.5⌋,⌊(y+50)/.5⌋] ↔ screen [left,up]",
    handoff: "With one frame understood, the final question is what the paper establishes—and what it does not.",
    evidence: "CHECKPOINT", source: "model525000.pt export · alignment diagnostics · reference LiDAR", illustration: "truth", stageView: "linked", lab: "bev"
  },
  {
    id: "evidence-action", act: "VIII · PLACE THE CLAIMS", title: "Results, robustness, Shoot—and the boundary",
    question: "Which conclusions belong to the ECCV 2020 evidence, and which parts of this laboratory are reconstructions?",
    reveal: "LSS shows that explicit geometry plus learned latent depth outperforms image-only and OFT baselines, while robustness still depends on training conditions.",
    explanation: [
      "For nuScenes vehicle segmentation, the paper reports IoU 24.25 for CNN, 30.05 for OFT and 32.07 for Lift-Splat. It also studies camera dropout, calibration noise, unseen camera arrangements and oracle depth; explicit geometry is helpful but not automatic immunity.",
      "Shoot learns a BEV cost field and converts the costs of 1,000 five-second trajectory templates into a Boltzmann distribution. No planning checkpoint was released, so this site’s trajectory lab is equation-level teaching reconstruction."
    ],
    steps: [
      { label: "WHAT HOLDS", text: "Camera-order invariance, ego-frame equivariance and strong segmentation results for the tested setup." },
      { label: "WHAT BREAKS", text: "Night, distance, occlusion, wrong calibration and missing views still degrade the representation." },
      { label: "WHAT FOLLOWS", text: "LSS influenced later camera-BEV work, but BEVDet/BEVDepth and temporal models are outside this site’s scope." }
    ],
    formula: "p(τᵢ|o)=exp(−cost(τᵢ)/T) / Σⱼ exp(−cost(τⱼ)/T)",
    handoff: "Return to any stage knowing exactly which coordinate frame, tensor and evidence source is on the page.",
    evidence: "PAPER", source: "Paper Tables 1–5 · Eq. 2 · Sec. 5", illustration: "shoot", stageView: "linked", lab: "robustness"
  }
];

export function sceneIndexFromHash(hash: string) {
  const id = hash.replace(/^#/, "");
  const index = SCENES.findIndex((scene) => scene.id === id);
  return index < 0 ? 0 : index;
}
