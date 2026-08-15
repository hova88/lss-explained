export type Evidence = "PAPER" | "OFFICIAL CODE" | "REAL SAMPLE" | "CHECKPOINT" | "TEACHING";
export type IllustrationKind = "overview" | "features" | "ray" | "depth" | "context" | "lift" | "geometry" | "rig" | "splat" | "learning" | "truth" | "shoot";
export type LabId = "geometry" | "bev" | "robustness";
export type NarrativeStep = { label:string; text:string };
export type TensorTransition = { input:string; operation:string; output:string; detail:string };

export type NarrativeScene = {
  id:string; act:string; title:string; question:string; reveal:string; explanation:string;
  steps:[NarrativeStep,NarrativeStep,NarrativeStep]; tensor:TensorTransition; formula:string; handoff:string;
  evidence:Evidence; source:string; illustration:IllustrationKind; lab?:LabId;
};

export const SCENES:NarrativeScene[] = [
  {
    id:"system-view",act:"I · THE ONE QUESTION",title:"How do six images become one BEV?",
    question:"What is the single problem LSS must solve?",
    reveal:"Give every 2D image feature a plausible depth, place it in shared ego 3D, then collapse it into BEV.",
    explanation:"An image says what a pixel looks like, but not where it lies in meters. LSS keeps that missing depth explicit. Lift creates depth-weighted 3D candidates; geometry moves them into the vehicle frame; Splat groups them on the ground plane.",
    steps:[{label:"ADD DEPTH",text:"Turn each feature pixel into several weighted positions on its camera ray."},{label:"SHARE SPACE",text:"Use calibration so every camera speaks ego coordinates."},{label:"POOL",text:"Sum candidates that land in the same metric BEV cell."}],
    tensor:{input:"[B,6,3,128,352]",operation:"LIFT → TRANSFORM → SPLAT",output:"[B,64,200,200]",detail:"images → shared metric feature map"},
    formula:"2D feature + latent depth + calibration → ego-space BEV",
    handoff:"First prove why an ordinary image feature is not already a BEV feature.",
    evidence:"PAPER",source:"Paper Fig. 1, Fig. 4 · official forward()",illustration:"overview"
  },
  {
    id:"image-features",act:"II · PREPROCESS, THEN SEE",title:"Images become perspective features",
    question:"What changes before geometry—and what is still missing?",
    reveal:"Resize and crop prepare the camera images; the shared encoder learns what appears, but still not where it lies in meters.",
    explanation:"Each 1600×900 JPEG is resized, cropped to 352×128 and normalized. The exact image warp is saved as post_rot and post_trans so geometry can undo it later. Six cameras are then folded into the batch, passed through one shared EfficientNet, and reduced to an 8×22 field with 105 channels per anchor.",
    steps:[{label:"PREPROCESS",text:"1600×900 → resize → crop → normalize → 352×128."},{label:"RESHAPE + ENCODE",text:"[B,6,3,128,352] becomes [B·6,3,128,352], then [B·6,105,8,22]."},{label:"REMEMBER WARP",text:"post_rot and post_trans preserve how network pixels relate to calibrated pixels."}],
    tensor:{input:"[B,6,3,900,1600]",operation:"resize/crop → normalize → fold → CamEncode",output:"[B·6,105,8,22]",detail:"network input is [B,6,3,128,352]; 105 = 41 + 64"},
    formula:"perspective feature ≠ BEV feature",
    handoff:"One anchor may say vehicle. Why can it not name one 3D point?",
    evidence:"OFFICIAL CODE",source:"models.py CamEncode · get_cam_feats()",illustration:"features",lab:"geometry"
  },
  {
    id:"pixel-ray",act:"III · THE MISSING DIMENSION",title:"One pixel means one ray",
    question:"Why can calibration not map [u,v] directly to one 3D position?",
    reveal:"Many points at different depths project to the same pixel.",
    explanation:"K⁻¹ recovers a direction, not a distance. A vehicle feature at 10 m, 25 m or 40 m can occupy the same image anchor. A unique 3D point therefore needs three coordinates: image position [u,v] and depth d.",
    steps:[{label:"PIXEL",text:"[u,v] identifies one direction from the optical center."},{label:"AMBIGUITY",text:"Every positive d chooses another point on the same ray."},{label:"REQUIREMENT",text:"Camera-to-BEV must represent [u,v,d], not [u,v] alone."}],
    tensor:{input:"anchor [u′,v′]",operation:"undo post-transform + K⁻¹",output:"camera ray r(u,v)",detail:"direction is known; metric depth is not"},
    formula:"p_cam(d)=K⁻¹[d·u,d·v,d]ᵀ",
    handoff:"LSS refuses to guess one hard distance. It builds a distribution.",
    evidence:"PAPER",source:"Paper Sec. 3.1 · models.py create_frustum()",illustration:"ray",lab:"geometry"
  },
  {
    id:"depth-distribution",act:"IV · WHERE?",title:"DepthNet keeps 41 hypotheses",
    question:"What exactly is the depth tensor?",
    reveal:"For every camera and every 8×22 anchor, it stores 41 weights from 4 m through 44 m.",
    explanation:"Softmax turns the first 41 output channels into a categorical allocation α(d). Think of the 2D feature map as one sheet, then stack 41 depth sheets behind it. The stack says where evidence may live, not what the evidence means.",
    steps:[{label:"SPLIT",text:"Take channels 0…40 from the 105-channel head."},{label:"SOFTMAX",text:"Normalize along D so Σd α(d)=1 at every anchor."},{label:"UNFOLD",text:"Expose a new D axis: one image anchor becomes 41 candidates."}],
    tensor:{input:"[B,6,41,8,22] logits",operation:"softmax(dim=D)",output:"α: [B,6,41,8,22]",detail:"41 depth sheets; each anchor sums to one"},
    formula:"α(d|u,v)=softmax(depth_logits)(d)",
    handoff:"Depth answers where. A different tensor must answer what.",
    evidence:"CHECKPOINT",source:"model525000.pt · get_depth_feat()",illustration:"depth",lab:"geometry"
  },
  {
    id:"context-feature",act:"IV · WHAT?",title:"Context carries the thing itself",
    question:"Why is depth probability not enough?",
    reveal:"Depth says where to place evidence; a 64D context vector says what evidence to place.",
    explanation:"The remaining 64 channels encode appearance and task-relevant semantics at each anchor. Depth and context are deliberately separate: one is a scalar distribution over positions, the other is the feature payload that the BEV network will later interpret.",
    steps:[{label:"DEPTH HEAD",text:"41 values answer where along the ray."},{label:"CONTEXT HEAD",text:"64 values describe the visual evidence at the anchor."},{label:"KEEP AXES",text:"Context has no D axis yet; it is one vector per [camera,v,u]."}],
    tensor:{input:"[B,6,105,8,22]",operation:"channel split",output:"depth 41 ⊕ context 64",detail:"where? [41] · what? [64]"},
    formula:"c(u,v)∈ℝ⁶⁴",
    handoff:"Now combine where and what without choosing one depth.",
    evidence:"OFFICIAL CODE",source:"models.py get_depth_feat()",illustration:"context",lab:"geometry"
  },
  {
    id:"outer-product",act:"V · LIFT",title:"Outer product performs the Lift",
    question:"What does α(d)×c actually compute?",
    reveal:"It copies the same 64D context to all 41 depths and scales each copy by that depth’s weight.",
    explanation:"Unsqueeze depth with a trailing channel axis and context with a leading depth axis. Broadcasting multiplies [D,1] by [1,C]. No loop or hard argmax is needed: one anchor becomes a [41,64] depth-feature slab, and all anchors form a frustum feature volume.",
    steps:[{label:"UNSQUEEZE",text:"α → [B,6,41,8,22,1]."},{label:"ALIGN",text:"context → [B,6,1,8,22,64]."},{label:"MULTIPLY",text:"Broadcast to [B,6,41,8,22,64]."}],
    tensor:{input:"α[...,D,H,W,1] × c[...,1,H,W,C]",operation:"broadcast multiply",output:"[B,6,41,8,22,64]",detail:"43,296 candidates · each carries 64 values"},
    formula:"F_frustum(d,u,v,c)=α(d|u,v)·c(u,v,c)",
    handoff:"The feature now has a D axis, but [u,v,d] is still not an XYZ coordinate.",
    evidence:"OFFICIAL CODE",source:"models.py get_cam_feats()",illustration:"lift",lab:"geometry"
  },
  {
    id:"intrinsic-unprojection",act:"VI · PIXEL TO CAMERA 3D",title:"Intrinsic turns direction plus depth into XYZ",
    question:"How does [u,v,d] become a metric camera point?",
    reveal:"K⁻¹ supplies direction; multiplying by d supplies distance.",
    explanation:"First invert resize and crop with post_rot and post_trans. Then form [du,dv,d] and apply K⁻¹. The output is measured in the selected camera frame: +x right, +y down, +z optical-forward.",
    steps:[{label:"UNDO IMAGE",text:"A⁻¹([u′,v′]−a) restores the raw calibrated pixel."},{label:"SCALE",text:"[u,v,1] becomes [du,dv,d]."},{label:"UNPROJECT",text:"K⁻¹ returns [Xc,Yc,Zc] in camera meters."}],
    tensor:{input:"geometry [B,6,41,8,22,3]",operation:"A⁻¹ then K⁻¹",output:"p_cam [B,6,41,8,22,3]",detail:"same shape; coordinate meaning changes"},
    formula:"p_cam=K⁻¹[d·A⁻¹(u′−a),d]ᵀ",
    handoff:"Six camera frames still cannot pool together. Move them into ego.",
    evidence:"REAL SAMPLE",source:"models.py get_geometry() · golden projection points",illustration:"geometry",lab:"geometry"
  },
  {
    id:"extrinsic-fusion",act:"VII · CAMERA TO EGO",title:"Six cameras meet in one coordinate system",
    question:"Where does multi-camera fusion really happen?",
    reveal:"Each camera is transformed separately; fusion happens only after all candidates speak ego XYZ.",
    explanation:"R rotates camera axes into ego axes and t places the optical center. Images are never stitched. If cameras and calibration are permuted together, the pooled result is unchanged because the later operation cares only about ego position and feature value.",
    steps:[{label:"ROTATE",text:"Rcam→ego changes the coordinate basis."},{label:"TRANSLATE",text:"+tcam→ego positions the camera on the vehicle."},{label:"CONCATENATE",text:"Flatten camera, depth, row and column into one candidate list."}],
    tensor:{input:"6 × [41,8,22,3]",operation:"R·p_cam+t; flatten N·D·H·W",output:"[B,43,296,3] + [B,43,296,64]",detail:"geometry and features share candidate index"},
    formula:"p_ego=Rcam→ego·p_cam+tcam→ego",
    handoff:"The list now shares meters, but a CNN needs a dense grid.",
    evidence:"PAPER",source:"Paper Sec. 3.2 · get_geometry()",illustration:"rig",lab:"geometry"
  },
  {
    id:"splat",act:"VIII · SPLAT",title:"Voxel pooling rebuilds a BEV tensor",
    question:"How do 43,296 irregular candidates become indexed BEV cells?",
    reveal:"Filter, floor-quantize, sort by voxel rank, sum collisions, then scatter.",
    explanation:"Each ego point receives integer [ix,iy,iz]. Equal voxel ranks become adjacent after sorting. QuickCumsum performs the same grouped sum as a naïve loop, then the compact groups are scattered into [B,C,Z,X,Y].",
    steps:[{label:"INDEX",text:"floor((p−origin)/Δ) maps meters to voxel indices."},{label:"REDUCE",text:"sort + QuickCumsum sum every 64D collision."},{label:"SCATTER",text:"Write each surviving group into its spatial tensor address."}],
    tensor:{input:"features [B,43,296,64] + XYZ",operation:"mask → floor → rank → sort → Σ → scatter",output:"[B,64,1,200,200]",detail:"irregular list becomes dense spatial memory"},
    formula:"Fvoxel[ix,iy,iz]=Σi∈voxel F_i",
    handoff:"The original configuration has one z bin. Collapse it to BEV.",
    evidence:"OFFICIAL CODE",source:"voxel_pooling() · QuickCumsum",illustration:"splat",lab:"bev"
  },
  {
    id:"bev-encoder",act:"IX · REASON IN BEV",title:"Collapse height, then learn spatial context",
    question:"When does the tensor finally become a camera BEV feature?",
    reveal:"After z is folded into channels, BevEncode operates on a regular [B,64,200,200] ground map.",
    explanation:"The geometric job ends at voxel pooling. BevEncode then uses a ResNet-18-style multiscale CNN to interpret neighborhoods and emit one vehicle logit per cell. BCE supervision travels backward through the BEV CNN, pooling and outer product into latent depth.",
    steps:[{label:"COLLAPSE Z",text:"Concatenate Z into channels; here Z=1, so C stays 64."},{label:"ENCODE",text:"Multiscale 2D convolution adds BEV context."},{label:"SUPERVISE",text:"BCEWithLogits compares [B,1,200,200] with the vehicle mask."}],
    tensor:{input:"[B,64,1,200,200]",operation:"collapse Z → BevEncode",output:"logits [B,1,200,200]",detail:"geometry ends; task learning begins"},
    formula:"L=BCEWithLogits(logits,vehicle_mask)",
    handoff:"Now audit one real output without expecting it to resemble a camera image.",
    evidence:"OFFICIAL CODE",source:"models.py BevEncode · train.py SimpleLoss",illustration:"learning",lab:"bev"
  },
  {
    id:"truth-lab",act:"X · TRACE THE WHOLE TENSOR",title:"Read one real frame end to end",
    question:"How do we verify the tensor orientation and output meaning?",
    reveal:"Use fixed ego axes, nuScenes GT and reference LiDAR to audit the map—not image-shaped similarity.",
    explanation:"At threshold 0.5 this pinned frame yields TP 282, FP 122, FN 120 and IoU 0.538. The LiDAR scan never enters inference; it exposes flips, offsets and object locations. The checkpoint output is semantic vehicle occupancy, not boxes or tracks.",
    steps:[{label:"DECODE",text:"logit → sigmoid → optional threshold."},{label:"ORIENT",text:"ego +x is screen up; ego +y is screen left."},{label:"VERIFY",text:"Linked GT and reference points check the same metric cells."}],
    tensor:{input:"logits [1,1,200,200]",operation:"sigmoid + threshold + compare GT",output:"TP 282 · FP 122 · FN 120 · IoU .538",detail:"single-frame diagnostic, not paper validation IoU"},
    formula:"ego [x,y] ↔ BEV [⌊(x+50)/.5⌋,⌊(y+50)/.5⌋]",
    handoff:"Finish by separating the core mechanism from paper results and Shoot.",
    evidence:"CHECKPOINT",source:"model525000.pt · nuScenes GT · reference LiDAR",illustration:"truth",lab:"bev"
  },
  {
    id:"evidence-action",act:"XI · REMEMBER THE CORE",title:"One sentence should survive",
    question:"What should remain after every shape and matrix is forgotten?",
    reveal:"LSS adds latent depth to 2D features, returns them to shared ego 3D, then pools them into BEV.",
    explanation:"The paper reports Lift-Splat vehicle IoU 32.07 versus OFT 30.05 and CNN 24.25, and studies missing cameras and calibration noise. Shoot consumes BEV costs for trajectory scoring, but the released segmentation checkpoint stops at vehicle logits.",
    steps:[{label:"LIFT",text:"where distribution × what feature."},{label:"SPLAT",text:"camera XYZ → ego XYZ → indexed BEV sums."},{label:"SHOOT",text:"optional downstream planning use, reconstructed here from the equation."}],
    tensor:{input:"images",operation:"feature → depth×context → geometry → pooling",output:"task-ready BEV",detail:"the complete computation in one line"},
    formula:"p(τi|o)∝exp(−cost(τi)/T)",
    handoff:"If you can explain each arrow and each shape, you understand original LSS.",
    evidence:"PAPER",source:"Paper Tables 1–5 · Eq. 2 · Sec. 5",illustration:"shoot",lab:"robustness"
  }
];

export function sceneIndexFromHash(hash:string){const id=hash.replace(/^#/,"");const index=SCENES.findIndex(scene=>scene.id===id);return index<0?0:index;}
