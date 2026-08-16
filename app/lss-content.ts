export type Evidence = "LSS PAPER" | "LSS CODE" | "BEVDEPTH PAPER" | "BEVDEPTH CODE" | "REAL SAMPLE" | "CHECKPOINT" | "TEACHING";
export type IllustrationKind = "overview" | "features" | "ray" | "depth" | "context" | "lift" | "image-ray" | "camera-point" | "ego-transform" | "rig" | "splat" | "bev-encoder" | "learning" | "truth";
export type LabId = "geometry" | "bev" | "robustness";
export type NarrativeStep = { label:string; text:string };
export type TensorTransition = { input:string; operation:string; output:string; detail:string };
export type MethodContrast = { lss:string; bevdepth:string };

export type NarrativeScene = {
  id:string; act:string; title:string; question:string; reveal:string; explanation:string;
  steps:[NarrativeStep,NarrativeStep,NarrativeStep]; tensor:TensorTransition; formula:string; handoff:string;
  evidence:Evidence; source:string; illustration:IllustrationKind; lab?:LabId; comparison?:MethodContrast;
};

export const SCENES:NarrativeScene[] = [
  {
    id:"ray-evidence",act:"01 · ONE RAY",title:"One ray, two predictions",
    question:"What does CamEncode attach to one image location?",
    reveal:"Depth says where. Context says what.",
    explanation:"Click the real image. Its nearest feature anchor owns one 41-bin depth allocation and one 64-channel context vector.",
    steps:[{label:"CLICK",text:"Choose one raw-image pixel."},{label:"ALIGN",text:"Apply resize/crop and select the nearest 8×22 anchor."},{label:"READ",text:"Inspect depth and context from the same checkpoint cell."}],
    tensor:{input:"image [B,6,3,128,352]",operation:"CamEncode → split 41 | 64",output:"D [B,6,41,8,22] + C [B,6,64,8,22]",detail:"one ray anchor · two different meanings"},
    formula:"F(d,c|u,v)=softmax(D)(d|u,v) · C(c|u,v)",
    handoff:"Lift gives every depth hypothesis a feature; calibration gives it a place.",
    evidence:"CHECKPOINT",source:"model525000.pt · CamEncode.get_depth_feat()",illustration:"overview"
  },
  {
    id:"lift-geometry",act:"02 · LIFT + GEOMETRY",title:"Give every hypothesis a place",
    question:"How does one feature anchor become 41 ego-frame candidates?",
    reveal:"Lift copies the context across depth; calibration turns every copy into ego meters.",
    explanation:"Follow one unchanged sample through image warp, intrinsics, metric depth and camera-to-ego extrinsics.",
    steps:[{label:"LIFT",text:"α(d)c creates one 64D feature per depth bin."},{label:"UNPROJECT",text:"A⁻¹ and K⁻¹ turn the network anchor into a metric camera point."},{label:"ALIGN",text:"R and t express all cameras in the ego frame."}],
    tensor:{input:"D [41] + C [64] + [u′,v′,d]",operation:"outer product + calibrated unprojection",output:"XYZego [41,3] + F [41,64]",detail:"features and coordinates keep the same depth index"},
    formula:"pₑ(d)=R K⁻¹[d·A⁻¹(p′−a), d]ᵀ+t",
    handoff:"The candidates are metric but irregular. Splat makes a grid.",
    evidence:"LSS CODE",source:"get_cam_feats() · get_geometry() · real nuScenes K,R,t",illustration:"image-ray",lab:"geometry"
  },
  {
    id:"splat-pooling",act:"03 · SPLAT",title:"Resolve collisions",
    question:"What survives when several candidates enter one cell?",
    reveal:"LSS floors, groups and sums.",
    explanation:"Move one candidate. Compare sum, mean, max and bilinear splatting without changing its feature value.",
    steps:[{label:"INDEX",text:"Continuous ego XYZ → integer voxel."},{label:"GROUP",text:"Equal ranks become contiguous."},{label:"REDUCE",text:"QuickCumsum returns the grouped sum."}],
    tensor:{input:"XYZ [B,43296,3] + F [B,43296,64]",operation:"floor → rank → sort → sum",output:"[B,64,1,200,200]",detail:"official LSS uses hard-cell sum pooling"},
    formula:"Fcell=Σᵢ:voxel(i)=cell Fᵢ",
    handoff:"A regular BEV tensor can now use ordinary convolution.",
    evidence:"LSS CODE",source:"voxel_pooling() · QuickCumsum",illustration:"splat"
  },
  {
    id:"bev-encoder",act:"04 · BEV ENCODER",title:"Reason on the ground plane",
    question:"Where does geometry end and learned spatial reasoning begin?",
    reveal:"After Splat, every axis is regular.",
    explanation:"Collapse Z, fuse BEV scales, and emit task logits. No camera projection remains inside BevEncode.",
    steps:[{label:"COLLAPSE",text:"[B,C,Z,X,Y] → [B,CZ,X,Y]."},{label:"ENCODE",text:"ResNet-18 mixes neighboring BEV cells."},{label:"PREDICT",text:"The task head emits one logit per cell."}],
    tensor:{input:"[B,64,1,200,200]",operation:"collapse Z → multiscale BevEncode",output:"logits [B,1,200,200]",detail:"geometry is finished before the BEV CNN starts"},
    formula:"BEV feature → ResNet-18 scales → task logits",
    handoff:"The remaining question is what teaches the depth branch.",
    evidence:"LSS CODE",source:"BevEncode.forward()",illustration:"bev-encoder"
  },
  {
    id:"supervision",act:"05 · SUPERVISION",title:"Where does depth learn?",
    question:"Why do LSS and BEVDepth share Lift–Splat but learn different depth?",
    reveal:"LSS learns depth through the task. BEVDepth adds direct depth supervision.",
    explanation:"BEVDepth projects training LiDAR, keeps the nearest valid depth per block, bins it, then applies sparse depth BCE.",
    steps:[{label:"LSS",text:"Task loss backpropagates through BEV, Splat and Lift."},{label:"TARGET",text:"LiDAR → image → nearest depth → one-hot bin."},{label:"BEVDEPTH",text:"Depth BCE joins the detection loss."}],
    tensor:{input:"Dpred + projected LiDAR",operation:"min nonzero → bin → masked BCE",output:"L=Ldet+3Ldepth",detail:"LiDAR supervises training; inference stays camera-only"},
    formula:"Ldepth=BCE(Dpred[valid],one_hot(Dgt)[valid])",
    handoff:"At inference, both systems start from cameras again.",
    evidence:"BEVDEPTH CODE",source:"get_downsampled_gt_depth() · get_depth_loss()",illustration:"learning"
  },
  {
    id:"inference",act:"06 · INFERENCE",title:"Decode only what was trained",
    question:"What comes after the BEV tensor?",
    reveal:"The head defines the product.",
    explanation:"The released LSS checkpoint predicts vehicle occupancy. BEVDepth uses a 3D detection head; this site does not mix their outputs.",
    steps:[{label:"LSS",text:"logit → sigmoid → threshold."},{label:"BEVDEPTH",text:"BEV feature → 3D detection head."},{label:"BOUNDARY",text:"No invented decoder or cross-task metric."}],
    tensor:{input:"LSS logits [B,1,200,200]",operation:"sigmoid → threshold",output:"vehicle occupancy mask",detail:"the public checkpoint stops here"},
    formula:"P(vehicle)=σ(logit)",
    handoff:"Now audit that mask against independent geometry.",
    evidence:"LSS CODE",source:"eval_model_iou() · model525000.pt",illustration:"truth"
  },
  {
    id:"truth-lab",act:"07 · TRUTH LAB",title:"Check the coordinate story",
    question:"Do prediction, GT and LiDAR agree in ego space?",
    reveal:"Compare them in one orientation—not by image resemblance.",
    explanation:"LiDAR and GT are reference evidence. They never enter this LSS checkpoint.",
    steps:[{label:"PREDICT",text:"Checkpoint probability."},{label:"REFERENCE",text:"nuScenes GT and one LiDAR sweep."},{label:"AUDIT",text:"TP, FP, FN and orientation."}],
    tensor:{input:"probability + GT + reference LiDAR",operation:"align in ego BEV",output:"TP 282 · FP 122 · FN 120 · IoU .538",detail:"single-frame diagnosis, not a paper benchmark"},
    formula:"ego +x=screen up · ego +y=screen left",
    handoff:"The full path is now visible: ray evidence → metric candidates → pooled BEV → task output.",
    evidence:"CHECKPOINT",source:"fixed nuScenes sample · model525000.pt",illustration:"truth",lab:"bev"
  }
];

export function sceneIndexFromHash(hash:string){const id=hash.replace(/^#/,"");const aliases:Record<string,string>={motivation:"ray-evidence","depth-distribution":"ray-evidence","context-feature":"ray-evidence","lift-outer-product":"lift-geometry","image-to-ray":"lift-geometry","ray-to-camera":"lift-geometry","camera-to-ego":"lift-geometry","encoder-supervision":"supervision"};const resolved=aliases[id]??id,index=SCENES.findIndex(scene=>scene.id===resolved);return index<0?0:index;}
