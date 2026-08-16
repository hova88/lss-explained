export type Evidence = "LSS PAPER" | "LSS CODE" | "BEVDEPTH PAPER" | "BEVDEPTH CODE" | "REAL SAMPLE" | "CHECKPOINT" | "TEACHING";
export type IllustrationKind = "overview" | "features" | "ray" | "depth" | "context" | "lift" | "image-ray" | "camera-point" | "ego-transform" | "rig" | "splat" | "learning" | "truth";
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
    id:"motivation",act:"01 · THE CONTRACT",title:"The view must change, not just the features",
    question:"Why is camera-to-BEV a geometric problem before it is a detection problem?",
    reveal:"Six perspective tensors must become one metric tensor whose axes belong to the ego vehicle.",
    explanation:"A camera CNN preserves image topology: neighboring activations are neighboring rays, not neighboring ground locations. LSS contributes a differentiable coordinate change between those topologies. BEVDepth keeps this Lift–Splat spine and asks whether its latent depth is accurate enough for 3D detection.",
    steps:[{label:"ENCODE",text:"Resize/crop six images, record the image warp, and extract perspective features."},{label:"CHANGE BASIS",text:"Attach depth, unproject through calibration, and express every candidate in ego meters."},{label:"RASTERIZE",text:"Pool irregular candidates into a regular BEV tensor for a task head."}],
    tensor:{input:"images [B,6,3,900,1600]",operation:"preprocess → encode → Lift → geometry → Splat",output:"ego BEV [B,C,X,Y]",detail:"the spatial meaning of the axes changes twice"},
    formula:"image topology → frustum topology → ego-metric topology",
    handoff:"The non-invertible part is depth. Start there.",
    evidence:"LSS PAPER",source:"LSS Sec. 1 & 3 · BEVDepth Sec. 1 & 3",illustration:"overview",
    comparison:{lss:"Introduces the differentiable camera-to-BEV construction.",bevdepth:"Retains that construction and targets unreliable intermediate depth."}
  },
  {
    id:"depth-distribution",act:"02 · LIFT / WHERE",title:"Depth is a distribution over locations",
    question:"What does one column of the depth tensor actually mean?",
    reveal:"For one feature anchor, 41 scalars allocate evidence across 41 metric positions on one ray.",
    explanation:"In the pinned LSS model, channels 0…40 become α(d) after softmax along D. This is not a conventional dense depth map and its expectation is not used by Lift. The entire categorical vector participates: one-hot imitates pseudo-LiDAR, uniform imitates OFT, and multimodal α preserves ambiguity.",
    steps:[{label:"SLICE",text:"Select depth logits at one [camera,row,column]."},{label:"NORMALIZE",text:"softmax(dim=D) makes all 41 non-negative weights sum to one."},{label:"KEEP ALL BINS",text:"Do not argmax: every bin remains available to the outer product."}],
    tensor:{input:"logits [B,6,41,8,22]",operation:"softmax along D",output:"α [B,6,41,8,22]",detail:"one categorical distribution per image-feature anchor"},
    formula:"α(d|u,v)=exp z_d / Σₖ exp z_k",
    handoff:"α says where evidence may go. It does not say what evidence is.",
    evidence:"LSS PAPER",source:"LSS Fig. 3, Eq. 1 · official get_depth_feat()",illustration:"depth",lab:"geometry",
    comparison:{lss:"Depth is latent: only downstream BEV loss constrains α.",bevdepth:"The same kind of categorical depth receives an additional sparse depth loss."}
  },
  {
    id:"context-feature",act:"03 · LIFT / WHAT",title:"Context is the payload, not the position",
    question:"Why does DepthNet output another 64 channels?",
    reveal:"A 64D context vector carries task-relevant appearance; α only decides how that vector is distributed in space.",
    explanation:"The 105-channel LSS head separates cleanly into 41 depth logits and 64 context channels. Context has axes [camera,feature-row,feature-column,channel] but no depth axis. This separation is the conceptual hinge of Lift: location uncertainty and semantic payload can change independently.",
    steps:[{label:"SHARED INPUT",text:"Both branches read the same encoded image feature."},{label:"SEPARATE MEANING",text:"Depth channels answer where; context channels answer what."},{label:"NO 3D YET",text:"The context vector still belongs to one perspective anchor."}],
    tensor:{input:"head [B,6,105,8,22]",operation:"split channels 41 | 64",output:"α logits ⊕ context c",detail:"where [41] is distinct from what [64]"},
    formula:"c(u,v)∈ℝ⁶⁴",
    handoff:"Lift is the exact operation that gives c a depth axis.",
    evidence:"LSS CODE",source:"models.py CamEncode.get_depth_feat()",illustration:"context",lab:"geometry",
    comparison:{lss:"One convolution emits depth and context together.",bevdepth:"Camera-aware SE conditions both branches using a 27D calibration/augmentation vector."}
  },
  {
    id:"lift-outer-product",act:"04 · LIFT / OUTER PRODUCT",title:"Lift is broadcast multiplication",
    question:"How does [41] × [64] become a frustum feature volume?",
    reveal:"Every depth receives the same context vector, scaled by its own α(d).",
    explanation:"Unsqueeze α to [...,D,H,W,1] and context to [...,1,H,W,C]. Broadcasting creates a [D,C] slab at every anchor. No hard 3D point is chosen. Geometry and features now share the candidate index [camera,depth,row,column], which must remain aligned through every later reshape.",
    steps:[{label:"UNSQUEEZE",text:"α gains a trailing singleton channel; c gains a singleton depth."},{label:"BROADCAST",text:"Multiply [D,1] by [1,C] without materializing repeated inputs."},{label:"PRESERVE INDEX",text:"Candidate i must refer to the same geometry row and feature row after flattening."}],
    tensor:{input:"α […,41,8,22,1] × c […,1,8,22,64]",operation:"broadcast multiply",output:"F [B,6,41,8,22,64]",detail:"43,296 depth-bearing candidates, each with 64 values"},
    formula:"F(d,u,v,c)=α(d|u,v)c(u,v,c)",
    handoff:"The tensor has a depth axis, but its coordinates are still [u′,v′,d].",
    evidence:"LSS CODE",source:"models.py get_cam_feats() · BEVDepth Eq. 1",illustration:"lift",lab:"geometry"
  },
  {
    id:"image-to-ray",act:"05 · GEOMETRY / IMAGE PLANE",title:"Undo the image before inverting the camera",
    question:"Which point lies on the image plane, and which vector starts at the optical center?",
    reveal:"The network anchor is first returned to the calibrated raw image; K⁻¹ then turns its homogeneous pixel into a ray direction.",
    explanation:"The optical center O is the origin of the camera frame—not a pixel. A network anchor p′=[u′,v′,1] was created after resize/crop, so A⁻¹(p′−a) restores p=[u,v,1]. Intrinsics K map camera rays to pixels; K⁻¹ therefore yields r=[(u−cₓ)/fₓ,(v−cᵧ)/fᵧ,1]. r has direction but no metric distance.",
    steps:[{label:"NETWORK → RAW",text:"Undo post_rot/post_trans; do not apply K⁻¹ to an augmented pixel."},{label:"PIXEL → RAY",text:"Subtract principal point and divide by focal lengths through K⁻¹."},{label:"KEEP FRAMES DISTINCT",text:"[u,v,1] is homogeneous image data; r is a camera-frame direction."}],
    tensor:{input:"frustum [B,6,41,8,22,3] as [u′,v′,d]",operation:"A⁻¹(p′−a) → K⁻¹[u,v,1]",output:"r_cam [B,6,41,8,22,3]",detail:"same shape; units change from pixels to a direction ratio"},
    formula:"r_cam=K⁻¹[u,v,1]ᵀ",
    handoff:"Only d can turn that direction into a metric camera point.",
    evidence:"LSS CODE",source:"create_frustum() · get_geometry()",illustration:"image-ray",lab:"geometry"
  },
  {
    id:"ray-to-camera",act:"06 · GEOMETRY / CAMERA XYZ",title:"Depth scales the ray into camera meters",
    question:"Why does official code construct [du,dv,d] instead of multiplying [u,v,1] by d later?",
    reveal:"They are algebraically identical: [du,dv,d]=d[u,v,1], so K⁻¹ produces d·r.",
    explanation:"Choose one bin d. The corresponding camera point is p_cam=dK⁻¹[u,v,1]ᵀ=[Xc,Yc,Zc]. With the vision convention used here, +x points image-right, +y image-down and +z along the optical axis. The selected point, every other depth candidate, and the image-plane pixel are collinear with O.",
    steps:[{label:"CHOOSE A BIN",text:"The frustum already contains the metric bin center d."},{label:"SCALE",text:"Multiply u and v by d and keep d as the third homogeneous component."},{label:"UNPROJECT",text:"K⁻¹ returns camera-frame meters, not ego-frame meters."}],
    tensor:{input:"[u,v,d] grid [B,6,41,8,22,3]",operation:"[u,v,d]→[du,dv,d]→K⁻¹",output:"p_cam [B,6,41,8,22,3]",detail:"one metric XYZ point for every depth candidate"},
    formula:"p_cam(d)=d·r_cam=K⁻¹[du,dv,d]ᵀ",
    handoff:"Camera XYZ is metric, but six different camera frames still cannot be pooled.",
    evidence:"LSS CODE",source:"models.py get_geometry() · calibrated pinhole model",illustration:"camera-point",lab:"geometry"
  },
  {
    id:"camera-to-ego",act:"07 · GEOMETRY / SHARED FRAME",title:"Extrinsics make six camera tensors commensurable",
    question:"What exactly do R and t change?",
    reveal:"R changes the axes; t moves the camera origin to its physical location in the ego frame.",
    explanation:"For column vectors, p_ego=Rcam→ego p_cam+tcam→ego. The third column of R is the camera optical axis expressed in ego coordinates, so every frustum must point outward from its real optical center. Only after this operation may camera/depth/row/column be flattened into one candidate axis and fused.",
    steps:[{label:"ROTATE BASIS",text:"The numerical coordinates change because camera and ego axes differ."},{label:"TRANSLATE ORIGIN",text:"Add the optical center location expressed in ego meters."},{label:"FLATTEN TOGETHER",text:"Geometry [N,D,H,W,3] and features [N,D,H,W,C] keep identical ordering."}],
    tensor:{input:"p_cam + F for 6×41×8×22",operation:"Rcam→ego·p_cam+t; flatten",output:"XYZ [B,43296,3] + F [B,43296,64]",detail:"six independent frustums now share ego meters"},
    formula:"p_ego=Rcam→ego p_cam+tcam→ego",
    handoff:"Shared continuous coordinates must now become discrete BEV addresses.",
    evidence:"REAL SAMPLE",source:"nuScenes calibrated_sensor · LSS get_geometry()",illustration:"ego-transform",lab:"geometry",
    comparison:{lss:"Calibration is used by geometry but not fed into the depth predictor.",bevdepth:"Calibration and augmentation parameters also condition DepthNet itself."}
  },
  {
    id:"splat-pooling",act:"08 · SPLAT / COLLISIONS",title:"Pooling decides what a BEV cell remembers",
    question:"When several lifted candidates hit one cell, why does LSS sum them?",
    reveal:"Splat is a grouped reduction over candidate features with identical voxel indices.",
    explanation:"LSS filters bounds, floors XYZ to integer voxels, constructs a rank, sorts candidates and uses QuickCumsum to sum each group before scattering. Sum is permutation-invariant and preserves accumulated evidence. Mean, max and bilinear splatting answer different questions; use the live comparison to see which information each discards or spreads.",
    steps:[{label:"INDEX",text:"floor((p−origin)/Δ) maps continuous ego meters to a half-open grid."},{label:"GROUP",text:"Equal [batch,ix,iy,iz] ranks become contiguous after sorting."},{label:"REDUCE + SCATTER",text:"One feature per group is written to [B,C,Z,X,Y]."}],
    tensor:{input:"F [B,43296,64] + voxel index",operation:"filter → rank → sort → grouped SUM → scatter",output:"[B,64,1,200,200]",detail:"official LSS uses sum; alternatives are teaching comparisons"},
    formula:"Fcell=Σᵢ:voxel(i)=cell Fᵢ",
    handoff:"Once the tensor is regular, geometry stops and BEV convolution begins.",
    evidence:"LSS CODE",source:"voxel_pooling() · QuickCumsum · BEVDepth efficient voxel pooling",illustration:"splat",
    comparison:{lss:"Sort + QuickCumsum implements grouped sum in PyTorch.",bevdepth:"A custom efficient voxel-pooling kernel preserves the view-transform purpose while reducing latency."}
  },
  {
    id:"encoder-supervision",act:"09 · LEARNING / THE FORK",title:"The decisive difference is the depth gradient",
    question:"How can two models share Lift–Splat yet learn very different depth?",
    reveal:"LSS supervises the final BEV task only; BEVDepth adds a sparse, explicit loss directly on the depth tensor.",
    explanation:"LSS collapses Z, applies BevEncode and sends BCE gradients backward from a BEV segmentation mask. BEVDepth projects training-time LiDAR into each image, keeps the nearest nonzero depth in each downsample block, discretizes it, one-hots the bin, and applies BCE only at valid sparse pixels. Its official loss is detection_loss + 3·depth_loss; LiDAR is absent at inference.",
    steps:[{label:"LSS PATH",text:"BEV task loss must discover useful α indirectly through pooling."},{label:"BUILD DEPTH GT",text:"ego point → camera → K projection → min nonzero depth → bin → one-hot."},{label:"BEVDEPTH PATH",text:"Direct depth BCE joins the detection gradient at DepthNet."}],
    tensor:{input:"Dpred [B,N,D,h,w] + projected LiDAR [B,N,H,W]",operation:"min-pool → bin → one-hot → masked BCE",output:"L=Ldet+3Ldepth",detail:"BEVDepth training only; camera-only inference remains"},
    formula:"L_depth=BCE(D_pred[valid],one_hot(D_gt)[valid])",
    handoff:"Training changes. The inference contract and the evidence boundary must remain explicit.",
    evidence:"BEVDEPTH CODE",source:"base_exp.py get_downsampled_gt_depth() & get_depth_loss()",illustration:"learning",
    comparison:{lss:"No depth sensor during training or testing; final BEV task loss only.",bevdepth:"Training-time LiDAR supervises depth; detection and depth losses are optimized jointly."}
  },
  {
    id:"truth-lab",act:"10 · INFERENCE / AUDIT",title:"Audit outputs in their own coordinate system",
    question:"What can this real LSS checkpoint prove—and what belongs only to BEVDepth?",
    reveal:"The pinned checkpoint produces a vehicle-occupancy logit map; LiDAR and GT verify orientation but are not model inputs.",
    explanation:"For LSS segmentation: logits → sigmoid → threshold is the complete deployed decoding shown here; there is no invented box decoder or NMS. BEVDepth instead feeds its BEV tensor to a 3D detection head for classes, box offsets and attributes. The shared insight is geometric lifting; the supervision and task heads are different experimental systems.",
    steps:[{label:"DECODE LSS",text:"[1,1,200,200] logits become vehicle occupancy probabilities."},{label:"AUDIT",text:"GT boxes and reference LiDAR expose mirrors, offsets and false regions in ego coordinates."},{label:"SEPARATE CLAIMS",text:"This frame is LSS checkpoint evidence; BEVDepth behavior is paper/code evidence."}],
    tensor:{input:"LSS logits [1,1,200,200]",operation:"sigmoid → threshold → compare",output:"TP 282 · FP 122 · FN 120 · IoU .538",detail:"single-frame diagnostic, not a cross-paper benchmark"},
    formula:"ego +x=screen up; ego +y=screen left",
    handoff:"If every axis, reshape and gradient path is now explainable, the mechanism is understood.",
    evidence:"CHECKPOINT",source:"model525000.pt · nuScenes GT · reference LiDAR",illustration:"truth",lab:"bev",
    comparison:{lss:"This site audits the released vehicle-segmentation checkpoint.",bevdepth:"Published system targets 3D detection and reports detection metrics, not this mask IoU."}
  }
];

export function sceneIndexFromHash(hash:string){const id=hash.replace(/^#/,"");const index=SCENES.findIndex(scene=>scene.id===id);return index<0?0:index;}
