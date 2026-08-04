"use client";

import { ChevronLeft, ChevronRight, Pause, Play, RotateCcw } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { applyRigid, boltzmannProbabilities, pixelToCamera, trajectoryCost, undoPostTransform } from "../lib/algorithm.mjs";
import { CAMERA_NAMES, LssScene, type SceneSelection, type Trajectory } from "./LssScene";

const STEPS = [
  {kicker:"A geometry-first architecture",title:"An arbitrary camera rig",line:"Six cameras observe one timestamp from different places. LSS never concatenates them in a privileged order: each view is lifted independently, then summed in the ego frame.",formula:"f(Tξ · x) = Tξ · f(x)",note:"Three intended symmetries: image translation, camera permutation, and ego-frame isometries."},
  {kicker:"Undo the image plane",title:"A pixel is not a point",line:"Click a pixel in the real front image. Intrinsics turn it into a ray—not a 3D point. Resize and crop must be undone before K⁻¹ can mean anything.",formula:"r = K⁻¹ · A⁻¹ [u, v, 1]ᵀ",note:"Every positive depth lies on the same ray. Monocular evidence alone does not choose one."},
  {kicker:"Learn where context might live",title:"Lift with latent depth",line:"At every 8×22 feature location, the camera encoder predicts 41 depth probabilities and a 64D context vector. Their outer product creates a feature at every candidate depth.",formula:"f(c,d) = α(d) · c  ∈ ℝ⁴¹ˣ⁶⁴",note:"One-hot resembles pseudo-LiDAR; uniform resembles OFT; LSS can learn multi-modal uncertainty."},
  {kicker:"One sample, every matrix",title:"Move into the ego frame",line:"Follow the selected sample: undo post-transform, unproject with K⁻¹, scale by depth, then rotate and translate with cam2ego.",formula:"pₑ = R · K⁻¹[d·u, d·v, d]ᵀ + t",note:"The matrix inspector and 3D frustum point refer to the same depth sample."},
  {kicker:"Pack · rank · sum",title:"Splat by pillar pooling",line:"Discard points outside the 100m square, quantize XYZ, sort equal voxel ranks together, then QuickCumsum sum-pools without padded pillars.",formula:"rank = x(YZB) + y(ZB) + zB + b",note:"Click the BEV to inspect one cell and trace its contributing cameras."},
  {kicker:"Dense reasoning after geometry",title:"Encode the BEV",line:"The pooled 64-channel pseudo-image enters a ResNet-18-style BEV encoder. Multi-scale features are fused back to a 200×200 vehicle logit map.",formula:"[1,64,200,200] → ResNet18 → [1,1,200,200]",note:"The colored raster is this repository’s verified output from model525000.pt—not an illustration."},
  {kicker:"What the loss can and cannot say",title:"What supervision teaches",line:"Vehicle masks and map layers supervise the final BEV. They do not directly supervise depth; useful depth and context emerge because they improve the downstream task.",formula:"Lseg = BCEWithLogits(ŷBEV, yBEV)",note:"Paper benchmarks are dataset-level. This single sample is a qualitative checkpoint demonstration."},
  {kicker:"Stress the rig",title:"Robustness and arbitrary rigs",line:"Disable a camera or perturb CAM_FRONT yaw. Every option switches to an offline, checkpoint-derived result so the browser stays static and auditable.",formula:"BEV = Σcamera Splat(Lift(Icamera))",note:"The paper’s robustness claims come from trained experiments; these toggles show one fixed sample only."},
  {kicker:"A teaching reconstruction",title:"Shoot trajectories",line:"Hover representative trajectories. Each samples a spatial cost map; the negative summed cost becomes a Boltzmann distribution, and test-time planning chooses its argmax.",formula:"p(τ|o) ∝ exp(− Σ(x,y)∈τ cₒ(x,y) / T)",note:"The official public repository has no planning checkpoint. This chapter reconstructs the paper equation, clearly labeled teaching."},
  {kicker:"Close the loop",title:"Return to the scene",line:"Six images become latent 3D features, sum into BEV, and predict vehicles in the ego frame. Compare the real checkpoint raster with annotation-derived ground truth.",formula:"images → Lift → Splat → BEV CNN → task",note:"One timestamp cannot resolve every occlusion, far object, or night-time depth ambiguity."},
] as const;

type CameraRecord={name:string;image:string;cam2img:number[][];cam2ego:number[][];lidar2cam:number[][];post_rot:number[][];post_trans:number[];augmentation:{resize:number;resize_dims:number[];crop:number[]}};
type RigContract={sample_token:string;cameras:CameraRecord[];grid:Record<string,number[]>;vehicles_ego:{center_ego:number[];dimensions:number[];yaw_ego:number}[]};
type FeatureContract={probe:{feature_cell:number[];depth_by_camera:number[][];context_by_camera:number[][]}};
type EncodedArray={shape:number[];dtype:string;data:string};
type Mat3=[[number,number,number],[number,number,number],[number,number,number]];
type Vec3=[number,number,number];
type ModelContract={checkpoint_sha256?:string;source_hashes:{checkpoint:string};variants:Record<string,{image:string;probability_min:number;probability_max:number;probability_mean:number;evidence:string}>;tensor_checks:{finite:boolean;depth_probability_sum_max_error:number};ground_truth:{image:string};geometry_contributors:{axis_order:string[];counts:EncodedArray}};
type TransformTrace={augmented:Vec3;original:Vec3;camera:Vec3;ego:Vec3;depth:number};

const FALLBACK_DEPTH=Array.from({length:41},(_,index)=>Math.exp(-Math.pow((index-15)/5,2))/8.86);

function asset(path:string){return `${process.env.NEXT_PUBLIC_BASE_PATH??""}${path}`}
function shortName(name:string){return name.replace("CAM_","").replaceAll("_"," ")}
function format(value:number,digits=3){return Number(value).toFixed(digits)}

function decodeUint16(encoded?:EncodedArray){
  if(!encoded)return null;const binary=window.atob(encoded.data),bytes=new Uint8Array(binary.length);for(let index=0;index<binary.length;index+=1)bytes[index]=binary.charCodeAt(index);
  const view=new DataView(bytes.buffer),values=new Uint16Array(binary.length/2);for(let index=0;index<values.length;index+=1)values[index]=view.getUint16(index*2,true);return values;
}

function CoordinateRail({step}:{step:number}){
  const labels=["augmented pixel","original pixel","camera ray","camera 3D","ego 3D","voxel [x,y,z]","BEV [x,y]"];
  const active=Math.min(6,step===0?0:step===1?2:step===2?3:step===3?4:step===4?6:6);
  return <div className="coordinate-rail" aria-label="Coordinate transformation pipeline">{labels.map((label,index)=><span key={label} className={index===active?"active":index<active?"past":""}><i>{index<2?"px":index===2?"ray":index<5?"m":"idx"}</i>{label}</span>)}</div>;
}

function DepthDistribution({values,selected,onSelect}:{values:number[];selected:number;onSelect:(value:number)=>void}){
  const maximum=Math.max(...values,.001);
  return <div className="depth-chart" aria-label="41 depth probabilities">{values.map((value,index)=><button key={index} style={{height:`${Math.max(3,value/maximum*100)}%`}} className={selected===index?"selected":""} title={`${4+index} m · p=${value.toFixed(4)}`} onClick={()=>onSelect(index)}/>)}</div>
}

function Matrix({values,label}:{values:number[][];label:string}){
  return <div className="matrix"><b>{label}</b>{values.map((row,index)=><code key={index}>{row.map(value=>value.toFixed(Math.abs(value)>99?1:3).padStart(8," ")).join(" ")}</code>)}</div>
}

function ImageRig({rig,selected,onSelect,pixel,onPixel}:{rig:RigContract|null;selected:number;onSelect:(index:number)=>void;pixel:[number,number];onPixel:(pixel:[number,number])=>void}){
  const records=rig?.cameras??CAMERA_NAMES.map(name=>({name,image:`/data/images/${name.toLowerCase().replaceAll("_","-")}.jpg`} as CameraRecord));
  return <div className="camera-rig">{records.map((camera,index)=><button key={camera.name} className={selected===index?"selected":""} onClick={event=>{
    onSelect(index);if(index===selected){const rect=event.currentTarget.getBoundingClientRect();onPixel([Math.round((event.clientX-rect.left)/rect.width*352),Math.round((event.clientY-rect.top)/rect.height*128)])}
  }}><img src={asset(camera.image)} alt={`${shortName(camera.name)} nuScenes camera view`}/><span>{shortName(camera.name)}</span>{index===selected&&<i style={{left:`${pixel[0]/352*100}%`,top:`${pixel[1]/128*100}%`}}/>}</button>)}</div>;
}

function ShapeLedger(){
  return <div className="shape-ledger"><span><b>[1,6,3,128,352]</b><small>normalized images</small></span><em>EfficientNet-B0</em><span><b>[1,6,41,8,22,64]</b><small>lifted features</small></span><em>sum pool</em><span><b>[1,64,200,200]</b><small>BEV pseudo-image</small></span><em>ResNet-18</em><span><b>[1,1,200,200]</b><small>vehicle logits</small></span></div>;
}

function PaperTable(){return <div className="paper-table"><div><b>Method</b><b>nuScenes car</b><b>vehicles</b></div>{[["CNN","22.78","24.25"],["Frozen encoder","25.51","26.83"],["OFT","29.72","30.05"],["Lift-Splat","32.06","32.07"]].map(row=><div key={row[0]} className={row[0]==="Lift-Splat"?"highlight":""}>{row.map(value=><span key={value}>{value}</span>)}</div>)}<small>Paper Table 1 · IoU (%) · dataset-level evaluation</small></div>}

function SelectionCard({selection,rig,onClose}:{selection:SceneSelection;rig:RigContract|null;onClose:()=>void}){
  let title="Inspection",label="GEOMETRY",rows:[string,string][]=[];
  if(selection.kind==="camera"){
    const camera=rig?.cameras[selection.index];title=shortName(CAMERA_NAMES[selection.index]);label="CALIBRATED SENSOR";
    rows=[["camera index",String(selection.index)],["image token",camera?"pinned demo asset":"loading"],["translation",camera?camera.cam2ego.slice(0,3).map(row=>row[3].toFixed(3)).join(", "):"…"],["fx · fy",camera?`${camera.cam2img[0][0].toFixed(2)} · ${camera.cam2img[1][1].toFixed(2)}`:"…"]];
  } else if(selection.kind==="ray"){title=`Pixel (${selection.pixel.join(", ")})`;label="PINHOLE RAY";rows=[["candidate depth",`${selection.depth} m`],["meaning","one point on an infinite ray"],["homogeneous input",`[${selection.pixel[0]}, ${selection.pixel[1]}, 1]ᵀ`]];
  } else if(selection.kind==="depth"){title=`Depth bin ${selection.bin}`;label="LATENT DEPTH";rows=[["metric depth",`${selection.meters} m`],["probability",selection.probability.toFixed(6)],["feature channels","64 context values"]];
  } else if(selection.kind==="cell"){title=`BEV cell [${selection.index.join(", ")}]`;label="SUM-POOLED PILLAR";rows=[["index order","[ego x cell, ego y cell]"],["ego center",`(${selection.center.map(v=>v.toFixed(2)).join(", ")}) m`],["camera samples",selection.contributors.length?selection.contributors.map((name,index)=>`${shortName(name)}:${selection.counts?.[index]??"?"}`).join(" · "):"none"],["pooling","sum after exact rank sort"]];
  } else {title=selection.index===0?"Minimum-cost trajectory":`Template ${selection.index+1}`;label="SHOOT PROBE";rows=[["summed cost",selection.cost.toFixed(3)],["Boltzmann p",selection.probability.toFixed(4)],["decision",selection.index===0?"selected in this probe":"candidate"]];}
  return <aside className="inspector" aria-live="polite"><button onClick={onClose} aria-label="Close inspection card">×</button><span>{label}</span><h2>{title}</h2>{rows.map(([key,value])=><p key={key}><b>{key}</b><em>{value}</em></p>)}<small>Press Esc to close.</small></aside>;
}

function ChapterControl({step,rig,model,trace,selectedCamera,setSelectedCamera,pixel,setPixel,depthMode,setDepthMode,depthIndex,setDepthIndex,depthProbability,enabled,setEnabled,yaw,setYaw,temperature,setTemperature,trajectories,selectedTrajectory,setSelectedTrajectory,bevView,setBevView}:{step:number;rig:RigContract|null;model:ModelContract|null;trace:TransformTrace|null;selectedCamera:number;setSelectedCamera:(v:number)=>void;pixel:[number,number];setPixel:(v:[number,number])=>void;depthMode:string;setDepthMode:(v:string)=>void;depthIndex:number;setDepthIndex:(v:number)=>void;depthProbability:number[];enabled:boolean[];setEnabled:(v:boolean[])=>void;yaw:number;setYaw:(v:number)=>void;temperature:number;setTemperature:(v:number)=>void;trajectories:Trajectory[];selectedTrajectory:number;setSelectedTrajectory:(v:number)=>void;bevView:string;setBevView:(v:string)=>void}){
  const camera=rig?.cameras[selectedCamera];
  if(step===0)return <div className="chapter-control rig-control"><ImageRig rig={rig} selected={selectedCamera} onSelect={setSelectedCamera} pixel={pixel} onPixel={setPixel}/><div className="frame-contract"><b>Column-vector contract</b><span>nuScenes ego: +x forward · +y left · +z up</span><code>pₑ = Tcam→ego · pcam</code><em>3D view maps screen-right = −ego y</em></div><div className="symmetry-row"><span>translation equivariant</span><span>camera permutation invariant</span><span>ego isometry equivariant</span></div>{camera&&<div className="matrix-pair"><Matrix label="K · cam2img" values={camera.cam2img}/><Matrix label="[R | t] · cam2ego" values={camera.cam2ego.slice(0,3)}/></div>}</div>;
  if(step===1)return <div className="chapter-control pixel-control"><div className="focus-image" onPointerDown={event=>{const rect=event.currentTarget.getBoundingClientRect();setPixel([Math.round((event.clientX-rect.left)/rect.width*352),Math.round((event.clientY-rect.top)/rect.height*128)])}}><img src={asset(camera?.image??"/data/images/cam-front.jpg")} alt="Selected nuScenes image for pixel ray inspection"/><i style={{left:`${pixel[0]/352*100}%`,top:`${pixel[1]/128*100}%`}}/></div><div><div className="operation-strip"><span>A⁻¹<small>processed ({pixel.join(", ")})<br/>→ original {trace?`${format(trace.original[0],1)}, ${format(trace.original[1],1)}`:"…"}</small></span><em>→</em><span>K⁻¹<small>focal {camera?.cam2img[0][0].toFixed(1)??"1266.4"}<br/>camera-frame ray</small></span><em>→</em><span>λr<small>zcam = λ<br/>depth ambiguous</small></span></div><div className="convention-line">A maps original → network image; geometry uses A⁻¹. K maps camera meters → original pixels.</div></div></div>;
  if(step===2)return <div className="chapter-control lift-control"><div className="mode-tabs">{["checkpoint","one-hot","uniform","multi-modal"].map(mode=><button className={mode===depthMode?"active":""} onClick={()=>setDepthMode(mode)} key={mode}>{mode}</button>)}</div><DepthDistribution values={depthProbability} selected={depthIndex} onSelect={setDepthIndex}/><input aria-label="Selected depth bin" type="range" min="0" max="40" value={depthIndex} onChange={event=>setDepthIndex(Number(event.target.value))}/><div className="lift-equation"><span><b>{(depthProbability[depthIndex]??0).toFixed(4)}</b><small>α({4+depthIndex}m) · camera z depth</small></span><em>⊗</em><span><b>64D</b><small>context c · image frame</small></span><em>=</em><span><b>α · c</b><small>[depth, channel]</small></span></div></div>;
  if(step===3)return <div className="chapter-control transform-control"><div className="trace-table"><div><b>1 · network pixel</b><code>{trace?trace.augmented.map(v=>format(v,2)).join(", "):"…"}</code><small>processed image · px</small></div><em>A⁻¹</em><div><b>2 · original pixel</b><code>{trace?trace.original.map(v=>format(v,2)).join(", "):"…"}</code><small>raw image · px</small></div><em>K⁻¹ · d</em><div><b>3 · camera point</b><code>{trace?trace.camera.map(v=>format(v,3)).join(", "):"…"}</code><small>xcam,ycam,zcam · m</small></div><em>R · p + t</em><div><b>4 · ego point</b><code>{trace?trace.ego.map(v=>format(v,3)).join(", "):"…"}</code><small>x forward,y left,z up · m</small></div></div>{camera&&<div className="matrix-pair"><Matrix label="A · post_rot" values={camera.post_rot}/><Matrix label="Tcam→ego · [R|t]" values={camera.cam2ego.slice(0,3)}/></div>}<div className="convention-line">Column vectors. `cam2ego` is not inverted here; `lidar2cam` is unrelated to camera-only LSS inference.</div></div>;
  if(step===4)return <div className="chapter-control splat-control"><div className="splat-steps"><span><b>43,296</b><small>ego-frame samples</small></span><em>filter</em><span><b>[ix,iy,iz,b]</b><small>ix=floor((x+50)/.5)</small></span><em>rank + sort</em><span><b>QuickCumsum</b><small>equal rank → exact sum</small></span><em>write</em><span><b>[B,C,Z,X,Y]</b><small>then collapse Z into C</small></span></div><div className="frame-contract compact"><b>Axis contract</b><span>grid X follows ego +x forward</span><span>grid Y follows ego +y left</span><code>rank=x·YZB+y·ZB+z·B+b</code></div><small className="control-note">Click any BEV cell: contributor counts are computed from the exact calibrated 41×8×22 frustum samples, per camera.</small></div>;
  if(step===5)return <div className="chapter-control encode-control"><ShapeLedger/><div className="frame-contract compact"><b>Memory → geometry</b><span>tensor rows = ego X</span><span>tensor columns = ego Y</span><code>BEV[:, :, ix, iy]</code><em>display right = −ego Y, up = +ego X</em></div><div className="view-tabs"><button className={bevView==="model"?"active":""} onClick={()=>setBevView("model")}>checkpoint output</button><button className={bevView==="gt"?"active":""} onClick={()=>setBevView("gt")}>nuScenes GT</button></div><p className="artifact-proof"><b>Checkpoint-derived</b><span>{model?.source_hashes.checkpoint.slice(0,16)??"loading"}…</span><em>{model?.tensor_checks.finite?"all tensors finite":"verifying"}</em></p></div>;
  if(step===6)return <div className="chapter-control supervision-control"><PaperTable/><div className="fact-row"><span><b>14.3M</b><small>trainable parameters</small></span><span><b>35 Hz</b><small>paper · Titan V</small></span><span><b>32.07</b><small>vehicles IoU · paper</small></span></div></div>;
  if(step===7)return <div className="chapter-control robustness-control"><div className="camera-switches">{CAMERA_NAMES.map((name,index)=><button key={name} aria-pressed={enabled[index]} className={enabled[index]?"on":"off"} onClick={()=>{const next=[...enabled];if(next.filter(Boolean).length===5&&!next[index])return;next[index]=!next[index];if(next.filter(Boolean).length<5){const firstMissing=next.findIndex(value=>!value);next.fill(true);next[firstMissing]=false}setEnabled(next)}}><i/>{shortName(name)}</button>)}</div><div className="yaw-picker"><span>CAM_FRONT yaw</span>{[-3,0,3].map(value=><button key={value} className={yaw===value?"active":""} onClick={()=>{setYaw(value);setEnabled(Array(6).fill(true))}}>{value>0?"+":""}{value}°</button>)}</div><div className="convention-line">Perturbation convention: R′cam→ego = Rz,ego(δ) · Rcam→ego; translation and all other cameras stay fixed.</div><button className="reset-button" onClick={()=>{setEnabled(Array(6).fill(true));setYaw(0)}}><RotateCcw size={14}/> reset full rig</button><small className="control-note">Only one perturbation is applied at once; each raster was exported offline from the same checkpoint.</small></div>;
  if(step===8)return <div className="chapter-control shoot-control"><div className="trajectory-list">{trajectories.map((trajectory,index)=><button key={trajectory.name} className={selectedTrajectory===index?"active":""} onMouseEnter={()=>setSelectedTrajectory(index)} onFocus={()=>setSelectedTrajectory(index)}><i style={{opacity:.2+trajectory.probability*.8}}/><span>{trajectory.name}</span><b>{trajectory.cost.toFixed(2)}</b><em>{(trajectory.probability*100).toFixed(1)}%</em></button>)}</div><label>temperature T <b>{temperature.toFixed(2)}</b><input type="range" min="0.25" max="2" step="0.05" value={temperature} onChange={event=>setTemperature(Number(event.target.value))}/></label><small className="teaching-label">TEACHING · paper equation · representative subset of 1K templates</small></div>;
  return <div className="chapter-control final-control"><div className="view-tabs"><button className={bevView==="model"?"active":""} onClick={()=>setBevView("model")}>real checkpoint</button><button className={bevView==="gt"?"active":""} onClick={()=>setBevView("gt")}>vehicle GT</button></div><div className="oracle-table"><span><b>nuScenes car</b><small>Lift-Splat 32.06 · oracle 1 scan 40.26</small></span><span><b>nuScenes vehicles</b><small>Lift-Splat 32.07 · oracle 1 scan 44.48</small></span></div><div className="limit-row"><span>single timestamp</span><span>far range</span><span>night / glare</span><span>occlusion</span></div></div>;
}

export default function LssExplainer(){
  const [step,setStep]=useState(0),[playing,setPlaying]=useState(false),[rig,setRig]=useState<RigContract|null>(null),[features,setFeatures]=useState<FeatureContract|null>(null),[model,setModel]=useState<ModelContract|null>(null);
  const [selectedCamera,setSelectedCamera]=useState(1),[pixel,setPixel]=useState<[number,number]>([176,64]),[depthIndex,setDepthIndex]=useState(15),[depthMode,setDepthMode]=useState("checkpoint"),[selection,setSelection]=useState<SceneSelection|null>(null);
  const [enabled,setEnabled]=useState<boolean[]>(Array(6).fill(true)),[yaw,setYaw]=useState(0),[temperature,setTemperature]=useState(.8),[selectedTrajectory,setSelectedTrajectory]=useState(0),[bevView,setBevView]=useState("model");
  useEffect(()=>{Promise.all([fetch(asset("/data/rig.json")).then(r=>r.json()),fetch(asset("/data/model-features.json")).then(r=>r.json()),fetch(asset("/data/model-artifacts.json")).then(r=>r.json())]).then(([a,b,c])=>{setRig(a);setFeatures(b);setModel(c)}).catch(error=>console.error("Evidence assets failed to load",error))},[]);
  const checkpointDepth=features?.probe.depth_by_camera[selectedCamera]??FALLBACK_DEPTH;
  const transformTrace=useMemo<TransformTrace|null>(()=>{
    const camera=rig?.cameras[selectedCamera];if(!camera)return null;const depth=4+depthIndex;
    const augmented=[pixel[0],pixel[1],1] as Vec3;
    const original=undoPostTransform(augmented,camera.post_rot as Mat3,camera.post_trans as Vec3);
    const cameraPoint=pixelToCamera(original,depth,camera.cam2img as Mat3);
    const rotation=camera.cam2ego.slice(0,3).map(row=>row.slice(0,3)) as Mat3,translation=camera.cam2ego.slice(0,3).map(row=>row[3]) as Vec3;
    return {augmented,original,camera:cameraPoint,ego:applyRigid(cameraPoint,rotation,translation),depth};
  },[rig,selectedCamera,pixel,depthIndex]);
  const liftedPoints=useMemo<[number,number,number][]>(()=>{
    const camera=rig?.cameras[selectedCamera];if(!camera)return Array.from({length:41},(_,index)=>[0,4+index,1.5] as [number,number,number]);
    const original=undoPostTransform([pixel[0],pixel[1],1],camera.post_rot as Mat3,camera.post_trans as Vec3),rotation=camera.cam2ego.slice(0,3).map(row=>row.slice(0,3)) as Mat3,translation=camera.cam2ego.slice(0,3).map(row=>row[3]) as Vec3;
    return Array.from({length:41},(_,index)=>{const ego=applyRigid(pixelToCamera(original,4+index,camera.cam2img as Mat3),rotation,translation);return [-ego[1],ego[0],ego[2]] as [number,number,number]});
  },[rig,selectedCamera,pixel]);
  const contributorCounts=useMemo(()=>decodeUint16(model?.geometry_contributors.counts),[model]);
  const depthProbability=useMemo(()=>{
    if(depthMode==="checkpoint")return checkpointDepth;
    if(depthMode==="one-hot")return Array.from({length:41},(_,index)=>index===depthIndex?1:0);
    if(depthMode==="uniform")return Array(41).fill(1/41);
    const raw=Array.from({length:41},(_,index)=>Math.exp(-Math.pow((index-11)/3.2,2))+0.72*Math.exp(-Math.pow((index-25)/4.2,2)));const sum=raw.reduce((a,b)=>a+b,0);return raw.map(value=>value/sum);
  },[checkpointDepth,depthMode,depthIndex]);
  const trajectories=useMemo<Trajectory[]>(()=>{
    const paths=Array.from({length:9},(_,index)=>Array.from({length:21},(_,point)=>{const y=point*.72;const curve=(index-4)*.017*y*y;return [curve,y] as [number,number]}));
    const map=([x,y]:[number,number])=>.055*Math.abs(x)+.85*Math.exp(-((x-2.3)**2+(y-9)**2)/5)+.5*Math.exp(-((x+1.4)**2+(y-13)**2)/3);
    const costs=paths.map(path=>trajectoryCost(path,map,.14));const probability=boltzmannProbabilities(costs,temperature);const order=costs.map((cost,index)=>({cost,index})).sort((a,b)=>a.cost-b.cost);
    return order.map((item,rank)=>({name:rank===0?"minimum cost":`template ${item.index+1}`,points:paths[item.index],cost:item.cost,probability:probability[item.index]}));
  },[temperature]);
  const activeDrop=enabled.findIndex(value=>!value);
  const variantKey=activeDrop>=0?`drop-${CAMERA_NAMES[activeDrop].toLowerCase().replaceAll("_","-")}`:`front-yaw-${yaw>=0?"+":""}${yaw}`;
  const bevImage=bevView==="gt"?(model?.ground_truth.image??"/data/model/vehicle-gt.png"):(model?.variants[variantKey]?.image??model?.variants["all-cameras"]?.image??"/data/model/bev-all-cameras.png");
  const go=useCallback((value:number)=>{setStep(Math.max(0,Math.min(9,value)));setPlaying(false);setSelection(null)},[]);
  useEffect(()=>{if(!playing)return;const id=window.setInterval(()=>setStep(value=>value===9?0:value+1),5200);return()=>window.clearInterval(id)},[playing]);
  useEffect(()=>{const handler=(event:KeyboardEvent)=>{if(event.key==="Escape")setSelection(null);if(event.key==="ArrowLeft")go(step-1);if(event.key==="ArrowRight")go(step+1);if(event.code==="Space"&&!(["INPUT","BUTTON"].includes((event.target as HTMLElement)?.tagName))){event.preventDefault();setPlaying(value=>!value)}};window.addEventListener("keydown",handler);return()=>window.removeEventListener("keydown",handler)},[go,step]);
  const handleSelection=(value:SceneSelection)=>{
    if(value.kind==="cell"&&contributorCounts){const [cellX,cellY]=value.index,cell=value;const raw=CAMERA_NAMES.map((_,camera)=>contributorCounts[camera*40000+cellX*200+cellY]);const contributing=CAMERA_NAMES.map((name,index)=>({name,count:raw[index]})).filter(item=>item.count>0);value={...cell,contributors:contributing.map(item=>item.name),counts:contributing.map(item=>item.count)}}
    setSelection(value);if(value.kind==="camera")setSelectedCamera(value.index);if(value.kind==="depth")setDepthIndex(value.bin);if(value.kind==="trajectory")setSelectedTrajectory(value.index)
  };
  return <main className={`lss-app step-${step}`}>
    <LssScene step={step} enabledCameras={enabled} selectedCamera={selectedCamera} pixel={pixel} depthIndex={depthIndex} depthProbability={depthProbability} bevImage={bevImage} trajectories={trajectories} selectedTrajectory={selectedTrajectory} cameraMatrices={rig?.cameras.map(camera=>camera.cam2ego)} liftedPoints={liftedPoints} onSelect={handleSelection}/>
    <header className="site-header"><a href={asset("/")}><span>LSS</span> EXPLAINED</a><div><a href={asset("/articles/lift-splat-shoot-explained.zh-CN.md")}>中文长文</a><a href="https://github.com/hova88/lss-explained">SOURCE ↗</a><b>{String(step+1).padStart(2,"0")} / 10</b></div></header>
    <section className="story-card"><span>{STEPS[step].kicker}</span><h1>{STEPS[step].title}</h1><p>{STEPS[step].line}</p><code>{STEPS[step].formula}</code></section>
    <CoordinateRail step={step}/>
    <div className="axis-key"><b>EGO FRAME</b><span>↑ +x forward</span><span>← +y left</span><span>⊙ +z up</span></div>
    <aside className="margin-note"><b>NOTE {String(step+1).padStart(2,"0")}</b><p>{STEPS[step].note}</p><span>{step===8?"TEACHING":step===6?"PAPER":"REAL SAMPLE"}</span></aside>
    <ChapterControl step={step} rig={rig} model={model} trace={transformTrace} selectedCamera={selectedCamera} setSelectedCamera={setSelectedCamera} pixel={pixel} setPixel={setPixel} depthMode={depthMode} setDepthMode={setDepthMode} depthIndex={depthIndex} setDepthIndex={setDepthIndex} depthProbability={depthProbability} enabled={enabled} setEnabled={setEnabled} yaw={yaw} setYaw={setYaw} temperature={temperature} setTemperature={setTemperature} trajectories={trajectories} selectedTrajectory={selectedTrajectory} setSelectedTrajectory={setSelectedTrajectory} bevView={bevView} setBevView={setBevView}/>
    {selection&&<SelectionCard selection={selection} rig={rig} onClose={()=>setSelection(null)}/>}
    <nav className="timeline" aria-label="Chapter timeline"><button className="round" onClick={()=>go(step-1)} disabled={step===0} aria-label="Previous chapter"><ChevronLeft/></button><button className="round play" onClick={()=>setPlaying(value=>!value)} aria-label={playing?"Pause autoplay":"Start autoplay"}>{playing?<Pause/>:<Play/>}</button><div>{STEPS.map((item,index)=><button key={item.title} className={index===step?"active":index<step?"past":""} onClick={()=>go(index)} aria-label={`Chapter ${index+1}: ${item.title}`}><i/><span>{index+1}</span></button>)}</div><button className="round" onClick={()=>go(step+1)} disabled={step===9} aria-label="Next chapter"><ChevronRight/></button></nav>
    <div className="interaction-hint">DRAG TO ORBIT · SCROLL / PINCH TO ZOOM · CLICK TO INSPECT</div>
  </main>;
}
