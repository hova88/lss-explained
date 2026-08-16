"use client";
/* eslint-disable @next/next/no-img-element */

import { ChevronLeft, ChevronRight, Menu, Pause, Play, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import { binaryStats, boltzmannProbabilities, float16LittleEndianToFloat32, nearestFeatureAnchor, trajectoryCost } from "../lib/algorithm.mjs";
import { IllustrationStage } from "./IllustrationStage";
import { SCENES, sceneIndexFromHash, type LabId } from "./lss-content";

type BevMode = "probability" | "threshold" | "gt" | "errors" | "lidar" | "contributors";
type PoolingMode = "sum" | "mean" | "max" | "bilinear";
type SceneCamera = { cam2img:number[][]; cam2ego:number[][]; lidar2cam:number[][] };
type Trajectory = { name:string; points:[number,number][]; cost:number; probability:number };
const CAMERA_NAMES=["CAM_FRONT_LEFT","CAM_FRONT","CAM_FRONT_RIGHT","CAM_BACK_RIGHT","CAM_BACK","CAM_BACK_LEFT"];
const GEOMETRY_DEFAULTS:Record<string,number>={"image-to-ray":2,"ray-to-camera":3,"camera-to-ego":4};
const GEOMETRY_TARGETS=["image-to-ray","image-to-ray","image-to-ray","ray-to-camera","camera-to-ego"] as const;
const TRACE_LABELS=["p′ network","p raw","ray r","p camera","p ego"] as const;
const TRACE_TENSORS=[
  {input:"feature anchor [h,w]",operation:"read frustum sample",detail:"network-image coordinates after resize and crop",output:"p′=[u′,v′,1] · network pixels"},
  {input:"p′ + post_rot A + post_trans a",operation:"A⁻¹(p′−a)",detail:"undo the exact resize/crop augmentation",output:"p=[u,v,1] · raw-image pixels"},
  {input:"p=[u,v,1] + intrinsics K",operation:"K⁻¹p",detail:"remove focal length and principal point",output:"r_cam=[x/z,y/z,1] · direction"},
  {input:"r_cam + selected depth d",operation:"d · r_cam",detail:"depth supplies metric scale along the same ray",output:"p_cam=[x,y,z] · camera meters"},
  {input:"p_cam + R_cam→ego,t_cam→ego",operation:"R p_cam + t",detail:"rotate the basis, then move the optical center",output:"p_ego=[x,y,z] · ego meters"},
] as const;

type EncodedArray = { shape:number[]; dtype:string; data:string };
type CameraRecord = SceneCamera & { name:string; image:string; network_image:string; post_rot:number[][]; post_trans:number[]; timestamp:number; augmentation:{resize:number;resize_dims:number[];crop:number[]} };
type Rig = { sample_token:string; timestamp:number; cameras:CameraRecord[]; lidar2ego:number[][]; ego2global:number[][] };
type Variant = { image:string; logits:EncodedArray; probability_min:number; probability_max:number; probability_mean:number; evidence:string };
type Model = { variants:Record<string,Variant>; ground_truth:{mask:EncodedArray;image:string}; geometry_contributors:{counts:EncodedArray}; tensor_checks:{finite:boolean;depth_probability_sum_max_error:number} };
type Features = { depth_probabilities:EncodedArray; context_features:EncodedArray; feature_anchors:{x:number[];y:number[]} };

const BASE = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const asset = (path:string) => `${BASE}${path.startsWith("/")?path:`/${path}`}`;
const short = (name:string) => name.replace("CAM_","").replaceAll("_"," ");
const f = (value:number,digits=2) => Number.isFinite(value)?value.toFixed(digits):"—";

function decodeBytes(encoded?:EncodedArray) {
  if(!encoded)return null;
  const binary=atob(encoded.data),bytes=new Uint8Array(binary.length);
  for(let i=0;i<binary.length;i+=1)bytes[i]=binary.charCodeAt(i);
  return bytes;
}
function decodeFloat(encoded?:EncodedArray){const bytes=decodeBytes(encoded);if(!bytes)return null;return encoded?.dtype==="float16"?float16LittleEndianToFloat32(bytes):new Float32Array(bytes.buffer,bytes.byteOffset,bytes.byteLength/4);}
function decodeUint8(encoded?:EncodedArray){const bytes=decodeBytes(encoded);return bytes?new Uint8Array(bytes.buffer):null;}

function EvidenceTag({ value }:{value:string}) { return <span className={`evidence-tag evidence-${value.toLowerCase().replaceAll(" ","-")}`}>{value}</span>; }

function StoryStep({ index }:{index:number}) {
  const scene=SCENES[index];
  return (
    <article id={scene.id} className="story-step active" data-scene={index}>
      <p className="act-label">{scene.act}</p>
      <h2>{scene.title}</h2>
      <p className="scene-reveal">{scene.reveal}</p>
      <p className="scene-question">{scene.question}</p>
      <p className="scene-explanation">{scene.explanation}</p>
      {scene.comparison&&<div className="method-contrast"><span><b>LSS</b>{scene.comparison.lss}</span><span><b>BEVDepth</b>{scene.comparison.bevdepth}</span></div>}
      <code className="scene-formula">{scene.formula}</code>
      <footer><EvidenceTag value={scene.evidence} /><span>{scene.source}</span></footer>
    </article>
  );
}

function DepthSketch({ values, selected, onSelect }:{values:number[];selected:number;onSelect:(index:number)=>void}) {
  const max=Math.max(...values,0.0001);
  return <div className="depth-sketch" aria-label="41-bin checkpoint depth allocation">{values.map((value,index)=><button key={index} className={index===selected?"active":""} style={{"--h":`${Math.max(4,(value/max)*100)}%`} as React.CSSProperties} onClick={()=>onSelect(index)} aria-label={`${index+4} meters, probability ${value}`}><i /></button>)}</div>;
}

function CameraStrip({ rig, selected, enabled, onSelect, onToggle }:{rig:Rig|null;selected:number;enabled:boolean[];onSelect:(index:number)=>void;onToggle?:(index:number)=>void}) {
  return <div className="camera-strip">{(rig?.cameras??[]).map((camera,index)=><figure key={camera.name} className={`${index===selected?"selected":""} ${enabled[index]?"":"disabled"}`}><button className="camera-photo" onClick={()=>onSelect(index)} aria-pressed={index===selected}><img src={asset(camera.image)} alt={short(camera.name)} /><span>{short(camera.name)}</span></button>{onToggle&&<button className="camera-toggle" onClick={()=>onToggle(index)}>{enabled[index]?"LIVE":"DROPPED"}</button>}</figure>)}</div>;
}

function LabShell({ id, title, eyebrow, children }:{id:LabId;title:string;eyebrow:string;children:React.ReactNode}) {
  return <section className={`lab-break lab-${id}`} data-lab={id}><div className="lab-heading"><p>{eyebrow}</p><h2>{title}</h2><span>Change one thing. Watch the whiteboard answer.</span></div><div className="lab-controls">{children}</div></section>;
}

function GeometryLab({ rig,selectedCamera,setSelectedCamera,depth,depthIndex,setDepthIndex }:{rig:Rig|null;selectedCamera:number;setSelectedCamera:(index:number)=>void;depth:number[];depthIndex:number;setDepthIndex:(index:number)=>void}) {
  const camera=rig?.cameras[selectedCamera];
  const anchor=nearestFeatureAnchor([184,73]) as {index:[number,number];anchor:[number,number];delta:[number,number]};
  return <LabShell id="geometry" eyebrow="PAUSE 01 · GEOMETRY" title="Trace one anchor">
    <CameraStrip rig={rig} selected={selectedCamera} enabled={Array(6).fill(true)} onSelect={setSelectedCamera} />
    <div className="geometry-workbench">
      <figure className="torn-photo"><img src={asset(camera?.network_image??"")} alt={`${short(camera?.name??"camera")} network input`} /><figcaption>network image · 352 × 128</figcaption></figure>
      <div className="transform-thread"><span><b>network</b><code>[{f(anchor.anchor[0],2)}, {f(anchor.anchor[1],2)}]</code></span><i>→</i><span><b>undo A,a</b><code>raw pixel</code></span><i>→</i><span><b>K⁻¹</b><code>camera ray</code></span><i>→</i><span><b>R,+t</b><code>ego meters</code></span></div>
      <div className="matrix-note"><span>K · camera intrinsics</span>{camera?.cam2img.map((row,index)=><code key={index}>{row.map((value)=>f(value,2).padStart(9," ")).join(" ")}</code>)}</div>
    </div>
    <div className="depth-panel"><div><b>latent depth allocation</b><span>selected {depthIndex+4} m</span></div><DepthSketch values={depth} selected={depthIndex} onSelect={setDepthIndex} /></div>
  </LabShell>;
}

function BevLab({ rig,selectedCamera,setSelectedCamera,bevMode,setBevMode,threshold,setThreshold,opacity,setOpacity,rawGrid,setRawGrid,stats,activeVariant }:{rig:Rig|null;selectedCamera:number;setSelectedCamera:(index:number)=>void;bevMode:BevMode;setBevMode:(mode:BevMode)=>void;threshold:number;setThreshold:(value:number)=>void;opacity:number;setOpacity:(value:number)=>void;rawGrid:boolean;setRawGrid:(value:boolean)=>void;stats:ReturnType<typeof binaryStats>|null;activeVariant?:Variant}) {
  const modes:[BevMode,string][]=[["probability","Watercolor probability"],["threshold","Threshold mask"],["gt","nuScenes GT"],["errors","TP / FP / FN"],["lidar","LiDAR occupancy"],["contributors","Frustum contributors"]];
  return <LabShell id="bev" eyebrow="PAUSE 02 · READ THE MAP" title="Separate prediction from evidence">
    <CameraStrip rig={rig} selected={selectedCamera} enabled={Array(6).fill(true)} onSelect={setSelectedCamera} />
    <div className="bev-workbench"><div className="ink-tabs">{modes.map(([value,label])=><button key={value} className={bevMode===value?"active":""} onClick={()=>setBevMode(value)}>{label}</button>)}</div><figure className="checkpoint-thumb"><img src={asset(activeVariant?.image??"/data/model/bev-all-cameras.png")} alt="Real checkpoint vehicle BEV" /><figcaption>CHECKPOINT OUTPUT · not LiDAR input</figcaption></figure></div>
    <div className="range-row"><label>threshold <b>{f(threshold,2)}</b><input type="range" min=".05" max=".95" step=".05" value={threshold} onChange={(event)=>setThreshold(Number(event.target.value))} /></label><label>pigment <b>{f(opacity,2)}</b><input type="range" min=".15" max="1" step=".05" value={opacity} onChange={(event)=>setOpacity(Number(event.target.value))} /></label><button className={rawGrid?"active":""} onClick={()=>setRawGrid(!rawGrid)}>raw 200 × 200 grid</button></div>
    {stats&&<div className="metric-stamps"><span><b>{stats.truePositive.toLocaleString()}</b>TP</span><span><b>{stats.falsePositive.toLocaleString()}</b>FP</span><span><b>{stats.falseNegative.toLocaleString()}</b>FN</span><span><b>{f(stats.iou,3)}</b>single-frame IoU</span></div>}
  </LabShell>;
}

function RobustnessLab({ rig,selectedCamera,setSelectedCamera,enabled,setEnabled,yaw,setYaw,trajectories,selectedTrajectory,setSelectedTrajectory,temperature,setTemperature }:{rig:Rig|null;selectedCamera:number;setSelectedCamera:(index:number)=>void;enabled:boolean[];setEnabled:(value:boolean[])=>void;yaw:number;setYaw:(value:number)=>void;trajectories:Trajectory[];selectedTrajectory:number;setSelectedTrajectory:(value:number)=>void;temperature:number;setTemperature:(value:number)=>void}) {
  const toggle=(index:number)=>{const next=Array(6).fill(true);if(enabled[index])next[index]=false;setEnabled(next);};
  return <LabShell id="robustness" eyebrow="PAUSE 03 · STRESS THE IDEA" title="Remove evidence, then choose a path">
    <CameraStrip rig={rig} selected={selectedCamera} enabled={enabled} onSelect={setSelectedCamera} onToggle={toggle} />
    <div className="robust-row"><div><b>CAM_FRONT calibration yaw</b><div className="ink-tabs">{[-3,0,3].map((value)=><button className={yaw===value?"active":""} key={value} onClick={()=>setYaw(value)}>{value>0?"+":""}{value}°</button>)}</div></div><label>Boltzmann temperature <b>{f(temperature,2)}</b><input type="range" min=".2" max="2" step=".1" value={temperature} onChange={(event)=>setTemperature(Number(event.target.value))} /></label></div>
    <div className="trajectory-ledger">{trajectories.slice(0,7).map((trajectory,index)=><button key={trajectory.name} className={selectedTrajectory===index?"active":""} onClick={()=>setSelectedTrajectory(index)}><i /><span>{trajectory.name}</span><code>cost {f(trajectory.cost,3)}</code><b>{f(trajectory.probability*100,1)}%</b></button>)}</div>
    <p className="teaching-warning"><EvidenceTag value="TEACHING" /> No planning checkpoint was released. The paths and cost field reconstruct the paper equation; dropout and yaw views use cached checkpoint outputs.</p>
  </LabShell>;
}

export default function LssExplainer() {
  const [activeScene,setActiveScene]=useState(0),[progress,setProgress]=useState(1),[contentsOpen,setContentsOpen]=useState(false),[playing,setPlaying]=useState(false),[labOpen,setLabOpen]=useState(false);
  const [rig,setRig]=useState<Rig|null>(null),[features,setFeatures]=useState<Features|null>(null),[model,setModel]=useState<Model|null>(null);
  const [selectedCamera,setSelectedCamera]=useState(1),[depthIndex,setDepthIndex]=useState(15);
  const [geometryStep,setGeometryStep]=useState(2),[poolingMode,setPoolingMode]=useState<PoolingMode>("sum"),[poolOffset,setPoolOffset]=useState(.34);
  const [bevMode,setBevMode]=useState<BevMode>("probability"),[threshold,setThreshold]=useState(.5),[bevOpacity,setBevOpacity]=useState(.82),[rawGrid,setRawGrid]=useState(false);
  const [enabled,setEnabled]=useState<boolean[]>(Array(6).fill(true)),[yaw,setYaw]=useState(0),[temperature,setTemperature]=useState(.8),[selectedTrajectory,setSelectedTrajectory]=useState(0);

  useEffect(()=>{Promise.all([
    fetch(asset("/data/rig.json")).then((response)=>response.json()),fetch(asset("/data/model-features.json")).then((response)=>response.json()),fetch(asset("/data/model-artifacts.json")).then((response)=>response.json())
  ]).then(([rigData,featuresData,modelData])=>{setRig(rigData);setFeatures(featuresData);setModel(modelData);}).catch((error)=>console.error("Evidence assets failed",error));},[]);

  useEffect(()=>{const frame=requestAnimationFrame(()=>{const initial=sceneIndexFromHash(location.hash);setActiveScene(initial);setGeometryStep(GEOMETRY_DEFAULTS[SCENES[initial].id]??2);});return()=>cancelAnimationFrame(frame);},[]);

  const go=useCallback((index:number,traceStep?:number)=>{const next=Math.max(0,Math.min(SCENES.length-1,index));setActiveScene(next);setGeometryStep(traceStep??GEOMETRY_DEFAULTS[SCENES[next].id]??2);setProgress(0);setLabOpen(false);setContentsOpen(false);history.replaceState(null,"",`#${SCENES[next].id}`);requestAnimationFrame(()=>setProgress(1));},[]);
  useEffect(()=>{const handler=(event:KeyboardEvent)=>{if(event.key==="Escape")setContentsOpen(false);if(event.key==="ArrowDown"||event.key==="ArrowRight"){event.preventDefault();go(activeScene+1);}if(event.key==="ArrowUp"||event.key==="ArrowLeft"){event.preventDefault();go(activeScene-1);}};addEventListener("keydown",handler);return()=>removeEventListener("keydown",handler);},[activeScene,go]);
  useEffect(()=>{if(!playing)return;const timer=setInterval(()=>setActiveScene(current=>{const next=(current+1)%SCENES.length;setGeometryStep(GEOMETRY_DEFAULTS[SCENES[next].id]??2);history.replaceState(null,"",`#${SCENES[next].id}`);setProgress(0);requestAnimationFrame(()=>setProgress(1));return next;}),6500);return()=>clearInterval(timer);},[playing]);

  const allDepth=useMemo(()=>decodeFloat(features?.depth_probabilities),[features]);
  const depth=useMemo(()=>Array.from({length:41},(_,d)=>allDepth?.[selectedCamera*41*8*22+d*8*22+4*22+11]??0),[allDepth,selectedCamera]);
  const activeDrop=enabled.findIndex((value)=>!value),variantKey=activeDrop>=0?`drop-${CAMERA_NAMES[activeDrop].toLowerCase().replaceAll("_","-")}`:`front-yaw-${yaw>=0?"+":""}${yaw}`;
  const activeVariant=model?.variants[variantKey]??model?.variants["all-cameras"];
  const logits=useMemo(()=>decodeFloat(activeVariant?.logits),[activeVariant]);
  const probability=useMemo(()=>logits?Float32Array.from(logits,(value)=>1/(1+Math.exp(-Math.max(-40,Math.min(40,value))))):null,[logits]);
  const groundTruth=useMemo(()=>decodeUint8(model?.ground_truth.mask),[model]);
  const stats=useMemo(()=>probability&&groundTruth?binaryStats(Array.from(probability),Array.from(groundTruth),threshold):null,[probability,groundTruth,threshold]);
  const trajectories=useMemo<Trajectory[]>(()=>{const paths=Array.from({length:9},(_,index)=>Array.from({length:21},(_,sample)=>{const y=sample*.72;return[(index-4)*.017*y*y,y] as [number,number];}));const map=([x,y]:[number,number])=>.055*Math.abs(x)+.85*Math.exp(-((x-2.3)**2+(y-9)**2)/5)+.5*Math.exp(-((x+1.4)**2+(y-13)**2)/3),costs=paths.map((path)=>trajectoryCost(path,map,.14)),probabilities=boltzmannProbabilities(costs,temperature);return costs.map((cost,index)=>({name:index===costs.indexOf(Math.min(...costs))?"minimum cost":`template ${index+1}`,points:paths[index],cost,probability:probabilities[index]})).sort((a,b)=>a.cost-b.cost);},[temperature]);
  const scene=SCENES[activeScene];
  const geometryScene=["image-to-ray","ray-to-camera","camera-to-ego"].includes(scene.id);
  const traceGeometry=(step:number)=>{const target=SCENES.findIndex(item=>item.id===GEOMETRY_TARGETS[step]);go(target,step);};
  const poolCopy:Record<PoolingMode,{operation:string;detail:string;formula:string}>={sum:{operation:"grouped SUM",detail:"preserves accumulated evidence · official LSS",formula:"Σ fᵢ"},mean:{operation:"grouped MEAN",detail:"normalizes away contributor count · comparison",formula:"Σ fᵢ / n"},max:{operation:"channel-wise MAX",detail:"keeps the strongest response · comparison",formula:"maxᵢ fᵢ"},bilinear:{operation:"4-neighbor weighted SPLAT",detail:"smoothly spreads one candidate · comparison",formula:"Σ wᵢⱼ fᵢ"}};
  const tensor=geometryScene?TRACE_TENSORS[geometryStep]:scene.id==="splat-pooling"?{...scene.tensor,operation:poolCopy[poolingMode].operation,detail:poolCopy[poolingMode].detail}:scene.tensor;

  const activeLab=scene.lab==="geometry"?<GeometryLab rig={rig} selectedCamera={selectedCamera} setSelectedCamera={setSelectedCamera} depth={depth} depthIndex={depthIndex} setDepthIndex={setDepthIndex} />:scene.lab==="bev"?<BevLab rig={rig} selectedCamera={selectedCamera} setSelectedCamera={setSelectedCamera} bevMode={bevMode} setBevMode={setBevMode} threshold={threshold} setThreshold={setThreshold} opacity={bevOpacity} setOpacity={setBevOpacity} rawGrid={rawGrid} setRawGrid={setRawGrid} stats={stats} activeVariant={activeVariant} />:scene.lab==="robustness"?<RobustnessLab rig={rig} selectedCamera={selectedCamera} setSelectedCamera={setSelectedCamera} enabled={enabled} setEnabled={setEnabled} yaw={yaw} setYaw={setYaw} trajectories={trajectories} selectedTrajectory={selectedTrajectory} setSelectedTrajectory={setSelectedTrajectory} temperature={temperature} setTemperature={setTemperature} />:null;

  return <main className={`visual-essay scene-${activeScene} ${labOpen?"lab-open":""}`}>
    <header className="essay-header"><a href={asset("/")}><span>LSS</span><b>EXPLAINED</b><sup>v10</sup></a><div><button onClick={()=>setContentsOpen(!contentsOpen)} aria-expanded={contentsOpen}><Menu />Contents</button><a href={asset("/articles/lift-splat-shoot-source-notes.md")}>Source notes</a><a href="https://github.com/hova88/lss-explained">GitHub ↗</a></div></header>
    {contentsOpen&&<nav className="contents-drawer" aria-label="Table of contents"><button className="drawer-close" onClick={()=>setContentsOpen(false)}><X /></button><p>FIELD INDEX · 10 SCENES</p>{SCENES.map((item,index)=><button key={item.id} className={index===activeScene?"active":""} onClick={()=>go(index)}><span>{String(index+1).padStart(2,"0")}</span><b>{item.title}</b><small>{item.act}</small></button>)}</nav>}

    <section className="persistent-stage" aria-live="polite">
      <IllustrationStage scene={scene} progress={progress} selectedCamera={selectedCamera} depthIndex={depthIndex} geometryStep={geometryStep} poolingMode={poolingMode} poolOffset={poolOffset} cameraPoses={rig?.cameras} onCameraSelect={setSelectedCamera} onDepthSelect={setDepthIndex} />
      <StoryStep index={activeScene} />
      {scene.id!=="encoder-supervision"&&<aside className="lecture-note"><b>{String(activeScene+1).padStart(2,"0")}.</b><span>{scene.steps[activeScene%3].text}</span></aside>}
      <div className="tensor-ledger" aria-label="Tensor operation for this scene">
        <code>{tensor.input}</code>
        <span><b>{tensor.operation}</b><small>{tensor.detail}</small></span>
        <code>{tensor.output}</code>
      </div>
      {geometryScene&&<div className="stage-control frame-control"><small>ONE SAMPLE · FIVE COORDINATE STATES</small><div>{TRACE_LABELS.map((label,index)=><button key={label} className={geometryStep===index?"active":""} aria-current={geometryStep===index?"step":undefined} title={`Show ${label} in ${GEOMETRY_TARGETS[index]}`} onClick={()=>traceGeometry(index)}>{index}<span>{label}</span></button>)}</div></div>}
      {scene.id==="splat-pooling"&&<div className="stage-control pool-control"><small>COLLISION RULE · <b>{poolCopy[poolingMode].formula}</b></small><div>{(["sum","mean","max","bilinear"] as PoolingMode[]).map(mode=><button key={mode} className={poolingMode===mode?"active":""} onClick={()=>setPoolingMode(mode)}>{mode}</button>)}</div><label>move candidate<input type="range" min="0" max="1" step=".01" value={poolOffset} onChange={event=>setPoolOffset(Number(event.target.value))} /></label></div>}
      <div className="gesture-note">{geometryScene?"SELECT A STATE · THE SAMPLE ID, CAMERA AND DEPTH STAY FIXED":scene.id==="splat-pooling"?"CHOOSE A REDUCTION · MOVE THE RUST CANDIDATE":"DRAG TO ROTATE · SCROLL TO ZOOM · CLICK CAMERA OR DEPTH"}</div>
      {scene.lab&&<button className="lab-toggle" onClick={()=>setLabOpen(!labOpen)}>{labOpen?"Close evidence":"Open evidence"}</button>}
    </section>
    {labOpen&&activeLab&&<div className="evidence-drawer">{activeLab}</div>}
    <nav className="lesson-timeline" aria-label="Lesson progress"><button onClick={()=>go(activeScene-1)} disabled={activeScene===0}><ChevronLeft /></button><button className="play-control" onClick={()=>setPlaying(!playing)} aria-label={playing?"Pause":"Play"}>{playing?<Pause />:<Play />}</button><div>{SCENES.map((item,index)=><button key={item.id} className={index===activeScene?"active":""} onClick={()=>go(index)} aria-label={`Scene ${index+1}: ${item.title}`}><i /><span>{item.title}</span></button>)}</div><button onClick={()=>go(activeScene+1)} disabled={activeScene===SCENES.length-1}><ChevronRight /></button></nav>
  </main>;
}
