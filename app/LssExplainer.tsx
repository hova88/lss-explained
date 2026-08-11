"use client";
/* eslint-disable @next/next/no-img-element */

import { ChevronDown, Menu, MousePointer2, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { binaryStats, boltzmannProbabilities, float16LittleEndianToFloat32, nearestFeatureAnchor, trajectoryCost } from "../lib/algorithm.mjs";
import { IllustrationStage } from "./IllustrationStage";
import { BevMode, CAMERA_NAMES, LssScene, SceneSelection, Trajectory, Vehicle, type SceneCamera } from "./LssScene";
import { SCENES, sceneIndexFromHash, type LabId } from "./lss-content";

type EncodedArray = { shape:number[]; dtype:string; data:string };
type CameraRecord = SceneCamera & { name:string; image:string; network_image:string; post_rot:number[][]; post_trans:number[]; timestamp:number; augmentation:{resize:number;resize_dims:number[];crop:number[]} };
type Rig = { sample_token:string; timestamp:number; cameras:CameraRecord[]; lidar2ego:number[][]; ego2global:number[][]; vehicles_ego:Vehicle[] };
type Variant = { image:string; logits:EncodedArray; probability_min:number; probability_max:number; probability_mean:number; evidence:string };
type Model = { variants:Record<string,Variant>; ground_truth:{mask:EncodedArray;image:string}; geometry_contributors:{counts:EncodedArray}; tensor_checks:{finite:boolean;depth_probability_sum_max_error:number} };
type Features = { depth_probabilities:EncodedArray; context_features:EncodedArray; feature_anchors:{x:number[];y:number[]} };
type Alignment = { lidar_occupancy:{counts:EncodedArray}; lidar:{point_count:number;sha256:string}; camera_projections:{camera:string;valid_points:number;delta_to_lidar_ms:number}[] };

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
function decodeUint16(encoded?:EncodedArray){const bytes=decodeBytes(encoded);return bytes?new Uint16Array(bytes.buffer):null;}

function EvidenceTag({ value }:{value:string}) { return <span className={`evidence-tag evidence-${value.toLowerCase().replaceAll(" ","-")}`}>{value}</span>; }

function StoryStep({ index, active, setRef }:{index:number;active:boolean;setRef:(element:HTMLElement|null)=>void}) {
  const scene=SCENES[index];
  return (
    <article ref={setRef} id={scene.id} className={`story-step ${active?"active":""}`} data-scene={index}>
      <div className="step-index"><span>{String(index+1).padStart(2,"0")}</span><i /></div>
      <p className="act-label">{scene.act}</p>
      <h2>{scene.title}</h2>
      <p className="scene-question">{scene.question}</p>
      <p className="scene-reveal">{scene.reveal}</p>
      <ol>{scene.beats.map((beat,beatIndex)=><li key={beat}><i>{beatIndex+1}</i><span>{beat}</span></li>)}</ol>
      <code className="scene-formula">{scene.formula}</code>
      <footer><EvidenceTag value={scene.evidence} /><span>{scene.source}</span></footer>
      {index<SCENES.length-1&&<div className="scroll-cue">keep tracing <ChevronDown /></div>}
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

function LabShell({ id, title, eyebrow, children, onExplore, mode }:{id:LabId;title:string;eyebrow:string;children:React.ReactNode;onExplore:()=>void;mode:"guided"|"explore"}) {
  return <section className={`lab-break lab-${id}`} data-lab={id}><div className="lab-heading"><p>{eyebrow}</p><h2>{title}</h2><button onClick={onExplore} className={mode==="explore"?"active":""}><MousePointer2 />{mode==="explore"?"Exploring — drag the stage":"Explore the stage"}</button></div><div className="lab-controls">{children}</div></section>;
}

function GeometryLab({ rig,selectedCamera,setSelectedCamera,depth,depthIndex,setDepthIndex,mode,setMode }:{rig:Rig|null;selectedCamera:number;setSelectedCamera:(index:number)=>void;depth:number[];depthIndex:number;setDepthIndex:(index:number)=>void;mode:"guided"|"explore";setMode:(mode:"guided"|"explore")=>void}) {
  const camera=rig?.cameras[selectedCamera];
  const anchor=nearestFeatureAnchor([184,73]) as {index:[number,number];anchor:[number,number];delta:[number,number]};
  return <LabShell id="geometry" eyebrow="LAB 01 · CALIBRATED VISION" title="Touch the exact path from pixel to ego" onExplore={()=>setMode(mode==="explore"?"guided":"explore")} mode={mode}>
    <CameraStrip rig={rig} selected={selectedCamera} enabled={Array(6).fill(true)} onSelect={setSelectedCamera} />
    <div className="geometry-workbench">
      <figure className="torn-photo"><img src={asset(camera?.network_image??"")} alt={`${short(camera?.name??"camera")} network input`} /><figcaption>network image · 352 × 128</figcaption></figure>
      <div className="transform-thread"><span><b>network</b><code>[{f(anchor.anchor[0],2)}, {f(anchor.anchor[1],2)}]</code></span><i>→</i><span><b>undo A,a</b><code>raw pixel</code></span><i>→</i><span><b>K⁻¹</b><code>camera ray</code></span><i>→</i><span><b>R,+t</b><code>ego meters</code></span></div>
      <div className="matrix-note"><span>K · camera intrinsics</span>{camera?.cam2img.map((row,index)=><code key={index}>{row.map((value)=>f(value,2).padStart(9," ")).join(" ")}</code>)}</div>
    </div>
    <div className="depth-panel"><div><b>latent depth allocation</b><span>selected {depthIndex+4} m</span></div><DepthSketch values={depth} selected={depthIndex} onSelect={setDepthIndex} /></div>
  </LabShell>;
}

function BevLab({ rig,selectedCamera,setSelectedCamera,bevMode,setBevMode,threshold,setThreshold,opacity,setOpacity,rawGrid,setRawGrid,stats,mode,setMode,activeVariant }:{rig:Rig|null;selectedCamera:number;setSelectedCamera:(index:number)=>void;bevMode:BevMode;setBevMode:(mode:BevMode)=>void;threshold:number;setThreshold:(value:number)=>void;opacity:number;setOpacity:(value:number)=>void;rawGrid:boolean;setRawGrid:(value:boolean)=>void;stats:ReturnType<typeof binaryStats>|null;mode:"guided"|"explore";setMode:(mode:"guided"|"explore")=>void;activeVariant?:Variant}) {
  const modes:[BevMode,string][]=[["probability","Watercolor probability"],["threshold","Threshold mask"],["gt","nuScenes GT"],["errors","TP / FP / FN"],["lidar","LiDAR occupancy"],["contributors","Frustum contributors"]];
  return <LabShell id="bev" eyebrow="LAB 02 · LINKED EVIDENCE" title="Read one place in three coordinate systems" onExplore={()=>setMode(mode==="explore"?"guided":"explore")} mode={mode}>
    <CameraStrip rig={rig} selected={selectedCamera} enabled={Array(6).fill(true)} onSelect={setSelectedCamera} />
    <div className="bev-workbench"><div className="ink-tabs">{modes.map(([value,label])=><button key={value} className={bevMode===value?"active":""} onClick={()=>setBevMode(value)}>{label}</button>)}</div><figure className="checkpoint-thumb"><img src={asset(activeVariant?.image??"/data/model/bev-all-cameras.png")} alt="Real checkpoint vehicle BEV" /><figcaption>CHECKPOINT OUTPUT · not LiDAR input</figcaption></figure></div>
    <div className="range-row"><label>threshold <b>{f(threshold,2)}</b><input type="range" min=".05" max=".95" step=".05" value={threshold} onChange={(event)=>setThreshold(Number(event.target.value))} /></label><label>pigment <b>{f(opacity,2)}</b><input type="range" min=".15" max="1" step=".05" value={opacity} onChange={(event)=>setOpacity(Number(event.target.value))} /></label><button className={rawGrid?"active":""} onClick={()=>setRawGrid(!rawGrid)}>raw 200 × 200 grid</button></div>
    {stats&&<div className="metric-stamps"><span><b>{stats.truePositive.toLocaleString()}</b>TP</span><span><b>{stats.falsePositive.toLocaleString()}</b>FP</span><span><b>{stats.falseNegative.toLocaleString()}</b>FN</span><span><b>{f(stats.iou,3)}</b>single-frame IoU</span></div>}
  </LabShell>;
}

function RobustnessLab({ rig,selectedCamera,setSelectedCamera,enabled,setEnabled,yaw,setYaw,trajectories,selectedTrajectory,setSelectedTrajectory,temperature,setTemperature,mode,setMode }:{rig:Rig|null;selectedCamera:number;setSelectedCamera:(index:number)=>void;enabled:boolean[];setEnabled:(value:boolean[])=>void;yaw:number;setYaw:(value:number)=>void;trajectories:Trajectory[];selectedTrajectory:number;setSelectedTrajectory:(value:number)=>void;temperature:number;setTemperature:(value:number)=>void;mode:"guided"|"explore";setMode:(mode:"guided"|"explore")=>void}) {
  const toggle=(index:number)=>{const next=Array(6).fill(true);if(enabled[index])next[index]=false;setEnabled(next);};
  return <LabShell id="robustness" eyebrow="LAB 03 · FAILURE AND ACTION" title="Remove evidence, perturb geometry, choose a path" onExplore={()=>setMode(mode==="explore"?"guided":"explore")} mode={mode}>
    <CameraStrip rig={rig} selected={selectedCamera} enabled={enabled} onSelect={setSelectedCamera} onToggle={toggle} />
    <div className="robust-row"><div><b>CAM_FRONT calibration yaw</b><div className="ink-tabs">{[-3,0,3].map((value)=><button className={yaw===value?"active":""} key={value} onClick={()=>setYaw(value)}>{value>0?"+":""}{value}°</button>)}</div></div><label>Boltzmann temperature <b>{f(temperature,2)}</b><input type="range" min=".2" max="2" step=".1" value={temperature} onChange={(event)=>setTemperature(Number(event.target.value))} /></label></div>
    <div className="trajectory-ledger">{trajectories.slice(0,7).map((trajectory,index)=><button key={trajectory.name} className={selectedTrajectory===index?"active":""} onClick={()=>setSelectedTrajectory(index)}><i /><span>{trajectory.name}</span><code>cost {f(trajectory.cost,3)}</code><b>{f(trajectory.probability*100,1)}%</b></button>)}</div>
    <p className="teaching-warning"><EvidenceTag value="TEACHING" /> No planning checkpoint was released. The paths and cost field reconstruct the paper equation; dropout and yaw views use cached checkpoint outputs.</p>
  </LabShell>;
}

function SelectionNote({ selection, onClose }:{selection:SceneSelection;onClose:()=>void}) {
  const rows = selection.kind==="camera"?[["camera",short(CAMERA_NAMES[selection.index])],["optical axis","cam2ego rotation column 3"]]:selection.kind==="depth"?[["depth bin",`${selection.bin}`],["metric depth",`${selection.meters} m`],["allocation",f(selection.probability,6)]]:selection.kind==="cell"?[["tensor index",`[${selection.index.join(", ")}]`],["ego center",`[${selection.center.map((value)=>f(value,2)).join(", ")}] m`]]:selection.kind==="lidar"?[["point id",`${selection.index}`],["LiDAR xyz",selection.lidar.map((value)=>f(value,3)).join(", ")],["ego xyz",selection.ego.map((value)=>f(value,3)).join(", ")]]:selection.kind==="trajectory"?[["template",`${selection.index+1}`],["cost",f(selection.cost,4)],["probability",f(selection.probability,5)]]:[["object",`${selection.index}`]];
  return <aside className="selection-note"><button onClick={onClose} aria-label="Close inspection"><X /></button><p>FIELD INSPECTION</p>{rows.map(([label,value])=><div key={label}><b>{label}</b><code>{value}</code></div>)}</aside>;
}

export default function LssExplainer() {
  const [activeScene,setActiveScene]=useState(0),[progress,setProgress]=useState(0),[mode,setMode]=useState<"guided"|"explore">("guided"),[contentsOpen,setContentsOpen]=useState(false);
  const [rig,setRig]=useState<Rig|null>(null),[features,setFeatures]=useState<Features|null>(null),[model,setModel]=useState<Model|null>(null),[alignment,setAlignment]=useState<Alignment|null>(null),[lidar,setLidar]=useState<Float32Array|null>(null);
  const [selectedCamera,setSelectedCamera]=useState(1),[depthIndex,setDepthIndex]=useState(15),[selection,setSelection]=useState<SceneSelection|null>(null),[selectedLidar,setSelectedLidar]=useState<number|null>(null);
  const [bevMode,setBevMode]=useState<BevMode>("probability"),[threshold,setThreshold]=useState(.5),[bevOpacity,setBevOpacity]=useState(.82),[rawGrid,setRawGrid]=useState(false);
  const [enabled,setEnabled]=useState<boolean[]>(Array(6).fill(true)),[yaw,setYaw]=useState(0),[temperature,setTemperature]=useState(.8),[selectedTrajectory,setSelectedTrajectory]=useState(0);
  const stepRefs=useRef<(HTMLElement|null)[]>([]);

  useEffect(()=>{Promise.all([
    fetch(asset("/data/rig.json")).then((response)=>response.json()),fetch(asset("/data/model-features.json")).then((response)=>response.json()),fetch(asset("/data/model-artifacts.json")).then((response)=>response.json()),fetch(asset("/data/alignment.json")).then((response)=>response.json()),fetch(asset("/data/lidar-frame.bin")).then((response)=>response.arrayBuffer())
  ]).then(([rigData,featuresData,modelData,alignmentData,lidarData])=>{setRig(rigData);setFeatures(featuresData);setModel(modelData);setAlignment(alignmentData);setLidar(new Float32Array(lidarData));}).catch((error)=>console.error("Evidence assets failed",error));},[]);

  useEffect(()=>{
    const initial=sceneIndexFromHash(location.hash);
    const observer=new IntersectionObserver((entries)=>{const visible=entries.filter((entry)=>entry.isIntersecting).sort((a,b)=>b.intersectionRatio-a.intersectionRatio)[0];if(!visible)return;const index=Number((visible.target as HTMLElement).dataset.scene);setActiveScene(index);history.replaceState(null,"",`#${SCENES[index].id}`);setMode("guided");}, {rootMargin:"-25% 0px -50% 0px",threshold:[.05,.2,.45,.7]});
    const frame=requestAnimationFrame(()=>{setActiveScene(initial);stepRefs.current[initial]?.scrollIntoView({block:"center"});stepRefs.current.forEach((element)=>element&&observer.observe(element));});
    return()=>{cancelAnimationFrame(frame);observer.disconnect();};
  },[]);

  useEffect(()=>{let frame=0;const update=()=>{frame=0;const element=stepRefs.current[activeScene];if(!element)return;const rect=element.getBoundingClientRect(),range=Math.max(1,rect.height+innerHeight*.35);setProgress(Math.max(0,Math.min(1,(innerHeight*.55-rect.top)/range)));};const handler=()=>{if(!frame)frame=requestAnimationFrame(update);};addEventListener("scroll",handler,{passive:true});update();return()=>{removeEventListener("scroll",handler);if(frame)cancelAnimationFrame(frame);};},[activeScene]);

  const go=useCallback((index:number)=>{stepRefs.current[Math.max(0,Math.min(SCENES.length-1,index))]?.scrollIntoView({behavior:matchMedia("(prefers-reduced-motion: reduce)").matches?"auto":"smooth",block:"center"});setContentsOpen(false);},[]);
  useEffect(()=>{const handler=(event:KeyboardEvent)=>{if(event.key==="Escape"){setMode("guided");setSelection(null);setContentsOpen(false);}if(event.key==="ArrowDown"||event.key==="ArrowRight"){event.preventDefault();go(activeScene+1);}if(event.key==="ArrowUp"||event.key==="ArrowLeft"){event.preventDefault();go(activeScene-1);}};addEventListener("keydown",handler);return()=>removeEventListener("keydown",handler);},[activeScene,go]);

  const allDepth=useMemo(()=>decodeFloat(features?.depth_probabilities),[features]);
  const depth=useMemo(()=>Array.from({length:41},(_,d)=>allDepth?.[selectedCamera*41*8*22+d*8*22+4*22+11]??0),[allDepth,selectedCamera]);
  const activeDrop=enabled.findIndex((value)=>!value),variantKey=activeDrop>=0?`drop-${CAMERA_NAMES[activeDrop].toLowerCase().replaceAll("_","-")}`:`front-yaw-${yaw>=0?"+":""}${yaw}`;
  const activeVariant=model?.variants[variantKey]??model?.variants["all-cameras"];
  const logits=useMemo(()=>decodeFloat(activeVariant?.logits),[activeVariant]);
  const probability=useMemo(()=>logits?Float32Array.from(logits,(value)=>1/(1+Math.exp(-Math.max(-40,Math.min(40,value))))):null,[logits]);
  const groundTruth=useMemo(()=>decodeUint8(model?.ground_truth.mask),[model]);
  const contributors=useMemo(()=>decodeUint16(model?.geometry_contributors.counts),[model]);
  const lidarOccupancy=useMemo(()=>decodeUint16(alignment?.lidar_occupancy.counts),[alignment]);
  const stats=useMemo(()=>probability&&groundTruth?binaryStats(Array.from(probability),Array.from(groundTruth),threshold):null,[probability,groundTruth,threshold]);
  const trajectories=useMemo<Trajectory[]>(()=>{const paths=Array.from({length:9},(_,index)=>Array.from({length:21},(_,sample)=>{const y=sample*.72;return[(index-4)*.017*y*y,y] as [number,number];}));const map=([x,y]:[number,number])=>.055*Math.abs(x)+.85*Math.exp(-((x-2.3)**2+(y-9)**2)/5)+.5*Math.exp(-((x+1.4)**2+(y-13)**2)/3),costs=paths.map((path)=>trajectoryCost(path,map,.14)),probabilities=boltzmannProbabilities(costs,temperature);return costs.map((cost,index)=>({name:index===costs.indexOf(Math.min(...costs))?"minimum cost":`template ${index+1}`,points:paths[index],cost,probability:probabilities[index]})).sort((a,b)=>a.cost-b.cost);},[temperature]);
  const handleSelect=(value:SceneSelection)=>{if(value.kind==="camera")setSelectedCamera(value.index);if(value.kind==="depth")setDepthIndex(value.bin);if(value.kind==="lidar")setSelectedLidar(value.index);if(value.kind==="trajectory")setSelectedTrajectory(value.index);setSelection(value);};
  const scene=SCENES[activeScene],showThree=scene.stageView!=="illustration"||mode==="explore";

  return <main className={`visual-essay scene-${activeScene} mode-${mode}`}>
    <header className="essay-header"><a href={asset("/")}><span>LSS</span><b>EXPLAINED</b><sup>v5</sup></a><div><button onClick={()=>setContentsOpen(!contentsOpen)} aria-expanded={contentsOpen}><Menu />Contents</button><a href={asset("/articles/lift-splat-shoot-source-notes.md")}>Source notes</a><a href="https://github.com/hova88/lss-explained">GitHub ↗</a></div></header>
    <div className="reading-progress" style={{"--progress":`${((activeScene+progress)/SCENES.length)*100}%`} as React.CSSProperties}><i /></div>
    {contentsOpen&&<nav className="contents-drawer" aria-label="Table of contents"><button className="drawer-close" onClick={()=>setContentsOpen(false)}><X /></button><p>FIELD INDEX · 12 SCENES</p>{SCENES.map((item,index)=><button key={item.id} className={index===activeScene?"active":""} onClick={()=>go(index)}><span>{String(index+1).padStart(2,"0")}</span><b>{item.title}</b><small>{item.act}</small></button>)}</nav>}

    <section className="persistent-stage" aria-live="polite">
      <div className={`stage-layer illustration-layer ${showThree?"behind":"front"}`}><IllustrationStage scene={scene} progress={progress} selectedCamera={selectedCamera} depthIndex={depthIndex} /></div>
      <div className={`stage-layer three-layer ${showThree?"front":"behind"}`}><LssScene sceneIndex={activeScene} mode={mode} cameras={rig?.cameras??[]} enabledCameras={enabled} selectedCamera={selectedCamera} depthIndex={depthIndex} depthProbability={depth} lidar={lidar} lidar2ego={rig?.lidar2ego??[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]} selectedLidar={selectedLidar} vehicles={rig?.vehicles_ego??[]} bevProbability={probability} groundTruth={groundTruth} lidarOccupancy={lidarOccupancy} contributors={contributors} bevMode={bevMode} threshold={threshold} bevOpacity={bevOpacity} rawGrid={rawGrid} trajectories={trajectories} selectedTrajectory={selectedTrajectory} onSelect={handleSelect} /></div>
      <div className="stage-caption"><span>{String(activeScene+1).padStart(2,"0")} / {SCENES.length}</span><b>{scene.title}</b><small>{showThree?mode==="explore"?"drag · pinch · click":"calibrated 3D stage":"deterministic ink plate"}</small></div>
      {scene.lab&&<button className="explore-toggle" onClick={()=>setMode(mode==="guided"?"explore":"guided")}><MousePointer2 />{mode==="guided"?"Explore":"Return to story"}</button>}
    </section>

    <div className="story-column">
      {SCENES.map((_,index)=><div key={SCENES[index].id}><StoryStep index={index} active={index===activeScene} setRef={(element)=>{stepRefs.current[index]=element;}} />
        {index===6&&<GeometryLab rig={rig} selectedCamera={selectedCamera} setSelectedCamera={setSelectedCamera} depth={depth} depthIndex={depthIndex} setDepthIndex={setDepthIndex} mode={mode} setMode={setMode} />}
        {index===10&&<BevLab rig={rig} selectedCamera={selectedCamera} setSelectedCamera={setSelectedCamera} bevMode={bevMode} setBevMode={setBevMode} threshold={threshold} setThreshold={setThreshold} opacity={bevOpacity} setOpacity={setBevOpacity} rawGrid={rawGrid} setRawGrid={setRawGrid} stats={stats} mode={mode} setMode={setMode} activeVariant={activeVariant} />}
        {index===11&&<RobustnessLab rig={rig} selectedCamera={selectedCamera} setSelectedCamera={setSelectedCamera} enabled={enabled} setEnabled={setEnabled} yaw={yaw} setYaw={setYaw} trajectories={trajectories} selectedTrajectory={selectedTrajectory} setSelectedTrajectory={setSelectedTrajectory} temperature={temperature} setTemperature={setTemperature} mode={mode} setMode={setMode} />}
      </div>)}
      <footer className="essay-footer"><p>One real frame. One explicit coordinate chain. No hidden depth sensor.</p><div><EvidenceTag value="PAPER" /><EvidenceTag value="OFFICIAL CODE" /><EvidenceTag value="CHECKPOINT" /><EvidenceTag value="REAL SAMPLE" /></div><a href="#the-problem" onClick={(event)=>{event.preventDefault();go(0);}}>Trace it again ↑</a></footer>
    </div>
    {selection&&<SelectionNote selection={selection} onClose={()=>setSelection(null)} />}
  </main>;
}
