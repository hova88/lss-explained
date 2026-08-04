"use client";

import { Canvas, ThreeEvent, useFrame, useLoader, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export const CAMERA_NAMES = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"] as const;

export type SceneSelection =
  | {kind:"camera"; index:number}
  | {kind:"ray"; pixel:[number,number]; depth:number}
  | {kind:"depth"; bin:number; meters:number; probability:number}
  | {kind:"cell"; index:[number,number]; center:[number,number]; contributors:string[]; counts?:number[]}
  | {kind:"trajectory"; index:number; cost:number; probability:number};

export type Trajectory = {name:string; points:[number,number][]; cost:number; probability:number};

const FALLBACK_CAMERAS:[number,number,number,number][] = [
  [-1.2,1.5,1.7,-.55], [0,1.7,1.75,0], [1.2,1.5,1.7,.55],
  [-1.1,-1.4,1.65,-2.55], [0,-1.5,1.7,Math.PI], [1.1,-1.4,1.65,2.55],
];

function asset(path:string){
  const prefix=process.env.NEXT_PUBLIC_BASE_PATH??"";
  return `${prefix}${path}`;
}

function Line({points,color="#161918",opacity=1,dashed=false}:{points:[number,number,number][];color?:string;opacity?:number;dashed?:boolean}){
  const positions=useMemo(()=>new Float32Array(points.flat()),[points]);
  return <lineSegments><bufferGeometry><bufferAttribute attach="attributes-position" args={[positions,3]}/></bufferGeometry>{dashed?<lineDashedMaterial color={color} transparent opacity={opacity} dashSize={.32} gapSize={.22}/>:<lineBasicMaterial color={color} transparent opacity={opacity}/>}</lineSegments>;
}

function PaperGrid(){
  const positions=useMemo(()=>{const values:number[]=[];for(let i=-20;i<=20;i+=2){values.push(-20,i,-.04,20,i,-.04,i,-20,-.04,i,20,-.04)}return new Float32Array(values)},[]);
  return <lineSegments><bufferGeometry><bufferAttribute attach="attributes-position" args={[positions,3]}/></bufferGeometry><lineBasicMaterial color="#a9aaa3" transparent opacity={.23}/></lineSegments>;
}

function EgoCar(){
  return <group>
    <mesh position={[0,0,.42]}><boxGeometry args={[1.9,4.2,.8]}/><meshPhysicalMaterial color="#e9e4d7" roughness={.55} clearcoat={.3}/></mesh>
    <mesh position={[0,.15,1]}><boxGeometry args={[1.55,1.9,.7]}/><meshPhysicalMaterial color="#7fa6b5" transparent opacity={.55} roughness={.2}/></mesh>
    <Line points={[[0,0,.1],[0,5,.1]]} color="#d95b35" opacity={.7}/>
  </group>;
}

function CameraGlyph({index,pose,enabled,selected,onSelect}:{index:number;pose:[number,number,number,number];enabled:boolean;selected:boolean;onSelect:(selection:SceneSelection)=>void}){
  const [x,y,z,yaw]=pose;
  const forward:[number,number,number]=[Math.sin(yaw)*5,Math.cos(yaw)*5,0];
  const left:[number,number,number]=[Math.sin(yaw-.42)*4.2,Math.cos(yaw-.42)*4.2,-1.25];
  const right:[number,number,number]=[Math.sin(yaw+.42)*4.2,Math.cos(yaw+.42)*4.2,-1.25];
  const topLeft:[number,number,number]=[Math.sin(yaw-.42)*4.2,Math.cos(yaw-.42)*4.2,1.25];
  const topRight:[number,number,number]=[Math.sin(yaw+.42)*4.2,Math.cos(yaw+.42)*4.2,1.25];
  const origin:[number,number,number]=[x,y,z];
  const corners=[left,right,topRight,topLeft].map(([a,b,c])=>[x+a,y+b,z+c] as [number,number,number]);
  const lines:[number,number,number][]=[];corners.forEach(corner=>lines.push(origin,corner));for(let i=0;i<4;i+=1)lines.push(corners[i],corners[(i+1)%4]);
  return <group>
    <mesh position={origin} rotation={[0,0,-yaw]} onPointerDown={event=>{event.stopPropagation();onSelect({kind:"camera",index})}} onPointerOver={event=>{event.stopPropagation();document.body.style.cursor="pointer"}} onPointerOut={()=>{document.body.style.cursor="default"}}>
      <boxGeometry args={[.68,.45,.4]}/><meshStandardMaterial color={selected?"#d85b35":enabled?"#171918":"#a7a69f"}/>
    </mesh>
    <Line points={lines} color={selected?"#d85b35":enabled?"#5d8797":"#bbb9af"} opacity={enabled?.55:.15}/>
    <Line points={[origin,[x+forward[0],y+forward[1],z]]} color={selected?"#d85b35":"#202322"} opacity={enabled?.42:.12}/>
  </group>;
}

function Ray({pixel,depthIndex,origin,points,onSelect}:{pixel:[number,number];depthIndex:number;origin:[number,number,number];points:[number,number,number][];onSelect:(selection:SceneSelection)=>void}){
  const depth=4+depthIndex;
  const selected=points[depthIndex]??points[0]??origin;
  const last=points.at(-1)??selected;
  return <group>
    <Line points={[origin,last]} color="#d85b35" opacity={.82}/>
    <mesh position={selected} onPointerDown={event=>{event.stopPropagation();onSelect({kind:"ray",pixel,depth})}}><sphereGeometry args={[.28,22,22]}/><meshBasicMaterial color="#d85b35"/></mesh>
  </group>;
}

function LiftedDepth({probabilities,depthIndex,points,onSelect}:{probabilities:number[];depthIndex:number;points:[number,number,number][];onSelect:(selection:SceneSelection)=>void}){
  return <group>{probabilities.map((probability,index)=>{
    const position=points[index]??[0,0,0];
    const radius=.08+Math.sqrt(Math.max(0,probability))*.75;
    return <mesh key={index} position={position} onPointerDown={(event:ThreeEvent<PointerEvent>)=>{event.stopPropagation();onSelect({kind:"depth",bin:index,meters:4+index,probability})}}>
      <sphereGeometry args={[radius,index===depthIndex?22:12,index===depthIndex?22:12]}/><meshBasicMaterial color={index===depthIndex?"#d85b35":"#6d4a8a"} transparent opacity={.24+Math.min(.7,probability*4)}/>
    </mesh>})}</group>;
}

function SplatCloud(){
  const points=useMemo(()=>{const values:number[]=[];for(let camera=0;camera<6;camera+=1){for(let i=0;i<90;i+=1){const angle=camera*Math.PI/3+(i%11-5)*.035;const r=3+(i*7%38)*.37;values.push(Math.sin(angle)*r,Math.cos(angle)*r,.08+(i%4)*.03)}}return new Float32Array(values)},[]);
  return <points><bufferGeometry><bufferAttribute attach="attributes-position" args={[points,3]}/></bufferGeometry><pointsMaterial color="#6e4b89" size={.12} transparent opacity={.54} sizeAttenuation/></points>;
}

function BevPlane({image,onSelect}:{image:string;onSelect:(selection:SceneSelection)=>void}){
  const loadedTexture=useLoader(THREE.TextureLoader,asset(image));
  const texture=useMemo(()=>{const copy=loadedTexture.clone();copy.colorSpace=THREE.SRGBColorSpace;copy.magFilter=THREE.NearestFilter;copy.wrapS=THREE.RepeatWrapping;copy.repeat.x=-1;copy.offset.x=1;copy.needsUpdate=true;return copy},[loadedTexture]);
  useEffect(()=>()=>texture.dispose(),[texture]);
  return <mesh position={[0,0,.02]} onPointerDown={event=>{
    event.stopPropagation();if(!event.uv)return;const x=Math.min(199,Math.max(0,Math.floor(event.uv.y*200)));const y=Math.min(199,Math.max(0,Math.floor((1-event.uv.x)*200)));
    onSelect({kind:"cell",index:[x,y],center:[x*.5-49.75,y*.5-49.75],contributors:[]});
  }} onPointerOver={event=>{event.stopPropagation();document.body.style.cursor="crosshair"}} onPointerOut={()=>{document.body.style.cursor="default"}}>
    <planeGeometry args={[36,36]}/><meshBasicMaterial map={texture} transparent opacity={.9} side={THREE.DoubleSide} depthWrite={false}/>
  </mesh>;
}

function TrajectoryLines({trajectories,selected,onSelect}:{trajectories:Trajectory[];selected:number;onSelect:(selection:SceneSelection)=>void}){
  return <group>{trajectories.map((trajectory,index)=>{
    const curve=new THREE.CatmullRomCurve3(trajectory.points.map(([x,y])=>new THREE.Vector3(x,y,.35)));
    const geometry=new THREE.TubeGeometry(curve,32,index===selected ? .14 : .075,8,false);
    return <mesh key={trajectory.name} geometry={geometry} onPointerMove={event=>{event.stopPropagation();onSelect({kind:"trajectory",index,cost:trajectory.cost,probability:trajectory.probability})}} onPointerDown={event=>{event.stopPropagation();onSelect({kind:"trajectory",index,cost:trajectory.cost,probability:trajectory.probability})}}>
      <meshBasicMaterial color={index===selected?"#d85b35":index===0?"#347f82":"#6c6d68"} transparent opacity={index===selected?1:.45}/>
    </mesh>})}</group>;
}

function CameraMotion({step}:{step:number}){
  const {camera,gl}=useThree();const controls=useRef<OrbitControls|null>(null);
  useEffect(()=>{const orbit=new OrbitControls(camera,gl.domElement);orbit.enableDamping=true;orbit.dampingFactor=.075;orbit.minDistance=7;orbit.maxDistance=75;orbit.target.set(0,4,0);orbit.update();controls.current=orbit;return()=>orbit.dispose()},[camera,gl]);
  useFrame(()=>{
    const destinations:Record<number,[THREE.Vector3,THREE.Vector3]>={
      0:[new THREE.Vector3(20,-23,17),new THREE.Vector3(0,0,1)],
      1:[new THREE.Vector3(10,-15,9),new THREE.Vector3(0,8,2)],
      2:[new THREE.Vector3(7,-11,6),new THREE.Vector3(0,11,1.5)],
      3:[new THREE.Vector3(20,-18,15),new THREE.Vector3(0,7,0)],
      4:[new THREE.Vector3(0,-1,39),new THREE.Vector3(0,0,0)],
      5:[new THREE.Vector3(0,-2,42),new THREE.Vector3(0,0,0)],
      6:[new THREE.Vector3(0,-3,40),new THREE.Vector3(0,0,0)],
      7:[new THREE.Vector3(0,-4,41),new THREE.Vector3(0,0,0)],
      8:[new THREE.Vector3(18,-21,23),new THREE.Vector3(0,8,0)],
      9:[new THREE.Vector3(18,-23,17),new THREE.Vector3(0,1,0)],
    };
    const [position,target]=destinations[step];camera.position.lerp(position,.045);controls.current?.target.lerp(target,.045);camera.up.set(0,0,1);controls.current?.update();
  });
  return null;
}

function World({step,enabledCameras,selectedCamera,pixel,depthIndex,depthProbability,bevImage,trajectories,selectedTrajectory,cameraPoses,liftedPoints,onSelect}:{step:number;enabledCameras:boolean[];selectedCamera:number;pixel:[number,number];depthIndex:number;depthProbability:number[];bevImage:string;trajectories:Trajectory[];selectedTrajectory:number;cameraPoses:[number,number,number,number][];liftedPoints:[number,number,number][];onSelect:(selection:SceneSelection)=>void}){
  return <>
    <color attach="background" args={["#f1eee5"]}/><ambientLight intensity={2.1}/><directionalLight position={[12,-8,22]} intensity={2.4}/>
    <PaperGrid/><EgoCar/><CameraMotion step={step}/>
    {(step===0||step===1||step===2||step===3||step===7||step===9)&&cameraPoses.map((pose,index)=><CameraGlyph key={index} index={index} pose={pose} enabled={enabledCameras[index]} selected={selectedCamera===index} onSelect={onSelect}/>)}
    {(step===1||step===2||step===3)&&<Ray pixel={pixel} depthIndex={depthIndex} origin={cameraPoses[selectedCamera].slice(0,3) as [number,number,number]} points={liftedPoints} onSelect={onSelect}/>}
    {(step===2||step===3)&&<LiftedDepth probabilities={depthProbability} depthIndex={depthIndex} points={liftedPoints} onSelect={onSelect}/>}
    {(step===3||step===4)&&<SplatCloud/>}
    {(step>=4&&step!==8)&&<BevPlane image={bevImage} onSelect={onSelect}/>}
    {step===8&&<><BevPlane image={bevImage} onSelect={onSelect}/><TrajectoryLines trajectories={trajectories} selected={selectedTrajectory} onSelect={onSelect}/></>}
    <fog attach="fog" args={["#f1eee5",35,78]}/>
  </>;
}

export function LssScene(props:{step:number;enabledCameras:boolean[];selectedCamera:number;pixel:[number,number];depthIndex:number;depthProbability:number[];bevImage:string;trajectories:Trajectory[];selectedTrajectory:number;cameraMatrices?:number[][][];liftedPoints:[number,number,number][];onSelect:(selection:SceneSelection)=>void}){
  const cameraPoses=useMemo(()=>props.cameraMatrices?.map(matrix=>{
    const translation=[-matrix[1][3],matrix[0][3],matrix[2][3]] as [number,number,number];
    const opticalAxis=[-matrix[1][2],matrix[0][2],matrix[2][2]];
    return [...translation,Math.atan2(opticalAxis[0],opticalAxis[1])] as [number,number,number,number];
  })??FALLBACK_CAMERAS,[props.cameraMatrices]);
  const {cameraMatrices,...worldProps}=props;
  void cameraMatrices;
  return <div className="lss-scene" aria-label="Interactive 3D LSS geometry scene"><Canvas dpr={[1,1.7]} camera={{position:[20,-23,17],fov:39,near:.1,far:130}} gl={{antialias:true,alpha:false}}><World {...worldProps} cameraPoses={cameraPoses}/></Canvas></div>;
}
