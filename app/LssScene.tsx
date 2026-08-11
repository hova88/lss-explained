"use client";

import { Canvas, ThreeEvent, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { cameraFrustumCorners } from "../lib/algorithm.mjs";

export const CAMERA_NAMES = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"] as const;
export type BevMode = "probability" | "threshold" | "gt" | "errors" | "lidar" | "contributors";
export type SceneSelection =
  | { kind: "camera"; index: number }
  | { kind: "depth"; bin: number; meters: number; probability: number }
  | { kind: "lidar"; index: number; lidar: [number, number, number]; ego: [number, number, number] }
  | { kind: "object"; index: number }
  | { kind: "cell"; index: [number, number]; center: [number, number] }
  | { kind: "trajectory"; index: number; cost: number; probability: number };
export type Trajectory = { name: string; points: [number, number][]; cost: number; probability: number };
export type Vehicle = { center_ego: number[]; dimensions: number[]; yaw_ego: number; label?: number; num_lidar_pts?: number };
export type SceneCamera = { cam2ego: number[][]; cam2img: number[][]; imageSize?: [number, number] };

const C = { paper: "#f3ecde", ink: "#232724", blue: "#315c9c", rust: "#c35e3c", sage: "#607b70", ochre: "#d4aa51" };
const egoScene = (point: number[]): [number, number, number] => [-point[1], point[0], point[2]];

function Line({ points, color = C.ink, opacity = 1, dashed = false }: { points: [number, number, number][]; color?: string; opacity?: number; dashed?: boolean }) {
  const geometry = useMemo(() => new THREE.BufferGeometry().setFromPoints(points.map((point) => new THREE.Vector3(...point))), [points]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return <lineSegments geometry={geometry}>{dashed ? <lineDashedMaterial color={color} transparent opacity={opacity} dashSize={0.45} gapSize={0.25} /> : <lineBasicMaterial color={color} transparent opacity={opacity} />}</lineSegments>;
}

function PaperGrid() {
  const lines = useMemo(() => {
    const values: [number, number, number][] = [];
    for (let i = -50; i <= 50; i += 5) values.push([-50, i, -0.05], [50, i, -0.05], [i, -50, -0.05], [i, 50, -0.05]);
    return values;
  }, []);
  return <group><Line points={lines} color="#898a82" opacity={0.1} /><Line points={[[-50,0,-.045],[50,0,-.045],[0,-50,-.045],[0,50,-.045]]} color={C.ink} opacity={.18} />{[-40,-20,0,20,40].map((x) => <mesh key={x} position={[x,0,-0.055]}><circleGeometry args={[0.08,12]} /><meshBasicMaterial color={C.ink} /></mesh>)}</group>;
}

function MetricHalo() {
  const rings=useMemo(()=>{const values:[number,number,number][]=[];for(const radius of [10,20,30,40])for(let i=0;i<72;i+=1){const a=i/72*Math.PI*2,b=(i+1)/72*Math.PI*2;values.push([Math.cos(a)*radius,Math.sin(a)*radius,.055],[Math.cos(b)*radius,Math.sin(b)*radius,.055]);}return values;},[]);
  return <group><Line points={rings} color={C.sage} opacity={.105} /><Line points={[[0,2.2,.07],[0,45,.07],[0,45,.07],[-.42,44.25,.07],[0,45,.07],[.42,44.25,.07]]} color={C.rust} opacity={.38} /></group>;
}

function InkEdges({ geometry, color = C.ink, opacity = 0.82 }: { geometry: THREE.BufferGeometry; color?: string; opacity?: number }) {
  const edges = useMemo(() => new THREE.EdgesGeometry(geometry, 22), [geometry]);
  useEffect(() => () => edges.dispose(), [edges]);
  return <group><lineSegments geometry={edges}><lineBasicMaterial color={color} transparent opacity={opacity} /></lineSegments><lineSegments geometry={edges} position={[0.025,-0.018,0.016]}><lineBasicMaterial color={color} transparent opacity={opacity*.2} /></lineSegments></group>;
}

function InkCar() {
  const body = useMemo(() => {
    const shape = new THREE.Shape();
    shape.moveTo(0,2.3);
    shape.bezierCurveTo(.62,2.23,.88,1.72,.91,.98);
    shape.lineTo(.91,-1.25);
    shape.bezierCurveTo(.88,-1.9,.57,-2.22,0,-2.28);
    shape.bezierCurveTo(-.57,-2.22,-.88,-1.9,-.91,-1.25);
    shape.lineTo(-.91,.98);
    shape.bezierCurveTo(-.88,1.72,-.62,2.23,0,2.3);
    shape.closePath();
    const geometry = new THREE.ExtrudeGeometry(shape, { depth: .34, bevelEnabled: true, bevelSize: .075, bevelThickness: .065, bevelSegments: 3, curveSegments: 16 });
    geometry.translate(0,0,.18);
    return geometry;
  }, []);
  const cabin = useMemo(() => {
    const shape = new THREE.Shape();
    shape.moveTo(0,1.02);shape.bezierCurveTo(.48,.96,.61,.54,.59,.05);shape.lineTo(.52,-.78);shape.bezierCurveTo(.3,-1.02,-.3,-1.02,-.52,-.78);shape.lineTo(-.59,.05);shape.bezierCurveTo(-.61,.54,-.48,.96,0,1.02);shape.closePath();
    const geometry = new THREE.ExtrudeGeometry(shape,{depth:.15,bevelEnabled:true,bevelSize:.035,bevelThickness:.025,bevelSegments:2,curveSegments:12});
    geometry.translate(0,.08,.6); return geometry;
  }, []);
  useEffect(() => () => { body.dispose(); cabin.dispose(); }, [body, cabin]);
  return (
    <group>
      <mesh geometry={body}><meshToonMaterial color="#eee6d7" /><InkEdges geometry={body} opacity={.74} /></mesh>
      <mesh geometry={cabin}><meshToonMaterial color="#7897a5" transparent opacity={.5} /><InkEdges geometry={cabin} color={C.blue} opacity={.48} /></mesh>
      <mesh position={[0,.08,.82]}><circleGeometry args={[.105,20]} /><meshBasicMaterial color={C.ochre} transparent opacity={.9} side={THREE.DoubleSide} /></mesh>
      <Line points={[[0,.25,.84],[0,1.42,.84],[0,1.42,.84],[-.17,1.16,.84],[0,1.42,.84],[.17,1.16,.84]]} color={C.rust} opacity={.9} />
    </group>
  );
}

function transformCameraPoint(point: [number, number, number], matrix: number[][]) {
  const ego = [
    matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2] * point[2] + matrix[0][3],
    matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2] * point[2] + matrix[1][3],
    matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2] * point[2] + matrix[2][3],
  ];
  return egoScene(ego);
}

function frustumCorners(camera: SceneCamera, depth: number) {
  const [width, height] = camera.imageSize ?? [1600, 900];
  return cameraFrustumCorners(camera.cam2img,camera.cam2ego,width,height,depth).map(egoScene);
}

function CameraInk({ camera, index, selected, enabled, onSelect }: { camera: SceneCamera; index: number; selected: boolean; enabled: boolean; onSelect: (value: SceneSelection) => void }) {
  const origin = egoScene(camera.cam2ego.slice(0,3).map((row) => row[3]));
  const near = frustumCorners(camera, 1.7), far = frustumCorners(camera, 7.4);
  const geometry = useMemo(() => {
    const vertices: number[] = [];
    for (let i=0;i<4;i+=1) vertices.push(...origin,...far[i],...far[(i+1)%4]);
    const g = new THREE.BufferGeometry(); g.setAttribute("position",new THREE.Float32BufferAttribute(vertices,3)); return g;
  }, [origin, far]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  const edgePoints: [number,number,number][] = [];
  far.forEach((corner) => edgePoints.push(origin,corner));
  for(let i=0;i<4;i+=1) edgePoints.push(near[i],near[(i+1)%4],far[i],far[(i+1)%4]);
  const axisEnd = transformCameraPoint([0,0,8.6],camera.cam2ego);
  const orientation = useMemo(() => {
    const direction = new THREE.Vector3(...axisEnd).sub(new THREE.Vector3(...origin)).normalize();
    return new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0,0,1),direction);
  },[axisEnd,origin]);
  const housing = useMemo(() => new THREE.BoxGeometry(.58,.44,.64),[]);
  useEffect(() => () => housing.dispose(),[housing]);
  return (
    <group>
      <group position={origin} quaternion={orientation} onPointerDown={(event) => { event.stopPropagation(); onSelect({kind:"camera",index}); }}>
        <mesh geometry={housing}><meshToonMaterial color={selected ? C.rust : enabled ? "#ddd4c4" : "#aaa59a"} /><InkEdges geometry={housing} opacity={enabled ? .8 : .18} /></mesh>
        <mesh position={[0,0,.38]} rotation={[Math.PI/2,0,0]}><cylinderGeometry args={[.17,.2,.17,18]} /><meshToonMaterial color={enabled?"#6f8c95":"#aaa59a"} transparent opacity={.9} /></mesh>
        <mesh position={[0,0,.48]}><circleGeometry args={[.13,20]} /><meshBasicMaterial color={selected?C.ochre:C.blue} transparent opacity={enabled ? .84 : .12} side={THREE.DoubleSide} /></mesh>
        <mesh position={[0,.29,-.09]}><boxGeometry args={[.28,.12,.16]} /><meshToonMaterial color={C.ink} /></mesh>
      </group>
      <mesh position={origin}><sphereGeometry args={[selected?0.48:0.31,18,18]} /><meshBasicMaterial color={C.ochre} transparent opacity={selected?0.23:0.04} depthWrite={false} /></mesh>
      <mesh geometry={geometry}><meshBasicMaterial color={selected?C.rust:C.blue} transparent opacity={enabled?(selected?0.09:0.026):0.008} side={THREE.DoubleSide} depthWrite={false} /></mesh>
      <Line points={edgePoints} color={selected?C.rust:C.blue} opacity={enabled?(selected?0.68:0.2):0.06} dashed={!selected} />
      <Line points={[origin,axisEnd]} color={selected?C.rust:C.ink} opacity={enabled ? (selected ? .8 : .32) : .06} />
    </group>
  );
}

function LiftedRay({ camera, depthIndex, probabilities, onSelect }: { camera: SceneCamera; depthIndex: number; probabilities: number[]; onSelect: (value: SceneSelection) => void }) {
  const u=800,v=450,fx=camera.cam2img[0][0],fy=camera.cam2img[1][1],cx=camera.cam2img[0][2],cy=camera.cam2img[1][2];
  const points = Array.from({length:41},(_,index) => transformCameraPoint([((u-cx)/fx)*(4+index),((v-cy)/fy)*(4+index),4+index],camera.cam2ego));
  return <group><Line points={[points[0],points.at(-1)!]} color={C.rust} opacity={0.72} />{points.map((point,index) => { const p=probabilities[index]??0; return <mesh key={index} position={point} onPointerDown={(event)=>{event.stopPropagation();onSelect({kind:"depth",bin:index,meters:4+index,probability:p});}}><sphereGeometry args={[index===depthIndex?0.26:0.055+Math.sqrt(Math.max(0,p))*0.5,index===depthIndex?18:8,index===depthIndex?18:8]} /><meshBasicMaterial color={index===depthIndex?C.rust:C.blue} transparent opacity={index===depthIndex?1:0.18+Math.min(0.6,p*5)} /></mesh>;})}</group>;
}

function rgba(prob: number, truth: boolean, mode: BevMode, threshold: number, opacity: number, lidar: number, contributors: number): [number,number,number,number] {
  if(mode==="gt") return truth?[96,123,112,Math.round(210*opacity)]:[0,0,0,0];
  if(mode==="lidar") return lidar>0?[49,92,156,Math.round(Math.min(225,60+Math.log2(lidar+1)*52)*opacity)]:[0,0,0,0];
  if(mode==="contributors") return contributors>0?[212,170,81,Math.round(Math.min(220,35+contributors*28)*opacity)]:[0,0,0,0];
  const predicted=prob>=threshold;
  if(mode==="threshold") return predicted?[195,94,60,Math.round(215*opacity)]:[0,0,0,0];
  if(mode==="errors"){if(predicted&&truth)return[96,123,112,220];if(predicted)return[195,94,60,220];if(truth)return[49,92,156,220];return[0,0,0,0];}
  const alpha=Math.max(0,Math.min(.88,(prob-.08)/.58))*opacity;
  return [195+Math.round(35*prob),94+Math.round(90*prob),60,Math.round(255*alpha)];
}

function BevPaper({ probability, groundTruth, mode, threshold, opacity, rawGrid, lidarOccupancy, contributors, onSelect }: { probability: Float32Array|null; groundTruth: Uint8Array|null; mode: BevMode; threshold:number; opacity:number; rawGrid:boolean; lidarOccupancy?:Uint16Array|null; contributors?:Uint16Array|null; onSelect:(value:SceneSelection)=>void }) {
  const texture = useMemo(() => {
    const data=new Uint8Array(200*200*4);
    for(let ix=0;ix<200;ix+=1)for(let iy=0;iy<200;iy+=1){const source=ix*200+iy,dest=(ix*200+(199-iy))*4;let contributor=0;if(contributors)for(let cam=0;cam<6;cam+=1)contributor+=contributors[cam*40000+source]??0;data.set(rgba(probability?.[source]??0,(groundTruth?.[source]??0)>0,mode,threshold,opacity,lidarOccupancy?.[source]??0,contributor),dest);}
    const t=new THREE.DataTexture(data,200,200,THREE.RGBAFormat);t.colorSpace=THREE.SRGBColorSpace;t.magFilter=rawGrid?THREE.NearestFilter:THREE.LinearFilter;t.minFilter=THREE.LinearFilter;t.needsUpdate=true;return t;
  },[probability,groundTruth,mode,threshold,opacity,rawGrid,lidarOccupancy,contributors]);
  useEffect(()=>()=>texture.dispose(),[texture]);
  return <group><mesh position={[0,0,0.015]} onPointerDown={(event)=>{event.stopPropagation();if(!event.uv)return;const ix=Math.min(199,Math.max(0,Math.floor(event.uv.y*200))),iy=Math.min(199,Math.max(0,Math.floor((1-event.uv.x)*200)));onSelect({kind:"cell",index:[ix,iy],center:[ix*.5-49.75,iy*.5-49.75]});}}><planeGeometry args={[100,100,rawGrid?200:1,rawGrid?200:1]} /><meshBasicMaterial map={texture} transparent side={THREE.DoubleSide} depthWrite={false} wireframe={false} /></mesh></group>;
}

function LidarInk({ points, lidar2ego, selected, onSelect }: { points:Float32Array|null; lidar2ego:number[][]; selected:number|null; onSelect:(value:SceneSelection)=>void }) {
  const prepared=useMemo(()=>{if(!points)return null;const count=points.length/5,positions=new Float32Array(count*3),colors=new Float32Array(count*3),egoPoints=new Float32Array(count*3);for(let i=0;i<count;i+=1){const x=points[i*5],y=points[i*5+1],z=points[i*5+2],ex=lidar2ego[0][0]*x+lidar2ego[0][1]*y+lidar2ego[0][2]*z+lidar2ego[0][3],ey=lidar2ego[1][0]*x+lidar2ego[1][1]*y+lidar2ego[1][2]*z+lidar2ego[1][3],ez=lidar2ego[2][0]*x+lidar2ego[2][1]*y+lidar2ego[2][2]*z+lidar2ego[2][3];positions.set([-ey,ex,ez],i*3);egoPoints.set([ex,ey,ez],i*3);const t=Math.max(0,Math.min(1,(ez+3)/7));colors.set([.19+.56*t,.36+.32*(1-t),.61-.34*t],i*3);}return{positions,colors,egoPoints};},[points,lidar2ego]);
  if(!prepared||!points)return null;const highlight=selected==null?null:[prepared.positions[selected*3],prepared.positions[selected*3+1],prepared.positions[selected*3+2]] as [number,number,number];
  return <group><points onPointerDown={(event:ThreeEvent<PointerEvent>)=>{event.stopPropagation();const index=event.index??0;onSelect({kind:"lidar",index,lidar:[points[index*5],points[index*5+1],points[index*5+2]],ego:[prepared.egoPoints[index*3],prepared.egoPoints[index*3+1],prepared.egoPoints[index*3+2]]});}}><bufferGeometry><bufferAttribute attach="attributes-position" args={[prepared.positions,3]} /><bufferAttribute attach="attributes-color" args={[prepared.colors,3]} /></bufferGeometry><pointsMaterial vertexColors size={0.11} transparent opacity={0.72} sizeAttenuation /></points>{highlight&&<mesh position={highlight}><sphereGeometry args={[0.3,16,16]} /><meshBasicMaterial color={C.ochre} /></mesh>}</group>;
}

function VehicleBoxes({ vehicles, onSelect }: { vehicles:Vehicle[]; onSelect:(value:SceneSelection)=>void }) {
  return <group>{vehicles.map((vehicle,index)=>{const [x,y,z]=vehicle.center_ego,[length,width,height]=vehicle.dimensions,yaw=vehicle.yaw_ego,corners:[number,number,number][]=[];for(const [lx,ly] of [[-length/2,-width/2],[length/2,-width/2],[length/2,width/2],[-length/2,width/2]] as [number,number][]){const ex=x+lx*Math.cos(yaw)-ly*Math.sin(yaw),ey=y+lx*Math.sin(yaw)+ly*Math.cos(yaw);corners.push(egoScene([ex,ey,z-height/2+.08]));}const lines:[number,number,number][]=[];for(let i=0;i<4;i+=1)lines.push(corners[i],corners[(i+1)%4]);return <group key={index} onPointerDown={(event)=>{event.stopPropagation();onSelect({kind:"object",index});}}><Line points={lines} color={C.sage} opacity={0.95} /></group>;})}</group>;
}

function TrajectoryInk({ items, selected, onSelect }: { items:Trajectory[]; selected:number; onSelect:(value:SceneSelection)=>void }) {
  return <group>{items.map((item,index)=>{const curve=new THREE.CatmullRomCurve3(item.points.map(([x,y])=>new THREE.Vector3(-x,y,.3))),geometry=new THREE.TubeGeometry(curve,34,index===selected?.15:.055,7,false);return <mesh key={item.name} geometry={geometry} onPointerMove={(event)=>{event.stopPropagation();onSelect({kind:"trajectory",index,cost:item.cost,probability:item.probability});}}><meshBasicMaterial color={index===selected?C.rust:C.blue} transparent opacity={index===selected?1:.32} /></mesh>;})}</group>;
}

function CameraDirector({ sceneIndex, mode }: { sceneIndex:number; mode:"guided"|"explore" }) {
  const {camera,gl}=useThree(); const controls=useRef<OrbitControls|null>(null); const key=useRef(""); const settled=useRef(false);
  useEffect(()=>{const orbit=new OrbitControls(camera,gl.domElement);orbit.enableDamping=true;orbit.enablePan=mode==="explore";orbit.enableRotate=mode==="explore";orbit.enableZoom=mode==="explore";orbit.dampingFactor=.07;orbit.minDistance=7;orbit.maxDistance=125;orbit.target.set(0,5,0);orbit.update();controls.current=orbit;return()=>orbit.dispose();},[camera,gl,mode]);
  useFrame(()=>{const next=`${sceneIndex}-${mode}`;if(key.current!==next){key.current=next;settled.current=false;}const top=sceneIndex>=7;let destination:THREE.Vector3,target:THREE.Vector3;if(top){destination=new THREE.Vector3(0,-5,74);target=new THREE.Vector3(0,0,0);}else if(sceneIndex===2){destination=new THREE.Vector3(14,-17,11);target=new THREE.Vector3(0,1.2,1);}else if(sceneIndex>=4){destination=new THREE.Vector3(21,-27,18);target=new THREE.Vector3(0,7,1);}else{destination=new THREE.Vector3(17,-21,14);target=new THREE.Vector3(0,3,1);}if(mode==="guided"&&!settled.current){camera.position.lerp(destination,.065);controls.current?.target.lerp(target,.065);if(camera.position.distanceTo(destination)<.04)settled.current=true;}camera.up.set(0,0,1);controls.current?.update();}); return null;
}

export function LssScene(props: {
  sceneIndex:number; mode:"guided"|"explore"; cameras:SceneCamera[]; enabledCameras:boolean[]; selectedCamera:number; depthIndex:number; depthProbability:number[];
  lidar:Float32Array|null; lidar2ego:number[][]; selectedLidar:number|null; vehicles:Vehicle[]; bevProbability:Float32Array|null; groundTruth:Uint8Array|null;
  lidarOccupancy?:Uint16Array|null; contributors?:Uint16Array|null; bevMode:BevMode; threshold:number; bevOpacity:number; rawGrid:boolean; trajectories:Trajectory[]; selectedTrajectory:number; onSelect:(value:SceneSelection)=>void;
}) {
  const showRig=props.sceneIndex>=2&&props.sceneIndex<=7,showRay=props.sceneIndex>=4&&props.sceneIndex<=6,showBev=props.sceneIndex>=8,showTruth=props.sceneIndex===10,showShoot=props.sceneIndex===11;
  return <div className={`lss-scene ${props.mode}`} aria-label="Interactive ink-rendered LSS geometry scene"><Canvas dpr={[1,1.55]} camera={{position:[17,-21,14],fov:34,near:.1,far:190}} gl={{antialias:true,alpha:true}}><ambientLight intensity={2.1} /><directionalLight position={[12,-9,26]} intensity={2.4} /><PaperGrid />{showBev&&<MetricHalo />}<InkCar /><CameraDirector sceneIndex={props.sceneIndex} mode={props.mode} />{showRig&&props.cameras.map((camera,index)=><CameraInk key={index} camera={camera} index={index} selected={props.selectedCamera===index} enabled={props.enabledCameras[index]} onSelect={props.onSelect} />)}{showRay&&props.cameras[props.selectedCamera]&&<LiftedRay camera={props.cameras[props.selectedCamera]} depthIndex={props.depthIndex} probabilities={props.depthProbability} onSelect={props.onSelect} />}{showBev&&<BevPaper probability={props.bevProbability} groundTruth={props.groundTruth} mode={props.bevMode} threshold={props.threshold} opacity={props.bevOpacity} rawGrid={props.rawGrid} lidarOccupancy={props.lidarOccupancy} contributors={props.contributors} onSelect={props.onSelect} />}{showTruth&&<><LidarInk points={props.lidar} lidar2ego={props.lidar2ego} selected={props.selectedLidar} onSelect={props.onSelect} /><VehicleBoxes vehicles={props.vehicles} onSelect={props.onSelect} /></>}{showShoot&&<TrajectoryInk items={props.trajectories} selected={props.selectedTrajectory} onSelect={props.onSelect} />}<fog attach="fog" args={[C.paper,70,155]} /></Canvas></div>;
}
