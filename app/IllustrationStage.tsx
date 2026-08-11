"use client";

import { useEffect, useRef } from "react";
import type { NarrativeScene } from "./lss-content";

type IllustrationStageProps = {
  scene: NarrativeScene;
  progress: number;
  selectedCamera: number;
  depthIndex: number;
};

const palette = {
  paper: "#f3ecde",
  ink: "#232724",
  pencil: "#8d8a80",
  blue: "#315c9c",
  rust: "#c35e3c",
  sage: "#607b70",
  ochre: "#d4aa51",
};

function seeded(seed: number) {
  let value = seed >>> 0;
  return () => ((value = (value * 1664525 + 1013904223) >>> 0) / 4294967296);
}

function sceneSeed(id: string) {
  return [...id].reduce((value, char) => (value * 31 + char.charCodeAt(0)) >>> 0, 2166136261);
}

function roughLine(ctx: CanvasRenderingContext2D, random: () => number, x1: number, y1: number, x2: number, y2: number, color = palette.ink, width = 1.35, alpha = 0.78) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineCap = "round";
  ctx.globalAlpha = alpha;
  for (let pass = 0; pass < 2; pass += 1) {
    ctx.lineWidth = width * (pass ? 0.55 : 1);
    ctx.beginPath();
    ctx.moveTo(x1 + (random() - 0.5) * 2.6, y1 + (random() - 0.5) * 2.6);
    const mx = (x1 + x2) / 2 + (random() - 0.5) * 5;
    const my = (y1 + y2) / 2 + (random() - 0.5) * 5;
    ctx.quadraticCurveTo(mx, my, x2 + (random() - 0.5) * 2.6, y2 + (random() - 0.5) * 2.6);
    ctx.stroke();
  }
  ctx.restore();
}

function roughRect(ctx: CanvasRenderingContext2D, random: () => number, x: number, y: number, w: number, h: number, color = palette.ink, fill?: string) {
  if (fill) {
    ctx.save(); ctx.fillStyle = fill; ctx.globalAlpha = 0.18; ctx.fillRect(x, y, w, h); ctx.restore();
  }
  roughLine(ctx, random, x, y, x + w, y, color);
  roughLine(ctx, random, x + w, y, x + w, y + h, color);
  roughLine(ctx, random, x + w, y + h, x, y + h, color);
  roughLine(ctx, random, x, y + h, x, y, color);
}

function wash(ctx: CanvasRenderingContext2D, random: () => number, x: number, y: number, rx: number, ry: number, color: string, alpha = 0.14) {
  ctx.save(); ctx.fillStyle = color; ctx.globalAlpha = alpha;
  ctx.beginPath();
  for (let index = 0; index < 18; index += 1) {
    const angle = (index / 18) * Math.PI * 2;
    const variance = 0.86 + random() * 0.25;
    const px = x + Math.cos(angle) * rx * variance;
    const py = y + Math.sin(angle) * ry * variance;
    if (index) ctx.lineTo(px, py); else ctx.moveTo(px, py);
  }
  ctx.closePath(); ctx.fill(); ctx.restore();
}

function dot(ctx: CanvasRenderingContext2D, x: number, y: number, radius: number, color: string, alpha = 0.8) {
  ctx.save(); ctx.fillStyle = color; ctx.globalAlpha = alpha; ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.fill(); ctx.restore();
}

function arrow(ctx: CanvasRenderingContext2D, random: () => number, x1: number, y1: number, x2: number, y2: number, color = palette.rust) {
  roughLine(ctx, random, x1, y1, x2, y2, color, 1.6, 0.88);
  const angle = Math.atan2(y2 - y1, x2 - x1);
  roughLine(ctx, random, x2, y2, x2 - Math.cos(angle - 0.5) * 11, y2 - Math.sin(angle - 0.5) * 11, color, 1.6, 0.88);
  roughLine(ctx, random, x2, y2, x2 - Math.cos(angle + 0.5) * 11, y2 - Math.sin(angle + 0.5) * 11, color, 1.6, 0.88);
}

function drawCar(ctx: CanvasRenderingContext2D, random: () => number, x: number, y: number, scale = 1) {
  wash(ctx, random, x, y, 42 * scale, 72 * scale, palette.ochre, 0.12);
  const points = [[-25,-55],[25,-55],[31,-22],[29,45],[18,61],[-18,61],[-29,45],[-31,-22]];
  ctx.save(); ctx.translate(x, y); ctx.scale(scale, scale); ctx.fillStyle = "#eee4d2"; ctx.globalAlpha = 0.92;
  ctx.beginPath(); points.forEach(([px,py], index) => index ? ctx.lineTo(px,py) : ctx.moveTo(px,py)); ctx.closePath(); ctx.fill(); ctx.restore();
  for (let index = 0; index < points.length; index += 1) {
    const a = points[index], b = points[(index + 1) % points.length];
    roughLine(ctx, random, x + a[0] * scale, y + a[1] * scale, x + b[0] * scale, y + b[1] * scale, palette.ink, 1.7);
  }
  roughRect(ctx, random, x - 20 * scale, y - 17 * scale, 40 * scale, 39 * scale, palette.blue, "#9bb4c6");
  arrow(ctx, random, x, y - 68 * scale, x, y - 102 * scale, palette.rust);
}

type Point2 = [number, number];
type Point3 = [number, number, number];

function iso(point: Point3, cx: number, cy: number, scale: number): Point2 {
  const [x,y,z]=point;
  return [cx+(x-y)*.866*scale,cy+(x+y)*.5*scale-z*scale];
}

function roughPolygon(ctx: CanvasRenderingContext2D, random: () => number, points: Point2[], color=palette.ink, fill?:string, alpha=.12) {
  if(fill){ctx.save();ctx.fillStyle=fill;ctx.globalAlpha=alpha;ctx.beginPath();points.forEach(([x,y],index)=>index?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.fill();ctx.restore();}
  points.forEach((point,index)=>roughLine(ctx,random,point[0],point[1],points[(index+1)%points.length][0],points[(index+1)%points.length][1],color,1.15,.62));
}

function drawIsoGrid(ctx: CanvasRenderingContext2D, random: () => number, cx:number, cy:number, scale:number, extent=7) {
  for(let index=-extent;index<=extent;index+=1){
    const a=iso([index,-extent,0],cx,cy,scale),b=iso([index,extent,0],cx,cy,scale),c=iso([-extent,index,0],cx,cy,scale),d=iso([extent,index,0],cx,cy,scale);
    roughLine(ctx,random,a[0],a[1],b[0],b[1],index===0?palette.rust:palette.pencil,index===0?1.05:.55,index===0?.42:.2);
    roughLine(ctx,random,c[0],c[1],d[0],d[1],index===0?palette.blue:palette.pencil,index===0?1.05:.55,index===0?.38:.2);
  }
}

function drawIsoCar(ctx:CanvasRenderingContext2D,random:()=>number,cx:number,cy:number,scale:number) {
  const footprint:Point3[]=[[-.72,-1.5,0],[.72,-1.5,0],[.88,-.85,0],[.82,1.35,0],[0,1.65,0],[-.82,1.35,0],[-.88,-.85,0]];
  const top=footprint.map(([x,y])=>iso([x,y,.38],cx,cy,scale)),bottom=footprint.map((point)=>iso(point,cx,cy,scale));
  roughPolygon(ctx,random,bottom,palette.ink,"#eee6d7",.8);roughPolygon(ctx,random,top,palette.ink,"#eee6d7",.94);
  for(let index=0;index<bottom.length;index+=1)roughLine(ctx,random,bottom[index][0],bottom[index][1],top[index][0],top[index][1],palette.ink,1,.42);
  const cabin=[[-.52,-.45,.42],[.52,-.45,.42],[.54,.72,.42],[0,.98,.42],[-.54,.72,.42]].map((point)=>iso(point as Point3,cx,cy,scale));
  roughPolygon(ctx,random,cabin,palette.blue,palette.blue,.17);
  const origin=iso([0,.15,.48],cx,cy,scale);dot(ctx,origin[0],origin[1],3.4,palette.ochre,.82);
  const forward=iso([0,2.45,.48],cx,cy,scale);arrow(ctx,random,origin[0],origin[1],forward[0],forward[1],palette.rust);
}

function drawSpatialCamera(ctx:CanvasRenderingContext2D,random:()=>number,from:Point2,to:Point2,selected=false) {
  roughRect(ctx,random,from[0]-8,from[1]-6,16,12,selected?palette.rust:palette.ink,selected?palette.rust:palette.ochre);
  const dx=to[0]-from[0],dy=to[1]-from[1],length=Math.hypot(dx,dy)||1,nx=-dy/length,ny=dx/length;
  const left:[number,number]=[to[0]+nx*34,to[1]+ny*34],right:[number,number]=[to[0]-nx*34,to[1]-ny*34];
  ctx.save();ctx.fillStyle=selected?palette.rust:palette.blue;ctx.globalAlpha=selected ? .09 : .035;ctx.beginPath();ctx.moveTo(...from);ctx.lineTo(...left);ctx.lineTo(...right);ctx.closePath();ctx.fill();ctx.restore();
  roughLine(ctx,random,from[0],from[1],left[0],left[1],selected?palette.rust:palette.blue,selected?1.5:.8,selected ? .72 : .28);
  roughLine(ctx,random,from[0],from[1],right[0],right[1],selected?palette.rust:palette.blue,selected?1.5:.8,selected ? .72 : .28);
  roughLine(ctx,random,from[0],from[1],to[0],to[1],selected?palette.rust:palette.pencil,selected?1.7:.8,selected ? .8 : .32);
}

function drawScene(ctx: CanvasRenderingContext2D, width: number, height: number, scene: NarrativeScene, progress: number, camera: number, depth: number) {
  const random = seeded(sceneSeed(scene.id));
  const cx = width * 0.5, cy = height * 0.52;
  ctx.clearRect(0, 0, width, height);
  ctx.save();
  ctx.globalAlpha = 0.14;
  for (let y = 18; y < height; y += 34) roughLine(ctx, random, 0, y, width, y, palette.pencil, 0.45, 0.32);
  ctx.restore();

  const eased = 0.45 + Math.min(1, Math.max(0, progress)) * 0.55;
  ctx.save(); ctx.globalAlpha = eased;
  if (scene.illustration === "overview") {
    drawCar(ctx, random, cx, cy + 65, 0.78);
    for (let i = 0; i < 6; i += 1) {
      const angle = -Math.PI / 2 + i * Math.PI / 3;
      const x = cx + Math.cos(angle) * Math.min(width, height) * 0.28;
      const y = cy + Math.sin(angle) * Math.min(width, height) * 0.25;
      roughRect(ctx, random, x - 42, y - 25, 84, 50, i === camera ? palette.rust : palette.blue, i === camera ? palette.rust : palette.blue);
      arrow(ctx, random, x, y, cx + Math.cos(angle) * 45, cy + 65 + Math.sin(angle) * 55, i === camera ? palette.rust : palette.pencil);
    }
    roughRect(ctx, random, width - 160, 45, 118, 118, palette.sage, palette.sage);
    for (let i = 0; i < 9; i += 1) dot(ctx, width - 135 + (i % 3) * 31, 70 + Math.floor(i / 3) * 31, 4 + (i % 2) * 3, palette.rust, 0.6);
  } else if (scene.illustration === "sample") {
    for (let i = 0; i < 6; i += 1) roughRect(ctx, random, cx - 260 + (i % 3) * 178, cy - 125 + Math.floor(i / 3) * 126, 142, 94, i === camera ? palette.rust : palette.ink, i === camera ? palette.ochre : palette.blue);
    roughRect(ctx, random, cx - 80, cy + 145, 160, 76, palette.sage, palette.sage);
    arrow(ctx, random, cx, cy + 108, cx, cy + 142);
  } else if (scene.illustration === "rig") {
    const s=Math.min(width,height)/18,baseY=cy+90;drawIsoGrid(ctx,random,cx,baseY,s,7);drawIsoCar(ctx,random,cx,baseY,s);
    for(let index=0;index<6;index+=1){const angle=Math.PI/2-index*Math.PI/3,origin=iso([Math.cos(angle)*1.25,Math.sin(angle)*1.25,.85],cx,baseY,s),target=iso([Math.cos(angle)*6.2,Math.sin(angle)*6.2,.65],cx,baseY,s);drawSpatialCamera(ctx,random,origin,target,index===camera);}
  } else if (scene.illustration === "features") {
    const s=Math.min(width,height)/17,baseY=cy+80;drawIsoGrid(ctx,random,cx,baseY,s,6);
    const image=[[-5,-2,1],[-5,3,1],[-5,3,4.1],[-5,-2,4.1]].map(point=>iso(point as Point3,cx,baseY,s));roughPolygon(ctx,random,image,palette.ink,palette.blue,.1);
    const feature=[[1,-2,1.25],[1,3,1.25],[1,3,3.65],[1,-2,3.65]].map(point=>iso(point as Point3,cx,baseY,s));roughPolygon(ctx,random,feature,palette.ink,palette.ochre,.1);
    for(let row=1;row<8;row+=1){const t=row/8,a:[number,number]=[feature[0][0]+(feature[3][0]-feature[0][0])*t,feature[0][1]+(feature[3][1]-feature[0][1])*t],b:[number,number]=[feature[1][0]+(feature[2][0]-feature[1][0])*t,feature[1][1]+(feature[2][1]-feature[1][1])*t];roughLine(ctx,random,a[0],a[1],b[0],b[1],palette.blue,.45,.22);}
    for(let col=1;col<11;col+=1){const t=col/11,a:[number,number]=[feature[0][0]+(feature[1][0]-feature[0][0])*t,feature[0][1]+(feature[1][1]-feature[0][1])*t],b:[number,number]=[feature[3][0]+(feature[2][0]-feature[3][0])*t,feature[3][1]+(feature[2][1]-feature[3][1])*t];roughLine(ctx,random,a[0],a[1],b[0],b[1],palette.blue,.45,.22);}
    arrow(ctx,random,image[1][0]+22,image[1][1]-8,feature[0][0]-25,feature[0][1]+4,palette.rust);
  } else if (scene.illustration === "ray" || scene.illustration === "lift") {
    const s=Math.min(width,height)/18,baseY=cy+115;drawIsoGrid(ctx,random,cx,baseY,s,7);const origin=iso([-4,-2,1.8],cx,baseY,s),end=iso([6,3,.5],cx,baseY,s);
    drawSpatialCamera(ctx,random,origin,end,true);roughLine(ctx,random,origin[0],origin[1],end[0],end[1],palette.rust,1.6,.8);
    for(let index=0;index<41;index+=1){const t=(index+1)/42,x=origin[0]+(end[0]-origin[0])*t,y=origin[1]+(end[1]-origin[1])*t,peak=Math.exp(-1*((index-depth)/5)**2),radius=scene.illustration==="lift"?1.8+peak*7:index===depth?6.5:1.7;dot(ctx,x,y,radius,index===depth?palette.rust:palette.blue,scene.illustration==="lift"?.22+peak*.68:.46);}
    if(scene.illustration==="lift")for(let index=0;index<9;index+=1){const t=.25+index*.055,x=origin[0]+(end[0]-origin[0])*t,y=origin[1]+(end[1]-origin[1])*t;wash(ctx,random,x,y,18+index*2,10+index,palette.blue,.025);}
  } else if (scene.illustration === "geometry") {
    const s=Math.min(width,height)/18,baseY=cy+95;drawIsoGrid(ctx,random,cx,baseY,s,7);const frames:[Point3,string][]=[[[-5,-2,2.7],palette.ochre],[[-1,-1,2.1],palette.blue],[[2,1,1.4],palette.rust],[[5,3,.25],palette.sage]];
    frames.forEach(([point,color],index)=>{const center=iso(point,cx,baseY,s),x=iso([point[0]+1,point[1],point[2]],cx,baseY,s),y=iso([point[0],point[1]+1,point[2]],cx,baseY,s),z=iso([point[0],point[1],point[2]+1],cx,baseY,s);dot(ctx,center[0],center[1],index===frames.length-1?7:4,color,.82);roughLine(ctx,random,center[0],center[1],x[0],x[1],palette.rust,.85,.6);roughLine(ctx,random,center[0],center[1],y[0],y[1],palette.blue,.85,.6);roughLine(ctx,random,center[0],center[1],z[0],z[1],palette.sage,.85,.6);if(index<frames.length-1){const next=iso(frames[index+1][0],cx,baseY,s);arrow(ctx,random,center[0]+10,center[1],next[0]-10,next[1],palette.pencil);}});drawIsoCar(ctx,random,cx,baseY,s*.72);
  } else if (scene.illustration === "splat") {
    const s=Math.min(width,height)/18,baseY=cy+135;drawIsoGrid(ctx,random,cx,baseY,s,7);for(let index=0;index<70;index+=1){const gx=-5+Math.floor(random()*10),gy=-5+Math.floor(random()*10),z=1.4+random()*5.4,point=iso([gx+random()*.8,gy+random()*.8,z],cx,baseY,s),ground=iso([gx+.5,gy+.5,.08],cx,baseY,s);dot(ctx,point[0],point[1],1.6+random()*1.8,palette.blue,.45);roughLine(ctx,random,point[0],point[1],ground[0],ground[1],palette.blue,.4,.13);if(index%13===0)wash(ctx,random,ground[0],ground[1],18,10,palette.rust,.11);}drawIsoCar(ctx,random,cx,baseY,s*.68);
  } else if (scene.illustration === "bev" || scene.illustration === "truth") {
    const s=Math.min(width,height)/18,baseY=cy+120;drawIsoGrid(ctx,random,cx,baseY,s,7);for(let index=0;index<18;index+=1){const point=iso([-5+random()*10,-5+random()*10,.05],cx,baseY,s);wash(ctx,random,point[0],point[1],13+random()*25,7+random()*13,index%3?palette.rust:palette.ochre,.055+random()*.08);}if(scene.illustration==="truth")for(let index=0;index<360;index+=1){const point=iso([-6+random()*12,-6+random()*12,.12+random()*.35],cx,baseY,s);dot(ctx,point[0],point[1],.6+random()*1.2,index%4?palette.blue:palette.sage,.32);}drawIsoCar(ctx,random,cx,baseY,s*.72);
  } else if (scene.illustration === "learning") {
    const s=Math.min(width,height)/18,baseY=cy+150;drawIsoGrid(ctx,random,cx,baseY,s,6);const layers=[0.15,1.5,2.85,4.2];layers.forEach((z,index)=>{const plane=[[-4,-3,z],[4,-3,z],[4,3,z],[-4,3,z]].map(point=>iso(point as Point3,cx,baseY,s));roughPolygon(ctx,random,plane,index===layers.length-1?palette.rust:palette.blue,index===layers.length-1?palette.rust:palette.blue,.04+index*.018);if(index<layers.length-1){const a=iso([4,0,z+.2],cx,baseY,s),b=iso([4,0,layers[index+1]-.2],cx,baseY,s);arrow(ctx,random,a[0],a[1],b[0],b[1],palette.rust);}});const backwardA=iso([-4,0,4],cx,baseY,s),backwardB=iso([-4,0,.35],cx,baseY,s);arrow(ctx,random,backwardA[0],backwardA[1],backwardB[0],backwardB[1],palette.blue);
  } else {
    const s=Math.min(width,height)/18,baseY=cy+145;drawIsoGrid(ctx,random,cx,baseY,s,7);for(let pathIndex=0;pathIndex<7;pathIndex+=1){const path=Array.from({length:20},(_,point)=>iso([(pathIndex-3)*.025*point**1.65,point*.42-1,.2],cx,baseY,s));path.slice(1).forEach((point,index)=>roughLine(ctx,random,path[index][0],path[index][1],point[0],point[1],pathIndex===3?palette.rust:palette.blue,pathIndex===3?2.5:.9,pathIndex===3?.9:.28));}drawIsoCar(ctx,random,cx,baseY,s*.72);
  }
  ctx.restore();
}

export function IllustrationStage(props: IllustrationStageProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const render = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(1.6, window.devicePixelRatio || 1);
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawScene(ctx, rect.width, rect.height, props.scene, props.progress, props.selectedCamera, props.depthIndex);
    };
    const observer = new ResizeObserver(render);
    observer.observe(canvas);
    render();
    return () => observer.disconnect();
  }, [props.scene, props.progress, props.selectedCamera, props.depthIndex]);

  return (
    <div className="illustration-stage" aria-hidden="true">
      <canvas ref={canvasRef} />
      <div className="sketch-note note-a">calibrated evidence</div>
      <div className="sketch-note note-b">ego +x ↑ · +y ←</div>
      <div className="stage-stamp">{props.scene.act}</div>
    </div>
  );
}
