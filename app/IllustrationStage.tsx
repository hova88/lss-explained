"use client";

import { useEffect, useRef, useState } from "react";
import type { NarrativeScene } from "./lss-content";

type IllustrationStageProps = {
  scene: NarrativeScene;
  progress: number;
  selectedCamera: number;
  depthIndex: number;
  onCameraSelect?: (index:number)=>void;
  onDepthSelect?: (index:number)=>void;
};

type HitRegion = { kind:"camera"|"depth"; index:number; x:number; y:number; radius:number };

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

function inkLabel(ctx:CanvasRenderingContext2D,text:string,x:number,y:number,color=palette.ink,align:CanvasTextAlign="left") {
  ctx.save();ctx.fillStyle=color;ctx.globalAlpha=.72;ctx.font="600 12px Caveat, cursive";ctx.textAlign=align;ctx.fillText(text,x,y);ctx.restore();
}

function arrow(ctx: CanvasRenderingContext2D, random: () => number, x1: number, y1: number, x2: number, y2: number, color = palette.rust) {
  roughLine(ctx, random, x1, y1, x2, y2, color, 1.6, 0.88);
  const angle = Math.atan2(y2 - y1, x2 - x1);
  roughLine(ctx, random, x2, y2, x2 - Math.cos(angle - 0.5) * 11, y2 - Math.sin(angle - 0.5) * 11, color, 1.6, 0.88);
  roughLine(ctx, random, x2, y2, x2 - Math.cos(angle + 0.5) * 11, y2 - Math.sin(angle + 0.5) * 11, color, 1.6, 0.88);
}

type Point2 = [number, number];
type Point3 = [number, number, number];
let viewYaw=0;
let viewZoom=1;

function iso(point: Point3, cx: number, cy: number, scale: number): Point2 {
  const [x,y,z]=point;
  const cosine=Math.cos(viewYaw),sine=Math.sin(viewYaw),rx=x*cosine-y*sine,ry=x*sine+y*cosine,s=scale*viewZoom;
  return [cx+(rx-ry)*.866*s,cy+(rx+ry)*.5*s-z*s];
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
  const footprint:Point3[]=[[-.68,-1.45,0],[.68,-1.45,0],[.82,-.8,0],[.78,1.2,0],[0,1.58,0],[-.78,1.2,0],[-.82,-.8,0]];
  const top=footprint.map(([x,y])=>iso([x,y,.38],cx,cy,scale)),bottom=footprint.map((point)=>iso(point,cx,cy,scale));
  roughPolygon(ctx,random,bottom,palette.ink,"#eee6d7",.8);roughPolygon(ctx,random,top,palette.ink,"#eee6d7",.94);
  for(let index=0;index<bottom.length;index+=1)roughLine(ctx,random,bottom[index][0],bottom[index][1],top[index][0],top[index][1],palette.ink,1,.42);
  const cabin=[[-.48,-.36,.42],[.48,-.36,.42],[.48,.7,.42],[0,.94,.42],[-.48,.7,.42]].map((point)=>iso(point as Point3,cx,cy,scale));
  roughPolygon(ctx,random,cabin,palette.blue,palette.blue,.1);
  const origin=iso([0,.15,.48],cx,cy,scale);dot(ctx,origin[0],origin[1],3.4,palette.ochre,.82);
  const forward=iso([0,2.45,.48],cx,cy,scale);arrow(ctx,random,origin[0],origin[1],forward[0],forward[1],palette.rust);
}

function drawVoxel(ctx:CanvasRenderingContext2D,random:()=>number,center:Point3,size:Point3,cx:number,cy:number,scale:number,color=palette.rust,alpha=.055) {
  const [x,y,z]=center,[sx,sy,sz]=size;
  const points:Point3[]=[[-sx,-sy,0],[sx,-sy,0],[sx,sy,0],[-sx,sy,0],[-sx,-sy,sz],[sx,-sy,sz],[sx,sy,sz],[-sx,sy,sz]].map(([dx,dy,dz])=>[x+dx,y+dy,z+dz]);
  const screen=points.map(point=>iso(point,cx,cy,scale));
  roughPolygon(ctx,random,[screen[0],screen[1],screen[2],screen[3]],color,color,alpha);
  roughPolygon(ctx,random,[screen[4],screen[5],screen[6],screen[7]],color,color,alpha*1.4);
  for(let index=0;index<4;index+=1)roughLine(ctx,random,screen[index][0],screen[index][1],screen[index+4][0],screen[index+4][1],color,.75,.34);
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

function drawFrustum3D(ctx:CanvasRenderingContext2D,random:()=>number,origin:Point3,angle:number,cx:number,cy:number,scale:number,selected=false) {
  const direction:[number,number]=[Math.cos(angle),Math.sin(angle)],perp:[number,number]=[-direction[1],direction[0]],far=5.7;
  const farCenter:[number,number,number]=[origin[0]+direction[0]*far,origin[1]+direction[1]*far,.8];
  const corners:Point3[]=[
    [farCenter[0]+perp[0]*2.15,farCenter[1]+perp[1]*2.15,.05],
    [farCenter[0]-perp[0]*2.15,farCenter[1]-perp[1]*2.15,.05],
    [farCenter[0]-perp[0]*1.75,farCenter[1]-perp[1]*1.75,3.1],
    [farCenter[0]+perp[0]*1.75,farCenter[1]+perp[1]*1.75,3.1],
  ];
  const o=iso(origin,cx,cy,scale),screenCorners=corners.map(point=>iso(point,cx,cy,scale)),color=selected?palette.rust:palette.blue;
  roughPolygon(ctx,random,screenCorners,color,color,selected?.075:.028);
  screenCorners.forEach(corner=>roughLine(ctx,random,o[0],o[1],corner[0],corner[1],color,selected?1.45:.72,selected?.74:.22));
  const axis=iso(farCenter,cx,cy,scale);roughLine(ctx,random,o[0],o[1],axis[0],axis[1],selected?palette.rust:palette.pencil,selected?1.65:.7,selected?.82:.34);
  roughRect(ctx,random,o[0]-8,o[1]-6,16,12,selected?palette.rust:palette.ink,selected?palette.rust:palette.ochre);
  return o;
}

function drawScene(ctx: CanvasRenderingContext2D, width: number, height: number, scene: NarrativeScene, progress: number, camera: number, depth: number, hits:HitRegion[], yaw:number, zoom:number) {
  viewYaw=yaw;viewZoom=zoom;
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
    const s=Math.min(width,height)/19,baseY=cy+125;drawIsoGrid(ctx,random,cx+45,baseY,s,7);drawIsoCar(ctx,random,cx+45,baseY,s*.78);
    for(let index=0;index<6;index+=1){const angle=Math.PI/2-index*Math.PI/3,origin:Point3=[Math.cos(angle)*1.25,Math.sin(angle)*1.25,.75],o=drawFrustum3D(ctx,random,origin,angle,cx+45,baseY,s,index===camera);hits.push({kind:"camera",index,x:o[0],y:o[1],radius:25});}
    const bev=[[-4,-4,.08],[4,-4,.08],[4,4,.08],[-4,4,.08]].map(point=>iso(point as Point3,cx+45,baseY-185,s*.65));roughPolygon(ctx,random,bev,palette.sage,palette.sage,.1);
    for(let index=0;index<13;index+=1){const p=iso([-3.2+random()*6.4,-3.2+random()*6.4,.12],cx+45,baseY-185,s*.65);wash(ctx,random,p[0],p[1],8+random()*16,5+random()*9,palette.rust,.08);}
    const carTop=iso([0,0,3.4],cx+45,baseY,s),bevBottom=iso([0,0,.08],cx+45,baseY-185,s*.65);arrow(ctx,random,carTop[0],carTop[1],bevBottom[0],bevBottom[1],palette.rust);
    inkLabel(ctx,"six calibrated frustums",cx+215,baseY-18,palette.blue);
    inkLabel(ctx,"shared ego BEV",bev[2][0]+8,bev[2][1]-4,palette.sage);
  } else if (scene.illustration === "sample") {
    const s=Math.min(width,height)/18,baseY=cy+145;drawIsoGrid(ctx,random,cx+30,baseY,s,7);drawIsoCar(ctx,random,cx+30,baseY,s*.65);
    const planes:{x:number;z:number;color:string}[]=[{x:-5,z:3.8,color:palette.blue},{x:-1.7,z:4.8,color:palette.ochre},{x:1.8,z:3.9,color:palette.rust},{x:5,z:4.8,color:palette.sage}];
    planes.forEach((item,index)=>{const plane=[[item.x,-1.5,item.z],[item.x,1.5,item.z],[item.x,1.5,item.z+2.1],[item.x,-1.5,item.z+2.1]].map(point=>iso(point as Point3,cx+30,baseY,s));roughPolygon(ctx,random,plane,item.color,item.color,.1);const ground=iso([item.x,0,.1],cx+30,baseY,s),center=iso([item.x,0,item.z],cx+30,baseY,s);roughLine(ctx,random,center[0],center[1],ground[0],ground[1],item.color,.8,.28);if(index===0)for(let dotIndex=0;dotIndex<6;dotIndex+=1)dot(ctx,plane[0][0]+18+dotIndex*7,plane[0][1]-10-dotIndex*2,1.5,palette.blue,.55);});
    for(let index=0;index<90;index+=1){const p=iso([-6+random()*12,-5+random()*10,.15+random()*1.3],cx+30,baseY,s);dot(ctx,p[0],p[1],.7+random()*.8,palette.sage,.34);}
    inkLabel(ctx,"images",cx-200,cy-115,palette.blue);inkLabel(ctx,"calibration",cx-30,cy-165,palette.ochre);inkLabel(ctx,"GT target",cx+140,cy-115,palette.rust);inkLabel(ctx,"LiDAR · audit only",cx+250,cy+115,palette.sage);
  } else if (scene.illustration === "rig") {
    const s=Math.min(width,height)/18,baseY=cy+90;drawIsoGrid(ctx,random,cx,baseY,s,7);drawIsoCar(ctx,random,cx,baseY,s);
    for(let index=0;index<6;index+=1){const angle=Math.PI/2-index*Math.PI/3,origin:Point3=[Math.cos(angle)*1.25,Math.sin(angle)*1.25,.85],o=drawFrustum3D(ctx,random,origin,angle,cx,baseY,s,index===camera);hits.push({kind:"camera",index,x:o[0],y:o[1],radius:24});}
  } else if (scene.illustration === "features") {
    const s=Math.min(width,height)/17,baseY=cy+80;drawIsoGrid(ctx,random,cx,baseY,s,6);
    const image=[[-5,-2,1],[-5,3,1],[-5,3,4.1],[-5,-2,4.1]].map(point=>iso(point as Point3,cx,baseY,s));roughPolygon(ctx,random,image,palette.ink,palette.blue,.1);
    const feature=[[1,-2,1.25],[1,3,1.25],[1,3,3.65],[1,-2,3.65]].map(point=>iso(point as Point3,cx,baseY,s));roughPolygon(ctx,random,feature,palette.ink,palette.ochre,.1);
    for(let row=1;row<8;row+=1){const t=row/8,a:[number,number]=[feature[0][0]+(feature[3][0]-feature[0][0])*t,feature[0][1]+(feature[3][1]-feature[0][1])*t],b:[number,number]=[feature[1][0]+(feature[2][0]-feature[1][0])*t,feature[1][1]+(feature[2][1]-feature[1][1])*t];roughLine(ctx,random,a[0],a[1],b[0],b[1],palette.blue,.45,.22);}
    for(let col=1;col<11;col+=1){const t=col/11,a:[number,number]=[feature[0][0]+(feature[1][0]-feature[0][0])*t,feature[0][1]+(feature[1][1]-feature[0][1])*t],b:[number,number]=[feature[3][0]+(feature[2][0]-feature[3][0])*t,feature[3][1]+(feature[2][1]-feature[3][1])*t];roughLine(ctx,random,a[0],a[1],b[0],b[1],palette.blue,.45,.22);}
    arrow(ctx,random,image[1][0]+22,image[1][1]-8,feature[0][0]-25,feature[0][1]+4,palette.rust);
    inkLabel(ctx,"352 × 128 image",image[3][0]-4,image[3][1]-12,palette.blue,"center");inkLabel(ctx,"8 × 22 anchors",feature[2][0]+12,feature[2][1],palette.ochre);
  } else if (scene.illustration === "ray" || scene.illustration === "lift") {
    const s=Math.min(width,height)/18,baseY=cy+115;drawIsoGrid(ctx,random,cx,baseY,s,7);const origin=iso([-4,-2,1.8],cx,baseY,s),end=iso([6,3,.5],cx,baseY,s);
    drawSpatialCamera(ctx,random,origin,end,true);roughLine(ctx,random,origin[0],origin[1],end[0],end[1],palette.rust,1.6,.8);
    for(let index=0;index<41;index+=1){const t=(index+1)/42,x=origin[0]+(end[0]-origin[0])*t,y=origin[1]+(end[1]-origin[1])*t,peak=Math.exp(-1*((index-depth)/5)**2),radius=scene.illustration==="lift"?1.8+peak*7:index===depth?6.5:1.7;dot(ctx,x,y,radius,index===depth?palette.rust:palette.blue,scene.illustration==="lift"?.22+peak*.68:.46);hits.push({kind:"depth",index,x,y,radius:9});}
    if(scene.illustration==="lift")for(let index=0;index<9;index+=1){const t=.25+index*.055,x=origin[0]+(end[0]-origin[0])*t,y=origin[1]+(end[1]-origin[1])*t;wash(ctx,random,x,y,18+index*2,10+index,palette.blue,.025);}
  } else if (scene.illustration === "geometry") {
    const s=Math.min(width,height)/18,baseY=cy+95;drawIsoGrid(ctx,random,cx,baseY,s,7);const frames:[Point3,string][]=[[[-5,-2,2.7],palette.ochre],[[-1,-1,2.1],palette.blue],[[2,1,1.4],palette.rust],[[5,3,.25],palette.sage]];
    frames.forEach(([point,color],index)=>{const center=iso(point,cx,baseY,s),x=iso([point[0]+1,point[1],point[2]],cx,baseY,s),y=iso([point[0],point[1]+1,point[2]],cx,baseY,s),z=iso([point[0],point[1],point[2]+1],cx,baseY,s);dot(ctx,center[0],center[1],index===frames.length-1?7:4,color,.82);roughLine(ctx,random,center[0],center[1],x[0],x[1],palette.rust,.85,.6);roughLine(ctx,random,center[0],center[1],y[0],y[1],palette.blue,.85,.6);roughLine(ctx,random,center[0],center[1],z[0],z[1],palette.sage,.85,.6);inkLabel(ctx,["network pixel","raw pixel + d","camera point","ego point"][index],center[0]+8,center[1]-9,color);if(index<frames.length-1){const next=iso(frames[index+1][0],cx,baseY,s);arrow(ctx,random,center[0]+10,center[1],next[0]-10,next[1],palette.pencil);}});drawIsoCar(ctx,random,cx,baseY,s*.72);
  } else if (scene.illustration === "splat") {
    const s=Math.min(width,height)/18,baseY=cy+135;drawIsoGrid(ctx,random,cx,baseY,s,7);for(let index=0;index<70;index+=1){const gx=-5+Math.floor(random()*10),gy=-5+Math.floor(random()*10),z=1.4+random()*5.4,point=iso([gx+random()*.8,gy+random()*.8,z],cx,baseY,s),ground=iso([gx+.5,gy+.5,.08],cx,baseY,s);dot(ctx,point[0],point[1],1.6+random()*1.8,palette.blue,.45);roughLine(ctx,random,point[0],point[1],ground[0],ground[1],palette.blue,index%9===0?.9:.4,index%9===0?.38:.1);if(index%13===0){drawVoxel(ctx,random,[gx+.5,gy+.5,.05],[.48,.48,2.7],cx,baseY,s,palette.rust,.045);wash(ctx,random,ground[0],ground[1],18,10,palette.rust,.11);}}drawIsoCar(ctx,random,cx,baseY,s*.68);inkLabel(ctx,"irregular lifted points",cx+120,cy-145,palette.blue);inkLabel(ctx,"sum inside each pillar",cx+170,cy+105,palette.rust);
  } else if (scene.illustration === "truth") {
    const s=Math.min(width,height)/18,baseY=cy+120;drawIsoGrid(ctx,random,cx,baseY,s,7);for(let index=0;index<18;index+=1){const point=iso([-5+random()*10,-5+random()*10,.05],cx,baseY,s);wash(ctx,random,point[0],point[1],13+random()*25,7+random()*13,index%3?palette.rust:palette.ochre,.055+random()*.08);}if(scene.illustration==="truth")for(let index=0;index<360;index+=1){const point=iso([-6+random()*12,-6+random()*12,.12+random()*.35],cx,baseY,s);dot(ctx,point[0],point[1],.6+random()*1.2,index%4?palette.blue:palette.sage,.32);}drawIsoCar(ctx,random,cx,baseY,s*.72);
  } else if (scene.illustration === "learning") {
    const s=Math.min(width,height)/18,baseY=cy+150;drawIsoGrid(ctx,random,cx,baseY,s,6);const layers=[0.15,1.5,2.85,4.2];layers.forEach((z,index)=>{const plane=[[-4,-3,z],[4,-3,z],[4,3,z],[-4,3,z]].map(point=>iso(point as Point3,cx,baseY,s));roughPolygon(ctx,random,plane,index===layers.length-1?palette.rust:palette.blue,index===layers.length-1?palette.rust:palette.blue,.04+index*.018);inkLabel(ctx,["pooled BEV","BEV context","multiscale fusion","vehicle logits"][index],plane[2][0]+8,plane[2][1],index===layers.length-1?palette.rust:palette.blue);if(index<layers.length-1){const a=iso([4,0,z+.2],cx,baseY,s),b=iso([4,0,layers[index+1]-.2],cx,baseY,s);arrow(ctx,random,a[0],a[1],b[0],b[1],palette.rust);}});const backwardA=iso([-4,0,4],cx,baseY,s),backwardB=iso([-4,0,.35],cx,baseY,s);arrow(ctx,random,backwardA[0],backwardA[1],backwardB[0],backwardB[1],palette.blue);inkLabel(ctx,"BCE gradient",backwardA[0]-8,(backwardA[1]+backwardB[1])*.5,palette.blue,"right");
  } else {
    const s=Math.min(width,height)/18,baseY=cy+145;drawIsoGrid(ctx,random,cx,baseY,s,7);for(let pathIndex=0;pathIndex<7;pathIndex+=1){const path=Array.from({length:20},(_,point)=>iso([(pathIndex-3)*.025*point**1.65,point*.42-1,.2],cx,baseY,s));path.slice(1).forEach((point,index)=>roughLine(ctx,random,path[index][0],path[index][1],point[0],point[1],pathIndex===3?palette.rust:palette.blue,pathIndex===3?2.5:.9,pathIndex===3?.9:.28));}drawIsoCar(ctx,random,cx,baseY,s*.72);
  }
  ctx.restore();
}

export function IllustrationStage(props: IllustrationStageProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hitsRef=useRef<HitRegion[]>([]);
  const dragRef=useRef<{x:number;y:number;yaw:number}|null>(null);
  const [view,setView]=useState({yaw:0,zoom:1});
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
      hitsRef.current=[];
      drawScene(ctx, rect.width, rect.height, props.scene, props.progress, props.selectedCamera, props.depthIndex,hitsRef.current,view.yaw,view.zoom);
    };
    const observer = new ResizeObserver(render);
    observer.observe(canvas);
    render();
    return () => observer.disconnect();
  }, [props.scene, props.progress, props.selectedCamera, props.depthIndex,view]);

  return (
    <div className="illustration-stage">
      <canvas ref={canvasRef}
        onPointerDown={(event)=>{event.currentTarget.setPointerCapture(event.pointerId);dragRef.current={x:event.clientX,y:event.clientY,yaw:view.yaw};}}
        onPointerMove={(event)=>{if(!dragRef.current)return;const delta=event.clientX-dragRef.current.x;if(Math.abs(delta)>3)setView(current=>({...current,yaw:dragRef.current!.yaw+delta*.006}));}}
        onPointerUp={(event)=>{const drag=dragRef.current;dragRef.current=null;if(!drag||Math.hypot(event.clientX-drag.x,event.clientY-drag.y)>8)return;const rect=event.currentTarget.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;const hit=hitsRef.current.reduce<HitRegion|null>((best,item)=>Math.hypot(item.x-x,item.y-y)<=item.radius?item:best,null);if(hit?.kind==="camera")props.onCameraSelect?.(hit.index);if(hit?.kind==="depth")props.onDepthSelect?.(hit.index);}}
        onWheel={(event)=>{event.preventDefault();setView(current=>({...current,zoom:Math.max(.72,Math.min(1.5,current.zoom-event.deltaY*.001))}));}}
        aria-label="Interactive spatial hand-drawn diagram" />
      <div className="sketch-note note-a">calibrated evidence</div>
      <div className="sketch-note note-b">ego +x ↑ · +y ←</div>
      <div className="stage-stamp">{props.scene.act}</div>
    </div>
  );
}
