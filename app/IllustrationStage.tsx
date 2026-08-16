"use client";

import { useEffect, useRef, useState } from "react";
import type { NarrativeScene } from "./lss-content";

type IllustrationStageProps = {
  scene: NarrativeScene;
  progress: number;
  selectedCamera: number;
  depthIndex: number;
  geometryStep: number;
  poolingMode: "sum" | "mean" | "max" | "bilinear";
  poolOffset: number;
  cameraPoses?: Array<{cam2ego:number[][];cam2img:number[][];post_rot:number[][];post_trans:number[];name?:string}>;
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

function indexSeed(row:number,column:number){return (row*11+column*7+row*column)%9;}

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

function drawCameraAxes(ctx:CanvasRenderingContext2D,random:()=>number,origin:Point2,scale=58) {
  arrow(ctx,random,origin[0],origin[1],origin[0]+scale,origin[1]+12,palette.rust);inkLabel(ctx,"+x right",origin[0]+scale+5,origin[1]+15,palette.rust);
  arrow(ctx,random,origin[0],origin[1],origin[0]-8,origin[1]+scale,palette.blue);inkLabel(ctx,"+y down",origin[0]-3,origin[1]+scale+14,palette.blue,"center");
}

function drawProjectionSketch(ctx:CanvasRenderingContext2D,random:()=>number,width:number,height:number,depth:number,step:number,showMetric=false) {
  const origin:Point2=[width*.43,height*.56],planeCenter:Point2=[width*.62,height*.42],objectCenter:Point2=[width*.82,height*.27];
  const raw=[[-82,-59],[82,-43],[82,64],[-82,48]].map(([x,y])=>[planeCenter[0]+x,planeCenter[1]+y] as Point2);
  const network=raw.map(([x,y])=>[planeCenter[0]+(x-planeCenter[0])*.78+18,planeCenter[1]+(y-planeCenter[1])*.78+8] as Point2);
  const selected:Point2=[planeCenter[0]+27,planeCenter[1]-4];
  const selectedNetwork:Point2=[planeCenter[0]+(selected[0]-planeCenter[0])*.78+18,planeCenter[1]+(selected[1]-planeCenter[1])*.78+8];
  roughRect(ctx,random,origin[0]-12,origin[1]-8,24,16,palette.ink,palette.ochre);dot(ctx,origin[0],origin[1],4,palette.ink,.9);inkLabel(ctx,"optical center O = [0,0,0]cam",origin[0]-18,origin[1]+28,palette.ink,"center");drawCameraAxes(ctx,random,origin);
  if(step<=1){roughPolygon(ctx,random,network,palette.blue,palette.blue,.055);dot(ctx,selectedNetwork[0],selectedNetwork[1],6,palette.blue,.9);inkLabel(ctx,"network anchor p′=[u′,v′,1]",selectedNetwork[0]+10,selectedNetwork[1]+19,palette.blue);}
  if(step===1){arrow(ctx,random,selectedNetwork[0]+8,selectedNetwork[1]-5,selected[0]+7,selected[1]-8,palette.rust);inkLabel(ctx,"A⁻¹(p′−a)",(selectedNetwork[0]+selected[0])*.5+9,(selectedNetwork[1]+selected[1])*.5-14,palette.rust,"center");}
  if(step>=1){roughPolygon(ctx,random,raw,palette.ochre,palette.ochre,.06);dot(ctx,selected[0],selected[1],6,palette.rust,.9);inkLabel(ctx,"raw pixel p=[u,v,1]",selected[0]+10,selected[1]-9,palette.rust);raw.forEach(corner=>roughLine(ctx,random,origin[0],origin[1],corner[0],corner[1],palette.pencil,.7,.32));arrow(ctx,random,origin[0]+8,origin[1]-3,planeCenter[0]-8,planeCenter[1]+3,palette.sage);inkLabel(ctx,"+z optical axis",(origin[0]+planeCenter[0])*.5,(origin[1]+planeCenter[1])*.5-12,palette.sage,"center");}
  if(step>=2){roughLine(ctx,random,origin[0],origin[1],objectCenter[0]+10,objectCenter[1]-2,palette.blue,1.7,.78);inkLabel(ctx,"r = K⁻¹p · direction only",(selected[0]+objectCenter[0])*.5,(selected[1]+objectCenter[1])*.5-14,palette.blue,"center");const boxW=128,boxH=94,dx=36,dy=-28;const front:[[number,number],[number,number],[number,number],[number,number]]=[[objectCenter[0]-boxW/2,objectCenter[1]-boxH/2],[objectCenter[0]+boxW/2,objectCenter[1]-boxH/2],[objectCenter[0]+boxW/2,objectCenter[1]+boxH/2],[objectCenter[0]-boxW/2,objectCenter[1]+boxH/2]];const back=front.map(([x,y])=>[x+dx,y+dy] as Point2);roughPolygon(ctx,random,front,palette.ink,undefined);roughPolygon(ctx,random,back,palette.ink,undefined);front.forEach((point,index)=>roughLine(ctx,random,point[0],point[1],back[index][0],back[index][1],palette.ink,1.2,.7));}
  if(showMetric){const t=(depth+1)/42,metric:Point2=[origin[0]+(objectCenter[0]-origin[0])*(.32+t*.52),origin[1]+(objectCenter[1]-origin[1])*(.32+t*.52)];for(let index=0;index<41;index+=1){const q=(index+1)/42,x=origin[0]+(objectCenter[0]-origin[0])*(.3+q*.57),y=origin[1]+(objectCenter[1]-origin[1])*(.3+q*.57);dot(ctx,x,y,index===depth?6:1.6,index===depth?palette.rust:palette.blue,index===depth?.95:.28);}dot(ctx,metric[0],metric[1],7,palette.rust,.92);inkLabel(ctx,`p_cam = d r · d=${depth+4}m`,metric[0]+12,metric[1]-10,palette.rust);}
}

function drawScene(ctx: CanvasRenderingContext2D, width: number, height: number, scene: NarrativeScene, progress: number, camera: number, depth: number, hits:HitRegion[], yaw:number, zoom:number, geometryStep:number, poolingMode:"sum"|"mean"|"max"|"bilinear",poolOffset:number,cameraPoses?:Array<{cam2ego:number[][];cam2img:number[][];post_rot:number[][];post_trans:number[];name?:string}>) {
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
  } else if (scene.illustration === "depth") {
    const s=Math.min(width,height)/18,baseY=cy+125;drawIsoGrid(ctx,random,cx,baseY,s,7);const origin:Point3=[-4,-2,2],end:Point3=[5.8,3,.4],o=iso(origin,cx,baseY,s),e=iso(end,cx,baseY,s);drawSpatialCamera(ctx,random,o,e,true);
    const sheets:Point2[][]=[];
    for(let index=0;index<9;index+=1){const t=.18+index*.075,z=.15+index*.03,center:[number,number,number]=[origin[0]+(end[0]-origin[0])*t,origin[1]+(end[1]-origin[1])*t,origin[2]+(end[2]-origin[2])*t],plane=[[-.62,-.42,0],[.62,-.42,0],[.62,.42,0],[-.62,.42,0]].map(([dx,dy,dz])=>iso([center[0]+dx,center[1]+dy,center[2]+dz+z],cx,baseY,s));sheets.push(plane);roughPolygon(ctx,random,plane,index===5?palette.rust:palette.blue,index===5?palette.rust:palette.blue,index===5?.12:.025);}
    roughLine(ctx,random,o[0],o[1],e[0],e[1],palette.rust,1.4,.7);
    for(let index=0;index<41;index+=1){const t=(index+1)/42,x=o[0]+(e[0]-o[0])*t,y=o[1]+(e[1]-o[1])*t,weight=Math.exp(-1*((index-depth)/6)**2);dot(ctx,x,y,1.5+weight*5,index===depth?palette.rust:palette.blue,.28+weight*.45);hits.push({kind:"depth",index,x,y,radius:9});}
    inkLabel(ctx,"41 depth sheets",sheets[8][2][0]+8,sheets[8][2][1],palette.blue);inkLabel(ctx,"Σ α(d) = 1",cx+150,cy-125,palette.rust);
  } else if (scene.illustration === "context") {
    const s=Math.min(width,height)/18,baseY=cy+130;drawIsoGrid(ctx,random,cx,baseY,s,7);const anchor=iso([-2,-1,2.4],cx,baseY,s),split=iso([1,1,3.4],cx,baseY,s);dot(ctx,anchor[0],anchor[1],7,palette.ochre,.8);arrow(ctx,random,anchor[0]+8,anchor[1]-5,split[0]-8,split[1]+3,palette.rust);
    const depthPlane=[[-.5,-2.3,3],[.5,-2.3,3],[.5,2.3,3],[-.5,2.3,3]].map(point=>iso(point as Point3,cx+150,baseY,s));roughPolygon(ctx,random,depthPlane,palette.rust,palette.rust,.07);
    for(let index=0;index<41;index+=1){const t=index/40,x=depthPlane[0][0]+(depthPlane[1][0]-depthPlane[0][0])*t,y=depthPlane[0][1]+(depthPlane[1][1]-depthPlane[0][1])*t;dot(ctx,x,y-8-Math.exp(-1*((index-depth)/6)**2)*38,1.7,index===depth?palette.rust:palette.ochre,.65);}
    const contextPlane=[[2,-2.8,1],[2,2.8,1],[2,2.8,4.5],[2,-2.8,4.5]].map(point=>iso(point as Point3,cx-20,baseY,s));roughPolygon(ctx,random,contextPlane,palette.blue,palette.blue,.07);
    for(let row=0;row<8;row+=1)for(let col=0;col<8;col+=1){const u=(col+.5)/8,v=(row+.5)/8,x=contextPlane[0][0]+(contextPlane[1][0]-contextPlane[0][0])*u+(contextPlane[3][0]-contextPlane[0][0])*v,y=contextPlane[0][1]+(contextPlane[1][1]-contextPlane[0][1])*u+(contextPlane[3][1]-contextPlane[0][1])*v;dot(ctx,x,y,1.2+(indexSeed(row,col)%3),palette.blue,.28+.05*(indexSeed(row,col)%4));}
    inkLabel(ctx,"WHERE · 41",depthPlane[2][0]+10,depthPlane[2][1],palette.rust);inkLabel(ctx,"WHAT · 64",contextPlane[3][0]-8,contextPlane[3][1]-8,palette.blue,"right");
  } else if (scene.illustration === "ray" || scene.illustration === "lift") {
    const s=Math.min(width,height)/18,baseY=cy+115;drawIsoGrid(ctx,random,cx,baseY,s,7);const origin=iso([-4,-2,1.8],cx,baseY,s),end=iso([6,3,.5],cx,baseY,s);
    drawSpatialCamera(ctx,random,origin,end,true);roughLine(ctx,random,origin[0],origin[1],end[0],end[1],palette.rust,1.6,.8);
    for(let index=0;index<41;index+=1){const t=(index+1)/42,x=origin[0]+(end[0]-origin[0])*t,y=origin[1]+(end[1]-origin[1])*t,peak=Math.exp(-1*((index-depth)/5)**2),radius=scene.illustration==="lift"?1.8+peak*7:index===depth?6.5:1.7;dot(ctx,x,y,radius,index===depth?palette.rust:palette.blue,scene.illustration==="lift"?.22+peak*.68:.46);hits.push({kind:"depth",index,x,y,radius:9});}
    if(scene.illustration==="lift")for(let index=0;index<9;index+=1){const t=.25+index*.055,x=origin[0]+(end[0]-origin[0])*t,y=origin[1]+(end[1]-origin[1])*t;wash(ctx,random,x,y,18+index*2,10+index,palette.blue,.025);}
  } else if (scene.illustration === "image-ray") {
    drawProjectionSketch(ctx,random,width,height,depth,geometryStep,false);
  } else if (scene.illustration === "camera-point") {
    drawProjectionSketch(ctx,random,width,height,depth,Math.max(2,geometryStep),true);
  } else if (scene.illustration === "ego-transform") {
    const s=Math.min(width,height)/18,baseY=cy+135;drawIsoGrid(ctx,random,cx+25,baseY,s,7);drawIsoCar(ctx,random,cx+25,baseY,s*.75);
    (cameraPoses??[]).forEach((pose,index)=>{
      const m=pose.cam2ego,origin:Point3=[m[0][3],m[1][3],m[2][3]],axis:[number,number]=[m[0][2],m[1][2]],angle=Math.atan2(axis[1],axis[0]);
      const o=drawFrustum3D(ctx,random,origin,angle,cx+25,baseY,s,index===camera);hits.push({kind:"camera",index,x:o[0],y:o[1],radius:24});
      if(index!==camera)return;
      const xEnd=iso([origin[0]+m[0][0]*1.5,origin[1]+m[1][0]*1.5,origin[2]+m[2][0]*1.5],cx+25,baseY,s),yEnd=iso([origin[0]+m[0][1]*1.5,origin[1]+m[1][1]*1.5,origin[2]+m[2][1]*1.5],cx+25,baseY,s),zEnd=iso([origin[0]+m[0][2]*2.2,origin[1]+m[1][2]*2.2,origin[2]+m[2][2]*2.2],cx+25,baseY,s);
      arrow(ctx,random,o[0],o[1],xEnd[0],xEnd[1],palette.rust);arrow(ctx,random,o[0],o[1],yEnd[0],yEnd[1],palette.blue);arrow(ctx,random,o[0],o[1],zEnd[0],zEnd[1],palette.sage);inkLabel(ctx,"camera basis R",o[0]+10,o[1]-18,palette.blue);
      const uPrime=351*11/21,vPrime=127*4/7,u=(uPrime-pose.post_trans[0])/pose.post_rot[0][0],v=(vPrime-pose.post_trans[1])/pose.post_rot[1][1],d=depth+4,k=pose.cam2img;
      const pCam:Point3=[(u-k[0][2])/k[0][0]*d,(v-k[1][2])/k[1][1]*d,d];
      const delta:Point3=[m[0][0]*pCam[0]+m[0][1]*pCam[1]+m[0][2]*pCam[2],m[1][0]*pCam[0]+m[1][1]*pCam[1]+m[1][2]*pCam[2],m[2][0]*pCam[0]+m[2][1]*pCam[1]+m[2][2]*pCam[2]],pEgo:Point3=[origin[0]+delta[0],origin[1]+delta[1],origin[2]+delta[2]];
      const compression=Math.min(1,6.5/Math.hypot(delta[0],delta[1],delta[2])),displayPoint:Point3=[origin[0]+delta[0]*compression,origin[1]+delta[1]*compression,origin[2]+delta[2]*compression],display=iso(displayPoint,cx+25,baseY,s);
      roughLine(ctx,random,o[0],o[1],display[0],display[1],palette.rust,2,.82);dot(ctx,display[0],display[1],7,palette.rust,.92);
      inkLabel(ctx,"same physical point",display[0]+9,display[1]-8,palette.rust);
      const noteX=width*.63,noteY=height*.19,noteW=Math.min(300,width*.36);roughRect(ctx,random,noteX,noteY,noteW,88,palette.pencil,palette.ochre);
      inkLabel(ctx,`${pose.name?.replace("CAM_","")??"CAMERA"} · selected d=${d}m`,noteX+12,noteY+18,palette.ink);
      inkLabel(ctx,`p_cam = [${pCam.map(value=>value.toFixed(2)).join(", ")}] m`,noteX+12,noteY+39,palette.blue);
      inkLabel(ctx,`R p_cam + t = [${pEgo.map(value=>value.toFixed(2)).join(", ")}] m`,noteX+12,noteY+60,palette.rust);
      if(compression<1)inkLabel(ctx,"ray compressed on page · numbers exact",noteX+12,noteY+79,palette.pencil);
    });
    const ego=iso([0,0,.6],cx+25,baseY,s),egoX=iso([2.2,0,.6],cx+25,baseY,s),egoY=iso([0,2.2,.6],cx+25,baseY,s);arrow(ctx,random,ego[0],ego[1],egoX[0],egoX[1],palette.rust);arrow(ctx,random,ego[0],ego[1],egoY[0],egoY[1],palette.blue);inkLabel(ctx,"ego frame",ego[0]-8,ego[1]+20,palette.ink,"center");
  } else if (scene.illustration === "splat") {
    const cell=Math.min(57,height*.085),gridX=width*.44,gridY=height*.19,values=[.25,.55,.9];for(let row=0;row<6;row+=1)for(let col=0;col<6;col+=1)roughRect(ctx,random,gridX+col*cell,gridY+row*cell,cell,cell,palette.pencil);
    const fixed=[[2.24,2.3],[2.62,2.68]],moving=[2.1+poolOffset*1.65,2.47] as [number,number],points=[...fixed,moving] as [number,number][];
    const hardX=Math.floor(moving[0]),hardY=Math.floor(moving[1]),result=poolingMode==="sum"?values.reduce((a,b)=>a+b,0):poolingMode==="mean"?values.reduce((a,b)=>a+b,0)/values.length:Math.max(...values);
    if(poolingMode!=="bilinear"){wash(ctx,random,gridX+(hardX+.5)*cell,gridY+(hardY+.5)*cell,cell*.45,cell*.43,poolingMode==="max"?palette.rust:palette.ochre,.16+.13*result);inkLabel(ctx,`${poolingMode} = ${result.toFixed(2)}`,gridX+(hardX+.5)*cell,gridY+(hardY+.5)*cell+4,palette.rust,"center");}else{const fx=moving[0]-hardX,fy=moving[1]-hardY,weights=[[0,0,(1-fx)*(1-fy)],[1,0,fx*(1-fy)],[0,1,(1-fx)*fy],[1,1,fx*fy]];weights.forEach(([dx,dy,w])=>{wash(ctx,random,gridX+(hardX+dx+.5)*cell,gridY+(hardY+dy+.5)*cell,cell*.44,cell*.42,palette.ochre,.08+.25*w);inkLabel(ctx,w.toFixed(2),gridX+(hardX+dx+.5)*cell,gridY+(hardY+dy+.5)*cell+4,palette.rust,"center");});}
    points.forEach(([px,py],index)=>{const x=gridX+px*cell,y=gridY+py*cell;dot(ctx,x,y,7,index===2?palette.rust:palette.blue,.9);inkLabel(ctx,`f${index+1}=${values[index]}`,x+9,y-7,index===2?palette.rust:palette.blue);if(poolingMode!=="bilinear"){const tx=gridX+(Math.floor(px)+.5)*cell,ty=gridY+(Math.floor(py)+.5)*cell;arrow(ctx,random,x,y+8,tx,ty-10,palette.pencil);}});
    inkLabel(ctx,"continuous ego XY",gridX,gridY-16,palette.blue);inkLabel(ctx,"floor → integer cell",gridX+6*cell,gridY+6*cell+20,palette.rust,"right");
  } else if (scene.illustration === "truth") {
    const s=Math.min(width,height)/18,baseY=cy+120;drawIsoGrid(ctx,random,cx,baseY,s,7);for(let index=0;index<18;index+=1){const point=iso([-5+random()*10,-5+random()*10,.05],cx,baseY,s);wash(ctx,random,point[0],point[1],13+random()*25,7+random()*13,index%3?palette.rust:palette.ochre,.055+random()*.08);}if(scene.illustration==="truth")for(let index=0;index<360;index+=1){const point=iso([-6+random()*12,-6+random()*12,.12+random()*.35],cx,baseY,s);dot(ctx,point[0],point[1],.6+random()*1.2,index%4?palette.blue:palette.sage,.32);}drawIsoCar(ctx,random,cx,baseY,s*.72);
  } else if (scene.illustration === "learning") {
    const split=width*.67,top=height*.19,rowGap=64,left=width*.42;roughLine(ctx,random,split,top-28,split,height*.78,palette.pencil,.8,.35);inkLabel(ctx,"LSS · indirect depth",left,top-14,palette.rust);inkLabel(ctx,"BEVDepth · explicit depth",split+24,top-14,palette.blue);
    const lss=["BEV target","task BCE","BEV Encoder","Splat","α × context","Depth logits"];lss.forEach((label,index)=>{const y=top+index*rowGap*.82,boxX=left;roughRect(ctx,random,boxX,y,125,28,index===5?palette.blue:palette.rust,index===5?palette.blue:palette.rust);inkLabel(ctx,label,boxX+62,y+18,index===5?palette.blue:palette.rust,"center");if(index<lss.length-1)arrow(ctx,random,boxX+62,y+31,boxX+62,y+rowGap*.82-4,palette.pencil);});inkLabel(ctx,"gradient travels through the task",left+140,top+135,palette.rust);
    const right=split+22,camY=top+18;roughRect(ctx,random,right,camY,92,62,palette.ink,palette.ochre);for(let index=0;index<18;index+=1)dot(ctx,right+8+random()*76,camY+8+random()*46,1.5,index%3?palette.blue:palette.sage,.65);inkLabel(ctx,"LiDAR → image",right+46,camY+78,palette.blue,"center");
    const ops=["min nonzero / block","depth bin","one-hot Dgt","masked BCE","Depth logits"];ops.forEach((label,index)=>{const x=right+130+(index%2)*128,y=top+(index<2?0:index<4?94:188);roughRect(ctx,random,x,y,112,31,index===4?palette.rust:palette.blue,index===4?palette.rust:palette.blue);inkLabel(ctx,label,x+56,y+20,index===4?palette.rust:palette.blue,"center");if(index<ops.length-1){const nx=right+130+((index+1)%2)*128,ny=top+((index+1)<2?0:(index+1)<4?94:188);arrow(ctx,random,x+112,y+15,nx-5,ny+15,palette.pencil);}});inkLabel(ctx,"training only · camera-only inference",right+130,top+260,palette.sage);
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
      drawScene(ctx, rect.width, rect.height, props.scene, props.progress, props.selectedCamera, props.depthIndex,hitsRef.current,view.yaw,view.zoom,props.geometryStep,props.poolingMode,props.poolOffset,props.cameraPoses);
    };
    const observer = new ResizeObserver(render);
    observer.observe(canvas);
    render();
    return () => observer.disconnect();
  }, [props.scene, props.progress, props.selectedCamera, props.depthIndex,props.geometryStep,props.poolingMode,props.poolOffset,props.cameraPoses,view]);

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
