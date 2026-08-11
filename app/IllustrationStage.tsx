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
    drawCar(ctx, random, cx, cy, 1.05);
    for (let i = 0; i < 6; i += 1) {
      const angle = -Math.PI / 2 + i * Math.PI / 3;
      const ox = cx + Math.cos(angle) * 48, oy = cy + Math.sin(angle) * 70;
      const fx = cx + Math.cos(angle) * 205, fy = cy + Math.sin(angle) * 205;
      roughLine(ctx, random, ox, oy, fx, fy, i === camera ? palette.rust : palette.blue, i === camera ? 2.4 : 1.1);
      roughLine(ctx, random, ox, oy, fx + Math.sin(angle) * 62, fy - Math.cos(angle) * 62, i === camera ? palette.rust : palette.blue, 0.9, 0.5);
      roughLine(ctx, random, ox, oy, fx - Math.sin(angle) * 62, fy + Math.cos(angle) * 62, i === camera ? palette.rust : palette.blue, 0.9, 0.5);
    }
  } else if (scene.illustration === "features") {
    roughRect(ctx, random, cx - 290, cy - 145, 290, 164, palette.ink, palette.blue);
    arrow(ctx, random, cx + 15, cy - 64, cx + 88, cy - 64);
    const gx = cx + 115, gy = cy - 145, cellW = 14, cellH = 20;
    for (let row = 0; row < 8; row += 1) for (let col = 0; col < 22; col += 1) {
      ctx.save(); ctx.fillStyle = (row + col) % 7 === 0 ? palette.rust : palette.blue; ctx.globalAlpha = 0.08 + ((row * 17 + col * 13) % 11) / 28; ctx.fillRect(gx + col * cellW, gy + row * cellH, cellW - 2, cellH - 2); ctx.restore();
    }
    roughRect(ctx, random, gx, gy, 22 * cellW, 8 * cellH, palette.ink);
  } else if (scene.illustration === "ray" || scene.illustration === "lift") {
    const originX = cx - 260, originY = cy + 130;
    roughRect(ctx, random, originX - 25, originY - 18, 50, 36, palette.ink, palette.ink);
    roughLine(ctx, random, originX, originY, cx + 310, cy - 130, palette.rust, 2);
    for (let i = 0; i < 41; i += 1) {
      const t = (i + 1) / 42;
      const x = originX + (cx + 310 - originX) * t, y = originY + (cy - 130 - originY) * t;
      const peak = Math.exp(-1 * ((i - depth) / 5) ** 2);
      dot(ctx, x, y, scene.illustration === "lift" ? 2.2 + peak * 8 : i === depth ? 7 : 2, i === depth ? palette.rust : palette.blue, scene.illustration === "lift" ? 0.25 + peak * 0.7 : 0.5);
    }
    if (scene.illustration === "lift") wash(ctx, random, cx + 80, cy + 5, 170, 65, palette.blue, 0.08);
  } else if (scene.illustration === "geometry") {
    const labels = ["u′", "A⁻¹", "K⁻¹", "R", "+t", "ego"];
    labels.forEach((_, i) => {
      const x = 55 + i * ((width - 110) / (labels.length - 1));
      dot(ctx, x, cy, i === labels.length - 1 ? 11 : 7, i < 2 ? palette.ochre : i < 4 ? palette.blue : palette.rust, 0.7);
      if (i < labels.length - 1) arrow(ctx, random, x + 13, cy, x + (width - 110) / (labels.length - 1) - 13, cy, palette.pencil);
    });
    drawCar(ctx, random, width - 85, cy + 130, 0.46);
  } else if (scene.illustration === "splat") {
    const grid = Math.min(width, height) * 0.68, left = cx - grid / 2, top = cy - grid / 2;
    for (let i = 0; i <= 12; i += 1) {
      roughLine(ctx, random, left + i * grid / 12, top, left + i * grid / 12, top + grid, palette.pencil, 0.5, 0.35);
      roughLine(ctx, random, left, top + i * grid / 12, left + grid, top + i * grid / 12, palette.pencil, 0.5, 0.35);
    }
    for (let i = 0; i < 76; i += 1) {
      const x = left + random() * grid, y = top - 60 - random() * 65;
      const targetX = left + Math.floor(random() * 12) * grid / 12 + grid / 24;
      const targetY = top + Math.floor(random() * 12) * grid / 12 + grid / 24;
      dot(ctx, x, y, 2.2, palette.blue, 0.55); roughLine(ctx, random, x, y, targetX, targetY, palette.blue, 0.45, 0.18);
    }
    wash(ctx, random, cx + 45, cy - 5, 68, 55, palette.rust, 0.24);
  } else if (scene.illustration === "bev" || scene.illustration === "truth") {
    const size = Math.min(width, height) * 0.72, left = cx - size / 2, top = cy - size / 2;
    roughRect(ctx, random, left, top, size, size, palette.ink);
    for (let i = 0; i < 15; i += 1) wash(ctx, random, left + random() * size, top + random() * size, 14 + random() * 38, 9 + random() * 26, i % 3 ? palette.rust : palette.ochre, 0.08 + random() * 0.1);
    drawCar(ctx, random, cx, cy + size * 0.22, 0.38);
    if (scene.illustration === "truth") for (let i = 0; i < 420; i += 1) dot(ctx, left + random() * size, top + random() * size, 0.7 + random() * 1.4, i % 4 ? palette.blue : palette.sage, 0.38);
  } else if (scene.illustration === "learning") {
    const nodes = [[cx-260,cy-120],[cx-80,cy-120],[cx+100,cy-120],[cx+280,cy-120],[cx+280,cy+100],[cx+100,cy+100],[cx-80,cy+100],[cx-260,cy+100]];
    nodes.forEach(([x,y], i) => { roughRect(ctx, random, x-48,y-27,96,54,i<4?palette.blue:palette.rust,i<4?palette.blue:palette.rust); if(i<nodes.length-1) arrow(ctx,random,x+48,y,nodes[i+1][0]-48,nodes[i+1][1],i<3?palette.blue:palette.rust); });
  } else {
    const paths = Array.from({ length: 7 }, (_, index) => Array.from({ length: 18 }, (_, point) => {
      const y = cy + 180 - point * 20;
      return [cx + (index - 3) * 0.017 * (point * 20) ** 1.55, y];
    }));
    paths.forEach((path, index) => path.slice(1).forEach((point, pointIndex) => roughLine(ctx, random, path[pointIndex][0], path[pointIndex][1], point[0], point[1], index === 3 ? palette.rust : palette.blue, index === 3 ? 2.8 : 1, index === 3 ? 0.95 : 0.35)));
    drawCar(ctx, random, cx, cy + 205, 0.5);
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
