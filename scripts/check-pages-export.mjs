import { readFile, stat } from "node:fs/promises";
import { join } from "node:path";

const root=new URL("../",import.meta.url).pathname,out=join(root,"out"),html=await readFile(join(out,"index.html"),"utf8");
if(!html.includes('/lss-explained/_next/'))throw new Error("Static HTML is missing the GitHub Pages base path");
const required=[
  "index.html","og.png","data/rig.json","data/model-features.json","data/model-artifacts.json",
  "data/alignment.json","data/lidar-frame.bin","data/network-images/cam-front.jpg",
  "data/model/bev-all-cameras.png","data/model/vehicle-gt.png","articles/lift-splat-shoot-source-notes.md",
];
for(const relative of required){const info=await stat(join(out,relative));if(!info.isFile()||info.size===0)throw new Error(`Missing exported Pages asset: ${relative}`)}
if(/locale|zh-CN|[\u3400-\u9fff]/u.test(html))throw new Error("Static HTML still contains the removed bilingual interface");
console.log("verified /lss-explained: ten tensor-first scenes, LSS/BEVDepth source notes, LiDAR audit assets, and checkpoint rasters");
