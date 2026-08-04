import { readFile, stat } from "node:fs/promises";
import { join } from "node:path";

const root=new URL("../",import.meta.url).pathname,out=join(root,"out"),html=await readFile(join(out,"index.html"),"utf8");
if(!html.includes('/lss-explained/_next/'))throw new Error("Static HTML is missing the GitHub Pages base path");
const required=[
  "index.html","og.png","data/rig.json","data/model-features.json","data/model-artifacts.json",
  "data/model/bev-all-cameras.png","data/model/vehicle-gt.png","articles/lift-splat-shoot-explained.zh-CN.md",
];
for(const relative of required){const info=await stat(join(out,relative));if(!info.isFile()||info.size===0)throw new Error(`Missing exported Pages asset: ${relative}`)}
console.log("verified static export under /lss-explained with article, evidence, and model rasters");
