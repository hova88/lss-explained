import { createHash } from "node:crypto";
import { readFile, stat } from "node:fs/promises";
import { join } from "node:path";

const root = new URL("../", import.meta.url).pathname;
const publicRoot = join(root, "public");
const required = [
  "data/rig.json", "data/model-features.json", "data/model-artifacts.json", "data/manifest.json",
  "data/model/bev-all-cameras.png", "data/model/vehicle-gt.png",
  ...["front-left", "front", "front-right", "back-left", "back", "back-right"].map(name=>`data/images/cam-${name}.jpg`),
];
for (const relative of required) {
  const info = await stat(join(publicRoot, relative));
  if (!info.isFile() || info.size === 0) throw new Error(`Missing or empty asset: ${relative}`);
}
const rig = JSON.parse(await readFile(join(publicRoot, "data/rig.json"), "utf8"));
const model = JSON.parse(await readFile(join(publicRoot, "data/model-artifacts.json"), "utf8"));
if (rig.schema_version !== "1.0.0" || model.schema_version !== "1.0.0") throw new Error("Unexpected evidence schema");
if (rig.sample_token !== "ca9a282c9e77460f8360f564131a8af5") throw new Error("Unexpected nuScenes sample");
if (rig.cameras.length !== 6 || Object.keys(model.variants).length !== 10) throw new Error("Incomplete rig variants");
if (!model.tensor_checks.finite || model.shapes.vehicle_logits.join() !== "1,1,200,200") throw new Error("Invalid checkpoint export");
if (model.source_hashes.checkpoint !== "4543030a339face9facb5651eb8f29add3407f8c7108f9eb21b0f8bceec921a0") throw new Error("Checkpoint hash drift");
const digest = createHash("sha256").update(await readFile(join(publicRoot, "data/model-artifacts.json"))).digest("hex");
const articleSource=await readFile(join(root,"articles/lift-splat-shoot-explained.zh-CN.md"));
const articlePublic=await readFile(join(publicRoot,"articles/lift-splat-shoot-explained.zh-CN.md"));
if(!articleSource.equals(articlePublic))throw new Error("Public Chinese article is out of sync with articles/");
console.log(`verified 6 cameras, 10 checkpoint variants, model contract ${digest.slice(0,16)}…`);
