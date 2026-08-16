import test from "node:test";
import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";
import {
  applyRigid, bevIndexToEgo, binaryStats, boltzmannProbabilities, cameraPointToEgo,
  cameraFrustumCorners, cameraOpticalAxis,
  determinantMat3, egoToBevIndex, egoToScreen, invertMat3, invertRigidMat4,
  float16LittleEndianToFloat32,
  mat3Multiply, mat4Multiply, mat4Vector, nearestFeatureAnchor, projectLidarPoint,
  quaternionToRotation, screenToBevIndex,
  bilinearSplatWeights, naiveSumPool, outerProduct, poolCameraContributions, quickCumsum, reduceFeatureGroup, softmax,
  trajectoryCost, transformPointSet, voxelIndex, voxelRank,
} from "../lib/algorithm.mjs";

const close = (actual, expected, epsilon = 1e-8) => assert.ok(Math.abs(actual - expected) < epsilon, `${actual} ≉ ${expected}`);

test("matrix inverse composes to identity", () => {
  const matrix = [[1266, 0, 816], [0, 1266, 492], [0, 0, 1]];
  const identity = mat3Multiply(matrix, invertMat3(matrix));
  identity.forEach((row, i) => row.forEach((value, j) => close(value, i === j ? 1 : 0)));
});

test("pixel ray and cam-to-ego composition", () => {
  const intrinsic = [[2, 0, 1], [0, 2, 1], [0, 0, 1]];
  const point = cameraPointToEgo([1, 1], 8, intrinsic, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], [2, 3, 1]);
  assert.deepEqual(point, [2, 3, 9]);
});

test("depth softmax is finite and normalized", () => {
  const probability = softmax([-1000, -1, 0, 2, 1000]);
  close(probability.reduce((sum, value) => sum + value, 0), 1);
  assert.ok(probability.every(Number.isFinite));
});

test("lift is the depth/context outer product", () => {
  assert.deepEqual(outerProduct([0.25, 0.75], [2, -4]), [[0.5, -1], [1.5, -3]]);
});

test("voxel boundaries are half-open and ranks are stable", () => {
  const bounds = [[-50, 50, 0.5], [-50, 50, 0.5], [-10, 10, 20]];
  assert.deepEqual(voxelIndex([-50, -50, -10], bounds), [0, 0, 0]);
  assert.deepEqual(voxelIndex([49.999, 49.999, 9.999], bounds), [199, 199, 0]);
  assert.equal(voxelIndex([50, 0, 0], bounds), null);
  assert.equal(voxelRank([3, 4, 0], [200, 200, 1]), 604);
});

test("QuickCumsum teaching implementation equals naive sum pooling", () => {
  const features = [[1, 4], [8, 2], [3, 1], [-2, 5], [1, 1]];
  const ranks = [7, 2, 7, 2, 9];
  assert.deepEqual(quickCumsum(features, ranks), naiveSumPool(features, ranks));
});

test("interactive pooling comparisons preserve their stated semantics",()=>{
  const features=[[.25,1],[.55,-2],[.9,.5]];
  assert.deepEqual(reduceFeatureGroup(features,"sum"),[1.7000000000000002,-.5]);
  assert.deepEqual(reduceFeatureGroup(features,"mean"),[1.7000000000000002/3,-.5/3]);
  assert.deepEqual(reduceFeatureGroup(features,"max"),[.9,1]);
  const weights=bilinearSplatWeights(2.25,3.6);
  close(weights.reduce((sum,item)=>sum+item.weight,0),1);
  assert.deepEqual(weights.map(item=>item.index),[[2,3],[3,3],[2,4],[3,4]]);
});

test("camera pooling is permutation invariant", () => {
  const cameras = [[1, 2, 3], [4, 5, 6], [-1, 2, 0]];
  assert.deepEqual(poolCameraContributions(cameras), poolCameraContributions([cameras[2], cameras[0], cameras[1]]));
});

test("rigid transforms are equivariant for a point set", () => {
  const rotation = [[0, -1, 0], [1, 0, 0], [0, 0, 1]];
  const translation = [4, -2, 0.5];
  const points = [[1, 2, 3], [-1, 0, 4]];
  assert.deepEqual(transformPointSet(points, rotation, translation), points.map((point) => applyRigid(point, rotation, translation)));
});

test("trajectory cost and Boltzmann probabilities select the low-cost path", () => {
  const costMap = ([x, y]) => Math.abs(x) + (y > 2 ? 0.5 : 0);
  const paths = [
    [[0, 0], [0, 1], [0, 2]],
    [[0, 0], [1, 1], [2, 2]],
    [[0, 0], [-2, 1], [-3, 2]],
  ];
  const costs = paths.map((path) => trajectoryCost(path, costMap));
  const probabilities = boltzmannProbabilities(costs, 0.8);
  close(probabilities.reduce((sum, value) => sum + value, 0), 1);
  assert.equal(probabilities.indexOf(Math.max(...probabilities)), costs.indexOf(Math.min(...costs)));
});

test("homogeneous rigid composition and inverse preserve a point",()=>{
  const transform=[[0,-1,0,2],[1,0,0,3],[0,0,1,.5],[0,0,0,1]];
  const identity=mat4Multiply(invertRigidMat4(transform),transform);
  identity.forEach((row,i)=>row.forEach((value,j)=>close(value,i===j?1:0)));
  const point=[4,-2,1,1],roundTrip=mat4Vector(invertRigidMat4(transform),mat4Vector(transform,point));
  roundTrip.forEach((value,index)=>close(value,point[index]));
});

test("quaternion conversion produces a proper rotation",()=>{
  const rotation=quaternionToRotation([Math.SQRT1_2,0,0,Math.SQRT1_2]);
  close(determinantMat3(rotation),1);
  const point=rotation.map(row=>row[0]);
  close(point[0],0);close(point[1],1);close(point[2],0);
});

test("official feature anchors use linspace and select the nearest cell",()=>{
  const selected=nearestFeatureAnchor([176,64]);
  assert.deepEqual(selected.index,[4,11]);
  close(selected.anchor[0],351*11/21);
  close(selected.anchor[1],127*4/7);
  assert.deepEqual(nearestFeatureAnchor([351,127]).index,[7,21]);
});

test("ego, BEV index and screen mappings share one orientation contract",()=>{
  assert.deepEqual(egoToBevIndex([0,0,0]),[100,100]);
  assert.deepEqual(bevIndexToEgo([100,100]),[.25,.25]);
  const frontLeft=egoToScreen([20,10,0]);
  assert.ok(frontLeft[0]<.5,"ego left must display left");
  assert.ok(frontLeft[1]<.5,"ego forward must display up");
  assert.deepEqual(screenToBevIndex(frontLeft),[140,120]);
  assert.equal(egoToBevIndex([50,0,0]),null);
});

test("LiDAR projection rejects points behind the camera",()=>{
  const identity=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],intrinsic=[[100,0,50],[0,100,40],[0,0,1]];
  assert.deepEqual(projectLidarPoint([1,2,10],identity,intrinsic),{camera:[1,2,10],image:[60,60,10]});
  assert.equal(projectLidarPoint([1,2,-10],identity,intrinsic),null);
});

test("camera optical axis and frustum corners follow real calibration",()=>{
  const intrinsic=[[100,0,50],[0,100,40],[0,0,1]],cameraToEgo=[[0,0,1,2],[-1,0,0,3],[0,-1,0,1],[0,0,0,1]];
  assert.deepEqual(cameraOpticalAxis(cameraToEgo),[1,0,0]);
  const corners=cameraFrustumCorners(intrinsic,cameraToEgo,100,80,10);
  assert.equal(corners.length,4);
  corners.forEach(point=>close(point[0],12));
  assert.ok(corners[0][1]>corners[1][1]);
  assert.ok(corners[0][2]>corners[3][2]);
});

test("single-frame diagnostic counts and IoU are explicit",()=>{
  assert.deepEqual(binaryStats([.9,.8,.1,.2],[1,0,1,0],.5),{truePositive:1,falsePositive:1,falseNegative:1,trueNegative:1,iou:1/3});
});

test("little-endian float16 evidence decodes without losing half the tensor",async()=>{
  const known=float16LittleEndianToFloat32(Uint8Array.from([0x00,0x3c,0x00,0xc0,0x00,0x38,0x00,0x00]));
  assert.deepEqual(Array.from(known),[1,-2,.5,0]);
  const [model,alignment]=await Promise.all([
    readFile(new URL("../public/data/model-artifacts.json",import.meta.url),"utf8").then(JSON.parse),
    readFile(new URL("../public/data/alignment.json",import.meta.url),"utf8").then(JSON.parse),
  ]);
  const bytes=Uint8Array.from(Buffer.from(model.variants["all-cameras"].logits.data,"base64"));
  const logits=float16LittleEndianToFloat32(bytes);
  const truth=Uint8Array.from(Buffer.from(model.ground_truth.mask.data,"base64"));
  const probability=Array.from(logits,value=>1/(1+Math.exp(-value)));
  const stats=binaryStats(probability,truth,.5),expected=alignment.single_frame_diagnostics.find(row=>row.threshold===.5);
  assert.equal(logits.length,40000);
  assert.deepEqual([stats.truePositive,stats.falsePositive,stats.falseNegative],[expected.tp,expected.fp,expected.fn]);
  close(stats.iou,expected.iou,1e-7);
});

test("pinned alignment contract and LiDAR binary are internally consistent",async()=>{
  const alignment=JSON.parse(await readFile(new URL("../public/data/alignment.json",import.meta.url),"utf8"));
  const lidar=await readFile(new URL("../public/data/lidar-frame.bin",import.meta.url));
  assert.equal(alignment.schema_version,"2.0.0");
  assert.deepEqual(alignment.lidar.shape,[34688,5]);
  assert.equal(lidar.byteLength,34688*5*4);
  assert.equal(alignment.camera_projections.length,6);
  assert.equal(alignment.geometry_gold.length,18);
  assert.ok(alignment.camera_projections.every(row=>row.visible_points>1000&&row.rotation_orthogonality_max_error<1e-6));
});

test("v10 public experience is English-only and defines ten LSS/BEVDepth tensor scenes",async()=>{
  const [content,explainer,layout]=await Promise.all([
    readFile(new URL("../app/lss-content.ts",import.meta.url),"utf8"),
    readFile(new URL("../app/LssExplainer.tsx",import.meta.url),"utf8"),
    readFile(new URL("../app/layout.tsx",import.meta.url),"utf8"),
  ]);
  const source=`${content}\n${explainer}\n${layout}`;
  assert.equal((content.match(/id:\s*\"[a-z-]+\"/g)??[]).length,10);
  assert.equal(/[\u3400-\u9fff]/u.test(source),false);
  assert.equal(/type\s+Locale|Localized|setLanguage|language-switch/.test(source),false);
  assert.ok(layout.includes('<html lang="en">'));
  assert.equal(/LssScene|@react-three|from ["']three["']|Open 3D|calibrated WebGL/.test(source),false);
  assert.equal((content.match(/explanation:\s*"/g)??[]).length,10);
  assert.equal((content.match(/tensor:\s*\{/g)??[]).length,10);
  assert.ok(content.split("\n").filter(line=>/explanation:\s*"/.test(line)).every(line=>line.length<560));
  assert.ok(explainer.includes("lesson-timeline"));
  assert.ok(explainer.includes("tensor-ledger"));
  assert.ok(explainer.includes("frame-control"));
  assert.ok(explainer.includes("pool-control"));
  assert.ok(content.includes("get_downsampled_gt_depth()"));
  assert.ok(content.includes("L=Ldet+3Ldepth"));
  assert.ok(explainer.includes("DRAG TO ROTATE"));
});
