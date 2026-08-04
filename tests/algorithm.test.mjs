import test from "node:test";
import assert from "node:assert/strict";
import {
  applyRigid, boltzmannProbabilities, cameraPointToEgo, invertMat3, mat3Multiply,
  naiveSumPool, outerProduct, poolCameraContributions, quickCumsum, softmax,
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
