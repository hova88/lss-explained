export function softmax(values) {
  const maximum = Math.max(...values);
  const exponents = values.map((value) => Math.exp(value - maximum));
  const total = exponents.reduce((sum, value) => sum + value, 0);
  return exponents.map((value) => value / total);
}

export function mat3Multiply(a, b) {
  return Array.from({ length: 3 }, (_, row) =>
    Array.from({ length: 3 }, (_, column) =>
      a[row][0] * b[0][column] + a[row][1] * b[1][column] + a[row][2] * b[2][column]),
  );
}

export function mat3Vector(a, value) {
  return a.map((row) => row[0] * value[0] + row[1] * value[1] + row[2] * value[2]);
}

export function invertMat3(m) {
  const determinant =
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  if (Math.abs(determinant) < 1e-12) throw new Error("Matrix is singular");
  const inverse = [
    [m[1][1] * m[2][2] - m[1][2] * m[2][1], m[0][2] * m[2][1] - m[0][1] * m[2][2], m[0][1] * m[1][2] - m[0][2] * m[1][1]],
    [m[1][2] * m[2][0] - m[1][0] * m[2][2], m[0][0] * m[2][2] - m[0][2] * m[2][0], m[0][2] * m[1][0] - m[0][0] * m[1][2]],
    [m[1][0] * m[2][1] - m[1][1] * m[2][0], m[0][1] * m[2][0] - m[0][0] * m[2][1], m[0][0] * m[1][1] - m[0][1] * m[1][0]],
  ];
  return inverse.map((row) => row.map((value) => value / determinant));
}

export function undoPostTransform(pixel, postRotation, postTranslation) {
  return mat3Vector(invertMat3(postRotation), pixel.map((value, index) => value - postTranslation[index]));
}

export function pixelToCamera(pixel, depth, intrinsic) {
  const ray = mat3Vector(invertMat3(intrinsic), [pixel[0] * depth, pixel[1] * depth, depth]);
  return ray;
}

export function applyRigid(point, rotation, translation) {
  return mat3Vector(rotation, point).map((value, index) => value + translation[index]);
}

export function cameraPointToEgo(pixel, depth, intrinsic, rotation, translation, postRotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], postTranslation = [0, 0, 0]) {
  const originalPixel = undoPostTransform([pixel[0], pixel[1], 1], postRotation, postTranslation);
  return applyRigid(pixelToCamera(originalPixel, depth, intrinsic), rotation, translation);
}

export function outerProduct(depthProbability, context) {
  return depthProbability.map((probability) => context.map((value) => probability * value));
}

export function voxelIndex(point, bounds) {
  return point.map((value, axis) => {
    const [minimum, maximum, step] = bounds[axis];
    if (value < minimum || value >= maximum) return null;
    return Math.floor((value - minimum) / step);
  }).every(Number.isInteger) ? point.map((value, axis) => Math.floor((value - bounds[axis][0]) / bounds[axis][2])) : null;
}

export function voxelRank(index, gridShape, batch = 0, batchSize = 1) {
  const [x, y, z] = index;
  return x * gridShape[1] * gridShape[2] * batchSize + y * gridShape[2] * batchSize + z * batchSize + batch;
}

export function quickCumsum(features, ranks) {
  if (!features.length) return { values: [], ranks: [] };
  const order = ranks.map((rank, index) => ({ rank, index })).sort((a, b) => a.rank - b.rank);
  const values = [];
  const uniqueRanks = [];
  for (const item of order) {
    const last = uniqueRanks.length - 1;
    if (last < 0 || uniqueRanks[last] !== item.rank) {
      uniqueRanks.push(item.rank);
      values.push([...features[item.index]]);
    } else {
      features[item.index].forEach((value, channel) => { values[last][channel] += value; });
    }
  }
  return { values, ranks: uniqueRanks };
}

export function naiveSumPool(features, ranks) {
  const buckets = new Map();
  features.forEach((feature, index) => {
    const current = buckets.get(ranks[index]) ?? Array(feature.length).fill(0);
    feature.forEach((value, channel) => { current[channel] += value; });
    buckets.set(ranks[index], current);
  });
  const sorted = [...buckets.entries()].sort((a, b) => a[0] - b[0]);
  return { ranks: sorted.map(([rank]) => rank), values: sorted.map(([, value]) => value) };
}

export function poolCameraContributions(cameraFeatures) {
  return cameraFeatures.reduce((sum, camera) => camera.map((value, index) => value + sum[index]), Array(cameraFeatures[0]?.length ?? 0).fill(0));
}

export function transformPointSet(points, rotation, translation) {
  return points.map((point) => applyRigid(point, rotation, translation));
}

export function trajectoryCost(points, costMap, comfortWeight = 0.12) {
  let cost = 0;
  for (let index = 0; index < points.length; index += 1) {
    cost += costMap(points[index]);
    if (index > 1) {
      const a = points[index - 2], b = points[index - 1], c = points[index];
      cost += comfortWeight * Math.hypot(c[0] - 2 * b[0] + a[0], c[1] - 2 * b[1] + a[1]);
    }
  }
  return cost;
}

export function boltzmannProbabilities(costs, temperature = 1) {
  if (!(temperature > 0)) throw new Error("Temperature must be positive");
  return softmax(costs.map((cost) => -cost / temperature));
}
