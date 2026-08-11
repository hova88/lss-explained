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

export function mat4Multiply(a, b) {
  return Array.from({ length: 4 }, (_, row) =>
    Array.from({ length: 4 }, (_, column) =>
      a[row].reduce((sum, value, index) => sum + value * b[index][column], 0)),
  );
}

export function mat4Vector(matrix, point) {
  const value = point.length === 4 ? point : [...point, 1];
  return matrix.map((row) => row.reduce((sum, entry, index) => sum + entry * value[index], 0));
}

export function rigidToMat4(rotation, translation) {
  return [
    [...rotation[0], translation[0]],
    [...rotation[1], translation[1]],
    [...rotation[2], translation[2]],
    [0, 0, 0, 1],
  ];
}

export function invertRigidMat4(matrix) {
  const rotation = matrix.slice(0, 3).map((row) => row.slice(0, 3));
  const transposed = Array.from({ length: 3 }, (_, row) => Array.from({ length: 3 }, (_, column) => rotation[column][row]));
  const translation = matrix.slice(0, 3).map((row) => row[3]);
  const inverseTranslation = mat3Vector(transposed, translation).map((value) => -value);
  return rigidToMat4(transposed, inverseTranslation);
}

export function quaternionToRotation([w, x, y, z]) {
  const norm = Math.hypot(w, x, y, z);
  if (norm < 1e-12) throw new Error("Quaternion has zero norm");
  [w, x, y, z] = [w / norm, x / norm, y / norm, z / norm];
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ];
}

export function determinantMat3(matrix) {
  return matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) -
    matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) +
    matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]);
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

export function cameraOpticalAxis(cameraToEgo) {
  const axis = [cameraToEgo[0][2], cameraToEgo[1][2], cameraToEgo[2][2]];
  const norm = Math.hypot(...axis);
  if (norm < 1e-12) throw new Error("Camera optical axis has zero norm");
  return axis.map((value) => value / norm);
}

export function cameraFrustumCorners(intrinsic, cameraToEgo, width, height, depth) {
  if (!(depth > 0)) throw new Error("Frustum depth must be positive");
  const rotation = cameraToEgo.slice(0, 3).map((row) => row.slice(0, 3));
  const translation = cameraToEgo.slice(0, 3).map((row) => row[3]);
  return [[0, 0], [width, 0], [width, height], [0, height]].map((pixel) =>
    applyRigid(pixelToCamera(pixel, depth, intrinsic), rotation, translation));
}

export function projectCameraPoint(point, intrinsic) {
  if (!(point[2] > 0)) return null;
  const projected = mat3Vector(intrinsic, point);
  return [projected[0] / projected[2], projected[1] / projected[2], point[2]];
}

export function projectLidarPoint(point, lidarToCamera, intrinsic) {
  const camera = mat4Vector(lidarToCamera, point).slice(0, 3);
  const image = projectCameraPoint(camera, intrinsic);
  return image ? { camera, image } : null;
}

export function linspace(start, end, count) {
  if (count === 1) return [start];
  return Array.from({ length: count }, (_, index) => start + (end - start) * index / (count - 1));
}

export function nearestFeatureAnchor(pixel, width = 352, height = 128, featureWidth = 22, featureHeight = 8) {
  const xs = linspace(0, width - 1, featureWidth);
  const ys = linspace(0, height - 1, featureHeight);
  const nearest = (values, target) => values.reduce((best, value, index) =>
    Math.abs(value - target) < Math.abs(values[best] - target) ? index : best, 0);
  const x = nearest(xs, pixel[0]);
  const y = nearest(ys, pixel[1]);
  return { index: [y, x], anchor: [xs[x], ys[y]], delta: [pixel[0] - xs[x], pixel[1] - ys[y]] };
}

export function egoToBevIndex(point, bounds = [[-50, 50, 0.5], [-50, 50, 0.5]]) {
  const index = point.slice(0, 2).map((value, axis) => Math.floor((value - bounds[axis][0]) / bounds[axis][2]));
  return index.every((value, axis) => value >= 0 && value < Math.round((bounds[axis][1] - bounds[axis][0]) / bounds[axis][2])) ? index : null;
}

export function bevIndexToEgo(index, bounds = [[-50, 50, 0.5], [-50, 50, 0.5]]) {
  return index.map((value, axis) => bounds[axis][0] + (value + 0.5) * bounds[axis][2]);
}

export function egoToScreen(point, bounds = [[-50, 50, 0.5], [-50, 50, 0.5]]) {
  const index = egoToBevIndex(point, bounds);
  if (!index) return null;
  const width = Math.round((bounds[1][1] - bounds[1][0]) / bounds[1][2]);
  const height = Math.round((bounds[0][1] - bounds[0][0]) / bounds[0][2]);
  return [1 - (index[1] + 0.5) / width, 1 - (index[0] + 0.5) / height];
}

export function screenToBevIndex(screen, gridSize = 200) {
  const x = Math.min(gridSize - 1, Math.max(0, Math.floor((1 - screen[1]) * gridSize)));
  const y = Math.min(gridSize - 1, Math.max(0, Math.floor((1 - screen[0]) * gridSize)));
  return [x, y];
}

export function binaryStats(probabilities, groundTruth, threshold = 0.5) {
  let truePositive = 0, falsePositive = 0, falseNegative = 0, trueNegative = 0;
  probabilities.forEach((probability, index) => {
    const predicted = probability >= threshold;
    const truth = groundTruth[index] > 0;
    if (predicted && truth) truePositive += 1;
    else if (predicted) falsePositive += 1;
    else if (truth) falseNegative += 1;
    else trueNegative += 1;
  });
  const union = truePositive + falsePositive + falseNegative;
  return { truePositive, falsePositive, falseNegative, trueNegative, iou: union ? truePositive / union : 1 };
}

export function float16LittleEndianToFloat32(bytes) {
  if (bytes.byteLength % 2 !== 0) throw new Error("float16 data must contain an even number of bytes");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const output = new Float32Array(bytes.byteLength / 2);
  for (let index = 0; index < output.length; index += 1) {
    const half = view.getUint16(index * 2, true);
    const sign = half & 0x8000 ? -1 : 1;
    const exponent = (half >> 10) & 0x1f;
    const fraction = half & 0x03ff;
    if (exponent === 0) output[index] = fraction === 0 ? sign * 0 : sign * 2 ** -14 * (fraction / 1024);
    else if (exponent === 0x1f) output[index] = fraction === 0 ? sign * Infinity : Number.NaN;
    else output[index] = sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
  }
  return output;
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
