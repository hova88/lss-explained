"use client";
/* Browser-native images keep the pinned nuScenes asset paths auditable in a static export. */
/* eslint-disable @next/next/no-img-element */

import { ChevronLeft, ChevronRight, Pause, Play } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  applyRigid,
  binaryStats,
  boltzmannProbabilities,
  determinantMat3,
  egoToBevIndex,
  invertMat3,
  invertRigidMat4,
  mat3Multiply,
  mat4Multiply,
  mat4Vector,
  nearestFeatureAnchor,
  pixelToCamera,
  projectLidarPoint,
  trajectoryCost,
  undoPostTransform,
} from "../lib/algorithm.mjs";
import {
  CAMERA_NAMES,
  LssScene,
  type BevMode,
  type SceneSelection,
  type Trajectory,
  type Vehicle,
} from "./LssScene";
import { ACTS, CHAPTERS, actForChapter, tx, type Locale } from "./lss-content";

type Mat3 = [[number, number, number], [number, number, number], [number, number, number]];
type Vec3 = [number, number, number];
type PanelMode = "lesson" | "lab";
type EncodedArray = { shape: number[]; dtype: string; data: string };
type CameraRecord = {
  name: string;
  image: string;
  network_image: string;
  cam2img: number[][];
  cam2ego: number[][];
  lidar2cam: number[][];
  post_rot: number[][];
  post_trans: number[];
  timestamp: number;
  augmentation: { resize: number; resize_dims: number[]; crop: number[] };
  matrix_metadata: Record<string, { from_frame: string; to_frame: string; source: string }>;
};
type Rig = {
  schema_version: string;
  sample_token: string;
  timestamp: number;
  cameras: CameraRecord[];
  lidar2ego: number[][];
  ego2global: number[][];
  grid: Record<string, number[]>;
  vehicles_ego: Vehicle[];
  frames: Record<string, { axes: string; unit: string; origin?: string; shape?: number[] }>;
};
type Features = {
  depth_probabilities: EncodedArray;
  context_features: EncodedArray;
  feature_anchors: { x: number[]; y: number[] };
};
type Variant = {
  image: string;
  logits: EncodedArray;
  probability_min: number;
  probability_max: number;
  probability_mean: number;
  evidence: string;
};
type Model = {
  source_hashes: { checkpoint: string };
  variants: Record<string, Variant>;
  ground_truth: { mask: EncodedArray; image: string };
  geometry_contributors: { counts: EncodedArray };
  tensor_checks: { finite: boolean; depth_probability_sum_max_error: number };
};
type Alignment = {
  lidar: { path: string; sha256: string; shape: number[]; fields: string[]; xyz_min: number[]; xyz_max: number[] };
  camera_projections: {
    camera: string;
    visible_points: number;
    delta_to_lidar_ms: number;
    direct_vs_static_chain_max_abs: number;
    rotation_det: number;
    rotation_orthogonality_max_error: number;
  }[];
  lidar_occupancy: { counts: EncodedArray };
  single_frame_diagnostics: { threshold: number; tp: number; fp: number; fn: number; tn: number; iou: number }[];
  geometry_gold: { camera: string; index: number[]; processed_anchor: number[]; depth: number; ego: number[] }[];
  warning: string;
};
type Trace = {
  clicked: Vec3;
  anchor: Vec3;
  original: Vec3;
  scaled: Vec3;
  camera: Vec3;
  ego: Vec3;
  depth: number;
};

const asset = (path: string) => `${process.env.NEXT_PUBLIC_BASE_PATH ?? ""}${path}`;
const short = (name: string) => name.replace("CAM_", "").replaceAll("_", " ");
const f = (value: number, digits = 3) => Number(value).toFixed(digits);

function half(value: number) {
  const sign = value & 0x8000 ? -1 : 1;
  const exponent = (value >> 10) & 0x1f;
  const fraction = value & 0x3ff;
  if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
  if (exponent === 31) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

function bytes(encoded?: EncodedArray) {
  if (!encoded || typeof window === "undefined") return null;
  const binary = atob(encoded.data);
  const array = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) array[i] = binary.charCodeAt(i);
  return array;
}

function decodeFloat(encoded?: EncodedArray) {
  const array = bytes(encoded);
  if (!array || !encoded) return null;
  if (encoded.dtype === "float16") {
    const view = new DataView(array.buffer);
    const out = new Float32Array(array.length / 2);
    for (let i = 0; i < out.length; i += 1) out[i] = half(view.getUint16(i * 2, true));
    return out;
  }
  return new Float32Array(array.buffer);
}

function decodeUint8(encoded?: EncodedArray) {
  const array = bytes(encoded);
  return array ? new Uint8Array(array) : null;
}

function decodeUint16(encoded?: EncodedArray) {
  const array = bytes(encoded);
  if (!array) return null;
  const view = new DataView(array.buffer);
  const out = new Uint16Array(array.length / 2);
  for (let i = 0; i < out.length; i += 1) out[i] = view.getUint16(i * 2, true);
  return out;
}

function identityResidual(matrix: number[][]) {
  const product = mat4Multiply(invertRigidMat4(matrix), matrix);
  return Math.max(...product.flatMap((row, i) => row.map((value, j) => Math.abs(value - (i === j ? 1 : 0)))));
}

function Matrix({ label, values }: { label: string; values: number[][] }) {
  return (
    <div className="matrix">
      <b>{label}</b>
      {values.map((row, index) => (
        <code key={index}>{row.map((value) => f(value, Math.abs(value) > 99 ? 1 : 4).padStart(9, " ")).join(" ")}</code>
      ))}
    </div>
  );
}

function Pills({ items }: { items: { name: string; value: string; kind?: string }[] }) {
  return (
    <div className="fact-row">
      {items.map((item) => (
        <span key={item.name} className={item.kind}>
          <b>{item.value}</b>
          <small>{item.name}</small>
        </span>
      ))}
    </div>
  );
}

function CurriculumRail({ step, locale, onGo }: { step: number; locale: Locale; onGo: (step: number) => void }) {
  return (
    <nav className="curriculum-rail" aria-label={locale === "zh-CN" ? "课程阶段" : "Course stages"}>
      {ACTS.map((act) => {
        const active = step >= act.range[0] && step <= act.range[1];
        const done = step > act.range[1];
        return (
          <button key={act.id} className={active ? "active" : done ? "done" : ""} aria-current={active ? "step" : undefined} onClick={() => onGo(act.range[0])}>
            <i />
            <span>{tx(locale, act.short)}</span>
            <small>{String(act.range[0] + 1).padStart(2, "0")}—{String(act.range[1] + 1).padStart(2, "0")}</small>
          </button>
        );
      })}
      <div className="rail-signal" style={{ left: `${((step + 0.5) / CHAPTERS.length) * 100}%` }} />
    </nav>
  );
}

function LessonPanel({
  step,
  locale,
  detail,
  onAdvance,
  onMinimize,
}: {
  step: number;
  locale: Locale;
  detail: number;
  onAdvance: () => void;
  onMinimize: () => void;
}) {
  const chapter = CHAPTERS[step];
  const act = actForChapter(step);
  const finalDetail = detail === chapter.layers.length - 1;
  return (
    <section className="lesson-panel" key={`${step}-${locale}`} aria-live="polite" aria-labelledby="lesson-title">
      <button className="card-minimize" onClick={onMinimize} aria-label={locale === "zh-CN" ? "收起卡片，专注场景" : "Minimize card and focus the scene"}>— {locale === "zh-CN" ? "沉浸场景" : "FOCUS SCENE"}</button>
      <div className="lesson-meta">
        <span>{tx(locale, act.label)}</span>
        <b>{String(step + 1).padStart(2, "0")} / {CHAPTERS.length}</b>
      </div>
      <small className="lesson-stage">{chapter.stage}</small>
      <h1 id="lesson-title">{tx(locale, chapter.title)}</h1>
      <p className="lesson-question">{tx(locale, chapter.question)}</p>
      <p className="lesson-answer">{tx(locale, chapter.answer)}</p>
      <ol className="lesson-layers">
        {chapter.layers.slice(0, detail + 1).map((layer, index) => (
          <li key={index} style={{ "--delay": `${index * 80}ms` } as React.CSSProperties}>
            <i>{index + 1}</i>
            <span>{tx(locale, layer)}</span>
          </li>
        ))}
      </ol>
      {finalDetail && <code className="lesson-formula">{chapter.formula}</code>}
      <div className="lesson-source">
        <span>{chapter.evidence}</span>
        <small>{chapter.source}</small>
      </div>
      <button className="deeper-button" onClick={onAdvance}>
        <span>{finalDetail ? (locale === "zh-CN" ? "进入交互实验" : "Enter the interactive lab") : (locale === "zh-CN" ? "继续深入" : "Go one layer deeper")}</span>
        <b>{finalDetail ? (locale === "zh-CN" ? "动手验证 →" : "EXPLORE →") : `${detail + 2} / 3`}</b>
      </button>
    </section>
  );
}

const FRAME_EDGES = ["raw→network", "network→camera", "camera→ego", "ego→BEV"] as const;

function FrameGraph({ locale, active, onActive }: { locale: Locale; active: string; onActive: (value: string) => void }) {
  const labels = locale === "zh-CN"
    ? ["原图像素", "网络像素", "相机 3D", "ego 3D", "BEV"]
    : ["raw pixel", "network pixel", "camera 3D", "ego 3D", "BEV"];
  return (
    <div className="frame-graph" aria-label="Coordinate frame graph">
      <div className="frame-graph-title">FRAME GRAPH <span>{locale === "zh-CN" ? "点击边查看真实矩阵" : "click an edge"}</span></div>
      <div className="frame-main">
        {labels.map((label, index) => (
          <span key={label}>
            <i>{index < 2 ? "px" : index === 4 ? "idx" : "m"}</i>{label}
            {index < labels.length - 1 && (
              <button className={active === FRAME_EDGES[index] ? "active" : ""} aria-pressed={active === FRAME_EDGES[index]} aria-label={`${labels[index]} → ${labels[index + 1]}`} onClick={() => onActive(FRAME_EDGES[index])}>→</button>
            )}
          </span>
        ))}
      </div>
      <div className="frame-lidar">
        <span><i>m</i>LiDAR</span>
        <button className={active === "lidar→ego" ? "active" : ""} aria-pressed={active === "lidar→ego"} onClick={() => onActive("lidar→ego")}>→ ego</button>
        <button className={active === "lidar→camera" ? "active" : ""} aria-pressed={active === "lidar→camera"} onClick={() => onActive("lidar→camera")}>→ camera</button>
        <button className={active === "ego→global" ? "active" : ""} aria-pressed={active === "ego→global"} onClick={() => onActive("ego→global")}>ego → global</button>
      </div>
    </div>
  );
}

function TraceRibbon({
  step,
  locale,
  camera,
  anchor,
  trace,
}: {
  step: number;
  locale: Locale;
  camera: CameraRecord | null;
  anchor: { index: [number, number] };
  trace: Trace | null;
}) {
  if (step < 4 || step > 14) return null;
  const bev = trace ? egoToBevIndex(trace.ego) : null;
  const active = step <= 4 ? 0 : step === 5 ? 1 : step <= 7 ? 2 : step <= 9 ? 3 : step <= 11 ? 4 : 5;
  const nodes = [
    { label: locale === "zh-CN" ? "图像" : "IMAGE", value: short(camera?.name ?? "camera") },
    { label: "FEATURE", value: `[${anchor.index.join(",")}]` },
    { label: locale === "zh-CN" ? "深度" : "DEPTH", value: trace ? `${f(trace.depth, 0)} m` : "—" },
    { label: "EGO", value: trace ? `[${f(trace.ego[0], 1)}, ${f(trace.ego[1], 1)}] m` : "—" },
    { label: "BEV", value: bev ? `[${bev[0]}, ${bev[1]}]` : locale === "zh-CN" ? "越界" : "outside" },
    { label: locale === "zh-CN" ? "任务" : "TASK", value: step === 12 ? "BCE loss" : step === 13 ? "σ(logit)" : "evidence" },
  ];
  return (
    <div className="trace-ribbon" aria-label={locale === "zh-CN" ? "同一真实采样点的完整数据链" : "One real sample traced through the pipeline"}>
      <b>{locale === "zh-CN" ? "同一数据线索" : "ONE DATA THREAD"}</b>
      <div>
        {nodes.map((node, index) => (
          <span key={node.label} className={index === active ? "active" : index < active ? "past" : ""}>
            <small>{node.label}</small><code>{node.value}</code>{index < nodes.length - 1 && <i>→</i>}
          </span>
        ))}
      </div>
    </div>
  );
}

function MatrixInspector({
  edge,
  camera,
  rig,
  alignment,
  locale,
  onClose,
}: {
  edge: string;
  camera: CameraRecord | null;
  rig: Rig | null;
  alignment: Alignment | null;
  locale: Locale;
  onClose: () => void;
}) {
  if (!camera || !rig) return null;
  let matrix: number[][] = camera.cam2ego;
  let label = "Tcam→ego";
  let source = camera.matrix_metadata.cam2ego.source;
  let check = `det R = ${f(determinantMat3(matrix.slice(0, 3).map((row) => row.slice(0, 3)) as Mat3), 8)} · T⁻¹T max = ${identityResidual(matrix).toExponential(2)}`;
  if (edge === "raw→network") {
    matrix = camera.post_rot;
    label = "A · post_rot";
    source = camera.matrix_metadata.post_transform.source;
    check = `resize ${camera.augmentation.resize} · crop y=${camera.augmentation.crop[1]}`;
  } else if (edge === "network→camera") {
    matrix = camera.cam2img;
    label = "K · cam2img";
    source = camera.matrix_metadata.cam2img.source;
    const id = mat3Multiply(invertMat3(matrix as Mat3), matrix as Mat3);
    check = `K⁻¹K max = ${Math.max(...id.flatMap((row, i) => row.map((value, j) => Math.abs(value - (i === j ? 1 : 0))))).toExponential(2)}`;
  } else if (edge === "ego→global") {
    matrix = rig.ego2global;
    label = "Tego→global";
    source = "nuScenes ego_pose";
    check = `T⁻¹T max = ${identityResidual(matrix).toExponential(2)}`;
  } else if (edge === "lidar→ego") {
    matrix = rig.lidar2ego;
    label = "Tlidar→ego";
    source = "nuScenes calibrated_sensor";
    check = `T⁻¹T max = ${identityResidual(matrix).toExponential(2)}`;
  } else if (edge === "lidar→camera") {
    matrix = camera.lidar2cam;
    label = "Tlidar→camera";
    source = camera.matrix_metadata.lidar2cam.source;
    const projection = alignment?.camera_projections.find((value) => value.camera === camera.name);
    check = `Δt=${f(projection?.delta_to_lidar_ms ?? 0, 2)} ms · direct/static residual=${f(projection?.direct_vs_static_chain_max_abs ?? 0, 4)}`;
  } else if (edge === "ego→BEV") {
    matrix = [[0.5, 0, -50], [0, 0.5, -50], [0, 0, 1]];
    label = "metric → integer grid";
    source = "LSS grid_conf";
    check = "ix=floor((x+50)/0.5) · iy=floor((y+50)/0.5)";
  }
  return (
    <aside className="matrix-inspector">
      <button onClick={onClose} aria-label="Close">×</button>
      <span>{locale === "zh-CN" ? "变换边检查" : "TRANSFORM EDGE"}</span>
      <h2>{edge}</h2>
      <Matrix label={label} values={matrix} />
      <p>{source}</p>
      <code>{check}</code>
      {edge === "lidar→camera" && (
        <small>{locale === "zh-CN" ? "直接矩阵含两个传感器时刻之间的 ego-pose 链，不能用静态外参乘积替代。" : "The direct matrix includes the ego-pose chain between sensor timestamps; static extrinsics are not equivalent."}</small>
      )}
    </aside>
  );
}

function CameraRig({ rig, selected, onSelect, locale }: { rig: Rig | null; selected: number; onSelect: (value: number) => void; locale: Locale }) {
  return (
    <div className="camera-rig">
      {(rig?.cameras ?? []).map((camera, index) => (
        <button key={camera.name} className={selected === index ? "selected" : ""} aria-pressed={selected === index} onClick={() => onSelect(index)}>
          <img src={asset(camera.image)} alt={short(camera.name)} />
          <span>{short(camera.name)}</span>
          <em>+z {locale === "zh-CN" ? "光轴" : "optical"}</em>
        </button>
      ))}
    </div>
  );
}

function NetworkPixel({
  camera,
  pixel,
  setPixel,
  anchor,
  locale,
}: {
  camera: CameraRecord | null;
  pixel: [number, number];
  setPixel: (value: [number, number]) => void;
  anchor: { index: [number, number]; anchor: [number, number]; delta: [number, number] };
  locale: Locale;
}) {
  return (
    <div className="network-pixel">
      <div
        className="focus-image"
        onPointerDown={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          setPixel([
            Math.max(0, Math.min(351, ((event.clientX - rect.left) / rect.width) * 352)),
            Math.max(0, Math.min(127, ((event.clientY - rect.top) / rect.height) * 128)),
          ]);
        }}
      >
        <img src={asset(camera?.network_image ?? "/data/network-images/cam-front.jpg")} alt="Network input" />
        <i className="clicked" style={{ left: `${(pixel[0] / 351) * 100}%`, top: `${(pixel[1] / 127) * 100}%` }} />
        <i className="anchor" style={{ left: `${(anchor.anchor[0] / 351) * 100}%`, top: `${(anchor.anchor[1] / 127) * 100}%` }} />
      </div>
      <div className="pixel-readout">
        <b>{locale === "zh-CN" ? "点击像素" : "click"} [{f(pixel[0], 1)}, {f(pixel[1], 1)}]</b>
        <b>{locale === "zh-CN" ? "官方 anchor" : "official anchor"} [{f(anchor.anchor[0], 2)}, {f(anchor.anchor[1], 2)}]</b>
        <code>feature [h={anchor.index[0]}, w={anchor.index[1]}]</code>
        <small>Δ=[{f(anchor.delta[0], 2)}, {f(anchor.delta[1], 2)}] px</small>
      </div>
    </div>
  );
}

function DepthChart({ values, selected, onSelect }: { values: number[]; selected: number; onSelect: (value: number) => void }) {
  const max = Math.max(...values, 0.001);
  return (
    <div className="depth-chart">
      {values.map((value, index) => (
        <button
          key={index}
          className={index === selected ? "selected" : ""}
          style={{ height: `${Math.max(3, (value / max) * 100)}%` }}
          title={`${4 + index}m · ${value.toFixed(6)}`}
          onClick={() => onSelect(index)}
        />
      ))}
    </div>
  );
}

function ContextChart({ values }: { values: number[] }) {
  const max = Math.max(...values.map(Math.abs), 0.001);
  return (
    <div className="context-chart">
      {values.map((value, index) => (
        <i key={index} title={`c${index}=${value.toFixed(4)}`} style={{ height: `${Math.max(2, (Math.abs(value) / max) * 100)}%`, background: value >= 0 ? "#4f8396" : "#d85b35" }} />
      ))}
    </div>
  );
}

function FeatureHeatmap({ values, selected }: { values: number[]; selected: [number, number] }) {
  const max = Math.max(...values.map(Math.abs), 0.001);
  return (
    <div className="feature-heatmap" aria-label="8 by 22 checkpoint context feature">
      {values.map((value, index) => {
        const h = Math.floor(index / 22);
        const w = index % 22;
        const strength = Math.abs(value) / max;
        return <i key={index} className={h === selected[0] && w === selected[1] ? "selected" : ""} style={{ background: value >= 0 ? `rgba(79,131,150,${0.08 + strength * 0.92})` : `rgba(216,91,53,${0.08 + strength * 0.92})` }} />;
      })}
    </div>
  );
}

function ProjectedCamera({
  camera,
  rig,
  lidar,
  selected,
  onSelect,
}: {
  camera: CameraRecord | null;
  rig: Rig | null;
  lidar: Float32Array | null;
  selected: number | null;
  onSelect: (value: SceneSelection) => void;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const projected = useMemo(() => {
    if (!camera || !lidar || !rig) return [] as { index: number; u: number; v: number; depth: number; ego: [number, number, number] }[];
    const rows: { index: number; u: number; v: number; depth: number; ego: [number, number, number] }[] = [];
    for (let i = 0; i < lidar.length / 5; i += 1) {
      const point = [lidar[i * 5], lidar[i * 5 + 1], lidar[i * 5 + 2]];
      const hit = projectLidarPoint(point, camera.lidar2cam, camera.cam2img as Mat3);
      if (!hit) continue;
      const [u, v, depth] = hit.image;
      if (u < 0 || u >= 1600 || v < 0 || v >= 900) continue;
      const ego = mat4Vector(rig.lidar2ego, point);
      rows.push({ index: i, u, v, depth, ego: [ego[0], ego[1], ego[2]] });
    }
    return rows;
  }, [camera, lidar, rig]);
  useEffect(() => {
    const context = canvas.current?.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, 1600, 900);
    for (const point of projected) {
      const t = Math.min(1, point.depth / 65);
      context.fillStyle = point.index === selected ? "#ffe36e" : `rgba(${Math.round(55 + 190 * t)},${Math.round(190 - 80 * t)},${Math.round(210 - 120 * t)},.8)`;
      context.beginPath();
      context.arc(point.u, point.v, point.index === selected ? 7 : 2.4, 0, Math.PI * 2);
      context.fill();
    }
  }, [projected, selected]);
  return (
    <div className="projected-camera">
      <img src={asset(camera?.image ?? "/data/images/cam-front.jpg")} alt="LiDAR projected into selected camera" />
      <canvas
        ref={canvas}
        width={1600}
        height={900}
        onPointerDown={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          const u = ((event.clientX - rect.left) / rect.width) * 1600;
          const v = ((event.clientY - rect.top) / rect.height) * 900;
          let best: null | (typeof projected)[number] = null;
          let distance = (18 * 1600) / rect.width;
          for (const point of projected) {
            const d = Math.hypot(point.u - u, point.v - v);
            if (d < distance) { distance = d; best = point; }
          }
          if (best && lidar) {
            onSelect({
              kind: "lidar",
              index: best.index,
              lidar: [lidar[best.index * 5], lidar[best.index * 5 + 1], lidar[best.index * 5 + 2]],
              ego: best.ego,
            });
          }
        }}
      />
    </div>
  );
}

function PaperTable() {
  return (
    <div className="paper-table">
      <div><b>Method</b><b>Car</b><b>Vehicles</b></div>
      {[["CNN", "22.78", "24.25"], ["Frozen encoder", "25.51", "26.83"], ["OFT", "29.72", "30.05"], ["Lift-Splat", "32.06", "32.07"]].map((row) => (
        <div key={row[0]} className={row[0] === "Lift-Splat" ? "highlight" : ""}>{row.map((value) => <span key={value}>{value}</span>)}</div>
      ))}
      <small>Paper Table 1 · nuScenes IoU (%)</small>
    </div>
  );
}

function SelectionCard({
  selection,
  rig,
  camera,
  modelProbability,
  gt,
  lidarOccupancy,
  contributors,
  locale,
  onClose,
}: {
  selection: SceneSelection;
  rig: Rig | null;
  camera: CameraRecord | null;
  modelProbability: Float32Array | null;
  gt: Uint8Array | null;
  lidarOccupancy: Uint16Array | null;
  contributors: Uint16Array | null;
  locale: Locale;
  onClose: () => void;
}) {
  let title = "Inspection";
  let label = "GEOMETRY";
  let rows: [string, string][] = [];
  if (selection.kind === "camera") {
    const selectedCamera = rig?.cameras[selection.index];
    title = short(CAMERA_NAMES[selection.index]);
    label = "CALIBRATED SENSOR";
    rows = [
      ["t cam→ego", selectedCamera ? selectedCamera.cam2ego.slice(0, 3).map((row) => f(row[3])).join(", ") : "…"],
      ["fx · fy", selectedCamera ? `${f(selectedCamera.cam2img[0][0], 2)} · ${f(selectedCamera.cam2img[1][1], 2)}` : "…"],
      ["optical axis", selectedCamera ? selectedCamera.cam2ego.slice(0, 3).map((row) => f(row[2])).join(", ") : "…"],
    ];
  } else if (selection.kind === "lidar") {
    title = `LiDAR #${selection.index}`;
    label = "REFERENCE LIDAR · NOT MODEL INPUT";
    const hit = camera ? projectLidarPoint(selection.lidar, camera.lidar2cam, camera.cam2img as Mat3) : null;
    const index = egoToBevIndex(selection.ego);
    rows = [
      ["lidar xyz", selection.lidar.map((value) => f(value)).join(", ")],
      ["ego xyz", selection.ego.map((value) => f(value)).join(", ")],
      ["BEV index", index ? `[${index.join(", ")}]` : "outside"],
      [short(camera?.name ?? "camera"), hit ? `uv=[${f(hit.image[0], 1)}, ${f(hit.image[1], 1)}], z=${f(hit.image[2], 2)}m` : "behind camera"],
    ];
  } else if (selection.kind === "cell") {
    title = `BEV [${selection.index.join(", ")}]`;
    label = "ALIGNED CELL";
    const flat = selection.index[0] * 200 + selection.index[1];
    const probability = modelProbability?.[flat] ?? 0;
    const truth = (gt?.[flat] ?? 0) > 0;
    const lidar = lidarOccupancy?.[flat] ?? 0;
    const byCamera = contributors ? CAMERA_NAMES.map((name, index) => `${short(name)}:${contributors[index * 40000 + flat]}`).join(" · ") : "…";
    rows = [
      ["ego center", `${selection.center.map((value) => f(value, 2)).join(", ")} m`],
      ["logit / sigmoid", `${f(Math.log(Math.max(probability, 1e-7) / Math.max(1 - probability, 1e-7)), 3)} / ${f(probability, 4)}`],
      ["GT / LiDAR pts", `${truth ? "vehicle" : "background"} / ${lidar}`],
      ["screen", `u=${f(1 - (selection.index[1] + 0.5) / 200, 4)} · v=${f(1 - (selection.index[0] + 0.5) / 200, 4)}`],
      ["frustum samples", byCamera],
    ];
  } else if (selection.kind === "object") {
    const vehicle = rig?.vehicles_ego[selection.index];
    title = `GT object ${selection.index}`;
    label = "NUSCENES GT";
    rows = [
      ["center ego", vehicle?.center_ego.map((value) => f(value)).join(", ") ?? "…"],
      ["l × w × h", vehicle?.dimensions.map((value) => f(value)).join(" × ") ?? "…"],
      ["yaw ego", f(vehicle?.yaw_ego ?? 0, 4)],
      ["LiDAR points", String(vehicle?.num_lidar_pts ?? "—")],
    ];
  } else if (selection.kind === "depth") {
    title = `depth bin ${selection.bin}`;
    label = "CHECKPOINT LATENT DEPTH";
    rows = [["zcam", `${selection.meters} m`], ["α(d)", f(selection.probability, 7)], ["channels", "64 context values"]];
  } else if (selection.kind === "trajectory") {
    title = `trajectory ${selection.index + 1}`;
    label = "TEACHING";
    rows = [["cost", f(selection.cost)], ["Boltzmann p", f(selection.probability, 5)]];
  } else {
    title = selection.kind === "feature" ? `feature [${selection.index.join(",")}]` : "selected ray";
    rows = [["meaning", selection.kind]];
  }
  return (
    <aside className="inspector">
      <button onClick={onClose} aria-label="Close">×</button>
      <span>{label}</span>
      <h2>{title}</h2>
      {rows.map(([key, value]) => <p key={key}><b>{key}</b><em>{value}</em></p>)}
      <small>{locale === "zh-CN" ? "Esc 关闭；所有数值来自当前联动选择。" : "Esc closes; all numbers follow the linked selection."}</small>
    </aside>
  );
}

function BevControls({
  bevMode,
  setBevMode,
  threshold,
  setThreshold,
  bevOpacity,
  setBevOpacity,
  locale,
}: {
  bevMode: BevMode;
  setBevMode: (value: BevMode) => void;
  threshold: number;
  setThreshold: (value: number) => void;
  bevOpacity: number;
  setBevOpacity: (value: number) => void;
  locale: Locale;
}) {
  const labels: Record<BevMode, string> = locale === "zh-CN"
    ? { probability: "概率", threshold: "阈值", gt: "真值", errors: "误差" }
    : { probability: "probability", threshold: "threshold", gt: "GT", errors: "errors" };
  return (
    <div className="bev-controls">
      <div className="view-tabs">
        {(["probability", "threshold", "gt", "errors"] as BevMode[]).map((mode) => (
          <button key={mode} className={bevMode === mode ? "active" : ""} aria-pressed={bevMode === mode} onClick={() => setBevMode(mode)}>{labels[mode]}</button>
        ))}
      </div>
      <label>p ≥ {f(threshold, 2)}<input type="range" min=".05" max=".95" step=".05" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} /></label>
      <label>α {f(bevOpacity, 2)}<input type="range" min=".1" max="1" step=".05" value={bevOpacity} onChange={(event) => setBevOpacity(Number(event.target.value))} /></label>
      <div className="heat-legend"><i /><span>0</span><span>.5</span><span>1 {locale === "zh-CN" ? "概率" : "probability"}</span></div>
    </div>
  );
}

type LabProps = {
  step: number;
  locale: Locale;
  rig: Rig | null;
  alignment: Alignment | null;
  camera: CameraRecord | null;
  selectedCamera: number;
  setSelectedCamera: (value: number) => void;
  pixel: [number, number];
  setPixel: (value: [number, number]) => void;
  anchor: { index: [number, number]; anchor: [number, number]; delta: [number, number] };
  trace: Trace | null;
  depth: number[];
  context: number[];
  contextPlane: number[];
  contextChannel: number;
  setContextChannel: (value: number) => void;
  depthIndex: number;
  setDepthIndex: (value: number) => void;
  depthMode: string;
  setDepthMode: (value: string) => void;
  geometryStage: number;
  setGeometryStage: (value: number) => void;
  lidar: Float32Array | null;
  selectedLidar: number | null;
  onSelect: (value: SceneSelection) => void;
  bevMode: BevMode;
  setBevMode: (value: BevMode) => void;
  threshold: number;
  setThreshold: (value: number) => void;
  bevOpacity: number;
  setBevOpacity: (value: number) => void;
  lidarColor: "height" | "distance" | "intensity";
  setLidarColor: (value: "height" | "distance" | "intensity") => void;
  lidarSize: number;
  setLidarSize: (value: number) => void;
  stats: ReturnType<typeof binaryStats> | null;
  enabled: boolean[];
  setEnabled: (value: boolean[]) => void;
  yaw: number;
  setYaw: (value: number) => void;
  activeVariant: Variant | undefined;
  model: Model | null;
  trajectories: Trajectory[];
  selectedTrajectory: number;
  setSelectedTrajectory: (value: number) => void;
  temperature: number;
  setTemperature: (value: number) => void;
  activeEdge: string;
  currentEdge: string;
  setActiveEdge: (value: string) => void;
  onBack: () => void;
  onComplete: () => void;
  onMinimize: () => void;
};

function ChapterLab(props: LabProps) {
  const { step, locale, rig, alignment, camera } = props;
  const zh = locale === "zh-CN";
  const cue = tx(locale, CHAPTERS[step].cue);
  let body: React.ReactNode;

  if (step === 0) {
    body = (
      <div className="io-overview">
        <div><small>{zh ? "输入 · 仅相机" : "INPUT · CAMERA ONLY"}</small><CameraRig rig={rig} selected={props.selectedCamera} onSelect={props.setSelectedCamera} locale={locale} /></div>
        <em>calibrated<br />Lift–Splat</em>
        <figure><small>{zh ? "输出 · 车辆 BEV LOGITS" : "OUTPUT · VEHICLE BEV LOGITS"}</small><img src={asset(props.activeVariant?.image ?? "/data/model/bev-all-cameras.png")} alt="Checkpoint BEV vehicle output" /><figcaption>1 × 200 × 200 · 0.5 m/cell</figcaption></figure>
      </div>
    );
  } else if (step === 1) {
    const stages = ["images", "CamEncode", "Lift", "geometry", "Splat", "BevEncode", "logits"];
    body = <div className="forward-map">{stages.map((stage, index) => <span key={stage}><b>{String(index + 1).padStart(2, "0")}</b>{stage}{index < stages.length - 1 && <i>→</i>}</span>)}</div>;
  } else if (step === 2) {
    body = (
      <div className="sample-packet">
        <CameraRig rig={rig} selected={props.selectedCamera} onSelect={props.setSelectedCamera} locale={locale} />
        <div className="packet-row">
          <span><b>imgs</b><code>[6,3,128,352]</code></span>
          <span><b>K, R, t</b><code>per camera</code></span>
          <span><b>post A, a</b><code>image geometry</code></span>
          <span><b>binimg</b><code>[1,200,200]</code></span>
          <span className="excluded"><b>LiDAR</b><code>{zh ? "不由加载器返回" : "not returned"}</code></span>
        </div>
      </div>
    );
  } else if (step === 3) {
    body = (
      <div className="calibration-lab">
        <CameraRig rig={rig} selected={props.selectedCamera} onSelect={props.setSelectedCamera} locale={locale} />
        <div className="matrix-pair">{camera && <><Matrix label="K · raw pixels" values={camera.cam2img} /><Matrix label="Tcam→ego · calibrated_sensor" values={camera.cam2ego} /></>}</div>
      </div>
    );
  } else if (step === 4) {
    body = (
      <div className="preprocess-lab">
        <div className="preprocess-images">
          <figure><img src={asset(camera?.image ?? "")} alt="raw" /><figcaption>1600×900 · {zh ? "原图" : "raw"}</figcaption></figure>
          <div><span>{zh ? "缩放" : "resize"} ×{camera?.augmentation.resize ?? 0.22}</span><span>{zh ? "裁剪" : "crop"} y={camera?.augmentation.crop[1] ?? 48}</span><span>{zh ? "RGB 标准化" : "normalize RGB"}</span></div>
          <figure><img src={asset(camera?.network_image ?? "")} alt="network" /><figcaption>352×128 · {zh ? "网络输入" : "network"}</figcaption></figure>
        </div>
        <div className="equation-strip"><span>{zh ? "原图" : "raw"} u</span><b>A u + a</b><span>{zh ? "网络" : "network"} u′</span><em>{zh ? "几何使用 A⁻¹" : "geometry uses A⁻¹"}</em></div>
      </div>
    );
  } else if (step === 5) {
    body = (
      <div className="camencode-lab">
        <div className="encoder-stack"><span>RGB<br /><b>3×128×352</b></span><i>→</i><span>EfficientNet-B0<br /><b>multiscale</b></span><i>→</i><span>Up(320+112)<br /><b>512×8×22</b></span><i>→</i><span>1×1 depthnet<br /><b>105×8×22</b></span></div>
        <div className="feature-row">
          <FeatureHeatmap values={props.contextPlane} selected={props.anchor.index} />
          <label>{zh ? "语义通道" : "context channel"} c={props.contextChannel}<input type="range" min="0" max="63" value={props.contextChannel} onChange={(event) => props.setContextChannel(Number(event.target.value))} /></label>
          <div className="channel-split"><span><b>41</b> {zh ? "深度 logits" : "depth logits"}</span><span><b>64</b> context</span></div>
        </div>
      </div>
    );
  } else if (step === 6) {
    body = (
      <div className="ray-lab">
        <NetworkPixel camera={camera} pixel={props.pixel} setPixel={props.setPixel} anchor={props.anchor} locale={locale} />
        <div className="operation-strip"><span>[u,v,1]<small>{props.anchor.anchor.slice(0, 2).map((value) => f(value, 1)).join(", ")}</small></span><em>K⁻¹</em><span>{zh ? "射线" : "ray"}<small>{zh ? "只有方向" : "direction only"}</small></span><em>× 41 d</em><span>4…44 m<small>{zh ? "投回同一像素" : "same pixel"}</small></span></div>
      </div>
    );
  } else if (step === 7) {
    body = (
      <div className="lift-lab">
        <div className="mode-tabs">{["checkpoint", "one-hot", "uniform", "multi-modal"].map((mode) => <button key={mode} className={props.depthMode === mode ? "active" : ""} aria-pressed={props.depthMode === mode} onClick={() => props.setDepthMode(mode)}>{mode}</button>)}</div>
        <DepthChart values={props.depth} selected={props.depthIndex} onSelect={props.setDepthIndex} />
        <input type="range" min="0" max="40" value={props.depthIndex} onChange={(event) => props.setDepthIndex(Number(event.target.value))} />
        <ContextChart values={props.context} />
        <div className="lift-equation"><span><b>{f(props.depth[props.depthIndex] ?? 0, 6)}</b><small>α({4 + props.depthIndex}m)</small></span><em>×</em><span><b>64D</b><small>context [{props.anchor.index.join(",")}]</small></span><em>=</em><span><b>41×64</b><small>frustum feature</small></span></div>
      </div>
    );
  } else if (step === 8) {
    const stages = ["anchor u′", "undo A,a", "[du,dv,d]", "K⁻¹", "R,+t"];
    const cards = [
      ["network", props.trace?.anchor.slice(0, 2), "px"],
      ["raw image", props.trace?.original.slice(0, 2), "px"],
      ["scaled", props.trace?.scaled, "homogeneous"],
      ["camera", props.trace?.camera, "m · x right, y down, z front"],
      ["ego", props.trace?.ego, "m · x forward, y left, z up"],
    ] as [string, number[] | undefined, string][];
    body = (
      <div className="geometry-lab">
        <div className="geometry-steps">{stages.map((stage, index) => <button key={stage} className={index === props.geometryStage ? "active" : index < props.geometryStage ? "done" : ""} aria-current={index === props.geometryStage ? "step" : undefined} onClick={() => props.setGeometryStage(index)}><i>{index + 1}</i><span>{stage}</span></button>)}</div>
        <div className="geometry-value" key={props.geometryStage}><small>{cards[props.geometryStage][0]}</small><b>[{cards[props.geometryStage][1]?.map((value) => f(value, 4)).join(", ")}]</b><span>{cards[props.geometryStage][2]}</span></div>
        <pre>points = frustum - post_trans{`\n`}points = inverse(post_rots) @ points{`\n`}points = [u*d, v*d, d]{`\n`}points = rots @ inverse(K) @ points + trans</pre>
        <small className="lab-note">{alignment?.geometry_gold.length ?? 0} official golden points · six cameras · first / center / last anchors</small>
      </div>
    );
  } else if (step === 9) {
    body = (
      <div className="multi-camera-lab">
        <CameraRig rig={rig} selected={props.selectedCamera} onSelect={props.setSelectedCamera} locale={locale} />
        <div className="frame-contract"><b>{zh ? "当前相机光轴" : "selected optical axis"}</b><code>{camera?.cam2ego.slice(0, 3).map((row) => f(row[2], 4)).join(", ")}</code><span>{zh ? "Rcam→ego 第三列" : "third column of Rcam→ego"}</span><em>{zh ? "必须朝车外" : "must point outward"}</em></div>
        <div className="symmetry-row"><span>{zh ? "共享编码器" : "shared encoder"}</span><span>{zh ? "逐相机 K,R,t" : "per-camera K,R,t"}</span><span>{zh ? "可交换求和" : "commutative sum"}</span></div>
      </div>
    );
  } else if (step === 10) {
    body = (
      <div className="splat-lab">
        <div className="splat-steps"><span><b>43,296</b><small>frustum points</small></span><em>filter</em><span><b>[ix,iy,iz,b]</b><small>half-open grid</small></span><em>rank</em><span><b>sort</b><small>same voxel adjacent</small></span><em>Σ</em><span><b>QuickCumsum</b><small>exact group sum</small></span></div>
        <div className="cumsum-demo"><span>a</span><span>a+b</span><span>a+b+c</span><b>keep group ends</b><span>a+b</span><span>c</span></div>
        <small className="lab-note">{zh ? "点击右侧 BEV 网格的任一 cell 查看真实 contributor 数量。" : "Click any BEV cell to inspect real contributor counts."}</small>
      </div>
    );
  } else if (step === 11) {
    body = (
      <div className="bevencode-lab">
        <div className="tensor-stages"><span><b>[B,C,Z,X,Y]</b><small>voxel grid</small></span><i>collapse Z</i><span><b>[1,64,200,200]</b><small>BEV features</small></span><i>ResNet-18</i><span><b>[1,1,200,200]</b><small>raw logits</small></span></div>
        <BevControls {...props} />
        <div className="frame-contract"><b>{zh ? "显示坐标契约" : "screen contract"}</b><span>↑ ego +x {zh ? "前" : "forward"}</span><span>← ego +y {zh ? "左" : "left"}</span><code>tensor [x,y]</code></div>
      </div>
    );
  } else if (step === 12) {
    const backwardStages = zh
      ? [["GT 车辆框", "栅格监督"], ["BCEWithLogits", "唯一终端 loss"], ["BevEncode", "空间上下文"], ["QuickCumsum", "组内共享梯度"], ["α(d) × context", "潜在几何"], ["EfficientNet", "原始像素"]]
      : [["GT boxes", "raster target"], ["BCEWithLogits", "one final loss"], ["BevEncode", "dense context"], ["QuickCumsum", "shared gradient"], ["α(d) × context", "latent geometry"], ["EfficientNet", "pixels"]];
    body = (
      <div className="supervision-lab">
        <div className="backprop-flow">
          {backwardStages.map(([name, note], index) => <span key={name} style={{ "--i": index } as React.CSSProperties}><b>{name}</b><small>{note}</small>{index < 5 && <i>←</i>}</span>)}
        </div>
        <div className="loss-contract"><span><b>PAPER</b> object pos_weight = 1.0</span><span><b>CODE</b> train.py default = 2.13</span><span><b>{zh ? "无深度 GT" : "NO DEPTH GT"}</b> {zh ? "梯度来自 BEV" : "gradient arrives from BEV"}</span></div>
      </div>
    );
  } else if (step === 13) {
    body = (
      <div className="inference-lab">
        <div className="inference-path"><span>{zh ? "确定性缩放/裁剪" : "deterministic resize/crop"}</span><i>→</i><span>{zh ? "前向 logits" : "forward logits"}</span><i>→</i><span>{zh ? "sigmoid 概率" : "sigmoid probability"}</span><i>→</i><span>{zh ? "阈值 mask" : "threshold mask"}</span></div>
        <BevControls {...props} />
        {props.stats && <Pills items={[{ name: zh ? "TP 网格" : "TP cells", value: String(props.stats.truePositive) }, { name: "FP", value: String(props.stats.falsePositive) }, { name: "FN", value: String(props.stats.falseNegative) }, { name: zh ? "单帧 IoU" : "single-frame IoU", value: f(props.stats.iou, 3) }]} />}
        <div className="not-in-lss"><b>{zh ? "原始 LSS 分割没有" : "NOT IN ORIGINAL LSS SEGMENTATION"}</b><span>{zh ? "检测框解码" : "box decode"}</span><span>NMS</span><span>{zh ? "跟踪" : "tracking"}</span><span>{zh ? "速度" : "velocity"}</span></div>
      </div>
    );
  } else if (step === 14) {
    body = (
      <div className="truth-lab">
        <div className="truth-toolbar"><div className="view-tabs">{CAMERA_NAMES.map((name, index) => <button key={name} className={props.selectedCamera === index ? "active" : ""} aria-pressed={props.selectedCamera === index} onClick={() => props.setSelectedCamera(index)}>{short(name).replace(" ", "·")}</button>)}</div><span>REFERENCE LIDAR · NOT MODEL INPUT</span></div>
        <ProjectedCamera camera={camera} rig={rig} lidar={props.lidar} selected={props.selectedLidar} onSelect={props.onSelect} />
        <div className="truth-settings"><BevControls {...props} /><label>LiDAR <select value={props.lidarColor} onChange={(event) => props.setLidarColor(event.target.value as "height" | "distance" | "intensity")}><option value="height">{zh ? "高度" : "height"}</option><option value="distance">{zh ? "距离" : "distance"}</option><option value="intensity">{zh ? "强度" : "intensity"}</option></select></label><label>{zh ? "点大小" : "point size"} <input type="range" min="0.05" max="0.3" step="0.01" value={props.lidarSize} onChange={(event) => props.setLidarSize(Number(event.target.value))} /></label></div>
      </div>
    );
  } else if (step === 15) {
    body = (
      <div className="results-lab">
        <PaperTable />
        <div className="robustness-controls">
          <div className="camera-switches">{CAMERA_NAMES.map((name, index) => <button key={name} className={props.enabled[index] ? "" : "off"} onClick={() => { const next = Array(6).fill(true); if (props.enabled[index]) next[index] = false; props.setEnabled(next); props.setYaw(0); }}><i />{short(name)}</button>)}</div>
          <div className="yaw-picker"><span>FRONT yaw</span>{[-3, 0, 3].map((value) => <button key={value} className={props.yaw === value ? "active" : ""} onClick={() => { props.setYaw(value); props.setEnabled(Array(6).fill(true)); }}>{value > 0 ? "+" : ""}{value}°</button>)}</div>
        </div>
        <Pills items={[{ name: zh ? "论文 · 参数量" : "paper · parameters", value: "14.3M" }, { name: "paper · Titan V", value: "35 Hz" }, { name: zh ? "车辆 IoU" : "vehicles IoU", value: "32.07" }, { name: zh ? "oracle 深度" : "oracle depth", value: "44.48" }]} />
      </div>
    );
  } else {
    body = (
      <div className="shoot-lab">
        <div className="trajectory-list">{props.trajectories.map((trajectory, index) => <button key={trajectory.name} className={props.selectedTrajectory === index ? "active" : ""} onMouseEnter={() => props.setSelectedTrajectory(index)} onFocus={() => props.setSelectedTrajectory(index)}><i /><span>{trajectory.name}</span><b>{f(trajectory.cost, 2)}</b><em>{f(trajectory.probability * 100, 1)}%</em></button>)}</div>
        <label>{zh ? "温度" : "temperature"} T={f(props.temperature, 2)}<input type="range" min=".25" max="2" step=".05" value={props.temperature} onChange={(event) => props.setTemperature(Number(event.target.value))} /></label>
        <small className="teaching-label">TEACHING · paper equation · representative subset of 1K templates</small>
        <div className="limit-row"><span>{zh ? "单时间戳" : "single timestamp"}</span><span>{zh ? "远距离" : "far range"}</span><span>{zh ? "夜间 / 眩光" : "night / glare"}</span><span>{zh ? "遮挡" : "occlusion"}</span><span>{zh ? "标定误差" : "calibration"}</span></div>
      </div>
    );
  }

  return (
    <section className="stage-dock" key={step}>
      <header>
        <button className="lab-back" onClick={props.onBack}>← {locale === "zh-CN" ? "返回讲解" : "EXPLANATION"}</button>
        <div><span>{locale === "zh-CN" ? "现在动手" : "TRY IT NOW"}</span><p>{cue}</p></div>
        <button className="card-minimize" onClick={props.onMinimize} aria-label={locale === "zh-CN" ? "收起卡片，专注场景" : "Minimize card and focus the scene"}>— {locale === "zh-CN" ? "沉浸场景" : "FOCUS SCENE"}</button>
      </header>
      <div className="stage-dock-body">
        <TraceRibbon step={step} locale={locale} camera={camera} anchor={props.anchor} trace={props.trace} />
        {[3, 4, 6, 8, 9, 10, 11, 13, 14].includes(step) && <FrameGraph locale={locale} active={props.activeEdge || props.currentEdge} onActive={props.setActiveEdge} />}
        {body}
      </div>
      <footer>
        <span>{locale === "zh-CN" ? "完成本章的交互验证后继续" : "Continue after testing the chapter claim"}</span>
        <button onClick={props.onComplete}>{step === CHAPTERS.length - 1 ? (locale === "zh-CN" ? "回到开场" : "REPLAY COURSE") : (locale === "zh-CN" ? "完成 · 下一章" : "COMPLETE · NEXT CHAPTER")} <ChevronRight /></button>
      </footer>
    </section>
  );
}

export default function LssExplainer() {
  const [step, setStep] = useState(0);
  const [detail, setDetail] = useState(0);
  const [panelMode, setPanelMode] = useState<PanelMode>("lesson");
  const [cardCollapsed, setCardCollapsed] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [locale, setLocale] = useState<Locale>("zh-CN");
  const [rig, setRig] = useState<Rig | null>(null);
  const [features, setFeatures] = useState<Features | null>(null);
  const [model, setModel] = useState<Model | null>(null);
  const [alignment, setAlignment] = useState<Alignment | null>(null);
  const [lidar, setLidar] = useState<Float32Array | null>(null);
  const [selectedCamera, setSelectedCamera] = useState(1);
  const [pixel, setPixel] = useState<[number, number]>([184, 73]);
  const [depthIndex, setDepthIndex] = useState(15);
  const [depthMode, setDepthMode] = useState("checkpoint");
  const [contextChannel, setContextChannel] = useState(12);
  const [geometryStage, setGeometryStage] = useState(0);
  const [selection, setSelection] = useState<SceneSelection | null>(null);
  const [selectedLidar, setSelectedLidar] = useState<number | null>(null);
  const [activeEdge, setActiveEdge] = useState("");
  const [bevMode, setBevMode] = useState<BevMode>("probability");
  const [threshold, setThreshold] = useState(0.5);
  const [bevOpacity, setBevOpacity] = useState(0.85);
  const [lidarColor, setLidarColor] = useState<"height" | "distance" | "intensity">("height");
  const [lidarSize, setLidarSize] = useState(0.12);
  const [enabled, setEnabled] = useState<boolean[]>(Array(6).fill(true));
  const [yaw, setYaw] = useState(0);
  const [temperature, setTemperature] = useState(0.8);
  const [selectedTrajectory, setSelectedTrajectory] = useState(0);

  useEffect(() => {
    const stored = localStorage.getItem("lss-locale");
    if (stored === "en" || stored === "zh-CN") {
      const frame = requestAnimationFrame(() => setLocale(stored));
      return () => cancelAnimationFrame(frame);
    }
  }, []);

  useEffect(() => {
    document.documentElement.lang = locale;
  }, [locale]);

  useEffect(() => {
    Promise.all([
      fetch(asset("/data/rig.json")).then((response) => response.json()),
      fetch(asset("/data/model-features.json")).then((response) => response.json()),
      fetch(asset("/data/model-artifacts.json")).then((response) => response.json()),
      fetch(asset("/data/alignment.json")).then((response) => response.json()),
      fetch(asset("/data/lidar-frame.bin")).then((response) => response.arrayBuffer()),
    ]).then(([rigData, featureData, modelData, alignmentData, lidarData]) => {
      setRig(rigData);
      setFeatures(featureData);
      setModel(modelData);
      setAlignment(alignmentData);
      setLidar(new Float32Array(lidarData));
    }).catch((error) => console.error("Evidence assets failed", error));
  }, []);

  const camera = rig?.cameras[selectedCamera] ?? null;
  const anchor = useMemo<{ index: [number, number]; anchor: [number, number]; delta: [number, number] }>(() => nearestFeatureAnchor(pixel) as { index: [number, number]; anchor: [number, number]; delta: [number, number] }, [pixel]);
  const allDepth = useMemo(() => decodeFloat(features?.depth_probabilities), [features]);
  const allContext = useMemo(() => decodeFloat(features?.context_features), [features]);
  const checkpointDepth = useMemo(() => Array.from({ length: 41 }, (_, d) => allDepth?.[selectedCamera * 41 * 8 * 22 + d * 8 * 22 + anchor.index[0] * 22 + anchor.index[1]] ?? 0), [allDepth, selectedCamera, anchor]);
  const context = useMemo(() => Array.from({ length: 64 }, (_, channel) => allContext?.[selectedCamera * 64 * 8 * 22 + channel * 8 * 22 + anchor.index[0] * 22 + anchor.index[1]] ?? 0), [allContext, selectedCamera, anchor]);
  const contextPlane = useMemo(() => Array.from({ length: 8 * 22 }, (_, index) => allContext?.[selectedCamera * 64 * 8 * 22 + contextChannel * 8 * 22 + index] ?? 0), [allContext, selectedCamera, contextChannel]);
  const depth = useMemo(() => {
    if (depthMode === "checkpoint") return checkpointDepth;
    if (depthMode === "one-hot") return Array.from({ length: 41 }, (_, index) => index === depthIndex ? 1 : 0);
    if (depthMode === "uniform") return Array(41).fill(1 / 41);
    const raw = Array.from({ length: 41 }, (_, index) => Math.exp(-1 * ((index - 11) / 3.2) ** 2) + 0.72 * Math.exp(-1 * ((index - 25) / 4.2) ** 2));
    const sum = raw.reduce((a, b) => a + b, 0);
    return raw.map((value) => value / sum);
  }, [checkpointDepth, depthMode, depthIndex]);

  const trace = useMemo<Trace | null>(() => {
    if (!camera) return null;
    const depthMeters = 4 + depthIndex;
    const clicked = [pixel[0], pixel[1], 1] as Vec3;
    const network = [anchor.anchor[0], anchor.anchor[1], 1] as Vec3;
    const original = undoPostTransform(network, camera.post_rot as Mat3, camera.post_trans as Vec3);
    const scaled = [original[0] * depthMeters, original[1] * depthMeters, depthMeters] as Vec3;
    const cameraPoint = pixelToCamera(original, depthMeters, camera.cam2img as Mat3);
    const rotation = camera.cam2ego.slice(0, 3).map((row) => row.slice(0, 3)) as Mat3;
    const translation = camera.cam2ego.slice(0, 3).map((row) => row[3]) as Vec3;
    return { clicked, anchor: network, original, scaled, camera: cameraPoint, ego: applyRigid(cameraPoint, rotation, translation), depth: depthMeters };
  }, [camera, pixel, anchor, depthIndex]);

  const liftedPoints = useMemo<[number, number, number][]>(() => {
    if (!camera) return [];
    const original = undoPostTransform([anchor.anchor[0], anchor.anchor[1], 1], camera.post_rot as Mat3, camera.post_trans as Vec3);
    const rotation = camera.cam2ego.slice(0, 3).map((row) => row.slice(0, 3)) as Mat3;
    const translation = camera.cam2ego.slice(0, 3).map((row) => row[3]) as Vec3;
    return Array.from({ length: 41 }, (_, index) => {
      const ego = applyRigid(pixelToCamera(original, 4 + index, camera.cam2img as Mat3), rotation, translation);
      return [-ego[1], ego[0], ego[2]] as [number, number, number];
    });
  }, [camera, anchor]);

  const activeDrop = enabled.findIndex((value) => !value);
  const variantKey = activeDrop >= 0 ? `drop-${CAMERA_NAMES[activeDrop].toLowerCase().replaceAll("_", "-")}` : `front-yaw-${yaw >= 0 ? "+" : ""}${yaw}`;
  const activeVariant = model?.variants[variantKey] ?? model?.variants["all-cameras"];
  // The decoded typed array stays stable while unrelated UI state changes.
  const logits = useMemo(() => decodeFloat(activeVariant?.logits), [activeVariant]);
  const probability = useMemo(() => logits ? Float32Array.from(logits, (value) => 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, value))))) : null, [logits]);
  const groundTruth = useMemo(() => decodeUint8(model?.ground_truth.mask), [model]);
  const contributors = useMemo(() => decodeUint16(model?.geometry_contributors.counts), [model]);
  const lidarOccupancy = useMemo(() => decodeUint16(alignment?.lidar_occupancy.counts), [alignment]);
  const stats = useMemo(() => probability && groundTruth ? binaryStats(Array.from(probability), Array.from(groundTruth), threshold) : null, [probability, groundTruth, threshold]);
  const trajectories = useMemo<Trajectory[]>(() => {
    const paths = Array.from({ length: 9 }, (_, index) => Array.from({ length: 21 }, (_, sample) => {
      const y = sample * 0.72;
      return [(index - 4) * 0.017 * y * y, y] as [number, number];
    }));
    const map = ([x, y]: [number, number]) => 0.055 * Math.abs(x) + 0.85 * Math.exp(-((x - 2.3) ** 2 + (y - 9) ** 2) / 5) + 0.5 * Math.exp(-((x + 1.4) ** 2 + (y - 13) ** 2) / 3);
    const costs = paths.map((path) => trajectoryCost(path, map, 0.14));
    const probabilities = boltzmannProbabilities(costs, temperature);
    return costs.map((cost, index) => ({ name: index === costs.indexOf(Math.min(...costs)) ? "minimum cost" : `template ${index + 1}`, points: paths[index], cost, probability: probabilities[index] })).sort((a, b) => a.cost - b.cost);
  }, [temperature]);

  const go = useCallback((value: number) => {
    setStep(Math.max(0, Math.min(CHAPTERS.length - 1, value)));
    setDetail(0);
    setPanelMode("lesson");
    setCardCollapsed(false);
    setPlaying(false);
    setSelection(null);
    setActiveEdge("");
  }, []);
  const advanceLesson = useCallback(() => {
    if (panelMode === "lesson" && detail < 2) setDetail((value) => value + 1);
    else if (panelMode === "lesson") { setPanelMode("lab"); setCardCollapsed(false); }
    else go(step === CHAPTERS.length - 1 ? 0 : step + 1);
  }, [panelMode, detail, step, go]);

  const retreatLesson = useCallback(() => {
    if (panelMode === "lab") setPanelMode("lesson");
    else if (detail > 0) setDetail((value) => value - 1);
    else if (step > 0) go(step - 1);
  }, [panelMode, detail, step, go]);

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(advanceLesson, panelMode === "lab" ? 5200 : 3600);
    return () => window.clearInterval(id);
  }, [playing, panelMode, advanceLesson]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape") { setSelection(null); setActiveEdge(""); }
      if (event.key === "ArrowLeft") retreatLesson();
      if (event.key === "ArrowRight") advanceLesson();
      if (event.code === "Space" && !["INPUT", "BUTTON", "SELECT"].includes((event.target as HTMLElement)?.tagName)) {
        event.preventDefault();
        setPlaying((value) => !value);
      }
    };
    addEventListener("keydown", handler);
    return () => removeEventListener("keydown", handler);
  }, [retreatLesson, advanceLesson]);

  const handleSelect = (value: SceneSelection) => {
    if (value.kind === "cell" && contributors) {
      const flat = value.index[0] * 200 + value.index[1];
      const rows = CAMERA_NAMES.map((name, index) => ({ name, count: contributors[index * 40000 + flat] })).filter((entry) => entry.count > 0);
      value = { ...value, contributors: rows.map((entry) => entry.name), counts: rows.map((entry) => entry.count) };
    }
    if (value.kind === "camera") setSelectedCamera(value.index);
    if (value.kind === "depth") setDepthIndex(value.bin);
    if (value.kind === "lidar") setSelectedLidar(value.index);
    if (value.kind === "trajectory") setSelectedTrajectory(value.index);
    setSelection(value);
  };

  const setLanguage = (next: Locale) => {
    setLocale(next);
    localStorage.setItem("lss-locale", next);
  };
  const currentEdge = ["", "", "", "camera→ego", "raw→network", "", "network→camera", "", geometryStage < 2 ? "raw→network" : geometryStage < 4 ? "network→camera" : "camera→ego", "camera→ego", "ego→BEV", "ego→BEV", "", "ego→BEV", "lidar→ego", "camera→ego", "ego→BEV"][step];
  const labProps: LabProps = { step, locale, rig, alignment, camera, selectedCamera, setSelectedCamera, pixel, setPixel, anchor, trace, depth, context, contextPlane, contextChannel, setContextChannel, depthIndex, setDepthIndex, depthMode, setDepthMode, geometryStage, setGeometryStage, lidar, selectedLidar, onSelect: handleSelect, bevMode, setBevMode, threshold, setThreshold, bevOpacity, setBevOpacity, lidarColor, setLidarColor, lidarSize, setLidarSize, stats, enabled, setEnabled, yaw, setYaw, activeVariant, model, trajectories, selectedTrajectory, setSelectedTrajectory, temperature, setTemperature, activeEdge, currentEdge, setActiveEdge, onBack: () => setPanelMode("lesson"), onComplete: advanceLesson, onMinimize: () => setCardCollapsed(true) };

  return (
    <main className={`lss-app step-${step} act-${CHAPTERS[step].act} phase-${panelMode} ${cardCollapsed ? "card-is-collapsed" : ""}`}>
      <LssScene
        step={step}
        panelMode={panelMode}
        cardVisible={!cardCollapsed}
        enabledCameras={enabled}
        selectedCamera={selectedCamera}
        pixel={pixel}
        depthIndex={depthIndex}
        depthProbability={depth}
        trajectories={trajectories}
        selectedTrajectory={selectedTrajectory}
        cameraMatrices={rig?.cameras.map((entry) => entry.cam2ego)}
        liftedPoints={liftedPoints}
        lidar={lidar}
        lidar2ego={rig?.lidar2ego ?? [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}
        lidarColor={lidarColor}
        lidarPointSize={lidarSize}
        selectedLidar={selectedLidar}
        vehicles={rig?.vehicles_ego ?? []}
        bevProbability={probability}
        groundTruth={groundTruth}
        bevMode={bevMode}
        threshold={threshold}
        bevOpacity={bevOpacity}
        onSelect={handleSelect}
      />
      <header className="site-header">
        <a href={asset("/")}><span>LSS</span> EXPLAINED <sup>v4</sup></a>
        <div>
          <div className="language-switch"><button className={locale === "zh-CN" ? "active" : ""} aria-pressed={locale === "zh-CN"} onClick={() => setLanguage("zh-CN")}>中文</button><button className={locale === "en" ? "active" : ""} aria-pressed={locale === "en"} onClick={() => setLanguage("en")}>EN</button></div>
          <a href={asset("/articles/lift-splat-shoot-explained.zh-CN.md")}>{locale === "zh-CN" ? "完整长文" : "LONG READ"}</a>
          <a href="https://github.com/hova88/lss-explained">SOURCE ↗</a>
        </div>
      </header>
      <CurriculumRail step={step} locale={locale} onGo={go} />
      <div className="scene-caption" aria-hidden="true">
        <b>{String(step + 1).padStart(2, "0")}</b>
        <span>{CHAPTERS[step].stage}</span>
        <small>{tx(locale, actForChapter(step).short)}</small>
      </div>
      <div className={`course-card-layer ${cardCollapsed ? "card-collapsed" : ""}`}>
        {cardCollapsed ? (
          <button className={`scene-card-handle ${panelMode}`} onClick={() => setCardCollapsed(false)}>
            <span>{String(step + 1).padStart(2, "0")} · {panelMode === "lesson" ? (locale === "zh-CN" ? "讲解" : "LEARN") : (locale === "zh-CN" ? "实验" : "LAB")}</span>
            <b>{tx(locale, CHAPTERS[step].title)}</b>
            <i>{locale === "zh-CN" ? "展开卡片" : "OPEN CARD"} ↗</i>
          </button>
        ) : panelMode === "lesson" ? <LessonPanel step={step} locale={locale} detail={detail} onAdvance={advanceLesson} onMinimize={() => setCardCollapsed(true)} /> : <ChapterLab {...labProps} />}
      </div>
      {activeEdge && <MatrixInspector edge={activeEdge} camera={camera} rig={rig} alignment={alignment} locale={locale} onClose={() => setActiveEdge("")} />}
      {selection && <SelectionCard selection={selection} rig={rig} camera={camera} modelProbability={probability} gt={groundTruth} lidarOccupancy={lidarOccupancy} contributors={contributors} locale={locale} onClose={() => setSelection(null)} />}
      <nav className="lesson-nav">
        <button className="round" disabled={step === 0 && detail === 0 && panelMode === "lesson"} onClick={retreatLesson} aria-label="Previous"><ChevronLeft /></button>
        <button className="round play" onClick={() => setPlaying((value) => !value)} aria-label={playing ? "Pause" : "Play"}>{playing ? <Pause /> : <Play />}</button>
        <div className="chapter-scrub">{CHAPTERS.map((chapter, index) => <button key={chapter.title.en} className={index === step ? "active" : index < step ? "past" : ""} aria-label={tx(locale, chapter.title)} aria-current={index === step ? "step" : undefined} onClick={() => go(index)} title={tx(locale, chapter.title)}><i /></button>)}</div>
        <div className="phase-switch" aria-label={locale === "zh-CN" ? "课程模式" : "Course mode"}><button className={panelMode === "lesson" ? "active" : ""} onClick={() => { setPanelMode("lesson"); setCardCollapsed(false); }}>{locale === "zh-CN" ? "理解" : "LEARN"}</button><button className={panelMode === "lab" ? "active" : ""} onClick={() => { setPanelMode("lab"); setCardCollapsed(false); }}>{locale === "zh-CN" ? "验证" : "LAB"}</button></div>
        <button className="next-lesson" onClick={advanceLesson}><small>{panelMode === "lab" ? (locale === "zh-CN" ? "完成本章" : "COMPLETE") : detail < 2 ? (locale === "zh-CN" ? "下一层" : "NEXT LAYER") : (locale === "zh-CN" ? "进入实验" : "ENTER LAB")}</small><b>{panelMode === "lab" ? (CHAPTERS[step + 1] ? tx(locale, CHAPTERS[step + 1].title) : (locale === "zh-CN" ? "回到开场" : "REPLAY")) : detail < 2 ? `${detail + 2}/3` : (locale === "zh-CN" ? "亲手验证本章" : "TEST THE CLAIM")}</b><ChevronRight /></button>
      </nav>
      <div className="interaction-hint">DRAG · PINCH · INSPECT · SPACE AUTOPLAY</div>
    </main>
  );
}
