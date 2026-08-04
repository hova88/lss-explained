#!/usr/bin/env python3
"""Export auditable browser assets from the pinned LSS checkpoint.

The official repository and checkpoint live in .cache/ and are never vendored.
This script imports the pinned model definition, loads the checkpoint strictly,
and writes only derived data under public/data.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import json
import math
import pickle
import shutil
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
SOURCE = CACHE / "openmmlab"
LSS_REPO = CACHE / "lss" / "repo"
CHECKPOINT = CACHE / "lss" / "model525000.pt"
PUBLIC = ROOT / "public" / "data"

LSS_COMMIT = "2903467c91ee9c12f0917a12c22ab1f04e607ae0"
MM3D_COMMIT = "fe25f7a51d36e3702f961e198894580d83c4387b"
SAMPLE_TOKEN = "ca9a282c9e77460f8360f564131a8af5"
LIDAR_FILE = "n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin"
CAMERAS = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
]
GRID = {
    "xbound": [-50.0, 50.0, 0.5],
    "ybound": [-50.0, 50.0, 0.5],
    "zbound": [-10.0, 10.0, 20.0],
    "dbound": [4.0, 45.0, 1.0],
}
DATA_AUG = {
    "resize_lim": [0.193, 0.225], "final_dim": [128, 352],
    "rot_lim": [-5.4, 5.4], "H": 900, "W": 1600,
    "rand_flip": True, "bot_pct_lim": [0.0, 0.22],
    "cams": CAMERAS, "Ncams": 5,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_array(value: np.ndarray, dtype: str = "float16") -> dict:
    array = np.asarray(value).astype(dtype)
    return {
        "shape": list(array.shape),
        "dtype": dtype,
        "encoding": "base64-little-endian",
        "data": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
    }


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def install_minimal_tools_module() -> None:
    """Let the pinned models.py import without pulling the nuScenes SDK stack."""
    package = types.ModuleType("src")
    package.__path__ = [str(LSS_REPO / "src")]
    sys.modules["src"] = package
    tools = types.ModuleType("src.tools")

    def gen_dx_bx(xbound, ybound, zbound):
        bounds = [xbound, ybound, zbound]
        dx = torch.tensor([row[2] for row in bounds], dtype=torch.float32)
        bx = torch.tensor([row[0] + row[2] / 2 for row in bounds], dtype=torch.float32)
        nx = torch.tensor([(row[1] - row[0]) / row[2] for row in bounds], dtype=torch.long)
        return dx, bx, nx

    def cumsum_trick(x, geom_feats, ranks):
        cumulative = x.cumsum(0)
        kept = torch.ones(cumulative.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = ranks[1:] != ranks[:-1]
        cumulative, geom_feats = cumulative[kept], geom_feats[kept]
        cumulative = torch.cat((cumulative[:1], cumulative[1:] - cumulative[:-1]))
        return cumulative, geom_feats

    class QuickCumsum(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, geom_feats, ranks):
            cumulative = x.cumsum(0)
            kept = torch.ones(cumulative.shape[0], device=x.device, dtype=torch.bool)
            kept[:-1] = ranks[1:] != ranks[:-1]
            cumulative, geom_feats = cumulative[kept], geom_feats[kept]
            cumulative = torch.cat((cumulative[:1], cumulative[1:] - cumulative[:-1]))
            ctx.save_for_backward(kept)
            ctx.mark_non_differentiable(geom_feats)
            return cumulative, geom_feats

        @staticmethod
        def backward(ctx, gradx, _gradgeom):
            (kept,) = ctx.saved_tensors
            back = torch.cumsum(kept, 0)
            back[kept] -= 1
            return gradx[back], None, None

    tools.gen_dx_bx = gen_dx_bx
    tools.cumsum_trick = cumsum_trick
    tools.QuickCumsum = QuickCumsum
    sys.modules["src.tools"] = tools


def load_model():
    sys.path.insert(0, str(LSS_REPO))
    install_minimal_tools_module()
    models = importlib.import_module("src.models")
    models.EfficientNet.from_pretrained = classmethod(
        lambda cls, name, *args, **kwargs: cls.from_name(name)
    )
    model = models.compile_model(GRID, DATA_AUG, outC=1)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    model.eval()
    return model


def preprocess_image(path: Path):
    image = Image.open(path).convert("RGB")
    resize = max(128 / 900, 352 / 1600)
    resized = (int(1600 * resize), int(900 * resize))
    crop_h = int((1 - np.mean([0.0, 0.22])) * resized[1]) - 128
    crop_w = int(max(0, resized[0] - 352) / 2)
    crop = (crop_w, crop_h, crop_w + 352, crop_h + 128)
    transformed = image.resize(resized, Image.Resampling.BILINEAR).crop(crop)
    array = np.asarray(transformed, dtype=np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
    post_rot = np.eye(3, dtype=np.float32)
    post_rot[0, 0] = resize
    post_rot[1, 1] = resize
    post_trans = np.array([-crop_w, -crop_h, 0], dtype=np.float32)
    return tensor, transformed, post_rot, post_trans, {
        "resize": resize, "resize_dims": list(resized), "crop": list(crop),
        "flip": False, "rotate_degrees": 0,
    }


def heatmap_rgba(probability: np.ndarray) -> Image.Image:
    p = np.clip(probability, 0, 1)
    red = np.clip((p - 0.18) * 1.7, 0, 1)
    green = np.clip((p - 0.45) * 1.8, 0, 1)
    blue = np.clip(0.72 - p * 0.62, 0.08, 0.72)
    alpha = np.clip((p - 0.08) / 0.62, 0, 0.94)
    rgba = np.stack([red, green, blue, alpha], axis=-1)
    return Image.fromarray(np.uint8(rgba * 255), "RGBA")


def vehicle_gt_mask(sample: dict) -> tuple[np.ndarray, list[dict]]:
    lidar2ego = np.asarray(sample["lidar_points"]["lidar2ego"], dtype=np.float64)
    mask = Image.new("L", (200, 200), 0)
    draw = ImageDraw.Draw(mask)
    vehicles = []
    for instance in sample["instances"]:
        if int(instance["bbox_label_3d"]) > 6 or not bool(instance["bbox_3d_isvalid"]):
            continue
        x, y, z, length, width, height, yaw = map(float, instance["bbox_3d"])
        center = lidar2ego @ np.array([x, y, z, 1.0])
        heading = lidar2ego[:3, :3] @ np.array([math.cos(yaw), math.sin(yaw), 0.0])
        ego_yaw = math.atan2(heading[1], heading[0])
        corners = []
        for lx, ly in [(-length / 2, -width / 2), (length / 2, -width / 2),
                       (length / 2, width / 2), (-length / 2, width / 2)]:
            px = center[0] + lx * math.cos(ego_yaw) - ly * math.sin(ego_yaw)
            py = center[1] + lx * math.sin(ego_yaw) + ly * math.cos(ego_yaw)
            corners.append((round((py + 50) / 0.5), round((px + 50) / 0.5)))
        draw.polygon(corners, fill=255)
        vehicles.append({
            "label": int(instance["bbox_label_3d"]),
            "center_ego": [float(center[0]), float(center[1]), float(center[2])],
            "dimensions": [length, width, height], "yaw_ego": ego_yaw,
            "num_lidar_pts": int(instance["num_lidar_pts"]),
            "num_radar_pts": int(instance["num_radar_pts"]),
        })
    return np.asarray(mask, dtype=np.uint8) / 255, vehicles


def main() -> None:
    PUBLIC.mkdir(parents=True, exist_ok=True)
    (PUBLIC / "images").mkdir(exist_ok=True)
    (PUBLIC / "network-images").mkdir(exist_ok=True)
    (PUBLIC / "model").mkdir(exist_ok=True)
    pkl_path = SOURCE / "n015-2018-07-24-11-22-45+0800.pkl"
    with pkl_path.open("rb") as handle:
        payload = pickle.load(handle)
    sample = payload["data_list"][0]
    if sample["token"] != SAMPLE_TOKEN:
        raise RuntimeError(f"Unexpected sample token {sample['token']}")

    lidar_source = SOURCE / LIDAR_FILE
    if not lidar_source.exists():
        raise RuntimeError(f"Missing pinned LiDAR input: {lidar_source}")
    lidar_output = PUBLIC / "lidar-frame.bin"
    shutil.copyfile(lidar_source, lidar_output)
    lidar_points = np.fromfile(lidar_source, dtype="<f4").reshape(-1, 5)
    if lidar_points.shape != (34688, 5) or not np.isfinite(lidar_points).all():
        raise RuntimeError(f"Unexpected LiDAR tensor: {lidar_points.shape}")

    tensors, rotations, translations, intrinsics, post_rots, post_trans = [], [], [], [], [], []
    camera_contract = []
    source_hashes = {
        "demo_pickle": sha256(pkl_path), "checkpoint": sha256(CHECKPOINT),
        "lidar_frame": sha256(lidar_source),
    }
    for camera in CAMERAS:
        record = sample["images"][camera]
        source_path = SOURCE / Path(record["img_path"]).name
        output_name = camera.lower().replace("_", "-") + ".jpg"
        output_path = PUBLIC / "images" / output_name
        shutil.copyfile(source_path, output_path)
        tensor, network_image, post_rot, post_tran, augmentation = preprocess_image(source_path)
        network_name = camera.lower().replace("_", "-") + ".jpg"
        network_image.save(PUBLIC / "network-images" / network_name, quality=91)
        cam2ego = np.asarray(record["cam2ego"], dtype=np.float32)
        tensors.append(tensor)
        rotations.append(torch.from_numpy(cam2ego[:3, :3]))
        translations.append(torch.from_numpy(cam2ego[:3, 3]))
        intrinsics.append(torch.tensor(record["cam2img"], dtype=torch.float32))
        post_rots.append(torch.from_numpy(post_rot))
        post_trans.append(torch.from_numpy(post_tran))
        source_hashes[camera] = sha256(source_path)
        camera_contract.append({
            "name": camera, "image": f"/data/images/{output_name}",
            "network_image": f"/data/network-images/{network_name}",
            "image_sha256": source_hashes[camera], "cam2img": record["cam2img"],
            "cam2ego": record["cam2ego"], "lidar2cam": record["lidar2cam"],
            "sample_data_token": record["sample_data_token"], "timestamp": record["timestamp"],
            "augmentation": augmentation, "post_rot": post_rot.tolist(),
            "post_trans": post_tran.tolist(),
            "matrix_metadata": {
                "cam2img": {"from_frame": "camera", "to_frame": "original_image", "source": "OpenMMLab pinned demo"},
                "cam2ego": {"from_frame": camera, "to_frame": "ego", "source": "nuScenes calibrated_sensor via OpenMMLab"},
                "lidar2cam": {"from_frame": "LIDAR_TOP", "to_frame": camera, "source": "OpenMMLab pose-compensated conversion"},
                "post_transform": {"from_frame": "original_image", "to_frame": "network_image", "source": "LSS validation preprocessing"},
            },
        })

    images = torch.stack(tensors).unsqueeze(0)
    rots = torch.stack(rotations).unsqueeze(0)
    trans = torch.stack(translations).unsqueeze(0)
    intrins = torch.stack(intrinsics).unsqueeze(0)
    post_rots_tensor = torch.stack(post_rots).unsqueeze(0)
    post_trans_tensor = torch.stack(post_trans).unsqueeze(0)

    model = load_model()
    with torch.inference_mode():
        flat = images.view(-1, 3, 128, 352)
        efficient = model.camencode.get_eff_depth(flat)
        raw = model.camencode.depthnet(efficient)
        depth = raw[:, :model.D].softmax(dim=1)
        context = raw[:, model.D:model.D + model.camC]
        lifted = depth.unsqueeze(1) * context.unsqueeze(2)
        camera_features = lifted.view(1, 6, 64, 41, 8, 22).permute(0, 1, 3, 4, 5, 2)
        geometry = model.get_geometry(rots, trans, intrins, post_rots_tensor, post_trans_tensor)
        geometry_numpy = geometry[0].detach().cpu().numpy()
        contributor_counts = np.zeros((6, 200, 200), dtype=np.uint16)
        for camera_index in range(6):
            points = geometry_numpy[camera_index].reshape(-1, 3)
            indices = np.floor((points[:, :2] + 50.0) / 0.5).astype(np.int64)
            kept = (
                (indices[:, 0] >= 0) & (indices[:, 0] < 200) &
                (indices[:, 1] >= 0) & (indices[:, 1] < 200) &
                (points[:, 2] >= -10.0) & (points[:, 2] < 10.0)
            )
            np.add.at(contributor_counts[camera_index], (indices[kept, 0], indices[kept, 1]), 1)

        variants = {}

        def infer_variant(name, variant_geometry, variant_features):
            bev_features = model.voxel_pooling(variant_geometry, variant_features)
            logits = model.bevencode(bev_features)[0, 0].detach().cpu().numpy()
            if not np.isfinite(logits).all() or logits.shape != (200, 200):
                raise RuntimeError(f"Invalid output for {name}: {logits.shape}")
            probability = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
            image_name = f"bev-{name}.png"
            heatmap_rgba(probability).save(PUBLIC / "model" / image_name)
            variants[name] = {
                "evidence": "official-checkpoint-derived", "shape": [1, 1, 200, 200],
                "logits": encode_array(logits), "probability_min": float(probability.min()),
                "probability_max": float(probability.max()), "probability_mean": float(probability.mean()),
                "image": f"/data/model/{image_name}",
                "image_sha256": sha256(PUBLIC / "model" / image_name),
            }
            return probability

        all_probability = infer_variant("all-cameras", geometry, camera_features)
        for camera_index, camera in enumerate(CAMERAS):
            kept = [index for index in range(6) if index != camera_index]
            infer_variant(
                "drop-" + camera.lower().replace("_", "-"),
                geometry[:, kept], camera_features[:, kept],
            )
        front_index = CAMERAS.index("CAM_FRONT")
        for degrees in (-3, 0, 3):
            perturbed_rots = rots.clone()
            radians = math.radians(degrees)
            yaw_matrix = torch.tensor([
                [math.cos(radians), -math.sin(radians), 0.0],
                [math.sin(radians), math.cos(radians), 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=torch.float32)
            perturbed_rots[0, front_index] = yaw_matrix @ perturbed_rots[0, front_index]
            perturbed_geometry = model.get_geometry(
                perturbed_rots, trans, intrins, post_rots_tensor, post_trans_tensor
            )
            infer_variant(f"front-yaw-{degrees:+d}", perturbed_geometry, camera_features)

    gt_mask, vehicles = vehicle_gt_mask(sample)
    gt_image = Image.fromarray(np.uint8(gt_mask * 255), "L")
    gt_image.save(PUBLIC / "model" / "vehicle-gt.png")

    lidar2ego = np.asarray(sample["lidar_points"]["lidar2ego"], dtype=np.float64)
    lidar_h = np.concatenate([lidar_points[:, :3].astype(np.float64), np.ones((len(lidar_points), 1))], axis=1)
    lidar_ego = (lidar2ego @ lidar_h.T).T[:, :3]
    lidar_occupancy = np.zeros((200, 200), dtype=np.uint16)
    lidar_indices = np.floor((lidar_ego[:, :2] + 50.0) / 0.5).astype(np.int64)
    lidar_kept = (
        (lidar_indices[:, 0] >= 0) & (lidar_indices[:, 0] < 200) &
        (lidar_indices[:, 1] >= 0) & (lidar_indices[:, 1] < 200)
    )
    np.add.at(lidar_occupancy, (lidar_indices[lidar_kept, 0], lidar_indices[lidar_kept, 1]), 1)

    projection_contract = []
    for camera in camera_contract:
        lidar2cam = np.asarray(camera["lidar2cam"], dtype=np.float64)
        camera_points = (lidar2cam @ lidar_h.T).T[:, :3]
        projected = (np.asarray(camera["cam2img"], dtype=np.float64) @ camera_points.T).T
        uv = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-12)
        visible = (
            (camera_points[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < 1600) &
            (uv[:, 1] >= 0) & (uv[:, 1] < 900)
        )
        static_chain = np.linalg.inv(np.asarray(camera["cam2ego"], dtype=np.float64)) @ lidar2ego
        rotation = np.asarray(camera["cam2ego"], dtype=np.float64)[:3, :3]
        projection_contract.append({
            "camera": camera["name"], "visible_points": int(visible.sum()),
            "camera_timestamp": camera["timestamp"],
            "delta_to_lidar_ms": float((camera["timestamp"] - sample["timestamp"]) * 1000),
            "direct_vs_static_chain_max_abs": float(np.max(np.abs(lidar2cam - static_chain))),
            "rotation_det": float(np.linalg.det(rotation)),
            "rotation_orthogonality_max_error": float(np.max(np.abs(rotation.T @ rotation - np.eye(3)))),
        })

    def single_frame_stats(threshold):
        predicted = all_probability >= threshold
        truth = gt_mask > 0
        tp = int(np.logical_and(predicted, truth).sum())
        fp = int(np.logical_and(predicted, ~truth).sum())
        fn = int(np.logical_and(~predicted, truth).sum())
        tn = int(np.logical_and(~predicted, ~truth).sum())
        union = tp + fp + fn
        return {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn, "iou": tp / union if union else 1.0}

    geometry_gold = []
    for camera_index in range(6):
        for depth_index, row, column in [(0, 0, 0), (20, 4, 11), (40, 7, 21)]:
            geometry_gold.append({
                "camera": CAMERAS[camera_index], "index": [depth_index, row, column],
                "processed_anchor": [float(model.frustum[depth_index, row, column, 0]),
                                     float(model.frustum[depth_index, row, column, 1])],
                "depth": float(model.frustum[depth_index, row, column, 2]),
                "ego": geometry_numpy[camera_index, depth_index, row, column].astype(float).tolist(),
            })

    alignment_contract = {
        "schema_version": "2.0.0", "sample_token": SAMPLE_TOKEN,
        "evidence": "pinned-lidar-and-calibration-derived",
        "lidar": {
            "path": "/data/lidar-frame.bin", "sha256": source_hashes["lidar_frame"],
            "shape": [int(lidar_points.shape[0]), 5], "dtype": "float32-little-endian",
            "fields": ["x", "y", "z", "intensity", "ring"], "frame": "LIDAR_TOP",
            "xyz_min": lidar_points[:, :3].min(axis=0).astype(float).tolist(),
            "xyz_max": lidar_points[:, :3].max(axis=0).astype(float).tolist(),
        },
        "camera_projections": projection_contract,
        "lidar_occupancy": {"axis_order": ["ego_x_cell", "ego_y_cell"], "counts": encode_array(lidar_occupancy, "uint16")},
        "single_frame_diagnostics": [single_frame_stats(value) for value in (0.3, 0.5, 0.7)],
        "geometry_gold": geometry_gold,
        "warning": "LiDAR is a reference visualization and is not an input to LSS camera inference.",
    }
    alignment_path = PUBLIC / "alignment.json"
    alignment_path.write_text(json.dumps(alignment_contract, separators=(",", ":")), encoding="utf-8")

    features_contract = {
        "schema_version": "2.0.0", "sample_token": SAMPLE_TOKEN,
        "evidence": "official-checkpoint-derived", "checkpoint_sha256": source_hashes["checkpoint"],
        "depth_probabilities": encode_array(depth.detach().cpu().numpy().reshape(6, 41, 8, 22)),
        "context_features": encode_array(context.detach().cpu().numpy().reshape(6, 64, 8, 22)),
        "probe": {
            "feature_cell": [4, 11],
            "depth_by_camera": depth[:, :, 4, 11].detach().cpu().numpy().tolist(),
            "context_by_camera": context[:, :, 4, 11].detach().cpu().numpy().tolist(),
        },
        "feature_anchors": {
            "x": np.linspace(0, 351, 22, dtype=np.float32).tolist(),
            "y": np.linspace(0, 127, 8, dtype=np.float32).tolist(),
        },
    }
    (PUBLIC / "model-features.json").write_text(json.dumps(features_contract, separators=(",", ":")), encoding="utf-8")

    model_contract = {
        "schema_version": "2.0.0", "sample_token": SAMPLE_TOKEN,
        "source_commits": {"lss": LSS_COMMIT, "mmdetection3d": MM3D_COMMIT},
        "source_hashes": source_hashes, "evidence": "official-checkpoint-derived",
        "shapes": {
            "images": [1, 6, 3, 128, 352], "depth_context": [1, 6, 41, 8, 22, 64],
            "bev_features": [1, 64, 200, 200], "vehicle_logits": [1, 1, 200, 200],
        },
        "variants": variants,
        "ground_truth": {
            "evidence": "nuscenes-annotation-derived", "shape": [1, 200, 200],
            "mask": encode_array(gt_mask, "uint8"), "image": "/data/model/vehicle-gt.png",
            "image_sha256": sha256(PUBLIC / "model" / "vehicle-gt.png"),
        },
        "tensor_checks": {
            "finite": True, "depth_probability_sum_max_error": float(
                np.abs(depth.detach().cpu().numpy().sum(axis=1) - 1).max()
            ),
        },
        "geometry_contributors": {
            "evidence": "calibration-derived-frustum-samples",
            "axis_order": ["camera", "ego_x_cell", "ego_y_cell"],
            "counts": encode_array(contributor_counts, "uint16"),
        },
    }
    model_path = PUBLIC / "model-artifacts.json"
    model_path.write_text(json.dumps(model_contract, separators=(",", ":")), encoding="utf-8")

    rig_contract = {
        "schema_version": "2.0.0", "sample_token": SAMPLE_TOKEN,
        "dataset": payload["metainfo"], "timestamp": sample["timestamp"],
        "source_commits": {"lss": LSS_COMMIT, "mmdetection3d": MM3D_COMMIT},
        "source_hashes": source_hashes, "grid": GRID,
        "camera_order": CAMERAS, "cameras": camera_contract,
        "lidar2ego": sample["lidar_points"]["lidar2ego"],
        "ego2global": sample["ego2global"], "vehicles_ego": vehicles,
        "frames": {
            "camera": {"axes": "+x right, +y down, +z optical forward", "unit": "meter"},
            "lidar": {"axes": "native LIDAR_TOP sensor frame", "unit": "meter"},
            "ego": {"axes": "+x vehicle forward, +y left, +z up", "unit": "meter", "origin": "rear axle midpoint"},
            "global": {"axes": "nuScenes map frame, z up", "unit": "meter"},
            "network_image": {"axes": "+u right, +v down", "unit": "pixel", "shape": [128, 352]},
            "bev": {"axes": "tensor [ego_x_cell, ego_y_cell]", "unit": "0.5 meter cell", "shape": [200, 200]},
        },
        "transform_convention": {
            "vectors": "column", "active_transform": "p_to = T_from_to @ p_from",
            "screen": "up = ego +x; left = ego +y",
        },
    }
    (PUBLIC / "rig.json").write_text(json.dumps(json_safe(rig_contract), indent=2), encoding="utf-8")
    manifest = {
        "model_artifacts_sha256": sha256(model_path),
        "model_features_sha256": sha256(PUBLIC / "model-features.json"),
        "rig_sha256": sha256(PUBLIC / "rig.json"),
        "alignment_sha256": sha256(alignment_path),
        "lidar_sha256": source_hashes["lidar_frame"],
        "variant_hashes": {name: value["image_sha256"] for name, value in variants.items()},
    }
    (PUBLIC / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "variants": len(variants), **manifest}, indent=2))


if __name__ == "__main__":
    main()
