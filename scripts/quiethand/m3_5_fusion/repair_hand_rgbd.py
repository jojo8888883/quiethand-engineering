#!/usr/bin/env python3
"""Repair one event's HaWoR hand root translations from SAM2 masks and RGB-D.

The method never reads native hand/object trajectories.  It preserves every
per-frame MANO mesh up to one shared 3-D translation and writes a new result.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts/quiethand/m3_jobs"))

from runtime_common import atomic_json, atomic_npy, status  # noqa: E402


SIDES = ("left", "right")


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def project(points: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    if not np.isfinite(points).all() or np.any(points[:, 2] <= 0.001):
        raise ValueError("hand vertices must be finite and in front of camera")
    uvz = points @ np.asarray(intrinsic, dtype=np.float64).T
    return uvz[:, :2] / uvz[:, 2:]


def expanded_box(uv: np.ndarray, width: int, height: int, fraction: float, minimum_px: int) -> np.ndarray:
    low = np.min(uv, axis=0)
    high = np.max(uv, axis=0)
    margin = np.maximum((high - low) * fraction, float(minimum_px))
    box = np.concatenate((low - margin, high + margin))
    box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        raise ValueError("projected hand box is empty")
    return box.astype(np.float32)


def component_at(mask: np.ndarray, point_xy: np.ndarray) -> np.ndarray:
    """Keep the 4-connected component containing the positive SAM prompt."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must be 2-D")
    x = int(np.clip(round(float(point_xy[0])), 0, mask.shape[1] - 1))
    y = int(np.clip(round(float(point_xy[1])), 0, mask.shape[0] - 1))
    if not mask[y, x]:
        raise ValueError("SAM mask excluded its positive hand prompt")
    result = np.zeros_like(mask)
    result[y, x] = True
    queue: deque[tuple[int, int]] = deque([(y, x)])
    while queue:
        row, col = queue.popleft()
        for rr, cc in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not result[rr, cc]:
                result[rr, cc] = True
                queue.append((rr, cc))
    return result


def backproject_mask(depth: np.ndarray, mask: np.ndarray, intrinsic: np.ndarray, max_points: int) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if depth.shape != mask.shape:
        raise ValueError("depth and hand mask shapes differ")
    valid = mask & np.isfinite(depth) & (depth > 0.001) & (depth < 10.0)
    rows, cols = np.nonzero(valid)
    if len(rows) < 100:
        raise ValueError("hand mask has fewer than 100 valid RGB-D pixels")
    if len(rows) > max_points:
        take = np.linspace(0, len(rows) - 1, max_points, dtype=np.int64)
        rows, cols = rows[take], cols[take]
    z = depth[rows, cols]
    inverse = np.linalg.inv(np.asarray(intrinsic, dtype=np.float64))
    rays = np.column_stack((cols, rows, np.ones_like(cols))) @ inverse.T
    points = rays * z[:, None]
    if not np.isfinite(points).all():
        raise ValueError("backprojected hand point cloud is non-finite")
    return points


def nearest_vertices(points: np.ndarray, vertices: np.ndarray, chunk_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic nearest-neighbour query without an optional scipy dependency."""
    indices = np.empty(len(points), dtype=np.int64)
    distances = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), chunk_size):
        stop = min(start + chunk_size, len(points))
        delta = points[start:stop, None, :] - vertices[None, :, :]
        squared = np.einsum("ijk,ijk->ij", delta, delta)
        local = np.argmin(squared, axis=1)
        indices[start:stop] = local
        distances[start:stop] = np.sqrt(squared[np.arange(stop - start), local])
    return distances, indices


def translation_icp(
    vertices: np.ndarray,
    observed: np.ndarray,
    *,
    iterations: int,
    trim_fraction: float,
    stop_delta_m: float,
    max_translation_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Align an unchanged hand vertex cloud to a partial observed RGB-D surface."""
    vertices = np.asarray(vertices, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or observed.ndim != 2 or observed.shape[1] != 3:
        raise ValueError("ICP inputs must be 3-D point clouds")
    if not 0.5 <= trim_fraction <= 1.0 or iterations < 1:
        raise ValueError("invalid ICP rule")
    translation = np.median(observed, axis=0) - np.median(vertices, axis=0)
    history = []
    for _ in range(iterations):
        shifted = vertices + translation
        distances, indices = nearest_vertices(observed, shifted)
        cutoff = float(np.quantile(distances, trim_fraction))
        keep = distances <= cutoff
        if int(keep.sum()) < 100:
            raise ValueError("translation ICP has fewer than 100 retained observations")
        delta = np.median(observed[keep] - shifted[indices[keep]], axis=0)
        translation = translation + delta
        history.append({
            "retained_points": int(keep.sum()),
            "trimmed_median_residual_m": float(np.median(distances[keep])),
            "translation_update_m": delta.tolist(),
        })
        if float(np.linalg.norm(delta)) <= stop_delta_m:
            break
    final_distances, _ = nearest_vertices(observed, vertices + translation)
    if not np.isfinite(translation).all() or float(np.linalg.norm(translation)) > max_translation_m:
        raise ValueError("RGB-D hand translation exceeds the frozen bound")
    return translation, {
        "initial_translation_m": (np.median(observed, axis=0) - np.median(vertices, axis=0)).tolist(),
        "final_translation_m": translation.tolist(),
        "final_translation_norm_m": float(np.linalg.norm(translation)),
        "final_median_observed_to_vertex_m": float(np.median(final_distances)),
        "iterations_run": len(history),
        "history": history,
    }


def relative_span(hand_vertices: np.ndarray, object_poses: np.ndarray) -> float:
    centers = np.mean(np.asarray(hand_vertices, dtype=np.float64), axis=1)
    poses = np.asarray(object_poses, dtype=np.float64)
    if centers.shape != (15, 3) or poses.shape != (15, 4, 4):
        raise ValueError("relative span requires the same 15 frames")
    local = np.einsum("tji,tj->ti", poses[:, :3, :3], centers - poses[:, :3, 3])
    return float(np.linalg.norm(np.quantile(local, 0.95, axis=0) - np.quantile(local, 0.05, axis=0)))


def load_obj_vertices(path: Path) -> np.ndarray:
    values = []
    for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
        if line.startswith("v "):
            fields = line.split()
            values.append([float(fields[1]), float(fields[2]), float(fields[3])])
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or len(result) < 4 or not np.isfinite(result).all():
        raise ValueError(f"invalid OBJ vertices: {path}")
    return result


def draw_points(image: Image.Image, points: np.ndarray, intrinsic: np.ndarray, color: str, radius: int = 2) -> None:
    draw = ImageDraw.Draw(image)
    uv = project(points, intrinsic) / 3.0
    for x, y in uv:
        if 0 <= x < image.width and 0 <= y < image.height:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def render_qa(
    output: Path,
    side: str,
    rgb_paths: list[Path],
    masks: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    intrinsic: np.ndarray,
    object_vertices: np.ndarray,
    object_poses: np.ndarray,
    selected: tuple[int, ...],
) -> str:
    canvas = Image.new("RGB", (1920, 382 * len(selected)), "white")
    painter = ImageDraw.Draw(canvas)
    for row, frame in enumerate(selected):
        top = 382 * row
        source = Image.open(rgb_paths[frame]).convert("RGB").resize((640, 360))
        mask = Image.fromarray((masks[frame].astype(np.uint8) * 255)).resize((640, 360), Image.Resampling.NEAREST)
        mask_array = np.asarray(mask, dtype=bool)
        mask_panel = np.asarray(source).copy()
        mask_panel[mask_array] = (0.55 * mask_panel[mask_array] + 0.45 * np.array([255, 0, 210])).astype(np.uint8)
        before_panel = Image.fromarray(mask_panel)
        after_panel = source.copy()
        draw_points(before_panel, before[frame, ::5], intrinsic, "#00e5ff")
        draw_points(after_panel, after[frame, ::5], intrinsic, "#7dff65")
        object_cloud = object_vertices @ object_poses[frame, :3, :3].T + object_poses[frame, :3, 3]
        draw_points(after_panel, object_cloud[::max(1, len(object_cloud) // 700)], intrinsic, "#ff9a2f", 1)
        canvas.paste(source, (0, top + 22))
        canvas.paste(before_panel, (640, top + 22))
        canvas.paste(after_panel, (1280, top + 22))
        painter.text((8, top + 4), f"source | sample {frame}", fill="black")
        painter.text((648, top + 4), "SAM hand mask + original HaWoR", fill="black")
        painter.text((1288, top + 4), "RGB-D translated hand + current object", fill="black")
    name = f"{side}_mask_repair_qa.jpg"
    canvas.save(output / name, quality=92)
    return name


def load_sam(repository: Path, config: str, checkpoint: Path):
    import torch

    sys.path.insert(0, str(repository))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(config, str(checkpoint), device="cuda").eval()
    return SAM2ImagePredictor(model), torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = read(args.input)
    if cfg.get("evaluation_results_opened") is not False or cfg.get("event_count") != 1:
        raise ValueError("repair input must be exactly one calibration event")
    event_id = cfg["event_id"]
    frames = cfg["source_frame_indices"]
    if len(frames) != 15 or frames != sorted(frames) or len(set(frames)) != 15:
        raise ValueError("repair input must retain the frozen 15 source frames")
    identity_input = read(args.workspace / cfg["identity_batch_input"])
    matches = [item for item in identity_input["events"] if item["event_id"] == event_id]
    if len(matches) != 1:
        raise ValueError("event is absent or duplicate in identity input")
    event = matches[0]
    if event["source_frame_indices"] != frames:
        raise ValueError("source frame binding changed")
    rgb_paths = [args.workspace / path for path in event["segmentation"]["ordered_rgb_frames"]]
    depth_paths = [args.workspace / path for path in event["object_state"]["ordered_depth_frames"]]
    intrinsic = np.load(args.workspace / event["object_state"]["intrinsic_path"], allow_pickle=False).astype(np.float64)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("camera intrinsic is invalid")
    record_path = args.workspace / cfg["entity_record"]
    record = read(record_path)
    if record["event_id"] != event_id or record["source_frame_indices"] != frames:
        raise ValueError("entity record binding changed")
    before = {
        side: np.load(args.workspace / record["input_provenance"][f"{side}_hand"], allow_pickle=False).astype(np.float64)
        for side in SIDES
    }
    for side, value in before.items():
        if value.shape != (15, 778, 3) or not np.isfinite(value).all():
            raise ValueError(f"{side} HaWoR hand array is invalid")

    algorithm = cfg["algorithm"]
    predictor, torch = load_sam(
        args.workspace / cfg["sam_repository"],
        cfg["sam_config"],
        args.workspace / cfg["sam_checkpoint"],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "masks").mkdir(exist_ok=True)
    (args.output / "hands").mkdir(exist_ok=True)
    all_masks = {side: [] for side in SIDES}
    repaired = {side: [] for side in SIDES}
    details = {side: [] for side in SIDES}
    started = time.monotonic()
    with torch.inference_mode():
        for frame_index, (rgb_path, depth_path) in enumerate(zip(rgb_paths, depth_paths, strict=True)):
            status(args.output / "status.json", "sam_rgbd_hand_repair", frame_index, 15)
            rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
            depth = np.load(depth_path, allow_pickle=False)
            if rgb.shape[:2] != depth.shape:
                raise ValueError("RGB and depth frame shapes differ")
            height, width = depth.shape
            predictor.set_image(rgb)
            projected = {side: project(before[side][frame_index], intrinsic) for side in SIDES}
            centers = {side: np.mean(projected[side], axis=0) for side in SIDES}
            for side in SIDES:
                box = expanded_box(
                    projected[side], width, height,
                    float(algorithm["box_margin_fraction"]), int(algorithm["box_minimum_margin_px"]),
                )
                points = [centers[side]]
                labels = [1]
                other = "right" if side == "left" else "left"
                ox, oy = centers[other]
                if box[0] <= ox <= box[2] and box[1] <= oy <= box[3]:
                    points.append(centers[other])
                    labels.append(0)
                masks, scores, _ = predictor.predict(
                    point_coords=np.asarray(points, dtype=np.float32),
                    point_labels=np.asarray(labels, dtype=np.int32),
                    box=box,
                    multimask_output=True,
                )
                order = np.argsort(-np.asarray(scores), kind="stable")
                selected_mask = None
                selected_score = None
                for candidate in order:
                    raw = np.asarray(masks[int(candidate)], dtype=bool)
                    px = int(np.clip(round(float(centers[side][0])), 0, width - 1))
                    py = int(np.clip(round(float(centers[side][1])), 0, height - 1))
                    if raw[py, px]:
                        selected_mask = raw
                        selected_score = float(scores[int(candidate)])
                        break
                if selected_mask is None:
                    raise ValueError(f"SAM returned no {side} candidate containing the positive prompt")
                clipped = np.zeros_like(selected_mask)
                x1, y1, x2, y2 = box
                clipped[int(np.floor(y1)):int(np.ceil(y2)) + 1, int(np.floor(x1)):int(np.ceil(x2)) + 1] = True
                selected_mask = component_at(selected_mask & clipped, centers[side])
                cloud = backproject_mask(depth, selected_mask, intrinsic, int(algorithm["max_depth_points"]))
                translation, trace = translation_icp(
                    before[side][frame_index], cloud,
                    iterations=int(algorithm["translation_icp_iterations"]),
                    trim_fraction=float(algorithm["translation_icp_trim_fraction"]),
                    stop_delta_m=float(algorithm["translation_icp_stop_delta_m"]),
                    max_translation_m=float(algorithm["max_translation_m"]),
                )
                all_masks[side].append(selected_mask)
                repaired[side].append(before[side][frame_index] + translation)
                details[side].append({
                    "sample_index": frame_index,
                    "source_frame_index": frames[frame_index],
                    "prompt_box_xyxy_px": box.tolist(),
                    "positive_prompt_xy_px": centers[side].tolist(),
                    "negative_other_hand_prompt_used": len(points) == 2,
                    "sam_predicted_iou": selected_score,
                    "mask_pixel_count": int(selected_mask.sum()),
                    "valid_depth_point_count": int(len(cloud)),
                    **trace,
                })

    masks_array = {side: np.stack(all_masks[side]).astype(bool) for side in SIDES}
    repaired_array = {side: np.stack(repaired[side]).astype(np.float32) for side in SIDES}
    for side in SIDES:
        atomic_npy(args.output / "masks" / f"{side}.npy", masks_array[side])
        atomic_npy(args.output / "hands" / f"{side}.npy", repaired_array[side])

    entity_catalog = {item["entity_id"]: item for item in event["object_state"]["entities"]}
    pairs = []
    qa_images = {}
    threshold = float(cfg["stable_span_threshold_m"])
    for pair in cfg["pairs"]:
        side, entity_id = pair["hand"], pair["entity_id"]
        object_poses = np.load(args.workspace / record["entities"][entity_id]["pose_path"], allow_pickle=False).astype(np.float64)
        before_span = relative_span(before[side], object_poses)
        after_span = relative_span(repaired_array[side], object_poses)
        pairs.append({
            **pair,
            "before_relative_span_m": before_span,
            "after_relative_span_m": after_span,
            "before_stable": before_span <= threshold,
            "after_stable": after_span <= threshold,
        })
        qa_images[side] = render_qa(
            args.output, side, rgb_paths, masks_array[side], before[side], repaired_array[side], intrinsic,
            load_obj_vertices(args.workspace / entity_catalog[entity_id]["mesh_path"]), object_poses,
            tuple(cfg["qa_sample_indices"]),
        )

    passed = all(item["after_stable"] for item in pairs)
    result = {
        "schema": "quiethand.m3_5.rgbd_hand_translation_repair.v1",
        "status": "PASS_PREDICTION_SIDE_HAND_REPAIR" if passed else "FAIL_PREDICTION_SIDE_HAND_REPAIR",
        "event_id": event_id,
        "source_frame_indices": frames,
        "method": "SAM2.1 per-frame hand mask plus trimmed translation-only RGB-D point-set ICP",
        "algorithm": algorithm,
        "native_hand_or_object_used": False,
        "human_correction_used_as_model_input": False,
        "articulation_or_rotation_changed": False,
        "evaluation_results_opened": False,
        "training_performed": False,
        "stable_span_threshold_m": threshold,
        "pairs": pairs,
        "hands": {
            side: {
                "mask_path": f"masks/{side}.npy",
                "hand_path": f"hands/{side}.npy",
                "qa_image": qa_images[side],
                "frames": details[side],
            }
            for side in SIDES
        },
        "elapsed_seconds": time.monotonic() - started,
        "next": "post_hoc_native_QA_only; never tune this frozen method from native results",
    }
    atomic_json(args.output / "repair.json", result)
    status(args.output / "status.json", "repair_complete", 15, 15)
    print(json.dumps({"status": result["status"], "pairs": pairs}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
