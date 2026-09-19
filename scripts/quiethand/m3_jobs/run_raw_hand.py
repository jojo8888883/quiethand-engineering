#!/usr/bin/env python3
"""Frozen HaWoR raw observed camera-space calibration adapter."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from ultralytics import YOLO


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_common import M3JobError, atomic_json, atomic_npy, load_exact_json, sha256_file, status, verify_bound_files, verify_repository_binding, verify_resource_manifest  # noqa: E402


INPUT_SHA256 = "5ade716e9a7b0b400db72abdc7d16bed9e3476d7f09023bd287581237ed47703"
REPOSITORY_REVISION = "66c7d4108d58a716deccd192cb7645170cdc7bd7"
CHECKPOINT_REVISION = "da6335f47f9806308992d5ae1002a4cc5f7252c2"
REPOSITORY_BINDING_SHA256 = "98208c39d4110b1c3b0c3a188796dc81e1d33f31b8dd93d62a4793b085968766"
MANO_BINDING = {
    "left": (3821391, "c4022f7083f2ca7c78b2b3d595abbab52debd32b09d372b16923a801f0ea6a30"),
    "right": (3821356, "45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767"),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--repository-parent", type=Path, required=True)
    parser.add_argument("--repository-binding", type=Path, required=True)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--resource-manifest", type=Path, required=True)
    parser.add_argument("--resource-manifest-sha256", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--mano-left", type=Path, required=True)
    parser.add_argument("--mano-right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-index", type=int)
    return parser.parse_args()


def _load_model(repository: Path, checkpoint: Path):
    os.chdir(repository)
    sys.path.insert(0, str(repository))
    from hawor.configs import get_config
    from lib.models.hawor import HAWOR

    model_cfg = get_config(str(checkpoint.parent.parent / "model_config.yaml"), update_cachedir=True)
    if model_cfg.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in model_cfg.MODEL:
        model_cfg.defrost()
        if model_cfg.MODEL.IMAGE_SIZE != 256:
            raise M3JobError("HaWoR ViT image size changed")
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()
    model = HAWOR.load_from_checkpoint(str(checkpoint), strict=False, cfg=model_cfg)
    return model.cuda().eval()


def _verify_mano(repository: Path, left: Path, right: Path) -> dict[str, str]:
    sources = {"left": left, "right": right}
    installed = {
        "left": repository / "_DATA" / "data_left" / "mano_left" / "MANO_LEFT.pkl",
        "right": repository / "_DATA" / "data" / "mano" / "MANO_RIGHT.pkl",
    }
    result = {}
    for side in ("left", "right"):
        source = sources[side]
        target = installed[side]
        expected_size, expected_sha = MANO_BINDING[side]
        if source.is_symlink() or not source.is_file() or source.stat().st_size != expected_size or sha256_file(source) != expected_sha:
            raise M3JobError(f"licensed MANO {side} source changed")
        if target.is_symlink() or not target.is_file() or not os.path.samefile(source, target):
            raise M3JobError(f"HaWoR MANO {side} installation is not the verified hard link")
        result[side] = expected_sha
    return result


def _boxes(detector: YOLO, paths: list[str]) -> dict[str, np.ndarray | None]:
    predictions = detector.predict(source=paths, conf=0.5, verbose=False, device=0)
    result: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    for prediction in predictions:
        for side, class_id in (("left", 0), ("right", 1)):
            boxes = prediction.boxes
            if boxes is None or len(boxes) == 0:
                result[side].append(None)
                continue
            classes = boxes.cls.detach().cpu().numpy().astype(np.int64)
            confidences = boxes.conf.detach().cpu().numpy()
            candidates = np.flatnonzero(classes == class_id)
            if candidates.size == 0:
                result[side].append(None)
                continue
            chosen = int(candidates[np.argmax(confidences[candidates])])
            xyxy = boxes.xyxy[chosen].detach().cpu().numpy().astype(np.float32)
            result[side].append(np.concatenate([xyxy, [confidences[chosen]]]).astype(np.float32))
    return {
        side: None if any(box is None for box in boxes) else np.stack(boxes).astype(np.float32)
        for side, boxes in result.items()
    }


def _vertices(model, paths: list[str], boxes: np.ndarray, side: str) -> np.ndarray:
    from hawor.utils.process import run_mano, run_mano_left
    from hawor.utils.rotation import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis

    do_flip = side == "left"
    prediction = model.inference(
        np.asarray(paths),
        boxes,
        img_focal=600.0,
        img_center=[960.0, 540.0],
        do_flip=do_flip,
    )
    root = prediction["pred_rotmat"][None, :, 0]
    hand_pose = prediction["pred_rotmat"][None, :, 1:]
    translation = prediction["pred_trans"][None, :, 0]
    betas = prediction["pred_shape"][None, :]
    root_aa = rotation_matrix_to_angle_axis(root)
    pose_aa = rotation_matrix_to_angle_axis(hand_pose)
    if do_flip:
        root_aa[..., 1] *= -1
        root_aa[..., 2] *= -1
        pose_aa[..., 1] *= -1
        pose_aa[..., 2] *= -1
        output = run_mano_left(translation, root_aa, pose_aa, betas=betas)
    else:
        output = run_mano(translation, root_aa, pose_aa, betas=betas)
    vertices = output["vertices"][0].detach().cpu().numpy().astype(np.float32)
    if vertices.shape != (15, 778, 3) or not np.isfinite(vertices).all():
        raise M3JobError("HaWoR vertex output is invalid")
    return vertices


def _artifact(output_root: Path, path: Path) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(output_root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "dtype": "float32",
        "shape": [15, 778, 3],
        "frame_indices": list(range(15)),
        "unit": "m",
        "frame_id": "camera",
    }


def main() -> int:
    args = arguments()
    progress = args.output / "status.json"
    started = time.monotonic()
    try:
        manifest = load_exact_json(args.input, INPUT_SHA256)
        events = manifest.get("events")
        if (
            manifest.get("adapter") != "raw_hand"
            or manifest.get("event_count") != 150
            or manifest.get("evaluation_event_count") != 0
            or manifest.get("evaluation_model_results_opened") is not False
            or not isinstance(events, list)
            or len(events) != 150
        ):
            raise M3JobError("raw-hand input envelope is not calibration-only")
        if args.event_index is not None:
            if not 0 <= args.event_index < 150:
                raise M3JobError("positive-control event index is outside calibration")
            events = [events[args.event_index]]
        total = len(events)
        run_mode = "positive_control" if args.event_index is not None else "full_calibration"
        verified_files = verify_bound_files(args.workspace, events)
        verified_resource_bytes = verify_resource_manifest(args.resource_manifest, args.resource_manifest_sha256, args.resource_root)
        verified_repository_files = verify_repository_binding(
            args.repository_binding,
            REPOSITORY_BINDING_SHA256,
            args.repository_parent,
            args.resource_root,
            "raw_hand",
        )
        if args.repository.is_symlink() or not args.repository.samefile(args.repository_parent / "HaWoR"):
            raise M3JobError("HaWoR repository path is not the verified source tree")
        if not args.checkpoint.samefile(args.resource_root / "raw_hand" / "hawor" / "checkpoints" / "hawor.ckpt"):
            raise M3JobError("HaWoR checkpoint path is not the verified resource")
        if not args.detector.samefile(args.resource_root / "raw_hand" / "external" / "detector.pt"):
            raise M3JobError("HaWoR detector path is not the verified resource")
        mano_sha256 = _verify_mano(args.repository, args.mano_left, args.mano_right)
        status(progress, "loading_models", 0, total)
        model = _load_model(args.repository, args.checkpoint)
        detector = YOLO(str(args.detector))
        counts = {"observed": 0, "abstain": 0, "invalid": 0}
        results_dir = args.output / "events"
        results_dir.mkdir(parents=True, exist_ok=True)
        for ordinal, event in enumerate(events, start=1):
            event_id = event["event_id"]
            paths = [str((args.workspace / relative).resolve()) for relative in event["ordered_rgb_frames"]]
            detections = _boxes(detector, paths)
            items = {}
            for side in ("left", "right"):
                boxes = detections[side]
                if boxes is None:
                    items[side] = {"status": "abstain", "artifact": None, "failure_reason": "detector_missing_one_or_more_observed_frames"}
                    counts["abstain"] += 1
                    continue
                try:
                    vertices = _vertices(model, paths, boxes, side)
                    event_dir = results_dir / event_id
                    event_dir.mkdir(exist_ok=True)
                    path = event_dir / f"{side}.npy"
                    atomic_npy(path, vertices)
                    items[side] = {"status": "observed", "artifact": _artifact(args.output, path), "failure_reason": None}
                    counts["observed"] += 1
                except Exception as exc:
                    items[side] = {"status": "invalid", "artifact": None, "failure_reason": f"{type(exc).__name__}: {exc}"}
                    counts["invalid"] += 1
            atomic_json(
                results_dir / f"{event_id}.json",
                {"schema": "quiethand.m3.adapter_result.v1", "event_id": event_id, "adapter": "raw_hand", "items": items},
            )
            status(progress, "running", ordinal, total)
        audit = {
            "schema": "quiethand.m3.adapter_job_audit.v1",
            "status": "COMPLETE",
            "adapter": "raw_hand",
            "event_count": total,
            "input_event_count": 150,
            "run_mode": run_mode,
            "event_index": args.event_index,
            "item_counts": counts,
            "input_sha256": INPUT_SHA256,
            "repository_revision": REPOSITORY_REVISION,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "detector_threshold": 0.5,
            "focal_length_px": 600.0,
            "missing_frame_policy": "whole-side abstain; no interpolation or infill",
            "verified_input_file_count": verified_files,
            "resource_manifest_sha256": args.resource_manifest_sha256,
            "verified_resource_bytes": verified_resource_bytes,
            "repository_binding_sha256": REPOSITORY_BINDING_SHA256,
            "verified_repository_file_count": verified_repository_files,
            "mano_sha256": mano_sha256,
            "runner_sha256": sha256_file(Path(__file__)),
            "elapsed_seconds": time.monotonic() - started,
            "runtime": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
            "slam_performed": False,
            "infiller_performed": False,
            "training_performed": False,
            "evaluation_results_opened": False,
        }
        atomic_json(args.output / "audit.json", audit)
        status(progress, "complete", total, total)
    except Exception as exc:
        status(progress, "failed", 0, 1 if args.event_index is not None else 150, f"{type(exc).__name__}: {exc}")
        print(f"[hold] HOLD_ADAPTER_INTEGRATION: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
