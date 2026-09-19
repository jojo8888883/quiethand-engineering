#!/usr/bin/env python3
"""Run HaWoR on a frozen QuietHand split using each event's real camera model."""

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
M3_V1_2 = HERE.parent / "m3_v1_2"
if str(M3_V1_2) not in sys.path:
    sys.path.insert(0, str(M3_V1_2))
M3_JOBS = HERE.parent / "m3_jobs"
if str(M3_JOBS) not in sys.path:
    sys.path.insert(0, str(M3_JOBS))

from run_raw_hand import (  # noqa: E402
    CHECKPOINT_REVISION,
    REPOSITORY_BINDING_SHA256,
    REPOSITORY_REVISION,
    _artifact,
    _boxes,
    _load_model,
    _verify_mano,
)
from runtime_common import (  # noqa: E402
    M3JobError,
    atomic_json,
    atomic_npy,
    load_exact_json,
    sha256_file,
    status,
    verify_bound_files,
    verify_repository_binding,
    verify_resource_manifest,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256", type=str, required=True)
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


def _validated_camera(event: dict[str, object]) -> tuple[float, list[float], dict[str, object]]:
    camera = event.get("camera_model")
    if not isinstance(camera, dict):
        raise M3JobError("metric hand event lacks camera_model")
    intrinsic = np.asarray(camera.get("intrinsic_px"), dtype=np.float64)
    focal = camera.get("hawor_focal_px")
    center = camera.get("hawor_principal_point_px")
    if (
        intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
        or isinstance(focal, bool)
        or not isinstance(focal, (int, float))
        or not np.isfinite(focal)
        or focal <= 0.0
        or not isinstance(center, list)
        or len(center) != 2
        or not all(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) for value in center)
        or not np.isclose(float(focal), (intrinsic[0, 0] + intrinsic[1, 1]) / 2.0, rtol=0.0, atol=1e-6)
        or not np.allclose(center, [intrinsic[0, 2], intrinsic[1, 2]], rtol=0.0, atol=1e-6)
    ):
        raise M3JobError("metric hand camera_model is invalid")
    return float(focal), [float(center[0]), float(center[1])], camera


def _vertices(model, paths: list[str], boxes: np.ndarray, side: str, focal: float, center: list[float]) -> np.ndarray:
    from hawor.utils.process import run_mano, run_mano_left
    from hawor.utils.rotation import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis

    do_flip = side == "left"
    prediction = model.inference(
        np.asarray(paths),
        boxes,
        img_focal=focal,
        img_center=center,
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


def main() -> int:
    args = arguments()
    progress = args.output / "status.json"
    started = time.monotonic()
    try:
        manifest = load_exact_json(args.input, args.input_sha256)
        events = manifest.get("events")
        expected_count = manifest.get("event_count")
        is_evaluation = manifest.get("schema") == "quiethand.m3_5.evaluation_raw_hand_metric_input.v1"
        calibration_envelope = (
            manifest.get("schema") == "quiethand.m3_5.raw_hand_metric_input.v1"
            and expected_count == 30
            and manifest.get("calibration_video_count") == 30
            and manifest.get("evaluation_event_count") == 0
            and manifest.get("evaluation_model_results_opened") is False
        )
        evaluation_envelope = (
            is_evaluation
            and expected_count == 30
            and manifest.get("evaluation_event_count") == 30
            and manifest.get("evaluation_model_results_authorized") is True
            and manifest.get("evaluation_results_opened") is True
            and manifest.get("training_performed") is False
        )
        if manifest.get("adapter") != "raw_hand_metric" or not (calibration_envelope or evaluation_envelope) or not isinstance(events, list) or len(events) != expected_count:
            raise M3JobError("metric raw-hand input is not a frozen 30-event calibration or authorized evaluation set")
        camera_models = [_validated_camera(event)[2] for event in events]
        if args.event_index is not None:
            if not 0 <= args.event_index < expected_count:
                raise M3JobError("event index is outside the frozen input")
            events = [events[args.event_index]]
        total = len(events)
        run_mode = "positive_control" if args.event_index is not None else ("full_evaluation" if is_evaluation else "full_calibration")
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
            focal, center, camera = _validated_camera(event)
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
                    vertices = _vertices(model, paths, boxes, side, focal, center)
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
                {
                    "schema": "quiethand.m3_5.raw_hand_metric_result.v1",
                    "event_id": event_id,
                    "adapter": "raw_hand_metric",
                    "camera_model": camera,
                    "items": items,
                },
            )
            status(progress, "running", ordinal, total)
        focal_values = [float(camera["hawor_focal_px"]) for camera in camera_models]
        anisotropy_values = [float(camera["focal_anisotropy_relative"]) for camera in camera_models]
        audit = {
            "schema": "quiethand.m3_5.raw_hand_metric_job_audit.v1",
            "status": "COMPLETE",
            "adapter": "raw_hand_metric",
            "event_count": total,
            "input_event_count": expected_count,
            "evaluation_event_count": expected_count if is_evaluation else 0,
            "run_mode": run_mode,
            "event_index": args.event_index,
            "item_counts": counts,
            "input_sha256": args.input_sha256,
            "repository_revision": REPOSITORY_REVISION,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "detector_threshold": 0.5,
            "camera_model_source": "per-event TACO egocentric intrinsic",
            "hawor_focal_policy": "arithmetic mean of fx and fy",
            "hawor_focal_px_range": [min(focal_values), max(focal_values)],
            "focal_anisotropy_relative_max": max(anisotropy_values),
            "principal_point_policy": "per-event cx and cy",
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
            "evaluation_results_opened": is_evaluation,
        }
        atomic_json(args.output / "audit.json", audit)
        status(progress, "complete", total, total)
    except Exception as exc:
        status(progress, "failed", 0, 1 if args.event_index is not None else 30, f"{type(exc).__name__}: {exc}")
        print(f"[hold] HOLD_ADAPTER_INTEGRATION: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
