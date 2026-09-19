#!/usr/bin/env python3
"""Frozen SAM2 + FoundationPose calibration-only adapter chain."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_common import M3JobError, atomic_json, atomic_npy, load_exact_json, sha256_file, status, verify_bound_files, verify_repository_binding, verify_resource_manifest  # noqa: E402
from quiethand.m3_adapter_contract import M3AdapterContractError, validate_semantic_candidate  # noqa: E402


INPUT_SHA256 = "1bbaf13d319a8df6fb971353a08b54ebd8c8d635bcab9009b871c65c5f1a0ccc"
SAM_REPOSITORY_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM_CHECKPOINT_REVISION = "b7320756a13354e7530a63935656d35b2f91a290"
FOUNDATION_REPOSITORY_REVISION = "a1b694b83e633c2cb6115b9063d940a687759392"
REPOSITORY_BINDING_SHA256 = "98208c39d4110b1c3b0c3a188796dc81e1d33f31b8dd93d62a4793b085968766"
HEIGHT = 1080
WIDTH = 1920


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--semantic-output", type=Path, required=True)
    parser.add_argument("--sam-repository", type=Path, required=True)
    parser.add_argument("--sam-config", type=str, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--foundation-repository", type=Path, required=True)
    parser.add_argument("--repository-parent", type=Path, required=True)
    parser.add_argument("--repository-binding", type=Path, required=True)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--resource-manifest", type=Path, required=True)
    parser.add_argument("--resource-manifest-sha256", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-index", type=int)
    return parser.parse_args()


def _semantic_candidates(root: Path) -> dict[str, dict[str, object] | None]:
    audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "COMPLETE" or audit.get("adapter") != "semantic" or audit.get("event_count") != 150 or audit.get("evaluation_results_opened") is not False:
        raise M3JobError("semantic dependency is not a completed calibration job")
    candidates = {}
    for path in sorted((root / "events").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        event_id = value.get("event_id")
        if not isinstance(event_id, str) or event_id in candidates:
            raise M3JobError("semantic event identity is invalid or duplicate")
        if value.get("status") == "observed":
            candidate = value.get("candidate")
            if not isinstance(candidate, dict):
                raise M3JobError("observed semantic result lacks candidate")
            validate_semantic_candidate(json.dumps(candidate, sort_keys=True, allow_nan=False))
            candidates[event_id] = candidate
        elif value.get("status") == "semantic_invalid":
            candidates[event_id] = None
        else:
            raise M3JobError("semantic event has an undeclared state")
    if len(candidates) != 150:
        raise M3JobError("semantic dependency coverage is not 150")
    return candidates


def _load_sam(repository: Path, config: str, checkpoint: Path):
    sys.path.insert(0, str(repository))
    from sam2.build_sam import build_sam2_video_predictor

    return build_sam2_video_predictor(config, str(checkpoint), device="cuda", vos_optimized=False).eval()


def _load_foundation(repository: Path):
    os.chdir(repository)
    sys.path.insert(0, str(repository))
    import nvdiffrast.torch as dr
    from estimater import FoundationPose
    from learning.training.predict_pose_refine import PoseRefinePredictor
    from learning.training.predict_score import ScorePredictor

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    context = dr.RasterizeCudaContext()
    return FoundationPose, scorer, refiner, context


def _verify_foundation_weight_links(repository: Path, resource_root: Path) -> None:
    for run_name in ("2023-10-28-18-33-37", "2024-01-11-20-02-45"):
        for name in ("config.yml", "model_best.pth"):
            source = resource_root / "object_state" / run_name / name
            target = repository / "weights" / run_name / name
            if source.is_symlink() or not source.is_file():
                raise M3JobError("FoundationPose verified source file is missing")
            if target.is_symlink() or not target.is_file() or not os.path.samefile(source, target):
                raise M3JobError("FoundationPose installed weight is not the verified hard link")


def _box_pixels(box: list[int] | None) -> np.ndarray | None:
    if box is None:
        return None
    x1, y1, x2, y2 = box
    result = np.asarray([x1 * WIDTH / 1000.0, y1 * HEIGHT / 1000.0, x2 * WIDTH / 1000.0, y2 * HEIGHT / 1000.0], dtype=np.float32)
    if not (0 <= result[0] < result[2] <= WIDTH and 0 <= result[1] < result[3] <= HEIGHT):
        raise M3JobError("semantic box does not map inside the image")
    return result


def _temporary_jpeg_aliases(paths: list[Path]):
    directory = Path(
        tempfile.mkdtemp(prefix=".qh-m3-sam-", dir=paths[0].parent)
    )
    try:
        for index, path in enumerate(paths):
            # SAM2's loader filters by suffix, while PIL detects content by magic.
            # Hard-linking preserves the exact PNG bytes and avoids lossy re-encoding.
            os.link(path, directory / f"{index:05d}.jpg")
        yield directory
    finally:
        shutil.rmtree(directory)


def _segment(predictor, paths: list[Path], boxes: dict[str, np.ndarray | None]) -> dict[str, np.ndarray | None]:
    # Local generator form keeps the temporary aliases alive for the full propagation.
    alias_generator = _temporary_jpeg_aliases(paths)
    directory = next(alias_generator)
    state = None
    try:
        state = predictor.init_state(video_path=str(directory), offload_video_to_cpu=True, offload_state_to_cpu=False)
        role_ids = {"tool": 1, "target": 2}
        active = []
        for role, object_id in role_ids.items():
            if boxes[role] is None:
                continue
            predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=object_id, box=boxes[role])
            active.append(role)
        masks: dict[str, list[np.ndarray]] = {role: [] for role in active}
        if active:
            for frame_index, object_ids, logits in predictor.propagate_in_video(state):
                if frame_index != len(next(iter(masks.values()), [])):
                    raise M3JobError("SAM2 propagation frame order changed")
                mapping = {int(object_id): (logits[index, 0] > 0.0).detach().cpu().numpy().astype(bool) for index, object_id in enumerate(object_ids)}
                for role in active:
                    mask = mapping.get(role_ids[role])
                    if mask is None or mask.shape != (HEIGHT, WIDTH):
                        raise M3JobError("SAM2 mask coverage or shape changed")
                    masks[role].append(mask)
        result = {}
        for role in role_ids:
            if role not in masks:
                result[role] = None
            else:
                array = np.stack(masks[role]).astype(bool)
                if array.shape != (15, HEIGHT, WIDTH):
                    raise M3JobError("SAM2 did not return exactly 15 masks")
                result[role] = array
        return result
    finally:
        if state is not None:
            predictor.reset_state(state)
        try:
            next(alias_generator)
        except StopIteration:
            pass


def _valid_pose(pose: np.ndarray) -> bool:
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return False
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-4, rtol=0):
        return False
    rotation = pose[:3, :3]
    return bool(np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3, rtol=0) and abs(np.linalg.det(rotation) - 1.0) <= 2e-3)


def _poses(FoundationPose, scorer, refiner, context, mesh_path: Path, rgbs: list[np.ndarray], depths: list[np.ndarray], intrinsic: np.ndarray, masks: np.ndarray) -> np.ndarray:
    import trimesh

    if masks[0].sum() < 4 or ((depths[0] >= 0.001) & masks[0]).sum() < 4:
        raise M3JobError("first-frame mask has fewer than four valid depth pixels")
    mesh = trimesh.load(mesh_path, process=False)
    estimator = FoundationPose(
        model_pts=mesh.vertices.copy(),
        model_normals=mesh.vertex_normals.copy(),
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir="/tmp/quiethand-foundationpose-debug",
        debug=0,
        glctx=context,
    )
    values = [estimator.register(K=intrinsic, rgb=rgbs[0], depth=depths[0], ob_mask=masks[0], iteration=5)]
    for rgb, depth in zip(rgbs[1:], depths[1:], strict=True):
        values.append(estimator.track_one(rgb=rgb, depth=depth, K=intrinsic, iteration=2))
    poses = np.stack(values).astype(np.float32)
    if poses.shape != (15, 4, 4) or not all(_valid_pose(pose.astype(np.float64)) for pose in poses):
        raise M3JobError("FoundationPose output is non-finite or not SE(3)")
    return poses


def _artifact(output_root: Path, path: Path, *, dtype: str, shape: list[int], unit: str, frame_id: str) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(output_root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "dtype": dtype,
        "shape": shape,
        "frame_indices": list(range(15)),
        "unit": unit,
        "frame_id": frame_id,
    }


def main() -> int:
    args = arguments()
    progress = args.output / "status.json"
    started = time.monotonic()
    try:
        manifest = load_exact_json(args.input, INPUT_SHA256)
        events = manifest.get("events")
        if (
            manifest.get("adapter") != "segmentation_object_state"
            or manifest.get("event_count") != 150
            or manifest.get("evaluation_event_count") != 0
            or manifest.get("evaluation_model_results_opened") is not False
            or not isinstance(events, list)
            or len(events) != 150
        ):
            raise M3JobError("segmentation/object input is not calibration-only")
        if args.event_index is not None:
            if not 0 <= args.event_index < 150:
                raise M3JobError("positive-control event index is outside calibration")
            events = [events[args.event_index]]
        total = len(events)
        run_mode = "positive_control" if args.event_index is not None else "full_calibration"
        verified_files = verify_bound_files(args.workspace, events)
        verified_resource_bytes = verify_resource_manifest(args.resource_manifest, args.resource_manifest_sha256, args.resource_root)
        verified_sam_files = verify_repository_binding(
            args.repository_binding,
            REPOSITORY_BINDING_SHA256,
            args.repository_parent,
            args.resource_root,
            "segmentation",
        )
        verified_foundation_files = verify_repository_binding(
            args.repository_binding,
            REPOSITORY_BINDING_SHA256,
            args.repository_parent,
            args.resource_root,
            "object_state",
        )
        if args.sam_repository.is_symlink() or not args.sam_repository.samefile(args.repository_parent / "sam2"):
            raise M3JobError("SAM2 repository path is not the verified source tree")
        if args.foundation_repository.is_symlink() or not args.foundation_repository.samefile(args.repository_parent / "FoundationPose"):
            raise M3JobError("FoundationPose repository path is not the verified source tree")
        if not args.sam_checkpoint.samefile(args.resource_root / "segmentation" / "sam2.1_hiera_base_plus.pt"):
            raise M3JobError("SAM2 checkpoint path is not the verified resource")
        _verify_foundation_weight_links(args.foundation_repository, args.resource_root)
        semantic = _semantic_candidates(args.semantic_output)
        status(progress, "loading_models", 0, total)
        sam = _load_sam(args.sam_repository, args.sam_config, args.sam_checkpoint)
        FoundationPose, scorer, refiner, context = _load_foundation(args.foundation_repository)
        results_dir = args.output / "events"
        results_dir.mkdir(parents=True, exist_ok=True)
        counts = {"segmentation_observed": 0, "segmentation_abstain": 0, "segmentation_invalid": 0, "object_observed": 0, "object_abstain": 0, "object_invalid": 0}
        for ordinal, event in enumerate(events, start=1):
            event_id = event["event_id"]
            candidate = semantic[event_id]
            boxes = {"tool": None, "target": None}
            if candidate is not None:
                boxes = {role: _box_pixels(candidate["first_frame_boxes_0_1000"][role]) for role in boxes}
            rgb_paths = [(args.workspace / relative).resolve() for relative in event["segmentation"]["ordered_rgb_frames"]]
            event_dir = results_dir / event_id
            event_dir.mkdir(exist_ok=True)
            segmentation_items = {}
            try:
                masks = _segment(sam, rgb_paths, boxes) if candidate is not None else {"tool": None, "target": None}
                segment_failure = None
            except Exception as exc:
                masks = {"tool": None, "target": None}
                segment_failure = f"{type(exc).__name__}: {exc}"
            for role in ("tool", "target"):
                if segment_failure is not None:
                    segmentation_items[role] = {"status": "invalid", "artifact": None, "failure_reason": segment_failure}
                    counts["segmentation_invalid"] += 1
                elif boxes[role] is None:
                    reason = "semantic_invalid" if candidate is None else "semantic_box_null"
                    segmentation_items[role] = {"status": "abstain", "artifact": None, "failure_reason": reason}
                    counts["segmentation_abstain"] += 1
                else:
                    mask_path = event_dir / f"{role}_mask.npy"
                    atomic_npy(mask_path, masks[role])
                    segmentation_items[role] = {"status": "observed", "artifact": _artifact(args.output, mask_path, dtype="bool", shape=[15, HEIGHT, WIDTH], unit="binary", frame_id="image"), "failure_reason": None}
                    counts["segmentation_observed"] += 1
            atomic_json(event_dir / "segmentation.json", {"schema": "quiethand.m3.adapter_result.v1", "event_id": event_id, "adapter": "segmentation", "items": segmentation_items})

            rgbs = [np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8) for path in rgb_paths]
            depths = [np.load(args.workspace / relative, allow_pickle=False).astype(np.float32, copy=False) for relative in event["object_state"]["ordered_depth_frames"]]
            intrinsic = np.load(args.workspace / event["object_state"]["intrinsic_path"], allow_pickle=False).astype(np.float64, copy=False)
            object_items = {}
            for role in ("tool", "target"):
                segment_item = segmentation_items[role]
                if segment_item["status"] != "observed":
                    object_items[role] = {"status": "abstain", "artifact": None, "failure_reason": f"segmentation_{segment_item['status']}"}
                    counts["object_abstain"] += 1
                    continue
                try:
                    mesh_path = (args.workspace / event["object_state"][f"{role}_mesh_path"]).resolve()
                    poses = _poses(FoundationPose, scorer, refiner, context, mesh_path, rgbs, depths, intrinsic, masks[role])
                    pose_path = event_dir / f"{role}_pose.npy"
                    atomic_npy(pose_path, poses)
                    object_items[role] = {"status": "observed", "artifact": _artifact(args.output, pose_path, dtype="float32", shape=[15, 4, 4], unit="m_SE3", frame_id="camera"), "failure_reason": None}
                    counts["object_observed"] += 1
                except Exception as exc:
                    object_items[role] = {"status": "invalid", "artifact": None, "failure_reason": f"{type(exc).__name__}: {exc}"}
                    counts["object_invalid"] += 1
            atomic_json(event_dir / "object_state.json", {"schema": "quiethand.m3.adapter_result.v1", "event_id": event_id, "adapter": "object_state", "items": object_items})
            status(progress, "running", ordinal, total)
        audit = {
            "schema": "quiethand.m3.adapter_job_audit.v1",
            "status": "COMPLETE",
            "adapter": "segmentation_object_state",
            "event_count": total,
            "input_event_count": 150,
            "run_mode": run_mode,
            "event_index": args.event_index,
            "item_counts": counts,
            "input_sha256": INPUT_SHA256,
            "semantic_audit_sha256": sha256_file(args.semantic_output / "audit.json"),
            "sam_repository_revision": SAM_REPOSITORY_REVISION,
            "sam_checkpoint_revision": SAM_CHECKPOINT_REVISION,
            "foundation_repository_revision": FOUNDATION_REPOSITORY_REVISION,
            "sam_prompt_rule": "first-frame semantic boxes only; exact PNG bytes exposed through extension aliases",
            "foundation_register_iterations": 5,
            "foundation_track_iterations": 2,
            "verified_input_file_count": verified_files,
            "resource_manifest_sha256": args.resource_manifest_sha256,
            "verified_resource_bytes": verified_resource_bytes,
            "repository_binding_sha256": REPOSITORY_BINDING_SHA256,
            "verified_sam_repository_file_count": verified_sam_files,
            "verified_foundation_repository_file_count": verified_foundation_files,
            "runner_sha256": sha256_file(Path(__file__)),
            "elapsed_seconds": time.monotonic() - started,
            "runtime": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
            "native_masks_used": False,
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
