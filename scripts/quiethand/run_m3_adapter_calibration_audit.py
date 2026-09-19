#!/usr/bin/env python3
"""Deterministically audit the complete calibration-only M3 adapter outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quiethand.m3_adapter_contract import (  # noqa: E402
    M3AdapterContractError,
    validate_itemized_adapter_result,
    validate_semantic_candidate,
)


class CalibrationAuditError(RuntimeError):
    """A completed adapter result violates the frozen calibration contract."""


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--raw-hand-root", type=Path, required=True)
    parser.add_argument("--perception-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _json_pairs(values: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in values:
        if key in result:
            raise CalibrationAuditError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise CalibrationAuditError(f"non-finite JSON constant: {value}")


def _stable_bytes(path: Path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CalibrationAuditError(f"not a regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    landed = os.lstat(path)
    snapshots = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ), (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ), (
        landed.st_dev,
        landed.st_ino,
        landed.st_size,
        landed.st_mtime_ns,
        landed.st_ctime_ns,
    )
    if snapshots[0] != snapshots[1] or snapshots[1] != snapshots[2]:
        raise CalibrationAuditError(f"file changed during audit: {path}")
    return b"".join(chunks)


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            _stable_bytes(path).decode("utf-8"),
            object_pairs_hook=_json_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationAuditError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CalibrationAuditError(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CalibrationAuditError(f"not a regular artifact: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    landed = os.lstat(path)
    snapshots = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ), (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ), (
        landed.st_dev,
        landed.st_ino,
        landed.st_size,
        landed.st_mtime_ns,
        landed.st_ctime_ns,
    )
    if snapshots[0] != snapshots[1] or snapshots[1] != snapshots[2]:
        raise CalibrationAuditError(f"artifact changed during audit: {path}")
    return digest.hexdigest()


def _safe_artifact(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise CalibrationAuditError("artifact path is not safe and relative")
    if root.is_symlink() or not root.is_dir():
        raise CalibrationAuditError(f"result root is missing or symlinked: {root}")
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise CalibrationAuditError(f"artifact path contains symlink: {relative}")
    if not current.is_file():
        raise CalibrationAuditError(f"artifact is missing: {relative}")
    return current


def _valid_se3(values: np.ndarray) -> bool:
    if values.shape != (15, 4, 4) or not np.isfinite(values).all():
        return False
    if not np.allclose(values[:, 3], np.asarray([0, 0, 0, 1]), atol=1e-4, rtol=0):
        return False
    rotations = values[:, :3, :3].astype(np.float64)
    products = np.swapaxes(rotations, 1, 2) @ rotations
    determinants = np.linalg.det(rotations)
    return bool(
        np.allclose(products, np.eye(3), atol=2e-3, rtol=0)
        and np.all(np.abs(determinants - 1.0) <= 2e-3)
    )


def _verify_array(
    root: Path,
    artifact: Mapping[str, object],
    *,
    adapter: str,
) -> tuple[str, int]:
    relative = artifact["relative_path"]
    if not isinstance(relative, str):
        raise CalibrationAuditError("artifact relative path is not a string")
    path = _safe_artifact(root, relative)
    if path.stat().st_size != artifact["bytes"]:
        raise CalibrationAuditError(f"artifact byte count differs: {relative}")
    if _sha256(path) != artifact["sha256"]:
        raise CalibrationAuditError(f"artifact SHA-256 differs: {relative}")
    try:
        values = np.load(path, allow_pickle=False, mmap_mode="r")
    except Exception as exc:
        raise CalibrationAuditError(f"artifact is not a safe NumPy array: {relative}") from exc
    expected_shape = tuple(artifact["shape"])
    if values.shape != expected_shape or str(values.dtype) != artifact["dtype"]:
        raise CalibrationAuditError(f"artifact array metadata differs: {relative}")
    if artifact["frame_indices"] != list(range(15)) or expected_shape[0] != 15:
        raise CalibrationAuditError(f"artifact frame coverage is not exactly 15: {relative}")
    if adapter == "raw_hand":
        if values.shape != (15, 778, 3) or not np.isfinite(values).all():
            raise CalibrationAuditError(f"raw-hand array is non-finite or malformed: {relative}")
    elif adapter == "segmentation":
        if values.shape != (15, 1080, 1920) or not np.logical_or(values == 0, values == 1).all():
            raise CalibrationAuditError(f"segmentation mask is non-binary or malformed: {relative}")
    elif adapter == "object_state":
        if not _valid_se3(values):
            raise CalibrationAuditError(f"object-state array is not finite SE(3): {relative}")
    else:
        raise CalibrationAuditError(f"unsupported artifact adapter: {adapter}")
    return relative, path.stat().st_size


def _input_events(path: Path, adapter: str) -> tuple[dict[str, object], list[str]]:
    manifest = _load_json(path)
    events = manifest.get("events")
    if (
        manifest.get("adapter") != adapter
        or manifest.get("event_count") != 150
        or manifest.get("evaluation_event_count") != 0
        or manifest.get("evaluation_model_results_opened") is not False
        or not isinstance(events, list)
        or len(events) != 150
    ):
        raise CalibrationAuditError(f"{adapter} input is not the frozen calibration envelope")
    event_ids = [event.get("event_id") if isinstance(event, Mapping) else None for event in events]
    if any(not isinstance(event_id, str) for event_id in event_ids) or len(set(event_ids)) != 150:
        raise CalibrationAuditError(f"{adapter} input event identities are invalid")
    return manifest, list(event_ids)


def _audit_envelope(
    path: Path,
    *,
    adapter: str,
    input_path: Path,
    runner_path: Path,
) -> dict[str, object]:
    value = _load_json(path)
    required = {
        "schema": "quiethand.m3.adapter_job_audit.v1",
        "status": "COMPLETE",
        "adapter": adapter,
        "event_count": 150,
        "input_event_count": 150,
        "run_mode": "full_calibration",
        "event_index": None,
        "input_sha256": _sha256(input_path),
        "runner_sha256": _sha256(runner_path),
        "training_performed": False,
        "evaluation_results_opened": False,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise CalibrationAuditError(f"{adapter} audit field differs: {key}")
    return value


def _semantic_results(root: Path, event_ids: list[str]) -> tuple[dict[str, object], dict[str, int]]:
    event_root = root / "events"
    files = {path.stem: path for path in event_root.glob("*.json")}
    if set(files) != set(event_ids) or len(files) != 150:
        raise CalibrationAuditError("semantic event-file coverage differs from frozen input")
    counts = {"observed": 0, "semantic_invalid": 0}
    candidates: dict[str, object] = {}
    for event_id in event_ids:
        value = _load_json(files[event_id])
        if set(value) != {"schema", "event_id", "adapter", "status", "candidate", "raw_text", "failure_reason"}:
            raise CalibrationAuditError(f"semantic result fields differ: {event_id}")
        if value["schema"] != "quiethand.m3.semantic_result.v1" or value["event_id"] != event_id or value["adapter"] != "semantic":
            raise CalibrationAuditError(f"semantic result envelope differs: {event_id}")
        raw_text = value["raw_text"]
        if not isinstance(raw_text, str):
            raise CalibrationAuditError(f"semantic raw text is missing: {event_id}")
        if value["status"] == "observed":
            candidate = validate_semantic_candidate(raw_text)
            if candidate != value["candidate"] or value["failure_reason"] is not None:
                raise CalibrationAuditError(f"semantic observed payload differs: {event_id}")
            candidates[event_id] = candidate
            counts["observed"] += 1
        elif value["status"] == "semantic_invalid":
            if value["candidate"] is not None or not isinstance(value["failure_reason"], str) or not value["failure_reason"].strip():
                raise CalibrationAuditError(f"semantic invalid payload is silent: {event_id}")
            try:
                validate_semantic_candidate(raw_text)
            except M3AdapterContractError as exc:
                if str(exc) != value["failure_reason"]:
                    raise CalibrationAuditError(f"semantic failure reason differs: {event_id}") from exc
            else:
                raise CalibrationAuditError(f"semantic invalid payload validates: {event_id}")
            candidates[event_id] = None
            counts["semantic_invalid"] += 1
        else:
            raise CalibrationAuditError(f"semantic state is undeclared: {event_id}")
    return candidates, counts


def _itemized_results(
    root: Path,
    event_ids: list[str],
    *,
    adapter: str,
    nested_name: str | None,
) -> tuple[dict[str, dict[str, object]], dict[str, int], set[str], int]:
    event_root = root / "events"
    expected_json = {
        (event_root / event_id / nested_name) if nested_name is not None else (event_root / f"{event_id}.json")
        for event_id in event_ids
    }
    actual_json = (
        set(event_root.rglob(nested_name))
        if nested_name is not None
        else set(event_root.glob("*.json"))
    )
    if actual_json != expected_json:
        raise CalibrationAuditError(f"{adapter} JSON coverage differs from frozen input")
    results: dict[str, dict[str, object]] = {}
    counts = {"observed": 0, "abstain": 0, "invalid": 0}
    referenced: set[str] = set()
    artifact_bytes = 0
    for event_id in event_ids:
        path = (event_root / event_id / nested_name) if nested_name is not None else (event_root / f"{event_id}.json")
        value = validate_itemized_adapter_result(_load_json(path), adapter=adapter, event_id=event_id)
        results[event_id] = value
        for item in value["items"].values():
            counts[item["status"]] += 1
            if item["status"] == "observed":
                relative, size = _verify_array(root, item["artifact"], adapter=adapter)
                if relative in referenced:
                    raise CalibrationAuditError(f"artifact is referenced twice: {relative}")
                referenced.add(relative)
                artifact_bytes += size
    return results, counts, referenced, artifact_bytes


def _canonical_sha256(value: object) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = arguments()
    inputs = {
        "semantic": args.workspace / "artifacts/quiethand/m3/QH_M3_SEMANTIC_INPUT.json",
        "raw_hand": args.workspace / "artifacts/quiethand/m3/QH_M3_RAW_HAND_INPUT.json",
        "segmentation_object_state": args.workspace / "artifacts/quiethand/m3/QH_M3_SEGMENTATION_OBJECT_INPUT.json",
    }
    runners = {
        "semantic": args.workspace / "scripts/quiethand/m3_jobs/run_semantic.py",
        "raw_hand": args.workspace / "scripts/quiethand/m3_jobs/run_raw_hand.py",
        "segmentation_object_state": args.workspace / "scripts/quiethand/m3_jobs/run_segmentation_object.py",
    }
    input_values = {}
    event_orders = {}
    for adapter, path in inputs.items():
        input_values[adapter], event_orders[adapter] = _input_events(path, adapter)
    semantic_ids = event_orders["semantic"]
    if set(semantic_ids) != set(event_orders["raw_hand"]) or set(semantic_ids) != set(event_orders["segmentation_object_state"]):
        raise CalibrationAuditError("adapter input event sets differ")

    semantic_audit = _audit_envelope(
        args.semantic_root / "audit.json",
        adapter="semantic",
        input_path=inputs["semantic"],
        runner_path=runners["semantic"],
    )
    raw_audit = _audit_envelope(
        args.raw_hand_root / "audit.json",
        adapter="raw_hand",
        input_path=inputs["raw_hand"],
        runner_path=runners["raw_hand"],
    )
    perception_audit = _audit_envelope(
        args.perception_root / "audit.json",
        adapter="segmentation_object_state",
        input_path=inputs["segmentation_object_state"],
        runner_path=runners["segmentation_object_state"],
    )
    if perception_audit.get("native_masks_used") is not False:
        raise CalibrationAuditError("perception audit does not prove native-mask exclusion")
    if raw_audit.get("slam_performed") is not False or raw_audit.get("infiller_performed") is not False:
        raise CalibrationAuditError("raw-hand audit does not prove SLAM/infill exclusion")
    if perception_audit.get("semantic_audit_sha256") != _sha256(args.semantic_root / "audit.json"):
        raise CalibrationAuditError("perception semantic dependency hash differs")

    candidates, semantic_counts = _semantic_results(args.semantic_root, semantic_ids)
    raw_results, raw_counts, raw_artifacts, raw_bytes = _itemized_results(
        args.raw_hand_root, semantic_ids, adapter="raw_hand", nested_name=None
    )
    segmentation_results, segmentation_counts, segmentation_artifacts, segmentation_bytes = _itemized_results(
        args.perception_root, semantic_ids, adapter="segmentation", nested_name="segmentation.json"
    )
    object_results, object_counts, object_artifacts, object_bytes = _itemized_results(
        args.perception_root, semantic_ids, adapter="object_state", nested_name="object_state.json"
    )

    if semantic_audit.get("observed_count") != semantic_counts["observed"] or semantic_audit.get("invalid_count") != semantic_counts["semantic_invalid"]:
        raise CalibrationAuditError("semantic audit counts do not replay")
    if raw_audit.get("item_counts") != raw_counts:
        raise CalibrationAuditError("raw-hand audit counts do not replay")
    perception_counts = {
        "segmentation_observed": segmentation_counts["observed"],
        "segmentation_abstain": segmentation_counts["abstain"],
        "segmentation_invalid": segmentation_counts["invalid"],
        "object_observed": object_counts["observed"],
        "object_abstain": object_counts["abstain"],
        "object_invalid": object_counts["invalid"],
    }
    if perception_audit.get("item_counts") != perception_counts:
        raise CalibrationAuditError("perception audit counts do not replay")

    for event_id in semantic_ids:
        boxes = None if candidates[event_id] is None else candidates[event_id]["first_frame_boxes_0_1000"]
        for role in ("tool", "target"):
            segmentation_item = segmentation_results[event_id]["items"][role]
            object_item = object_results[event_id]["items"][role]
            if boxes is None or boxes[role] is None:
                expected_reason = "semantic_invalid" if boxes is None else "semantic_box_null"
                if segmentation_item["status"] != "abstain" or segmentation_item["failure_reason"] != expected_reason:
                    raise CalibrationAuditError(f"segmentation abstention provenance differs: {event_id}/{role}")
            if segmentation_item["status"] != "observed":
                if object_item["status"] != "abstain" or object_item["failure_reason"] != f"segmentation_{segmentation_item['status']}":
                    raise CalibrationAuditError(f"object abstention provenance differs: {event_id}/{role}")

    actual_raw_arrays = {path.relative_to(args.raw_hand_root).as_posix() for path in args.raw_hand_root.rglob("*.npy")}
    actual_perception_arrays = {path.relative_to(args.perception_root).as_posix() for path in args.perception_root.rglob("*.npy")}
    if actual_raw_arrays != raw_artifacts:
        raise CalibrationAuditError("raw-hand artifact file set differs from references")
    if actual_perception_arrays != segmentation_artifacts | object_artifacts:
        raise CalibrationAuditError("perception artifact file set differs from references")

    adapter_valid_events = {
        "semantic": sum(candidates[event_id] is not None for event_id in semantic_ids),
        "raw_hand": sum(any(item["status"] == "observed" for item in raw_results[event_id]["items"].values()) for event_id in semantic_ids),
        "segmentation": sum(any(item["status"] == "observed" for item in segmentation_results[event_id]["items"].values()) for event_id in semantic_ids),
        "object_state": sum(any(item["status"] == "observed" for item in object_results[event_id]["items"].values()) for event_id in semantic_ids),
    }
    if any(value < 1 for value in adapter_valid_events.values()):
        raise CalibrationAuditError("at least one adapter lacks a valid calibration event")

    result = {
        "schema": "quiethand.m3.adapter_calibration_audit.v1",
        "status": "PASS_QH_M3_003_ADAPTER_CALIBRATION",
        "claim_boundary": "ENGINEERING_ONLY_CALIBRATION_NO_SCIENTIFIC_RESULT",
        "event_count": 150,
        "event_set_sha256": _canonical_sha256(sorted(semantic_ids)),
        "input_sha256": {adapter: _sha256(path) for adapter, path in inputs.items()},
        "source_audit_sha256": {
            "semantic": _sha256(args.semantic_root / "audit.json"),
            "raw_hand": _sha256(args.raw_hand_root / "audit.json"),
            "segmentation_object_state": _sha256(args.perception_root / "audit.json"),
        },
        "runner_sha256": {adapter: _sha256(path) for adapter, path in runners.items()},
        "semantic_counts": semantic_counts,
        "raw_hand_counts": raw_counts,
        "segmentation_counts": segmentation_counts,
        "object_state_counts": object_counts,
        "adapter_valid_event_counts": adapter_valid_events,
        "artifact_file_count": len(raw_artifacts | segmentation_artifacts | object_artifacts),
        "artifact_bytes": raw_bytes + segmentation_bytes + object_bytes,
        "silent_null_count": 0,
        "undeclared_artifact_count": 0,
        "evaluation_results_opened": False,
        "native_masks_used": False,
        "slam_performed": False,
        "infiller_performed": False,
        "training_performed": False,
        "full_calibration_a800_hours": (
            float(semantic_audit["elapsed_seconds"])
            + float(raw_audit["elapsed_seconds"])
            + float(perception_audit["elapsed_seconds"])
        ) / 3600.0,
        "auditor_sha256": _sha256(Path(__file__)),
    }
    _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
