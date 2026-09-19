#!/usr/bin/env python3
"""Verify that corrected HaWoR and FoundationPose outputs share the TACO camera model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


EXPECTED_EVENTS = 30


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-input", type=Path, required=True)
    parser.add_argument("--temporal-plan", type=Path, required=True)
    parser.add_argument("--taco-root", type=Path, required=True)
    parser.add_argument("--hand-results", type=Path, required=True)
    parser.add_argument("--perception-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observed_array(root: Path, document: dict[str, object], name: str, shape: tuple[int, ...]) -> np.ndarray | None:
    items = document.get("items")
    item = items.get(name) if isinstance(items, dict) else None
    if not isinstance(item, dict):
        raise ValueError(f"missing item: {name}")
    if item.get("status") in {"abstain", "invalid"}:
        if item.get("artifact") is not None or not item.get("failure_reason"):
            raise ValueError(f"invalid missing-evidence item: {name}")
        return None
    if item.get("status") != "observed":
        raise ValueError(f"unexpected item status: {name}={item.get('status')}")
    artifact = item.get("artifact")
    if (
        not isinstance(artifact, dict)
        or artifact.get("shape") != list(shape)
        or artifact.get("frame_id") != "camera"
        or not isinstance(artifact.get("relative_path"), str)
        or not isinstance(artifact.get("sha256"), str)
    ):
        raise ValueError(f"invalid artifact declaration: {name}")
    path = root / str(artifact["relative_path"])
    if not path.is_file() or sha256(path) != artifact["sha256"]:
        raise ValueError(f"artifact binding failed: {path}")
    value = np.load(path, allow_pickle=False).astype(np.float64, copy=False)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"artifact array failed: {path}")
    return value


def main() -> int:
    args = arguments()
    manifest = load_json(args.metric_input)
    plan = load_json(args.temporal_plan)
    hand_audit = load_json(args.hand_results / "audit.json")
    events = manifest.get("events")
    plan_events = plan.get("events")
    if (
        manifest.get("adapter") != "raw_hand_metric"
        or manifest.get("event_count") != EXPECTED_EVENTS
        or manifest.get("evaluation_event_count") != 0
        or manifest.get("evaluation_model_results_opened") is not False
        or hand_audit.get("status") != "COMPLETE"
        or hand_audit.get("event_count") != EXPECTED_EVENTS
        or hand_audit.get("evaluation_results_opened") is not False
        or hand_audit.get("training_performed") is not False
        or hand_audit.get("item_counts") != {"observed": 58, "abstain": 2, "invalid": 0}
        or plan.get("calibration_video_count") != EXPECTED_EVENTS
        or plan.get("evaluation_video_count") != 0
        or plan.get("evaluation_model_results_opened") is not False
        or not isinstance(events, list)
        or not isinstance(plan_events, list)
        or len(events) != EXPECTED_EVENTS
        or len(plan_events) != EXPECTED_EVENTS
    ):
        raise SystemExit("corrected metric hand envelope is not complete 30/0 calibration output")
    plan_by_id = {event["event_id"]: event for event in plan_events}
    mismatch = 0
    hand_depths: list[float] = []
    object_depths: list[float] = []
    records = []
    for event in events:
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or event_id not in plan_by_id:
            raise SystemExit("event identity mismatch")
        plan_event = plan_by_id[event_id]
        binding = plan_event.get("source_binding")
        intrinsic_binding = binding.get("intrinsic") if isinstance(binding, dict) else None
        relative = intrinsic_binding.get("relative_path") if isinstance(intrinsic_binding, dict) else None
        if not isinstance(relative, str):
            raise SystemExit(f"{event_id}: intrinsic source missing")
        true_k = np.loadtxt(args.taco_root / relative, dtype=np.float64)
        input_camera = event.get("camera_model")
        input_k = np.asarray(input_camera.get("intrinsic_px") if isinstance(input_camera, dict) else None, dtype=np.float64)
        result = load_json(args.hand_results / "events" / f"{event_id}.json")
        result_camera = result.get("camera_model")
        result_k = np.asarray(result_camera.get("intrinsic_px") if isinstance(result_camera, dict) else None, dtype=np.float64)
        camera_match = true_k.shape == input_k.shape == result_k.shape == (3, 3) and np.array_equal(true_k, input_k) and np.array_equal(input_k, result_k)
        mismatch += int(not camera_match)
        side_depth = {}
        for side in ("left", "right"):
            array = observed_array(args.hand_results, result, side, (15, 778, 3))
            if array is None:
                side_depth[side] = None
            else:
                median = float(np.median(array[..., 2]))
                if median <= 0.0:
                    raise SystemExit(f"{event_id}: non-positive hand depth")
                hand_depths.append(median)
                side_depth[side] = median
        objects = load_json(args.perception_results / "events" / event_id / "object_state.json")
        role_depth = {}
        for role in ("tool", "target"):
            array = observed_array(args.perception_results, objects, role, (15, 4, 4))
            if array is None:
                role_depth[role] = None
            else:
                median = float(np.median(array[:, 2, 3]))
                if median <= 0.0:
                    raise SystemExit(f"{event_id}: non-positive object depth")
                object_depths.append(median)
                role_depth[role] = median
        records.append({
            "event_id": event_id,
            "camera_model_match": camera_match,
            "hand_median_depth_m": side_depth,
            "object_median_depth_m": role_depth,
        })
    result = {
        "schema": "quiethand.m3_5.metric_hand_verification.v1",
        "status": "PASS_METRIC_ALIGNMENT" if mismatch == 0 else "HOLD_METRIC_ALIGNMENT",
        "calibration_event_count": EXPECTED_EVENTS,
        "evaluation_event_count": 0,
        "evaluation_results_opened": False,
        "camera_model_mismatch_count": mismatch,
        "hand_item_counts": hand_audit["item_counts"],
        "hand_median_depth_m_range": [min(hand_depths), max(hand_depths)],
        "object_median_depth_m_range": [min(object_depths), max(object_depths)],
        "fusion_performed": False,
        "records": records,
    }
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in (
        "status", "calibration_event_count", "evaluation_event_count", "camera_model_mismatch_count",
        "hand_item_counts", "hand_median_depth_m_range", "object_median_depth_m_range",
    )}, ensure_ascii=False, indent=2))
    return 0 if mismatch == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
