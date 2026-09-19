#!/usr/bin/env python
"""Check whether M3-v1.2 hand and object outputs share one metric camera model."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


FOCAL_RELATIVE_TOLERANCE = 0.01
EXPECTED_CALIBRATION_EVENTS = 30


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--taco-root", type=Path, required=True)
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


def finite_array(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    value = np.load(path, allow_pickle=False).astype(np.float64, copy=False)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"array contract failed: {path}")
    return value


def observed_item_path(root: Path, event_id: str, item: dict[str, object]) -> Path | None:
    if item.get("status") != "observed":
        return None
    artifact = item.get("artifact")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("relative_path"), str):
        raise ValueError(f"observed item lacks artifact: {event_id}")
    return root / str(artifact["relative_path"])


def main() -> int:
    args = arguments()
    project = args.project_root.resolve()
    taco = args.taco_root.resolve()
    artifact_root = project / "artifacts/quiethand/m3_v1_2"
    plan = load_json(artifact_root / "QH_M3_V1_2_TEMPORAL_PLAN.json")
    raw_audit = load_json(artifact_root / "results/raw_hand/audit.json")

    events = plan.get("events")
    if (
        plan.get("calibration_video_count") != EXPECTED_CALIBRATION_EVENTS
        or plan.get("evaluation_video_count") != 0
        or not isinstance(events, list)
        or len(events) != EXPECTED_CALIBRATION_EVENTS
        or raw_audit.get("event_count") != EXPECTED_CALIBRATION_EVENTS
        or raw_audit.get("evaluation_results_opened") is not False
    ):
        raise SystemExit("input is not the frozen 30/0 calibration-only set")

    hand_focal = float(raw_audit.get("focal_length_px", math.nan))
    principal_point_recorded = "principal_point_px" in raw_audit
    records: list[dict[str, object]] = []
    mismatch_count = 0
    focal_ratios: list[float] = []
    hand_z_values: list[float] = []
    object_z_values: list[float] = []

    raw_root = artifact_root / "results/raw_hand"
    perception_root = artifact_root / "results/perception"
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("event_id"), str):
            raise SystemExit("temporal plan contains an invalid event")
        event_id = str(event["event_id"])
        binding = event.get("source_binding")
        if not isinstance(binding, dict) or not isinstance(binding.get("intrinsic"), dict):
            raise SystemExit(f"{event_id}: intrinsic binding missing")
        relative = binding["intrinsic"].get("relative_path")
        if not isinstance(relative, str):
            raise SystemExit(f"{event_id}: intrinsic path missing")
        intrinsic = np.loadtxt(taco / relative, dtype=np.float64)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise SystemExit(f"{event_id}: intrinsic is invalid")
        fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
        cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
        sensor_focal = (fx + fy) / 2.0
        ratio = sensor_focal / hand_focal
        focal_ratios.append(ratio)
        relative_error = abs(sensor_focal - hand_focal) / sensor_focal

        raw = load_json(raw_root / "events" / f"{event_id}.json")
        perception_event = perception_root / "events" / event_id
        objects = load_json(perception_event / "object_state.json")
        hand_depths: dict[str, float | None] = {}
        object_depths: dict[str, float | None] = {}
        for side in ("left", "right"):
            item = raw.get("items", {}).get(side) if isinstance(raw.get("items"), dict) else None
            if not isinstance(item, dict):
                raise SystemExit(f"{event_id}: raw hand item missing: {side}")
            path = observed_item_path(raw_root, event_id, item)
            if path is None:
                hand_depths[side] = None
            else:
                array = finite_array(path, (15, 778, 3))
                depth = float(np.median(array[..., 2]))
                hand_depths[side] = depth
                hand_z_values.append(depth)
        for role in ("tool", "target"):
            item = objects.get("items", {}).get(role) if isinstance(objects.get("items"), dict) else None
            if not isinstance(item, dict):
                raise SystemExit(f"{event_id}: object item missing: {role}")
            path = observed_item_path(perception_root, event_id, item)
            if path is None:
                object_depths[role] = None
            else:
                array = finite_array(path, (15, 4, 4))
                depth = float(np.median(array[:, 2, 3]))
                object_depths[role] = depth
                object_z_values.append(depth)

        event_mismatch = relative_error > FOCAL_RELATIVE_TOLERANCE or not principal_point_recorded
        mismatch_count += int(event_mismatch)
        records.append(
            {
                "event_id": event_id,
                "sensor_intrinsic_px": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
                "hand_adapter_focal_px": hand_focal,
                "hand_adapter_principal_point_recorded": principal_point_recorded,
                "sensor_to_hand_focal_ratio": ratio,
                "focal_relative_error": relative_error,
                "hand_median_depth_m": hand_depths,
                "object_median_depth_m": object_depths,
                "status": "mismatch" if event_mismatch else "aligned",
            }
        )

    status = "HOLD_METRIC_ALIGNMENT" if mismatch_count else "PASS_METRIC_ALIGNMENT"
    result = {
        "schema": "quiethand.m3_5.metric_alignment_audit.v1",
        "status": status,
        "calibration_event_count": EXPECTED_CALIBRATION_EVENTS,
        "evaluation_event_count": 0,
        "evaluation_results_opened": False,
        "focal_relative_tolerance": FOCAL_RELATIVE_TOLERANCE,
        "mismatched_event_count": mismatch_count,
        "hand_adapter_focal_px": hand_focal,
        "hand_adapter_principal_point_recorded": principal_point_recorded,
        "sensor_to_hand_focal_ratio_range": [min(focal_ratios), max(focal_ratios)],
        "observed_hand_median_depth_m_range": [min(hand_z_values), max(hand_z_values)],
        "observed_object_median_depth_m_range": [min(object_z_values), max(object_z_values)],
        "fusion_performed": False,
        "c_oame_performed": False,
        "reason": (
            "HaWoR outputs were generated with one fixed focal length and no recorded per-event principal point, "
            "while TACO sensor intrinsics are event-specific. Hand and object outputs therefore do not yet share "
            "one metric camera model."
            if mismatch_count
            else None
        ),
        "next_action": (
            "rerun the calibration-only HaWoR adapter with each event's TACO intrinsic before fusion"
            if mismatch_count
            else "compute four hand-object temporal evidence tracks"
        ),
        "records": records,
    }
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in (
        "status", "calibration_event_count", "evaluation_event_count",
        "mismatched_event_count", "sensor_to_hand_focal_ratio_range",
        "observed_hand_median_depth_m_range", "observed_object_median_depth_m_range",
        "next_action",
    )}, ensure_ascii=False, indent=2))
    return 3 if mismatch_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
