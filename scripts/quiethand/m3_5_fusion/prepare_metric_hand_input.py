#!/usr/bin/env python3
"""Build the calibration-only HaWoR input with the real TACO camera model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np


EXPECTED_EVENTS = 30
IMAGE_SIZE = [1920, 1080]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-input", type=Path, required=True)
    parser.add_argument("--temporal-plan", type=Path, required=True)
    parser.add_argument("--taco-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_object(path: Path) -> dict[str, object]:
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


def validate_calibration_envelope(raw: dict[str, object], plan: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw_events = raw.get("events")
    plan_events = plan.get("events")
    if (
        raw.get("adapter") != "raw_hand"
        or raw.get("calibration_video_count") != EXPECTED_EVENTS
        or raw.get("evaluation_event_count") != 0
        or raw.get("evaluation_model_results_opened") is not False
        or raw.get("event_count") != EXPECTED_EVENTS
        or plan.get("calibration_video_count") != EXPECTED_EVENTS
        or plan.get("evaluation_video_count") != 0
        or plan.get("evaluation_model_results_opened") is not False
        or not isinstance(raw_events, list)
        or not isinstance(plan_events, list)
        or len(raw_events) != EXPECTED_EVENTS
        or len(plan_events) != EXPECTED_EVENTS
        or any(not isinstance(event, dict) for event in raw_events + plan_events)
    ):
        raise ValueError("inputs are not the frozen 30/0 calibration-only set")
    raw_ids = [event.get("event_id") for event in raw_events]
    plan_ids = [event.get("event_id") for event in plan_events]
    if raw_ids != plan_ids or len(set(raw_ids)) != EXPECTED_EVENTS:
        raise ValueError("raw-hand input and temporal plan event order do not match")
    return raw_events, plan_events


def camera_model(plan_event: dict[str, object], taco_root: Path) -> dict[str, object]:
    binding = plan_event.get("source_binding")
    intrinsic_binding = binding.get("intrinsic") if isinstance(binding, dict) else None
    relative = intrinsic_binding.get("relative_path") if isinstance(intrinsic_binding, dict) else None
    if not isinstance(relative, str):
        raise ValueError(f"{plan_event.get('event_id')}: source intrinsic binding is missing")
    intrinsic = np.loadtxt(taco_root / relative, dtype=np.float64)
    if (
        intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
        or not np.allclose(intrinsic[2], [0.0, 0.0, 1.0], rtol=0.0, atol=1e-9)
    ):
        raise ValueError(f"{plan_event.get('event_id')}: source intrinsic is invalid")
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    if fx <= 0.0 or fy <= 0.0 or not 0.0 <= cx < IMAGE_SIZE[0] or not 0.0 <= cy < IMAGE_SIZE[1]:
        raise ValueError(f"{plan_event.get('event_id')}: source intrinsic is outside the image")
    focal = (fx + fy) / 2.0
    return {
        "source": "TACO egocentric intrinsic",
        "source_relative_path": relative,
        "intrinsic_px": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "image_size_px": IMAGE_SIZE,
        "hawor_focal_px": focal,
        "hawor_principal_point_px": [cx, cy],
        "hawor_focal_policy": "arithmetic mean of fx and fy because HaWoR accepts one scalar focal length",
        "focal_anisotropy_relative": abs(fx - fy) / focal,
    }


def build(raw: dict[str, object], plan: dict[str, object], taco_root: Path) -> dict[str, object]:
    raw_events, plan_events = validate_calibration_envelope(raw, plan)
    events = []
    for raw_event, plan_event in zip(raw_events, plan_events, strict=True):
        event = dict(raw_event)
        event["camera_model"] = camera_model(plan_event, taco_root)
        events.append(event)
    return {
        "schema": "quiethand.m3_5.raw_hand_metric_input.v1",
        "adapter": "raw_hand_metric",
        "calibration_video_count": EXPECTED_EVENTS,
        "evaluation_event_count": 0,
        "evaluation_model_results_opened": False,
        "event_count": EXPECTED_EVENTS,
        "events": events,
    }


def main() -> int:
    args = arguments()
    result = build(load_object(args.raw_input), load_object(args.temporal_plan), args.taco_root.resolve())
    atomic_json(args.output, result)
    anisotropy = [float(event["camera_model"]["focal_anisotropy_relative"]) for event in result["events"]]
    focal = [float(event["camera_model"]["hawor_focal_px"]) for event in result["events"]]
    print(json.dumps({
        "status": "READY_METRIC_HAND_INPUT",
        "calibration_event_count": result["event_count"],
        "evaluation_event_count": result["evaluation_event_count"],
        "hawor_focal_px_range": [min(focal), max(focal)],
        "focal_anisotropy_relative_max": max(anisotropy),
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
