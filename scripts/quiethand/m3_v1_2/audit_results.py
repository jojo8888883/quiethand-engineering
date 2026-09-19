#!/usr/bin/env python3
"""Targeted end-to-end audit for the user-authorized M3-v1.2 calibration rerun."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--perception-root", type=Path, required=True)
    parser.add_argument("--preview-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"not a JSON object: {path}")
    return value


def main() -> int:
    cfg = args()
    root = cfg.workspace / "artifacts/quiethand/m3_v1_2"
    temporal = load(root / "QH_M3_V1_2_TEMPORAL_INPUT.json")
    geometry = load(root / "QH_M3_V1_2_GEOMETRY_MATERIALIZATION.json")
    semantic_audit = load(cfg.semantic_root / "audit.json")
    raw_audit = load(cfg.raw_root / "audit.json")
    perception_audit = load(cfg.perception_root / "audit.json")
    preview = load(cfg.preview_root / "data.json")
    failures: list[str] = []
    for label, value in (
        ("temporal", temporal.get("evaluation_event_count")),
        ("geometry", geometry.get("evaluation_event_count")),
    ):
        if value != 0:
            failures.append(f"{label} evaluation count is not zero")
    for label, audit in (("semantic", semantic_audit), ("raw", raw_audit), ("perception", perception_audit)):
        if audit.get("status") != "COMPLETE" or audit.get("evaluation_results_opened") is not False:
            failures.append(f"{label} audit is not complete calibration-only output")
        if audit.get("training_performed") is not False:
            failures.append(f"{label} reports training")
    semantic_ids = {event["event_id"] for event in temporal["events"]}
    geometry_rows = {event["event_id"]: event for event in geometry["events"]}
    if len(semantic_ids) != 30 or set(geometry_rows) != semantic_ids:
        failures.append("30-video semantic/geometry identity coverage differs")
    ready_ids = {event_id for event_id, row in geometry_rows.items() if row["status"] == "ready_geometry_inference"}
    explicit_nonready = semantic_ids - ready_ids
    for event_id in ready_ids:
        frames = geometry_rows[event_id].get("frame_indices")
        if not isinstance(frames, list) or len(frames) != 15 or frames != sorted(set(frames)):
            failures.append(f"{event_id} does not have 15 unique chronological frames")
    if raw_audit.get("input_event_count") != len(ready_ids) or perception_audit.get("input_event_count") != len(ready_ids):
        failures.append("geometry adapter input counts differ from ready intervals")
    modality_counts = {"semantic": {}, "raw_hand": {}, "segmentation": {}, "object_state": {}}
    for event_id in semantic_ids:
        semantic = load(cfg.semantic_root / "events" / f"{event_id}.json")
        status = str(semantic.get("status"))
        modality_counts["semantic"][status] = modality_counts["semantic"].get(status, 0) + 1
        if event_id not in ready_ids:
            continue
        raw = load(cfg.raw_root / "events" / f"{event_id}.json")
        segmentation = load(cfg.perception_root / "events" / event_id / "segmentation.json")
        objects = load(cfg.perception_root / "events" / event_id / "object_state.json")
        for name, document in (("raw_hand", raw), ("segmentation", segmentation), ("object_state", objects)):
            items = document.get("items")
            expected_items = {"left", "right"} if name == "raw_hand" else {"tool", "target"}
            if not isinstance(items, dict) or set(items) != expected_items:
                failures.append(f"{event_id} {name} lacks item states")
                continue
            for item in items.values():
                item_status = str(item.get("status"))
                modality_counts[name][item_status] = modality_counts[name].get(item_status, 0) + 1
                if item_status not in {"observed", "abstain", "invalid"}:
                    failures.append(f"{event_id} {name} has undeclared state {item_status}")
                if item_status == "observed" and not isinstance(item.get("artifact"), dict):
                    failures.append(f"{event_id} {name} observed item lacks artifact")
                if item_status != "observed" and not item.get("failure_reason"):
                    failures.append(f"{event_id} {name} non-observed item lacks reason")
    if preview.get("event_count") != 30 or len(preview.get("records", [])) != 30:
        failures.append("preview does not contain all 30 complete-source records")
    result = {
        "schema": "quiethand.m3_v1_2.calibration_audit.v1",
        "status": "PASS_ENGINEERING_M3_V1_2_CALIBRATION" if not failures else "HOLD_ENGINEERING_INCOMPLETE",
        "calibration_video_count": 30,
        "ready_geometry_count": len(ready_ids),
        "explicit_nonready_count": len(explicit_nonready),
        "modality_item_counts": modality_counts,
        "preview_event_count": preview.get("event_count"),
        "evaluation_results_opened": False,
        "training_performed": False,
        "failures": failures,
    }
    cfg.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if not failures else 3


if __name__ == "__main__":
    raise SystemExit(main())
