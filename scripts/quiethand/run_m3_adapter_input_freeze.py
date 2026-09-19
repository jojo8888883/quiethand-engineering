#!/usr/bin/env python3
"""Freeze three model-facing M3 manifests with no TACO metadata fields."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quiethand.m3_materialization import (  # noqa: E402
    M3MaterializationError,
    TASK_PLAN_SHA256,
    verify_materialization_manifest,
)


MATERIALIZATION_PATH = ROOT / "artifacts/quiethand/m3/QH_M3_CALIBRATION_MATERIALIZATION.json"
TASK_PATH = ROOT / "artifacts/quiethand/m3/QH_M3_CALIBRATION_TASK_PLAN.json"
EXPECTED_MATERIALIZATION_SHA256 = "6c4a6c1083b977ee9bd0352bdbef88bf24ebb1b6218919cb3c89b267d965f145"
OUTPUTS = {
    "semantic": ROOT / "artifacts/quiethand/m3/QH_M3_SEMANTIC_INPUT.json",
    "raw_hand": ROOT / "artifacts/quiethand/m3/QH_M3_RAW_HAND_INPUT.json",
    "segmentation_object_state": ROOT / "artifacts/quiethand/m3/QH_M3_SEGMENTATION_OBJECT_INPUT.json",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic(path: Path, value: dict[str, object]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    try:
        if _sha(TASK_PATH) != TASK_PLAN_SHA256 or _sha(MATERIALIZATION_PATH) != EXPECTED_MATERIALIZATION_SHA256:
            raise M3MaterializationError("frozen calibration source changed")
        task_plan = json.loads(TASK_PATH.read_text(encoding="utf-8"))
        materialization = json.loads(MATERIALIZATION_PATH.read_text(encoding="utf-8"))
        verify_materialization_manifest(materialization, ROOT)
        tasks = task_plan.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != 150:
            raise M3MaterializationError("task plan coverage changed")
        event_files = {event["event_id"]: event["files"] for event in materialization["events"]}
        projected: dict[str, list[dict[str, object]]] = {name: [] for name in OUTPUTS}
        forbidden_keys = {
            "rank", "sequence_id", "sequence_name", "triplet", "source_binding",
            "center_frame", "frame_indices", "event_index", "orchestration_only_never_model_input",
        }
        for task in tasks:
            event_id = task["event_id"]
            visible = task["adapter_visible"]
            files = event_files.get(event_id)
            if files is None:
                raise M3MaterializationError("materialized event is missing")
            semantic = json.loads(json.dumps(visible["semantic"], allow_nan=False))
            raw_hand = json.loads(json.dumps(visible["raw_hand"], allow_nan=False))
            seg_object = {
                "event_id": event_id,
                "segmentation": json.loads(json.dumps(visible["segmentation"], allow_nan=False)),
                "object_state": json.loads(json.dumps(visible["object_state"], allow_nan=False)),
            }
            # Bind only hashes and shapes of already model-visible files.  No native label,
            # pose, action/tool/target identity, rank, or split is copied.
            bound = {}
            for kind in ("rgb", "depth", "intrinsic", "tool_mesh", "target_mesh"):
                records = files[kind]
                if not isinstance(records, list):
                    records = [records]
                bound[kind] = [
                    {
                        key: record[key]
                        for key in ("relative_path", "bytes", "sha256", "dtype", "shape", "unit")
                    }
                    for record in records
                ]
            semantic["materialized_file_binding"] = {"rgb": bound["rgb"]}
            raw_hand["materialized_file_binding"] = {"rgb": bound["rgb"]}
            seg_object["materialized_file_binding"] = bound
            projected["semantic"].append(semantic)
            projected["raw_hand"].append(raw_hand)
            projected["segmentation_object_state"].append(seg_object)
        for adapter, events in projected.items():
            payload = {
                "schema": "quiethand.m3.adapter_input.v1",
                "status": "READY_FOR_CALIBRATION_INFERENCE",
                "adapter": adapter,
                "event_count": 150,
                "evaluation_event_count": 0,
                "evaluation_model_results_opened": False,
                "task_plan_sha256": TASK_PLAN_SHA256,
                "materialization_sha256": EXPECTED_MATERIALIZATION_SHA256,
                "events": sorted(events, key=lambda event: event["event_id"]),
            }
            serialized = json.dumps(payload, sort_keys=True, allow_nan=False)
            if any(f'"{key}"' in serialized for key in forbidden_keys):
                raise M3MaterializationError(f"orchestration metadata leaked into {adapter}")
            _atomic(OUTPUTS[adapter], payload)
            print(f"[artifact] {OUTPUTS[adapter]}")
    except (OSError, ValueError, TypeError, json.JSONDecodeError, M3MaterializationError) as exc:
        print(f"[hold] HOLD_ENGINEERING_INCOMPLETE: {exc}", file=sys.stderr)
        return 3
    print("[status] 3 BLINDED CALIBRATION-ONLY ADAPTER INPUTS CLOSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
