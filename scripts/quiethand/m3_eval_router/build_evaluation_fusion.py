#!/usr/bin/env python3
"""Build frozen evaluation hand-object evidence records without human targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
FUSION = ROOT / "scripts/quiethand/m3_5_fusion"
if str(FUSION) not in sys.path:
    sys.path.insert(0, str(FUSION))

from build_fusion_preview import (  # noqa: E402
    CLOSE_M,
    FRAME_COUNT,
    MAX_SURFACE_POINTS,
    MIN_CLOSE_RUN,
    MIN_EVIDENCE_FRAMES,
    NEAR_M,
    STABLE_SPAN_M,
    atomic_json,
    classify_contact,
    item_array,
    nearest_distance,
    relative_envelope,
)


EXPECTED_EVENTS = 30


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--perception-input", type=Path, required=True)
    parser.add_argument("--semantic-results", type=Path, required=True)
    parser.add_argument("--hand-results", type=Path, required=True)
    parser.add_argument("--perception-results", type=Path, required=True)
    parser.add_argument("--identity-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def require_audit(root: Path, adapter: str | None = None) -> dict[str, Any]:
    audit = load(root / "audit.json")
    if (
        audit.get("status") != "COMPLETE"
        or audit.get("event_count") != EXPECTED_EVENTS
        or audit.get("evaluation_event_count") != EXPECTED_EVENTS
        or audit.get("evaluation_results_opened") is not True
        or audit.get("training_performed") is not False
        or (adapter is not None and audit.get("adapter") != adapter)
    ):
        raise ValueError(f"dependency audit is not complete evaluation output: {root}")
    return audit


def visible_surface_points_m(depth_m: np.ndarray, mask: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(depth_m) & (depth_m >= 0.001)
    flat = np.flatnonzero(valid.reshape(-1))
    if flat.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if flat.size > MAX_SURFACE_POINTS:
        flat = flat[np.linspace(0, flat.size - 1, MAX_SURFACE_POINTS, dtype=np.int64)]
    height, width = depth_m.shape
    rows, cols = np.divmod(flat, width)
    z = depth_m.reshape(-1)[flat].astype(np.float32)
    fx, fy = np.float32(intrinsic[0, 0]), np.float32(intrinsic[1, 1])
    cx, cy = np.float32(intrinsic[0, 2]), np.float32(intrinsic[1, 2])
    x = (cols.astype(np.float32) - cx) * z / fx
    y = (rows.astype(np.float32) - cy) * z / fy
    return np.column_stack((x, y, z)).astype(np.float32, copy=False)


def pair_evidence_m(
    hand: np.ndarray | None,
    mask: np.ndarray | None,
    poses: np.ndarray | None,
    depths_m: np.ndarray,
    intrinsic: np.ndarray,
    missing_reasons: list[str | None],
) -> dict[str, Any]:
    if hand is None or mask is None or poses is None:
        return {
            "status": "abstain",
            "failure_reason": "; ".join(reason for reason in missing_reasons if reason) or "required_input_missing",
            "distance_m": [None] * FRAME_COUNT,
            "contact": classify_contact([None] * FRAME_COUNT),
            "relative_motion": None,
            "surface_point_count": [0] * FRAME_COUNT,
        }
    if hand.shape != (FRAME_COUNT, 778, 3) or mask.shape != (FRAME_COUNT, 1080, 1920) or poses.shape != (FRAME_COUNT, 4, 4) or depths_m.shape != (FRAME_COUNT, 1080, 1920):
        raise ValueError("fusion dependency shape changed")
    distances: list[float | None] = []
    counts: list[int] = []
    for index in range(FRAME_COUNT):
        points = visible_surface_points_m(depths_m[index], mask[index], intrinsic)
        counts.append(len(points))
        distances.append(nearest_distance(hand[index], points))
    return {
        "status": "observed",
        "failure_reason": None,
        "distance_m": distances,
        "contact": classify_contact(distances),
        "relative_motion": relative_envelope(np.asarray(hand), np.asarray(poses)),
        "surface_point_count": counts,
    }


def main() -> int:
    cfg = arguments()
    workspace = cfg.workspace.resolve()
    manifest = load(cfg.perception_input)
    events = manifest.get("events")
    if (
        manifest.get("schema") != "quiethand.m3_5.evaluation_perception_input.v1"
        or manifest.get("event_count") != EXPECTED_EVENTS
        or manifest.get("evaluation_event_count") != EXPECTED_EVENTS
        or manifest.get("evaluation_results_opened") is not True
        or not isinstance(events, list)
        or len(events) != EXPECTED_EVENTS
    ):
        raise SystemExit("perception input is not the frozen evaluation set")
    require_audit(cfg.semantic_results, "temporal_semantic")
    require_audit(cfg.hand_results, "raw_hand_metric")
    require_audit(cfg.perception_results, "segmentation_object_state")
    require_audit(cfg.identity_results)
    expected_ids = {event["event_id"] for event in events}
    if len(expected_ids) != EXPECTED_EVENTS:
        raise SystemExit("evaluation event ids are not unique")

    records = []
    for ordinal, event in enumerate(events, start=1):
        event_id = event["event_id"]
        semantic_document = load(cfg.semantic_results / "events" / f"{event_id}.json")
        candidate = semantic_document.get("candidate") or {}
        hand_document = load(cfg.hand_results / "events" / f"{event_id}.json")
        segmentation_document = load(cfg.perception_results / "events" / event_id / "segmentation.json")
        object_document = load(cfg.perception_results / "events" / event_id / "object_state.json")
        binding = load(cfg.identity_results / "events" / f"{event_id}.json")
        depths_m = np.stack([np.load(workspace / relative, allow_pickle=False) for relative in event["object_state"]["ordered_depth_frames"]]).astype(np.float32, copy=False)
        intrinsic = np.load(workspace / event["object_state"]["intrinsic_path"], allow_pickle=False).astype(np.float64, copy=False)
        hands = {}
        for side in ("left", "right"):
            hand, hand_reason = item_array(cfg.hand_results, hand_document, side)
            pairs = {}
            for role in ("tool", "target"):
                mask, mask_reason = item_array(cfg.perception_results, segmentation_document, role)
                poses, pose_reason = item_array(cfg.perception_results, object_document, role)
                pairs[role] = pair_evidence_m(hand, mask, poses, depths_m, intrinsic, [hand_reason, mask_reason, pose_reason])
            hands[side] = {"qwen": candidate.get(side) or {"role": "unknown", "contact": "unknown"}, "pairs": pairs}
        records.append({
            "event_id": event_id,
            "vlm_coarse_label": candidate,
            "binding": {"role_to_entity": binding.get("role_to_entity"), "source": binding.get("source"), "is_ground_truth": False},
            "hands": hands,
            "human_targets_used": False,
            "evaluation_results_opened": True,
        })
        print(f"[fusion] {ordinal}/{EXPECTED_EVENTS} {event_id}", flush=True)
    if {record["event_id"] for record in records} != expected_ids:
        raise SystemExit("fusion records are not aligned one-to-one")
    cfg.output.mkdir(parents=True, exist_ok=False)
    atomic_json(cfg.output / "data.json", {
        "schema": "quiethand.m3_5.evaluation_fusion.v1",
        "status": "COMPLETE",
        "event_count": EXPECTED_EVENTS,
        "evaluation_event_count": EXPECTED_EVENTS,
        "records": records,
    })
    pair_states = [record["hands"][side]["pairs"][role]["contact"]["state"] for record in records for side in ("left", "right") for role in ("tool", "target")]
    atomic_json(cfg.output / "audit.json", {
        "schema": "quiethand.m3_5.evaluation_fusion_audit.v1",
        "status": "COMPLETE",
        "event_count": EXPECTED_EVENTS,
        "evaluation_event_count": EXPECTED_EVENTS,
        "explicit_pair_state_count": len(pair_states),
        "contact_state_counts": {state: pair_states.count(state) for state in sorted(set(pair_states))},
        "thresholds": {
            "close_m": CLOSE_M,
            "near_m": NEAR_M,
            "minimum_evidence_frames": MIN_EVIDENCE_FRAMES,
            "minimum_close_run": MIN_CLOSE_RUN,
            "stable_relative_span_m": STABLE_SPAN_M,
            "surface_points_per_frame_max": MAX_SURFACE_POINTS,
        },
        "human_targets_used_for_fusion": False,
        "training_performed": False,
        "evaluation_results_opened": True,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
