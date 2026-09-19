"""Frozen M3 calibration-task and adapter-result contracts.

This module prepares deterministic, calibration-only work without importing or
running any perception model.  Source paths live in an orchestration-only
binding; every adapter-visible payload uses opaque event paths so TACO action,
tool and target metadata cannot leak into the semantic model input.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping, Sequence

from .taco_contract import TacoSplitEntry


M3_CHECKPOINT_CAP_BYTES = 30 * 1024**3
SEMANTIC_FRAME_OFFSETS = (-7, -3, 0, 3, 7)
WINDOW_FRAME_OFFSETS = tuple(range(-7, 8))
SEMANTIC_PROMPT = """You receive five ordered frames from a 0.5-second egocentric bimanual manipulation window.
Use only visible evidence. Do not infer a support hand merely because it moves slowly.
An active hand directly executes the manipulation, usually through the tool.
A support hand restrains or stabilizes the target object or a part of it.
Return JSON only with this schema:
{"action_candidate":string|null,"tool_candidate":string|null,"target_candidate":string|null,
"left":{"role":"active|support|both|neither|unknown","contact":"tool|target|both|neither|unknown"},
"right":{"role":"active|support|both|neither|unknown","contact":"tool|target|both|neither|unknown"},
"first_frame_boxes_0_1000":{"tool":[x1,y1,x2,y2]|null,"target":[x1,y1,x2,y2]|null},
"support_part_candidate":string|null,"evidence":"visible|ambiguous|not_visible"}
Coordinates must be integers in [0,1000] with x1<x2 and y1<y2. Use null/unknown when evidence is insufficient."""
SEMANTIC_PROMPT_SHA256 = hashlib.sha256(SEMANTIC_PROMPT.encode("utf-8")).hexdigest()

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EVENT_ID = re.compile(r"^qh-m3-cal-[0-9a-f]{24}-e0[1-5]$")
_ROLE_VALUES = {"active", "support", "both", "neither", "unknown"}
_CONTACT_VALUES = {"tool", "target", "both", "neither", "unknown"}
_EVIDENCE_VALUES = {"visible", "ambiguous", "not_visible"}


class M3AdapterContractError(RuntimeError):
    """An M3 task, model result, or checkpoint manifest violates protocol."""


def _exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise M3AdapterContractError(
            f"{label} fields differ: missing={sorted(keys-set(value))}, "
            f"extra={sorted(set(value)-keys)}"
        )


def _strict_json(text: str) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result = {}
        for key, value in values:
            if key in result:
                raise M3AdapterContractError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise M3AdapterContractError(f"non-finite JSON constant: {value}")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=reject_constant)
    except (json.JSONDecodeError, TypeError) as exc:
        raise M3AdapterContractError("semantic output is not strict JSON") from exc


def _nullable_string(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise M3AdapterContractError(f"{label} must be string or null")
    return value


def _box(value: object, label: str) -> list[int] | None:
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise M3AdapterContractError(f"{label} must be four integer coordinates or null")
    x1, y1, x2, y2 = value
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        raise M3AdapterContractError(f"{label} coordinates are outside contract")
    return value


def validate_semantic_candidate(text: str) -> dict[str, object]:
    """Parse exactly one frozen Qwen semantic candidate; never repair output."""

    value = _strict_json(text)
    if not isinstance(value, dict):
        raise M3AdapterContractError("semantic output must be one JSON object")
    _exact_keys(
        value,
        {
            "action_candidate",
            "tool_candidate",
            "target_candidate",
            "left",
            "right",
            "first_frame_boxes_0_1000",
            "support_part_candidate",
            "evidence",
        },
        "semantic output",
    )
    result: dict[str, object] = {}
    for key in (
        "action_candidate",
        "tool_candidate",
        "target_candidate",
        "support_part_candidate",
    ):
        result[key] = _nullable_string(value[key], key)
    for side in ("left", "right"):
        hand = value[side]
        if not isinstance(hand, dict):
            raise M3AdapterContractError(f"{side} must be an object")
        _exact_keys(hand, {"role", "contact"}, side)
        if hand["role"] not in _ROLE_VALUES or hand["contact"] not in _CONTACT_VALUES:
            raise M3AdapterContractError(f"{side} enum is invalid")
        result[side] = {"role": hand["role"], "contact": hand["contact"]}
    boxes = value["first_frame_boxes_0_1000"]
    if not isinstance(boxes, dict):
        raise M3AdapterContractError("first_frame_boxes_0_1000 must be an object")
    _exact_keys(boxes, {"tool", "target"}, "first_frame_boxes_0_1000")
    result["first_frame_boxes_0_1000"] = {
        "tool": _box(boxes["tool"], "tool box"),
        "target": _box(boxes["target"], "target box"),
    }
    if value["evidence"] not in _EVIDENCE_VALUES:
        raise M3AdapterContractError("semantic evidence enum is invalid")
    result["evidence"] = value["evidence"]
    return result


def opaque_calibration_event_id(sequence_id: str, event_index: int) -> str:
    if event_index not in range(1, 6):
        raise M3AdapterContractError("event index must be 1..5")
    digest = hashlib.sha256(
        f"QH-M3-CAL\0{sequence_id}\0{event_index}".encode("utf-8")
    ).hexdigest()[:24]
    return f"qh-m3-cal-{digest}-e{event_index:02d}"


def _receipt_record(
    completed: Mapping[str, Mapping[str, object]], suffix: str
) -> dict[str, object]:
    matches = [(key, value) for key, value in completed.items() if key.endswith(suffix)]
    if len(matches) != 1:
        raise M3AdapterContractError(f"expected one landed source ending {suffix}")
    path, value = matches[0]
    sha = value.get("sha256")
    size = value.get("bytes")
    if not isinstance(sha, str) or _SHA256.fullmatch(sha) is None:
        raise M3AdapterContractError("landed source lacks SHA-256")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise M3AdapterContractError("landed source lacks positive byte count")
    return {"relative_path": path, "sha256": sha, "bytes": size}


def _planned_frames(event_id: str, kind: str, count: int, suffix: str) -> list[str]:
    return [
        f"artifacts/quiethand/m3/calibration_inputs/{event_id}/{kind}_{index:02d}.{suffix}"
        for index in range(count)
    ]


def build_calibration_task_manifest(
    entries: Sequence[TacoSplitEntry],
    native_report: Mapping[str, object],
    completed: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Build exactly 150 opaque calibration tasks and no evaluation task."""

    calibration = [entry for entry in entries if entry.split == "calibration"]
    evaluation = [entry for entry in entries if entry.split == "evaluation"]
    if len(calibration) != 30 or len(evaluation) != 30:
        raise M3AdapterContractError("frozen 30/30 split is not present")
    geometry = native_report.get("native_contact_geometry")
    if (
        native_report.get("status") != "READY_NATIVE"
        or native_report.get("native_contact_geometry_status") != "CLOSED_300_EVENTS"
        or not isinstance(geometry, Mapping)
        or geometry.get("closed") is not True
    ):
        raise M3AdapterContractError("native diagnostic is not closed for planning")
    events = geometry.get("events")
    if not isinstance(events, list) or len(events) != 300:
        raise M3AdapterContractError("native report does not contain 300 events")
    by_key = {}
    for event in events:
        if not isinstance(event, Mapping):
            raise M3AdapterContractError("native event is not an object")
        key = (event.get("sequence_id"), event.get("event_index"))
        if key in by_key:
            raise M3AdapterContractError("native report contains a duplicate event")
        by_key[key] = event

    tasks = []
    for entry in calibration:
        prefix = f"/{entry.triplet}/{entry.sequence_name}/"
        sequence_records = {
            key: value for key, value in completed.items() if prefix in "/" + key
        }
        rgb = _receipt_record(sequence_records, "/color.mp4")
        depth = _receipt_record(sequence_records, "/egocentric_depth.avi")
        intrinsic = _receipt_record(sequence_records, "/egocentric_intrinsic.txt")
        extrinsic = _receipt_record(
            sequence_records, "/egocentric_frame_extrinsic.npy"
        )
        for event_index in range(1, 6):
            native = by_key.get((entry.sequence_id, event_index))
            if native is None or native.get("split") != "calibration":
                raise M3AdapterContractError("calibration native event is missing")
            object_ids = native.get("object_ids")
            if (
                not isinstance(object_ids, Mapping)
                or not isinstance(object_ids.get("tool"), str)
                or not isinstance(object_ids.get("target"), str)
            ):
                raise M3AdapterContractError("native event object IDs are invalid")
            tool_id = object_ids["tool"]
            target_id = object_ids["target"]
            tool_pose = _receipt_record(sequence_records, f"/tool_{tool_id}.npy")
            target_pose = _receipt_record(sequence_records, f"/target_{target_id}.npy")
            tool_mesh = _receipt_record(completed, f"/{tool_id}_cm.obj")
            target_mesh = _receipt_record(completed, f"/{target_id}_cm.obj")
            center = native.get("center_frame")
            frames = native.get("frame_indices")
            if (
                isinstance(center, bool)
                or not isinstance(center, int)
                or frames != list(range(center - 7, center + 8))
            ):
                raise M3AdapterContractError("native event frame window changed")
            event_id = opaque_calibration_event_id(entry.sequence_id, event_index)
            rgb_frames = _planned_frames(event_id, "rgb", 15, "png")
            depth_frames = _planned_frames(event_id, "depth", 15, "npy")
            semantic_indices = [offset + 7 for offset in SEMANTIC_FRAME_OFFSETS]
            task = {
                "event_id": event_id,
                "split": "calibration",
                "materialization_status": "PLANNED_NOT_MATERIALIZED",
                "orchestration_only_never_model_input": {
                    "rank": entry.rank,
                    "sequence_id": entry.sequence_id,
                    "event_index": event_index,
                    "center_frame": center,
                    "frame_indices": frames,
                    "source_binding": {
                        "rgb": rgb,
                        "depth": depth,
                        "intrinsic": intrinsic,
                        "extrinsic": extrinsic,
                        "tool_pose": tool_pose,
                        "target_pose": target_pose,
                        "tool_mesh": tool_mesh,
                        "target_mesh": target_mesh,
                    },
                },
                "adapter_visible": {
                    "semantic": {
                        "event_id": event_id,
                        "ordered_rgb_frames": [rgb_frames[index] for index in semantic_indices],
                        "frame_offsets": list(SEMANTIC_FRAME_OFFSETS),
                        "prompt": SEMANTIC_PROMPT,
                        "prompt_sha256": SEMANTIC_PROMPT_SHA256,
                        "do_sample": False,
                        "max_new_tokens": 512,
                    },
                    "raw_hand": {
                        "event_id": event_id,
                        "ordered_rgb_frames": rgb_frames,
                        "frame_offsets": list(WINDOW_FRAME_OFFSETS),
                        "coordinate_frame": "camera",
                        "infill_allowed": False,
                        "slam_allowed": False,
                    },
                    "segmentation": {
                        "event_id": event_id,
                        "ordered_rgb_frames": rgb_frames,
                        "box_source": "semantic.first_frame_boxes_0_1000",
                        "native_mask_allowed": False,
                    },
                    "object_state": {
                        "event_id": event_id,
                        "ordered_rgb_frames": rgb_frames,
                        "ordered_depth_frames": depth_frames,
                        "intrinsic_path": f"artifacts/quiethand/m3/calibration_inputs/{event_id}/intrinsic.npy",
                        "tool_mesh_path": f"artifacts/quiethand/m3/calibration_inputs/{event_id}/tool.obj",
                        "target_mesh_path": f"artifacts/quiethand/m3/calibration_inputs/{event_id}/target.obj",
                        "mask_source": "segmentation",
                        "coordinate_frame": "camera",
                    },
                },
            }
            visible = json.dumps(task["adapter_visible"], sort_keys=True)
            forbidden = [entry.sequence_id, entry.triplet, entry.sequence_name]
            if any(token and token in visible for token in forbidden):
                raise M3AdapterContractError("TACO metadata leaked into adapter-visible input")
            tasks.append(task)
    if len(tasks) != 150 or len({task["event_id"] for task in tasks}) != 150:
        raise M3AdapterContractError("calibration task coverage is not exactly 150")
    if any(task["split"] != "calibration" for task in tasks):
        raise M3AdapterContractError("evaluation task leaked into calibration plan")
    return {
        "schema": "quiethand.m3.calibration_task_plan.v1",
        "status": "READY_FOR_CALIBRATION_MATERIALIZATION",
        "task_count": 150,
        "calibration_sequence_count": 30,
        "evaluation_task_count": 0,
        "evaluation_model_results_opened": False,
        "semantic_prompt_sha256": SEMANTIC_PROMPT_SHA256,
        "tasks": tasks,
    }


def _hf_asset(path: str, size: int, lfs_sha256: str | None = None) -> dict[str, object]:
    result: dict[str, object] = {"path": path, "bytes": size}
    if lfs_sha256 is not None:
        result["expected_sha256"] = lfs_sha256
    return result


def build_checkpoint_plan() -> dict[str, object]:
    """Return the exact minimal frozen M3 profile without downloading bytes."""

    qwen_assets = [
        _hf_asset("chat_template.json", 5_499),
        _hf_asset("config.json", 1_474),
        _hf_asset("generation_config.json", 269),
        _hf_asset("merges.txt", 1_671_839),
        _hf_asset("model-00001-of-00004.safetensors", 4_902_275_944, "d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa"),
        _hf_asset("model-00002-of-00004.safetensors", 4_915_962_496, "8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5"),
        _hf_asset("model-00003-of-00004.safetensors", 4_999_831_048, "83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192"),
        _hf_asset("model-00004-of-00004.safetensors", 2_716_270_024, "0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5"),
        _hf_asset("model.safetensors.index.json", 67_759),
        _hf_asset("preprocessor_config.json", 390),
        _hf_asset("tokenizer.json", 7_032_403),
        _hf_asset("tokenizer_config.json", 10_868),
        _hf_asset("video_preprocessor_config.json", 385),
        _hf_asset("vocab.json", 2_776_833),
    ]
    hawor_assets = [
        _hf_asset("external/detector.pt", 53_582_271, "5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033"),
        _hf_asset("hawor/checkpoints/hawor.ckpt", 3_267_481_572, "4d1cc43853c190d6f2c10d9b6295c73109f0faf9ef41ac817a2b31d94b4823f2"),
        _hf_asset("hawor/model_config.yaml", 2_743),
    ]
    sam_assets = [
        _hf_asset("sam2.1_hiera_b+.yaml", 3_650),
        _hf_asset("sam2.1_hiera_base_plus.pt", 323_606_802, "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5"),
    ]
    adapters = {
        "semantic": {
            "repository": None,
            "checkpoint_provider": "huggingface",
            "checkpoint_id": "Qwen/Qwen3-VL-8B-Instruct",
            "resolved_revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
            "assets": qwen_assets,
        },
        "raw_hand": {
            "repository": "https://github.com/ThunderVVV/HaWoR",
            "repository_commit": "66c7d4108d58a716deccd192cb7645170cdc7bd7",
            "checkpoint_provider": "huggingface",
            "checkpoint_id": "ThunderVVV/HaWoR",
            "resolved_revision": "da6335f47f9806308992d5ae1002a4cc5f7252c2",
            "assets": hawor_assets,
            "excluded_upstream_assets": [
                "external/droid.pth",
                "external/metric_depth_vit_large_800k.pth",
                "hawor/checkpoints/infiller.pt",
            ],
            "reason": "M3 uses raw observed camera-space hand output only; SLAM, metric-depth scaling and infiller are prohibited",
        },
        "segmentation": {
            "repository": "https://github.com/facebookresearch/sam2",
            "repository_commit": "2b90b9f5ceec907a1c18123530e92e794ad901a4",
            "checkpoint_provider": "huggingface",
            "checkpoint_id": "facebook/sam2.1-hiera-base-plus",
            "resolved_revision": "b7320756a13354e7530a63935656d35b2f91a290",
            "assets": sam_assets,
            "excluded_upstream_assets": ["model.safetensors"],
            "reason": "the official SAM2 repository consumes the .pt checkpoint; a duplicate Transformers-format weight is not part of this profile",
        },
        "object_state": {
            "repository": "https://github.com/NVlabs/FoundationPose",
            "repository_commit": "a1b694b83e633c2cb6115b9063d940a687759392",
            "checkpoint_provider": "google_drive",
            "checkpoint_id": "1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i",
            "resolved_revision": "google_drive_folder:1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i",
            "required_assets": [
                "2023-10-28-18-33-37/config.yml",
                "2023-10-28-18-33-37/model_best.pth",
                "2024-01-11-20-02-45/config.yml",
                "2024-01-11-20-02-45/model_best.pth",
            ],
        },
    }
    repositories = {
        "raw_hand": {
            "relative_path": "_repositories/HaWoR-66c7d4108d58a716deccd192cb7645170cdc7bd7.tar.gz",
            "source_url": "https://github.com/ThunderVVV/HaWoR/archive/66c7d4108d58a716deccd192cb7645170cdc7bd7.tar.gz",
            "resolved_revision": "66c7d4108d58a716deccd192cb7645170cdc7bd7",
        },
        "segmentation": {
            "relative_path": "_repositories/sam2-2b90b9f5ceec907a1c18123530e92e794ad901a4.tar.gz",
            "source_url": "https://github.com/facebookresearch/sam2/archive/2b90b9f5ceec907a1c18123530e92e794ad901a4.tar.gz",
            "resolved_revision": "2b90b9f5ceec907a1c18123530e92e794ad901a4",
        },
        "object_state": {
            "relative_path": "_repositories/FoundationPose-a1b694b83e633c2cb6115b9063d940a687759392.tar.gz",
            "source_url": "https://github.com/NVlabs/FoundationPose/archive/a1b694b83e633c2cb6115b9063d940a687759392.tar.gz",
            "resolved_revision": "a1b694b83e633c2cb6115b9063d940a687759392",
        },
    }
    known_bytes = sum(
        int(asset["bytes"])
        for adapter in adapters.values()
        for asset in adapter.get("assets", [])
    )
    if known_bytes >= M3_CHECKPOINT_CAP_BYTES:
        raise M3AdapterContractError("known checkpoint bytes already exceed cap")
    return {
        "schema": "quiethand.m3.checkpoint_plan.v1",
        "status": "PLAN_ONLY_NO_BYTES_DOWNLOADED",
        "adapters": adapters,
        "repository_archives": repositories,
        "known_checkpoint_bytes": known_bytes,
        "checkpoint_and_repo_cap_bytes": M3_CHECKPOINT_CAP_BYTES,
        "remaining_bytes_before_unversioned_foundationpose_and_repositories": M3_CHECKPOINT_CAP_BYTES - known_bytes,
        "landing_rule": "every landed file requires source URL, resolved revision, exact bytes and local SHA-256 before inference",
    }


def validate_artifact_reference(
    value: Mapping[str, object],
    *,
    expected_shape_tail: tuple[int, ...],
    allowed_dtypes: set[str],
) -> dict[str, object]:
    _exact_keys(
        value,
        {"relative_path", "sha256", "bytes", "dtype", "shape", "frame_indices", "unit", "frame_id"},
        "artifact reference",
    )
    path = value["relative_path"]
    if not isinstance(path, str):
        raise M3AdapterContractError("artifact path must be a string")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise M3AdapterContractError("artifact path must be safe and relative")
    sha = value["sha256"]
    if not isinstance(sha, str) or _SHA256.fullmatch(sha) is None:
        raise M3AdapterContractError("artifact SHA-256 is invalid")
    size = value["bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise M3AdapterContractError("artifact byte count must be positive")
    dtype = value["dtype"]
    if dtype not in allowed_dtypes:
        raise M3AdapterContractError("artifact dtype is invalid")
    shape = value["shape"]
    frames = value["frame_indices"]
    if (
        not isinstance(shape, list)
        or len(shape) != 1 + len(expected_shape_tail)
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape)
        or tuple(shape[1:]) != expected_shape_tail
        or not isinstance(frames, list)
        or len(frames) != shape[0]
        or len(set(frames)) != len(frames)
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in frames)
    ):
        raise M3AdapterContractError("artifact shape/frame contract is invalid")
    if not isinstance(value["unit"], str) or not isinstance(value["frame_id"], str):
        raise M3AdapterContractError("artifact unit/frame_id must be strings")
    return dict(value)


def validate_itemized_adapter_result(
    value: Mapping[str, object], *, adapter: str, event_id: str
) -> dict[str, object]:
    """Validate non-semantic adapter outputs with explicit per-item abstention."""

    profiles = {
        "raw_hand": ({"left", "right"}, (778, 3), {"float32"}, "m", "camera"),
        "segmentation": ({"tool", "target"}, (1080, 1920), {"bool", "uint8"}, "binary", "image"),
        "object_state": ({"tool", "target"}, (4, 4), {"float32", "float64"}, "m_SE3", "camera"),
    }
    if adapter not in profiles or _EVENT_ID.fullmatch(event_id) is None:
        raise M3AdapterContractError("adapter or event ID is outside contract")
    _exact_keys(value, {"schema", "event_id", "adapter", "items"}, "adapter result")
    if (
        value["schema"] != "quiethand.m3.adapter_result.v1"
        or value["event_id"] != event_id
        or value["adapter"] != adapter
        or not isinstance(value["items"], Mapping)
    ):
        raise M3AdapterContractError("adapter result envelope is invalid")
    expected, shape_tail, dtypes, unit, frame_id = profiles[adapter]
    items = value["items"]
    _exact_keys(items, expected, "adapter result items")
    normalized = {}
    for name, item in items.items():
        if not isinstance(item, Mapping):
            raise M3AdapterContractError("adapter item must be an object")
        _exact_keys(item, {"status", "artifact", "failure_reason"}, "adapter item")
        status = item["status"]
        artifact = item["artifact"]
        reason = item["failure_reason"]
        if status == "observed":
            if not isinstance(artifact, Mapping) or reason is not None:
                raise M3AdapterContractError("observed item requires artifact and no failure")
            parsed = validate_artifact_reference(
                artifact, expected_shape_tail=shape_tail, allowed_dtypes=dtypes
            )
            if parsed["unit"] != unit or parsed["frame_id"] != frame_id:
                raise M3AdapterContractError("artifact unit or coordinate frame is invalid")
            normalized[name] = {"status": status, "artifact": parsed, "failure_reason": None}
        elif status in {"abstain", "invalid"}:
            if artifact is not None or not isinstance(reason, str) or not reason.strip():
                raise M3AdapterContractError("unusable item requires null artifact and reason")
            normalized[name] = dict(item)
        else:
            raise M3AdapterContractError("adapter item status is invalid")
    return {
        "schema": value["schema"],
        "event_id": event_id,
        "adapter": adapter,
        "items": normalized,
    }


def _expected_landed_resources(plan: Mapping[str, object]) -> dict[tuple[str, str], dict[str, object]]:
    expected: dict[tuple[str, str], dict[str, object]] = {}
    adapters = plan["adapters"]
    if not isinstance(adapters, Mapping):
        raise M3AdapterContractError("checkpoint plan adapters are invalid")
    for adapter, profile in adapters.items():
        if not isinstance(adapter, str) or not isinstance(profile, Mapping):
            raise M3AdapterContractError("checkpoint plan profile is invalid")
        provider = profile.get("checkpoint_provider")
        checkpoint_id = profile.get("checkpoint_id")
        revision = profile.get("resolved_revision")
        assets = profile.get("assets", profile.get("required_assets"))
        if (
            provider not in {"huggingface", "google_drive"}
            or not isinstance(checkpoint_id, str)
            or not isinstance(revision, str)
            or not isinstance(assets, list)
        ):
            raise M3AdapterContractError("checkpoint plan source is invalid")
        for asset in assets:
            details = dict(asset) if isinstance(asset, Mapping) else {"path": asset}
            relative_asset = details.get("path")
            if not isinstance(relative_asset, str):
                raise M3AdapterContractError("checkpoint plan asset path is invalid")
            relative_path = f"{adapter}/{relative_asset}"
            if provider == "huggingface":
                source_url = (
                    f"https://huggingface.co/{checkpoint_id}/resolve/"
                    f"{revision}/{relative_asset}"
                )
            else:
                source_url = f"https://drive.google.com/drive/folders/{checkpoint_id}"
            expected[(adapter, relative_path)] = {
                "asset_class": "checkpoint",
                "source_url": source_url,
                "resolved_revision": revision,
                **{key: value for key, value in details.items() if key != "path"},
            }
    repositories = plan.get("repository_archives")
    if not isinstance(repositories, Mapping):
        raise M3AdapterContractError("checkpoint plan repositories are invalid")
    for adapter, source in repositories.items():
        if not isinstance(adapter, str) or not isinstance(source, Mapping):
            raise M3AdapterContractError("repository plan row is invalid")
        relative_path = source.get("relative_path")
        if not isinstance(relative_path, str):
            raise M3AdapterContractError("repository archive path is invalid")
        expected[(adapter, relative_path)] = {
            "asset_class": "repository_archive",
            "source_url": source.get("source_url"),
            "resolved_revision": source.get("resolved_revision"),
        }
    return expected


def _regular_file_without_symlink(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise M3AdapterContractError("checkpoint path must be safe and relative")
    if root.is_symlink() or not root.is_dir():
        raise M3AdapterContractError("checkpoint root is missing or symlinked")
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise M3AdapterContractError("checkpoint path contains a symlink")
    if not current.is_file():
        raise M3AdapterContractError("checkpoint file is missing")
    return current


def _stable_file_sha256(path: Path) -> tuple[int, str]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise M3AdapterContractError("checkpoint asset is not a nonempty regular file")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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
        raise M3AdapterContractError("checkpoint file changed during verification")
    return before.st_size, digest.hexdigest()


def verify_landed_checkpoint_manifest(
    manifest: Mapping[str, object], root: Path
) -> dict[str, object]:
    """Fail closed on any missing, extra, symlinked, oversized, or changed asset."""

    plan = build_checkpoint_plan()
    _exact_keys(manifest, {"schema", "files"}, "checkpoint manifest")
    if manifest["schema"] != "quiethand.m3.checkpoint_landing.v1" or not isinstance(
        manifest["files"], list
    ):
        raise M3AdapterContractError("checkpoint landing envelope is invalid")
    rows = manifest["files"]
    expected = _expected_landed_resources(plan)
    if not rows:
        raise M3AdapterContractError("checkpoint landing manifest is empty")
    indexed: dict[tuple[object, object], Mapping[str, object]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise M3AdapterContractError("checkpoint row must be an object")
        _exact_keys(
            row,
            {
                "adapter",
                "asset_class",
                "relative_path",
                "source_url",
                "resolved_revision",
                "bytes",
                "sha256",
            },
            "checkpoint row",
        )
        adapter = row["adapter"]
        relative = row["relative_path"]
        if adapter not in plan["adapters"] or not isinstance(relative, str):
            raise M3AdapterContractError("checkpoint adapter/path is invalid")
        key = (adapter, relative)
        if key in indexed:
            raise M3AdapterContractError("duplicate checkpoint row")
        if key not in expected:
            raise M3AdapterContractError("unexpected checkpoint or repository file")
        requirement = expected[key]
        for field in ("asset_class", "source_url", "resolved_revision"):
            if row[field] != requirement[field]:
                raise M3AdapterContractError(f"checkpoint {field} differs from plan")
        size = row["bytes"]
        sha = row["sha256"]
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(sha, str)
            or _SHA256.fullmatch(sha) is None
        ):
            raise M3AdapterContractError("checkpoint size/SHA declaration is invalid")
        expected_size = requirement.get("bytes")
        expected_sha = requirement.get("expected_sha256")
        if expected_size is not None and size != expected_size:
            raise M3AdapterContractError("checkpoint byte count differs from plan")
        if expected_sha is not None and sha != expected_sha:
            raise M3AdapterContractError("checkpoint SHA-256 differs from plan")
        indexed[key] = row
    if set(indexed) != set(expected):
        raise M3AdapterContractError("checkpoint landing file set is incomplete")
    total = 0
    for key in sorted(indexed):
        row = indexed[key]
        relative = row["relative_path"]
        if not isinstance(relative, str):
            raise M3AdapterContractError("checkpoint path is invalid")
        path = _regular_file_without_symlink(root, relative)
        actual_size, actual_sha = _stable_file_sha256(path)
        if actual_size != row["bytes"] or actual_sha != row["sha256"]:
            raise M3AdapterContractError("checkpoint file size or SHA-256 changed")
        size = row["bytes"]
        if isinstance(size, bool) or not isinstance(size, int):
            raise M3AdapterContractError("checkpoint byte count is invalid")
        total += size
    if total > M3_CHECKPOINT_CAP_BYTES:
        raise M3AdapterContractError("checkpoint and repository cap exceeded")
    return {"closed": True, "file_count": len(rows), "total_bytes": total}
