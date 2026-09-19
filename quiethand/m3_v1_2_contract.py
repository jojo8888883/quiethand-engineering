"""QuietHand M3-v1.2 complete-action semantic contract."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


SAMPLE_COUNT = 8
GEOMETRY_FRAME_COUNT = 15
PROMPT = """You receive eight ordered RGB frames sampled uniformly from one complete egocentric bimanual-manipulation video.
Use only visible evidence across the full sequence. Identify the inclusive sample interval that contains the purposeful manipulation; exclude stationary preparation before it and idle/withdrawal after it. Sample indices are 0 through 7.
An active hand directly executes the manipulation, usually through the tool. A support hand restrains or stabilizes the target object or a part of it. Slow motion alone is not evidence of support.
Return JSON only with this schema:
{"action_start_sample":integer|null,"action_end_sample":integer|null,
"action_candidate":string|null,"tool_candidate":string|null,"target_candidate":string|null,
"left":{"role":"active|support|both|neither|unknown","contact":"tool|target|both|neither|unknown"},
"right":{"role":"active|support|both|neither|unknown","contact":"tool|target|both|neither|unknown"},
"start_sample_boxes_0_1000":{"tool":[x1,y1,x2,y2]|null,"target":[x1,y1,x2,y2]|null},
"support_part_candidate":string|null,"evidence":"visible|ambiguous|not_visible"}
For visible or ambiguous evidence, start/end must be integers with 0<=start<end<=7. Boxes refer only to the selected start-sample image. Coordinates are integers in [0,1000] with x1<x2 and y1<y2. If no purposeful manipulation is visible, use null start/end/action and evidence not_visible. Use null/unknown when other evidence is insufficient."""
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()

ROLE_VALUES = {"active", "support", "both", "neither", "unknown"}
CONTACT_VALUES = {"tool", "target", "both", "neither", "unknown"}
EVIDENCE_VALUES = {"visible", "ambiguous", "not_visible"}


class M3V12ContractError(RuntimeError):
    pass


def _exact(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise M3V12ContractError(f"{label} fields differ")


def _strict_json(text: str) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise M3V12ContractError(f"duplicate key: {key}")
            result[key] = value
        return result

    def reject(value: str) -> object:
        raise M3V12ContractError(f"non-finite constant: {value}")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=reject)
    except (json.JSONDecodeError, TypeError) as exc:
        raise M3V12ContractError("output is not strict JSON") from exc


def _nullable_string(value: object, label: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise M3V12ContractError(f"{label} must be string or null")
    return value


def _box(value: object, label: str) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4 or any(isinstance(x, bool) or not isinstance(x, int) for x in value):
        raise M3V12ContractError(f"{label} must be four integers or null")
    x1, y1, x2, y2 = value
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        raise M3V12ContractError(f"{label} is outside 0..1000")
    return value


def validate_candidate(text: str) -> dict[str, object]:
    value = _strict_json(text)
    if not isinstance(value, dict):
        raise M3V12ContractError("output must be one object")
    _exact(value, {
        "action_start_sample", "action_end_sample", "action_candidate",
        "tool_candidate", "target_candidate", "left", "right",
        "start_sample_boxes_0_1000", "support_part_candidate", "evidence",
    }, "semantic output")
    evidence = value["evidence"]
    if evidence not in EVIDENCE_VALUES:
        raise M3V12ContractError("evidence enum is invalid")
    start = value["action_start_sample"]
    end = value["action_end_sample"]
    if evidence == "not_visible":
        if start is not None or end is not None or value["action_candidate"] is not None:
            raise M3V12ContractError("not_visible must have null interval and action")
    elif (
        isinstance(start, bool) or not isinstance(start, int)
        or isinstance(end, bool) or not isinstance(end, int)
        or not 0 <= start < end < SAMPLE_COUNT
    ):
        raise M3V12ContractError("visible interval must satisfy 0<=start<end<=7")
    result: dict[str, object] = {
        "action_start_sample": start,
        "action_end_sample": end,
        "evidence": evidence,
    }
    for key in ("action_candidate", "tool_candidate", "target_candidate", "support_part_candidate"):
        result[key] = _nullable_string(value[key], key)
    for side in ("left", "right"):
        hand = value[side]
        if not isinstance(hand, dict):
            raise M3V12ContractError(f"{side} must be an object")
        _exact(hand, {"role", "contact"}, side)
        if hand["role"] not in ROLE_VALUES or hand["contact"] not in CONTACT_VALUES:
            raise M3V12ContractError(f"{side} enum is invalid")
        result[side] = {"role": hand["role"], "contact": hand["contact"]}
    boxes = value["start_sample_boxes_0_1000"]
    if not isinstance(boxes, dict):
        raise M3V12ContractError("boxes must be an object")
    _exact(boxes, {"tool", "target"}, "boxes")
    result["start_sample_boxes_0_1000"] = {
        "tool": _box(boxes["tool"], "tool box"),
        "target": _box(boxes["target"], "target box"),
    }
    return result
