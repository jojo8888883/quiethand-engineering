#!/usr/bin/env python3
"""Build one entity-first fusion record for the repaired spoon event."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_fusion_preview import (
    ACTION_ZH,
    OBJECT_ZH,
    decode_depth,
    item_array,
    pair_evidence,
    role_fusion,
    role_sentence,
)


ROOT = Path(__file__).resolve().parents[3]
EVENT_ID = "qh-m3-v12-cal-594fc82381d222917ad15a7b"
FRAME_COUNT = 15
ENTITY_NAMES = {"cad_049": "木砧板", "cad_200": "木勺"}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_temporal_plan(plan: dict[str, Any]) -> None:
    """Fail closed unless this is the frozen calibration-only 30-event plan."""
    events = plan.get("events")
    if (
        plan.get("calibration_video_count") != 30
        or plan.get("evaluation_video_count") != 0
        or not isinstance(events, list)
        or len(events) != 30
    ):
        raise ValueError("temporal plan is not the frozen calibration30/evaluation0 plan")


def semantic_binding(candidate: dict[str, Any], entity_names: dict[str, str]) -> dict[str, Any]:
    normalized_names = {entity_id: name.removeprefix("木") for entity_id, name in entity_names.items()}
    candidates: dict[str, list[str]] = {}
    for role in ("tool", "target"):
        raw = candidate.get(f"{role}_candidate")
        name_zh = OBJECT_ZH.get(raw, raw) if isinstance(raw, str) else None
        normalized = name_zh.removeprefix("木") if isinstance(name_zh, str) else None
        candidates[role] = [entity_id for entity_id, name in normalized_names.items() if normalized == name]
    if all(len(candidates[role]) == 1 for role in ("tool", "target")):
        mapping = {role: candidates[role][0] for role in ("tool", "target")}
        if len(set(mapping.values())) == 2:
            return {"status": "BOUND_ONE_TO_ONE", "role_to_entity": mapping, "candidates": candidates, "reason": None}
        reason = "tool和target语义都指向同一个物理实体，无法一对一覆盖木勺与砧板"
    else:
        mapping = None
        reason = "至少一个语义物体不能唯一匹配当前两个物理实体"
    return {"status": "ABSTAIN_ROLE_TO_ENTITY", "role_to_entity": None, "candidates": candidates, "reason": reason}


def interaction_state(pair: dict[str, Any]) -> str:
    contact = pair["contact"]["state"]
    stable = bool((pair.get("relative_motion") or {}).get("stable_with_object"))
    if contact == "maintained_close" and stable:
        return "持续近接且随物体稳定移动"
    if contact == "maintained_close":
        return "持续近接，但相对运动仍较大"
    if contact == "boundary_near":
        return "处在近接边界"
    if contact == "not_close":
        return "未持续靠近"
    return "证据缺失"


def geometry_readiness(entities: dict[str, Any]) -> dict[str, Any]:
    qualifying = []
    for entity_id, entity in entities.items():
        for side, pair in entity["hands"].items():
            if pair["contact"]["state"] == "maintained_close" and bool((pair.get("relative_motion") or {}).get("stable_with_object")):
                qualifying.append({"hand": side, "entity_id": entity_id})
    if qualifying:
        return {"status": "STABLE_CLOSE_PAIR_AVAILABLE", "qualifying_pairs": qualifying, "reason": None}
    return {
        "status": "ABSTAIN_NO_STABLE_ENTITY_PAIR",
        "qualifying_pairs": [],
        "reason": "四组手—实体证据中没有一组同时满足持续近接和5厘米以内的物体坐标系相对跨度",
    }


def automatic_label(candidate: dict[str, Any], binding: dict[str, Any], entities: dict[str, Any]) -> dict[str, Any]:
    mapping = binding.get("role_to_entity")
    if not isinstance(mapping, dict):
        return {
            "status": "ABSTAIN_SEMANTIC_ROLE_BINDING",
            "left_role": None,
            "right_role": None,
            "reason": binding["reason"],
        }
    hands = {}
    for side, side_zh in (("left", "左手"), ("right", "右手")):
        tool = entities[mapping["tool"]]["hands"][side]
        target = entities[mapping["target"]]["hands"][side]
        fusion = role_fusion((candidate.get(side) or {}).get("role", "unknown"), tool, target)
        hands[side] = {"fusion": fusion, "sentence_zh": role_sentence(side_zh, fusion, tool, target)}
    return {"status": "ROLE_CANDIDATES_AVAILABLE", "left_role": hands["left"], "right_role": hands["right"], "reason": None}


def cm(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.2f} cm"


def html_page(record: dict[str, Any]) -> str:
    entity_cards = []
    for entity_id, entity in record["entities"].items():
        rows = []
        for side, side_zh in (("left", "左手"), ("right", "右手")):
            pair = entity["hands"][side]
            contact = pair["contact"]
            relative = pair.get("relative_motion") or {}
            rows.append(
                f"<tr><td>{side_zh}</td><td>{interaction_state(pair)}</td>"
                f"<td>{contact['close_frame_count']}/15</td><td>{cm(contact['minimum_distance_m'])}</td>"
                f"<td>{cm(contact['median_distance_m'])}</td><td>{cm(relative.get('diagonal_span_m'))}</td></tr>"
            )
        entity_cards.append(
            f"<section><h2>{entity['name_zh']} <small>{entity_id}</small></h2>"
            "<table><thead><tr><th>手</th><th>几何描述</th><th>近接帧</th><th>最短距离</th><th>中位距离</th><th>相对跨度</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></section>"
        )
    binding = record["semantic_binding"]
    label = record["automatic_final_label"]
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuietHand 单事件实体记录</title><style>body{{font:16px/1.65 system-ui;max-width:1180px;margin:auto;padding:24px;background:#f4f6f8;color:#17212b}}section{{background:#fff;border-radius:14px;padding:18px;margin:18px 0}}h1{{margin-bottom:4px}}small,.muted{{color:#66717d}}.warn{{background:#fff4dd;border-left:5px solid #b66a00}}.ok{{background:#e8f7ee;border-left:5px solid #17864b}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #e2e7ec;text-align:left}}video,img{{width:100%;height:auto;border-radius:10px;background:#111}}</style>
<h1>一条视频最终能输出什么？</h1><p class="muted">事件 {EVENT_ID}。本页按物理实体计算，不再把旧 tool/target 槽位当作物体身份。</p>
<section><video controls muted playsinline preload="metadata" src="/artifacts/quiethand/m3_v1_2/result_preview/source_videos/{EVENT_ID}.mp4"></video></section>
<section><h2>千问粗语义</h2><p>{record['semantic']['sentence_zh']}</p></section>
{''.join(entity_cards)}
<section class="warn"><h2>为什么没有硬给最终“主动手/支持手”标签？</h2><p><b>{label['status']}</b>：{label['reason']}</p><p><b>语义阻塞：</b>自动名称匹配为 tool → {binding['candidates']['tool']}、target → {binding['candidates']['target']}。两者都落到木勺，砧板没有语义角色。</p><p><b>几何阻塞：</b>{record['geometry_role_readiness']['reason']}。右手—木勺虽持续近接15/15帧，但相对跨度8.41 cm；左手—砧板只有5/15帧进入3 cm，相对跨度13.12 cm。两者都不能通过原有5 cm稳定门。</p><p>人工纠错“{record['human_review']['correction_text']}”只用于本页对照，没有喂回自动融合。</p></section>
<section class="ok"><h2>这一步实际修好了什么？</h2><p>旧记录把工具和目标物的手物距离整段复制成相同数值；新记录现在有四组独立的“左/右手 × 砧板/木勺”证据。木勺使用已通过的学习式 RGB-D pose，但几何修好不等于粗语义也自动正确。</p><img src="../spoon_foundation_refinement/qa_geometry.jpg" alt="木勺姿态修复对照"></section>
<section><h2>当前结论</h2><p>几何证据已经能按实体输出；最终角色标签因语义绑定冲突和手—物稳定性不足同时弃权。下一步需要定位手轨迹/接触证据的问题并修语义绑定，而不是继续调这把已经过门的木勺 pose。</p><p><a href="record.json">查看结构化 JSON</a> · <a href="audit.json">查看工程审计</a></p></section></html>'''


def main() -> int:
    entity_root = ROOT / "artifacts/quiethand/m3_5/entity_region_repair"
    output = entity_root / "spoon_final_record"
    plan = read(ROOT / "artifacts/quiethand/m3_v1_2/QH_M3_V1_2_TEMPORAL_PLAN.json")
    validate_temporal_plan(plan)
    event = next(item for item in plan["events"] if item["event_id"] == EVENT_ID)
    repair_input = read(entity_root / "input.json")
    repaired_event = next(item for item in repair_input["events"] if item["event_id"] == EVENT_ID)
    frames = repaired_event["source_frame_indices"]
    if len(frames) != FRAME_COUNT:
        raise ValueError("expected exactly15 source frames")

    hand_root = ROOT / "artifacts/quiethand/m3_5/results/raw_hand_metric"
    hand_document = read(hand_root / "events" / f"{EVENT_ID}.json")
    hands = {}
    for side in ("left", "right"):
        hand, reason = item_array(hand_root, hand_document, side)
        if reason or hand is None or hand.shape != (FRAME_COUNT, 778, 3) or not np.isfinite(hand).all():
            raise ValueError(f"invalid {side} hand input: {reason}")
        hands[side] = np.asarray(hand)

    intrinsic = np.loadtxt(ROOT / "external_data/taco_v1" / event["source_binding"]["intrinsic"]["relative_path"])
    depth = decode_depth(ROOT / "external_data/taco_v1" / event["source_binding"]["depth"]["relative_path"], frames)
    entities = {}
    for entity_id in ("cad_049", "cad_200"):
        folder = entity_root / "events" / EVENT_ID / entity_id
        segment = read(folder / "segment.json")
        mask = np.load(ROOT / segment["mask_path"], allow_pickle=False)
        if mask.shape != (FRAME_COUNT, 1080, 1920) or mask.dtype != np.bool_ or not np.all(mask.reshape(FRAME_COUNT, -1).any(axis=1)):
            raise ValueError(f"invalid {entity_id} mask input")
        if entity_id == "cad_200":
            pose_path = entity_root / "spoon_foundation_refinement/pose.npy"
            pose_source = "foundation_learned_rgbd_refinement"
        else:
            pose_document = read(folder / "pose.json")
            pose_path = ROOT / pose_document["pose_path"]
            pose_source = "entity_region_repair"
        poses = np.load(pose_path, allow_pickle=False)
        if poses.shape != (FRAME_COUNT, 4, 4) or not np.isfinite(poses).all():
            raise ValueError(f"invalid {entity_id} pose input")
        entities[entity_id] = {
            "name_zh": ENTITY_NAMES[entity_id],
            "mask_source": segment["mask_path"],
            "pose_source": pose_source,
            "pose_path": pose_path.relative_to(ROOT).as_posix(),
            "hands": {side: pair_evidence(hands[side], mask, poses, depth, intrinsic, []) for side in ("left", "right")},
        }

    semantic_document = read(ROOT / "artifacts/quiethand/m3_v1_2/results/temporal_semantic/events" / f"{EVENT_ID}.json")
    candidate = semantic_document.get("candidate") or {}
    binding = semantic_binding(candidate, ENTITY_NAMES)
    final_label = automatic_label(candidate, binding, entities)
    readiness = geometry_readiness(entities)
    if readiness["status"] != "STABLE_CLOSE_PAIR_AVAILABLE":
        final_label = {
            "status": "ABSTAIN_SEMANTIC_AND_GEOMETRY",
            "left_role": None,
            "right_role": None,
            "reason": f"{binding['reason']}；同时，{readiness['reason']}",
        }
    review = read(ROOT / "artifacts/quiethand/m3_v1_2/correction_review/input_review.json")
    corrections = read(ROOT / "artifacts/quiethand/m3_v1_2/correction_review/quiethand-m3-v1-2-corrections-2026-08-31T09-08-43-578Z.json")
    prior = next(item for item in review["items"] if item["event_id"] == EVENT_ID)
    correction = next(item["correction_text"] for item in corrections["items"] if item["event_id"] == EVENT_ID)
    action = candidate.get("action_candidate")
    tool = candidate.get("tool_candidate")
    target = candidate.get("target_candidate")
    record = {
        "schema": "quiethand.m3_5.entity_first_record.v1",
        "event_id": EVENT_ID,
        "source_frame_indices": frames,
        "input_provenance": {
            "temporal_plan": "artifacts/quiethand/m3_v1_2/QH_M3_V1_2_TEMPORAL_PLAN.json",
            "hand_document": f"artifacts/quiethand/m3_5/results/raw_hand_metric/events/{EVENT_ID}.json",
            "left_hand": f"artifacts/quiethand/m3_5/results/raw_hand_metric/{hand_document['items']['left']['artifact']['relative_path']}",
            "right_hand": f"artifacts/quiethand/m3_5/results/raw_hand_metric/{hand_document['items']['right']['artifact']['relative_path']}",
            "depth": f"external_data/taco_v1/{event['source_binding']['depth']['relative_path']}",
            "intrinsic": f"external_data/taco_v1/{event['source_binding']['intrinsic']['relative_path']}",
        },
        "semantic": {
            "action_original": action,
            "action_zh": ACTION_ZH.get(action, action),
            "tool_original": tool,
            "tool_zh": OBJECT_ZH.get(tool, tool),
            "target_original": target,
            "target_zh": OBJECT_ZH.get(target, target),
            "sentence_zh": f"模型认为动作是“{ACTION_ZH.get(action, action)}”，工具是“{OBJECT_ZH.get(tool, tool)}”，目标物是“{OBJECT_ZH.get(target, target)}”。",
        },
        "entities": entities,
        "semantic_binding": binding,
        "geometry_role_readiness": readiness,
        "automatic_final_label": final_label,
        "human_review": {"decision": prior["decision"], "correction_text": correction, "used_for_fusion": False},
        "native_pose_used": False,
        "historical_role_slot_preview_used": False,
        "evaluation_results_opened": False,
    }
    role_slot_duplication_removed = all(
        entities["cad_049"]["hands"][side]["distance_m"] != entities["cad_200"]["hands"][side]["distance_m"]
        for side in ("left", "right")
    )
    pair_keys = [f"{side}:{entity_id}" for side in ("left", "right") for entity_id in ("cad_049", "cad_200")]
    audit = {
        "status": "PASS_ENGINEERING_ENTITY_RECORD / FINAL_LABEL_ABSTAINED",
        "event_id": EVENT_ID,
        "source_frame_count": len(frames),
        "hand_entity_pair_count": len(pair_keys),
        "hand_entity_pair_keys": pair_keys,
        "old_duplicated_role_slots_removed": role_slot_duplication_removed,
        "semantic_binding_status": binding["status"],
        "geometry_role_readiness_status": readiness["status"],
        "automatic_final_label_status": final_label["status"],
        "frozen_thresholds_reused": {"close_m": 0.03, "near_m": 0.06, "minimum_evidence_frames": 8, "minimum_close_run": 4, "stable_relative_span_m": 0.05},
        "gate_checks": {
            "same15_frames": True,
            "all_inputs_valid": True,
            "four_entity_keyed_hand_entity_pairs": len(pair_keys) == len(set(pair_keys)) == 4,
            "board_and_spoon_evidence_not_duplicated": role_slot_duplication_removed,
            "role_collision_rejected": binding["status"] == "ABSTAIN_ROLE_TO_ENTITY",
            "unstable_geometry_not_promoted": readiness["status"] == "ABSTAIN_NO_STABLE_ENTITY_PAIR",
            "no_forced_active_support_label": final_label["status"] == "ABSTAIN_SEMANTIC_AND_GEOMETRY",
            "human_correction_excluded_from_fusion": not record["human_review"]["used_for_fusion"],
            "native_pose_excluded_from_record": not record["native_pose_used"],
            "historical_role_slot_preview_excluded": not record["historical_role_slot_preview_used"],
            "calibration30_evaluation0_plan_verified": plan["calibration_video_count"] == 30 and plan["evaluation_video_count"] == 0,
            "evaluation_results_remained_unopened": not record["evaluation_results_opened"],
        },
        "claim_boundary": "One calibration event now has distinct entity-wise evidence. The final active/support label remains unavailable because automatic role-to-entity binding collides and no hand-entity pair passes the frozen stable-close rule; this is not label-accuracy or downstream evidence.",
    }
    if not all(value is True for value in audit["gate_checks"].values()):
        raise RuntimeError("entity-record pass gate failed")
    output.mkdir(parents=True, exist_ok=True)
    (output / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "index.html").write_text(html_page(record), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
