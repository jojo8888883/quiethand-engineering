#!/usr/bin/env python3
"""Build grouped left/right-swapped views from creator-reviewed role labels."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import html
import json
from pathlib import Path
import re
from typing import Any


LABEL_FIELDS = {"action", "tool", "target", "left", "right"}
HAND_FIELDS = {"contact", "role"}
TRACKED_FIELDS = (
    "action",
    "tool",
    "target",
    "left.contact",
    "left.role",
    "right.contact",
    "right.role",
)
SIDE_TEXT = re.compile(r"(?:\b(?:left|right)(?:\s+hand)?\b|左手|右手)", re.IGNORECASE)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def field(label: dict[str, Any], name: str) -> Any:
    value: Any = label
    for key in name.split("."):
        value = value[key]
    return value


def changed_fields(original: dict[str, Any], corrected: dict[str, Any]) -> list[str]:
    return [name for name in TRACKED_FIELDS if field(original, name) != field(corrected, name)]


def validate_label(label: dict[str, Any]) -> None:
    if set(label) != LABEL_FIELDS:
        raise ValueError("label fields changed")
    for name in ("action", "tool", "target"):
        if not isinstance(label[name], str) or not label[name].strip():
            raise ValueError(f"empty {name}")
        if SIDE_TEXT.search(label[name]):
            raise ValueError(f"{name} explicitly binds a hand side: {label[name]}")
    for side in ("left", "right"):
        if not isinstance(label[side], dict) or set(label[side]) != HAND_FIELDS:
            raise ValueError(f"{side} fields changed")
        if label[side]["contact"] not in {"tool", "target", "none", "uncertain"}:
            raise ValueError(f"invalid {side} contact")
        if label[side]["role"] not in {"active", "support", "none", "uncertain"}:
            raise ValueError(f"invalid {side} role")


def swap_label(label: dict[str, Any]) -> dict[str, Any]:
    validate_label(label)
    return {
        "action": label["action"],
        "tool": label["tool"],
        "target": label["target"],
        "left": copy.deepcopy(label["right"]),
        "right": copy.deepcopy(label["left"]),
    }


def swap_changed_fields(changed_fields: list[str]) -> list[str]:
    swapped = set()
    for name in changed_fields:
        if name not in TRACKED_FIELDS:
            raise ValueError(f"unsupported changed field: {name}")
        if name.startswith("left."):
            swapped.add("right." + name[len("left."):])
        elif name.startswith("right."):
            swapped.add("left." + name[len("right."):])
        else:
            swapped.add(name)
    return [name for name in TRACKED_FIELDS if name in swapped]


def validate_source(data: dict[str, Any]) -> None:
    if data.get("schema") != "quiethand.m3_5.role_correction_calibration.v1":
        raise ValueError("unexpected source schema")
    items = data.get("items")
    if not isinstance(items, list) or len(items) != 30:
        raise ValueError("expected exactly 30 source records")
    if data.get("evaluation_results_opened") is not False or data.get("training_performed") is not False:
        raise ValueError("source evaluation/training boundary changed")
    if data.get("active_side_counts") != {"left": 2, "right": 28}:
        raise ValueError("source active-side distribution changed")

    seen = set()
    for item in items:
        event_id = item.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in seen:
            raise ValueError("invalid or duplicate source event id")
        seen.add(event_id)
        if item.get("evaluation") is not False or item.get("training_use_status") != "not_authorized":
            raise ValueError(f"source boundary changed: {event_id}")
        validate_label(item["original_label"])
        validate_label(item["calibration_label"])
        expected_changed = changed_fields(item["original_label"], item["calibration_label"])
        if item.get("changed_fields") != expected_changed:
            raise ValueError(f"changed_fields disagree with labels: {event_id}")


def make_view(item: dict[str, Any], swapped: bool) -> dict[str, Any]:
    event_id = item["event_id"]
    view = "hand_swapped" if swapped else "original"
    transform = swap_label if swapped else copy.deepcopy
    changed_fields = (
        swap_changed_fields(item["changed_fields"])
        if swapped else copy.deepcopy(item["changed_fields"])
    )
    return {
        "view_id": f"{event_id}:{view}",
        "source_event_id": event_id,
        "split_group": event_id,
        "view": view,
        "synthetic_symmetry_view": swapped,
        "creator_reviewed_source": not swapped,
        "independent_new_label": False,
        "source_review_decision": item["review_decision"],
        "source_label_source": item["label_source"],
        "derived_label_source": (
            "left_right_swap_of_creator_reviewed_source"
            if swapped else "creator_reviewed_source"
        ),
        "correction_text": item["correction_text"],
        "correction_text_usage": "provenance_only_excluded_from_symmetric_training",
        "original_label": transform(item["original_label"]),
        "calibration_label": transform(item["calibration_label"]),
        "changed_fields": changed_fields,
        "evaluation": False,
        "training_use_status": "not_authorized",
    }


def build_dataset(source: dict[str, Any]) -> dict[str, Any]:
    validate_source(source)
    views = []
    for item in source["items"]:
        views.extend((make_view(item, False), make_view(item, True)))

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for view in views:
        groups[view["split_group"]].append(view)

    group_violations = []
    invariant_violations = []
    involution_violations = []
    changed_field_violations = []
    for group_id, pair in groups.items():
        if len(pair) != 2 or {item["view"] for item in pair} != {"original", "hand_swapped"}:
            group_violations.append(group_id)
            continue
        by_view = {item["view"]: item for item in pair}
        original = by_view["original"]
        swapped = by_view["hand_swapped"]
        for label_name in ("original_label", "calibration_label"):
            for name in ("action", "tool", "target"):
                if original[label_name][name] != swapped[label_name][name]:
                    invariant_violations.append(f"{group_id}:{label_name}:{name}")
            if swap_label(swapped[label_name]) != original[label_name]:
                involution_violations.append(f"{group_id}:{label_name}")
        if swap_changed_fields(swapped["changed_fields"]) != original["changed_fields"]:
            involution_violations.append(f"{group_id}:changed_fields")
        for view in pair:
            expected_changed = changed_fields(view["original_label"], view["calibration_label"])
            if view["changed_fields"] != expected_changed:
                changed_field_violations.append(view["view_id"])

    active_side_counts = Counter()
    for view in views:
        label = view["calibration_label"]
        if label["left"]["role"] == "active" and label["right"]["role"] != "active":
            active_side_counts["left"] += 1
        elif label["right"]["role"] == "active" and label["left"]["role"] != "active":
            active_side_counts["right"] += 1
        else:
            active_side_counts["other"] += 1

    audit = {
        "status": "PASS_STRUCTURED_HAND_SWAP_SYMMETRY",
        "source_group_count": len(groups),
        "view_count": len(views),
        "original_view_count": sum(not item["synthetic_symmetry_view"] for item in views),
        "synthetic_symmetry_view_count": sum(item["synthetic_symmetry_view"] for item in views),
        "creator_reviewed_source_count": sum(item["creator_reviewed_source"] for item in views),
        "independent_new_label_count": sum(item["independent_new_label"] for item in views),
        "calibration_label_active_side_counts": dict(sorted(active_side_counts.items())),
        "group_violation_count": len(group_violations),
        "invariant_violation_count": len(invariant_violations),
        "involution_violation_count": len(involution_violations),
        "changed_field_violation_count": len(changed_field_violations),
        "group_violations": group_violations,
        "invariant_violations": invariant_violations,
        "involution_violations": involution_violations,
        "changed_field_violations": changed_field_violations,
        "evaluation_results_opened": False,
        "training_performed": False,
    }
    expected = {
        "source_group_count": 30,
        "view_count": 60,
        "original_view_count": 30,
        "synthetic_symmetry_view_count": 30,
        "creator_reviewed_source_count": 30,
        "independent_new_label_count": 0,
        "calibration_label_active_side_counts": {"left": 30, "right": 30},
        "group_violation_count": 0,
        "invariant_violation_count": 0,
        "involution_violation_count": 0,
        "changed_field_violation_count": 0,
    }
    for name, value in expected.items():
        if audit[name] != value:
            audit["status"] = "HOLD_STRUCTURED_HAND_SWAP_SYMMETRY"
            raise ValueError(f"symmetry contract failed: {name}={audit[name]!r}, expected {value!r}")

    training_projection = [
        {
            "view_id": item["view_id"],
            "split_group": item["split_group"],
            "view": item["view"],
            "synthetic_symmetry_view": item["synthetic_symmetry_view"],
            "input_label": copy.deepcopy(item["original_label"]),
            "target_label": copy.deepcopy(item["calibration_label"]),
        }
        for item in views
    ]
    return {
        "schema": "quiethand.m3_5.role_swap_symmetry.v1",
        "source_schema": source["schema"],
        "source_group_count": 30,
        "view_count": 60,
        "source_calibration_label_active_side_counts": source["active_side_counts"],
        "calibration_label_active_side_counts": audit["calibration_label_active_side_counts"],
        "views": views,
        "training_projection": training_projection,
        "audit": audit,
        "training_contract": {
            "split_unit": "split_group",
            "pair_must_remain_atomic": True,
            "eligible_input": "structured_fields_only",
            "eligible_path": "training_projection",
            "eligible_fields": [
                "view_id",
                "split_group",
                "view",
                "synthetic_symmetry_view",
                "input_label",
                "target_label",
            ],
            "excluded_inputs": ["rgb", "mask", "point_cloud", "pose", "correction_text"],
            "training_authorized": False,
        },
        "claim_boundary": (
            "Grouped involutive structured symmetry views only; not new independent labels, "
            "real left-active evidence, RGB augmentation, model improvement, held-out generalization, "
            "annotation-cost reduction or downstream utility."
        ),
        "evaluation_results_opened": False,
        "training_performed": False,
    }


def label_line(label: dict[str, Any]) -> str:
    return (
        f"左：{html.escape(label['left']['role'])} / {html.escape(label['left']['contact'])}；"
        f"右：{html.escape(label['right']['role'])} / {html.escape(label['right']['contact'])}"
    )


def render_page(data: dict[str, Any], output: Path) -> None:
    source_ids = [
        "qh-m3-v12-cal-b9805bc0caea27ed3c97766d",
        "qh-m3-v12-cal-6f4c29963e05a124b3d99786",
    ]
    original_by_id = {
        item["source_event_id"]: item
        for item in data["views"] if item["view"] == "original"
    }
    real_cards = []
    for event_id in source_ids:
        item = original_by_id[event_id]
        label = item["calibration_label"]
        video = f"/artifacts/quiethand/m3_v1_2/result_preview/source_videos/{event_id}.mp4"
        real_cards.append(
            f'<article><video controls muted playsinline preload="metadata" src="{video}"></video>'
            f'<p><b>{html.escape(label["action"])}</b> · {html.escape(label["tool"])} → '
            f'{html.escape(label["target"])}</p><p>{label_line(label)}</p></article>'
        )

    examples = []
    for event_id in source_ids + ["qh-m3-v12-cal-35e9607ec879423a945ee0e5"]:
        pair = [item for item in data["views"] if item["source_event_id"] == event_id]
        by_view = {item["view"]: item for item in pair}
        before = by_view["original"]["calibration_label"]
        after = by_view["hand_swapped"]["calibration_label"]
        examples.append(
            f'<article><p><b>{html.escape(before["action"])}</b></p>'
            f'<p>原记录　{label_line(before)}</p><p>交换后　{label_line(after)}</p>'
            f'<p class="small">同组：<code>{html.escape(event_id)}</code></p></article>'
        )

    audit = data["audit"]
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>QuietHand 左右手对称数据层</title>
<style>body{{font:16px/1.65 system-ui;max-width:1180px;margin:auto;padding:24px;background:#f3f5f8;color:#18202a}}section{{background:#fff;border-radius:14px;padding:20px;margin:18px 0}}.hero{{border-left:6px solid #4768e8}}.warn{{border-left:6px solid #d68a18}}.ok{{border-left:6px solid #219653}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}article{{border:1px solid #dfe4eb;border-radius:12px;padding:14px}}video{{width:100%;background:#111;border-radius:9px}}code{{word-break:break-all;background:#eef1f5;padding:2px 5px;border-radius:5px}}.big{{font-size:30px;font-weight:750}}.small{{font-size:13px;color:#53606f}}@media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}</style>
<h1>去掉这批结构记录的“右手主动”边际偏斜</h1>
<section class="hero"><p>真实复盘源数据仍然只有 <b>30条</b>，主动手分布是左 <b>2</b> / 右 <b>28</b>。现在每条记录增加一个严格左右交换视图，并把原记录与交换记录锁在同一个划分组里。</p><p class="big">30条真实复盘记录 → 30个原视图 + 30个交换视图</p></section>
<section class="ok"><h2>机器检查结果：PASS</h2><p>共 {audit['source_group_count']} 个组、{audit['view_count']} 个视图；交换后目标标签中的主动手为左 {audit['calibration_label_active_side_counts']['left']} / 右 {audit['calibration_label_active_side_counts']['right']}。分组违规 {audit['group_violation_count']}，不变字段违规 {audit['invariant_violation_count']}，双交换恒等违规 {audit['involution_violation_count']}，变更字段顺序违规 {audit['changed_field_violation_count']}。</p></section>
<section class="warn"><h2>这一步没有凭空增加真实数据</h2><p><b>新增独立人工标签：0条。</b>交换只消除了这60个构造视图中目标标签的主动手边际偏斜；尚未训练模型，所以不能声称模型已经不再利用手侧捷径。它也不能证明真实左手主动场景已经覆盖，不能直接喂给看RGB视频的模型；RGB、mask、点云和pose都没有被镜像。</p><p>可供未来结构模型读取的<code>training_projection</code>只含白名单字段，物理上不包含带左右手文字的纠错评语。原记录与交换记录以后必须按<code>split_group</code>放进同一个划分。本门没有执行划分、训练或evaluation。</p></section>
<section><h2>两条真实左手主动源样本</h2><div class="grid">{''.join(real_cards)}</div></section>
<section><h2>交换到底改了什么</h2><div class="grid">{''.join(examples)}</div><p>动作、工具、目标不变；只交换左右手的接触对象和active/support角色。原评语保留作来源证据，但不会被当成交换后的训练文本。</p></section>
<section><a href="dataset.json">完整60视图数据</a> · <a href="audit.json">机器审计</a> · <a href="PLAN.md">冻结计划</a></section></html>'''
    (output / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = args.output if args.output.is_absolute() else workspace / args.output
    output.mkdir(parents=True, exist_ok=True)
    source = load(workspace / "artifacts/quiethand/m3_5/role_correction_calibration/calibration.json")
    data = build_dataset(source)
    (output / "dataset.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "audit.json").write_text(
        json.dumps(data["audit"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    render_page(data, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
