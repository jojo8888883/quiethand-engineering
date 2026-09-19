#!/usr/bin/env python3
"""Compile the user's completed 30-event review into a provenance-bearing calibration set."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import html
import json
from pathlib import Path
from typing import Any


CORRECTED_LABELS: dict[str, dict[str, Any]] = {
    "qh-m3-v12-cal-594fc82381d222917ad15a7b": {
        "action": "操作木勺",
        "tool": "wooden spoon",
        "target": "wooden board",
        "left": {"contact": "target", "role": "support"},
        "right": {"contact": "tool", "role": "active"},
    },
    "qh-m3-v12-cal-cb52358278979fb42b1a1344": {
        "action": "将蓝色碗里的东西倒入黑色锅",
        "tool": "blue bowl",
        "target": "black pot",
        "left": {"contact": "target", "role": "support"},
        "right": {"contact": "tool", "role": "active"},
    },
    "qh-m3-v12-cal-9e71200d66c4d9be4acf8bfc": {
        "action": "将木盒内容物倒入白色漏碗",
        "tool": "wooden box",
        "target": "white colander",
        "left": {"contact": "target", "role": "support"},
        "right": {"contact": "tool", "role": "active"},
    },
    "qh-m3-v12-cal-05df271ef9413bfd71b7142d": {
        "action": "用滚筒在木盒上滚动",
        "tool": "roller tool",
        "target": "wooden box",
        "left": {"contact": "target", "role": "support"},
        "right": {"contact": "tool", "role": "active"},
    },
    "qh-m3-v12-cal-533f4c219792c721eda87419": {
        "action": "用打磨工具打磨锅",
        "tool": "grinder",
        "target": "pot",
        "left": {"contact": "target", "role": "support"},
        "right": {"contact": "tool", "role": "active"},
    },
}

FIELDS = ("action", "tool", "target", "left.contact", "left.role", "right.contact", "right.role")


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def field(label: dict[str, Any], name: str) -> Any:
    value: Any = label
    for key in name.split("."):
        value = value[key]
    return value


def validate_label(label: dict[str, Any]) -> None:
    if set(label) != {"action", "tool", "target", "left", "right"}:
        raise ValueError("label fields changed")
    for side in ("left", "right"):
        if set(label[side]) != {"contact", "role"}:
            raise ValueError(f"{side} fields changed")
        if label[side]["contact"] not in {"tool", "target", "none", "uncertain"}:
            raise ValueError(f"invalid {side} contact")
        if label[side]["role"] not in {"active", "support", "none", "uncertain"}:
            raise ValueError(f"invalid {side} role")


def compile_set(review: dict[str, Any], corrections: dict[str, Any]) -> dict[str, Any]:
    review_items = review.get("items", [])
    correction_items = corrections.get("items", [])
    if len(review_items) != 30 or len(correction_items) != 5:
        raise ValueError("expected the completed 30-event review and five corrections")
    review_by_id = {item["event_id"]: item for item in review_items}
    correction_by_id = {item["event_id"]: item for item in correction_items}
    if len(review_by_id) != 30 or len(correction_by_id) != 5:
        raise ValueError("duplicate event id")
    rejected = {event_id for event_id, item in review_by_id.items() if item["decision"] == "bad"}
    if rejected != set(CORRECTED_LABELS) or rejected != set(correction_by_id):
        raise ValueError("bad-event set and structured corrections disagree")

    compiled = []
    for source in review_items:
        event_id = source["event_id"]
        original = copy.deepcopy(source["original_label"])
        validate_label(original)
        if source["decision"] == "ok":
            corrected = original
            label_source = "user_accepted_model_output"
            correction_text = None
        elif source["decision"] == "bad":
            corrected = copy.deepcopy(CORRECTED_LABELS[event_id])
            validate_label(corrected)
            label_source = "user_correction_interpreted"
            correction_source = correction_by_id[event_id]
            if correction_source.get("original_decision") != "bad":
                raise ValueError("correction is not bound to a rejected review")
            if correction_source.get("original_model_label") != original:
                raise ValueError("correction and review disagree on the original label")
            correction_text = correction_source["correction_text"]
            if not isinstance(correction_text, str) or not correction_text.strip():
                raise ValueError("correction text is empty")
        else:
            raise ValueError(f"unsupported review decision: {source['decision']}")
        changed = [name for name in FIELDS if field(original, name) != field(corrected, name)]
        compiled.append({
            "event_id": event_id,
            "review_decision": source["decision"],
            "label_source": label_source,
            "correction_text": correction_text,
            "original_label": original,
            "calibration_label": corrected,
            "changed_fields": changed,
            "evaluation": False,
            "training_use_status": "not_authorized",
        })

    pair_counts = Counter(
        (item["calibration_label"]["left"]["role"], item["calibration_label"]["right"]["role"])
        for item in compiled
    )
    active_side_counts = Counter(
        "left" if item["calibration_label"]["left"]["role"] == "active"
        else "right" if item["calibration_label"]["right"]["role"] == "active"
        else "other"
        for item in compiled
    )
    return {
        "schema": "quiethand.m3_5.role_correction_calibration.v1",
        "event_count": len(compiled),
        "accepted_count": sum(item["label_source"] == "user_accepted_model_output" for item in compiled),
        "corrected_count": sum(item["label_source"] == "user_correction_interpreted" for item in compiled),
        "ambiguous_correction_count": 0,
        "role_changed_event_count": sum(
            any(name.endswith(".role") for name in item["changed_fields"]) for item in compiled
        ),
        "role_pair_counts": {f"left={left},right={right}": count for (left, right), count in sorted(pair_counts.items())},
        "active_side_counts": dict(sorted(active_side_counts.items())),
        "items": compiled,
        "evaluation_results_opened": False,
        "training_performed": False,
        "claim_boundary": "Creator-reviewed calibration labels with interpreted corrections; not independent gold, train-readiness, batch accuracy, annotation-cost reduction or downstream utility.",
    }


def render_page(data: dict[str, Any], output: Path) -> None:
    corrected = [item for item in data["items"] if item["review_decision"] == "bad"]
    cards = []
    for item in corrected:
        before = item["original_label"]
        after = item["calibration_label"]
        video = f"/artifacts/quiethand/m3_v1_2/result_preview/source_videos/{item['event_id']}.mp4"
        cards.append(f"""<section class="card"><h2>{html.escape(item['correction_text'])}</h2>
<video controls muted playsinline preload="metadata" src="{video}"></video>
<div class="cols"><div><h3>原模型</h3><p>{html.escape(before['action'])}</p><p>左：{before['left']['role']} / {before['left']['contact']}<br>右：{before['right']['role']} / {before['right']['contact']}</p></div>
<div><h3>整理后</h3><p>{html.escape(after['action'])}</p><p>左：{after['left']['role']} / {after['left']['contact']}<br>右：{after['right']['role']} / {after['right']['contact']}</p><p>变更：{html.escape(', '.join(item['changed_fields']))}</p></div></div></section>""")
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>QuietHand 角色纠错校准集</title>
<style>body{{font:16px/1.6 system-ui;max-width:1180px;margin:auto;padding:24px;background:#f3f5f8;color:#18202a}}section{{background:#fff;border-radius:14px;padding:20px;margin:18px 0}}.summary{{border-left:6px solid #5267e8}}.warn{{border-left:6px solid #d68a18}}.cols{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}video{{width:100%;max-height:520px;background:#111;border-radius:10px}}code{{background:#eef1f5;padding:2px 5px;border-radius:5px}}@media(max-width:760px){{.cols{{grid-template-columns:1fr}}}}</style>
<h1>30条复盘现在整理成了什么</h1>
<section class="summary"><p><b>已闭合：</b>{data['event_count']}条，其中{data['accepted_count']}条沿用你确认 OK 的标签，{data['corrected_count']}条按你已有评语结构化，0条需要你重新判断。</p></section>
<section class="warn"><p><b>暂时不能直接训练：</b>整理后右手主动 {data['active_side_counts'].get('right',0)} 条，左手主动 {data['active_side_counts'].get('left',0)} 条。直接训练很可能只学会“猜右手主动”。这不是数据错误，而是当前30条的动作分布太偏。</p><p>本页只交付校准纠错集，没有打开 evaluation，也没有训练模型。</p></section>
{''.join(cards)}
<section><a href="calibration.json">完整结构化数据</a> · <a href="audit.json">编译审计</a></section></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = args.output if args.output.is_absolute() else workspace / args.output
    output.mkdir(parents=True, exist_ok=True)
    review = load(workspace / "artifacts/quiethand/m3_v1_2/correction_review/input_review.json")
    corrections = load(workspace / "artifacts/quiethand/m3_v1_2/correction_review/quiethand-m3-v1-2-corrections-2026-08-31T09-08-43-578Z.json")
    data = compile_set(review, corrections)
    (output / "calibration.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {
        "status": "PASS_COMPILED_CALIBRATION_SET",
        "event_count": data["event_count"],
        "accepted_count": data["accepted_count"],
        "corrected_count": data["corrected_count"],
        "ambiguous_correction_count": data["ambiguous_correction_count"],
        "role_changed_event_count": data["role_changed_event_count"],
        "active_side_counts": data["active_side_counts"],
        "training_readiness": "HOLD_CLASS_SKEW_AND_NO_TRAINING_GATE",
        "evaluation_results_opened": False,
        "training_performed": False,
    }
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render_page(data, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
