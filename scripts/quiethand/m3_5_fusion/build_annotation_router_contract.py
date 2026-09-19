#!/usr/bin/env python3
"""Build the calibration-only data contract for QuietHand annotation routing."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FORBIDDEN_INPUT_KEYS = {
    "review_decision",
    "review_needed",
    "calibration_label",
    "changed_fields",
    "correction_text",
    "label_source",
    "human_review",
}
SUPPORTED_CONTACT_STATES = {"maintained_close", "boundary_near"}
ALLOWED_CONTACT_STATES = SUPPORTED_CONTACT_STATES | {"not_close", "abstain"}
ALLOWED_PAIR_STATUS = {"observed", "abstain"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def duplicate_values(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def forbidden_key_paths(value: Any, prefix: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{prefix}.{key}"
            if key in FORBIDDEN_INPUT_KEYS:
                found.append(child_path)
            found.extend(forbidden_key_paths(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(forbidden_key_paths(child, f"{prefix}[{index}]"))
    return found


def claimed_contact_evidence(pair: dict[str, Any] | None) -> str:
    if not pair or pair.get("status") != "observed":
        return "missing"
    state = pair.get("contact", {}).get("state")
    if state in SUPPORTED_CONTACT_STATES:
        return "supported"
    if state == "not_close":
        return "contradicted"
    return "missing"


def active_side(label: dict[str, Any]) -> str:
    sides = [side for side in ("left", "right") if label[side]["role"] == "active"]
    if len(sides) != 1:
        raise ValueError(f"expected exactly one active hand, got {sides}")
    return sides[0]


def compact_pair(pair: dict[str, Any]) -> dict[str, Any]:
    status = pair.get("status")
    contact = pair.get("contact", {})
    relative = pair.get("relative_motion") or {}
    return {
        "status": status,
        "failure_reason": pair.get("failure_reason"),
        "contact": {
            "state": contact.get("state"),
            "valid_frame_count": contact.get("valid_frame_count"),
            "near_frame_count": contact.get("near_frame_count"),
            "close_frame_count": contact.get("close_frame_count"),
            "longest_close_run": contact.get("longest_close_run"),
            "median_distance_m": contact.get("median_distance_m"),
            "minimum_distance_m": contact.get("minimum_distance_m"),
        },
        "relative_motion": {
            "state": relative.get("state"),
            "diagonal_span_m": relative.get("diagonal_span_m"),
            "axis_span_m": relative.get("axis_span_m"),
            "stable_with_object": relative.get("stable_with_object"),
        },
    }


def compile_contract(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    base = repo_root / "artifacts/quiethand/m3_5"
    calibration = load_json(base / "role_correction_calibration/calibration.json")
    fusion = load_json(base / "identity_batch/fusion_data/data.json")
    fusion_audit = load_json(base / "identity_batch/fusion_data/audit.json")
    binding_dir = base / "identity_batch/binding/events"
    bindings = [load_json(path) for path in sorted(binding_dir.glob("*.json"))]

    errors: list[str] = []
    calibration_items = calibration.get("items", [])
    fusion_records = fusion.get("records", [])

    cal_ids = [item.get("event_id") for item in calibration_items]
    fusion_ids = [record.get("event_id") for record in fusion_records]
    binding_ids = [record.get("event_id") for record in bindings]
    for source, ids in (("calibration", cal_ids), ("fusion", fusion_ids), ("binding", binding_ids)):
        duplicates = duplicate_values(ids)
        if duplicates:
            errors.append(f"{source} duplicate event ids: {duplicates}")

    id_sets = {"calibration": set(cal_ids), "fusion": set(fusion_ids), "binding": set(binding_ids)}
    if not (id_sets["calibration"] == id_sets["fusion"] == id_sets["binding"]):
        all_ids = set().union(*id_sets.values())
        for source, ids in id_sets.items():
            missing = sorted(all_ids - ids)
            extra = sorted(ids - id_sets["calibration"])
            if missing:
                errors.append(f"{source} missing event ids: {missing}")
            if source != "calibration" and extra:
                errors.append(f"{source} extra event ids: {extra}")

    if calibration.get("event_count") != 30 or len(calibration_items) != 30:
        errors.append("calibration must contain exactly 30 events")
    if calibration.get("training_performed") is not False:
        errors.append("calibration training_performed must remain false")
    if calibration.get("evaluation_results_opened") is not False:
        errors.append("calibration evaluation_results_opened must remain false")
    if fusion_audit.get("training_performed") is not False:
        errors.append("fusion training_performed must remain false")
    if fusion_audit.get("evaluation_results_opened") is not False:
        errors.append("fusion evaluation_results_opened must remain false")
    if fusion_audit.get("evaluation_event_count") != 0:
        errors.append("fusion evaluation_event_count must remain zero")
    if fusion_audit.get("human_corrections_used_for_fusion") is not False:
        errors.append("human corrections must not have been used for fusion features")

    fusion_by_id = {record["event_id"]: record for record in fusion_records}
    binding_by_id = {record["event_id"]: record for record in bindings}
    input_projection: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    pair_status_counts: Counter[str] = Counter()
    contact_state_counts: Counter[str] = Counter()
    claimed_evidence_counts: Counter[str] = Counter()

    for item in calibration_items:
        event_id = item["event_id"]
        if event_id not in fusion_by_id or event_id not in binding_by_id:
            continue
        record = fusion_by_id[event_id]
        binding = binding_by_id[event_id]
        original = item["original_label"]

        if record.get("evaluation_results_opened") is not False:
            errors.append(f"{event_id}: evaluation_results_opened must be false")
        if record.get("human_review", {}).get("used_for_fusion") is not False:
            errors.append(f"{event_id}: human review was used for fusion")
        if binding.get("human_corrections_used") is not False:
            errors.append(f"{event_id}: human corrections were used for binding")
        if binding.get("native_pose_used") is not False:
            errors.append(f"{event_id}: native pose unexpectedly used for binding")

        geometry: dict[str, Any] = {}
        per_hand_evidence: dict[str, str] = {}
        for side in ("left", "right"):
            hand = record.get("hands", {}).get(side, {})
            if hand.get("qwen") != original[side]:
                errors.append(f"{event_id}: {side} Qwen fields do not match original VLM label")
            geometry[side] = {}
            for object_role in ("tool", "target"):
                pair = hand.get("pairs", {}).get(object_role)
                if not isinstance(pair, dict):
                    errors.append(f"{event_id}: missing {side}/{object_role} pair")
                    continue
                status = pair.get("status")
                state = pair.get("contact", {}).get("state")
                pair_status_counts[status] += 1
                contact_state_counts[state] += 1
                if status not in ALLOWED_PAIR_STATUS:
                    errors.append(f"{event_id}: invalid pair status {status!r}")
                if state not in ALLOWED_CONTACT_STATES:
                    errors.append(f"{event_id}: invalid contact state {state!r}")
                geometry[side][object_role] = compact_pair(pair)

            claimed_role = original[side].get("contact")
            claimed_pair = hand.get("pairs", {}).get(claimed_role) if claimed_role in {"tool", "target"} else None
            evidence = claimed_contact_evidence(claimed_pair)
            per_hand_evidence[side] = evidence
            claimed_evidence_counts[evidence] += 1

        consumer_input = {
            "event_id": event_id,
            "vlm_coarse_label": original,
            "binding": {
                "role_to_entity": binding.get("role_to_entity"),
                "source": binding.get("source"),
                "is_ground_truth": False,
            },
            "geometry": geometry,
            "existing_fusion_classification": record.get("classification"),
        }
        input_projection.append(consumer_input)

        target = {
            "event_id": event_id,
            "review_needed": item.get("review_decision") == "bad",
            "review_decision": item.get("review_decision"),
            "corrected_label": item.get("calibration_label"),
            "changed_fields": item.get("changed_fields"),
            "correction_text": item.get("correction_text"),
            "active_side": active_side(item["calibration_label"]),
        }
        targets.append(target)

        diagnostic_rows.append(
            {
                "event_id": event_id,
                "left_claimed_contact_evidence": per_hand_evidence["left"],
                "right_claimed_contact_evidence": per_hand_evidence["right"],
                "supported_hand_count": sum(value == "supported" for value in per_hand_evidence.values()),
                "contradicted_hand_count": sum(value == "contradicted" for value in per_hand_evidence.values()),
                "missing_hand_count": sum(value == "missing" for value in per_hand_evidence.values()),
                "fusion_classification": record.get("classification"),
                "review_needed": target["review_needed"],
            }
        )

    leakage_paths = forbidden_key_paths(input_projection)
    if leakage_paths:
        errors.append(f"target information leaked into input projection: {leakage_paths}")
    if len(input_projection) != 30 or len(targets) != 30:
        errors.append("input projection and targets must each contain 30 records")
    pair_slot_count = sum(
        len(row["geometry"][side])
        for row in input_projection
        for side in ("left", "right")
    )
    if pair_slot_count != 120:
        errors.append(f"expected 120 explicit pair slots, got {pair_slot_count}")

    contingency: dict[str, dict[str, int]] = defaultdict(lambda: {"accepted": 0, "review_needed": 0})
    classification_by_review: dict[str, dict[str, int]] = defaultdict(
        lambda: {"accepted": 0, "review_needed": 0}
    )
    for row in diagnostic_rows:
        bucket = str(row["contradicted_hand_count"])
        outcome = "review_needed" if row["review_needed"] else "accepted"
        contingency[bucket][outcome] += 1
        classification_by_review[row["fusion_classification"]][outcome] += 1

    accepted_count = sum(not target["review_needed"] for target in targets)
    review_needed_count = len(targets) - accepted_count
    active_counts = Counter(target["active_side"] for target in targets)
    status = "PASS" if not errors else "FAIL"

    dataset = {
        "schema": "quiethand.m3_5.annotation_router_contract.v1",
        "consumer": {
            "name": "annotation_quality_router",
            "purpose": "route VLM coarse annotations to auto_accept, human_review, or abstain before dataset release",
            "future_output_space": ["auto_accept", "human_review", "abstain"],
            "rule_or_model_selected": False,
            "direct_h_rdt_input": False,
        },
        "source_boundary": {
            "calibration_only": True,
            "training_performed": False,
            "evaluation_results_opened": False,
            "human_targets_used_as_input": False,
        },
        "input_projection": input_projection,
        "targets": targets,
        "diagnostics": diagnostic_rows,
    }
    audit = {
        "schema": "quiethand.m3_5.annotation_router_contract_audit.v1",
        "status": status,
        "errors": errors,
        "join": {
            "calibration_events": len(cal_ids),
            "fusion_events": len(fusion_ids),
            "binding_events": len(binding_ids),
            "joined_inputs": len(input_projection),
            "targets": len(targets),
        },
        "leakage": {
            "forbidden_input_key_paths": leakage_paths,
            "passed": not leakage_paths,
        },
        "feature_coverage": {
            "explicit_pair_slots": pair_slot_count,
            "pair_status_counts": dict(sorted(pair_status_counts.items())),
            "contact_state_counts": dict(sorted(contact_state_counts.items())),
            "claimed_contact_evidence_counts": dict(sorted(claimed_evidence_counts.items())),
        },
        "calibration_targets": {
            "accepted": accepted_count,
            "review_needed": review_needed_count,
            "active_side_counts": dict(sorted(active_counts.items())),
        },
        "fixed_diagnostics": {
            "contradicted_hand_count_by_review": dict(sorted(contingency.items())),
            "fusion_classification_by_review": dict(sorted(classification_by_review.items())),
            "thresholds_retuned": False,
        },
        "readiness": {
            "consumer_interface_ready": status == "PASS",
            "defensible_model_training_ready": False,
            "reason": "Only 5 review-needed targets and 2 real left-active labels; no held-out evaluation is opened.",
        },
        "claim_boundary": (
            "Calibration-only consumer contract and signal coverage; not router performance, automatic-label accuracy, "
            "annotation-cost reduction, train-readiness, H-RDT compatibility, or downstream utility."
        ),
    }
    return dataset, audit


def render_result(audit: dict[str, Any]) -> str:
    coverage = audit["feature_coverage"]
    targets = audit["calibration_targets"]
    diagnostics = audit["fixed_diagnostics"]
    rows = []
    for contradicted, counts in diagnostics["contradicted_hand_count_by_review"].items():
        rows.append(
            f"| {contradicted} | {counts['accepted']} | {counts['review_needed']} |"
        )
    table = "\n".join(rows)
    return f"""# QuietHand M3.5 标注质检路由器：校准就绪结果

## 结论

**{audit['status']}：消费者接口闭环，但还不能训练一个可信的路由器。**

30 条 VLM 粗标、fusion 几何证据和人工目标已一一对齐；输入投影中没有人工审核答案。现有几何信号可以成为“审核前输入”，但人工判为需要修正的样本只有 {targets['review_needed']} 条，真实左手主动样本只有 {targets['active_side_counts'].get('left', 0)} 条，当前数据不足以支撑训练或性能结论。

## 这套数据现在给谁用

直接消费者不是 H-RDT，而是一个未来的标注质检路由器：

`VLM 粗标 + 手物几何摘要 -> 自动接受 / 送人工 / 弃权 -> 通过质检的结构化数据`

H-RDT 后续若要使用这些数据，还需要另做它所需的连续双手轨迹与语言适配；本关没有声称已经接通。

## 接口检查

- 对齐事件：{audit['join']['joined_inputs']} / 30
- 几何配对槽位：{coverage['explicit_pair_slots']} / 120
- observed / abstain：{coverage['pair_status_counts'].get('observed', 0)} / {coverage['pair_status_counts'].get('abstain', 0)}
- 人工答案泄漏：0
- training：关闭
- evaluation：关闭

## 固定信号诊断

这里没有调阈值，只看既有 fusion 状态。`矛盾手数` 表示 VLM 声称某只手接触某物体，但几何状态为 `not_close` 的手数。

| 每条样本的矛盾手数 | 人工接受 | 人工认为需修正 |
|---:|---:|---:|
{table}

这个表只能说明现有接触信号与人工审核之间有没有结构关系，不能当作分类准确率。

## 下一步判断

工程接口已经准备好；当前真正缺的是更多真实的、尤其是左手主动和失败样本，而不是再堆一个模型。拿到这批证据后，才值得做按原始视频分组的训练/验证切分并训练路由器。

## 证据边界

{audit['claim_boundary']}
"""


def render_html(audit: dict[str, Any]) -> str:
    coverage = audit["feature_coverage"]
    targets = audit["calibration_targets"]
    rows = "".join(
        f"<tr><td>{html.escape(key)}</td><td>{value['accepted']}</td><td>{value['review_needed']}</td></tr>"
        for key, value in audit["fixed_diagnostics"]["contradicted_hand_count_by_review"].items()
    )
    status_class = "pass" if audit["status"] == "PASS" else "fail"
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuietHand 标注质检路由器</title>
<style>
body{{margin:0;background:#f4f1ea;color:#191919;font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}}
main{{max-width:980px;margin:auto;padding:48px 24px 80px}}h1{{font-size:38px;margin:0 0 14px}}h2{{margin-top:38px}}
.lead{{font-size:20px;line-height:1.7}}.badge{{display:inline-block;padding:6px 12px;border-radius:999px;font-weight:700}}
.pass{{background:#d8f3df;color:#176b35}}.fail{{background:#ffd9d9;color:#8c1c1c}}
.flow{{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;gap:12px;align-items:center;margin:28px 0}}
.box,.card{{background:#fff;border:1px solid #ded8cb;border-radius:16px;padding:20px;box-shadow:0 8px 24px #4d46330d}}
.arrow{{font-size:24px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.number{{font-size:30px;font-weight:750}}
table{{width:100%;border-collapse:collapse;background:#fff;border-radius:14px;overflow:hidden}}th,td{{padding:13px;border-bottom:1px solid #eee;text-align:left}}
.boundary{{border-left:5px solid #d19a2a;background:#fff8e8;padding:16px 18px;border-radius:8px;line-height:1.7}}
@media(max-width:760px){{.flow{{grid-template-columns:1fr}}.arrow{{transform:rotate(90deg);text-align:center}}.cards{{grid-template-columns:1fr 1fr}}}}
</style></head><body><main>
<span class="badge {status_class}">{audit['status']}</span><h1>QuietHand 标注质检路由器</h1>
<p class="lead">接口已经闭环，但现在还不能训练出一个可信的路由器。真正缺的是更多真实失败样本和左手主动样本。</p>
<div class="flow"><div class="box"><b>审核前输入</b><br>VLM 粗标 + 手物几何</div><div class="arrow">→</div><div class="box"><b>质检路由器</b><br>接受 / 人工 / 弃权</div><div class="arrow">→</div><div class="box"><b>通过质检的数据</b><br>再适配下游模型</div></div>
<p><b>关键澄清：</b>这些 role/contact 字段不是 H-RDT 的直接输入。当前最短消费者是质检路由器。</p>
<h2>接口检查</h2><div class="cards">
<div class="card"><div class="number">{audit['join']['joined_inputs']}/30</div>事件一一对齐</div>
<div class="card"><div class="number">{coverage['explicit_pair_slots']}/120</div>几何配对槽位</div>
<div class="card"><div class="number">0</div>人工答案泄漏</div>
<div class="card"><div class="number">{targets['review_needed']}</div>需修正样本</div></div>
<h2>固定信号诊断</h2><p>没有调阈值。矛盾手数 = VLM 声称接触，但既有几何状态为 <code>not_close</code>。</p>
<table><thead><tr><th>每条样本的矛盾手数</th><th>人工接受</th><th>人工认为需修正</th></tr></thead><tbody>{rows}</tbody></table>
<h2>现在能说什么</h2><p class="boundary">可以说：接口与信号覆盖已闭环。<br>不能说：路由器有效、自动标注准确、已经降本、可直接喂 H-RDT，或有下游收益。</p>
<h2>下一步</h2><p>补真实样本，优先补左手主动和 VLM 粗标失败；然后按原始视频分组切分，再训练质检路由器。当前 training 与 evaluation 都保持关闭。</p>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to artifacts/quiethand/m3_5/annotation_router_contract",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir or repo_root / "artifacts/quiethand/m3_5/annotation_router_contract"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset, audit = compile_contract(repo_root)

    (output_dir / "consumer_matrix.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "RESULT.md").write_text(render_result(audit), encoding="utf-8")
    (output_dir / "index.html").write_text(render_html(audit), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
