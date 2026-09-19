#!/usr/bin/env python3
"""Score the two sealed review rankings against completed blind targets."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


EXPECTED_EVENTS = 30
EXPECTED_BUDGET = 6


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def method_metrics(rows: list[dict[str, Any]], targets: dict[str, bool], total_errors: int) -> dict[str, Any]:
    selected = [row["event_id"] for row in rows if row.get("selected_for_review") is True]
    if len(selected) != EXPECTED_BUDGET or len(set(selected)) != EXPECTED_BUDGET:
        raise ValueError("sealed method does not select exactly six unique events")
    caught_ids = [event_id for event_id in selected if targets[event_id]]
    caught = len(caught_ids)
    return {
        "errors_caught_at_6": caught,
        "error_recall_at_6": None if total_errors == 0 else caught / total_errors,
        "precision_at_6": caught / EXPECTED_BUDGET,
        "false_accept_count_at_24": total_errors - caught,
        "selected_event_ids": selected,
        "caught_event_ids": caught_ids,
    }


def decide_verdict(vlm: dict[str, Any], geometry: dict[str, Any]) -> tuple[str, str]:
    if geometry["errors_caught_at_6"] > vlm["errors_caught_at_6"] and geometry["false_accept_count_at_24"] < vlm["false_accept_count_at_24"]:
        return (
            "SUPPORTS_GEOMETRY_GAIN_ON_THIS_30",
            "这 30 条支持：在同样只复核 6 条的预算下，加入既有手物几何比只看 VLM 不确定性抓到了更多核心标注错误。",
        )
    if geometry["errors_caught_at_6"] == vlm["errors_caught_at_6"]:
        return (
            "NO_SUPPORT_TIE_ON_THIS_30",
            "这 30 条没有显示几何排序带来增益：两种方法在同样 6 条复核预算下抓到的核心错误一样多。",
        )
    return (
        "NEGATIVE_GEOMETRY_WORSE_ON_THIS_30",
        "这 30 条得到负结果：加入既有手物几何后，在同样 6 条复核预算下反而抓到更少核心错误。",
    )


def result_html(result: dict[str, Any], target_rows: dict[str, dict[str, Any]]) -> str:
    def fmt(value: Any) -> str:
        return "未定义（全体无核心错误）" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value))

    methods = []
    for name, label in (("vlm_only", "仅 VLM"), ("vlm_plus_geometry", "VLM + 几何")):
        value = result["methods"][name]
        methods.append(
            f"<tr><th>{label}</th><td>{value['errors_caught_at_6']}</td><td>{fmt(value['error_recall_at_6'])}</td>"
            f"<td>{fmt(value['precision_at_6'])}</td><td>{value['false_accept_count_at_24']}</td></tr>"
        )
    cases = []
    for event_id in sorted(event_id for event_id, needed in result["target_map"].items() if needed):
        target = target_rows[event_id]
        selected_by = []
        if event_id in result["methods"]["vlm_only"]["selected_event_ids"]:
            selected_by.append("仅 VLM")
        if event_id in result["methods"]["vlm_plus_geometry"]["selected_event_ids"]:
            selected_by.append("VLM + 几何")
        cases.append(
            f"<tr><td>{html.escape(event_id)}</td><td>{html.escape(target['correction_summary'])}</td>"
            f"<td>{html.escape('、'.join(selected_by) or '两者都未选中')}</td></tr>"
        )
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>QuietHand evaluation 结果</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:1050px;margin:36px auto;padding:0 18px;color:#17202a}}.lead{{background:#eef4ff;border-radius:12px;padding:18px;font-size:18px}}table{{border-collapse:collapse;width:100%;margin:18px 0}}th,td{{border:1px solid #ccd4dc;padding:10px;text-align:left;vertical-align:top}}thead{{background:#eef1f4}}code{{font-size:12px}}</style></head>
<body><h1>QuietHand 30 条 evaluation 盲测</h1><p class="lead">{html.escape(result['plain_chinese_conclusion'])}</p>
<p>全体核心错误：{result['total_core_errors']} / {EXPECTED_EVENTS}；两种方法均只复核 {EXPECTED_BUDGET} 条。</p>
<table><thead><tr><th>方法</th><th>抓到错误</th><th>错误召回</th><th>复核精度</th><th>其余 24 条误放</th></tr></thead><tbody>{''.join(methods)}</tbody></table>
<h2>核心错误明细</h2><table><thead><tr><th>事件</th><th>盲审修正</th><th>谁选中了它</th></tr></thead><tbody>{''.join(cases)}</tbody></table>
<p>结论只限这 30 条 evaluation；没有训练、改阈值或测试下游机器人。</p></body></html>"""


def main() -> int:
    cfg = arguments()
    rankings = load(cfg.rankings)
    targets_doc = load(cfg.targets)
    rows = targets_doc.get("targets")
    if (
        rankings.get("status") != "SEALED_BEFORE_TARGET_REVIEW"
        or rankings.get("event_count") != EXPECTED_EVENTS
        or rankings.get("human_review_budget_each") != EXPECTED_BUDGET
        or rankings.get("human_targets_used") is not False
        or targets_doc.get("schema") != "quiethand.m3_5.evaluation_human_targets.v1"
        or targets_doc.get("status") != "COMPLETE"
        or targets_doc.get("event_count") != EXPECTED_EVENTS
        or targets_doc.get("review_basis") != "original_complete_video_and_vlm_coarse_label_only"
        or not isinstance(rows, list)
        or len(rows) != EXPECTED_EVENTS
    ):
        raise SystemExit("sealed rankings or completed blind targets are invalid")
    target_rows = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("event_id"), str) or not isinstance(row.get("review_needed"), bool):
            raise SystemExit("every target requires event_id and boolean review_needed")
        if not isinstance(row.get("correction_summary"), str) or not isinstance(row.get("minor_issue"), str):
            raise SystemExit("target text fields must be strings")
        if row["review_needed"] and not row["correction_summary"].strip():
            raise SystemExit("every core error requires a correction summary")
        if row["event_id"] in target_rows:
            raise SystemExit("target event ids are not unique")
        target_rows[row["event_id"]] = row
    rank_ids = {row["event_id"] for row in rankings["vlm_only"]}
    geometry_ids = {row["event_id"] for row in rankings["vlm_plus_geometry"]}
    if len(rank_ids) != EXPECTED_EVENTS or rank_ids != geometry_ids or rank_ids != set(target_rows):
        raise SystemExit("rankings and targets do not align one-to-one")

    target_map = {event_id: row["review_needed"] for event_id, row in target_rows.items()}
    total_errors = sum(target_map.values())
    vlm = method_metrics(rankings["vlm_only"], target_map, total_errors)
    geometry = method_metrics(rankings["vlm_plus_geometry"], target_map, total_errors)
    verdict, conclusion = decide_verdict(vlm, geometry)
    result = {
        "schema": "quiethand.m3_5.evaluation_router_result.v1",
        "status": "COMPLETE",
        "verdict": verdict,
        "plain_chinese_conclusion": conclusion,
        "event_count": EXPECTED_EVENTS,
        "human_review_budget_each": EXPECTED_BUDGET,
        "total_core_errors": total_errors,
        "methods": {"vlm_only": vlm, "vlm_plus_geometry": geometry},
        "target_map": target_map,
        "retuning_performed": False,
        "training_performed": False,
        "scope": "this frozen 30-event evaluation only",
    }
    if cfg.output.exists():
        raise SystemExit(f"destination exists: {cfg.output}")
    cfg.output.mkdir(parents=True)
    (cfg.output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (cfg.output / "index.html").write_text(result_html(result, target_rows), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "verdict": verdict, "total_core_errors": total_errors}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
