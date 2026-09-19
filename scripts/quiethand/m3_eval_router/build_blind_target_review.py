#!/usr/bin/env python3
"""Build the post-ranking blind review bundle from full evaluation videos."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from materialize_temporal import source_path  # noqa: E402


EXPECTED_EVENTS = 30


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--temporal-plan", type=Path, required=True)
    parser.add_argument("--semantic-results", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def make_contact_sheet(source: Path, frame_count: int, destination: Path) -> None:
    indices = [round(index * (frame_count - 1) / 15) for index in range(16)]
    expression = "+".join(f"eq(n\\,{index})" for index in indices)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-threads", "1",
        "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", f"select={expression},scale=400:-1,tile=4x4:padding=4:margin=4",
        "-frames:v", "1", "-q:v", "3", str(destination),
    ]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0 or not destination.is_file():
        raise RuntimeError(f"contact sheet failed: {result.stderr.decode(errors='replace').strip()}")


def transcode_for_browser(source: Path, destination: Path) -> None:
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-threads", "1",
        "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(destination),
    ]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0 or not destination.is_file():
        raise RuntimeError(f"browser video transcode failed: {result.stderr.decode(errors='replace').strip()}")


def natural_label(candidate: dict[str, Any]) -> str:
    role = {"active": "主动", "support": "支撑", "both": "兼具两种角色", "neither": "无角色", "unknown": "未知"}
    contact = {"tool": "工具", "target": "目标物", "both": "工具和目标物", "neither": "均未接触", "unknown": "接触未知"}
    left = candidate.get("left") or {}
    right = candidate.get("right") or {}
    return (
        f"动作：{candidate.get('action_candidate') or '未识别'}；"
        f"工具：{candidate.get('tool_candidate') or '未识别'}；"
        f"目标物：{candidate.get('target_candidate') or '未识别'}。"
        f"左手：{role.get(left.get('role'), left.get('role') or '未知')}，{contact.get(left.get('contact'), left.get('contact') or '接触未知')}；"
        f"右手：{role.get(right.get('role'), right.get('role') or '未知')}，{contact.get(right.get('contact'), right.get('contact') or '接触未知')}。"
    )


def build_html(rows: list[dict[str, Any]]) -> str:
    cards = []
    for index, row in enumerate(rows, start=1):
        event_id = html.escape(row["event_id"])
        label = html.escape(row["vlm_label_zh"])
        cards.append(f"""
<article class="card" data-event="{event_id}">
  <h2>{index:02d} · {event_id}</h2>
  <video controls preload="none" src="{html.escape(row['video'])}"></video>
  <details><summary>16 帧总览（只辅助定位，判断以完整视频为准）</summary><img src="{html.escape(row['contact_sheet'])}"></details>
  <p class="label">{label}</p>
  <div class="decision">
    <label><input type="radio" name="d-{event_id}" value="false"> 核心结构正确</label>
    <label><input type="radio" name="d-{event_id}" value="true"> 需要修正</label>
  </div>
  <label>核心修正（动作、工具、目标物、左右手角色/接触）<textarea class="correction"></textarea></label>
  <label>小问题（颜色、宽泛但不错误的措辞等）<textarea class="minor"></textarea></label>
</article>""")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuietHand evaluation 盲审</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f3f5f7;color:#17202a}}
header{{position:sticky;top:0;z-index:2;background:#17202a;color:white;padding:16px 24px}}
header p{{margin:6px 0 0;color:#dce3ea}} main{{max-width:1100px;margin:24px auto;padding:0 16px}}
.card{{background:white;border-radius:14px;padding:18px;margin:0 0 22px;box-shadow:0 2px 12px #0001}}
h2{{font-size:17px;margin:0 0 12px}} video,img{{display:block;width:100%;background:#111;border-radius:10px}}
details{{margin-top:10px}} summary{{cursor:pointer;margin-bottom:8px}} .label{{font-size:16px;line-height:1.7;background:#eef4ff;padding:12px;border-radius:8px}}
.decision{{display:flex;gap:28px;margin:14px 0}} label{{display:block;margin-top:10px}} textarea{{box-sizing:border-box;width:100%;min-height:64px;margin-top:6px;padding:8px}}
button{{border:0;border-radius:8px;background:#2d6cdf;color:white;padding:10px 18px;font-size:15px;cursor:pointer}}
#progress{{font-weight:700;margin-right:18px}}
</style></head><body>
<header><div><span id="progress">0 / {EXPECTED_EVENTS}</span><button id="export">导出盲审 JSON</button></div>
<p>只看完整视频和 VLM 粗标。不要查看几何分数、两种方法排名或 TACO 三元组。核心字段错/缺才选“需要修正”；小问题单独记。</p></header>
<main>{''.join(cards)}</main>
<script>
const key='quiethand-eval-targets-v1';
const cards=[...document.querySelectorAll('.card')];
function read(){{return cards.map(card=>{{const chosen=card.querySelector('input:checked');return {{event_id:card.dataset.event,review_needed:chosen?chosen.value==='true':null,correction_summary:card.querySelector('.correction').value.trim(),minor_issue:card.querySelector('.minor').value.trim()}}}})}}
function save(){{const rows=read();localStorage.setItem(key,JSON.stringify(rows));document.getElementById('progress').textContent=`${{rows.filter(x=>x.review_needed!==null).length}} / {EXPECTED_EVENTS}`}}
const old=JSON.parse(localStorage.getItem(key)||'[]');
for(const row of old){{const card=cards.find(x=>x.dataset.event===row.event_id);if(!card)continue;if(row.review_needed!==null)card.querySelector(`input[value="${{row.review_needed}}"]`).checked=true;card.querySelector('.correction').value=row.correction_summary||'';card.querySelector('.minor').value=row.minor_issue||''}}
document.addEventListener('input',save); save();
document.getElementById('export').onclick=()=>{{const targets=read();if(targets.some(x=>x.review_needed===null)){{alert('还有未判断的条目');return}}const payload={{schema:'quiethand.m3_5.evaluation_human_targets.v1',status:'COMPLETE',event_count:{EXPECTED_EVENTS},review_basis:'original_complete_video_and_vlm_coarse_label_only',targets}};const blob=new Blob([JSON.stringify(payload,null,2)+'\\n'],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='quiethand-evaluation-human-targets.json';a.click();URL.revokeObjectURL(a.href)}};
</script></body></html>"""


def main() -> int:
    cfg = arguments()
    workspace = cfg.workspace.resolve()
    plan = load(cfg.temporal_plan)
    rankings = load(cfg.rankings)
    semantic_audit = load(cfg.semantic_results / "audit.json")
    events = plan.get("events")
    if (
        plan.get("evaluation_video_count") != EXPECTED_EVENTS
        or not isinstance(events, list)
        or len(events) != EXPECTED_EVENTS
        or rankings.get("status") != "SEALED_BEFORE_TARGET_REVIEW"
        or rankings.get("event_count") != EXPECTED_EVENTS
        or rankings.get("human_targets_used") is not False
        or semantic_audit.get("status") != "COMPLETE"
        or semantic_audit.get("event_count") != EXPECTED_EVENTS
    ):
        raise SystemExit("blind-review dependencies are not sealed and complete")
    if cfg.output.exists():
        raise SystemExit(f"destination exists: {cfg.output}")
    ids = [event.get("event_id") for event in events if isinstance(event, dict)]
    ranking_ids = {row["event_id"] for name in ("vlm_only", "vlm_plus_geometry") for row in rankings[name]}
    if len(set(ids)) != EXPECTED_EVENTS or set(ids) != ranking_ids:
        raise SystemExit("blind-review event ids do not align")

    video_root = cfg.output / "videos"
    sheet_root = cfg.output / "contact_sheets"
    video_root.mkdir(parents=True)
    sheet_root.mkdir()
    rows = []
    try:
        for ordinal, event in enumerate(events, start=1):
            event_id = event["event_id"]
            semantic = load(cfg.semantic_results / "events" / f"{event_id}.json")
            candidate = semantic.get("candidate") or {}
            source = source_path(cfg.source_root, event["source_binding"]["rgb"])
            video = video_root / f"{event_id}.mp4"
            sheet = sheet_root / f"{event_id}.jpg"
            transcode_for_browser(source, video)
            make_contact_sheet(source, int(event["source_frame_count"]), sheet)
            rows.append({
                "event_id": event_id,
                "video": f"videos/{video.name}",
                "contact_sheet": f"contact_sheets/{sheet.name}",
                "vlm_label": candidate,
                "vlm_label_zh": natural_label(candidate),
            })
            print(f"[blind-review] {ordinal}/{EXPECTED_EVENTS} {event_id}", flush=True)
        manifest = {
            "schema": "quiethand.m3_5.evaluation_blind_review_manifest.v1",
            "status": "READY_FOR_BLIND_TARGET_REVIEW",
            "event_count": EXPECTED_EVENTS,
            "review_basis": "original_complete_video_and_vlm_coarse_label_only",
            "geometry_exposed": False,
            "rankings_exposed": False,
            "taco_triplets_exposed": False,
            "events": rows,
        }
        write_json(cfg.output / "review_manifest.json", manifest)
        write_json(cfg.output / "human_targets_template.json", {
            "schema": "quiethand.m3_5.evaluation_human_targets.v1",
            "status": "INCOMPLETE",
            "event_count": EXPECTED_EVENTS,
            "review_basis": "original_complete_video_and_vlm_coarse_label_only",
            "targets": [
                {"event_id": row["event_id"], "review_needed": None, "correction_summary": "", "minor_issue": ""}
                for row in rows
            ],
        })
        (cfg.output / "index.html").write_text(build_html(rows), encoding="utf-8")
    except Exception:
        shutil.rmtree(cfg.output)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
