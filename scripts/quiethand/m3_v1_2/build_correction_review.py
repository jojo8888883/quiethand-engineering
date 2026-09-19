#!/usr/bin/env python3
"""Build a focused free-text review page for rejected M3-v1.2 records."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def html_document(records: list[dict], source_exported_at: str) -> str:
    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuietHand M3-v1.2：5条错误复盘</title>
<style>
:root {{ color-scheme:dark; --bg:#081018; --card:#111d29; --line:#2b4052; --text:#edf5fb; --muted:#9cb0c0; --accent:#7dd3fc; --bad:#fb7185; --good:#4ade80; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:radial-gradient(circle at top,#14283a 0,#081018 48%); color:var(--text); font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
header,main {{ max-width:1120px; margin:auto; padding-inline:24px; }}
header {{ padding-top:38px; padding-bottom:22px; }}
h1 {{ margin:0 0 8px; font-size:32px; }}
.lead {{ color:var(--muted); margin:0; }}
.toolbar {{ position:sticky; top:0; z-index:5; display:flex; align-items:center; justify-content:space-between; gap:14px; padding:12px 24px; border-block:1px solid var(--line); background:#081018ed; backdrop-filter:blur(12px); }}
.progress {{ font-weight:700; }}
button {{ border:1px solid #4d7897; border-radius:10px; background:#17344a; color:var(--text); padding:9px 14px; cursor:pointer; }}
main {{ padding-top:22px; padding-bottom:70px; display:grid; gap:22px; }}
.card {{ overflow:hidden; border:1px solid var(--line); border-top:3px solid var(--bad); border-radius:16px; background:linear-gradient(145deg,#132230,#0d1822); box-shadow:0 12px 30px #0005; }}
.card.done {{ border-top-color:var(--good); }}
video {{ display:block; width:100%; aspect-ratio:16/9; object-fit:cover; background:#000; }}
.body {{ padding:17px 18px 20px; }}
.id {{ color:#b9cbd8; font:12px ui-monospace,SFMono-Regular,Menlo,monospace; }}
.model {{ margin:10px 0 14px; padding:12px 14px; border-radius:10px; background:#0a151f; color:#dbe9f3; }}
.model b {{ color:var(--bad); }}
label {{ display:block; color:var(--accent); font-weight:700; margin-bottom:7px; }}
textarea {{ display:block; width:100%; min-height:130px; resize:vertical; border:1px solid #3b5870; border-radius:11px; background:#09131c; color:var(--text); padding:12px 13px; font:15px/1.65 inherit; }}
textarea:focus {{ outline:2px solid #38bdf866; border-color:#38bdf8; }}
.state {{ margin-top:7px; color:var(--muted); font-size:13px; }}
.done .state {{ color:var(--good); }}
@media(max-width:700px) {{ header,main {{ padding-inline:13px; }} .toolbar {{ padding-inline:13px; align-items:flex-start; flex-direction:column; }} }}
</style>
</head>
<body>
<header>
  <h1>5条“不行”：请写实际发生了什么</h1>
  <p class="lead">只显示你在上一轮判为“不行”的视频。每条看完后，用自己的话写真实动作、物体和双手分工；输入会自动保存在当前浏览器。</p>
</header>
<div class="toolbar"><div class="progress" id="progress"></div><button id="export" type="button">导出复盘 JSON</button></div>
<main id="cards"></main>
<script>
const records={payload};
const sourceExportedAt={json.dumps(source_exported_at, ensure_ascii=False)};
const storageKey="quiethand-m3-v1-2-bad-corrections-v1";
const actionZh={{"manipulating spoon":"操作勺子","pouring":"倾倒","lifting and rotating":"抬起并旋转","assembling a wooden box with a roller tool":"用滚筒工具组装木盒","grinding":"打磨"}};
const objectZh={{"wooden spoon":"木勺","blue bowl":"蓝色碗","black pot":"黑色锅","unknown":"未识别","wooden box":"木盒","roller tool":"滚筒工具","grinder":"打磨工具","pot":"锅"}};
const roleZh={{active:"主动作手",support:"辅助/稳定手",unknown:"未识别"}};
const contactZh={{tool:"接触工具",target:"接触作用对象",none:"未接触",unknown:"未识别"}};
const esc=s=>String(s??"").replace(/[&<>\"']/g,c=>({{"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#039;"}}[c]||"&quot;"));
let corrections={{}};
try {{ corrections=JSON.parse(localStorage.getItem(storageKey)||"{{}}"); }} catch (_) {{ corrections={{}}; }}
function zh(map,value) {{ return map[value]||value||"未识别"; }}
function handText(hand) {{ return `${{zh(roleZh,hand?.role)}}，${{zh(contactZh,hand?.contact)}}`; }}
function modelText(record) {{
  return `模型原始标签：动作“${{zh(actionZh,record.action)}}”；工具“${{zh(objectZh,record.tool)}}”；作用对象“${{zh(objectZh,record.target)}}”；左手“${{handText(record.left)}}”；右手“${{handText(record.right)}}”。`;
}}
function save() {{ localStorage.setItem(storageKey,JSON.stringify(corrections)); update(); }}
function update() {{
  let complete=0;
  records.forEach(record=>{{
    const value=(corrections[record.event_id]||"").trim();
    const card=document.querySelector(`[data-event="${{record.event_id}}"]`);
    card.classList.toggle("done",Boolean(value));
    card.querySelector(".state").textContent=value?"已自动保存":"尚未填写";
    if(value) complete++;
  }});
  document.querySelector("#progress").textContent=`已填写 ${{complete}} / ${{records.length}}`;
}}
document.querySelector("#cards").innerHTML=records.map(record=>`
  <article class="card" data-event="${{esc(record.event_id)}}">
    <video controls muted playsinline preload="metadata" src="../result_preview/source_videos/${{esc(record.event_id)}}.mp4?v=20260830-h264"></video>
    <div class="body">
      <div class="id">${{esc(record.event_id)}}</div>
      <div class="model"><b>你上一轮判定：不行</b><br>${{esc(modelText(record))}}</div>
      <label for="text-${{esc(record.event_id)}}">实际情况是什么？</label>
      <textarea id="text-${{esc(record.event_id)}}" placeholder="例如：实际是右手……，左手……；模型把动作/物体/双手分工识别错了。"></textarea>
      <div class="state"></div>
    </div>
  </article>`).join("");
records.forEach(record=>{{
  const area=document.querySelector(`[data-event="${{record.event_id}}"] textarea`);
  area.value=corrections[record.event_id]||"";
  area.addEventListener("input",()=>{{
    if(area.value) corrections[record.event_id]=area.value;
    else delete corrections[record.event_id];
    save();
  }});
}});
document.querySelector("#export").addEventListener("click",()=>{{
  const items=records.map(record=>({{
    event_id:record.event_id,
    correction_text:(corrections[record.event_id]||"").trim(),
    original_decision:"bad",
    original_model_label:{{action:record.action,tool:record.tool,target:record.target,left:record.left,right:record.right}}
  }}));
  const payload={{format:"quiethand-m3-v1-2-corrections",version:1,exported_at:new Date().toISOString(),source_review_exported_at:sourceExportedAt,event_count:records.length,completed_count:items.filter(item=>item.correction_text).length,items}};
  const blob=new Blob([`${{JSON.stringify(payload,null,2)}}\n`],{{type:"application/json"}});
  const link=document.createElement("a");
  link.href=URL.createObjectURL(blob);
  link.download=`quiethand-m3-v1-2-corrections-${{new Date().toISOString().replace(/[:.]/g,"-")}}.json`;
  document.body.appendChild(link); link.click(); link.remove();
  setTimeout(()=>URL.revokeObjectURL(link.href),1000);
}});
update();
</script>
</body>
</html>
"""


def main() -> int:
    args = arguments()
    review = load_json(args.review)
    data = load_json(args.data)
    if review.get("format") != "quiethand-m3-v1-2-user-review":
        raise ValueError("unexpected review format")
    record_by_id = {record["event_id"]: record for record in data["records"]}
    rejected = [item for item in review.get("items", []) if item.get("decision") == "bad"]
    records = []
    for item in rejected:
        event_id = item["event_id"]
        if event_id not in record_by_id:
            raise ValueError(f"review event missing from preview data: {event_id}")
        video = args.data.parent / "source_videos" / f"{event_id}.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        records.append(record_by_id[event_id])
    if not records:
        raise ValueError("review contains no rejected records")

    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.review, args.output / "input_review.json")
    with (args.output / "data.json").open("w", encoding="utf-8") as handle:
        json.dump({"status": "READY_CORRECTION_REVIEW", "event_count": len(records), "records": records}, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    (args.output / "index.html").write_text(
        html_document(records, review.get("exported_at", "unknown")), encoding="utf-8"
    )
    print(json.dumps({"status": "READY_CORRECTION_REVIEW", "events": len(records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
