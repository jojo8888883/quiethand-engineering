#!/usr/bin/env python3
"""Compare old/new poses for the SAME physical entity; native poses are QA only."""

import html
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from diagnose_trajectories import draw_points, read, source_frame, trajectory_errors, transform_points
from quiethand.taco_native import _load_obj_vertices


def main():
    repair = ROOT / "artifacts/quiethand/m3_5/identity_repair"
    output = repair / "preview"
    base = ROOT / "artifacts/quiethand/m3_v1_2"
    native = ROOT / "external_data/taco_v1"
    old_root = base / "results/perception"
    new_root = repair / "perception"
    manifest = read(repair / "input.json")
    if manifest["event_count"] != 1 or manifest["evaluation_event_count"] != 0:
        raise ValueError("this preview is the single approved repair case")
    event = manifest["events"][0]
    eid = event["event_id"]
    plan = read(base / "QH_M3_V1_2_TEMPORAL_PLAN.json")
    source = next(e for e in plan["events"] if e["event_id"] == eid)
    frames = event["source_frame_indices"]
    binding = read(repair / "binding/events" / f"{eid}.json")
    old_doc = read(old_root / "events" / eid / "object_state.json")
    new_doc = read(new_root / "events" / eid / "object_state.json")
    source_binding = source["source_binding"]
    K = np.loadtxt(native / source_binding["intrinsic"]["relative_path"])
    cameras = np.load(native / source_binding["extrinsic"]["relative_path"], allow_pickle=False)[frames]
    meshes, old_poses, new_poses, references, metrics = {}, {}, {}, {}, {}
    # Native slot names are used only here to obtain QA references, NOT matching.
    for slot in ("tool", "target"):
        mesh_source = native / source_binding[f"{slot}_mesh"]["relative_path"]
        identity = "cad_" + mesh_source.stem.removesuffix("_cm")
        vertices, _ = _load_obj_vertices(mesh_source)
        meshes[identity] = vertices * .01
        references[identity] = cameras @ np.load(native / source_binding[f"{slot}_pose"]["relative_path"], allow_pickle=False)[frames]
        old_item = old_doc["items"][slot]
        old_poses[identity] = None if old_item["status"] != "observed" else np.load(old_root / old_item["artifact"]["relative_path"], allow_pickle=False)
        items = [item for item in new_doc["items"].values() if item.get("entity_id") == identity and item["status"] == "observed"]
        if len(items) > 1:
            raise ValueError("duplicate physical-entity output")
        new_poses[identity] = None if not items else np.load(new_root / items[0]["artifact"]["relative_path"], allow_pickle=False)
        center = (meshes[identity].min(0) + meshes[identity].max(0)) / 2
        centers = np.broadcast_to(center, (len(frames), 1, 3))
        reference_center = transform_points(references[identity], centers)[:, 0]
        metrics[identity] = {}
        for label, poses in (("before", old_poses[identity]), ("after", new_poses[identity])):
            metrics[identity][label] = None if poses is None else trajectory_errors(transform_points(poses, centers)[:, 0], reference_center)
    output.mkdir(parents=True, exist_ok=True)
    identities = sorted(meshes)
    colors = dict(zip(identities, ("#ec398d", "#58cd41")))
    figures = []
    for index in (0, 7, 14):
        picture = source_frame(native / source_binding["rgb"]["relative_path"], frames[index])
        for column, poses_by_id in (("source", {}), ("before", old_poses), ("after", new_poses), ("reference", references)):
            image = picture.copy()
            for identity, poses in poses_by_id.items():
                if poses is None:
                    continue
                vertices = meshes[identity][::max(1, len(meshes[identity]) // 600)]
                xyz = vertices @ poses[index, :3, :3].T + poses[index, :3, 3]
                draw_points(image, xyz, K, colors[identity], 1)
            name = f"{index:02d}_{column}.jpg"
            image.save(output / name, quality=90)
            figures.append(f'<figure><img src="{name}"><figcaption>{frames[index]/30:.2f} 秒 · 原帧 {frames[index]}</figcaption></figure>')
    table = []
    for identity, values in metrics.items():
        cells = []
        for label in ("before", "after"):
            value = values[label]
            cells.append("未知" if value is None else f'{100*value["mean_position_error_m"]:.2f} cm / {100*value["mean_displacement_error_m"]:.2f} cm')
        table.append(f'<tr><td>{identity}</td><td>{cells[0]}</td><td>{cells[1]}</td></tr>')
    payload = {"event_id": eid, "source_frame_indices": frames, "binding": binding,
        "metrics": metrics, "reference_usage": "same-entity QA only, never supplied to matching or pose inference",
        "evaluation_results_opened": False, "training_performed": False}
    (output / "comparison.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>QuietHand：物体错接修复前后</title>
<style>body{font:16px/1.65 system-ui;background:#f4f6f8;color:#182631;margin:0}main{max-width:1600px;margin:auto;padding:24px}section{background:white;padding:22px;margin:18px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}figure{margin:0}img{width:100%}figcaption{font-size:13px;color:#647381}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}pre{white-space:pre-wrap}a{color:#1b56ad}@media(max-width:900px){.grid{min-width:900px}section{overflow:auto}}</style>
<main><h1>碗配碗，盘配盘：第一条修复前后</h1><p>这里只修物体身份与 CAD 的对应关系。原来的动作描述、手部输出、分割结果及你的评语没有修改。</p>
<section><h2>同一时刻直接对比</h2><p>粉色始终是盘（cad_020），绿色始终是碗（cad_146），前后没有偷偷交换颜色。点云为透明叠加。</p><div class="grid"><b>原视频</b><b>修复前</b><b>重新匹配并估计后</b><b>原生参考，仅用于检查</b>'''+"".join(figures)+'''</div></section>
<section><h2>自动匹配结果</h2><p>同一个千问模型根据 RGB 区域和 CAD 多视角图作匹配；没给它原生位姿或人工改好的答案。</p><pre>'''+html.escape(json.dumps(binding, ensure_ascii=False, indent=2))+'''</pre></section>
<section><h2>同一个物体的误差变化</h2><p>每格：15 个采样点的平均中心位置误差 / 平均相邻采样位移误差。后者扣除真实物体运动，不把真实动作算成抖动。没有事后对齐或缩放。</p><table><tr><th>物体</th><th>修复前</th><th>修复后</th></tr>'''+"".join(table)+'''</table><p>这是单条工程修复检查，不是全数据集准确率。原生参考只用于此处对比，不进入匹配或姿态估计。</p></section>
<p><a href="../../trajectory_diagnosis/index.html">之前的诊断页</a> · <a href="../../fusion_preview/index.html">原融合页与评语</a> · <a href="comparison.json">本次对比数据</a></p></main></html>'''
    (output / "index.html").write_text(page)
    print(json.dumps({identity: {label: None if v is None else {k: v[k] for k in ("mean_position_error_m", "mean_displacement_error_m")} for label, v in value.items()} for identity, value in metrics.items()}, indent=2))


if __name__ == "__main__":
    main()
