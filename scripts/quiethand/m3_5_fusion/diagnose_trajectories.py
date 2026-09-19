#!/usr/bin/env python3
"""CPU-only same-frame calibration diagnosis. Never changes model outputs/labels."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import subprocess
import sys

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from quiethand.arctic_native_validation import load_mano_model  # noqa: E402
from quiethand.taco_native import _hand_parameters, _load_obj_vertices, reconstruct_taco_mano  # noqa: E402


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def transform_points(poses, points):
    return np.einsum("tij,tvj->tvi", poses[:, :3, :3], points) + poses[:, None, :3, 3]


def trajectory_errors(pred, reference):
    error = np.asarray(pred, dtype=float) - np.asarray(reference, dtype=float)
    return {
        "mean_position_error_m": float(np.linalg.norm(error, axis=-1).mean()),
        "mean_error_vector_m": error.mean(axis=0).tolist(),
        "mean_displacement_error_m": float(np.linalg.norm(np.diff(error, axis=0), axis=-1).mean()),
        "position_error_m": np.linalg.norm(error, axis=-1).tolist(),
        "predicted_position_m": np.asarray(pred).tolist(),
        "reference_position_m": np.asarray(reference).tolist(),
    }


def span(points):
    return float(np.linalg.norm(np.quantile(points, .95, axis=0) - np.quantile(points, .05, axis=0)))


def relative_positions(centers, poses):
    return np.einsum("tji,tj->ti", poses[:, :3, :3], centers - poses[:, :3, 3])


def load_observed(root, document, key):
    item = document["items"][key]
    if item["status"] != "observed":
        return None
    return np.load(root / item["artifact"]["relative_path"], allow_pickle=False).astype(float)


def source_frame(path, frame):
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-threads", "1",
               "-i", str(path), "-vf", f"select=eq(n\\,{frame}),scale=640:360", "-frames:v", "1",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    result = subprocess.run(command, capture_output=True, check=True)
    if len(result.stdout) != 640 * 360 * 3:
        raise ValueError("source frame decode size mismatch")
    return Image.frombytes("RGB", (640, 360), result.stdout)


def draw_points(image, points, intrinsic, color, radius):
    draw = ImageDraw.Draw(image)
    points = np.asarray(points)
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > .001)]
    uvz = points @ intrinsic.T
    uv = uvz[:, :2] / uvz[:, 2:] / 3.0
    for x, y in uv:
        if 0 <= x < image.width and 0 <= y < image.height:
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)


def summarize(records):
    result = {}
    for key in ("left", "right", "tool", "target"):
        values = [r["components"][key] for r in records if r["components"].get(key)]
        result[key] = {
            "observed_count": len(values),
            "median_clip_mean_position_error_m": float(np.median([v["mean_position_error_m"] for v in values])),
            "median_clip_mean_displacement_error_m": float(np.median([v["mean_displacement_error_m"] for v in values])),
        }
    pairs = [p for r in records for p in r["pairs"]]
    result["pair_spans"] = {
        "available_count": len(pairs),
        "predicted_under_5cm_count": sum(p["predicted_span_m"] <= .05 for p in pairs),
        "reference_under_5cm_count": sum(p["reference_span_m"] <= .05 for p in pairs),
        "meaning": "Full-action centroid envelope, NOT contact correctness or jitter accuracy.",
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/quiethand/m3_5/trajectory_diagnosis")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = ROOT / "artifacts/quiethand/m3_v1_2"
    data = ROOT / "external_data/taco_v1"
    hand_root = ROOT / "artifacts/quiethand/m3_5/results/raw_hand_metric"
    object_root = base / "results/perception"
    plan = read(base / "QH_M3_V1_2_TEMPORAL_PLAN.json")
    materialized = read(base / "QH_M3_V1_2_GEOMETRY_MATERIALIZATION.json")
    if plan["calibration_video_count"] != 30 or plan["evaluation_video_count"] != 0 or plan["evaluation_model_results_opened"]:
        raise ValueError("diagnosis requires the existing calibration-only plan")
    if materialized["evaluation_event_count"] != 0 or materialized["evaluation_model_results_opened"]:
        raise ValueError("not calibration-only materialization")
    frames_by_id = {e["event_id"]: e["frame_indices"] for e in materialized["events"]}
    models = {side: load_mano_model(ROOT / "external_data/mano_v1_2/models" / f"MANO_{side.upper()}.pkl", side)
              for side in ("left", "right")}
    records, visual = [], {}
    for event in plan["events"]:
        eid, binding = event["event_id"], event["source_binding"]
        frames = frames_by_id[eid]
        if len(frames) != 15 or np.any(np.diff(frames) <= 0):
            raise ValueError(f"invalid source frame mapping: {eid}")
        camera = np.load(data / binding["extrinsic"]["relative_path"], allow_pickle=False)[frames].astype(float)
        intrinsic = np.loadtxt(data / binding["intrinsic"]["relative_path"])
        hand_document = read(hand_root / "events" / f"{eid}.json")
        object_document = read(object_root / "events" / eid / "object_state.json")
        semantic = read(base / "results/temporal_semantic/events" / f"{eid}.json")["candidate"]
        native_hands = data / "Hand_Poses" / Path(binding["rgb"]["relative_path"]).parent.relative_to("Egocentric_RGB_Videos")
        pred_h, ref_h, pred_o, ref_o, meshes, components = {}, {}, {}, {}, {}, {}
        for side in ("left", "right"):
            betas, params = _hand_parameters(native_hands / f"{side}_hand.pkl", native_hands / f"{side}_hand_shape.pkl", event["source_frame_count"])
            native = np.stack([reconstruct_taco_mano(models[side], params[f, :48], betas, params[f, 48:]) for f in frames])
            ref_h[side] = transform_points(camera, native)
            pred_h[side] = load_observed(hand_root, hand_document, side)
            if pred_h[side] is None:
                components[side] = None
                continue
            p, g = pred_h[side], ref_h[side]
            components[side] = trajectory_errors(p.mean(1), g.mean(1))
            components[side]["mean_centered_vertex_error_m"] = float(np.linalg.norm(
                (p - p.mean(1, keepdims=True)) - (g - g.mean(1, keepdims=True)), axis=-1).mean())
        for role in ("tool", "target"):
            ref_o[role] = camera @ np.load(data / binding[f"{role}_pose"]["relative_path"], allow_pickle=False)[frames]
            pred_o[role] = load_observed(object_root, object_document, role)
            vertices, _ = _load_obj_vertices(data / binding[f"{role}_mesh"]["relative_path"])
            meshes[role] = vertices * .01
            if pred_o[role] is None:
                components[role] = None
                continue
            # Use the mesh bounding-box center, not an arbitrary mesh origin.
            center = (meshes[role].min(0) + meshes[role].max(0)) / 2
            p = transform_points(pred_o[role], np.broadcast_to(center, (15, 1, 3)))[:, 0]
            g = transform_points(ref_o[role], np.broadcast_to(center, (15, 1, 3)))[:, 0]
            components[role] = trajectory_errors(p, g)
            components[role]["mesh"] = binding[f"{role}_mesh"]["relative_path"]
            # Pose angle is NOT used as an error verdict: some objects are symmetric.
        pairs = []
        for side in ("left", "right"):
            for role in ("tool", "target"):
                if pred_h[side] is None or pred_o[role] is None:
                    continue
                p = relative_positions(pred_h[side].mean(1), pred_o[role])
                g = relative_positions(ref_h[side].mean(1), ref_o[role])
                pairs.append({"hand": side, "object": role, "predicted_span_m": span(p), "reference_span_m": span(g)})
        record = {"event_id": eid, "ordinal": event["rank"], "sequence_id": event["sequence_id"],
                  "source_frame_indices": frames, "source_interval_s": [frames[0]/30, frames[-1]/30],
                  "semantic_candidate": semantic, "components": components, "pairs": pairs}
        records.append(record)
        visual[eid] = (event, intrinsic, pred_h, ref_h, pred_o, ref_o, meshes)
        print(f"{len(records):02d}/30 {eid}", flush=True)
    summary = summarize(records)
    payload = {"status": "COMPLETE_CALIBRATION_DIAGNOSIS", "calibration_events": len(records),
               "evaluation_events": 0, "model_calls": 0, "training_performed": False,
               "native_reference_usage": "diagnosis only; no correction or label generation",
               "summary": summary, "records": records}
    (output / "measurements.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    # Fixed examples: namespace mismatch, typical tool interaction, more stable source.
    examples = [records[i] for i in (0, 12, 25)]
    cards = []
    for r in examples:
        event, K, ph, gh, po, go, meshes = visual[r["event_id"]]
        rows = []
        for index in (0, 7, 14):
            source = source_frame(data / event["source_binding"]["rgb"]["relative_path"], r["source_frame_indices"][index])
            for column, hands, objects in (("source", {}, {}), ("pred", ph, po), ("reference", gh, go)):
                picture = source.copy()
                for side, color in (("left", "#ffbf47"), ("right", "#54d9fa")):
                    if hands.get(side) is not None:
                        draw_points(picture, hands[side][index, ::5], K, color, 1)
                for role, color in (("tool", "#f961a4"), ("target", "#83e16c")):
                    if objects.get(role) is not None:
                        v = meshes[role][::max(1, len(meshes[role]) // 400)]
                        xyz = v @ objects[role][index, :3, :3].T + objects[role][index, :3, 3]
                        draw_points(picture, xyz, K, color, 1)
                filename = f"{r['ordinal']:02d}_{index:02d}_{column}.jpg"
                picture.save(output / filename, quality=87)
                rows.append(f'<figure><img loading="lazy" src="{filename}"><figcaption>原视频第 {r["source_frame_indices"][index]} 帧 · {r["source_frame_indices"][index]/30:.2f} 秒</figcaption></figure>')
        notes = "本例千问 tool=碗、target=盘；数据集 tool=盘、target=碗。两种 tool/target 的含义不同，但当前代码直接同名配对，导致 mask 与 CAD 身份不一致。" if r["ordinal"] == 1 else "这里展示原始预测，不做平滑、不换标签。原生参考只用来定位误差。"
        cards.append(f'<section><h2>样本 {r["ordinal"]}：{html.escape(r["sequence_id"])}</h2><p>{notes}</p><div class="grid"><b>原视频</b><b>当前模型预测</b><b>TACO 原生参考</b>{"".join(rows)}</div></section>')
    table_rows = []
    for r in records:
        cells = []
        for key in ("left", "right", "tool", "target"):
            value = r["components"][key]
            cells.append("缺失" if value is None else f'{value["mean_position_error_m"]*100:.1f} / {value["mean_displacement_error_m"]*100:.1f}')
        table_rows.append(f'<tr><td>{r["ordinal"]}</td><td>{html.escape(r["sequence_id"])}</td>'+"".join(f"<td>{v}</td>" for v in cells)+"</tr>")
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>QuietHand：先看清轨迹哪里错</title>
<style>body{font:16px/1.65 system-ui;margin:0;background:#f4f6f8;color:#17212b}main{max-width:1500px;margin:auto;padding:28px}section{background:white;border-radius:12px;padding:22px;margin:24px 0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}figure{margin:0}img{width:100%}figcaption{color:#65717b;font-size:13px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;border-bottom:1px solid #ddd;padding:8px}.note{background:#fff2d4;padding:15px;border-radius:10px}a{color:#235bc1}@media(max-width:750px){main{padding:12px}.grid{min-width:720px}section{overflow:auto}}</style>
<main><h1>先看清：手、物体，还是对接出了问题？</h1><p>现有 30 条校准视频的本地诊断，没有新模型推理，没有训练，没有打开 evaluation。旧网页和评语保持不变。</p>
<p class="note">发现的明确对接问题：千问的“工具/目标物”是语义角色，数据集的同名字段是固定物体身份；不能直接按字段名把分割区域和 CAD 模型接起来。另一个限制：整段相对运动大，不等于模型抖动大。</p>
<p>颜色：橙色=左手；蓝色=右手；粉色=工具；绿色=目标物。参考栏的工具/目标物按数据集定义，预测栏按当前代码输出。点云只作透明叠加，不做遮挡消隐。</p>
<p><a href="../fusion_preview/index.html">原融合网页（原评语保留）</a> · <a href="measurements.json">本轮测量数据</a></p>'''+"".join(cards)+'''<section><h2>同帧对照：30 条现有结果</h2><p>每格为“平均位置误差 / 平均位移误差”，单位 cm。位移误差比较两个相邻采样点之间的预测位移与参考位移，因此扣除了真实运动；这些采样间隔不是一帧。手比较网格中心，物体比较 CAD 包围盒中心。没有做刚体对齐或事后尺度拟合。这是定位问题，不是正式精度榜单。</p><table><tr><th>#</th><th>片段</th><th>左手</th><th>右手</th><th>工具</th><th>目标物</th></tr>'''+"".join(table_rows)+"</table></section></main></html>"
    (output / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
