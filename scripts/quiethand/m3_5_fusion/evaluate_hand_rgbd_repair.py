#!/usr/bin/env python3
"""Post-hoc native QA for the frozen single-event RGB-D hand repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from repair_hand_rgbd import load_obj_vertices, project, read, relative_span  # noqa: E402
from diagnose_trajectories import transform_points  # noqa: E402
from quiethand.arctic_native_validation import load_mano_model  # noqa: E402
from quiethand.taco_native import _hand_parameters, reconstruct_taco_mano  # noqa: E402


SIDES = ("left", "right")


def centroid_errors(hand: np.ndarray, native: np.ndarray) -> dict[str, Any]:
    error = np.linalg.norm(hand.mean(axis=1) - native.mean(axis=1), axis=1)
    return {
        "mean_m": float(error.mean()),
        "max_m": float(error.max()),
        "per_frame_m": error.tolist(),
    }


def draw_points(image: Image.Image, points: np.ndarray, intrinsic: np.ndarray, color: str, radius: int = 2) -> None:
    draw = ImageDraw.Draw(image)
    uv = project(points, intrinsic) / 3.0
    for x, y in uv:
        if 0 <= x < image.width and 0 <= y < image.height:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def decode_rgb(video: Path, frames: list[int]) -> np.ndarray:
    expression = "+".join(f"eq(n\\,{frame})" for frame in frames)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-threads", "1", "-i", str(video),
        "-vf", f"select={expression},scale=640:360", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    raw = subprocess.check_output(command)
    if len(raw) != len(frames) * 640 * 360 * 3:
        raise ValueError("native source RGB frame count changed")
    return np.frombuffer(raw, dtype=np.uint8).reshape(len(frames), 360, 640, 3)


def comparison_image(
    output: Path,
    side: str,
    rgbs: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    native: np.ndarray,
    intrinsic: np.ndarray,
    selected: tuple[int, ...],
) -> str:
    canvas = Image.new("RGB", (1920, 382 * len(selected)), "white")
    painter = ImageDraw.Draw(canvas)
    for row, frame in enumerate(selected):
        top = 382 * row
        source = Image.fromarray(rgbs[frame]).convert("RGB")
        before_panel, after_panel, native_panel = source.copy(), source.copy(), source.copy()
        draw_points(before_panel, before[frame, ::5], intrinsic, "#00e5ff")
        draw_points(after_panel, after[frame, ::5], intrinsic, "#7dff65")
        draw_points(native_panel, native[frame, ::5], intrinsic, "#ef4cff")
        canvas.paste(before_panel, (0, top + 22))
        canvas.paste(after_panel, (640, top + 22))
        canvas.paste(native_panel, (1280, top + 22))
        painter.text((8, top + 4), f"original HaWoR | sample {frame}", fill="black")
        painter.text((648, top + 4), "RGB-D repaired", fill="black")
        painter.text((1288, top + 4), "native hand | post-hoc QA", fill="black")
    name = f"{side}_native_comparison.jpg"
    canvas.save(output / name, quality=92)
    return name


def render_html(audit: dict[str, Any]) -> str:
    rows = []
    cards = []
    for pair in audit["pairs"]:
        rows.append(
            f"<tr><th>{pair['name_zh']}</th>"
            f"<td>{100 * pair['prediction_object_before_span_m']:.2f} → {100 * pair['prediction_object_after_span_m']:.2f} cm</td>"
            f"<td>{100 * pair['native_object_before_span_m']:.2f} → {100 * pair['native_object_after_span_m']:.2f} cm</td>"
            f"<td>{100 * pair['native_centroid_before']['mean_m']:.2f} → {100 * pair['native_centroid_after']['mean_m']:.2f} cm</td>"
            f"<td>{'通过' if pair['pair_pass'] else '未通过'}</td></tr>"
        )
        side = pair["hand"]
        cards.append(
            f"<section><h2>{pair['name_zh']}：SAM mask 与修复</h2><img src=\"{audit['repair_qa_images'][side]}\"></section>"
            f"<section><h2>{pair['name_zh']}：原始 / 修复 / 原生事后对照</h2><img src=\"{pair['native_comparison_image']}\"></section>"
        )
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuietHand 单视频手轨迹 RGB-D 修复</title><style>body{{font:16px/1.65 system-ui;max-width:1500px;margin:auto;padding:24px;background:#f4f6f8;color:#17212b}}section{{background:white;padding:18px;margin:18px 0;border-radius:14px}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:10px;border-bottom:1px solid #dfe5ea;text-align:left}}img,video{{width:100%;height:auto;border-radius:10px;background:#111}}.lead{{background:#fff4dd;border-left:5px solid #b66a00}}</style>
<h1>同一视频：手轨迹修复前后</h1>
<p>只处理事件 {audit['event_id']} 的15个采样帧。SAM与RGB-D只修整只手的三维位置，HaWoR的手型、关节和朝向没有改变。</p>
<section><video controls muted playsinline preload="metadata" src="/artifacts/quiethand/m3_v1_2/result_preview/source_videos/{audit['event_id']}.mp4"></video></section>
<section class="lead"><h2>结论</h2><p>{audit['conclusion_zh']}</p></section>
<section><h2>5厘米门与真实手事后检查</h2><table><thead><tr><th>手—物体</th><th>配预测物体</th><th>配原生物体</th><th>手中心误差</th><th>结论</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
{''.join(cards)}
<section><p>原生手和原生物体只在修复文件已经写完之后做检查，没有进入SAM、点云对齐或选参。该页证明范围仅限这个校准视频的工程修复，不代表整批准确率或下游收益。</p><p><a href="audit.json">查看结构化结果</a></p></section></html>'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--repair-output", type=Path, required=True)
    args = parser.parse_args()

    cfg = read(args.input)
    repair = read(args.repair_output / "repair.json")
    event_id = cfg["event_id"]
    frames = cfg["source_frame_indices"]
    if repair["event_id"] != event_id or repair["source_frame_indices"] != frames:
        raise ValueError("repair and frozen event binding differ")
    if repair["native_hand_or_object_used"] is not False or repair["evaluation_results_opened"] is not False:
        raise ValueError("repair provenance violates the gate")
    record = read(args.workspace / cfg["entity_record"])
    before = {
        side: np.load(args.workspace / record["input_provenance"][f"{side}_hand"], allow_pickle=False).astype(np.float64)
        for side in SIDES
    }
    after = {
        side: np.load(args.repair_output / repair["hands"][side]["hand_path"], allow_pickle=False).astype(np.float64)
        for side in SIDES
    }

    plan = read(args.workspace / cfg["temporal_plan"])
    source_event = next(item for item in plan["events"] if item["event_id"] == event_id)
    binding = source_event["source_binding"]
    data = args.workspace / "external_data/taco_v1"
    intrinsic = np.loadtxt(data / binding["intrinsic"]["relative_path"]).astype(np.float64)
    rgbs = decode_rgb(data / binding["rgb"]["relative_path"], frames)
    camera = np.load(data / binding["extrinsic"]["relative_path"], allow_pickle=False)[frames].astype(np.float64)
    sequence_rel = Path(binding["rgb"]["relative_path"]).parent.relative_to("Egocentric_RGB_Videos")
    native_root = data / "Hand_Poses" / sequence_rel
    native_hands = {}
    for side in SIDES:
        model = load_mano_model(args.workspace / f"external_data/mano_v1_2/models/MANO_{side.upper()}.pkl", side)
        shape, parameters = _hand_parameters(
            native_root / f"{side}_hand.pkl", native_root / f"{side}_hand_shape.pkl", source_event["source_frame_count"]
        )
        world = np.stack([
            reconstruct_taco_mano(model, parameters[frame, :48], shape, parameters[frame, 48:]) for frame in frames
        ])
        native_hands[side] = transform_points(camera, world)
    native_objects = {
        "cad_049": camera @ np.load(data / binding["target_pose"]["relative_path"], allow_pickle=False)[frames],
        "cad_200": camera @ np.load(data / binding["tool_pose"]["relative_path"], allow_pickle=False)[frames],
    }
    predicted_objects = {
        entity_id: np.load(args.workspace / record["entities"][entity_id]["pose_path"], allow_pickle=False).astype(np.float64)
        for entity_id in ("cad_049", "cad_200")
    }

    pairs = []
    threshold = float(cfg["stable_span_threshold_m"])
    for pair in cfg["pairs"]:
        side, entity_id = pair["hand"], pair["entity_id"]
        before_error = centroid_errors(before[side], native_hands[side])
        after_error = centroid_errors(after[side], native_hands[side])
        pred_before = relative_span(before[side], predicted_objects[entity_id])
        pred_after = relative_span(after[side], predicted_objects[entity_id])
        native_before = relative_span(before[side], native_objects[entity_id])
        native_after = relative_span(after[side], native_objects[entity_id])
        pair_pass = pred_after <= threshold and native_after <= threshold and after_error["mean_m"] < before_error["mean_m"]
        pairs.append({
            **pair,
            "prediction_object_before_span_m": pred_before,
            "prediction_object_after_span_m": pred_after,
            "native_object_before_span_m": native_before,
            "native_object_after_span_m": native_after,
            "native_centroid_before": before_error,
            "native_centroid_after": after_error,
            "native_centroid_improved_frame_count": int(sum(
                a < b for a, b in zip(after_error["per_frame_m"], before_error["per_frame_m"], strict=True)
            )),
            "pair_pass": pair_pass,
            "native_comparison_image": comparison_image(
                args.repair_output, side, rgbs, before[side], after[side], native_hands[side], intrinsic,
                tuple(cfg["qa_sample_indices"]),
            ),
        })
    passed = repair["status"] == "PASS_PREDICTION_SIDE_HAND_REPAIR" and all(item["pair_pass"] for item in pairs)
    if passed:
        conclusion = "两只手在预测物体和原生物体对照下都回到5厘米以内，且相对原生手的平均中心误差同时下降；这一个样例的手根部位置修复通过。"
    else:
        failed = "、".join(item["name_zh"] for item in pairs if not item["pair_pass"])
        conclusion = f"冻结的一次修复没有闭合：{failed} 未同时满足5厘米门和原生手事后改善。结果保留，不据此调参或扩批。"
    audit = {
        "schema": "quiethand.m3_5.rgbd_hand_translation_repair_audit.v1",
        "status": "PASS_ENGINEERING_SINGLE_EVENT_HAND_REPAIR" if passed else "TERMINAL_FAIL_SINGLE_EVENT_HAND_REPAIR",
        "event_id": event_id,
        "source_frame_indices": frames,
        "stable_span_threshold_m": threshold,
        "repair_method_frozen_before_native_QA": True,
        "native_reference_usage": "post_hoc_calibration_QA_only",
        "evaluation_results_opened": False,
        "training_performed": False,
        "pairs": pairs,
        "repair_qa_images": {side: repair["hands"][side]["qa_image"] for side in SIDES},
        "conclusion_zh": conclusion,
        "claim_boundary": "One calibration event engineering repair only; not batch accuracy, annotation correctness, downstream utility, simulation, or robot evidence.",
    }
    atomic_json = args.repair_output / "audit.json"
    temporary = atomic_json.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(atomic_json)
    (args.repair_output / "index.html").write_text(render_html(audit), encoding="utf-8")
    print(json.dumps({"status": audit["status"], "pairs": pairs}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
