#!/usr/bin/env python3
"""Attribute one event's stability abstention without changing predictions."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from build_identity_batch_preview import decode_rgb, draw_cloud  # noqa: E402
from build_fusion_preview import decode_depth, pair_evidence  # noqa: E402
from build_spoon_entity_record import validate_temporal_plan  # noqa: E402
from diagnose_trajectories import trajectory_errors, transform_points  # noqa: E402
from quiethand.arctic_native_validation import load_mano_model  # noqa: E402
from quiethand.taco_native import _hand_parameters, _load_obj_vertices, reconstruct_taco_mano  # noqa: E402


EVENT_ID = "qh-m3-v12-cal-594fc82381d222917ad15a7b"
FRAMES = [24, 33, 41, 50, 58, 67, 75, 84, 93, 101, 110, 118, 127, 135, 144]
SELECTED = (0, 3, 4, 7, 11, 14)
STABLE_SPAN_M = 0.05
PAIRS = (
    {"hand": "left", "entity_id": "cad_049", "native_role": "target", "name_zh": "左手—木砧板"},
    {"hand": "right", "entity_id": "cad_200", "native_role": "tool", "name_zh": "右手—木勺"},
)


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def relative_span(hand_vertices: np.ndarray, object_poses: np.ndarray) -> float:
    if hand_vertices.shape != (15, 778, 3) or object_poses.shape != (15, 4, 4):
        raise ValueError("relative-span inputs must cover the same15 frames")
    centers = np.mean(np.asarray(hand_vertices, dtype=np.float64), axis=1)
    local = np.stack(
        [pose[:3, :3].T @ (center - pose[:3, 3]) for center, pose in zip(centers, object_poses, strict=True)]
    )
    low = np.quantile(local, 0.05, axis=0)
    high = np.quantile(local, 0.95, axis=0)
    value = float(np.linalg.norm(high - low))
    if not np.isfinite(value):
        raise ValueError("relative span is non-finite")
    return value


def classify_attribution(spans: dict[str, float], threshold_m: float = STABLE_SPAN_M) -> str:
    predicted = spans["predicted_hand_predicted_object"]
    hand_only = spans["predicted_hand_native_object"]
    object_only = spans["native_hand_predicted_object"]
    reference = spans["native_hand_native_object"]
    if reference > threshold_m:
        return "SOURCE_MOTION_OR_RULE_MISMATCH"
    if predicted <= threshold_m:
        return "NO_PREDICTED_STABILITY_FAILURE"
    hand_fails = hand_only > threshold_m
    object_fails = object_only > threshold_m
    if hand_fails and object_fails:
        return "BOTH_PREDICTIONS_INDIVIDUALLY_SUFFICIENT"
    if hand_fails:
        return "HAND_PREDICTION_SUFFICIENT"
    if object_fails:
        return "OBJECT_PREDICTION_SUFFICIENT"
    return "COUPLED_PREDICTION_ERROR"


def describe(code: str) -> str:
    return {
        "HAND_PREDICTION_SUFFICIENT": "只保留预测手、把物体换成原生轨迹，仍会越过5厘米门；手轨迹误差本身足以造成这次弃权。",
        "OBJECT_PREDICTION_SUFFICIENT": "只保留预测物体、把手换成原生轨迹，仍会越过5厘米门；物体轨迹误差本身足以造成这次弃权。",
        "BOTH_PREDICTIONS_INDIVIDUALLY_SUFFICIENT": "手或物体任意一个保持预测值都足以越门，两边都需要修。",
        "COUPLED_PREDICTION_ERROR": "单独替换任一组件都能过门，只有两种预测叠加时失败，属于耦合误差。",
        "SOURCE_MOTION_OR_RULE_MISMATCH": "连原生手和原生物体也越门，说明动作本身或5厘米规则不适合这段，而不是单纯预测错误。",
        "NO_PREDICTED_STABILITY_FAILURE": "完整预测本应通过稳定门，与上一步记录不一致。",
    }[code]


def make_qa(
    output: Path,
    pair: dict[str, str],
    rgbs: np.ndarray,
    intrinsic: np.ndarray,
    predicted_hand: np.ndarray,
    native_hand: np.ndarray,
    predicted_object: np.ndarray,
    native_object: np.ndarray,
    object_vertices: np.ndarray,
) -> str:
    canvas = Image.new("RGB", (1920, len(SELECTED) * 382), "white")
    painter = ImageDraw.Draw(canvas)
    hand_step = max(1, predicted_hand.shape[1] // 120)
    object_step = max(1, len(object_vertices) // 700)
    for row, index in enumerate(SELECTED):
        y = row * 382
        painter.text((8, y + 4), f"source | frame {FRAMES[index]}", fill="black")
        painter.text((648, y + 4), "prediction: cyan hand + orange object", fill="black")
        painter.text((1288, y + 4), "native QA: magenta hand + green object", fill="black")
        for col in range(3):
            canvas.paste(Image.fromarray(rgbs[index]).copy(), (col * 640, y + 22))
        predicted_panel = canvas.crop((640, y + 22, 1280, y + 382))
        predicted_cloud = object_vertices @ predicted_object[index, :3, :3].T + predicted_object[index, :3, 3]
        draw_cloud(predicted_panel, predicted_hand[index, ::hand_step], intrinsic, "#00e2ff", 2)
        draw_cloud(predicted_panel, predicted_cloud[::object_step], intrinsic, "#ff8a22", 1)
        canvas.paste(predicted_panel, (640, y + 22))
        native_panel = canvas.crop((1280, y + 22, 1920, y + 382))
        native_cloud = object_vertices @ native_object[index, :3, :3].T + native_object[index, :3, 3]
        draw_cloud(native_panel, native_hand[index, ::hand_step], intrinsic, "#ef4cff", 2)
        draw_cloud(native_panel, native_cloud[::object_step], intrinsic, "#54d66b", 1)
        canvas.paste(native_panel, (1280, y + 22))
    name = f"{pair['hand']}_{pair['entity_id']}_qa.jpg"
    canvas.save(output / name, quality=93)
    return name


def render_html(result: dict[str, Any]) -> str:
    rows = []
    cards = []
    keys = (
        ("predicted_hand_predicted_object", "预测手 + 预测物体"),
        ("predicted_hand_native_object", "预测手 + 原生物体"),
        ("native_hand_predicted_object", "原生手 + 预测物体"),
        ("native_hand_native_object", "原生手 + 原生物体"),
    )
    for pair in result["pairs"]:
        cells = "".join(
            f"<td>{100 * pair['spans_m'][key]:.2f} cm<br><small>{'通过' if pair['stable'][key] else '未通过'}</small></td>"
            for key, _ in keys
        )
        contact = pair["contact_comparison"]
        rows.append(f"<tr><th>{pair['name_zh']}<br><small>原记录接触：{contact['predicted_hand']['state']}；换原生手：{contact['native_hand']['state']}</small></th>{cells}<td><b>{pair['attribution']}</b><br>{pair['explanation_zh']}<br><small>手中心均误差 {100 * pair['hand_mean_centroid_error_m']:.2f} cm；物体中心均误差 {100 * pair['object_mean_center_error_m']:.2f} cm；最差手帧 {pair['worst_hand_frame']}。</small></td></tr>")
        cards.append(f"<section><h2>{pair['name_zh']} 可视对照</h2><img src=\"{pair['qa_image']}\"></section>")
    headers = "".join(f"<th>{label}</th>" for _, label in keys)
    return f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuietHand 手物稳定性归因</title><style>body{{font:16px/1.65 system-ui;max-width:1500px;margin:auto;padding:24px;background:#f4f6f8;color:#17212b}}section{{background:white;padding:18px;margin:18px 0;border-radius:14px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #dfe5ea;text-align:left;vertical-align:top}}small{{color:#66717d}}img,video{{width:100%;height:auto;border-radius:10px;background:#111}}.lead{{background:#fff4dd;border-left:5px solid #b66a00}}</style>
<h1>为什么本来应该跟着物体的手，被判成“不稳定”？</h1><p>只诊断事件 {EVENT_ID}；没有重跑模型，也没有修改上一条自动标签。</p>
<section><video controls muted playsinline preload="metadata" src="/artifacts/quiethand/m3_v1_2/result_preview/source_videos/{EVENT_ID}.mp4"></video></section>
<section class="lead"><h2>结论</h2><p>{result['conclusion_zh']}</p></section>
<section><h2>四种替换实验</h2><p>只替换手或物体，就能判断哪一边的预测误差足以把相对跨度推过原来的5厘米门。原生数据只做事后诊断，不回填自动标签。</p><table><thead><tr><th>诊断对</th>{headers}<th>归因</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>
{''.join(cards)}
<section><h2>能说明什么</h2><p>这能定位当前两次稳定性弃权的工程来源；不能说明全批次手姿态准确率，也不能把人工纠错直接变成训练标签。语义端tool/target都指向木勺的问题仍未修。</p><p><a href="result.json">查看结构化结果</a></p></section></html>'''


def main() -> int:
    entity_root = ROOT / "artifacts/quiethand/m3_5/entity_region_repair"
    record_path = entity_root / "spoon_final_record/record.json"
    record_bytes = record_path.read_bytes()
    record = json.loads(record_bytes)
    plan = read(ROOT / "artifacts/quiethand/m3_v1_2/QH_M3_V1_2_TEMPORAL_PLAN.json")
    validate_temporal_plan(plan)
    event = next(item for item in plan["events"] if item["event_id"] == EVENT_ID)
    if record["source_frame_indices"] != FRAMES or record["evaluation_results_opened"] is not False:
        raise ValueError("record is not the frozen same15-frame calibration event")

    data = ROOT / "external_data/taco_v1"
    binding = event["source_binding"]
    camera = np.load(data / binding["extrinsic"]["relative_path"], allow_pickle=False)[FRAMES].astype(np.float64)
    intrinsic = np.loadtxt(data / binding["intrinsic"]["relative_path"])
    sequence_rel = Path(binding["rgb"]["relative_path"]).parent.relative_to("Egocentric_RGB_Videos")
    native_hand_root = data / "Hand_Poses" / sequence_rel
    models = {
        side: load_mano_model(ROOT / f"external_data/mano_v1_2/models/MANO_{side.upper()}.pkl", side)
        for side in ("left", "right")
    }
    predicted_hands = {
        side: np.load(ROOT / record["input_provenance"][f"{side}_hand"], allow_pickle=False).astype(np.float64)
        for side in ("left", "right")
    }
    native_hands = {}
    for side in ("left", "right"):
        shape, parameters = _hand_parameters(
            native_hand_root / f"{side}_hand.pkl",
            native_hand_root / f"{side}_hand_shape.pkl",
            event["source_frame_count"],
        )
        world = np.stack(
            [reconstruct_taco_mano(models[side], parameters[frame, :48], shape, parameters[frame, 48:]) for frame in FRAMES]
        )
        native_hands[side] = transform_points(camera, world)

    predicted_objects = {
        entity_id: np.load(ROOT / record["entities"][entity_id]["pose_path"], allow_pickle=False).astype(np.float64)
        for entity_id in ("cad_049", "cad_200")
    }
    native_objects = {
        "cad_049": camera @ np.load(data / binding["target_pose"]["relative_path"], allow_pickle=False)[FRAMES],
        "cad_200": camera @ np.load(data / binding["tool_pose"]["relative_path"], allow_pickle=False)[FRAMES],
    }
    object_vertices = {
        entity_id: _load_obj_vertices(ROOT / f"artifacts/quiethand/m3_5/identity_batch/inputs/cads/{entity_id}.obj")[0]
        for entity_id in ("cad_049", "cad_200")
    }
    depth = decode_depth(data / binding["depth"]["relative_path"], FRAMES)
    masks = {
        entity_id: np.load(ROOT / record["entities"][entity_id]["mask_source"], allow_pickle=False)
        for entity_id in ("cad_049", "cad_200")
    }
    for value in [*predicted_hands.values(), *native_hands.values(), *predicted_objects.values(), *native_objects.values()]:
        if not np.isfinite(value).all():
            raise ValueError("diagnostic input is non-finite")

    output = entity_root / "hand_stability_attribution"
    output.mkdir(parents=True, exist_ok=True)
    rgbs = decode_rgb(data / binding["rgb"]["relative_path"], FRAMES)
    results = []
    for pair in PAIRS:
        side, entity_id = pair["hand"], pair["entity_id"]
        spans = {
            "predicted_hand_predicted_object": relative_span(predicted_hands[side], predicted_objects[entity_id]),
            "predicted_hand_native_object": relative_span(predicted_hands[side], native_objects[entity_id]),
            "native_hand_predicted_object": relative_span(native_hands[side], predicted_objects[entity_id]),
            "native_hand_native_object": relative_span(native_hands[side], native_objects[entity_id]),
        }
        recorded = record["entities"][entity_id]["hands"][side]["relative_motion"]["diagonal_span_m"]
        if not np.isclose(spans["predicted_hand_predicted_object"], recorded, rtol=0.0, atol=1e-9):
            raise ValueError("prediction span does not reproduce the entity record")
        attribution = classify_attribution(spans)
        if attribution in {"NO_PREDICTED_STABILITY_FAILURE"}:
            raise ValueError("recorded stability failure did not reproduce")
        hand_error = trajectory_errors(predicted_hands[side].mean(1), native_hands[side].mean(1))
        center = (object_vertices[entity_id].min(0) + object_vertices[entity_id].max(0)) / 2
        centers = np.broadcast_to(center, (15, 1, 3))
        object_error = trajectory_errors(
            transform_points(predicted_objects[entity_id], centers)[:, 0],
            transform_points(native_objects[entity_id], centers)[:, 0],
        )
        native_pair = pair_evidence(native_hands[side], masks[entity_id], predicted_objects[entity_id], depth, intrinsic, [])
        hand_frame_errors = np.asarray(hand_error["position_error_m"])
        qa_image = make_qa(
            output, pair, rgbs, intrinsic, predicted_hands[side], native_hands[side],
            predicted_objects[entity_id], native_objects[entity_id], object_vertices[entity_id],
        )
        results.append({
            **pair,
            "spans_m": spans,
            "stable": {key: value <= STABLE_SPAN_M for key, value in spans.items()},
            "attribution": attribution,
            "explanation_zh": describe(attribution),
            "hand_mean_centroid_error_m": hand_error["mean_position_error_m"],
            "hand_mean_displacement_error_m": hand_error["mean_displacement_error_m"],
            "hand_centroid_error_m": hand_error["position_error_m"],
            "worst_hand_frame": FRAMES[int(np.argmax(hand_frame_errors))],
            "worst_hand_centroid_error_m": float(hand_frame_errors.max()),
            "object_mean_center_error_m": object_error["mean_position_error_m"],
            "object_mean_displacement_error_m": object_error["mean_displacement_error_m"],
            "contact_comparison": {
                "predicted_hand": record["entities"][entity_id]["hands"][side]["contact"],
                "native_hand": native_pair["contact"],
                "surface": "same predicted entity mask plus sensor depth; native hand is post-hoc QA",
            },
            "qa_image": qa_image,
        })

    wording = "；".join(f"{item['name_zh']}：{item['explanation_zh']}" for item in results)
    result = {
        "schema": "quiethand.m3_5.hand_object_stability_attribution.v1",
        "status": "PASS_ENGINEERING_STABILITY_ATTRIBUTION",
        "event_id": EVENT_ID,
        "source_frame_indices": FRAMES,
        "stable_span_threshold_m": STABLE_SPAN_M,
        "diagnostic_pairs_selected_from_human_correction": True,
        "human_correction_used_to_rewrite_automatic_label": False,
        "native_reference_usage": "post_hoc_calibration_QA_only",
        "model_calls": 0,
        "gpu_used": False,
        "training_performed": False,
        "evaluation_results_opened": False,
        "pairs": results,
        "conclusion_zh": wording,
        "claim_boundary": "Same-event engineering root-cause attribution only; not batch hand/object accuracy, label correctness, semantic repair, or downstream utility.",
    }
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "index.html").write_text(render_html(result), encoding="utf-8")
    if record_path.read_bytes() != record_bytes:
        raise RuntimeError("prior automatic record changed during diagnosis")
    print(json.dumps({"status": result["status"], "pairs": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
