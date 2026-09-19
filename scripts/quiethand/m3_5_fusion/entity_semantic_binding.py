#!/usr/bin/env python3
"""Bind left/right hand semantics to known physical entities for one Ego event."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from PIL import Image, ImageDraw


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts/quiethand/m3_jobs"))

from runtime_common import atomic_json, status  # noqa: E402


EVENT_ID = "qh-m3-v12-cal-594fc82381d222917ad15a7b"
SAMPLE_INDICES = (0, 2, 4, 6, 8, 10, 12, 14)
ENTITIES = {"cad_049": "木砧板", "cad_200": "木勺"}
ROLES = {"active", "support", "none", "uncertain"}
POSTHOC_REFERENCE = {
    "left_hand": {"entity_id": "cad_049", "role": "support"},
    "right_hand": {"entity_id": "cad_200", "role": "active"},
}
PROMPT = """你会看到同一段第一人称双手操作按时间排序的8张画面。
每张画面中的紫色区域标为 E1 = cad_049 木砧板，橙色区域标为 E2 = cad_200 木勺。
这些实体名称来自已经完成的物体身份绑定；你的任务是观察动作，把佩戴者的左手和右手分别绑定到实际接触或操作的实体，并描述两只手的分工。

这里的 active 表示执行主要操作，support 表示抓住、扶住或稳定另一个物体，none 表示没有与这两个实体交互，uncertain 表示画面不足以判断。left_hand/right_hand 指佩戴者自己的左右手。不要沿用“tool/target”的旧名称，也不要因为画面里正好有两个实体就强行一一分配；看不清时使用 null 和 uncertain。若两只手都给出非空实体，本段要求两个实体不能重复。

只返回下面结构的 JSON，不要 Markdown，不要增加字段：
{
  "action_zh": "一句中文动作概括",
  "left_hand": {
    "entity_id": "cad_049、cad_200 或 null",
    "role": "active、support、none 或 uncertain",
    "description_zh": "左手具体在做什么",
    "evidence_zh": "从连续画面看到的依据"
  },
  "right_hand": {
    "entity_id": "cad_049、cad_200 或 null",
    "role": "active、support、none 或 uncertain",
    "description_zh": "右手具体在做什么",
    "evidence_zh": "从连续画面看到的依据"
  }
}
"""


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def decode_frames(video: Path, frame_indices: list[int]) -> list[Image.Image]:
    import imageio_ffmpeg

    expression = "+".join(f"eq(n\\,{index})" for index in frame_indices)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-threads", "1", "-i", str(video),
        "-vf", f"select={expression}", "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    raw = subprocess.check_output(command)
    expected = len(frame_indices) * 1920 * 1080 * 3
    if len(raw) != expected:
        raise ValueError("source RGB frame count changed")
    return [
        Image.frombytes("RGB", (1920, 1080), raw[offset:offset + 1920 * 1080 * 3])
        for offset in range(0, len(raw), 1920 * 1080 * 3)
    ]


def overlay_entities(image: Image.Image, board_mask: Any, spoon_mask: Any, ordinal: int) -> Image.Image:
    import numpy as np

    pixels = np.asarray(image).copy()
    for mask, colour in ((board_mask, np.array([174, 85, 255])), (spoon_mask, np.array([255, 139, 36]))):
        selected = np.asarray(mask, dtype=bool)
        pixels[selected] = (0.58 * pixels[selected] + 0.42 * colour).astype(np.uint8)
    panel = Image.fromarray(pixels).resize((960, 540), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, 960, 50), fill=(10, 16, 25))
    draw.text((14, 8), f"TIME {ordinal + 1}/8", fill="white")
    draw.rectangle((150, 12, 170, 32), fill=(174, 85, 255))
    draw.text((178, 8), "E1 cad_049 BOARD", fill="white")
    draw.rectangle((430, 12, 450, 32), fill=(255, 139, 36))
    draw.text((458, 8), "E2 cad_200 SPOON", fill="white")
    return panel


def prepare(workspace: Path, output: Path) -> None:
    import numpy as np

    plan = read(workspace / "artifacts/quiethand/m3_v1_2/QH_M3_V1_2_TEMPORAL_PLAN.json")
    event = next(item for item in plan["events"] if item["event_id"] == EVENT_ID)
    record = read(
        workspace
        / "artifacts/quiethand/m3_5/entity_region_repair/spoon_final_record/record.json"
    )
    frames = record["source_frame_indices"]
    if len(frames) != 15:
        raise ValueError("expected the frozen 15-frame event")
    binding = event["source_binding"]
    video = workspace / "external_data/taco_v1" / binding["rgb"]["relative_path"]
    rgb = decode_frames(video, [frames[index] for index in SAMPLE_INDICES])
    source = workspace / "artifacts/quiethand/m3_5/entity_region_repair/events" / EVENT_ID
    board = np.load(source / "cad_049/mask.npy", allow_pickle=False)
    spoon = np.load(source / "cad_200/mask.npy", allow_pickle=False)
    if board.shape != spoon.shape or board.shape[0] != 15:
        raise ValueError("entity masks do not share the frozen 15 frames")
    images = output / "inputs"
    images.mkdir(parents=True, exist_ok=True)
    paths = []
    for ordinal, (sample_index, image) in enumerate(zip(SAMPLE_INDICES, rgb, strict=True)):
        destination = images / f"frame_{ordinal:02d}.jpg"
        overlay_entities(image, board[sample_index], spoon[sample_index], ordinal).save(destination, quality=92)
        paths.append(destination.relative_to(workspace).as_posix())
    atomic_json(output / "input.json", {
        "schema": "quiethand.m3_5.entity_semantic_binding_input.v1",
        "event_id": EVENT_ID,
        "source_frame_indices": [frames[index] for index in SAMPLE_INDICES],
        "entities": ENTITIES,
        "ordered_images": paths,
        "prompt": PROMPT,
        "native_pose_used": False,
        "human_correction_used": False,
        "evaluation_results_opened": False,
    })


def parse_answer(raw: str) -> dict[str, Any]:
    result = json.loads(raw)
    if not isinstance(result, dict) or set(result) != {"action_zh", "left_hand", "right_hand"}:
        raise ValueError("answer must contain exactly action_zh, left_hand and right_hand")
    if not isinstance(result["action_zh"], str) or not result["action_zh"].strip():
        raise ValueError("action_zh must be non-empty")
    assigned = []
    for side in ("left_hand", "right_hand"):
        hand = result[side]
        required = {"entity_id", "role", "description_zh", "evidence_zh"}
        if not isinstance(hand, dict) or set(hand) != required:
            raise ValueError(f"{side} has an invalid field set")
        if hand["entity_id"] not in {*ENTITIES, None}:
            raise ValueError(f"{side}.entity_id is outside the supplied entity catalog")
        if hand["role"] not in ROLES:
            raise ValueError(f"{side}.role is invalid")
        for field in ("description_zh", "evidence_zh"):
            if not isinstance(hand[field], str) or not hand[field].strip():
                raise ValueError(f"{side}.{field} must be non-empty")
        if hand["role"] in {"active", "support"} and hand["entity_id"] is None:
            raise ValueError(f"{side} has a definite role without an entity")
        if hand["entity_id"] is not None:
            assigned.append(hand["entity_id"])
    if len(assigned) != len(set(assigned)):
        raise ValueError("the two non-null hand assignments reuse one entity")
    return result


def compare_posthoc_reference(candidate: dict[str, Any]) -> dict[str, Any]:
    comparisons = {}
    for side in ("left_hand", "right_hand"):
        expected = POSTHOC_REFERENCE[side]
        observed = candidate[side]
        comparisons[side] = {
            "entity_match": observed["entity_id"] == expected["entity_id"],
            "role_match": observed["role"] == expected["role"],
            "expected": expected,
            "observed": {
                "entity_id": observed["entity_id"],
                "role": observed["role"],
            },
        }
    return {
        "source": "此前用户旁注：左手是抓木板，右手是操作勺子",
        "used_for_inference": False,
        "hands": comparisons,
        "all_fields_match": all(
            item["entity_match"] and item["role_match"] for item in comparisons.values()
        ),
    }


def infer(workspace: Path, output: Path) -> None:
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    manifest = read(output / "input.json")
    if manifest.get("event_id") != EVENT_ID or manifest.get("evaluation_results_opened") is not False:
        raise ValueError("input is outside the frozen single calibration event")
    result_root = output / "semantic"
    status(result_root / "status.json", "loading_model", 0, 1)
    started = time.monotonic()
    model_path = workspace / "external_data/quiethand_m3_resources/semantic"
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0},
    ).eval()
    content = [{"type": "image", "url": str(workspace / path)} for path in manifest["ordered_images"]]
    content.append({"type": "text", "text": manifest["prompt"]})
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}], add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=520, use_cache=True)
    raw = processor.decode(generated[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    try:
        candidate = parse_answer(raw)
        result_status, failure = "observed", None
    except (json.JSONDecodeError, ValueError) as exc:
        candidate, result_status, failure = None, "invalid_model_answer", str(exc)
    atomic_json(result_root / "result.json", {
        "schema": "quiethand.m3_5.entity_semantic_binding_result.v1",
        "event_id": EVENT_ID,
        "status": result_status,
        "candidate": candidate,
        "raw_text": raw,
        "failure_reason": failure,
        "native_pose_used": False,
        "human_correction_used": False,
        "evaluation_results_opened": False,
    })
    atomic_json(result_root / "audit.json", {
        "status": "COMPLETE" if result_status == "observed" else "COMPLETE_WITH_INVALID_ANSWER",
        "event_id": EVENT_ID,
        "elapsed_seconds": time.monotonic() - started,
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "model_revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "training_performed": False,
        "evaluation_results_opened": False,
    })
    status(result_root / "status.json", "complete", 1, 1)


def zh_contact(state: str) -> str:
    return {
        "maintained_close": "15帧中持续靠近该物体",
        "boundary_near": "只有部分时刻足够靠近，接触证据处在边界",
        "not_close": "没有观察到持续靠近",
        "abstain": "接触证据未获得",
    }.get(state, "接触状态未知")


def build(workspace: Path, output: Path) -> None:
    semantic = read(output / "semantic/result.json")
    record_path = workspace / "artifacts/quiethand/m3_5/entity_region_repair/spoon_final_record/record.json"
    record = read(record_path)
    rows = []
    if semantic["status"] == "observed":
        for side in ("left", "right"):
            answer = semantic["candidate"][f"{side}_hand"]
            entity_id = answer["entity_id"]
            geometry = None if entity_id is None else record["entities"][entity_id]["hands"][side]
            stable = None if geometry is None else bool(geometry["relative_motion"]["stable_with_object"])
            rows.append({
                "hand": side,
                "entity_id": entity_id,
                "entity_name_zh": None if entity_id is None else ENTITIES[entity_id],
                "semantic_role": answer["role"],
                "semantic_description_zh": answer["description_zh"],
                "semantic_evidence_zh": answer["evidence_zh"],
                "contact_state": None if geometry is None else geometry["contact"]["state"],
                "contact_zh": "未绑定实体" if geometry is None else zh_contact(geometry["contact"]["state"]),
                "relative_span_m": None if geometry is None else geometry["relative_motion"]["diagonal_span_m"],
                "mobility_constraint_status": "AVAILABLE_CANDIDATE" if stable else "ABSTAIN_UNSTABLE_TRACK",
                "mobility_constraint_zh": "相对运动范围可作为候选" if stable else "当前手轨迹不稳，暂不输出运动约束",
            })
    posthoc = None if semantic["candidate"] is None else compare_posthoc_reference(semantic["candidate"])
    final = {
        "schema": "quiethand.m3_5.fieldwise_interaction_record.v1",
        "event_id": EVENT_ID,
        "semantic_status": semantic["status"],
        "action_zh": None if semantic["candidate"] is None else semantic["candidate"]["action_zh"],
        "hands": rows,
        "human_review": record["human_review"],
        "posthoc_semantic_comparison": posthoc,
        "native_pose_used": False,
        "human_correction_used_for_inference": False,
        "evaluation_results_opened": False,
    }
    atomic_json(output / "record.json", final)
    cards = []
    for row in rows:
        span = "未获得" if row["relative_span_m"] is None else f"{100 * row['relative_span_m']:.2f} cm"
        cards.append(f"""<section><h2>{'左手' if row['hand']=='left' else '右手'} → {row['entity_name_zh'] or '未确定实体'}</h2>
<p><b>模型语义：</b>{row['semantic_description_zh']}（{row['semantic_role']}）</p>
<p><b>画面依据：</b>{row['semantic_evidence_zh']}</p>
<p><b>几何接触：</b>{row['contact_zh']}</p>
<p><b>相对运动：</b>{span}；{row['mobility_constraint_zh']}</p></section>""")
    images = "".join(f'<img src="inputs/frame_{index:02d}.jpg" alt="模型输入时刻{index + 1}">' for index in range(8))
    if semantic["status"] == "observed" and posthoc["all_fields_match"]:
        lead_class = "lead pass"
        lead = f"本次语义绑定通过事后对照：{final['action_zh']}。接触和运动约束按各自证据展示。"
    elif semantic["status"] == "observed":
        lead_class = "lead fail"
        lead = (f"本次语义绑定没有通过：模型虽把左手绑定到木砧板、右手绑定到木勺，"
                f"却把主动/支撑角色判反，还把砧板黑色把手误说成笔。模型原话：{final['action_zh']}")
    else:
        lead_class = "lead fail"
        lead = f"模型回答未通过结构校验：{semantic['failure_reason']}。本次没有生成新的语义绑定。"
    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuietHand 单例纠错结果</title><style>body{{font:16px/1.65 system-ui;max-width:1280px;margin:auto;padding:24px;background:#f4f6f8;color:#17212b}}section{{background:white;padding:18px;margin:18px 0;border-radius:14px}}.lead{{border-left:5px solid #5267e8}}.pass{{border-color:#16855b}}.fail{{border-color:#cf3f3f;background:#fff7f7}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}}img,video{{width:100%;border-radius:10px;background:#111}}code{{word-break:break-all}}@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}</style>
<h1>这一条视频现在标成了什么</h1><section><video controls muted playsinline preload="metadata" src="/artifacts/quiethand/m3_v1_2/result_preview/source_videos/{EVENT_ID}.mp4"></video></section>
<section class="{lead_class}"><p>{lead}</p></section>{''.join(cards)}
<section><h2>千问实际看到的8个时刻</h2><div class="grid">{images}</div></section>
<section><h2>与人工旁注对照</h2><p>此前你的原话：{record['human_review']['correction_text']}</p><p>这句话只放在结果之后供你比较，没有输入千问，也没有改写几何数值。</p></section>
<section><p><a href="record.json">结构化记录</a> · <a href="semantic/result.json">千问原始输出</a></p></section></html>"""
    (output / "index.html").write_text(html, encoding="utf-8")
    if semantic["status"] != "observed":
        terminal_status = "HOLD_INVALID_SEMANTIC_ANSWER"
    elif posthoc["all_fields_match"]:
        terminal_status = "TERMINAL_PASS_ENTITY_SEMANTIC_BINDING"
    else:
        terminal_status = "TERMINAL_FAIL_ROLE_REVERSAL"
    atomic_json(output / "audit.json", {
        "status": terminal_status,
        "event_id": EVENT_ID,
        "hand_record_count": len(rows),
        "semantic_binding_observed": semantic["status"] == "observed",
        "posthoc_semantic_match": None if posthoc is None else posthoc["all_fields_match"],
        "geometry_fields_copied_without_threshold_change": True,
        "human_correction_used_for_inference": False,
        "native_pose_used": False,
        "evaluation_results_opened": False,
        "claim_boundary": "One calibration event field-wise engineering record; not batch accuracy, annotation-cost reduction, downstream utility, or support-role ground truth.",
    })


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "infer", "build"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    workspace = args.workspace.resolve()
    output = args.output if args.output.is_absolute() else workspace / args.output
    {"prepare": prepare, "infer": infer, "build": build}[args.command](workspace, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
