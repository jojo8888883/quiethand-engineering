#!/usr/bin/env python3
"""Build calibration-only hand-object evidence fusion records and a Chinese preview."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import imageio_ffmpeg
import numpy as np


WIDTH = 1920
HEIGHT = 1080
DEPTH_SCALE = 4000.0
FRAME_COUNT = 15
MAX_SURFACE_POINTS = 2048
CLOSE_M = 0.03
NEAR_M = 0.06
MIN_EVIDENCE_FRAMES = 8
MIN_CLOSE_RUN = 4
STABLE_SPAN_M = 0.05


ACTION_ZH = {
    "aligning objects": "对齐两个物体",
    "stirring": "搅拌",
    "cutting": "切割",
    "manipulating spoon": "操作勺子",
    "pouring": "倾倒",
    "cleaning": "清洁",
    "measuring": "测量",
    "lifting and rotating": "拿起并转动",
    "manipulating objects with a slotted spoon": "用漏勺操作物体",
    "applying adhesive or coating": "滚涂材料",
    "assembling a wooden box with a roller tool": "用滚筒处理木盒",
    "removing a small object from the wheel": "从轮子上取下小物体",
    "poking": "戳刺",
    "using a lint roller on a kettle": "用粘毛滚筒滚压水壶",
    "erasing": "擦除",
    "brushing": "刷洗",
    "grinding": "打磨",
    "washing": "清洗",
    "pouring tea": "倒茶",
    "pouring liquid": "倾倒液体",
    "straining": "用漏勺捞取或过滤",
}

OBJECT_ZH = {
    "black bowl": "黑色碗",
    "cream tray": "浅色托盘",
    "wooden spoon": "木勺",
    "white container": "白色容器",
    "cleaver": "菜刀",
    "tray": "托盘",
    "blue bowl": "蓝色碗",
    "black pot": "黑色锅",
    "wooden spatula": "木铲",
    "pot": "锅",
    "brush": "刷子",
    "teapot": "茶壶",
    "green lint roller": "绿色粘毛滚筒",
    "black wok": "黑色炒锅",
    "ruler": "尺子",
    "wheel": "轮子",
    "unknown": "未知物体",
    "wooden box": "木盒",
    "slotted spoon": "漏勺",
    "roller": "滚筒",
    "wooden cutting board": "木砧板",
    "yellow bowl": "黄色碗",
    "roller tool": "滚筒工具",
    "screwdriver": "螺丝刀",
    "knife": "刀",
    "spatula": "铲子",
    "bowl": "碗",
    "lint roller": "粘毛滚筒",
    "kettle": "水壶",
    "red eraser": "红色橡皮",
    "white perforated tray": "白色漏孔托盘",
    "black tray": "黑色托盘",
    "digital caliper": "数显卡尺",
    "ceramic cup": "陶瓷杯",
    "white rectangular device": "白色长方形工具",
    "pan": "平底锅",
    "grinder": "打磨工具",
    "tea tray": "茶盘",
    "metal plate": "金属盘",
    "yellow pitcher": "黄色壶",
    "white pitcher": "白色壶",
    "contents of the container": "容器中的内容物",
    "ladle": "长柄勺",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--hand-results", type=Path, required=True)
    parser.add_argument("--perception-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def item_array(root: Path, document: dict[str, Any], name: str) -> tuple[np.ndarray | None, str | None]:
    item = document.get("items", {}).get(name, {})
    status = item.get("status", "missing")
    artifact = item.get("artifact")
    if status != "observed" or not isinstance(artifact, dict):
        return None, item.get("failure_reason") or f"{name}_{status}"
    path = root / artifact["relative_path"]
    return np.load(path, allow_pickle=False, mmap_mode="r"), None


def decode_depth(video: Path, frame_indices: list[int]) -> np.ndarray:
    expression = "+".join(f"eq(n\\,{index})" for index in frame_indices)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-threads", "1", "-i", str(video),
        "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", f"select={expression}",
        "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "gray16le", "pipe:1",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    expected = FRAME_COUNT * HEIGHT * WIDTH * 2
    if result.returncode != 0 or result.stderr.strip() or len(result.stdout) != expected:
        error = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"depth decode failed: {error or f'{len(result.stdout)} bytes'}")
    return np.frombuffer(result.stdout, dtype="<u2").reshape(FRAME_COUNT, HEIGHT, WIDTH)


def visible_surface_points(depth_raw: np.ndarray, mask: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    valid = np.asarray(mask, dtype=bool) & (depth_raw >= 4)
    flat = np.flatnonzero(valid.reshape(-1))
    if flat.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if flat.size > MAX_SURFACE_POINTS:
        chosen = np.linspace(0, flat.size - 1, MAX_SURFACE_POINTS, dtype=np.int64)
        flat = flat[chosen]
    rows, cols = np.divmod(flat, WIDTH)
    z = depth_raw.reshape(-1)[flat].astype(np.float32) / np.float32(DEPTH_SCALE)
    fx, fy = np.float32(intrinsic[0, 0]), np.float32(intrinsic[1, 1])
    cx, cy = np.float32(intrinsic[0, 2]), np.float32(intrinsic[1, 2])
    x = (cols.astype(np.float32) - cx) * z / fx
    y = (rows.astype(np.float32) - cy) * z / fy
    return np.column_stack((x, y, z)).astype(np.float32, copy=False)


def nearest_distance(hand_vertices: np.ndarray, surface_points: np.ndarray) -> float | None:
    hand = np.asarray(hand_vertices, dtype=np.float32)
    surface = np.asarray(surface_points, dtype=np.float32)
    if hand.shape != (778, 3) or surface.ndim != 2 or surface.shape[1:] != (3,) or len(surface) == 0:
        return None
    hand_sq = np.sum(hand * hand, axis=1, keepdims=True)
    surface_sq = np.sum(surface * surface, axis=1)[None, :]
    squared = hand_sq + surface_sq - np.float32(2.0) * (hand @ surface.T)
    minimum = float(np.min(squared))
    return math.sqrt(max(0.0, minimum))


def longest_true_run(values: list[bool]) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def classify_contact(distances: list[float | None]) -> dict[str, Any]:
    valid = [value for value in distances if value is not None]
    close_flags = [value is not None and value <= CLOSE_M for value in distances]
    near_flags = [value is not None and value <= NEAR_M for value in distances]
    close_count = sum(close_flags)
    near_count = sum(near_flags)
    longest = longest_true_run(close_flags)
    if len(valid) < MIN_EVIDENCE_FRAMES:
        state = "abstain"
    elif close_count >= MIN_EVIDENCE_FRAMES and longest >= MIN_CLOSE_RUN:
        state = "maintained_close"
    elif near_count >= MIN_EVIDENCE_FRAMES:
        state = "boundary_near"
    else:
        state = "not_close"
    return {
        "state": state,
        "valid_frame_count": len(valid),
        "close_frame_count": close_count,
        "near_frame_count": near_count,
        "longest_close_run": longest,
        "minimum_distance_m": min(valid) if valid else None,
        "median_distance_m": float(np.median(valid)) if valid else None,
    }


def relative_envelope(hand_vertices: np.ndarray, poses: np.ndarray) -> dict[str, Any] | None:
    if hand_vertices.shape != (FRAME_COUNT, 778, 3) or poses.shape != (FRAME_COUNT, 4, 4):
        return None
    centers = np.mean(hand_vertices.astype(np.float64), axis=1)
    local = np.stack([pose[:3, :3].T @ (center - pose[:3, 3]) for center, pose in zip(centers, poses, strict=True)])
    low = np.quantile(local, 0.05, axis=0)
    high = np.quantile(local, 0.95, axis=0)
    axis_span = high - low
    diagonal = float(np.linalg.norm(axis_span))
    return {
        "relative_position_p05_m": low.tolist(),
        "relative_position_p95_m": high.tolist(),
        "axis_span_m": axis_span.tolist(),
        "diagonal_span_m": diagonal,
        "stable_with_object": diagonal <= STABLE_SPAN_M,
        "state": "uncalibrated_candidate",
    }


def pair_evidence(
    hand: np.ndarray | None,
    mask: np.ndarray | None,
    poses: np.ndarray | None,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    missing_reasons: list[str],
) -> dict[str, Any]:
    if hand is None or mask is None or poses is None:
        return {
            "status": "abstain",
            "failure_reason": "; ".join(reason for reason in missing_reasons if reason) or "required_input_missing",
            "distance_m": [None] * FRAME_COUNT,
            "contact": classify_contact([None] * FRAME_COUNT),
            "relative_motion": None,
            "surface_point_count": [0] * FRAME_COUNT,
        }
    distances: list[float | None] = []
    counts = []
    for index in range(FRAME_COUNT):
        points = visible_surface_points(depth[index], mask[index], intrinsic)
        counts.append(len(points))
        distances.append(nearest_distance(hand[index], points))
    return {
        "status": "observed",
        "failure_reason": None,
        "distance_m": distances,
        "contact": classify_contact(distances),
        "relative_motion": relative_envelope(np.asarray(hand), np.asarray(poses)),
        "surface_point_count": counts,
    }


def role_fusion(qwen_role: str, tool: dict[str, Any], target: dict[str, Any]) -> dict[str, str]:
    tool_active = tool["contact"]["state"] == "maintained_close" and bool((tool.get("relative_motion") or {}).get("stable_with_object"))
    target_support = target["contact"]["state"] == "maintained_close" and bool((target.get("relative_motion") or {}).get("stable_with_object"))
    geometry = ({"active"} if tool_active else set()) | ({"support"} if target_support else set())
    qwen = {qwen_role} if qwen_role in {"active", "support"} else ({"active", "support"} if qwen_role == "both" else set())
    if geometry == {"active", "support"}:
        geometry_role = "both_candidates"
    elif geometry == {"active"}:
        geometry_role = "active_candidate"
    elif geometry == {"support"}:
        geometry_role = "support_candidate"
    else:
        geometry_role = "undetermined"
    if not geometry:
        relation = "geometry_undetermined"
        final = "qwen_coarse_only" if qwen else "undetermined"
    elif geometry == qwen:
        relation = "consistent"
        final = geometry_role
    elif geometry & qwen:
        relation = "partial_consistency"
        final = geometry_role
    else:
        relation = "conflict"
        final = "ambiguous_conflict"
    return {"qwen_role": qwen_role, "geometry_role": geometry_role, "relation": relation, "final_candidate": final}


def zh_object(value: str | None) -> str:
    if not value:
        return "未知物体"
    return OBJECT_ZH.get(value, value)


def zh_action(value: str | None) -> str:
    if not value:
        return "动作未知"
    return ACTION_ZH.get(value, value)


def zh_role(value: str | None) -> str:
    return {
        "active": "主动操作手",
        "support": "扶持稳定手",
        "both": "同时承担两种角色",
        "neither": "未承担主动或扶持角色",
        "unknown": "角色未知",
    }.get(value or "unknown", value or "角色未知")


def role_sentence(side_zh: str, fusion: dict[str, str], tool: dict[str, Any], target: dict[str, Any]) -> str:
    geometry = fusion["geometry_role"]
    relation = fusion["relation"]
    if geometry == "support_candidate":
        base = f"{side_zh}持续靠近目标物并随目标物稳定移动，属于支持手候选"
    elif geometry == "active_candidate":
        base = f"{side_zh}持续靠近工具并随工具稳定移动，属于主动手候选"
    elif geometry == "both_candidates":
        base = f"{side_zh}同时满足工具和目标物的稳定近接条件，角色仍有歧义"
    else:
        tool_state = tool["contact"]["state"]
        target_state = target["contact"]["state"]
        base = f"{side_zh}的几何证据不足以定角色（工具：{tool_state}；目标物：{target_state}）"
    suffix = {
        "consistent": "，与千问粗标签一致。",
        "partial_consistency": "，与千问粗标签部分一致。",
        "conflict": "，与千问粗标签冲突，因此不自动定论。",
        "geometry_undetermined": "；这里只保留千问粗标签，不把它升级为几何确认。",
    }[relation]
    return base + suffix


def event_class(left: dict[str, str], right: dict[str, str]) -> str:
    relations = {left["relation"], right["relation"]}
    if "conflict" in relations:
        return "conflict"
    if relations <= {"consistent", "partial_consistency"}:
        return "consistent"
    if relations & {"consistent", "partial_consistency"}:
        return "partial"
    return "undetermined"


def html_document(records: list[dict[str, Any]], counts: dict[str, int]) -> str:
    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuietHand M3.5 细粒度融合结果</title>
<style>
:root{{--bg:#f4f6f8;--card:#fff;--ink:#17212b;--muted:#66717d;--line:#dce2e8;--blue:#2563eb;--green:#17864b;--amber:#b66a00;--red:#c63e4c}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}header{{background:#17212b;color:white;padding:38px max(24px,calc((100vw - 1450px)/2)) 26px}}h1{{margin:0 0 10px;font-size:32px}}header p{{max-width:1000px;margin:7px 0;color:#d5dde5}}.metrics{{display:flex;gap:12px;flex-wrap:wrap;margin-top:20px}}.metric{{min-width:135px;padding:12px 15px;border:1px solid #40505f;border-radius:12px;background:#202d39}}.metric b{{display:block;font-size:27px}}.toolbar{{position:sticky;top:0;z-index:4;display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:12px max(24px,calc((100vw - 1450px)/2));background:#fffffff2;border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}}button,input,textarea{{border:1px solid var(--line);border-radius:9px;background:white;padding:8px 12px;font:inherit}}button{{cursor:pointer}}.toolbar button.active{{color:white;background:#17212b}}input{{min-width:260px}}.review-progress{{margin-left:auto;color:var(--muted);font-weight:700;white-space:nowrap}}.export{{border-color:#2563eb;color:#164bb8;font-weight:700}}main{{max-width:1450px;margin:auto;padding:24px}}.card{{background:var(--card);border:1px solid var(--line);border-radius:15px;overflow:hidden;margin-bottom:22px;box-shadow:0 8px 24px #22303c0c}}.card.consistent{{border-left:5px solid var(--green)}}.card.partial,.card.undetermined{{border-left:5px solid var(--amber)}}.card.conflict{{border-left:5px solid var(--red)}}.card.review-ok{{box-shadow:0 0 0 2px #17864b55,0 8px 24px #22303c0c}}.card.review-bad{{box-shadow:0 0 0 2px #c63e4c55,0 8px 24px #22303c0c}}.card.review-other{{box-shadow:0 0 0 2px #2563eb55,0 8px 24px #22303c0c}}.layout{{display:grid;grid-template-columns:minmax(360px,0.85fr) minmax(520px,1.45fr)}}video{{width:100%;height:100%;min-height:330px;object-fit:contain;background:#0b1015}}.body{{padding:20px}}.top{{display:flex;align-items:start;justify-content:space-between;gap:15px}}code{{color:var(--muted);font-size:12px}}.pill{{padding:3px 10px;border-radius:999px;font-size:12px;font-weight:700;background:#eef2f5}}.consistent .pill{{color:var(--green);background:#e6f6ed}}.conflict .pill{{color:var(--red);background:#fdecef}}.partial .pill,.undetermined .pill{{color:var(--amber);background:#fff4dd}}h2{{margin:8px 0 2px;font-size:23px}}.coarse{{margin:11px 0;padding:12px 14px;background:#f3f7fb;border-radius:10px}}.hands{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}}.hand{{border:1px solid var(--line);border-radius:11px;padding:12px}}.hand h3{{margin:0 0 7px}}.pair{{margin-top:8px;padding-top:8px;border-top:1px dashed var(--line)}}.pair b{{display:inline-block;min-width:62px}}.muted{{color:var(--muted)}}.correction{{margin-top:12px;padding:11px 13px;border-radius:10px;background:#fff8e6;border:1px solid #f1d899}}.review{{margin-top:15px;padding:15px;border:1px solid #cfd8e1;border-radius:12px;background:#f8fafc}}.review-title{{font-size:16px;font-weight:800}}.review-help{{margin:3px 0 10px;color:var(--muted)}}.review-actions{{display:flex;gap:8px;flex-wrap:wrap}}.review-actions button{{min-width:90px}}.review-actions button.selected[data-choice="ok"]{{border-color:var(--green);background:#e6f6ed;color:var(--green);font-weight:800}}.review-actions button.selected[data-choice="bad"]{{border-color:var(--red);background:#fdecef;color:var(--red);font-weight:800}}.review-actions button.selected[data-choice="other"]{{border-color:var(--blue);background:#eaf1ff;color:var(--blue);font-weight:800}}.review textarea{{display:block;width:100%;min-height:82px;margin-top:10px;resize:vertical;line-height:1.55}}.review textarea:focus{{outline:2px solid #2563eb33;border-color:var(--blue)}}.review-state{{margin-top:6px;color:var(--muted);font-size:13px}}.spark{{width:100%;height:34px;margin-top:4px}}details{{margin-top:12px}}summary{{cursor:pointer;color:var(--blue)}}.limit{{max-width:1450px;margin:18px auto 0;padding:0 24px;color:#cfd8e1}}@media(max-width:900px){{.layout{{grid-template-columns:1fr}}video{{min-height:240px}}.hands{{grid-template-columns:1fr}}.review-progress{{margin-left:0}}}}
</style></head><body>
<header><h1>QuietHand M3.5：从“在做什么”到“手如何约束物体”</h1><p>左侧是完整动作视频。右侧先保留千问直接看视频得到的粗语义，再展示 HaWoR 手网格、SAM 物体区域、传感器深度和 FoundationPose 物体位姿融合出的时序证据。</p><p>看完每条后，请在卡片底部选择“OK / 不行 / 其他”并按需写评语。这里的“支持手/主动手”是校准集候选；运动范围是未校准候选，不是真值、物理证书或机器人控制指令。</p>
<div class="metrics"><div class="metric"><b>{len(records)}</b><span>校准视频</span></div><div class="metric"><b>{counts.get('consistent',0)}</b><span>几何与粗标签一致</span></div><div class="metric"><b>{counts.get('partial',0)}</b><span>部分一致</span></div><div class="metric"><b>{counts.get('conflict',0)}</b><span>明确冲突</span></div><div class="metric"><b>{counts.get('undetermined',0)}</b><span>几何未定</span></div></div></header>
<div class="limit">固定规则：≤3 cm 且至少 8/15 帧、连续至少 4 帧才算持续近接；3–6 cm 只标边界；物体坐标系内跨度≤5 cm 才算稳定随物。你的旧审核只作旁注展示，没有参与规则或结果生成。</div>
<div class="toolbar"><button class="active" data-filter="all">全部</button><button data-filter="consistent">一致</button><button data-filter="partial">部分一致</button><button data-filter="conflict">冲突</button><button data-filter="undetermined">未定</button><button data-filter="human_bad">你上次判不行</button><input id="search" placeholder="搜索动作、物体或事件"><span class="review-progress" id="review-progress">已评 0 / {len(records)}</span><button class="export" id="export-review" type="button">导出本轮评审</button></div><main id="root"></main>
<script>
const records={payload};let active='all';const root=document.querySelector('#root'),search=document.querySelector('#search');
const reviewStorageKey='quiethand-m3-5-fusion-review-v1';let reviews={{}};
try{{reviews=JSON.parse(localStorage.getItem(reviewStorageKey)||'{{}}')}}catch(_){{reviews={{}}}}
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}}[c]));
const stateZh=s=>({{maintained_close:'持续近接',boundary_near:'边界近接',not_close:'未持续靠近',abstain:'证据缺失'}}[s]||s);
const relationZh=s=>({{consistent:'一致',partial_consistency:'部分一致',conflict:'冲突',geometry_undetermined:'几何未定'}}[s]||s);
function cm(v){{return v==null?'—':(v*100).toFixed(1)+' cm'}}
function spark(values,color){{const ys=values.map(v=>v==null?null:Math.min(8,v*100));const pts=ys.map((v,i)=>v==null?null:`${{i*100/14}},${{30-v*3.4}}`).filter(Boolean).join(' ');return `<svg class="spark" viewBox="0 0 100 34" preserveAspectRatio="none"><line x1="0" x2="100" y1="19.8" y2="19.8" stroke="#e8b45e" stroke-dasharray="2 2"/><line x1="0" x2="100" y1="9.6" y2="9.6" stroke="#6bb78d" stroke-dasharray="2 2"/><polyline points="${{pts}}" fill="none" stroke="${{color}}" stroke-width="1.7" vector-effect="non-scaling-stroke"/></svg>`}}
function pair(name,p,color){{const c=p.contact,rel=p.relative_motion;const bounds=rel?`<details><summary>三轴相对位置范围</summary><div class="muted">x：${{cm(rel.relative_position_p05_m[0])}} 到 ${{cm(rel.relative_position_p95_m[0])}}；y：${{cm(rel.relative_position_p05_m[1])}} 到 ${{cm(rel.relative_position_p95_m[1])}}；z：${{cm(rel.relative_position_p05_m[2])}} 到 ${{cm(rel.relative_position_p95_m[2])}}。状态：未校准候选。</div></details>`:'';return `<div class="pair"><b>${{name}}</b> ${{stateZh(c.state)}} · 近接 ${{c.close_frame_count}}/15 · 最短 ${{cm(c.minimum_distance_m)}} · 中位 ${{cm(c.median_distance_m)}}${{rel?` · 相对跨度 ${{cm(rel.diagonal_span_m)}} · ${{rel.stable_with_object?'稳定随物':'相对移动较大'}}`:''}}${{spark(p.distance_m,color)}}${{bounds}}</div>`}}
function hand(sideZh,h){{return `<div class="hand"><h3>${{sideZh}}：${{relationZh(h.fusion.relation)}}</h3><div>${{esc(h.sentence_zh)}}</div>${{pair('工具',h.pairs.tool,'#2563eb')}}${{pair('目标物',h.pairs.target,'#c63e4c')}}<details><summary>查看运动限制候选</summary><div class="muted">${{[h.pairs.tool,h.pairs.target].some(p=>p.relative_motion&&p.contact.state==='maintained_close')?'列出的相对跨度和三轴范围只表示当前15帧模型证据。':'没有满足持续近接条件，因此不生成运动限制候选。'}}</div></details></div>`}}
function reviewBox(r){{return `<div class="review"><div class="review-title">你认为这整条结果怎么样？</div><div class="review-help">语义和手物关系都对就选 OK；核心内容错误选“不行”；颜色、措辞或局部问题选“其他”并说明。</div><div class="review-actions"><button type="button" data-choice="ok">OK</button><button type="button" data-choice="bad">不行</button><button type="button" data-choice="other">其他</button></div><textarea aria-label="对 ${{esc(r.event_id)}} 的评语" placeholder="可写：实际发生了什么、哪只手/哪个物体判断错了，或哪里只是小问题。"></textarea><div class="review-state">尚未评价</div></div>`}}
function completed(review){{return Boolean(review&&review.decision)}}
function saveReviews(){{localStorage.setItem(reviewStorageKey,JSON.stringify(reviews))}}
function updateProgress(){{const count=records.filter(r=>completed(reviews[r.event_id])).length;document.querySelector('#review-progress').textContent=`已评 ${{count}} / ${{records.length}}`}}
function applyReviewState(card){{const id=card.dataset.event,review=reviews[id]||{{}},decision=review.decision||'';card.classList.remove('review-ok','review-bad','review-other');if(decision)card.classList.add(`review-${{decision}}`);card.querySelectorAll('[data-choice]').forEach(button=>{{const selected=button.dataset.choice===decision;button.classList.toggle('selected',selected);button.setAttribute('aria-pressed',selected?'true':'false')}});const label={{ok:'OK',bad:'不行',other:'其他'}}[decision];card.querySelector('.review-state').textContent=label?`已自动保存：${{label}}${{review.comment_text?.trim()?'，评语也已保存':''}}`:(review.comment_text?.trim()?'评语已自动保存，请再选择一项':'尚未评价')}}
function bindReviews(){{root.querySelectorAll('.card').forEach(card=>{{const id=card.dataset.event,area=card.querySelector('textarea'),review=reviews[id]||{{}};area.value=review.comment_text||'';card.querySelectorAll('[data-choice]').forEach(button=>button.onclick=()=>{{reviews[id]={{...(reviews[id]||{{}}),decision:button.dataset.choice,comment_text:area.value}};saveReviews();applyReviewState(card);updateProgress()}});area.oninput=()=>{{const decision=(reviews[id]||{{}}).decision;reviews[id]={{decision,comment_text:area.value}};if(!decision&&!area.value)delete reviews[id];saveReviews();applyReviewState(card)}};applyReviewState(card)}});updateProgress()}}
function render(){{const q=search.value.trim().toLowerCase();const shown=records.filter(r=>{{const match=active==='all'||r.classification===active||(active==='human_bad'&&r.human_review.decision==='bad');return match&&(!q||JSON.stringify(r).toLowerCase().includes(q))}});root.innerHTML=shown.map(r=>`<article class="card ${{r.classification}}" data-event="${{esc(r.event_id)}}"><div class="layout"><video controls muted playsinline preload="none" data-src="../../m3_v1_2/result_preview/source_videos/${{esc(r.event_id)}}.mp4"></video><div class="body"><div class="top"><code>${{esc(r.event_id)}}</code><span class="pill">${{esc(r.classification_zh)}}</span></div><h2>${{esc(r.semantic.action_zh)}}</h2><div class="coarse"><b>千问粗标签：</b>${{esc(r.semantic.sentence_zh)}}</div><div class="hands">${{hand('左手',r.hands.left)}}${{hand('右手',r.hands.right)}}</div>${{r.human_review.decision==='bad'?`<div class="correction"><b>你上次判定：不行</b><br>${{esc(r.human_review.correction_text||'未填写纠错')}}</div>`:`<div class="muted" style="margin-top:10px">你上次判定：OK</div>`}}${{reviewBox(r)}}</div></div></article>`).join('');bindReviews();lazy()}}
function lazy(){{const io=new IntersectionObserver(es=>es.forEach(e=>{{if(e.isIntersecting){{e.target.src=e.target.dataset.src;e.target.load();io.unobserve(e.target)}}}}),{{rootMargin:'500px'}});document.querySelectorAll('video[data-src]').forEach(v=>io.observe(v))}}
document.querySelectorAll('.toolbar button[data-filter]').forEach(b=>b.onclick=()=>{{document.querySelector('.toolbar button[data-filter].active').classList.remove('active');b.classList.add('active');active=b.dataset.filter;render()}});search.oninput=render;
document.querySelector('#export-review').onclick=()=>{{const items=records.map(r=>{{const review=reviews[r.event_id]||{{}};return{{event_id:r.event_id,decision:review.decision||'unreviewed',comment_text:(review.comment_text||'').trim(),fusion_classification:r.classification,previous_review_decision:r.human_review.decision,previous_correction_text:r.human_review.correction_text||null}}}});const output={{format:'quiethand-m3-5-fusion-review',version:1,exported_at:new Date().toISOString(),event_count:records.length,completed_count:items.filter(item=>item.decision!=='unreviewed').length,items}};const blob=new Blob([JSON.stringify(output,null,2)+'\\n'],{{type:'application/json'}});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download=`quiethand-m3-5-fusion-review-${{new Date().toISOString().replace(/[:.]/g,'-')}}.json`;document.body.appendChild(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(link.href),1000)}};
render();
</script></body></html>"""


def main() -> int:
    args = arguments()
    workspace = args.workspace.resolve()
    p0 = load_json(workspace / "artifacts/quiethand/m3_5/QH_M3_5_METRIC_HAND_VERIFICATION.json")
    if p0.get("status") != "PASS_METRIC_ALIGNMENT" or p0.get("evaluation_event_count") != 0:
        raise SystemExit("P0 metric alignment has not passed")
    plan = load_json(workspace / "artifacts/quiethand/m3_v1_2/QH_M3_V1_2_TEMPORAL_PLAN.json")
    materialization = load_json(workspace / "artifacts/quiethand/m3_v1_2/QH_M3_V1_2_GEOMETRY_MATERIALIZATION.json")
    if plan.get("calibration_video_count") != 30 or plan.get("evaluation_video_count") != 0 or len(plan.get("events", [])) != 30:
        raise SystemExit("temporal plan is not calibration 30 / evaluation 0")
    frame_by_id = {event["event_id"]: event for event in materialization["events"]}
    review = load_json(workspace / "artifacts/quiethand/m3_v1_2/correction_review/input_review.json")
    corrections = load_json(workspace / "artifacts/quiethand/m3_v1_2/correction_review/quiethand-m3-v1-2-corrections-2026-08-31T09-08-43-578Z.json")
    review_by_id = {item["event_id"]: item for item in review["items"]}
    correction_by_id = {item["event_id"]: item["correction_text"] for item in corrections["items"]}
    records = []
    for ordinal, event in enumerate(plan["events"], start=1):
        event_id = event["event_id"]
        frame_record = frame_by_id[event_id]
        if frame_record.get("status") != "ready_geometry_inference" or len(frame_record.get("frame_indices", [])) != FRAME_COUNT:
            raise RuntimeError(f"{event_id}: geometry interval unavailable")
        semantic_document = load_json(workspace / "artifacts/quiethand/m3_v1_2/results/temporal_semantic/events" / f"{event_id}.json")
        candidate = semantic_document.get("candidate") or {}
        hand_document = load_json(args.hand_results / "events" / f"{event_id}.json")
        segmentation_document = load_json(args.perception_results / "events" / event_id / "segmentation.json")
        object_document = load_json(args.perception_results / "events" / event_id / "object_state.json")
        intrinsic_relative = event["source_binding"]["intrinsic"]["relative_path"]
        intrinsic = np.loadtxt(workspace / "external_data/taco_v1" / intrinsic_relative, dtype=np.float64)
        depth_path = workspace / "external_data/taco_v1" / event["source_binding"]["depth"]["relative_path"]
        depth = decode_depth(depth_path, frame_record["frame_indices"])
        hands = {}
        for side, side_zh in (("left", "左手"), ("right", "右手")):
            hand, hand_reason = item_array(args.hand_results, hand_document, side)
            pairs = {}
            for role in ("tool", "target"):
                mask, mask_reason = item_array(args.perception_results, segmentation_document, role)
                poses, pose_reason = item_array(args.perception_results, object_document, role)
                pairs[role] = pair_evidence(hand, mask, poses, depth, intrinsic, [hand_reason, mask_reason, pose_reason])
            fusion = role_fusion((candidate.get(side) or {}).get("role", "unknown"), pairs["tool"], pairs["target"])
            hands[side] = {
                "qwen": candidate.get(side) or {"role": "unknown", "contact": "unknown"},
                "pairs": pairs,
                "fusion": fusion,
                "sentence_zh": role_sentence(side_zh, fusion, pairs["tool"], pairs["target"]),
            }
        classification = event_class(hands["left"]["fusion"], hands["right"]["fusion"])
        human = review_by_id.get(event_id, {"decision": "unreviewed"})
        action = candidate.get("action_candidate")
        tool = candidate.get("tool_candidate")
        target = candidate.get("target_candidate")
        records.append({
            "event_id": event_id,
            "classification": classification,
            "classification_zh": {"consistent": "一致", "partial": "部分一致", "conflict": "冲突", "undetermined": "几何未定"}[classification],
            "semantic": {
                "action_original": action,
                "action_zh": zh_action(action),
                "tool_original": tool,
                "tool_zh": zh_object(tool),
                "target_original": target,
                "target_zh": zh_object(target),
                "sentence_zh": f"模型认为动作是“{zh_action(action)}”，工具是“{zh_object(tool)}”，目标物是“{zh_object(target)}”；左手是{zh_role((candidate.get('left') or {}).get('role'))}，右手是{zh_role((candidate.get('right') or {}).get('role'))}。",
            },
            "hands": hands,
            "human_review": {
                "decision": human.get("decision", "unreviewed"),
                "correction_text": correction_by_id.get(event_id),
                "used_for_fusion": False,
            },
            "evaluation_results_opened": False,
        })
        print(f"[{ordinal:02d}/30] {classification} {event_id}", flush=True)
    counts = {name: sum(record["classification"] == name for record in records) for name in ("consistent", "partial", "conflict", "undetermined")}
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "data.json", {"status": "READY_M3_5_FUSION_PREVIEW", "event_count": 30, "classification_counts": counts, "records": records})
    audit = {
        "schema": "quiethand.m3_5.fusion_audit.v1",
        "status": "READY_M3_5_FUSION_PREVIEW",
        "calibration_event_count": 30,
        "evaluation_event_count": 0,
        "evaluation_results_opened": False,
        "classification_counts": counts,
        "explicit_pair_state_count": sum(len(record["hands"]) * 2 for record in records),
        "thresholds": {
            "close_m": CLOSE_M,
            "near_m": NEAR_M,
            "minimum_evidence_frames": MIN_EVIDENCE_FRAMES,
            "minimum_close_run": MIN_CLOSE_RUN,
            "stable_relative_span_m": STABLE_SPAN_M,
            "surface_points_per_frame_max": MAX_SURFACE_POINTS,
        },
        "evidence": ["Qwen3-VL coarse semantics", "HaWoR metric hand mesh", "SAM2.1 object mask", "TACO sensor depth", "FoundationPose object pose"],
        "human_corrections_used_for_fusion": False,
        "c_oame_status": "uncalibrated_candidates_only",
        "training_performed": False,
    }
    atomic_json(args.output / "audit.json", audit)
    (args.output / "index.html").write_text(html_document(records, counts), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
