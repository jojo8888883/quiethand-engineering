#!/usr/bin/env python3
"""Build the M3-v1.2 complete-action calibration result page."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np


COLORS = {
    "left": (255, 170, 50),
    "right": (60, 210, 255),
    "tool": (40, 130, 255),
    "target": (255, 190, 40),
    "x": (40, 40, 255),
    "y": (40, 220, 40),
    "z": (255, 80, 40),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--semantic-root", type=Path, required=True)
    parser.add_argument("--raw-hand-root", type=Path, required=True)
    parser.add_argument("--perception-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def item_summary(document: dict[str, Any], names: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    items = document.get("items", {})
    return {
        name: {
            "status": items.get(name, {}).get("status", "missing"),
            "failure_reason": items.get(name, {}).get("failure_reason"),
        }
        for name in names
    }


def load_array(root: Path, document: dict[str, Any], item: str) -> np.ndarray | None:
    payload = document.get("items", {}).get(item, {})
    artifact = payload.get("artifact")
    if payload.get("status") != "observed" or not artifact:
        return None
    return np.load(root / artifact["relative_path"], allow_pickle=False)


def project(points: np.ndarray, intrinsic: np.ndarray, width: int, height: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-5)
    points = points[valid]
    if not len(points):
        return np.empty((0, 2), dtype=np.int32)
    pixels = (intrinsic @ points.T).T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    visible = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    return np.rint(pixels[visible]).astype(np.int32)


def draw_hand(frame: np.ndarray, vertices: np.ndarray | None, frame_index: int, intrinsic: np.ndarray, color: tuple[int, int, int]) -> None:
    if vertices is None:
        return
    height, width = frame.shape[:2]
    pixels = project(vertices[frame_index][::8], intrinsic, width, height)
    for x, y in pixels:
        cv2.circle(frame, (int(x), int(y)), 2, color, -1, cv2.LINE_AA)


def draw_axes(frame: np.ndarray, poses: np.ndarray | None, frame_index: int, intrinsic: np.ndarray) -> None:
    if poses is None:
        return
    pose = np.asarray(poses[frame_index], dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return
    origin = pose[:3, 3]
    axis_length = 0.06
    points = np.vstack(
        [
            origin,
            origin + pose[:3, 0] * axis_length,
            origin + pose[:3, 1] * axis_length,
            origin + pose[:3, 2] * axis_length,
        ]
    )
    pixels = project(points, intrinsic, frame.shape[1], frame.shape[0])
    if len(pixels) != 4:
        return
    origin_pixel = tuple(int(value) for value in pixels[0])
    for endpoint, key in zip(pixels[1:], ("x", "y", "z"), strict=True):
        cv2.line(
            frame,
            origin_pixel,
            tuple(int(value) for value in endpoint),
            COLORS[key],
            3,
            cv2.LINE_AA,
        )


def blend_mask(frame: np.ndarray, mask: np.ndarray | None, frame_index: int, color: tuple[int, int, int]) -> None:
    if mask is None:
        return
    active = np.asarray(mask[frame_index], dtype=bool)
    if active.shape != frame.shape[:2]:
        return
    frame[active] = np.clip(
        frame[active].astype(np.float32) * 0.52 + np.asarray(color, dtype=np.float32) * 0.48,
        0,
        255,
    ).astype(np.uint8)


def draw_first_frame_boxes(frame: np.ndarray, candidate: dict[str, Any], frame_index: int) -> None:
    if frame_index != 0:
        return
    boxes = candidate.get("start_sample_boxes_0_1000", {})
    height, width = frame.shape[:2]
    for name in ("tool", "target"):
        box = boxes.get(name)
        if not isinstance(box, list) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(value) for value in box]
        start = (round(x1 * width / 1000), round(y1 * height / 1000))
        end = (round(x2 * width / 1000), round(y2 * height / 1000))
        cv2.rectangle(frame, start, end, COLORS[name], 3, cv2.LINE_AA)
        cv2.putText(frame, f"initial {name}", (start[0], max(28, start[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS[name], 2, cv2.LINE_AA)


def draw_banner(frame: np.ndarray, candidate: dict[str, Any], status_line: str) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 106), (10, 14, 22), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    action = candidate.get("action_candidate") or "unknown action"
    tool = candidate.get("tool_candidate") or "unknown tool"
    target = candidate.get("target_candidate") or "unknown target"
    left = candidate.get("left", {})
    right = candidate.get("right", {})
    lines = [
        f"{action} | tool: {tool} | target: {target}",
        f"LEFT {left.get('role', '?')} / {left.get('contact', '?')}    RIGHT {right.get('role', '?')} / {right.get('contact', '?')}",
        status_line,
    ]
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (24, 31 + index * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)


def classification(semantic: dict[str, Any], raw: dict[str, Any], segmentation: dict[str, Any], objects: dict[str, Any]) -> str:
    statuses = [semantic.get("status", "missing")]
    statuses.extend(item.get("status", "missing") for item in raw.get("items", {}).values())
    statuses.extend(item.get("status", "missing") for item in segmentation.get("items", {}).values())
    statuses.extend(item.get("status", "missing") for item in objects.get("items", {}).values())
    if statuses and all(status == "observed" for status in statuses):
        return "complete"
    if any(status == "observed" for status in statuses):
        return "partial"
    return "failed"


def render_event(
    event_id: str,
    input_root: Path,
    output_video: Path,
    fps: int,
    semantic: dict[str, Any],
    raw: dict[str, Any],
    segmentation: dict[str, Any],
    objects: dict[str, Any],
    raw_root: Path,
    perception_root: Path,
) -> None:
    intrinsic = np.load(input_root / event_id / "intrinsic.npy", allow_pickle=False)
    candidate = semantic.get("candidate") or {}
    hands = {name: load_array(raw_root, raw, name) for name in ("left", "right")}
    masks = {name: load_array(perception_root, segmentation, name) for name in ("tool", "target")}
    poses = {name: load_array(perception_root, objects, name) for name in ("tool", "target")}
    output_video.parent.mkdir(parents=True, exist_ok=True)
    first_frame = cv2.imread(str(input_root / event_id / "rgb_00.png"), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"failed to read RGB for {event_id}")
    height, width = first_frame.shape[:2]
    hand_intrinsic = np.asarray(
        [[600.0, 0.0, 960.0], [0.0, 600.0, 540.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "24",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video),
    ]
    status_line = (
        f"semantic={semantic.get('status', 'missing')} | "
        f"hands={','.join(raw.get('items', {}).get(name, {}).get('status', 'missing') for name in ('left', 'right'))} | "
        f"masks={','.join(segmentation.get('items', {}).get(name, {}).get('status', 'missing') for name in ('tool', 'target'))}"
    )
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        if process.stdin is None:
            raise RuntimeError("ffmpeg stdin unavailable")
        for frame_index in range(15):
            frame = cv2.imread(str(input_root / event_id / f"rgb_{frame_index:02d}.png"), cv2.IMREAD_COLOR)
            if frame is None:
                process.kill()
                raise RuntimeError(f"failed to read frame {frame_index} for {event_id}")
            blend_mask(frame, masks["tool"], frame_index, COLORS["tool"])
            blend_mask(frame, masks["target"], frame_index, COLORS["target"])
            draw_hand(frame, hands["left"], frame_index, hand_intrinsic, COLORS["left"])
            draw_hand(frame, hands["right"], frame_index, hand_intrinsic, COLORS["right"])
            draw_axes(frame, poses["tool"], frame_index, intrinsic)
            draw_axes(frame, poses["target"], frame_index, intrinsic)
            draw_first_frame_boxes(frame, candidate, frame_index)
            draw_banner(frame, candidate, status_line)
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed for {event_id}: {return_code}")


def transcode_source_video(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ],
        check=True,
    )
    temporary.replace(destination)


def html_document(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    metrics = {
        "events": len(records),
        "semantic": sum(record["semantic_status"] == "observed" for record in records),
        "geometry": sum(record["geometry_status"] == "ready_geometry_inference" for record in records),
        "complete": sum(record["classification"] == "complete" for record in records),
    }
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuietHand M3-v1.2 完整动作结果</title>
<style>
:root {{ color-scheme: dark; --bg:#081018; --card:#111d29; --line:#24384a; --text:#edf5fb; --muted:#91a6b8; --good:#4ade80; --partial:#fbbf24; --bad:#fb7185; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:radial-gradient(circle at top,#132638 0,#081018 48%); color:var(--text); font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
header {{ max-width:1500px; margin:auto; padding:42px 28px 18px; }} h1 {{ margin:0 0 8px; font-size:34px; }} .lead {{ color:var(--muted); max-width:920px; margin:0; }}
.metrics {{ display:grid; grid-template-columns:repeat(5,minmax(130px,1fr)); gap:12px; margin-top:24px; }} .metric {{ background:#0d1924cc; border:1px solid var(--line); border-radius:14px; padding:15px; }} .metric b {{ display:block; font-size:28px; }} .metric span {{ color:var(--muted); }}
.toolbar {{ position:sticky; top:0; z-index:10; backdrop-filter:blur(16px); background:#081018e8; border-block:1px solid var(--line); padding:12px 28px; display:flex; gap:10px; flex-wrap:wrap; }}
button,input {{ border:1px solid var(--line); background:#10202d; color:var(--text); padding:9px 13px; border-radius:10px; }} button.active {{ border-color:#69b7ef; background:#17344a; }} input {{ min-width:260px; }}
main {{ max-width:1500px; margin:auto; padding:24px 28px 60px; }} .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(420px,1fr)); gap:18px; }}
.card {{ border:1px solid var(--line); background:linear-gradient(145deg,#132230,#0d1822); border-radius:16px; overflow:hidden; box-shadow:0 12px 30px #0005; }} .card.complete {{ border-top:3px solid var(--good); }} .card.partial {{ border-top:3px solid var(--partial); }} .card.failed {{ border-top:3px solid var(--bad); }}
video {{ display:block; width:100%; aspect-ratio:16/9; object-fit:cover; background:#000; }} .body {{ padding:14px 16px 17px; }} .title {{ display:flex; justify-content:space-between; gap:12px; align-items:center; }} .title code {{ color:#c9d9e6; font-size:12px; }} .pill {{ border-radius:999px; padding:3px 9px; font-size:12px; font-weight:700; }} .complete .pill {{ color:var(--good); background:#123622; }} .partial .pill {{ color:var(--partial); background:#382c0e; }} .failed .pill {{ color:var(--bad); background:#3a1620; }} .interval {{ margin:8px 0; color:#a7d8ff; }} .overlay {{ margin-top:10px; }}
.judgement {{ font-size:17px; font-weight:700; margin:10px 0 4px; }} .detail {{ color:var(--muted); }} .states {{ margin-top:10px; display:grid; grid-template-columns:repeat(2,1fr); gap:6px 14px; font-size:13px; }} .states b {{ color:#dceaf4; }} a {{ color:#7dd3fc; }}
.legend,.limit {{ margin:18px 0; padding:13px 15px; border:1px solid var(--line); background:#0d1924; border-radius:12px; color:var(--muted); }} .swatch {{ display:inline-block; width:11px; height:11px; margin:0 5px 0 14px; border-radius:50%; }}
@media(max-width:700px) {{ .metrics {{ grid-template-columns:repeat(2,1fr); }} .grid {{ grid-template-columns:1fr; }} header,main {{ padding-inline:14px; }} .toolbar {{ padding-inline:14px; }} }}
</style>
</head>
<body>
<header><h1>QuietHand M3-v1.2：完整动作重跑结果</h1><p class="lead">每张卡片先播放 3.7–8.9 秒的完整源视频；模型看完整段之后选择真正发生动作的区间，再只在该区间运行手、分割和物体位姿。</p>
<div class="metrics">
<div class="metric"><b>{metrics['events']}</b><span>完整视频</span></div><div class="metric"><b>{metrics['semantic']}</b><span>有语义输出</span></div><div class="metric"><b>{metrics['geometry']}</b><span>动作区间有效</span></div><div class="metric"><b>{metrics['complete']}</b><span>全部模态完整</span></div>
</div><div class="legend"><span class="swatch" style="background:#ff8228"></span>tool mask <span class="swatch" style="background:#28beff"></span>target mask <span class="swatch" style="background:#32aaff"></span>left hand <span class="swatch" style="background:#ffd23c"></span>right hand</div>
<div class="limit">这次修正的是评估输入：不再把一段动作硬切成五个 0.5 秒片段。旧 150 条审核没有被带进来；这是独立的新一轮 30 条结果。</div></header>
<div class="toolbar"><button class="active" data-filter="all">全部</button><button data-filter="complete">完整</button><button data-filter="partial">部分</button><button data-filter="failed">失败</button><input id="search" placeholder="搜索动作、物体或事件 ID"></div>
<main><div class="grid" id="grid"></div></main>
<script>
const records={payload}; let active='all'; const grid=document.querySelector('#grid'); const search=document.querySelector('#search');
const esc=s=>String(s??'').replace(/[&<>\"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',"'":'&#039;'}}[c]));
function stateText(items){{return Object.entries(items).map(([k,v])=>`<div><b>${{esc(k)}}</b> ${{esc(v.status)}}${{v.failure_reason?` · ${{esc(v.failure_reason)}}`:''}}</div>`).join('')}}
function render(){{const q=search.value.trim().toLowerCase(); const shown=records.filter(r=>(active==='all'||r.classification===active)&&(!q||JSON.stringify(r).toLowerCase().includes(q))); grid.innerHTML=shown.map(r=>`<article class="card ${{r.classification}}"><video controls muted playsinline preload="none" data-src="source_videos/${{esc(r.event_id)}}.mp4?v=20260830-h264"></video><div class="body"><div class="title"><code>${{esc(r.event_id)}}</code><span class="pill">${{r.classification==='complete'?'完整':r.classification==='partial'?'部分':'失败'}}</span></div><div class="interval">模型选择的动作区间：${{esc(r.interval_text)}}</div><div class="judgement">${{esc(r.action)}} · ${{esc(r.tool)}} → ${{esc(r.target)}}</div><div class="detail">左手：${{esc(r.left.role)}} / ${{esc(r.left.contact)}}　右手：${{esc(r.right.role)}} / ${{esc(r.right.contact)}}</div><div class="states"><div><b>semantic</b> ${{esc(r.semantic_status)}}</div><div><b>temporal</b> ${{esc(r.geometry_status)}}</div>${{stateText(r.raw_hand)}}${{stateText(r.segmentation)}}${{stateText(r.object_state)}}</div>${{r.has_overlay?`<details class="overlay"><summary>查看动作区间的几何叠加</summary><video controls muted loop playsinline preload="none" data-src="overlay_videos/${{esc(r.event_id)}}.mp4?v=20260830-h264"></video></details>`:''}}</div></article>`).join(''); lazy();}}
function lazy(){{const load=v=>{{if(!v.src){{v.src=v.dataset.src;v.load();}}}}; const io=new IntersectionObserver(entries=>entries.forEach(e=>{{if(e.isIntersecting){{load(e.target);io.unobserve(e.target);}}}}),{{rootMargin:'500px'}}); document.querySelectorAll('video[data-src]').forEach(v=>io.observe(v)); document.querySelectorAll('details.overlay').forEach(d=>d.addEventListener('toggle',()=>{{if(d.open)load(d.querySelector('video'));}}));}}
document.querySelectorAll('button[data-filter]').forEach(b=>b.onclick=()=>{{document.querySelector('button.active').classList.remove('active');b.classList.add('active');active=b.dataset.filter;render();}}); search.oninput=render; render();
</script>
<script src="review.js?v=20260830-v12"></script>
</body></html>"""


def main() -> int:
    args = arguments()
    workspace = args.workspace.resolve()
    task_plan = load_json(workspace / "artifacts/quiethand/m3_v1_2/QH_M3_V1_2_TEMPORAL_PLAN.json")
    geometry = load_json(workspace / "artifacts/quiethand/m3_v1_2/QH_M3_V1_2_GEOMETRY_MATERIALIZATION.json")
    input_root = workspace / "artifacts/quiethand/m3_v1_2/geometry_inputs"
    geometry_by_id = {event["event_id"]: event for event in geometry["events"]}
    records: list[dict[str, Any]] = []
    videos = args.output / "overlay_videos"
    source_videos = args.output / "source_videos"
    source_videos.mkdir(parents=True, exist_ok=True)
    tasks = task_plan["events"][: args.limit] if args.limit else task_plan["events"]
    for index, task in enumerate(tasks, start=1):
        event_id = task["event_id"]
        semantic = load_json(args.semantic_root / "events" / f"{event_id}.json")
        geometry_event = geometry_by_id[event_id]
        ready = geometry_event["status"] == "ready_geometry_inference"
        raw_path = args.raw_hand_root / "events" / f"{event_id}.json"
        segmentation_path = args.perception_root / "events" / event_id / "segmentation.json"
        object_path = args.perception_root / "events" / event_id / "object_state.json"
        raw = load_json(raw_path) if raw_path.is_file() else {"items": {}}
        segmentation = load_json(segmentation_path) if segmentation_path.is_file() else {"items": {}}
        objects = load_json(object_path) if object_path.is_file() else {"items": {}}
        candidate = semantic.get("candidate") or {}
        start = candidate.get("action_start_sample")
        end = candidate.get("action_end_sample")
        if isinstance(start, int) and isinstance(end, int):
            start_frame = task["sample_frame_indices"][start]
            end_frame = task["sample_frame_indices"][end]
            interval_text = f"{start_frame / 30:.2f}–{(end_frame + 1) / 30:.2f} 秒"
        else:
            interval_text = "未选出有效动作区间"
        record = {
            "event_id": event_id,
            "classification": classification(semantic, raw, segmentation, objects),
            "semantic_status": semantic.get("status", "missing"),
            "semantic_failure_reason": semantic.get("failure_reason"),
            "geometry_status": geometry_event["status"],
            "interval_text": interval_text,
            "has_overlay": ready,
            "action": candidate.get("action_candidate") or "unknown",
            "tool": candidate.get("tool_candidate") or "unknown",
            "target": candidate.get("target_candidate") or "unknown",
            "left": candidate.get("left") or {"role": "unknown", "contact": "unknown"},
            "right": candidate.get("right") or {"role": "unknown", "contact": "unknown"},
            "evidence": candidate.get("evidence") or "unknown",
            "raw_hand": item_summary(raw, ("left", "right")),
            "segmentation": item_summary(segmentation, ("tool", "target")),
            "object_state": item_summary(objects, ("tool", "target")),
        }
        records.append(record)
        source = workspace / "external_data/taco_v1" / task["source_rgb"]["relative_path"]
        source_destination = source_videos / f"{event_id}.mp4"
        if args.force or not source_destination.is_file():
            transcode_source_video(source, source_destination)
        output_video = videos / f"{event_id}.mp4"
        if ready and (args.force or not output_video.is_file() or output_video.stat().st_size == 0):
            render_event(
                event_id,
                input_root,
                output_video,
                args.fps,
                semantic,
                raw,
                segmentation,
                objects,
                args.raw_hand_root,
                args.perception_root,
            )
        print(f"[{index:03d}/{len(tasks):03d}] {record['classification']} {event_id}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__).with_name("review.js"), args.output / "review.js")
    summary = {kind: sum(record["classification"] == kind for record in records) for kind in ("complete", "partial", "failed")}
    with (args.output / "data.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "status": "READY_M3_V1_2_RESULT_PREVIEW",
                "event_count": len(records),
                "classification_counts": summary,
                "records": records,
            },
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
    (args.output / "index.html").write_text(html_document(records), encoding="utf-8")
    print(json.dumps({"status": "READY_M3_V1_2_RESULT_PREVIEW", "events": len(records), **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
