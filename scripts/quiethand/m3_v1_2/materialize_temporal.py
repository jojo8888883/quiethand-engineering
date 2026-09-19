#!/usr/bin/env python3
"""Materialize eight full-clip calibration samples per TACO source video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quiethand.m3_v1_2_contract import PROMPT, PROMPT_SHA256, SAMPLE_COUNT


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--old-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_path(root: Path, binding: dict[str, object]) -> Path:
    relative = binding.get("relative_path")
    if not isinstance(relative, str):
        raise RuntimeError("source binding path is invalid")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise RuntimeError("source binding escapes root")
    path = root.joinpath(*pure.parts)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"source file missing: {relative}")
    if path.stat().st_size != binding.get("bytes") or sha256_file(path) != binding.get("sha256"):
        raise RuntimeError(f"source binding changed: {relative}")
    return path


def video_metadata(path: Path) -> tuple[int, int, int, str]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames,avg_frame_rate",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
    streams = json.loads(result.stdout).get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise RuntimeError("video stream metadata is missing")
    row = streams[0]
    width, height, frames, rate = row.get("width"), row.get("height"), row.get("nb_frames"), row.get("avg_frame_rate")
    try:
        frames_int = int(frames)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("video frame count is unavailable") from exc
    if (width, height) != (1920, 1080) or rate != "30/1" or frames_int < SAMPLE_COUNT:
        raise RuntimeError("RGB timebase or dimensions changed")
    return int(width), int(height), frames_int, rate


def record(workspace: Path, path: Path, source_index: int) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(workspace).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "source_frame_index": source_index,
    }


def extract(source: Path, indices: list[int], destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=False)
    expression = "+".join(f"eq(n\\,{index})" for index in indices)
    output = destination / "rgb_%02d.png"
    command = [
        "ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-i", str(source),
        "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", f"select={expression}",
        "-vsync", "0", "-start_number", "0", str(output),
    ]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    paths = [destination / f"rgb_{index:02d}.png" for index in range(SAMPLE_COUNT)]
    if result.returncode != 0 or any(not path.is_file() for path in paths) or len(list(destination.glob("rgb_*.png"))) != SAMPLE_COUNT:
        raise RuntimeError(f"ffmpeg extraction failed: {result.stderr.decode(errors='replace').strip()}")
    return paths


def main() -> int:
    cfg = args()
    old = json.loads(cfg.old_plan.read_text(encoding="utf-8"))
    tasks = old.get("tasks")
    if old.get("task_count") != 150 or old.get("evaluation_task_count") != 0 or not isinstance(tasks, list):
        raise SystemExit("old calibration plan is not the closed 150/0 source")
    first_by_rank: dict[int, dict[str, object]] = {}
    for task in tasks:
        orchestration = task.get("orchestration_only_never_model_input")
        if not isinstance(orchestration, dict):
            raise SystemExit("old task lacks orchestration binding")
        rank = orchestration.get("rank")
        if not isinstance(rank, int):
            raise SystemExit("old task rank is invalid")
        first_by_rank.setdefault(rank, orchestration)
    if len(first_by_rank) != 30:
        raise SystemExit("calibration source coverage is not 30 videos")
    temporal_root = cfg.output_root / "temporal_inputs"
    if temporal_root.exists():
        raise SystemExit(f"destination exists: {temporal_root}")
    temporal_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".temporal_inputs.", dir=temporal_root.parent))
    plan_rows: list[dict[str, object]] = []
    input_rows: list[dict[str, object]] = []
    try:
        for ordinal, rank in enumerate(sorted(first_by_rank), start=1):
            orchestration = first_by_rank[rank]
            sequence_id = orchestration.get("sequence_id")
            binding = orchestration.get("source_binding")
            if not isinstance(sequence_id, str) or not isinstance(binding, dict):
                raise RuntimeError("source identity/binding is invalid")
            rgb_binding = binding.get("rgb")
            if not isinstance(rgb_binding, dict):
                raise RuntimeError("RGB binding is invalid")
            source = source_path(cfg.source_root, rgb_binding)
            width, height, frame_count, rate = video_metadata(source)
            indices = [round(index * (frame_count - 1) / (SAMPLE_COUNT - 1)) for index in range(SAMPLE_COUNT)]
            if len(set(indices)) != SAMPLE_COUNT:
                raise RuntimeError("sample indices are not unique")
            digest = hashlib.sha256(f"QH-M3-V1.2-CAL\0{sequence_id}".encode()).hexdigest()[:24]
            event_id = f"qh-m3-v12-cal-{digest}"
            paths = extract(source, indices, staging / event_id)
            records = [record(cfg.workspace, path, source_index) for path, source_index in zip(paths, indices, strict=True)]
            plan_rows.append({
                "event_id": event_id,
                "rank": rank,
                "sequence_id": sequence_id,
                "source_frame_count": frame_count,
                "source_rate": rate,
                "source_rgb": rgb_binding,
                "source_binding": binding,
                "sample_frame_indices": indices,
            })
            input_rows.append({
                "event_id": event_id,
                "ordered_rgb_frames": [item["relative_path"].replace(staging.relative_to(cfg.workspace).as_posix(), temporal_root.relative_to(cfg.workspace).as_posix()) for item in records],
                "sample_frame_indices": indices,
                "prompt": PROMPT,
                "prompt_sha256": PROMPT_SHA256,
                "do_sample": False,
                "max_new_tokens": 640,
                "materialized_file_binding": {"rgb": [dict(item, relative_path=item["relative_path"].replace(staging.relative_to(cfg.workspace).as_posix(), temporal_root.relative_to(cfg.workspace).as_posix())) for item in records]},
            })
            print(f"[temporal] {ordinal}/30 {event_id}", flush=True)
        os.replace(staging, temporal_root)
        staging = Path("/nonexistent")
        atomic_json(cfg.output_root / "QH_M3_V1_2_TEMPORAL_PLAN.json", {
            "schema": "quiethand.m3_v1_2.temporal_plan.v1", "status": "READY_TEMPORAL_INFERENCE",
            "calibration_video_count": 30, "evaluation_video_count": 0,
            "evaluation_model_results_opened": False, "events": plan_rows,
        })
        atomic_json(cfg.output_root / "QH_M3_V1_2_TEMPORAL_INPUT.json", {
            "schema": "quiethand.m3_v1_2.temporal_input.v1", "adapter": "temporal_semantic",
            "event_count": 30, "evaluation_event_count": 0,
            "evaluation_model_results_opened": False, "events": input_rows,
        })
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
