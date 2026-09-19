#!/usr/bin/env python3
"""Materialize the authorized 30-video evaluation temporal input."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile

import imageio_ffmpeg


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quiethand.m3_v1_2_contract import PROMPT, PROMPT_SHA256, SAMPLE_COUNT
from quiethand.taco_contract import canonical_selection_bytes, deterministic_taco_split, load_frozen_sequence_ids


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sequence-list", type=Path, required=True)
    parser.add_argument("--native-contract", type=Path, required=True)
    parser.add_argument("--landing-receipt", type=Path, required=True)
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
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def receipt_record(completed: dict[str, object], suffix: str) -> dict[str, object]:
    matches = [(key, value) for key, value in completed.items() if key.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one landed source ending {suffix!r}, got {len(matches)}")
    path, raw = matches[0]
    if not isinstance(raw, dict):
        raise RuntimeError("landing receipt record is invalid")
    digest, size = raw.get("sha256"), raw.get("bytes")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise RuntimeError("landing receipt SHA-256 is invalid")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise RuntimeError("landing receipt byte count is invalid")
    return {"relative_path": path, "sha256": digest, "bytes": size}


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
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(reader)
    finally:
        reader.close()
    frame_count, _ = imageio_ffmpeg.count_frames_and_secs(str(path))
    width, height = metadata.get("source_size", (None, None))
    fps = metadata.get("fps")
    if (width, height) != (1920, 1080) or fps != 30.0 or frame_count < SAMPLE_COUNT:
        raise RuntimeError("RGB timebase or dimensions changed")
    return int(width), int(height), int(frame_count), "30/1"


def extract(source: Path, indices: list[int], destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=False)
    expression = "+".join(f"eq(n\\,{index})" for index in indices)
    result = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-threads", "1", "-i", str(source),
            "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", f"select={expression}",
            "-vsync", "0", "-start_number", "0", str(destination / "rgb_%02d.png"),
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    paths = [destination / f"rgb_{index:02d}.png" for index in range(SAMPLE_COUNT)]
    if result.returncode != 0 or any(not path.is_file() for path in paths) or len(list(destination.glob("rgb_*.png"))) != SAMPLE_COUNT:
        raise RuntimeError(f"ffmpeg extraction failed: {result.stderr.decode(errors='replace').strip()}")
    return paths


def file_record(workspace: Path, path: Path, source_index: int) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(workspace).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "source_frame_index": source_index,
    }


def object_ids(sequence: dict[str, object]) -> dict[str, str]:
    values = sequence.get("object_pose_files")
    if not isinstance(values, list):
        raise RuntimeError("native object pose files are missing")
    result: dict[str, str] = {}
    for value in values:
        if not isinstance(value, str):
            continue
        match = re.fullmatch(r"(tool|target)_([0-9]+)\.npy", value)
        if match:
            result[match.group(1)] = match.group(2)
    if set(result) != {"tool", "target"}:
        raise RuntimeError("native tool/target object IDs are incomplete")
    return result


def main() -> int:
    cfg = arguments()
    native = json.loads(cfg.native_contract.read_text(encoding="utf-8"))
    receipt = json.loads(cfg.landing_receipt.read_text(encoding="utf-8"))
    if native.get("status") != "READY_NATIVE" or receipt.get("status") != "READY_FOR_NATIVE_VALIDATION":
        raise SystemExit("native source provenance is not ready")
    completed = receipt.get("completed")
    sequences = native.get("sequences")
    if not isinstance(completed, dict) or not isinstance(sequences, list):
        raise SystemExit("native source contract is incomplete")
    entries = deterministic_taco_split(load_frozen_sequence_ids(cfg.sequence_list))
    canonical_selection_bytes(entries)
    evaluation = [entry for entry in entries if entry.split == "evaluation"]
    native_by_id = {row.get("sequence_id"): row for row in sequences if isinstance(row, dict)}
    if len(evaluation) != 30 or len(native_by_id) != 60:
        raise SystemExit("frozen 30/30 split is unavailable")

    temporal_root = cfg.output_root / "temporal_inputs"
    if temporal_root.exists():
        raise SystemExit(f"destination exists: {temporal_root}")
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".temporal_inputs.", dir=cfg.output_root))
    plan_rows: list[dict[str, object]] = []
    input_rows: list[dict[str, object]] = []
    try:
        for ordinal, entry in enumerate(evaluation, start=1):
            sequence = native_by_id.get(entry.sequence_id)
            if not isinstance(sequence, dict) or sequence.get("split") != "evaluation" or sequence.get("rank") != entry.rank:
                raise RuntimeError("evaluation/native sequence alignment changed")
            prefix = f"/{entry.triplet}/{entry.sequence_name}/"
            sequence_records = {key: value for key, value in completed.items() if prefix in "/" + key}
            ids = object_ids(sequence)
            binding = {
                "rgb": receipt_record(sequence_records, "/color.mp4"),
                "depth": receipt_record(sequence_records, "/egocentric_depth.avi"),
                "intrinsic": receipt_record(sequence_records, "/egocentric_intrinsic.txt"),
                "extrinsic": receipt_record(sequence_records, "/egocentric_frame_extrinsic.npy"),
                "tool_pose": receipt_record(sequence_records, f"/tool_{ids['tool']}.npy"),
                "target_pose": receipt_record(sequence_records, f"/target_{ids['target']}.npy"),
                "tool_mesh": receipt_record(completed, f"/{ids['tool']}_cm.obj"),
                "target_mesh": receipt_record(completed, f"/{ids['target']}_cm.obj"),
            }
            source = source_path(cfg.source_root, binding["rgb"])
            _, _, frame_count, rate = video_metadata(source)
            indices = [round(index * (frame_count - 1) / (SAMPLE_COUNT - 1)) for index in range(SAMPLE_COUNT)]
            if len(set(indices)) != SAMPLE_COUNT:
                raise RuntimeError("sample indices are not unique")
            digest = hashlib.sha256(f"QH-M3-V1.2-EVAL\0{entry.sequence_id}".encode()).hexdigest()[:24]
            event_id = f"qh-m3-v12-eval-{digest}"
            paths = extract(source, indices, staging / event_id)
            records = [file_record(cfg.workspace, path, frame_index) for path, frame_index in zip(paths, indices, strict=True)]
            old_prefix = staging.relative_to(cfg.workspace).as_posix()
            new_prefix = temporal_root.relative_to(cfg.workspace).as_posix()
            promoted = [dict(row, relative_path=row["relative_path"].replace(old_prefix, new_prefix)) for row in records]
            plan_rows.append({
                "event_id": event_id,
                "rank": entry.rank,
                "sequence_id": entry.sequence_id,
                "source_frame_count": frame_count,
                "source_rate": rate,
                "source_binding": binding,
                "sample_frame_indices": indices,
            })
            input_rows.append({
                "event_id": event_id,
                "ordered_rgb_frames": [row["relative_path"] for row in promoted],
                "sample_frame_indices": indices,
                "prompt": PROMPT,
                "prompt_sha256": PROMPT_SHA256,
                "do_sample": False,
                "max_new_tokens": 640,
                "materialized_file_binding": {"rgb": promoted},
            })
            print(f"[temporal-eval] {ordinal}/30 {event_id}", flush=True)
        serialized_input = json.dumps(input_rows, sort_keys=True, allow_nan=False)
        leaked = [entry.sequence_id for entry in evaluation if entry.sequence_id in serialized_input]
        if leaked:
            raise RuntimeError("evaluation metadata leaked into model-visible input")
        os.replace(staging, temporal_root)
        staging = Path("/nonexistent")
        provenance = {
            "native_contract_sha256": sha256_file(cfg.native_contract),
            "landing_receipt_sha256": sha256_file(cfg.landing_receipt),
            "selection_sha256": hashlib.sha256(canonical_selection_bytes(entries)).hexdigest(),
        }
        atomic_json(cfg.output_root / "QH_M3_EVALUATION_TEMPORAL_PLAN.json", {
            "schema": "quiethand.m3_v1_2.evaluation_temporal_plan.v1",
            "status": "READY_EVALUATION_TEMPORAL_INFERENCE",
            "evaluation_video_count": 30,
            "evaluation_model_results_authorized": True,
            "evaluation_model_results_opened": False,
            "training_performed": False,
            "source_provenance": provenance,
            "events": plan_rows,
        })
        atomic_json(cfg.output_root / "QH_M3_EVALUATION_TEMPORAL_INPUT.json", {
            "schema": "quiethand.m3_v1_2.evaluation_temporal_input.v1",
            "status": "READY_EVALUATION_TEMPORAL_INFERENCE",
            "adapter": "temporal_semantic",
            "event_count": 30,
            "evaluation_event_count": 30,
            "evaluation_model_results_authorized": True,
            "evaluation_model_results_opened": False,
            "training_performed": False,
            "events": input_rows,
        })
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
