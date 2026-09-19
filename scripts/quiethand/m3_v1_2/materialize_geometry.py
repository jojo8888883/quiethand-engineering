#!/usr/bin/env python3
"""Materialize 15 RGB-D frames inside each valid v1.2 action interval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quiethand.m3_materialization import DEPTH_SCALE, HEIGHT, WIDTH, scale_obj_cm_to_m
from quiethand.m3_v1_2_contract import GEOMETRY_FRAME_COUNT, validate_candidate


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--temporal-plan", type=Path, required=True)
    parser.add_argument("--temporal-input", type=Path, required=True)
    parser.add_argument("--semantic-output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha(path: Path) -> str:
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


def source(root: Path, binding: dict[str, object]) -> Path:
    relative = binding.get("relative_path")
    if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
        raise RuntimeError("source path is invalid")
    path = root / relative
    if not path.is_file() or path.is_symlink() or path.stat().st_size != binding.get("bytes") or sha(path) != binding.get("sha256"):
        raise RuntimeError(f"source binding changed: {relative}")
    return path


def selected_indices(first: int, last: int) -> list[int]:
    if last - first + 1 < GEOMETRY_FRAME_COUNT:
        return []
    values = [round(first + index * (last - first) / (GEOMETRY_FRAME_COUNT - 1)) for index in range(GEOMETRY_FRAME_COUNT)]
    return values if len(set(values)) == GEOMETRY_FRAME_COUNT else []


def decode(ffmpeg_source: Path, indices: list[int], pixel_format: str, bytes_per_pixel: int):
    expression = "+".join(f"eq(n\\,{index})" for index in indices)
    command = [
        "ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-i", str(ffmpeg_source),
        "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", f"select={expression}",
        "-vsync", "0", "-f", "rawvideo", "-pix_fmt", pixel_format, "pipe:1",
    ]
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None and process.stderr is not None
    frame_bytes = WIDTH * HEIGHT * bytes_per_pixel
    try:
        for frame_index in indices:
            payload = process.stdout.read(frame_bytes)
            if len(payload) != frame_bytes:
                raise RuntimeError("decoder ended early")
            yield frame_index, payload
        if process.stdout.read(1):
            raise RuntimeError("decoder emitted extra frames")
        error = process.stderr.read().decode(errors="replace").strip()
        code = process.wait()
        if code != 0 or error:
            raise RuntimeError(f"decoder failed: {error or code}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def file_record(workspace: Path, path: Path, frame_index: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "relative_path": path.relative_to(workspace).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha(path),
    }
    if frame_index is not None:
        result["source_frame_index"] = frame_index
    return result


def main() -> int:
    cfg = args()
    plan = json.loads(cfg.temporal_plan.read_text(encoding="utf-8"))
    temporal_input = json.loads(cfg.temporal_input.read_text(encoding="utf-8"))
    semantic_audit = json.loads((cfg.semantic_output / "audit.json").read_text(encoding="utf-8"))
    if plan.get("calibration_video_count") != 30 or plan.get("evaluation_video_count") != 0:
        raise SystemExit("temporal plan is not 30/0")
    if temporal_input.get("event_count") != 30 or semantic_audit.get("status") != "COMPLETE" or semantic_audit.get("event_count") != 30 or semantic_audit.get("evaluation_results_opened") is not False:
        raise SystemExit("semantic dependency is not complete calibration-only output")
    input_by_id = {row["event_id"]: row for row in temporal_input["events"]}
    geometry_root = cfg.output_root / "geometry_inputs"
    if geometry_root.exists():
        raise SystemExit(f"destination exists: {geometry_root}")
    staging = Path(tempfile.mkdtemp(prefix=".geometry_inputs.", dir=cfg.output_root))
    raw_events: list[dict[str, object]] = []
    perception_events: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    try:
        for ordinal, row in enumerate(plan["events"], start=1):
            event_id = row["event_id"]
            semantic_path = cfg.semantic_output / "events" / f"{event_id}.json"
            semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
            if semantic.get("status") != "observed":
                records.append({"event_id": event_id, "status": "semantic_invalid", "frame_indices": [], "failure_reason": semantic.get("failure_reason")})
                continue
            candidate = validate_candidate(json.dumps(semantic["candidate"], sort_keys=True, allow_nan=False))
            if candidate["evidence"] == "not_visible":
                records.append({"event_id": event_id, "status": "abstain_not_visible", "frame_indices": [], "failure_reason": "semantic_not_visible"})
                continue
            samples = row["sample_frame_indices"]
            start_sample, end_sample = candidate["action_start_sample"], candidate["action_end_sample"]
            frame_indices = selected_indices(samples[start_sample], samples[end_sample])
            if not frame_indices:
                records.append({"event_id": event_id, "status": "temporal_invalid", "frame_indices": [], "failure_reason": "selected interval contains fewer than 15 unique frames"})
                continue
            binding = row["source_binding"]
            rgb_source = source(cfg.source_root, binding["rgb"])
            depth_source = source(cfg.source_root, binding["depth"])
            intrinsic_source = source(cfg.source_root, binding["intrinsic"])
            event_dir = staging / event_id
            event_dir.mkdir()
            for position, (_, payload) in enumerate(decode(rgb_source, frame_indices, "rgb24", 3)):
                Image.frombytes("RGB", (WIDTH, HEIGHT), payload).save(event_dir / f"rgb_{position:02d}.png", format="PNG", compress_level=6, optimize=False)
            temporal_start = cfg.workspace / input_by_id[event_id]["ordered_rgb_frames"][start_sample]
            first_rgb = event_dir / "rgb_00.png"
            first_rgb.unlink()
            os.link(temporal_start, first_rgb)
            for position, (_, payload) in enumerate(decode(depth_source, frame_indices, "gray16le", 2)):
                raw = np.frombuffer(payload, dtype="<u2").reshape(HEIGHT, WIDTH)
                np.save(event_dir / f"depth_{position:02d}.npy", (raw.astype(np.float32) / np.float32(DEPTH_SCALE)).astype(np.float32), allow_pickle=False)
            intrinsic = np.loadtxt(intrinsic_source, dtype=np.float64)
            if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
                raise RuntimeError("intrinsic is invalid")
            np.save(event_dir / "intrinsic.npy", intrinsic, allow_pickle=False)
            entity_meshes = {}
            for role in ("tool", "target"):
                mesh_source = source(cfg.source_root, binding[f"{role}_mesh"])
                entity_id = "cad_" + mesh_source.stem.removesuffix("_cm")
                payload, _ = scale_obj_cm_to_m(mesh_source)
                (event_dir / f"{entity_id}.obj").write_bytes(payload)
                entity_meshes[entity_id] = event_dir / f"{entity_id}.obj"
            rgb_records = [file_record(cfg.workspace, event_dir / f"rgb_{index:02d}.png", frame) for index, frame in enumerate(frame_indices)]
            depth_records = [file_record(cfg.workspace, event_dir / f"depth_{index:02d}.npy", frame) for index, frame in enumerate(frame_indices)]
            def promoted(record: dict[str, object]) -> dict[str, object]:
                return dict(record, relative_path=record["relative_path"].replace(staging.relative_to(cfg.workspace).as_posix(), geometry_root.relative_to(cfg.workspace).as_posix()))
            rgb_records = [promoted(item) for item in rgb_records]
            depth_records = [promoted(item) for item in depth_records]
            intrinsic_record = promoted(file_record(cfg.workspace, event_dir / "intrinsic.npy"))
            mesh_records = {entity_id: promoted(file_record(cfg.workspace, path)) for entity_id, path in sorted(entity_meshes.items())}
            ordered_rgb = [item["relative_path"] for item in rgb_records]
            ordered_depth = [item["relative_path"] for item in depth_records]
            common_binding = {"rgb": rgb_records}
            raw_events.append({
                "event_id": event_id, "ordered_rgb_frames": ordered_rgb,
                "source_frame_indices": frame_indices,
                "materialized_file_binding": common_binding,
            })
            perception_events.append({
                "event_id": event_id,
                "segmentation": {"ordered_rgb_frames": ordered_rgb, "box_source": "temporal_semantic.start_sample_boxes_0_1000"},
                "object_state": {
                    "ordered_depth_frames": ordered_depth,
                    "intrinsic_path": intrinsic_record["relative_path"],
                    "entities": [{"entity_id": entity_id, "mesh_path": record["relative_path"]}
                                 for entity_id, record in mesh_records.items()],
                },
                "materialized_file_binding": {"rgb": rgb_records, "depth": depth_records, "intrinsic": [intrinsic_record], "mesh": list(mesh_records.values())},
            })
            records.append({"event_id": event_id, "status": "ready_geometry_inference", "frame_indices": frame_indices, "failure_reason": None})
            print(f"[geometry] {ordinal}/30 {event_id}", flush=True)
        os.replace(staging, geometry_root)
        staging = Path("/nonexistent")
        atomic_json(cfg.output_root / "QH_M3_V1_2_GEOMETRY_MATERIALIZATION.json", {
            "schema": "quiethand.m3_v1_2.geometry_materialization.v1", "status": "READY_GEOMETRY_INFERENCE",
            "calibration_video_count": 30, "geometry_event_count": len(raw_events),
            "evaluation_event_count": 0, "evaluation_model_results_opened": False, "events": records,
        })
        atomic_json(cfg.output_root / "QH_M3_V1_2_RAW_HAND_INPUT.json", {
            "schema": "quiethand.m3_v1_2.raw_hand_input.v1", "adapter": "raw_hand",
            "event_count": len(raw_events), "calibration_video_count": 30,
            "evaluation_event_count": 0, "evaluation_model_results_opened": False, "events": raw_events,
        })
        atomic_json(cfg.output_root / "QH_M3_V1_2_PERCEPTION_INPUT.json", {
            "schema": "quiethand.m3_v1_2.perception_input.v1", "adapter": "segmentation_object_state",
            "event_count": len(perception_events), "calibration_video_count": 30,
            "evaluation_event_count": 0, "evaluation_model_results_opened": False, "events": perception_events,
        })
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
