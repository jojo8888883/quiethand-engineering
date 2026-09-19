#!/usr/bin/env python3
"""Materialize evaluation RGB-D intervals without exposing source labels to models."""

from __future__ import annotations

import argparse
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
M3_V1_2 = ROOT / "scripts/quiethand/m3_v1_2"
M3_5 = ROOT / "scripts/quiethand/m3_5_fusion"
for directory in (ROOT, M3_V1_2, M3_5):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from materialize_geometry import (  # noqa: E402
    DEPTH_SCALE,
    HEIGHT,
    WIDTH,
    atomic_json,
    file_record,
    scale_obj_cm_to_m,
    selected_indices,
    source,
)
from prepare_metric_hand_input import camera_model  # noqa: E402
from quiethand.m3_v1_2_contract import validate_candidate  # noqa: E402


EXPECTED_EVENTS = 30


def decode(ffmpeg_source: Path, indices: list[int], pixel_format: str, bytes_per_pixel: int):
    import imageio_ffmpeg

    expression = "+".join(f"eq(n\\,{index})" for index in indices)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-threads", "1",
        "-i", str(ffmpeg_source), "-map", "0:v:0", "-an", "-sn", "-dn",
        "-vf", f"select={expression}", "-vsync", "0", "-f", "rawvideo",
        "-pix_fmt", pixel_format, "pipe:1",
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


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--temporal-plan", type=Path, required=True)
    parser.add_argument("--temporal-input", type=Path, required=True)
    parser.add_argument("--semantic-output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _promote(workspace: Path, staging: Path, geometry_root: Path, record: dict[str, object]) -> dict[str, object]:
    staging_relative = staging.relative_to(workspace).as_posix()
    geometry_relative = geometry_root.relative_to(workspace).as_posix()
    return dict(record, relative_path=record["relative_path"].replace(staging_relative, geometry_relative, 1))


def main() -> int:
    cfg = arguments()
    plan = json.loads(cfg.temporal_plan.read_text(encoding="utf-8"))
    temporal_input = json.loads(cfg.temporal_input.read_text(encoding="utf-8"))
    semantic_audit = json.loads((cfg.semantic_output / "audit.json").read_text(encoding="utf-8"))
    plan_events = plan.get("events")
    temporal_events = temporal_input.get("events")
    if (
        plan.get("evaluation_video_count") != EXPECTED_EVENTS
        or plan.get("evaluation_model_results_authorized") is not True
        or plan.get("evaluation_model_results_opened") is not False
        or not isinstance(plan_events, list)
        or len(plan_events) != EXPECTED_EVENTS
    ):
        raise SystemExit("temporal plan is not the frozen 30-event evaluation set")
    if (
        temporal_input.get("event_count") != EXPECTED_EVENTS
        or temporal_input.get("evaluation_event_count") != EXPECTED_EVENTS
        or temporal_input.get("evaluation_model_results_authorized") is not True
        or not isinstance(temporal_events, list)
        or len(temporal_events) != EXPECTED_EVENTS
        or semantic_audit.get("status") != "COMPLETE"
        or semantic_audit.get("event_count") != EXPECTED_EVENTS
        or semantic_audit.get("evaluation_event_count") != EXPECTED_EVENTS
        or semantic_audit.get("evaluation_results_opened") is not True
        or semantic_audit.get("training_performed") is not False
    ):
        raise SystemExit("semantic dependency is not the completed 30-event evaluation output")
    plan_ids = [event.get("event_id") for event in plan_events]
    input_ids = [event.get("event_id") for event in temporal_events]
    if plan_ids != input_ids or len(set(plan_ids)) != EXPECTED_EVENTS:
        raise SystemExit("evaluation plan and temporal input are not aligned one-to-one")

    input_by_id = {row["event_id"]: row for row in temporal_events}
    geometry_root = cfg.output_root / "geometry_inputs"
    if geometry_root.exists():
        raise SystemExit(f"destination exists: {geometry_root}")
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".geometry_inputs.", dir=cfg.output_root))
    raw_events: list[dict[str, object]] = []
    perception_events: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    try:
        for ordinal, row in enumerate(plan_events, start=1):
            event_id = row["event_id"]
            semantic = json.loads((cfg.semantic_output / "events" / f"{event_id}.json").read_text(encoding="utf-8"))
            if semantic.get("status") != "observed":
                records.append({"event_id": event_id, "status": "semantic_invalid", "frame_indices": [], "failure_reason": semantic.get("failure_reason")})
                continue
            candidate = validate_candidate(json.dumps(semantic["candidate"], sort_keys=True, allow_nan=False))
            if candidate["evidence"] == "not_visible":
                records.append({"event_id": event_id, "status": "abstain_not_visible", "frame_indices": [], "failure_reason": "semantic_not_visible"})
                continue
            samples = row["sample_frame_indices"]
            frame_indices = selected_indices(samples[candidate["action_start_sample"]], samples[candidate["action_end_sample"]])
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
            first_rgb = event_dir / "rgb_00.png"
            first_rgb.unlink()
            os.link(cfg.workspace / input_by_id[event_id]["ordered_rgb_frames"][candidate["action_start_sample"]], first_rgb)
            for position, (_, payload) in enumerate(decode(depth_source, frame_indices, "gray16le", 2)):
                raw = np.frombuffer(payload, dtype="<u2").reshape(HEIGHT, WIDTH)
                np.save(event_dir / f"depth_{position:02d}.npy", (raw.astype(np.float32) / np.float32(DEPTH_SCALE)).astype(np.float32), allow_pickle=False)
            intrinsic = np.loadtxt(intrinsic_source, dtype=np.float64)
            if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
                raise RuntimeError("intrinsic is invalid")
            np.save(event_dir / "intrinsic.npy", intrinsic, allow_pickle=False)

            entity_meshes: dict[str, Path] = {}
            for binding_key in ("tool_mesh", "target_mesh"):
                mesh_source = source(cfg.source_root, binding[binding_key])
                entity_id = "cad_" + mesh_source.stem.removesuffix("_cm")
                mesh_path = event_dir / f"{entity_id}.obj"
                mesh_path.write_bytes(scale_obj_cm_to_m(mesh_source)[0])
                entity_meshes[entity_id] = mesh_path

            rgb_records = [_promote(cfg.workspace, staging, geometry_root, file_record(cfg.workspace, event_dir / f"rgb_{index:02d}.png", frame)) for index, frame in enumerate(frame_indices)]
            depth_records = [_promote(cfg.workspace, staging, geometry_root, file_record(cfg.workspace, event_dir / f"depth_{index:02d}.npy", frame)) for index, frame in enumerate(frame_indices)]
            intrinsic_record = _promote(cfg.workspace, staging, geometry_root, file_record(cfg.workspace, event_dir / "intrinsic.npy"))
            mesh_records = {entity_id: _promote(cfg.workspace, staging, geometry_root, file_record(cfg.workspace, path)) for entity_id, path in sorted(entity_meshes.items())}
            ordered_rgb = [item["relative_path"] for item in rgb_records]
            ordered_depth = [item["relative_path"] for item in depth_records]
            raw_events.append({
                "event_id": event_id,
                "ordered_rgb_frames": ordered_rgb,
                "source_frame_indices": frame_indices,
                "camera_model": camera_model(row, cfg.source_root),
                "materialized_file_binding": {"rgb": rgb_records},
            })
            perception_events.append({
                "event_id": event_id,
                "segmentation": {"ordered_rgb_frames": ordered_rgb, "box_source": "temporal_semantic.start_sample_boxes_0_1000"},
                "object_state": {
                    "ordered_depth_frames": ordered_depth,
                    "intrinsic_path": intrinsic_record["relative_path"],
                    "entities": [{"entity_id": entity_id, "mesh_path": record["relative_path"]} for entity_id, record in mesh_records.items()],
                },
                "source_frame_indices": frame_indices,
                "materialized_file_binding": {"rgb": rgb_records, "depth": depth_records, "intrinsic": [intrinsic_record], "mesh": list(mesh_records.values())},
            })
            records.append({"event_id": event_id, "status": "ready_geometry_inference", "frame_indices": frame_indices, "failure_reason": None})
            print(f"[geometry] {ordinal}/{EXPECTED_EVENTS} {event_id}", flush=True)

        os.replace(staging, geometry_root)
        staging = Path("/nonexistent")
        common = {
            "evaluation_event_count": EXPECTED_EVENTS,
            "evaluation_model_results_authorized": True,
            "evaluation_results_opened": True,
            "training_performed": False,
        }
        atomic_json(cfg.output_root / "QH_M3_EVALUATION_GEOMETRY_MATERIALIZATION.json", {
            "schema": "quiethand.m3_5.evaluation_geometry_materialization.v1",
            "status": "READY_GEOMETRY_INFERENCE",
            "geometry_event_count": len(raw_events),
            **common,
            "events": records,
        })
        atomic_json(cfg.output_root / "QH_M3_EVALUATION_RAW_HAND_INPUT.json", {
            "schema": "quiethand.m3_5.evaluation_raw_hand_metric_input.v1",
            "adapter": "raw_hand_metric",
            "event_count": len(raw_events),
            **common,
            "events": raw_events,
        })
        atomic_json(cfg.output_root / "QH_M3_EVALUATION_PERCEPTION_INPUT.json", {
            "schema": "quiethand.m3_5.evaluation_perception_input.v1",
            "adapter": "segmentation_object_state",
            "event_count": len(perception_events),
            **common,
            "events": perception_events,
        })
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    if len(raw_events) != EXPECTED_EVENTS:
        raise SystemExit(f"geometry coverage is {len(raw_events)}/{EXPECTED_EVENTS}; do not submit partial evaluation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
