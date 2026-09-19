#!/usr/bin/env python3
"""Materialize exactly the 150 frozen M3 calibration events."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import imageio_ffmpeg
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quiethand.m3_materialization import (  # noqa: E402
    DEPTH_SCALE,
    HEIGHT,
    MATERIALIZATION_SCHEMA,
    M3MaterializationError,
    TASK_PLAN_SHA256,
    WIDTH,
    atomic_json,
    clean_staging,
    decode_selected_frames,
    file_record,
    promote_event,
    scale_obj_cm_to_m,
    sha256_file,
    verify_materialization_manifest,
    verify_source_binding,
)


DATA_ROOT = ROOT / "external_data" / "taco_v1"
PLAN_PATH = ROOT / "artifacts" / "quiethand" / "m3" / "QH_M3_CALIBRATION_TASK_PLAN.json"
INPUT_ROOT = ROOT / "artifacts" / "quiethand" / "m3" / "calibration_inputs"
MANIFEST_PATH = ROOT / "artifacts" / "quiethand" / "m3" / "QH_M3_CALIBRATION_MATERIALIZATION.json"


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _decoder_identity(ffmpeg: Path) -> dict[str, object]:
    result = subprocess.run(
        [str(ffmpeg), "-hide_banner", "-version"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.splitlines():
        raise M3MaterializationError("bundled FFmpeg identity is unavailable")
    return {
        "imageio_ffmpeg_version": imageio_ffmpeg.__version__,
        "ffmpeg_first_line": result.stdout.splitlines()[0],
        "ffmpeg_sha256": sha256_file(ffmpeg),
    }


def _load_plan() -> dict[str, object]:
    payload = PLAN_PATH.read_bytes()
    if hashlib.sha256(payload).hexdigest() != TASK_PLAN_SHA256:
        raise M3MaterializationError("calibration task plan changed")
    plan = json.loads(payload)
    if (
        not isinstance(plan, dict)
        or plan.get("status") != "READY_FOR_CALIBRATION_MATERIALIZATION"
        or plan.get("task_count") != 150
        or plan.get("evaluation_task_count") != 0
        or plan.get("evaluation_model_results_opened") is not False
        or not isinstance(plan.get("tasks"), list)
        or any(task.get("split") != "calibration" for task in plan["tasks"])
    ):
        raise M3MaterializationError("calibration-only task plan is not closed")
    return plan


def _mesh_cache_file(
    source: Path,
    source_sha: str,
    mesh_cache: Path,
    memory_cache: dict[str, tuple[bytes, dict[str, object]]],
) -> tuple[Path, dict[str, object]]:
    cached = memory_cache.get(source_sha)
    if cached is None:
        cached = scale_obj_cm_to_m(source)
        memory_cache[source_sha] = cached
    payload, geometry = cached
    path = mesh_cache / f"{source_sha}.obj"
    if path.exists():
        if path.is_symlink() or sha256_file(path) != hashlib.sha256(payload).hexdigest():
            raise M3MaterializationError("scaled mesh cache changed")
    else:
        temporary = mesh_cache / f".{source_sha}.{os.getpid()}.tmp"
        _write_bytes(temporary, payload)
        os.replace(temporary, path)
    return path, geometry


def _materialize_sequence(
    tasks: list[dict[str, object]],
    *,
    ffmpeg: Path,
    source_cache: dict[str, tuple[int, str]],
    mesh_memory_cache: dict[str, tuple[bytes, dict[str, object]]],
    mesh_cache: Path,
    staging_root: Path,
) -> list[dict[str, object]]:
    bindings = [task["orchestration_only_never_model_input"]["source_binding"] for task in tasks]
    first = bindings[0]
    for binding in bindings[1:]:
        for key in ("rgb", "depth", "intrinsic", "extrinsic"):
            if binding[key] != first[key]:
                raise M3MaterializationError("sequence-level source binding is inconsistent")
    rgb_source = verify_source_binding(DATA_ROOT, first["rgb"], source_cache)
    depth_source = verify_source_binding(DATA_ROOT, first["depth"], source_cache)
    intrinsic_source = verify_source_binding(DATA_ROOT, first["intrinsic"], source_cache)
    verify_source_binding(DATA_ROOT, first["extrinsic"], source_cache)
    intrinsic = np.loadtxt(intrinsic_source, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all() or intrinsic[2].tolist() != [0.0, 0.0, 1.0]:
        raise M3MaterializationError("camera intrinsic is invalid")

    occurrences: dict[int, list[tuple[dict[str, object], int]]] = {}
    staging: dict[str, Path] = {}
    for task in tasks:
        orchestration = task["orchestration_only_never_model_input"]
        frames = orchestration["frame_indices"]
        if not isinstance(frames, list) or len(frames) != 15:
            raise M3MaterializationError("event frame window is invalid")
        event_id = task["event_id"]
        event_staging = Path(tempfile.mkdtemp(prefix=f"{event_id}.", dir=staging_root))
        staging[event_id] = event_staging
        for position, frame_index in enumerate(frames):
            occurrences.setdefault(frame_index, []).append((task, position))

    selected = sorted(occurrences)
    try:
        for frame_index, payload in decode_selected_frames(
            ffmpeg, rgb_source, selected, pixel_format="rgb24", bytes_per_pixel=3
        ):
            image = Image.frombytes("RGB", (WIDTH, HEIGHT), payload)
            for task, position in occurrences[frame_index]:
                image.save(
                    staging[task["event_id"]] / f"rgb_{position:02d}.png",
                    format="PNG",
                    compress_level=6,
                    optimize=False,
                )
        for frame_index, payload in decode_selected_frames(
            ffmpeg, depth_source, selected, pixel_format="gray16le", bytes_per_pixel=2
        ):
            raw = np.frombuffer(payload, dtype="<u2").reshape(HEIGHT, WIDTH)
            depth_m = (raw.astype(np.float32) / np.float32(DEPTH_SCALE)).astype(np.float32)
            if not np.isfinite(depth_m).all() or np.any(depth_m < 0):
                raise M3MaterializationError("decoded depth contains invalid values")
            for task, position in occurrences[frame_index]:
                np.save(
                    staging[task["event_id"]] / f"depth_{position:02d}.npy",
                    depth_m,
                    allow_pickle=False,
                )

        records = []
        for task, binding in zip(tasks, bindings, strict=True):
            event_id = task["event_id"]
            event_staging = staging[event_id]
            np.save(event_staging / "intrinsic.npy", intrinsic, allow_pickle=False)
            mesh_records = {}
            for role in ("tool", "target"):
                source_binding = binding[f"{role}_mesh"]
                source = verify_source_binding(DATA_ROOT, source_binding, source_cache)
                verify_source_binding(DATA_ROOT, binding[f"{role}_pose"], source_cache)
                cache_path, geometry = _mesh_cache_file(
                    source,
                    source_binding["sha256"],
                    mesh_cache,
                    mesh_memory_cache,
                )
                destination = event_staging / f"{role}.obj"
                os.link(cache_path, destination)
                mesh_records[role] = (destination, geometry, source_binding["sha256"])

            frames = task["orchestration_only_never_model_input"]["frame_indices"]
            event_files = {
                "rgb": [
                    file_record(
                        ROOT,
                        event_staging / f"rgb_{position:02d}.png",
                        dtype="uint8",
                        shape=(HEIGHT, WIDTH, 3),
                        unit="srgb_0_255",
                        source_frame_index=frame_index,
                    )
                    for position, frame_index in enumerate(frames)
                ],
                "depth": [
                    file_record(
                        ROOT,
                        event_staging / f"depth_{position:02d}.npy",
                        dtype="float32",
                        shape=(HEIGHT, WIDTH),
                        unit="m",
                        source_frame_index=frame_index,
                    )
                    for position, frame_index in enumerate(frames)
                ],
                "intrinsic": file_record(
                    ROOT,
                    event_staging / "intrinsic.npy",
                    dtype="float64",
                    shape=(3, 3),
                    unit="pixel",
                ),
            }
            for role in ("tool", "target"):
                path, geometry, source_sha = mesh_records[role]
                record = file_record(ROOT, path, dtype="ascii_obj", shape=(geometry["vertex_count"], 3), unit="m")
                record["source_cm_obj_sha256"] = source_sha
                record["geometry"] = geometry
                event_files[f"{role}_mesh"] = record
            destination = INPUT_ROOT / event_id
            promote_event(event_staging, destination)
            staging.pop(event_id)
            prefix = event_staging.relative_to(ROOT).as_posix()
            final_prefix = destination.relative_to(ROOT).as_posix()
            serialized = json.dumps(event_files, sort_keys=True)
            event_files = json.loads(serialized.replace(prefix, final_prefix))
            records.append(
                {
                    "event_id": event_id,
                    "split": "calibration",
                    "frame_indices": frames,
                    "files": event_files,
                }
            )
        return records
    finally:
        for path in staging.values():
            clean_staging(path)


def main() -> int:
    try:
        plan = _load_plan()
        if MANIFEST_PATH.exists():
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            closure = verify_materialization_manifest(manifest, ROOT)
            print(f"[reuse] {closure['event_count']} exact calibration events")
            print(f"[artifact] {MANIFEST_PATH}")
            return 0
        INPUT_ROOT.mkdir(parents=True, exist_ok=True)
        if any(path.name != "_mesh_cache" for path in INPUT_ROOT.iterdir()):
            raise M3MaterializationError("unmanifested calibration input already exists")
        mesh_cache = INPUT_ROOT / "_mesh_cache"
        mesh_cache.mkdir(exist_ok=True)
        staging_root = INPUT_ROOT / ".staging"
        staging_root.mkdir(exist_ok=False)
        ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
        decoder = _decoder_identity(ffmpeg)
        tasks_by_rank: dict[int, list[dict[str, object]]] = {}
        for task in plan["tasks"]:
            rank = task["orchestration_only_never_model_input"]["rank"]
            tasks_by_rank.setdefault(rank, []).append(task)
        if len(tasks_by_rank) != 30 or any(len(tasks) != 5 for tasks in tasks_by_rank.values()):
            raise M3MaterializationError("calibration sequence/event coverage is not 30 x 5")
        events = []
        source_cache: dict[str, tuple[int, str]] = {}
        mesh_memory_cache: dict[str, tuple[bytes, dict[str, object]]] = {}
        try:
            for ordinal, rank in enumerate(sorted(tasks_by_rank), start=1):
                events.extend(
                    _materialize_sequence(
                        tasks_by_rank[rank],
                        ffmpeg=ffmpeg,
                        source_cache=source_cache,
                        mesh_memory_cache=mesh_memory_cache,
                        mesh_cache=mesh_cache,
                        staging_root=staging_root,
                    )
                )
                print(f"[materialized] calibration sequence {ordinal}/30", flush=True)
        finally:
            clean_staging(staging_root)
        manifest = {
            "schema": MATERIALIZATION_SCHEMA,
            "status": "READY_FOR_CALIBRATION_INFERENCE",
            "task_plan_sha256": TASK_PLAN_SHA256,
            "event_count": 150,
            "calibration_sequence_count": 30,
            "evaluation_event_count": 0,
            "evaluation_model_results_opened": False,
            "rgb_rule": "input frame index decode to RGB uint8 PNG; no resampling",
            "depth_rule": "input frame index decode as gray16le then float32(raw/4000) metres; no resampling",
            "mesh_rule": "OBJ vertex coordinates exactly multiplied by decimal 0.01 from cm to m",
            "decoder": decoder,
            "events": sorted(events, key=lambda event: event["event_id"]),
        }
        atomic_json(MANIFEST_PATH, manifest)
        closure = verify_materialization_manifest(manifest, ROOT)
    except (
        OSError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
        M3AdapterContractError,
        M3MaterializationError,
    ) as exc:
        print(f"[hold] HOLD_ENGINEERING_INCOMPLETE: {exc}", file=sys.stderr, flush=True)
        return 3
    print(f"[artifact] {MANIFEST_PATH}")
    print(f"[status] READY_FOR_CALIBRATION_INFERENCE ({closure['total_bytes']} referenced bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
