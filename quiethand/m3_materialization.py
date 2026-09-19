"""Deterministic, calibration-only M3 input materialization.

The model-facing tree contains opaque event identifiers only.  Source identity,
TACO labels, and native geometry remain in the orchestration manifest and are
never passed to an adapter.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

from .m3_adapter_contract import M3AdapterContractError


WIDTH = 1920
HEIGHT = 1080
DEPTH_SCALE = 4000.0
TASK_PLAN_SHA256 = "11fdaff266bebb0ed2fdb46d529b35eefe6ea493842c769fc83d64fe91833531"
MATERIALIZATION_SCHEMA = "quiethand.m3.calibration_materialization.v1"


class M3MaterializationError(RuntimeError):
    """Materialized inputs do not close the frozen M3 contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_regular(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise M3MaterializationError("source path is not safe and relative")
    if root.is_symlink() or not root.is_dir():
        raise M3MaterializationError("source root is missing or symlinked")
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise M3MaterializationError("source path contains a symlink")
    try:
        mode = current.stat(follow_symlinks=False).st_mode
    except OSError as exc:
        raise M3MaterializationError("source path is unreadable") from exc
    if not stat.S_ISREG(mode):
        raise M3MaterializationError("source path is not a regular file")
    return current


def verify_source_binding(
    data_root: Path,
    binding: Mapping[str, object],
    cache: dict[str, tuple[int, str]],
) -> Path:
    relative = binding.get("relative_path")
    expected_size = binding.get("bytes")
    expected_sha = binding.get("sha256")
    if (
        not isinstance(relative, str)
        or isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size <= 0
        or not isinstance(expected_sha, str)
        or len(expected_sha) != 64
    ):
        raise M3MaterializationError("source binding is malformed")
    path = _safe_regular(data_root, relative)
    key = str(path)
    observed = cache.get(key)
    if observed is None:
        observed = (path.stat(follow_symlinks=False).st_size, sha256_file(path))
        cache[key] = observed
    if observed != (expected_size, expected_sha):
        raise M3MaterializationError(f"source binding changed: {relative}")
    return path


def _read_exact(stream, byte_count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise M3MaterializationError("decoder ended before a selected frame completed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def decode_selected_frames(
    ffmpeg: Path,
    source: Path,
    frame_indices: Sequence[int],
    *,
    pixel_format: str,
    bytes_per_pixel: int,
):
    """Yield exact selected frames by input index, without any time resampling."""

    if not frame_indices or list(frame_indices) != sorted(set(frame_indices)):
        raise M3MaterializationError("selected frame indices are not unique and sorted")
    expression = "+".join(f"eq(n\\,{index})" for index in frame_indices)
    command = [
        str(ffmpeg),
        "-v",
        "error",
        "-nostdin",
        "-threads",
        "1",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        f"select={expression}",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pixel_format,
        "pipe:1",
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    frame_bytes = WIDTH * HEIGHT * bytes_per_pixel
    try:
        for index in frame_indices:
            yield index, _read_exact(process.stdout, frame_bytes)
        if process.stdout.read(1):
            raise M3MaterializationError("decoder emitted more selected frames than requested")
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if return_code != 0 or stderr.strip():
            raise M3MaterializationError(
                f"decoder failed for {source.name}: {stderr.strip() or return_code}"
            )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def _decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def scale_obj_cm_to_m(source: Path) -> tuple[bytes, dict[str, object]]:
    output = [b"# QuietHand M3: vertex coordinates converted exactly cm -> m\n"]
    vertex_count = 0
    face_count = 0
    minimum = [Decimal("Infinity")] * 3
    maximum = [Decimal("-Infinity")] * 3
    for raw in source.read_bytes().splitlines(keepends=True):
        if raw.startswith(b"v "):
            try:
                text = raw.decode("ascii").strip()
                parts = text.split()
                if len(parts) < 4:
                    raise ValueError("short vertex")
                values = [Decimal(parts[index]) * Decimal("0.01") for index in range(1, 4)]
            except (UnicodeDecodeError, InvalidOperation, ValueError) as exc:
                raise M3MaterializationError("object mesh contains an invalid vertex") from exc
            if not all(value.is_finite() for value in values):
                raise M3MaterializationError("object mesh contains a non-finite vertex")
            for axis, value in enumerate(values):
                minimum[axis] = min(minimum[axis], value)
                maximum[axis] = max(maximum[axis], value)
            suffix = "" if len(parts) == 4 else " " + " ".join(parts[4:])
            output.append(
                (
                    "v "
                    + " ".join(_decimal_text(value) for value in values)
                    + suffix
                    + "\n"
                ).encode("ascii")
            )
            vertex_count += 1
        else:
            output.append(raw if raw.endswith((b"\n", b"\r")) else raw + b"\n")
            if raw.startswith(b"f "):
                face_count += 1
    if vertex_count < 3 or face_count < 1:
        raise M3MaterializationError("object mesh lacks vertices or faces")
    bounds = [float(value) for value in minimum + maximum]
    if not all(np.isfinite(bounds)) or max(abs(value) for value in bounds) > 10.0:
        raise M3MaterializationError("meter-scaled object mesh bounds are invalid")
    payload = b"".join(output)
    return payload, {
        "vertex_count": vertex_count,
        "face_count": face_count,
        "bounds_m": {
            "minimum": bounds[:3],
            "maximum": bounds[3:],
        },
    }


def file_record(
    root: Path,
    path: Path,
    *,
    dtype: str,
    shape: Sequence[int],
    unit: str,
    source_frame_index: int | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": path.stat(follow_symlinks=False).st_size,
        "sha256": sha256_file(path),
        "dtype": dtype,
        "shape": list(shape),
        "unit": unit,
    }
    if source_frame_index is not None:
        record["source_frame_index"] = source_frame_index
    return record


def verify_materialization_manifest(
    manifest: Mapping[str, object], workspace_root: Path
) -> dict[str, int]:
    if (
        manifest.get("schema") != MATERIALIZATION_SCHEMA
        or manifest.get("status") != "READY_FOR_CALIBRATION_INFERENCE"
        or manifest.get("task_plan_sha256") != TASK_PLAN_SHA256
        or manifest.get("event_count") != 150
        or manifest.get("evaluation_event_count") != 0
        or manifest.get("evaluation_model_results_opened") is not False
    ):
        raise M3MaterializationError("materialization envelope is not closed")
    events = manifest.get("events")
    if not isinstance(events, list) or len(events) != 150:
        raise M3MaterializationError("materialization event coverage is not exact")
    event_ids: set[str] = set()
    file_count = 0
    total_bytes = 0
    for event in events:
        if not isinstance(event, Mapping) or event.get("split") != "calibration":
            raise M3MaterializationError("non-calibration materialization event found")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or event_id in event_ids:
            raise M3MaterializationError("materialization event ID is invalid or duplicate")
        event_ids.add(event_id)
        files = event.get("files")
        if not isinstance(files, Mapping):
            raise M3MaterializationError("materialization event files are missing")
        rgb = files.get("rgb")
        depth = files.get("depth")
        if not isinstance(rgb, list) or len(rgb) != 15 or not isinstance(depth, list) or len(depth) != 15:
            raise M3MaterializationError("event does not contain exactly 15 RGB-D frames")
        records = [*rgb, *depth, files.get("intrinsic"), files.get("tool_mesh"), files.get("target_mesh")]
        for record in records:
            if not isinstance(record, Mapping):
                raise M3MaterializationError("materialized file record is malformed")
            relative = record.get("relative_path")
            expected_size = record.get("bytes")
            expected_sha = record.get("sha256")
            if not isinstance(relative, str) or not isinstance(expected_size, int) or not isinstance(expected_sha, str):
                raise M3MaterializationError("materialized file identity is malformed")
            path = _safe_regular(workspace_root, relative)
            size = path.stat(follow_symlinks=False).st_size
            if size != expected_size or sha256_file(path) != expected_sha:
                raise M3MaterializationError("materialized file changed after creation")
            file_count += 1
            total_bytes += size
        for record in rgb:
            path = workspace_root / str(record["relative_path"])
            with Image.open(path) as image:
                if image.mode != "RGB" or image.size != (WIDTH, HEIGHT):
                    raise M3MaterializationError("RGB frame shape or mode changed")
        for record in depth:
            array = np.load(workspace_root / str(record["relative_path"]), allow_pickle=False)
            if array.dtype != np.float32 or array.shape != (HEIGHT, WIDTH) or not np.isfinite(array).all() or np.any(array < 0):
                raise M3MaterializationError("depth frame contract changed")
        intrinsic = np.load(workspace_root / str(files["intrinsic"]["relative_path"]), allow_pickle=False)
        if intrinsic.dtype != np.float64 or intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise M3MaterializationError("intrinsic matrix contract changed")
    return {"event_count": len(event_ids), "file_count": file_count, "total_bytes": total_bytes}


def promote_event(staging: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise M3MaterializationError(f"refusing to overwrite existing event: {destination.name}")
    os.replace(staging, destination)


def clean_staging(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
