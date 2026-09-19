"""Shared fail-closed helpers for frozen M3 GPU inference jobs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Mapping

import numpy as np


class M3JobError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, object]) -> None:
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
        if temporary.exists():
            temporary.unlink()


def atomic_npy(path: Path, value: np.ndarray) -> None:
    """Write one NumPy artifact without exposing a partial final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npy", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, value, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_exact_json(path: Path, expected_sha256: str) -> dict[str, object]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise M3JobError(f"frozen JSON changed: {path.name}")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise M3JobError(f"frozen JSON is not an object: {path.name}")
    return value


def safe_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise M3JobError("model input path is not safe and relative")
    if root.is_symlink() or not root.is_dir():
        raise M3JobError("workspace root is missing or symlinked")
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise M3JobError("model input path contains a symlink")
    if not current.is_file():
        raise M3JobError("model input file is missing")
    return current


def verify_bound_files(root: Path, events: list[dict[str, object]]) -> int:
    cache: dict[str, tuple[int, str]] = {}
    for event in events:
        binding = event.get("materialized_file_binding")
        if not isinstance(binding, dict):
            raise M3JobError("event lacks materialized file binding")
        for records in binding.values():
            if not isinstance(records, list):
                raise M3JobError("file binding is not itemized")
            for record in records:
                if not isinstance(record, dict):
                    raise M3JobError("file record is malformed")
                relative = record.get("relative_path")
                expected_size = record.get("bytes")
                expected_sha = record.get("sha256")
                if not isinstance(relative, str) or not isinstance(expected_size, int) or not isinstance(expected_sha, str):
                    raise M3JobError("file record identity is malformed")
                path = safe_file(root, relative)
                observed = cache.get(relative)
                if observed is None:
                    observed = (path.stat().st_size, sha256_file(path))
                    cache[relative] = observed
                if observed != (expected_size, expected_sha):
                    raise M3JobError(f"model input changed: {relative}")
    return len(cache)


def verify_resource_manifest(manifest_path: Path, expected_sha256: str, resource_root: Path) -> int:
    """Verify the complete 26-file frozen model/repository resource closure."""
    manifest = load_exact_json(manifest_path, expected_sha256)
    rows = manifest.get("files")
    if manifest.get("schema") != "quiethand.m3.checkpoint_landing.v1" or not isinstance(rows, list) or len(rows) != 26:
        raise M3JobError("checkpoint resource manifest is not the complete frozen closure")
    if resource_root.is_symlink() or not resource_root.is_dir():
        raise M3JobError("checkpoint resource root is missing or symlinked")
    observed_paths: set[str] = set()
    total = 0
    for row in rows:
        if not isinstance(row, dict):
            raise M3JobError("checkpoint resource row is malformed")
        relative = row.get("relative_path")
        expected_size = row.get("bytes")
        expected_sha = row.get("sha256")
        if (
            not isinstance(relative, str)
            or relative in observed_paths
            or not isinstance(expected_size, int)
            or expected_size <= 0
            or not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or not isinstance(row.get("source_url"), str)
            or not isinstance(row.get("resolved_revision"), str)
        ):
            raise M3JobError("checkpoint resource identity is malformed or duplicate")
        observed_paths.add(relative)
        path = safe_file(resource_root, relative)
        if path.stat().st_size != expected_size or sha256_file(path) != expected_sha:
            raise M3JobError(f"checkpoint resource changed: {relative}")
        total += expected_size
    closure = manifest.get("closure")
    if not isinstance(closure, dict) or closure.get("file_count") != 26 or closure.get("total_bytes") != total:
        raise M3JobError("checkpoint resource closure summary is inconsistent")
    return total


def verify_repository_binding(
    binding_path: Path,
    expected_sha256: str,
    repository_parent: Path,
    resource_root: Path,
    adapter: str,
) -> int:
    """Verify one extracted source tree against the byte-bound official archive."""
    binding = load_exact_json(binding_path, expected_sha256)
    repositories = binding.get("repositories")
    if (
        binding.get("schema") != "quiethand.m3.repository_binding.v1"
        or binding.get("status") != "READY_REPOSITORY_BINDING"
        or binding.get("repository_count") != 3
        or not isinstance(repositories, list)
    ):
        raise M3JobError("repository binding envelope is invalid")
    matches = [row for row in repositories if isinstance(row, dict) and row.get("adapter") == adapter]
    if len(matches) != 1:
        raise M3JobError("repository binding does not identify exactly one adapter")
    row = matches[0]
    name = row.get("repository_name")
    archive_relative = row.get("archive_relative_path")
    files = row.get("files")
    if not isinstance(name, str) or not isinstance(archive_relative, str) or not isinstance(files, list) or row.get("file_count") != len(files):
        raise M3JobError("repository binding row is malformed")
    archive = safe_file(resource_root, archive_relative)
    if archive.stat().st_size != row.get("archive_bytes") or sha256_file(archive) != row.get("archive_sha256"):
        raise M3JobError("repository archive changed")
    root = repository_parent / name
    if root.is_symlink() or not root.is_dir():
        raise M3JobError("extracted repository root is missing or symlinked")
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict):
            raise M3JobError("repository file row is malformed")
        relative = item.get("relative_path")
        if not isinstance(relative, str) or relative in seen:
            raise M3JobError("repository file identity is malformed or duplicate")
        seen.add(relative)
        if item.get("type") == "symlink":
            pure = PurePosixPath(relative)
            path = root.joinpath(*pure.parts)
            if not path.is_symlink() or os.readlink(path) != item.get("target"):
                raise M3JobError(f"repository symlink changed: {relative}")
        elif item.get("type") == "file":
            path = safe_file(root, relative)
            if path.stat().st_size != item.get("bytes") or sha256_file(path) != item.get("sha256"):
                raise M3JobError(f"repository source changed: {relative}")
        else:
            raise M3JobError("repository member type is unsupported")
    return len(seen)


def status(path: Path, state: str, completed: int, total: int, detail: str | None = None) -> None:
    value: dict[str, object] = {"state": state, "completed": completed, "total": total}
    if detail is not None:
        value["detail"] = detail
    atomic_json(path, value)
