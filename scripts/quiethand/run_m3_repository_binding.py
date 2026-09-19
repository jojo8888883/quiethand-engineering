#!/usr/bin/env python3
"""Bind each extracted M3 repository byte-for-byte to its frozen archive."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "artifacts" / "quiethand" / "m3" / "QH_M3_CHECKPOINT_PLAN.json"
PLAN_SHA256 = "f00a048c1b81109959e1279d96eef808af8e06938a679c67dd12985a10b0d703"
RESOURCE_ROOT = ROOT / "external_data" / "quiethand_m3_resources"
REPOSITORY_ROOT = ROOT / "external_repos" / "quiethand_m3"
OUTPUT = ROOT / "artifacts" / "quiethand" / "m3" / "QH_M3_REPOSITORY_BINDING.json"
ROOT_NAMES = {
    "object_state": "FoundationPose",
    "raw_hand": "HaWoR",
    "segmentation": "sam2",
}


class RepositoryBindingError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, object]) -> None:
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


def safe_regular(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise RepositoryBindingError("repository member path is unsafe")
    if root.is_symlink() or not root.is_dir():
        raise RepositoryBindingError("repository root is missing or symlinked")
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise RepositoryBindingError("repository member path contains a symlink")
    if not current.is_file():
        raise RepositoryBindingError(f"repository member is missing: {relative}")
    return current


def bind_archive(archive: Path, extracted: Path, expected_top: str) -> list[dict[str, object]]:
    if archive.is_symlink() or not archive.is_file():
        raise RepositoryBindingError("repository archive is missing or symlinked")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    with tarfile.open(archive, mode="r:gz") as handle:
        for member in handle.getmembers():
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts or not pure.parts or pure.parts[0] != expected_top:
                raise RepositoryBindingError("archive has an unsafe or unexpected root")
            if member.isdir():
                continue
            relative = PurePosixPath(*pure.parts[1:]).as_posix()
            if not relative or relative in seen:
                raise RepositoryBindingError("archive has an empty or duplicate file member")
            seen.add(relative)
            if member.issym():
                target = PurePosixPath(member.linkname)
                combined = PurePosixPath(relative).parent / target
                if target.is_absolute() or ".." in combined.parts:
                    raise RepositoryBindingError("archive symlink escapes the repository")
                path = extracted / relative
                if not path.is_symlink() or os.readlink(path) != member.linkname:
                    raise RepositoryBindingError(f"extracted repository symlink differs: {relative}")
                rows.append({"relative_path": relative, "type": "symlink", "target": member.linkname})
                continue
            if not member.isfile():
                raise RepositoryBindingError("archive contains an unsupported member type")
            source = handle.extractfile(member)
            if source is None:
                raise RepositoryBindingError("archive member could not be opened")
            digest = hashlib.sha256()
            size = 0
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
            if size != member.size:
                raise RepositoryBindingError("archive member byte count changed while reading")
            path = safe_regular(extracted, relative)
            if path.stat().st_size != size or sha256_file(path) != digest.hexdigest():
                raise RepositoryBindingError(f"extracted repository differs from archive: {relative}")
            rows.append({"relative_path": relative, "type": "file", "bytes": size, "sha256": digest.hexdigest()})
    actual = {
        path.relative_to(extracted).as_posix()
        for path in extracted.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual != seen:
        raise RepositoryBindingError("extracted repository has missing or extra files")
    return sorted(rows, key=lambda row: row["relative_path"])


def main() -> int:
    try:
        if sha256_file(PLAN) != PLAN_SHA256:
            raise RepositoryBindingError("frozen checkpoint plan changed")
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        repositories = []
        for adapter in sorted(ROOT_NAMES):
            source = plan["repository_archives"][adapter]
            commit = source["resolved_revision"]
            root_name = ROOT_NAMES[adapter]
            archive = RESOURCE_ROOT / source["relative_path"]
            extracted = REPOSITORY_ROOT / root_name
            files = bind_archive(archive, extracted, f"{root_name}-{commit}")
            repositories.append(
                {
                    "adapter": adapter,
                    "repository_name": root_name,
                    "repository_commit": commit,
                    "archive_relative_path": source["relative_path"],
                    "archive_bytes": archive.stat().st_size,
                    "archive_sha256": sha256_file(archive),
                    "file_count": len(files),
                    "files": files,
                }
            )
        artifact = {
            "schema": "quiethand.m3.repository_binding.v1",
            "status": "READY_REPOSITORY_BINDING",
            "checkpoint_plan_sha256": PLAN_SHA256,
            "repository_count": len(repositories),
            "repositories": repositories,
            "scientific_result": False,
        }
        atomic_json(OUTPUT, artifact)
    except (OSError, ValueError, KeyError, json.JSONDecodeError, tarfile.TarError, RepositoryBindingError) as exc:
        print(f"[hold] HOLD_ENGINEERING_INCOMPLETE: {exc}", file=sys.stderr)
        return 3
    print(f"[artifact] {OUTPUT}")
    print(f"[sha256] {sha256_file(OUTPUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
