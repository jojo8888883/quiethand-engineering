"""Exact archive-to-consumed-byte bindings for the frozen QuietHand M2 inputs.

The ARCTIC object containers are legacy NumPy object arrays.  They are only
deserialized from bytes that are re-hashed against a manifest derived from the
three exact-hash official ZIP files.  The same read-once bytes are then passed
to NumPy, so a post-check path swap cannot change what is consumed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Mapping
import zipfile

from quiethand.arctic_contract import (
    M2_ASSETS,
    ResourceAccountingError,
    sha256_exact_file,
)


MANO_ARCHIVE_SHA256 = "50976831790ea9657d8110e0c94e50e90eaf35cd76169f0b27e5d32f3fcd951f"
MANO_ARCHIVE_SIZE = 175_200_815
MANO_MODEL_SHA256 = {
    "left": "c4022f7083f2ca7c78b2b3d595abbab52debd32b09d372b16923a801f0ea6a30",
    "right": "45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767",
}
MANO_MODEL_SIZE = {"left": 3_821_391, "right": 3_821_356}
READ_CHUNK_BYTES = 1024 * 1024


class ArchiveBindingError(RuntimeError):
    """A sanitized exact-input binding failure."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_zip_member(info: zipfile.ZipInfo) -> bool:
    relative = PurePosixPath(info.filename)
    mode = info.external_attr >> 16
    return (
        bool(relative.parts)
        and not relative.is_absolute()
        and ".." not in relative.parts
        and stat.S_IFMT(mode) != stat.S_IFLNK
    )


def _safe_relative(value: str | Path) -> Path:
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ArchiveBindingError("unsafe relative input path")
    return relative


def _assert_regular_without_symlink(path: Path, root: Path) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ArchiveBindingError("input path escaped canonical root") from exc
    current = path
    while current != root:
        try:
            if current.is_symlink():
                raise ArchiveBindingError(f"{path.name}: symlink input is forbidden")
        except OSError as exc:
            raise ArchiveBindingError(f"{path.name}: path inspection failed") from exc
        current = current.parent
    if root.is_symlink() or not path.is_file():
        raise ArchiveBindingError(f"{path.name}: regular canonical file required")


def _assert_canonical_directory(path: Path, *, label: str) -> None:
    try:
        if path.is_symlink() or not path.is_dir():
            raise ArchiveBindingError(f"{label}: non-symlink directory required")
    except OSError as exc:
        raise ArchiveBindingError(f"{label}: directory inspection failed") from exc


def _read_exact_bounded(path: Path, *, root: Path, expected_size: int) -> bytes:
    """Stat before allocation, then read at most exact-size+1 from one open fd."""

    if expected_size < 0:
        raise ArchiveBindingError(f"{path.name}: negative expected size")
    _assert_regular_without_symlink(path, root)
    try:
        before = path.stat()
        if before.st_size != expected_size:
            raise ArchiveBindingError(f"{path.name}: file-size mismatch before read")
        chunks: list[bytes] = []
        remaining = expected_size
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if opened.st_size != expected_size:
                raise ArchiveBindingError(f"{path.name}: file changed before bounded read")
            while remaining:
                chunk = handle.read(min(READ_CHUNK_BYTES, remaining))
                if not chunk:
                    raise ArchiveBindingError(f"{path.name}: short bounded read")
                chunks.append(chunk)
                remaining -= len(chunk)
            if handle.read(1):
                raise ArchiveBindingError(f"{path.name}: oversized bounded read")
            after = os.fstat(handle.fileno())
            if after.st_size != expected_size:
                raise ArchiveBindingError(f"{path.name}: file changed during bounded read")
        return b"".join(chunks)
    except OSError as exc:
        raise ArchiveBindingError(f"{path.name}: bounded read failed") from exc


@dataclass(frozen=True)
class FrozenArcticBinding:
    data_root: Path
    manifest: Mapping[str, tuple[int, str]]
    proof: Mapping[str, object]

    def read(self, root_name: str, relative_path: str | Path) -> bytes:
        relative = _safe_relative(relative_path)
        key = (Path(root_name) / relative).as_posix()
        expected = self.manifest.get(key)
        if expected is None:
            raise ArchiveBindingError(f"{key}: file is not in the frozen archive manifest")
        path = self.data_root / "data" / root_name / relative
        _assert_canonical_directory(self.data_root, label="ARCTIC data_root")
        _assert_canonical_directory(self.data_root / "data", label="ARCTIC data/")
        _assert_canonical_directory(
            self.data_root / "data" / root_name, label=f"ARCTIC {root_name}/"
        )
        expected_size, expected_sha256 = expected
        data = _read_exact_bounded(
            path,
            root=self.data_root / "data" / root_name,
            expected_size=expected_size,
        )
        if sha256_bytes(data) != expected_sha256:
            raise ArchiveBindingError(f"{key}: extracted bytes differ from frozen archive")
        return data


@dataclass(frozen=True)
class FrozenManoBinding:
    mano_root: Path
    model_bytes: Mapping[str, bytes]
    proof: Mapping[str, object]

    def read_model(self, side: str) -> bytes:
        if side not in MANO_MODEL_SHA256:
            raise ArchiveBindingError("MANO side must be left or right")
        path = self.mano_root / "models" / f"MANO_{side.upper()}.pkl"
        _assert_canonical_directory(self.mano_root, label="MANO root")
        _assert_canonical_directory(self.mano_root / "models", label="MANO models/")
        data = _read_exact_bounded(
            path,
            root=self.mano_root / "models",
            expected_size=MANO_MODEL_SIZE[side],
        )
        if data != self.model_bytes[side]:
            raise ArchiveBindingError(f"{path.name}: changed after archive binding")
        return data


def verify_frozen_arctic_extraction(data_root: Path) -> FrozenArcticBinding:
    """Prove exact canonical extracted trees against all three pinned ZIPs."""

    data_root = Path(data_root)
    canonical_parent = data_root / "data"
    manifest: dict[str, tuple[int, str]] = {}
    asset_proofs: list[dict[str, object]] = []
    manifest_lines: list[str] = []
    try:
        _assert_canonical_directory(data_root, label="ARCTIC data_root")
        _assert_canonical_directory(canonical_parent, label="ARCTIC data/")
        for asset in M2_ASSETS:
            alternate = data_root / asset.name
            if alternate.exists() or alternate.is_symlink():
                raise ArchiveBindingError(
                    f"{asset.name}: alternate top-level extraction root is forbidden"
                )
            archive_path = data_root / asset.relative_path
            _assert_regular_without_symlink(archive_path, canonical_parent)
            archive_sha = sha256_exact_file(
                archive_path,
                root=canonical_parent,
                expected_size=asset.size_bytes,
            )
            if archive_sha != asset.sha256:
                raise ArchiveBindingError(f"{asset.name}: frozen archive SHA-256 mismatch")
            root = canonical_parent / asset.name
            _assert_canonical_directory(root, label=f"ARCTIC {asset.name}/")

            expected_paths: set[str] = set()
            archive_total = 0
            with zipfile.ZipFile(archive_path) as archive:
                infos = archive.infolist()
                if not infos or any(not _safe_zip_member(info) for info in infos):
                    raise ArchiveBindingError(f"{asset.name}: unsafe or empty ZIP")
                for info in infos:
                    if info.is_dir():
                        continue
                    member = PurePosixPath(info.filename)
                    if not member.parts or member.parts[0] != asset.name or len(member.parts) < 2:
                        raise ArchiveBindingError(
                            f"{asset.name}: ZIP member is outside its frozen root"
                        )
                    relative = Path(*member.parts[1:])
                    key = (Path(asset.name) / relative).as_posix()
                    if key in manifest:
                        raise ArchiveBindingError(f"{asset.name}: duplicate ZIP member")
                    archive_bytes = archive.read(info)
                    if len(archive_bytes) != info.file_size:
                        raise ArchiveBindingError(f"{asset.name}: short ZIP member read")
                    member_sha = sha256_bytes(archive_bytes)
                    manifest[key] = (info.file_size, member_sha)
                    expected_paths.add(relative.as_posix())
                    archive_total += info.file_size
                    manifest_lines.append(f"{key}\t{info.file_size}\t{member_sha}\n")

            actual_paths: set[str] = set()
            for path in root.rglob("*"):
                if path.is_symlink():
                    raise ArchiveBindingError(f"{asset.name}: symlink in extracted tree")
                if path.is_file():
                    actual_paths.add(path.relative_to(root).as_posix())
            if actual_paths != expected_paths:
                missing = len(expected_paths - actual_paths)
                extra = len(actual_paths - expected_paths)
                raise ArchiveBindingError(
                    f"{asset.name}: extracted member set mismatch (missing={missing}, extra={extra})"
                )
            for relative_string in sorted(expected_paths):
                key = (Path(asset.name) / relative_string).as_posix()
                path = root / relative_string
                _assert_regular_without_symlink(path, root)
                expected_size, expected_sha = manifest[key]
                data = _read_exact_bounded(
                    path, root=root, expected_size=expected_size
                )
                if sha256_bytes(data) != expected_sha:
                    raise ArchiveBindingError(
                        f"{asset.name}/{relative_string}: extracted byte mismatch"
                    )
            asset_proofs.append(
                {
                    "name": asset.name,
                    "archive_path": archive_path.as_posix(),
                    "archive_sha256": archive_sha,
                    "file_count": len(expected_paths),
                    "uncompressed_bytes": archive_total,
                    "missing_file_count": 0,
                    "extra_file_count": 0,
                    "byte_mismatch_count": 0,
                    "symlink_count": 0,
                    "closed": True,
                }
            )
    except (OSError, zipfile.BadZipFile, ResourceAccountingError, RuntimeError) as exc:
        if isinstance(exc, ArchiveBindingError):
            raise
        raise ArchiveBindingError(f"ARCTIC extraction binding failed ({type(exc).__name__})") from exc

    proof = {
        "canonical_parent": canonical_parent.as_posix(),
        "alternate_roots_allowed": False,
        "consumption_mode": "read-once bytes rehashed against exact-ZIP manifest",
        "assets": asset_proofs,
        "total_file_count": len(manifest),
        "manifest_sha256": sha256_bytes("".join(sorted(manifest_lines)).encode("utf-8")),
        "closed": len(asset_proofs) == len(M2_ASSETS),
    }
    return FrozenArcticBinding(data_root=data_root, manifest=manifest, proof=proof)


def verify_frozen_mano_extraction(mano_root: Path) -> FrozenManoBinding:
    """Bind both installed MANO files to the pinned official archive and bytes."""

    mano_root = Path(mano_root)
    _assert_canonical_directory(mano_root, label="MANO root")
    _assert_canonical_directory(mano_root / "models", label="MANO models/")
    archive_path = mano_root / "mano_v1_2.zip"
    _assert_regular_without_symlink(archive_path, mano_root)
    try:
        archive_sha = sha256_exact_file(
            archive_path,
            root=mano_root,
            expected_size=MANO_ARCHIVE_SIZE,
        )
    except ResourceAccountingError as exc:
        raise ArchiveBindingError("MANO archive exact-file binding failed") from exc
    if archive_sha != MANO_ARCHIVE_SHA256:
        raise ArchiveBindingError("MANO archive SHA-256 mismatch")
    selected: dict[str, bytes] = {}
    members: dict[str, dict[str, object]] = {}
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if not infos or any(not _safe_zip_member(info) for info in infos):
                raise ArchiveBindingError("MANO archive is unsafe or empty")
            for info in infos:
                if info.is_dir():
                    continue
                basename = PurePosixPath(info.filename).name
                for side in ("left", "right"):
                    wanted = f"MANO_{side.upper()}.pkl"
                    if basename == wanted:
                        if side in selected:
                            raise ArchiveBindingError(f"MANO archive has duplicate {wanted}")
                        data = archive.read(info)
                        digest = sha256_bytes(data)
                        if len(data) != info.file_size or digest != MANO_MODEL_SHA256[side]:
                            raise ArchiveBindingError(f"{wanted}: frozen model mismatch")
                        selected[side] = data
                        members[side] = {
                            "archive_member": info.filename,
                            "bytes": len(data),
                            "sha256": digest,
                            "crc32": info.CRC,
                        }
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, ArchiveBindingError):
            raise
        raise ArchiveBindingError(f"MANO archive binding failed ({type(exc).__name__})") from exc
    if set(selected) != {"left", "right"}:
        raise ArchiveBindingError("MANO archive must contain exactly one model per side")
    for side, archive_bytes in selected.items():
        path = mano_root / "models" / f"MANO_{side.upper()}.pkl"
        _assert_regular_without_symlink(path, mano_root / "models")
        extracted = _read_exact_bounded(
            path,
            root=mano_root / "models",
            expected_size=MANO_MODEL_SIZE[side],
        )
        if extracted != archive_bytes:
            raise ArchiveBindingError(f"{path.name}: extracted bytes differ from archive")
    proof = {
        "archive_path": archive_path.as_posix(),
        "archive_sha256": archive_sha,
        "models": members,
        "consumption_mode": "read-once exact model bytes",
        "closed": True,
    }
    return FrozenManoBinding(mano_root=mano_root, model_bytes=selected, proof=proof)
