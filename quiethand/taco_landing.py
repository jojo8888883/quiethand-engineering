"""Selective TACO ZIP-member landing with CRC, SHA-256, and progress receipts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import time
from typing import Mapping, Sequence
import zipfile
import zlib

from .http_range_zip import HTTPRangeReader
from .taco_contract import TACO_DATA_CAP_BYTES, TACO_SELECTION_SHA256
from .taco_source import (
    TacoSourceError,
    bind_required_archives,
    catalog_line,
    fetch_dropbox_listing,
)


CHUNK_BYTES = 1024 * 1024
FREE_SPACE_RESERVE_BYTES = 2 * 1024**3


class TacoLandingError(RuntimeError):
    """Selective TACO landing could not be completed fail-closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def crc32_and_sha256_file(path: Path) -> tuple[str, str, int]:
    crc = 0
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            crc = zlib.crc32(chunk, crc)
            digest.update(chunk)
            size += len(chunk)
    return f"{crc & 0xFFFFFFFF:08x}", digest.hexdigest(), size


def _assert_regular_path(path: Path, root: Path, *, allow_missing: bool) -> None:
    root = root.resolve(strict=True)
    candidate = path
    while True:
        if candidate.exists() or candidate.is_symlink():
            try:
                mode = candidate.lstat().st_mode
            except OSError as exc:
                raise TacoLandingError("cannot stat a managed TACO path") from exc
            if stat.S_ISLNK(mode):
                raise TacoLandingError("symlinks are forbidden in the TACO data root")
        elif not allow_missing:
            raise TacoLandingError("required managed TACO path is missing")
        if candidate == root:
            return
        if root not in candidate.parents:
            raise TacoLandingError("managed TACO path escapes its root")
        candidate = candidate.parent


def member_destination(data_root: Path, member_path: str) -> Path:
    pure = PurePosixPath(member_path)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "\\" in member_path
        or any(ord(character) < 32 for character in member_path)
    ):
        raise TacoLandingError("unsafe selected TACO member path")
    destination = data_root.joinpath(*pure.parts)
    if data_root not in destination.parents:
        raise TacoLandingError("selected TACO member escapes its root")
    return destination


def expected_info_matches(info: zipfile.ZipInfo, expected: Mapping[str, object]) -> bool:
    return (
        info.filename == expected.get("path")
        and f"{info.CRC:08x}" == expected.get("crc32")
        and info.compress_size == expected.get("compressed_bytes")
        and info.file_size == expected.get("uncompressed_bytes")
        and info.header_offset == expected.get("header_offset")
        and info.compress_type == expected.get("compression")
        and info.flag_bits == expected.get("flag_bits")
        and not info.is_dir()
        and not (info.flag_bits & 0x1)
    )


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _move_invalid(path: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    target = path.with_name(f"{path.name}.invalid.{timestamp}")
    counter = 1
    while target.exists():
        target = path.with_name(f"{path.name}.invalid.{timestamp}.{counter}")
        counter += 1
    path.rename(target)
    return target


def _extract_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    expected: Mapping[str, object],
    destination: Path,
    data_root: Path,
) -> dict[str, object]:
    _assert_regular_path(destination.parent, data_root, allow_missing=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _assert_regular_path(destination.parent, data_root, allow_missing=False)

    if destination.exists() or destination.is_symlink():
        _assert_regular_path(destination, data_root, allow_missing=False)
        if not destination.is_file():
            raise TacoLandingError("selected TACO destination is not a regular file")
        crc, digest, size = crc32_and_sha256_file(destination)
        if size == expected["uncompressed_bytes"] and crc == expected["crc32"]:
            return {
                "path": expected["path"],
                "bytes": size,
                "crc32": crc,
                "sha256": digest,
                "reused": True,
            }
        _move_invalid(destination)

    partial = destination.with_name(f".{destination.name}.part")
    if partial.is_symlink() or (partial.exists() and not partial.is_file()):
        raise TacoLandingError("managed TACO partial path is unsafe")
    crc = 0
    digest = hashlib.sha256()
    written = 0
    try:
        with archive.open(info, "r") as source, partial.open("wb") as output:
            while chunk := source.read(CHUNK_BYTES):
                output.write(chunk)
                crc = zlib.crc32(chunk, crc)
                digest.update(chunk)
                written += len(chunk)
                if written > int(expected["uncompressed_bytes"]):
                    raise TacoLandingError("selected TACO member exceeded pinned size")
            output.flush()
            os.fsync(output.fileno())
    except zipfile.BadZipFile as exc:
        raise TacoLandingError("selected TACO member failed ZIP CRC validation") from exc
    observed_crc = f"{crc & 0xFFFFFFFF:08x}"
    if written != expected["uncompressed_bytes"] or observed_crc != expected["crc32"]:
        raise TacoLandingError("selected TACO member differs from its catalog")
    os.replace(partial, destination)
    _assert_regular_path(destination, data_root, allow_missing=False)
    return {
        "path": expected["path"],
        "bytes": written,
        "crc32": observed_crc,
        "sha256": digest.hexdigest(),
        "reused": False,
    }


def _catalog_digest(infos: Sequence[zipfile.ZipInfo]) -> str:
    digest = hashlib.sha256()
    seen: set[str] = set()
    for info in infos:
        if info.filename in seen:
            raise TacoLandingError("remote ZIP contains duplicate members")
        seen.add(info.filename)
        digest.update(catalog_line(info))
    return digest.hexdigest()


def land_probe_members(
    *,
    probe_path: Path,
    data_root: Path,
    receipt_path: Path,
) -> dict[str, object]:
    probe_bytes = probe_path.read_bytes()
    probe_sha256 = hashlib.sha256(probe_bytes).hexdigest()
    probe = json.loads(probe_bytes)
    if (
        probe.get("schema") != "quiethand.m3.taco_probe.v1"
        or probe.get("status") != "READY_FOR_SELECTIVE_LANDING"
        or probe.get("selection_sha256") != TACO_SELECTION_SHA256
    ):
        raise TacoLandingError("M3 TACO probe is not authorized for landing")
    catalogs = probe.get("catalogs")
    if not isinstance(catalogs, list) or len(catalogs) != 7:
        raise TacoLandingError("M3 TACO probe archive set is incomplete")

    data_root.parent.mkdir(parents=True, exist_ok=True)
    _assert_regular_path(data_root.parent, data_root.parent, allow_missing=False)
    if data_root.is_symlink():
        raise TacoLandingError("TACO data root may not be a symlink")
    data_root.mkdir(parents=True, exist_ok=True)
    _assert_regular_path(data_root, data_root, allow_missing=False)

    fresh_listing_bytes, fresh_listing = fetch_dropbox_listing()
    fresh_archives = bind_required_archives(fresh_listing)
    expected_archives = {str(item["filename"]): item for item in catalogs}
    for name, expected in expected_archives.items():
        fresh = fresh_archives[name]
        for field in ("bytes", "file_id", "revision_id", "href"):
            if fresh[field] != expected[field]:
                raise TacoLandingError(f"{name}: Dropbox source identity changed")

    all_members = [
        member for catalog in catalogs for member in catalog["selected_members"]
    ]
    total_expected = sum(int(item["uncompressed_bytes"]) for item in all_members)
    if total_expected != probe.get("selected_uncompressed_bytes"):
        raise TacoLandingError("probe selected-byte total does not close")
    if total_expected <= 0 or total_expected > TACO_DATA_CAP_BYTES:
        raise TacoLandingError("selected TACO slice violates the data cap")
    free_bytes = shutil.disk_usage(data_root).free
    if free_bytes < total_expected + FREE_SPACE_RESERVE_BYTES:
        raise TacoLandingError("insufficient free space for selected TACO slice")

    receipt: dict[str, object] = {
        "schema": "quiethand.m3.taco_landing_receipt.v1",
        "status": "LANDING_IN_PROGRESS",
        "scientific_result": False,
        "probe_sha256": probe_sha256,
        "selection_sha256": TACO_SELECTION_SHA256,
        "selected_member_manifest_sha256": probe["selected_member_manifest_sha256"],
        "fresh_listing_sha256": hashlib.sha256(fresh_listing_bytes).hexdigest(),
        "expected_file_count": len(all_members),
        "expected_uncompressed_bytes": total_expected,
        "completed": {},
        "started_unix_seconds": int(time.time()),
    }
    if receipt_path.is_file():
        prior = json.loads(receipt_path.read_bytes())
        if (
            prior.get("probe_sha256") == probe_sha256
            and prior.get("selected_member_manifest_sha256")
            == probe["selected_member_manifest_sha256"]
            and isinstance(prior.get("completed"), dict)
        ):
            receipt["completed"] = prior["completed"]
            receipt["started_unix_seconds"] = prior.get(
                "started_unix_seconds", receipt["started_unix_seconds"]
            )

    completed: dict[str, object] = receipt["completed"]  # type: ignore[assignment]
    completed_bytes = 0
    completed_files = 0
    next_report = 256 * 1024**2
    for catalog in catalogs:
        name = str(catalog["filename"])
        print(f"[archive] {name}", flush=True)
        fresh = fresh_archives[name]
        reader = HTTPRangeReader(
            str(fresh["href"]).replace("dl=0", "dl=1"), int(fresh["bytes"])
        )
        with zipfile.ZipFile(reader) as archive:
            infos = archive.infolist()
            if _catalog_digest(infos) != catalog["catalog_sha256"]:
                raise TacoLandingError(f"{name}: remote ZIP catalog changed")
            by_name = {info.filename: info for info in infos}
            expected_members = sorted(
                catalog["selected_members"], key=lambda item: int(item["header_offset"])
            )
            for expected in expected_members:
                info = by_name.get(str(expected["path"]))
                if info is None or not expected_info_matches(info, expected):
                    raise TacoLandingError(f"{name}: selected member metadata changed")
                destination = member_destination(data_root, str(expected["path"]))
                record = _extract_member(
                    archive, info, expected, destination, data_root
                )
                completed[str(expected["path"])] = {"archive": name, **record}
                if not record["reused"]:
                    completed_files += 1
                    completed_bytes += int(record["bytes"])
                receipt["completed_file_count"] = len(completed)
                receipt["completed_uncompressed_bytes"] = sum(
                    int(item["bytes"]) for item in completed.values()
                )
                _atomic_json(receipt_path, receipt)
                if not record["reused"] and (
                    completed_bytes >= next_report or completed_files % 50 == 0
                ):
                    print(
                        f"  {len(completed)}/{len(all_members)} files, "
                        f"{int(receipt['completed_uncompressed_bytes']) / 1024**3:.2f} GiB",
                        flush=True,
                    )
                    next_report = completed_bytes + 256 * 1024**2

    if len(completed) != len(all_members):
        raise TacoLandingError("selective TACO landing file count is incomplete")
    observed_bytes = sum(int(item["bytes"]) for item in completed.values())
    if observed_bytes != total_expected:
        raise TacoLandingError("selective TACO landing byte total is incomplete")
    receipt["status"] = "READY_FOR_NATIVE_VALIDATION"
    receipt["completed_file_count"] = len(completed)
    receipt["completed_uncompressed_bytes"] = observed_bytes
    receipt["completed_unix_seconds"] = int(time.time())
    _atomic_json(receipt_path, receipt)
    return receipt
