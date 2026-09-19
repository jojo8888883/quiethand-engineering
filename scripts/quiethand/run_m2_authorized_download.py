#!/usr/bin/env python3
"""Credential-safe, minimal ARCTIC/MANO downloader for QuietHand M2.

Credentials are prompted interactively, kept in memory for this process only,
and never written to files, command arguments, reports, or logs.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import time
from typing import BinaryIO
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quiethand.arctic_contract import (  # noqa: E402
    ARCTIC_RELEASE_ID,
    ARCTIC_SOURCE_COMMIT,
    M2_DATA_CAP_BYTES,
    M2_ASSETS,
    ResourceAccountingError,
    build_m2_preflight,
    build_resource_snapshot,
    sha256_exact_file,
    verify_resource_snapshot,
)
from quiethand.arctic_archive_binding import (  # noqa: E402
    MANO_ARCHIVE_SHA256,
    MANO_ARCHIVE_SIZE,
    MANO_MODEL_SIZE,
)
from quiethand.arctic_native_validation import MAX_MANO_PICKLE_BYTES  # noqa: E402


ARCTIC_ROOT = ROOT / "external_data" / "arctic_v1_0"
MANO_ROOT = ROOT / "external_data" / "mano_v1_2"
MANO_URL = (
    "https://download.is.tue.mpg.de/download.php?"
    "domain=mano&resume=1&sfile=mano_v1_2.zip"
)
MANO_ARCHIVE = MANO_ROOT / "mano_v1_2.zip"
CHUNK_BYTES = 1024 * 1024
PROGRESS_BYTES = 16 * 1024 * 1024
MAX_ZIP_MEMBER_COUNT = 5_000
MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES = 4 * 1024**3
MAX_ZIP_MEMBER_UNCOMPRESSED_BYTES = 512 * 1024**2
MAX_ZIP_COMPRESSION_RATIO = 200.0


class DownloadError(RuntimeError):
    """A sanitized download or archive-validation failure."""


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _move_aside(path: Path, label: str) -> Path:
    destination = path.with_name(f"{path.name}.{label}.{_timestamp()}")
    suffix = 1
    while destination.exists():
        destination = path.with_name(
            f"{path.name}.{label}.{_timestamp()}.{suffix}"
        )
        suffix += 1
    path.rename(destination)
    return destination


def _post_download(
    *,
    label: str,
    url: str,
    destination: Path,
    username: str,
    password: str,
    expected_sha256: str,
    expected_size: int,
    max_bytes: int,
) -> dict[str, object]:
    if max_bytes <= 0 or expected_size > max_bytes:
        raise DownloadError(f"{label}: no M2 download budget remains")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        try:
            current_hash = sha256_exact_file(
                destination,
                root=destination.parent,
                expected_size=expected_size,
            )
        except ResourceAccountingError:
            current_hash = None
        if current_hash == expected_sha256:
            print(f"[skip] {label}: existing archive hash is valid", flush=True)
            return {
                "label": label,
                "path": destination.as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": current_hash,
                "reused": True,
            }
        moved = _move_aside(destination, "invalid")
        print(f"[safe] moved invalid prior archive to {moved.name}", flush=True)

    partial = destination.with_name(f"{destination.name}.part")
    if partial.exists():
        moved = _move_aside(partial, "stale")
        print(f"[safe] moved stale partial file to {moved.name}", flush=True)

    form = urlencode({"username": username, "password": password}).encode("utf-8")
    request = Request(
        url,
        data=form,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "QuietHand-M2/1.0",
        },
    )
    print(f"[download] {label}", flush=True)
    digest = hashlib.sha256()
    downloaded = 0
    next_progress = PROGRESS_BYTES
    try:
        with urlopen(request, timeout=300) as response:  # noqa: S310 - frozen HTTPS URLs
            content_type = response.headers.get_content_type().lower()
            if content_type in {"text/html", "application/xhtml+xml"}:
                raise DownloadError(
                    f"{label}: server returned a web/login page; verify this account's credentials and license access"
                )
            declared = response.headers.get("Content-Length")
            expected_bytes = int(declared) if declared and declared.isdigit() else None
            if expected_bytes is not None and expected_bytes != expected_size:
                raise DownloadError(
                    f"{label}: declared response differs from the pinned archive size"
                )
            if expected_bytes is not None and expected_bytes > max_bytes:
                raise DownloadError(
                    f"{label}: declared response exceeds remaining M2 byte budget"
                )
            over_budget = False
            with partial.open("xb") as handle:
                while chunk := response.read(CHUNK_BYTES):
                    if downloaded + len(chunk) > expected_size:
                        over_budget = True
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    downloaded += len(chunk)
                    if downloaded >= next_progress:
                        if expected_bytes:
                            print(
                                f"  {downloaded / 1024**2:.1f} / {expected_bytes / 1024**2:.1f} MiB",
                                flush=True,
                            )
                        else:
                            print(f"  {downloaded / 1024**2:.1f} MiB", flush=True)
                        next_progress += PROGRESS_BYTES
    except HTTPError as exc:
        raise DownloadError(f"{label}: server returned HTTP {exc.code}") from None
    except URLError as exc:
        reason = type(exc.reason).__name__
        raise DownloadError(f"{label}: network/TLS connection failed ({reason})") from None
    except FileExistsError:
        raise DownloadError(f"{label}: partial-file collision; retry after inspection") from None

    if over_budget:
        moved = _move_aside(partial, "over_budget")
        raise DownloadError(
            f"{label}: streamed response exceeded remaining M2 byte budget; retained rejected partial as {moved.name}"
        )

    if downloaded != expected_size:
        moved = _move_aside(partial, "length_mismatch")
        raise DownloadError(
            f"{label}: response length mismatch; retained the rejected payload as {moved.name}"
        )
    if downloaded == 0:
        raise DownloadError(f"{label}: server returned an empty response")
    computed = digest.hexdigest()
    if computed != expected_sha256:
        moved = _move_aside(partial, "hash_mismatch")
        raise DownloadError(
            f"{label}: SHA-256 mismatch; retained the rejected payload as {moved.name}"
        )
    os.replace(partial, destination)
    print(f"[ok] {label}: {downloaded / 1024**2:.1f} MiB", flush=True)
    return {
        "label": label,
        "path": destination.as_posix(),
        "bytes": downloaded,
        "sha256": computed,
        "reused": False,
    }


def _safe_member(info: zipfile.ZipInfo) -> bool:
    name = PurePosixPath(info.filename)
    if name.is_absolute() or ".." in name.parts:
        return False
    mode = info.external_attr >> 16
    return stat.S_IFMT(mode) != stat.S_IFLNK


def _audit_zip_budget(
    infos: list[zipfile.ZipInfo], *, label: str
) -> dict[str, int | float]:
    files = [info for info in infos if not info.is_dir()]
    if not files or len(infos) > MAX_ZIP_MEMBER_COUNT:
        raise DownloadError(f"{label}: ZIP member-count contract violated")
    total = 0
    maximum_ratio = 0.0
    for info in files:
        if info.file_size < 0 or info.compress_size < 0:
            raise DownloadError(f"{label}: negative ZIP size metadata")
        if info.file_size > MAX_ZIP_MEMBER_UNCOMPRESSED_BYTES:
            raise DownloadError(f"{label}: ZIP member exceeds expanded-size cap")
        ratio = info.file_size / max(info.compress_size, 1)
        if ratio > MAX_ZIP_COMPRESSION_RATIO:
            raise DownloadError(f"{label}: ZIP compression-ratio cap exceeded")
        total += info.file_size
        maximum_ratio = max(maximum_ratio, ratio)
        if total > MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES:
            raise DownloadError(f"{label}: ZIP total expanded-size cap exceeded")
    return {
        "member_count": len(infos),
        "file_count": len(files),
        "total_uncompressed_bytes": total,
        "maximum_compression_ratio": maximum_ratio,
    }


def _remaining_m2_bytes() -> int:
    try:
        snapshot = build_resource_snapshot(ARCTIC_ROOT, MANO_ROOT, ROOT)
        verify_resource_snapshot(snapshot)
        used = snapshot.m2_usage_bytes
    except ResourceAccountingError as exc:
        raise DownloadError("M2 resource accounting is incomplete") from exc
    return max(0, M2_DATA_CAP_BYTES - used)


def _enforce_extraction_budget(required_bytes: int, *, label: str) -> None:
    if required_bytes < 0 or required_bytes > _remaining_m2_bytes():
        raise DownloadError(f"{label}: extraction would exceed the M2 data cap")


def _copy_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.part")
    if temporary.exists():
        _move_aside(temporary, "stale")
    with archive.open(info, "r") as source, temporary.open("xb") as sink:
        shutil.copyfileobj(source, sink, length=CHUNK_BYTES)
    os.replace(temporary, target)


def _extract_arctic_archive(archive_path: Path, expected_root: str) -> Path:
    staging_parent = ARCTIC_ROOT / ".extract"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = staging_parent / expected_root
    if staging.exists():
        moved = _move_aside(staging, "stale")
        print(f"[safe] moved stale extraction to {moved.name}", flush=True)
    staging.mkdir()

    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if not infos or any(not _safe_member(info) for info in infos):
                raise DownloadError(f"{archive_path.name}: unsafe or empty ZIP archive")
            budget = _audit_zip_budget(infos, label=archive_path.name)
            _enforce_extraction_budget(
                int(budget["total_uncompressed_bytes"]), label=archive_path.name
            )
            for info in infos:
                if info.is_dir():
                    continue
                relative = PurePosixPath(info.filename)
                target = staging.joinpath(*relative.parts)
                _copy_member(archive, info, target)
    except zipfile.BadZipFile:
        raise DownloadError(f"{archive_path.name}: invalid ZIP archive") from None

    roots = [path for path in staging.rglob(expected_root) if path.is_dir()]
    if staging.name == expected_root and any(staging.iterdir()):
        direct_entries = list(staging.iterdir())
        if not roots and any(entry.name != expected_root for entry in direct_entries):
            roots = [staging]
    if len(roots) != 1:
        raise DownloadError(
            f"{archive_path.name}: expected exactly one {expected_root}/ directory, found {len(roots)}"
        )

    source_root = roots[0]
    target_root = ARCTIC_ROOT / "data" / expected_root
    target_root.parent.mkdir(parents=True, exist_ok=True)
    if target_root.exists():
        moved = _move_aside(target_root, "previous")
        print(f"[safe] moved prior {expected_root} directory to {moved.name}", flush=True)
    if source_root == staging:
        os.replace(staging, target_root)
    else:
        os.replace(source_root, target_root)
        _move_aside(staging, "container")
    print(f"[ok] extracted {archive_path.name}", flush=True)
    return target_root


def _extract_mano_models(archive_path: Path) -> list[Path]:
    wanted = {"MANO_LEFT.pkl", "MANO_RIGHT.pkl"}
    staging_parent = MANO_ROOT / ".extract"
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = staging_parent / "models"
    if staging.exists():
        moved = _move_aside(staging, "stale")
        print(f"[safe] moved stale MANO extraction to {moved.name}", flush=True)
    staging.mkdir()
    target_root = MANO_ROOT / "models"
    target_root.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            if not infos or any(not _safe_member(info) for info in infos):
                raise DownloadError("MANO: unsafe or empty ZIP archive")
            _audit_zip_budget(infos, label="MANO")
            if archive.testzip() is not None:
                raise DownloadError("MANO: ZIP CRC validation failed")
            selected: dict[str, zipfile.ZipInfo] = {}
            for info in infos:
                basename = PurePosixPath(info.filename).name
                if basename in wanted and not info.is_dir():
                    if basename in selected:
                        raise DownloadError(f"MANO: duplicate {basename} in archive")
                    selected[basename] = info
            missing = sorted(wanted - selected.keys())
            if missing:
                raise DownloadError(f"MANO: archive is missing {', '.join(missing)}")
            selected_bytes = sum(info.file_size for info in selected.values())
            if any(info.file_size > MAX_MANO_PICKLE_BYTES for info in selected.values()):
                raise DownloadError("MANO: selected model exceeds pickle-size cap")
            _enforce_extraction_budget(selected_bytes, label="MANO")
            for basename in sorted(wanted):
                staged = staging / basename
                _copy_member(archive, selected[basename], staged)
                if staged.stat().st_size == 0:
                    raise DownloadError(f"MANO: extracted empty {basename}")
    except zipfile.BadZipFile:
        raise DownloadError("MANO: invalid ZIP archive") from None

    backup = MANO_ROOT / ".rollback" / _timestamp()
    backup.mkdir(parents=True, exist_ok=False)
    backed_up: list[str] = []
    installed: list[str] = []
    try:
        for basename in sorted(wanted):
            target = target_root / basename
            if target.exists():
                os.replace(target, backup / basename)
                backed_up.append(basename)
        for basename in sorted(wanted):
            os.replace(staging / basename, target_root / basename)
            installed.append(basename)
    except OSError as exc:
        try:
            for basename in reversed(installed):
                target = target_root / basename
                if target.exists():
                    os.replace(target, staging / basename)
            for basename in reversed(backed_up):
                prior = backup / basename
                if prior.exists():
                    os.replace(prior, target_root / basename)
        except OSError:
            raise DownloadError(
                "MANO: transactional install failed and rollback was incomplete; inspect .extract and .rollback"
            ) from None
        raise DownloadError(
            f"MANO: transactional install failed ({type(exc).__name__}); prior pair was restored"
        ) from None

    outputs = [target_root / basename for basename in sorted(wanted)]
    print("[ok] extracted MANO left/right models", flush=True)
    return outputs


def _write_json_versioned(directory: Path, stem: str, payload: dict[str, object]) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    timestamped = directory / f"{stem}_{_timestamp()}.json"
    alias = directory / f"{stem}.json"
    serialized = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    for destination in (timestamped, alias):
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".part", dir=directory
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        finally:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    return timestamped, alias


def _enforce_pre_download_resource_gate(report: dict[str, object]) -> None:
    status = str(report.get("status"))
    blocked = {
        "HOLD_RESOURCE_ACCOUNTING_INCOMPLETE",
        "HOLD_WORKSPACE_HARD_CAP",
        "HOLD_WORKSPACE_WARNING_USER_ACTION",
        "HOLD_M2_DATA_CAP",
    }
    if status in blocked:
        raise DownloadError(f"resource preflight blocked network I/O with status {status}")


def main() -> int:
    print("QuietHand M2 authorized minimal download", flush=True)
    print("Credentials stay in this process and are never logged or saved.", flush=True)
    arctic_username = getpass.getpass("ARCTIC 邮箱（输入不显示）: ").strip()
    arctic_password = getpass.getpass("ARCTIC 密码（输入不显示）: ")
    mano_username = getpass.getpass("MANO 邮箱（输入不显示）: ").strip()
    mano_password = getpass.getpass("MANO 密码（输入不显示）: ")
    if not all((arctic_username, arctic_password, mano_username, mano_password)):
        print("[stop] all four credential fields are required", file=sys.stderr)
        return 2

    receipts: list[dict[str, object]] = []
    try:
        environment = {
            "ARCTIC_LICENSE_ACCEPTED": "yes",
            "MANO_LICENSE_ACCEPTED": "yes",
            "ARCTIC_USERNAME": arctic_username,
            "ARCTIC_PASSWORD": arctic_password,
        }
        before_download = build_m2_preflight(
            data_root=ARCTIC_ROOT,
            workspace_root=ROOT,
            mano_root=MANO_ROOT,
            env=environment,
        )
        _enforce_pre_download_resource_gate(before_download)

        for asset in M2_ASSETS:
            destination = ARCTIC_ROOT / asset.relative_path
            receipts.append(
                _post_download(
                    label=f"ARCTIC {asset.name}",
                    url=asset.url,
                    destination=destination,
                    username=arctic_username,
                    password=arctic_password,
                    expected_sha256=asset.sha256,
                    expected_size=asset.size_bytes,
                    max_bytes=_remaining_m2_bytes(),
                )
            )
        receipts.append(
            _post_download(
                label="MANO v1.2",
                url=MANO_URL,
                destination=MANO_ARCHIVE,
                username=mano_username,
                password=mano_password,
                expected_sha256=MANO_ARCHIVE_SHA256,
                expected_size=MANO_ARCHIVE_SIZE,
                max_bytes=_remaining_m2_bytes(),
            )
        )

        for asset in M2_ASSETS:
            _extract_arctic_archive(
                ARCTIC_ROOT / asset.relative_path,
                expected_root=asset.name,
            )
        mano_outputs = _extract_mano_models(MANO_ARCHIVE)

        report = build_m2_preflight(
            data_root=ARCTIC_ROOT,
            workspace_root=ROOT,
            mano_root=MANO_ROOT,
            env=environment,
        )
        artifacts = ROOT / "artifacts" / "quiethand" / "m2"
        result_path, _ = _write_json_versioned(
            artifacts, "QH_M2_PREFLIGHT_RESULT", report
        )
        receipt = {
            "schema_version": 1,
            "milestone": "QH-E1-M2",
            "release_id": ARCTIC_RELEASE_ID,
            "source_commit": ARCTIC_SOURCE_COMMIT,
            "credential_values_recorded": False,
            "user_license_attestations": {"arctic": True, "mano": True},
            "downloads": receipts,
            "mano_models": [
                {
                    "path": path.as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_exact_file(
                        path,
                        root=path.parent,
                        expected_size=MANO_MODEL_SIZE[
                            "left" if path.name == "MANO_LEFT.pkl" else "right"
                        ],
                    ),
                }
                for path in mano_outputs
            ],
            "preflight_status": report["status"],
            "preflight_artifact": result_path.as_posix(),
        }
        receipt_path, _ = _write_json_versioned(
            artifacts, "QH_M2_DOWNLOAD_RECEIPT", receipt
        )
    except (DownloadError, OSError) as exc:
        print(f"[stop] {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        del arctic_password
        del mano_password

    if report["status"] != "READY_FOR_ARRAY_AND_GEOMETRY_VALIDATION":
        print(f"[hold] M2 preflight status: {report['status']}", file=sys.stderr, flush=True)
        print(f"[artifact] {result_path}", flush=True)
        print(f"[receipt] {receipt_path}", flush=True)
        return 3

    print(f"[done] M2 preflight status: {report['status']}", flush=True)
    print(f"[artifact] {result_path}", flush=True)
    print(f"[receipt] {receipt_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
