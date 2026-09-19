#!/usr/bin/env python3
"""Land the exact frozen M3 repositories/checkpoints and close their manifest."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import time
from typing import Mapping

import gdown
import requests


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quiethand.m3_adapter_contract import (  # noqa: E402
    M3AdapterContractError,
    M3_CHECKPOINT_CAP_BYTES,
    verify_landed_checkpoint_manifest,
)


PLAN_PATH = ROOT / "artifacts" / "quiethand" / "m3" / "QH_M3_CHECKPOINT_PLAN.json"
PLAN_SHA256 = "f00a048c1b81109959e1279d96eef808af8e06938a679c67dd12985a10b0d703"
RESOURCE_ROOT = ROOT / "external_data" / "quiethand_m3_resources"
MANIFEST_PATH = (
    ROOT / "artifacts" / "quiethand" / "m3" / "QH_M3_CHECKPOINT_LANDING.json"
)
PARTIAL_MANIFEST_PATH = (
    ROOT / "artifacts" / "quiethand" / "m3" / "QH_M3_CHECKPOINT_LANDING_PARTIAL.json"
)
CHUNK_BYTES = 8 * 1024 * 1024
FOUNDATIONPOSE_FILE_IDS = {
    "2023-10-28-18-33-37/config.yml": "1477-st1s1TxXN6oqfM5ZnsQwd8BCzVg1",
    "2023-10-28-18-33-37/model_best.pth": "1E9FPB5WFIBMLrOJqZLpoVOK4Mjzrrxhv",
    "2024-01-11-20-02-45/config.yml": "1kQkQG-q_VvLRozv30hyeLB7P_jiEEqiE",
    "2024-01-11-20-02-45/model_best.pth": "1Zdjnkn4EHOI5_k08apofwRgTjWpai4E4",
}
FOUNDATIONPOSE_IDENTITIES = {
    "2023-10-28-18-33-37/config.yml": (
        708,
        "28a6ba94a33230ee5fc3c51939486281578b0972542bd9e38ca6123e75605686",
    ),
    "2023-10-28-18-33-37/model_best.pth": (
        68_220_109,
        "774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60",
    ),
    "2024-01-11-20-02-45/config.yml": (
        778,
        "a79db4de3b95885dd5ae86833b37b8698a75dad81e87d1086cd50b2fcd8dda3f",
    ),
    "2024-01-11-20-02-45/model_best.pth": (
        190_229_389,
        "81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26",
    ),
}


class M3CheckpointLandingError(RuntimeError):
    """The exact frozen resource set could not be landed safely."""


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--foundationpose-cache-root",
        type=Path,
        help="local staging root containing the exact two cached weight paths",
    )
    parser.add_argument(
        "--foundationpose-cache-provenance",
        help="original server cache root recorded in the landing artifact",
    )
    args = parser.parse_args()
    if (args.foundationpose_cache_root is None) != (
        args.foundationpose_cache_provenance is None
    ):
        parser.error("cache root and cache provenance must be supplied together")
    if args.foundationpose_cache_provenance is not None:
        provenance = PurePosixPath(args.foundationpose_cache_provenance)
        if (
            not provenance.is_absolute()
            or ".." in provenance.parts
            or "\n" in args.foundationpose_cache_provenance
        ):
            parser.error("cache provenance must be one absolute server path")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
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


def _safe_destination(relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise M3CheckpointLandingError("resource path is not safe and relative")
    destination = RESOURCE_ROOT.joinpath(*pure.parts)
    current = RESOURCE_ROOT
    for part in pure.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise M3CheckpointLandingError("resource parent contains a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise M3CheckpointLandingError("resource destination is a symlink")
    return destination


def _validate_existing(
    destination: Path, expected_size: int | None, expected_sha256: str | None
) -> tuple[int, str] | None:
    if not destination.exists():
        return None
    if destination.is_symlink() or not destination.is_file():
        raise M3CheckpointLandingError("existing resource is not a regular file")
    size = destination.stat().st_size
    if expected_size is not None and size != expected_size:
        raise M3CheckpointLandingError(f"existing byte count changed: {destination}")
    digest = _sha256(destination)
    if expected_sha256 is not None and digest != expected_sha256:
        raise M3CheckpointLandingError(f"existing SHA-256 changed: {destination}")
    return size, digest


def _download_http(
    source_url: str,
    relative_path: str,
    *,
    expected_size: int | None,
    expected_sha256: str | None,
) -> tuple[int, str]:
    destination = _safe_destination(relative_path)
    existing = _validate_existing(destination, expected_size, expected_sha256)
    if existing is not None:
        print(f"[reuse] {relative_path} ({existing[0]} bytes)", flush=True)
        return existing
    partial = destination.with_name(f".{destination.name}.partial")
    for attempt in range(1, 7):
        current = partial.stat().st_size if partial.exists() else 0
        if expected_size is not None and current > expected_size:
            raise M3CheckpointLandingError(f"partial file exceeds expected size: {relative_path}")
        headers = {"Range": f"bytes={current}-"} if current else {}
        try:
            with requests.get(
                source_url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=(30, 120),
            ) as response:
                if current:
                    content_range = response.headers.get("Content-Range", "")
                    if response.status_code != 206 or not content_range.startswith(
                        f"bytes {current}-"
                    ):
                        raise M3CheckpointLandingError(
                            f"server refused exact resume for {relative_path}"
                        )
                elif response.status_code != 200:
                    raise M3CheckpointLandingError(
                        f"HTTP {response.status_code} for {relative_path}"
                    )
                print(
                    f"[download] {relative_path} from byte {current} (attempt {attempt})",
                    flush=True,
                )
                with partial.open("ab" if current else "wb") as handle:
                    for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        current += len(chunk)
                        if expected_size is not None and current > expected_size:
                            raise M3CheckpointLandingError(
                                f"stream exceeds expected size: {relative_path}"
                            )
                    handle.flush()
                    os.fsync(handle.fileno())
        except (requests.RequestException, OSError) as exc:
            if attempt == 6:
                raise M3CheckpointLandingError(
                    f"download failed after retries: {relative_path}: {exc}"
                ) from exc
            time.sleep(min(2**attempt, 20))
            continue
        if expected_size is not None and current != expected_size:
            if attempt == 6:
                raise M3CheckpointLandingError(
                    f"download ended at {current}, expected {expected_size}: {relative_path}"
                )
            time.sleep(min(2**attempt, 20))
            continue
        digest = _sha256(partial)
        if expected_sha256 is not None and digest != expected_sha256:
            raise M3CheckpointLandingError(f"downloaded SHA-256 differs: {relative_path}")
        os.replace(partial, destination)
        print(f"[closed] {relative_path} ({current} bytes)", flush=True)
        return current, digest
    raise AssertionError("unreachable")


def _cache_source(cache_root: Path, relative_asset: str) -> Path:
    if cache_root.is_symlink() or not cache_root.is_dir():
        raise M3CheckpointLandingError("FoundationPose cache root is missing or symlinked")
    source = cache_root
    for part in PurePosixPath(relative_asset).parts:
        source = source / part
        if source.is_symlink():
            raise M3CheckpointLandingError("FoundationPose cache path contains a symlink")
    if not source.is_file():
        raise M3CheckpointLandingError(
            f"FoundationPose cache asset is missing: {relative_asset}"
        )
    return source


def _import_foundationpose_cache(
    cache_root: Path, relative_asset: str, destination: Path
) -> tuple[int, str]:
    source = _cache_source(cache_root, relative_asset)
    expected_size, expected_sha = FOUNDATIONPOSE_IDENTITIES[relative_asset]
    observed = _validate_existing(source, expected_size, expected_sha)
    if observed is None:
        raise AssertionError("validated cache source unexpectedly vanished")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".cache-import", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_handle, os.fdopen(
            descriptor, "wb"
        ) as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=CHUNK_BYTES)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        imported = _validate_existing(temporary, expected_size, expected_sha)
        if imported is None:
            raise AssertionError("cache import temporary unexpectedly vanished")
        os.replace(temporary, destination)
        print(f"[cache-import] object_state/{relative_asset} ({imported[0]} bytes)", flush=True)
        return imported
    finally:
        if temporary.exists():
            temporary.unlink()


def _download_foundationpose(
    relative_asset: str, cache_root: Path | None
) -> tuple[int, str, str]:
    file_id = FOUNDATIONPOSE_FILE_IDS.get(relative_asset)
    identity = FOUNDATIONPOSE_IDENTITIES.get(relative_asset)
    if file_id is None or identity is None:
        raise M3CheckpointLandingError("FoundationPose Drive file set changed")
    expected_size, expected_sha = identity
    relative_path = f"object_state/{relative_asset}"
    destination = _safe_destination(relative_path)
    existing = _validate_existing(destination, expected_size, expected_sha)
    if existing is not None:
        if cache_root is not None and relative_asset.endswith("model_best.pth"):
            source = _cache_source(cache_root, relative_asset)
            source_identity = _validate_existing(source, expected_size, expected_sha)
            if source_identity != existing:
                raise M3CheckpointLandingError(
                    f"FoundationPose cache identity differs: {relative_asset}"
                )
            print(f"[cache-bound-reuse] {relative_path} ({existing[0]} bytes)", flush=True)
            return (*existing, "verified_server_cache_import")
        print(f"[reuse] {relative_path} ({existing[0]} bytes)", flush=True)
        return (*existing, "existing_exact_canonical")
    if cache_root is not None and relative_asset.endswith("model_best.pth"):
        imported = _import_foundationpose_cache(cache_root, relative_asset, destination)
        return (*imported, "verified_server_cache_import")
    if relative_asset.endswith("config.yml"):
        direct_url = (
            "https://drive.usercontent.google.com/download?id="
            + file_id
            + "&export=download&confirm=t"
        )
        result = _download_http(
            direct_url,
            relative_path,
            expected_size=expected_size,
            expected_sha256=expected_sha,
        )
        return (*result, "official_drive_download")
    staged = destination.with_name(f".{destination.name}.gdown")
    print(f"[download] {relative_path} from official Drive file {file_id}", flush=True)
    result = gdown.download(
        id=file_id,
        output=str(staged),
        quiet=False,
        use_cookies=False,
        resume=True,
    )
    if result is None or not staged.is_file() or staged.stat().st_size <= 0:
        raise M3CheckpointLandingError(
            f"official FoundationPose Drive access failed: {relative_asset}"
        )
    size = staged.stat().st_size
    digest = _sha256(staged)
    if (size, digest) != (expected_size, expected_sha):
        raise M3CheckpointLandingError(
            f"official FoundationPose identity differs: {relative_asset}"
        )
    os.replace(staged, destination)
    print(f"[closed] {relative_path} ({size} bytes)", flush=True)
    return size, digest, "official_drive_download"


def _row(
    adapter: str,
    asset_class: str,
    relative_path: str,
    source_url: str,
    revision: str,
    size: int,
    digest: str,
) -> dict[str, object]:
    return {
        "adapter": adapter,
        "asset_class": asset_class,
        "relative_path": relative_path,
        "source_url": source_url,
        "resolved_revision": revision,
        "bytes": size,
        "sha256": digest,
    }


def main() -> int:
    try:
        args = _arguments()
        if _sha256(PLAN_PATH) != PLAN_SHA256:
            raise M3CheckpointLandingError("frozen checkpoint plan changed")
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        if (
            plan.get("download_authorized_at_current_gate") is not True
            or plan.get("protocol_v1_1_status") != "USER_APPROVED_EFFECTIVE"
            or plan.get("checkpoint_and_repo_cap_bytes") != M3_CHECKPOINT_CAP_BYTES
        ):
            raise M3CheckpointLandingError("checkpoint plan is not authorized/effective")
        RESOURCE_ROOT.mkdir(parents=True, exist_ok=True)
        if RESOURCE_ROOT.is_symlink():
            raise M3CheckpointLandingError("resource root is symlinked")
        free_bytes = shutil.disk_usage(RESOURCE_ROOT).free
        if free_bytes < M3_CHECKPOINT_CAP_BYTES:
            raise M3CheckpointLandingError("less than 30 GiB free before landing")

        rows: list[dict[str, object]] = []
        acquisition_rows: list[dict[str, object]] = []
        hf_jobs = []
        adapters = plan["adapters"]
        for adapter, profile in adapters.items():
            if profile["checkpoint_provider"] != "huggingface":
                continue
            revision = profile["resolved_revision"]
            for asset in profile["assets"]:
                relative_path = f"{adapter}/{asset['path']}"
                source_url = (
                    f"https://huggingface.co/{profile['checkpoint_id']}/resolve/"
                    f"{revision}/{asset['path']}"
                )
                hf_jobs.append((adapter, revision, source_url, relative_path, asset))
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(
                    _download_http,
                    source_url,
                    relative_path,
                    expected_size=asset["bytes"],
                    expected_sha256=asset.get("expected_sha256"),
                ): (adapter, revision, source_url, relative_path)
                for adapter, revision, source_url, relative_path, asset in hf_jobs
            }
            for future in as_completed(futures):
                adapter, revision, source_url, relative_path = futures[future]
                size, digest = future.result()
                rows.append(
                    _row(
                        adapter,
                        "checkpoint",
                        relative_path,
                        source_url,
                        revision,
                        size,
                        digest,
                    )
                )

        for adapter, source in plan["repository_archives"].items():
            size, digest = _download_http(
                source["source_url"],
                source["relative_path"],
                expected_size=None,
                expected_sha256=None,
            )
            rows.append(
                _row(
                    adapter,
                    "repository_archive",
                    source["relative_path"],
                    source["source_url"],
                    source["resolved_revision"],
                    size,
                    digest,
                )
            )

        object_profile = adapters["object_state"]
        drive_source = (
            "https://drive.google.com/drive/folders/"
            + object_profile["checkpoint_id"]
        )
        required_assets = object_profile["required_assets"]
        ordered_assets = [asset for asset in required_assets if asset.endswith("config.yml")] + [
            asset for asset in required_assets if not asset.endswith("config.yml")
        ]
        if sorted(ordered_assets) != sorted(required_assets):
            raise M3CheckpointLandingError("FoundationPose asset ordering changed")
        for relative_asset in ordered_assets:
            size, digest, acquisition_method = _download_foundationpose(
                relative_asset, args.foundationpose_cache_root
            )
            rows.append(
                _row(
                    "object_state",
                    "checkpoint",
                    f"object_state/{relative_asset}",
                    drive_source,
                    object_profile["resolved_revision"],
                    size,
                    digest,
                )
            )
            if relative_asset.endswith("model_best.pth"):
                acquisition_rows.append(
                    {
                        "relative_path": f"object_state/{relative_asset}",
                        "method": acquisition_method,
                        "source_cache_path": (
                            f"{args.foundationpose_cache_provenance.rstrip('/')}/{relative_asset}"
                            if acquisition_method == "verified_server_cache_import"
                            else None
                        ),
                        "bytes": size,
                        "sha256": digest,
                    }
                )

        manifest = {
            "schema": "quiethand.m3.checkpoint_landing.v1",
            "files": sorted(rows, key=lambda row: (row["adapter"], row["relative_path"])),
            "foundationpose_acquisition": acquisition_rows,
        }
        strict_manifest = {
            "schema": manifest["schema"],
            "files": manifest["files"],
        }
        closure = verify_landed_checkpoint_manifest(strict_manifest, RESOURCE_ROOT)
        manifest["closure"] = closure
        manifest["checkpoint_plan_sha256"] = PLAN_SHA256
        manifest["scientific_result"] = False
        verify_landed_checkpoint_manifest(strict_manifest, RESOURCE_ROOT)
        _atomic_json(MANIFEST_PATH, manifest)
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        requests.RequestException,
        gdown.exceptions.FileURLRetrievalError,
        gdown.exceptions.FolderContentsMaximumLimitError,
        M3AdapterContractError,
        M3CheckpointLandingError,
    ) as exc:
        completed_rows = sorted(
            locals().get("rows", []),
            key=lambda row: (row["adapter"], row["relative_path"]),
        )
        expected_paths = set()
        loaded_plan = locals().get("plan")
        if isinstance(loaded_plan, dict):
            for adapter, profile in loaded_plan.get("adapters", {}).items():
                assets = profile.get("assets", profile.get("required_assets", []))
                for asset in assets:
                    relative_asset = asset["path"] if isinstance(asset, dict) else asset
                    expected_paths.add(f"{adapter}/{relative_asset}")
            for source in loaded_plan.get("repository_archives", {}).values():
                expected_paths.add(source["relative_path"])
        completed_paths = {row["relative_path"] for row in completed_rows}
        partial = {
            "schema": "quiethand.m3.checkpoint_landing_partial.v1",
            "status": "HOLD_MODEL_ACCESS",
            "checkpoint_plan_sha256": PLAN_SHA256,
            "completed_file_count": len(completed_rows),
            "completed_bytes": sum(row["bytes"] for row in completed_rows),
            "completed_files": completed_rows,
            "missing_relative_paths": sorted(expected_paths - completed_paths),
            "hold_reason": f"{type(exc).__name__}: {exc}",
            "scientific_result": False,
            "inference_started": False,
        }
        _atomic_json(PARTIAL_MANIFEST_PATH, partial)
        print(f"[artifact] {PARTIAL_MANIFEST_PATH}", file=sys.stderr, flush=True)
        print(f"[hold] HOLD_MODEL_ACCESS: {exc}", file=sys.stderr, flush=True)
        return 3
    print(f"[artifact] {MANIFEST_PATH}")
    print(f"[status] READY_FOR_ADAPTER_PREFLIGHT ({closure['total_bytes']} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
