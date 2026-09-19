#!/usr/bin/env python3
"""Probe and freeze the selective TACO V1 landing plan without downloading members."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quiethand.taco_contract import (  # noqa: E402
    TACO_DATA_CAP_BYTES,
    TACO_SELECTION_SHA256,
    TACO_SEQUENCE_LIST,
    TACO_SOURCE_COMMIT,
    canonical_selection_bytes,
    deterministic_taco_split,
    load_frozen_sequence_ids,
)
from quiethand.taco_source import (  # noqa: E402
    SEQUENCE_ARCHIVES,
    bind_required_archives,
    catalog_object_models,
    catalog_sequence_archive,
    fetch_dropbox_listing,
    object_ids_from_pose_catalog,
    selected_member_manifest_sha256,
    validate_selected_budget,
)


REPOSITORY_ROOT = ROOT / "external_repos" / "TACO-Instructions"
ARTIFACT_ROOT = ROOT / "artifacts" / "quiethand" / "m3"
SELECTION_PATH = ARTIFACT_ROOT / "TACO_M3_SELECTION.tsv"
RESULT_PATH = ARTIFACT_ROOT / "QH_M3_TACO_PROBE.json"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _git_head(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> int:
    if _git_head(REPOSITORY_ROOT) != TACO_SOURCE_COMMIT:
        raise RuntimeError("TACO repository is not at the frozen commit")
    sequence_ids = load_frozen_sequence_ids(REPOSITORY_ROOT / TACO_SEQUENCE_LIST)
    split = deterministic_taco_split(sequence_ids)
    selection_payload = canonical_selection_bytes(split)

    listing_payload, listing = fetch_dropbox_listing()
    archives = bind_required_archives(listing)
    catalogs = []
    for name in SEQUENCE_ARCHIVES:
        print(f"[catalog] {name}", flush=True)
        catalogs.append(catalog_sequence_archive(archives[name], split))
    pose_catalog = next(
        item for item in catalogs if item["filename"] == "Object_Poses.zip"
    )
    object_ids = object_ids_from_pose_catalog(pose_catalog)
    print("[catalog] Object_Models.zip", flush=True)
    catalogs.append(catalog_object_models(archives["Object_Models.zip"], object_ids))

    selected_bytes = validate_selected_budget(catalogs)
    free_bytes = os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize
    if free_bytes <= selected_bytes:
        raise RuntimeError("local free space is smaller than the selected TACO slice")

    result = {
        "schema": "quiethand.m3.taco_probe.v1",
        "status": "READY_FOR_SELECTIVE_LANDING",
        "created_unix_seconds": int(time.time()),
        "scientific_result": False,
        "source_commit": TACO_SOURCE_COMMIT,
        "selection_sha256": TACO_SELECTION_SHA256,
        "selection_count": len(split),
        "calibration_count": 30,
        "sealed_evaluation_count": 30,
        "dropbox_listing_sha256": hashlib.sha256(listing_payload).hexdigest(),
        "dropbox_folder_name": listing.get("folder", {}).get("filename"),
        "top_level_archive_bytes": sum(int(item["bytes"]) for item in archives.values()),
        "whole_archive_plan_within_cap": False,
        "selected_uncompressed_bytes": selected_bytes,
        "selected_data_cap_bytes": TACO_DATA_CAP_BYTES,
        "free_bytes_at_probe": free_bytes,
        "selected_object_ids": list(object_ids),
        "selected_member_manifest_sha256": selected_member_manifest_sha256(catalogs),
        "catalogs": catalogs,
    }
    _atomic_write(SELECTION_PATH, selection_payload)
    _atomic_write(
        RESULT_PATH,
        (json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    print(
        f"[ready] {len(split)} clips, {selected_bytes / 1024**3:.2f} GiB selected",
        flush=True,
    )
    print(f"[artifact] {RESULT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
