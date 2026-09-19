#!/usr/bin/env python3
"""Prepare frozen M3 calibration-only tasks and resource plans.

This runner performs no network access, checkpoint loading, model inference, or
evaluation task materialization.  It consumes the exact READY_NATIVE artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quiethand.m3_adapter_contract import (  # noqa: E402
    M3AdapterContractError,
    build_calibration_task_manifest,
    build_checkpoint_plan,
)
from quiethand.taco_contract import (  # noqa: E402
    TACO_SELECTION_SHA256,
    TACO_SEQUENCE_LIST,
    TacoContractError,
    canonical_selection_bytes,
    deterministic_taco_split,
    load_frozen_sequence_ids,
)


REPOSITORY = ROOT / "external_repos" / "TACO-Instructions"
NATIVE_PATH = (
    ROOT / "artifacts" / "quiethand" / "m3" / "QH_M3_TACO_NATIVE_CONTRACT.json"
)
RECEIPT_PATH = (
    ROOT / "artifacts" / "quiethand" / "m3" / "QH_M3_TACO_LANDING_RECEIPT.json"
)
TASK_PATH = (
    ROOT / "artifacts" / "quiethand" / "m3" / "QH_M3_CALIBRATION_TASK_PLAN.json"
)
CHECKPOINT_PATH = (
    ROOT / "artifacts" / "quiethand" / "m3" / "QH_M3_CHECKPOINT_PLAN.json"
)
EXPECTED_NATIVE_SHA256 = (
    "4f3d5876882cd6aa9a738f458e6b728468afab4d52ba425be13cf088eed2c4a4"
)
EXPECTED_RECEIPT_SHA256 = (
    "cda13e41af86682901c837994248d87a461686672cda9e224e70f329599682bc"
)
EXPECTED_GEOMETRY_SHA256 = (
    "e8472184d4042aff96b72b775301955b9885ef57e360fbd5bc766aa43674a042"
)
PROTOCOL_V1_1_SHA256 = (
    "6f3216350d47b9eee2f029f3e039b15f8929c7b1b9950ce4e163811be673a4cc"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
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


def _load_exact(path: Path, expected_sha256: str) -> tuple[dict[str, object], str]:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise M3AdapterContractError(f"frozen source changed: {path.name}")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise M3AdapterContractError(f"frozen source is not an object: {path.name}")
    return value, digest


def _verify_native_code_binding(native: dict[str, object]) -> None:
    declared = native.get("code_sha256")
    paths = {
        "native_core": ROOT / "quiethand" / "taco_native.py",
        "safe_torch_pickle": ROOT / "quiethand" / "safe_torch_pickle.py",
        "mano_lbs_core": ROOT / "quiethand" / "arctic_native_validation.py",
        "runner": ROOT / "scripts" / "quiethand" / "run_m3_taco_native.py",
    }
    if not isinstance(declared, dict) or set(declared) != set(paths):
        raise M3AdapterContractError("native code binding is incomplete")
    for name, path in paths.items():
        if declared[name] != _sha256(path):
            raise M3AdapterContractError(f"native code changed after validation: {name}")


def main() -> int:
    try:
        native, native_sha = _load_exact(NATIVE_PATH, EXPECTED_NATIVE_SHA256)
        receipt, receipt_sha = _load_exact(RECEIPT_PATH, EXPECTED_RECEIPT_SHA256)
        if (
            native.get("selection_sha256") != TACO_SELECTION_SHA256
            or native.get("landing_receipt_sha256") != receipt_sha
            or native.get("native_contact_geometry_sha256")
            != EXPECTED_GEOMETRY_SHA256
            or native.get("status") != "READY_NATIVE"
            or not isinstance(native.get("protocol_v1_1"), dict)
            or native["protocol_v1_1"].get("sha256") != PROTOCOL_V1_1_SHA256
            or receipt.get("selection_sha256") != TACO_SELECTION_SHA256
            or receipt.get("status") != "READY_FOR_NATIVE_VALIDATION"
            or not isinstance(receipt.get("completed"), dict)
        ):
            raise M3AdapterContractError("native/landing provenance does not close")
        _verify_native_code_binding(native)
        entries = deterministic_taco_split(
            load_frozen_sequence_ids(REPOSITORY / TACO_SEQUENCE_LIST)
        )
        canonical_selection_bytes(entries)
        task_plan = build_calibration_task_manifest(
            entries, native, receipt["completed"]
        )
        checkpoint_plan = build_checkpoint_plan()
        provenance = {
            "native_contract_sha256": native_sha,
            "landing_receipt_sha256": receipt_sha,
            "selection_sha256": TACO_SELECTION_SHA256,
            "native_geometry_sha256": EXPECTED_GEOMETRY_SHA256,
            "adapter_contract_sha256": _sha256(
                ROOT / "quiethand" / "m3_adapter_contract.py"
            ),
            "preparation_runner_sha256": _sha256(Path(__file__).resolve()),
        }
        task_plan["protocol_v1_1_status"] = "USER_APPROVED_EFFECTIVE"
        task_plan["ready_native"] = True
        task_plan["source_provenance"] = provenance
        checkpoint_plan["protocol_v1_1_status"] = "USER_APPROVED_EFFECTIVE"
        checkpoint_plan["download_authorized_at_current_gate"] = True
        checkpoint_plan["source_provenance"] = provenance
        _atomic_json(TASK_PATH, task_plan)
        _atomic_json(CHECKPOINT_PATH, checkpoint_plan)
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        TacoContractError,
        M3AdapterContractError,
    ) as exc:
        print(f"[hold] HOLD_ENGINEERING_INCOMPLETE: {exc}", file=sys.stderr)
        return 3
    print(f"[artifact] {TASK_PATH}")
    print(f"[artifact] {CHECKPOINT_PATH}")
    print("[plan] 150 calibration tasks; 0 evaluation tasks; no model bytes or inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
