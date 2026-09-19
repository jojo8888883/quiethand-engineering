#!/usr/bin/env python3
"""Run the frozen QuietHand M2 ARCTIC array/native-geometry validation."""

from __future__ import annotations

import argparse
import copy
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quiethand.arctic_contract import (  # noqa: E402
    ARCTIC_RELEASE_ID,
    ARCTIC_SOURCE_COMMIT,
    build_m2_preflight,
    verify_resource_report,
)
from quiethand.arctic_archive_binding import (  # noqa: E402
    ArchiveBindingError,
    verify_frozen_arctic_extraction,
    verify_frozen_mano_extraction,
)
from quiethand.arctic_native_validation import (  # noqa: E402
    NativeValidationError,
    UNIT_COORDINATE_EVIDENCE,
    load_mano_model,
    mano_model_facts,
    summarize_array_facts,
    validate_geometry_sentinels,
    validate_sequence_arrays,
)
from quiethand.m2_terminal_contract import (  # noqa: E402
    build_primary_input_certificate,
    native_terminal_report_closed,
    terminal_payload_sha256,
)


TERMINAL_PASS = "PASS_ARRAY_AND_NATIVE_GEOMETRY_VALIDATION"


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_versioned(payload: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"QH_M2_NATIVE_VALIDATION_RESULT_{_timestamp()}"
    timestamped = output_dir / f"{stem}.json"
    suffix = 1
    while timestamped.exists():
        timestamped = output_dir / f"{stem}_{suffix}.json"
        suffix += 1
    alias = output_dir / "QH_M2_NATIVE_VALIDATION_RESULT.json"
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
            prefix=f".{destination.name}.", suffix=".part", dir=output_dir
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


def _base_report() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "milestone": "QH-E1-M2",
        "status": "HOLD_INTERNAL_INCOMPLETE",
        "engineering_result": "NO_SCIENTIFIC_RESULT",
        "release": {
            "release_id": ARCTIC_RELEASE_ID,
            "source_commit": ARCTIC_SOURCE_COMMIT,
        },
        "scope": {
            "gpu_required": False,
            "model_call_required": False,
            "training_performed": False,
            "semantic_support_role_validated": False,
            "scientific_coverage_computed": False,
        },
    }


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in ("numpy", "scipy", "smplx", "torch"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _code_hashes() -> dict[str, str]:
    paths = {
        "arctic_contract": ROOT / "quiethand" / "arctic_contract.py",
        "archive_binding": ROOT / "quiethand" / "arctic_archive_binding.py",
        "native_validation": ROOT / "quiethand" / "arctic_native_validation.py",
        "terminal_contract": ROOT / "quiethand" / "m2_terminal_contract.py",
        "native_runner": Path(__file__).resolve(),
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _terminal_provenance_closed(
    report: dict[str, Any],
    *,
    expected_certificate: dict[str, object],
    expected_seal: str,
) -> bool:
    return native_terminal_report_closed(
        report,
        expected_certificate=expected_certificate,
        expected_seal=expected_seal,
        expected_code_hashes=_code_hashes(),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "external_data" / "arctic_v1_0",
    )
    parser.add_argument(
        "--mano-root",
        type=Path,
        default=ROOT / "external_data" / "mano_v1_2",
    )
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "quiethand" / "m2",
    )
    parser.add_argument(
        "--licenses-attested",
        action="store_true",
        help="Assert that the user personally accepted both ARCTIC and MANO licenses.",
    )
    parser.add_argument(
        "--geometry-samples-per-sequence",
        type=int,
        default=3,
        choices=(1, 2, 3),
    )
    args = parser.parse_args()

    started = time.process_time()
    report = _base_report()
    primary_certificate: dict[str, object] | None = None
    report["input_paths"] = {
        "data_root": args.data_root.as_posix(),
        "mano_root": args.mano_root.as_posix(),
    }
    report["license_attestation"] = {
        "user_attested_arctic": bool(args.licenses_attested),
        "user_attested_mano": bool(args.licenses_attested),
        "secret_values_recorded": False,
    }
    environment = {
        "ARCTIC_LICENSE_ACCEPTED": "yes" if args.licenses_attested else "no",
        "MANO_LICENSE_ACCEPTED": "yes" if args.licenses_attested else "no",
    }
    try:
        preflight = build_m2_preflight(
            data_root=args.data_root,
            workspace_root=args.workspace_root,
            mano_root=args.mano_root,
            env=environment,
        )
        report["preflight"] = {
            "status": preflight["status"],
            "release": preflight["release"],
            "license_and_access": preflight["license_and_access"],
            "native_reference_contract": preflight["native_reference_contract"],
            "paths": preflight["paths"],
            "field_audit": preflight["field_audit"],
            "split_contract": preflight["split_contract"],
            "resources": preflight["resources"],
        }
        report["frozen_split_contract"] = preflight["split_contract"]
        report["native_reference_contract"] = preflight["native_reference_contract"]
        if preflight["status"] != "READY_FOR_ARRAY_AND_GEOMETRY_VALIDATION":
            report["status"] = "HOLD_INPUT_PREFLIGHT"
            report["hold_reason"] = str(preflight["status"])
        else:
            arctic_binding = verify_frozen_arctic_extraction(args.data_root)
            mano_binding = verify_frozen_mano_extraction(args.mano_root)
            primary_certificate = build_primary_input_certificate(
                preflight, arctic_binding, mano_binding
            )
            report["primary_input_certificate"] = copy.deepcopy(primary_certificate)
            report["exact_input_binding"] = copy.deepcopy(
                primary_certificate["exact_input_binding"]
            )
            sequence_ids = sorted(
                key[len("raw_seqs/") : -len(".mano.npy")]
                for key in arctic_binding.manifest
                if key.startswith("raw_seqs/") and key.endswith(".mano.npy")
            )
            eligible = int(preflight["field_audit"]["eligible_sequence_count"])
            candidate = int(preflight["field_audit"]["candidate_sequence_count"])
            if len(sequence_ids) != eligible or candidate != eligible:
                raise NativeValidationError(
                    "archive manifest and complete-sequence preflight counts disagree"
                )
            facts: list[dict[str, object]] = []
            failures: list[dict[str, str]] = []
            for sequence_id in sequence_ids:
                try:
                    facts.append(validate_sequence_arrays(arctic_binding, sequence_id))
                except NativeValidationError as exc:
                    if len(failures) < 100:
                        failures.append({"sequence_id": sequence_id, "error": str(exc)})
            report["array_audit"] = {
                "attempted_sequence_count": len(sequence_ids),
                "passed_sequence_count": len(facts),
                "failure_count": len(sequence_ids) - len(facts),
                "failures_capped_at_100": failures,
            }
            if failures or len(facts) != len(sequence_ids):
                report["status"] = "HOLD_ARRAY_CONTRACT"
            else:
                report["array_audit"].update(summarize_array_facts(facts))
                report["array_audit"]["unit_and_coordinate_evidence"] = UNIT_COORDINATE_EVIDENCE
                models = {
                    side: load_mano_model(mano_binding.read_model(side), side)
                    for side in ("left", "right")
                }
                report["mano_model_audit"] = {
                    side: {
                        **mano_model_facts(models[side]),
                        "path": (args.mano_root / "models" / f"MANO_{side.upper()}.pkl").as_posix(),
                        "bytes": len(mano_binding.read_model(side)),
                        "sha256": models[side].source_sha256,
                    }
                    for side in ("left", "right")
                }
                split_entries = preflight["split_contract"]["entries"]
                selected = [str(entry["sequence_id"]) for entry in split_entries]
                selected_set = set(selected)
                if len(selected) != 40 or len(selected_set) != 40 or not selected_set.issubset(sequence_ids):
                    report["status"] = "HOLD_SPLIT_CONTRACT"
                else:
                    report["geometry_sentinel_audit"] = validate_geometry_sentinels(
                        binding=arctic_binding,
                        models=models,
                        sequence_ids=selected,
                        samples_per_sequence=args.geometry_samples_per_sequence,
                    )
                    report["object_parts_contract"] = {
                        "official_code_behavior": "original parts==0 is articulated top",
                        "documentation_text_conflicts": True,
                        "implementation_follows_frozen_official_code": True,
                    }
                    if report["geometry_sentinel_audit"]["closed"]:
                        verify_resource_report(
                            preflight["resources"],
                            data_root=args.data_root,
                            mano_root=args.mano_root,
                            workspace_root=args.workspace_root,
                        )
                        report["status"] = TERMINAL_PASS
                    else:
                        report["status"] = "HOLD_NATIVE_GEOMETRY"
    except (Exception, MemoryError) as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        report["status"] = "HOLD_VALIDATION_EXCEPTION"
        report["hold_reason"] = f"{type(exc).__name__}: {exc}"

    report["runtime"] = {
        "cpu_seconds": time.process_time() - started,
        "dependencies": _dependency_versions(),
        "validated_code_sha256": _code_hashes(),
    }
    if report["status"] == TERMINAL_PASS:
        if primary_certificate is None:
            report["status"] = "HOLD_TERMINAL_PROVENANCE"
            report["hold_reason"] = "primary-input certificate is missing"
        else:
            terminal_seal = terminal_payload_sha256(report)
            report["terminal_seal_sha256"] = terminal_seal
            if not _terminal_provenance_closed(
                report,
                expected_certificate=primary_certificate,
                expected_seal=terminal_seal,
            ):
                report.pop("terminal_seal_sha256", None)
                report["status"] = "HOLD_TERMINAL_PROVENANCE"
                report["hold_reason"] = "externally pinned terminal contract is incomplete"
    try:
        result_path, alias_path = _write_versioned(report, args.output_dir)
    except (TypeError, ValueError) as exc:
        report = {
            **_base_report(),
            "status": "HOLD_NONFINITE_OR_UNSERIALIZABLE_ARTIFACT",
            "hold_reason": type(exc).__name__,
            "license_attestation": {
                "user_attested_arctic": bool(args.licenses_attested),
                "user_attested_mano": bool(args.licenses_attested),
                "secret_values_recorded": False,
            },
        }
        result_path, alias_path = _write_versioned(report, args.output_dir)
    print(f"[status] {report['status']}")
    print(f"[artifact] {result_path}")
    print(f"[alias] {alias_path}")
    if report["status"] != TERMINAL_PASS:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
