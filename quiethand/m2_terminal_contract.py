"""Primary-input certificates and externally pinned terminal seals for M2."""

from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import math
from typing import Mapping

from quiethand.arctic_archive_binding import (
    FrozenArcticBinding,
    FrozenManoBinding,
    MANO_ARCHIVE_SHA256,
    MANO_MODEL_SHA256,
)
from quiethand.arctic_contract import (
    ARCTIC_RELEASE_ID,
    ARCTIC_SOURCE_COMMIT,
    CALIBRATION_GROUPS,
    EVALUATION_GROUPS,
    M2_ASSETS,
    M2_DATA_CAP_BYTES,
    REQUIRED_SEQUENCE_SUFFIXES,
    WINDOWS_PER_GROUP_CAP,
    WORKSPACE_HARD_CAP_BYTES,
    WORKSPACE_WARNING_BYTES,
    deterministic_group_split,
    native_reference_contract,
)


class TerminalContractError(RuntimeError):
    """The verified primary inputs cannot support a terminal M2 report."""


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _immutable_resource_contract(resources: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "m2_data_cap_bytes",
        "data_usage_bytes",
        "data_usage_scope",
        "m2_manifest_sha256",
        "m2_entry_count",
        "m2_regular_file_count",
        "workspace_warning_bytes",
        "workspace_hard_cap_bytes",
        "workspace_usage_bytes",
        "workspace_manifest_sha256",
        "workspace_entry_count",
        "accounting_complete",
        "snapshot_stable",
        "accounting_errors",
        "gpu_required",
        "model_call_required",
    )
    return {key: resources.get(key) for key in keys}


def build_primary_input_certificate(
    preflight: Mapping[str, object],
    arctic_binding: FrozenArcticBinding,
    mano_binding: FrozenManoBinding,
) -> dict[str, object]:
    """Derive one certificate from verified bytes, never from terminal claims."""

    try:
        release = preflight["release"]
        field_audit = preflight["field_audit"]
        supplied_split = preflight["split_contract"]
        resources = preflight["resources"]
        if not all(
            isinstance(item, Mapping)
            for item in (release, field_audit, supplied_split, resources)
        ):
            raise TerminalContractError("preflight contract has an invalid shape")

        suffix_sets: list[set[str]] = []
        for suffix in REQUIRED_SEQUENCE_SUFFIXES:
            prefix = "raw_seqs/"
            suffix_sets.append(
                {
                    key[len(prefix) : -len(suffix)]
                    for key in arctic_binding.manifest
                    if key.startswith(prefix) and key.endswith(suffix)
                }
            )
        if not suffix_sets or not suffix_sets[0] or any(
            item != suffix_sets[0] for item in suffix_sets[1:]
        ):
            raise TerminalContractError("required archive sequence fields do not align")
        eligible_ids = sorted(suffix_sets[0])
        split_entries = [
            asdict(item) for item in deterministic_group_split(eligible_ids)
        ]
        expected_split = {
            "hash_expression": "sha256(utf8(release_id || sequence_id))",
            "calibration_groups_required": CALIBRATION_GROUPS,
            "evaluation_groups_required": EVALUATION_GROUPS,
            "windows_per_group_cap": WINDOWS_PER_GROUP_CAP,
            "calibration_groups_selected": CALIBRATION_GROUPS,
            "evaluation_groups_selected": EVALUATION_GROUPS,
            "entries": split_entries,
            "subject_object_distribution_used_for_recut": False,
        }
        expected_assets = [asdict(asset) for asset in M2_ASSETS]
        if (
            preflight.get("status") != "READY_FOR_ARRAY_AND_GEOMETRY_VALIDATION"
            or release.get("release_id") != ARCTIC_RELEASE_ID
            or release.get("source_commit") != ARCTIC_SOURCE_COMMIT
            or release.get("assets") != expected_assets
            or preflight.get("native_reference_contract")
            != native_reference_contract()
            or supplied_split != expected_split
            or field_audit.get("eligible_sequence_count") != len(eligible_ids)
            or field_audit.get("candidate_sequence_count") != len(eligible_ids)
        ):
            raise TerminalContractError("preflight disagrees with verified primary inputs")

        resource_contract = _immutable_resource_contract(resources)
        data_usage = resource_contract["data_usage_bytes"]
        workspace_usage = resource_contract["workspace_usage_bytes"]
        if (
            resource_contract["m2_data_cap_bytes"] != M2_DATA_CAP_BYTES
            or resource_contract["workspace_warning_bytes"] != WORKSPACE_WARNING_BYTES
            or resource_contract["workspace_hard_cap_bytes"] != WORKSPACE_HARD_CAP_BYTES
            or resource_contract["accounting_complete"] is not True
            or resource_contract["snapshot_stable"] is not True
            or resource_contract["accounting_errors"] != []
            or not isinstance(data_usage, int)
            or data_usage > M2_DATA_CAP_BYTES
            or not isinstance(workspace_usage, int)
            or workspace_usage >= WORKSPACE_WARNING_BYTES
        ):
            raise TerminalContractError("resource contract is not terminal-ready")

        arctic_proof = copy.deepcopy(dict(arctic_binding.proof))
        mano_proof = copy.deepcopy(dict(mano_binding.proof))
        arctic_hashes = {
            str(item["name"]): str(item["archive_sha256"])
            for item in arctic_proof["assets"]
        }
        mano_hashes = {
            side: str(mano_proof["models"][side]["sha256"])
            for side in ("left", "right")
        }
        if (
            arctic_proof.get("closed") is not True
            or mano_proof.get("closed") is not True
            or arctic_hashes != {asset.name: asset.sha256 for asset in M2_ASSETS}
            or mano_proof.get("archive_sha256") != MANO_ARCHIVE_SHA256
            or mano_hashes != MANO_MODEL_SHA256
        ):
            raise TerminalContractError("archive binding proof is incomplete")

        core: dict[str, object] = {
            "schema_version": 1,
            "release": {
                "release_id": ARCTIC_RELEASE_ID,
                "source_commit": ARCTIC_SOURCE_COMMIT,
                "assets": expected_assets,
            },
            "eligible_sequence_contract": {
                "count": len(eligible_ids),
                "sequence_ids": eligible_ids,
                "sequence_ids_sha256": _canonical_sha256(eligible_ids),
                "required_suffixes": list(REQUIRED_SEQUENCE_SUFFIXES),
            },
            "split_contract": expected_split,
            "native_reference_contract": native_reference_contract(),
            "exact_input_binding": {
                "arctic": arctic_proof,
                "mano": mano_proof,
                "closed": True,
            },
        }
        return {**core, "certificate_sha256": _canonical_sha256(core)}
    except TerminalContractError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise TerminalContractError("primary-input certificate construction failed") from exc


def terminal_payload_sha256(report: Mapping[str, object]) -> str:
    payload = {key: value for key, value in report.items() if key != "terminal_seal_sha256"}
    return _canonical_sha256(payload)


def _common_terminal_closed(
    report: Mapping[str, object],
    *,
    expected_certificate: Mapping[str, object],
    expected_seal: str,
    expected_status: str,
    expected_code_hashes: Mapping[str, str],
) -> bool:
    try:
        attestation = report["license_attestation"]
        release = report["release"]
        runtime = report["runtime"]
        return bool(
            report.get("schema_version") == 2
            and report.get("status") == expected_status
            and report.get("primary_input_certificate") == expected_certificate
            and report.get("terminal_seal_sha256") == expected_seal
            and terminal_payload_sha256(report) == expected_seal
            and report.get("exact_input_binding")
            == expected_certificate["exact_input_binding"]
            and report.get("frozen_split_contract")
            == expected_certificate["split_contract"]
            and report.get("native_reference_contract")
            == expected_certificate["native_reference_contract"]
            and release.get("release_id") == ARCTIC_RELEASE_ID
            and release.get("source_commit") == ARCTIC_SOURCE_COMMIT
            and attestation.get("user_attested_arctic") is True
            and attestation.get("user_attested_mano") is True
            and attestation.get("secret_values_recorded") is False
            and runtime.get("validated_code_sha256") == expected_code_hashes
        )
    except (KeyError, TypeError, ValueError):
        return False


def native_terminal_report_closed(
    report: Mapping[str, object],
    *,
    expected_certificate: Mapping[str, object],
    expected_seal: str,
    expected_code_hashes: Mapping[str, str],
) -> bool:
    if not _common_terminal_closed(
        report,
        expected_certificate=expected_certificate,
        expected_seal=expected_seal,
        expected_status="PASS_ARRAY_AND_NATIVE_GEOMETRY_VALIDATION",
        expected_code_hashes=expected_code_hashes,
    ):
        return False
    try:
        eligible_count = expected_certificate["eligible_sequence_contract"]["count"]
        array_audit = report["array_audit"]
        geometry = report["geometry_sentinel_audit"]
        mano_audit = report["mano_model_audit"]
        return bool(
            array_audit.get("attempted_sequence_count") == eligible_count
            and array_audit.get("passed_sequence_count") == eligible_count
            and array_audit.get("failure_count") == 0
            and array_audit.get("failures_capped_at_100") == []
            and geometry.get("closed") is True
            and {
                side: mano_audit[side]["sha256"] for side in ("left", "right")
            }
            == MANO_MODEL_SHA256
        )
    except (KeyError, TypeError, ValueError):
        return False


def oracle_terminal_report_closed(
    report: Mapping[str, object],
    *,
    expected_certificate: Mapping[str, object],
    expected_seal: str,
    expected_code_hashes: Mapping[str, str],
    maximum_error_m: float,
) -> bool:
    if not _common_terminal_closed(
        report,
        expected_certificate=expected_certificate,
        expected_seal=expected_seal,
        expected_status="PASS_SMPLX_NUMERICAL_ORACLE",
        expected_code_hashes=expected_code_hashes,
    ):
        return False
    try:
        comparisons = report["comparisons"]
        observed = [float(item["maximum_absolute_error_m"]) for item in comparisons]
        rms = [float(item["root_mean_square_error_m"]) for item in comparisons]
        selected = report["input"]["selected_sequences"]
        entries = expected_certificate["split_contract"]["entries"]
        expected_selected = [entries[0]["sequence_id"], entries[20]["sequence_id"]]
        maximum = float(report["maximum_observed_vertex_absolute_error_m"])
        return bool(
            report.get("threshold")
            == {"maximum_vertex_absolute_error_m": maximum_error_m, "comparator": "<="}
            and report.get("comparison_count") == 12
            and len(comparisons) == 12
            and all(math.isfinite(value) for value in observed + rms + [maximum])
            and maximum == max(observed)
            and maximum <= maximum_error_m
            and selected == expected_selected
            and report["input"]["mano_model_sha256"] == MANO_MODEL_SHA256
        )
    except (KeyError, TypeError, ValueError, IndexError):
        return False
