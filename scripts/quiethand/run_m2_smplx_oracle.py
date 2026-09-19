#!/usr/bin/env python3
"""Compare QuietHand's NumPy MANO LBS with the independent smplx implementation."""

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from quiethand.arctic_contract import (  # noqa: E402
    ARCTIC_RELEASE_ID,
    ARCTIC_SOURCE_COMMIT,
    build_m2_preflight,
    verify_resource_report,
)
from quiethand.arctic_archive_binding import (  # noqa: E402
    MANO_MODEL_SHA256,
    verify_frozen_arctic_extraction,
    verify_frozen_mano_extraction,
)
from quiethand.arctic_native_validation import (  # noqa: E402
    NativeValidationError,
    _load_scalar_dict,
    load_mano_model,
    reconstruct_mano,
)
from quiethand.m2_terminal_contract import (  # noqa: E402
    build_primary_input_certificate,
    oracle_terminal_report_closed,
    terminal_payload_sha256,
)


MAX_VERTEX_ABS_ERROR_M = 1e-7
PASS_STATUS = "PASS_SMPLX_NUMERICAL_ORACLE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_result(output_dir: Path, payload: dict[str, object]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    timestamped = output_dir / f"QH_M2_SMPLX_ORACLE_RESULT_{stamp}.json"
    suffix = 1
    while timestamped.exists():
        timestamped = output_dir / f"QH_M2_SMPLX_ORACLE_RESULT_{stamp}_{suffix}.json"
        suffix += 1
    alias = output_dir / "QH_M2_SMPLX_ORACLE_RESULT.json"
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


def _smplx_struct(model: object) -> object:
    """Create a minimal inert struct from an already fully validated model."""

    from scipy.sparse import csc_matrix
    from smplx.utils import Struct

    parents = np.asarray(model.parents, dtype=np.int64).copy()
    parents[0] = 0
    kintree = np.vstack((parents, np.arange(16, dtype=np.int64)))
    return Struct(
        f=np.asarray(model.faces, dtype=np.int64),
        v_template=np.asarray(model.v_template, dtype=np.float64),
        shapedirs=np.asarray(model.shapedirs, dtype=np.float64),
        J_regressor=csc_matrix(np.asarray(model.joint_regressor, dtype=np.float64)),
        posedirs=np.asarray(model.posedirs, dtype=np.float64),
        kintree_table=kintree,
        weights=np.asarray(model.weights, dtype=np.float64),
        hands_components=np.asarray(model.hand_components, dtype=np.float64),
        hands_mean=np.asarray(model.hand_mean, dtype=np.float64),
    )


def _code_hashes() -> dict[str, str]:
    paths = {
        "arctic_contract": ROOT / "quiethand" / "arctic_contract.py",
        "archive_binding": ROOT / "quiethand" / "arctic_archive_binding.py",
        "native_validation": ROOT / "quiethand" / "arctic_native_validation.py",
        "terminal_contract": ROOT / "quiethand" / "m2_terminal_contract.py",
        "oracle_runner": Path(__file__).resolve(),
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _oracle_pass_contract(
    report: dict[str, object],
    *,
    expected_certificate: dict[str, object],
    expected_seal: str,
) -> bool:
    return oracle_terminal_report_closed(
        report,
        expected_certificate=expected_certificate,
        expected_seal=expected_seal,
        expected_code_hashes=_code_hashes(),
        maximum_error_m=MAX_VERTEX_ABS_ERROR_M,
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
    args = parser.parse_args()

    started = time.process_time()
    report: dict[str, object] = {
        "schema_version": 2,
        "milestone": "QH-E1-M2",
        "status": "HOLD_ORACLE_INCOMPLETE",
        "engineering_result": "NO_SCIENTIFIC_RESULT",
        "release": {
            "release_id": ARCTIC_RELEASE_ID,
            "source_commit": ARCTIC_SOURCE_COMMIT,
        },
        "license_attestation": {
            "user_attested_arctic": bool(args.licenses_attested),
            "user_attested_mano": bool(args.licenses_attested),
            "secret_values_recorded": False,
        },
        "threshold": {
            "maximum_vertex_absolute_error_m": MAX_VERTEX_ABS_ERROR_M,
            "comparator": "<=",
        },
        "scope": {
            "cpu_only": True,
            "checkpoint_used": False,
            "model_inference": False,
            "training": False,
            "shared_parser_boundary": (
                "legacy Chumpy/sparse wrappers are decoded by QuietHand, then "
                "the official smplx PyTorch LBS is evaluated independently"
            ),
        },
    }
    primary_certificate: dict[str, object] | None = None
    try:
        import scipy
        import smplx
        import torch

        report["versions"] = {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "smplx": metadata.version("smplx"),
            "torch": torch.__version__,
            "python": sys.version.split()[0],
        }
        preflight = build_m2_preflight(
            data_root=args.data_root,
            workspace_root=ROOT,
            mano_root=args.mano_root,
            env={
                "ARCTIC_LICENSE_ACCEPTED": "yes" if args.licenses_attested else "no",
                "MANO_LICENSE_ACCEPTED": "yes" if args.licenses_attested else "no",
            },
        )
        if preflight["status"] != "READY_FOR_ARRAY_AND_GEOMETRY_VALIDATION":
            raise NativeValidationError(f"preflight status {preflight['status']}")
        report["preflight"] = {
            "status": preflight["status"],
            "license_and_access": preflight["license_and_access"],
            "native_reference_contract": preflight["native_reference_contract"],
            "split_contract": preflight["split_contract"],
            "release": preflight["release"],
            "resources": preflight["resources"],
        }
        report["frozen_split_contract"] = preflight["split_contract"]
        report["native_reference_contract"] = preflight["native_reference_contract"]
        arctic_binding = verify_frozen_arctic_extraction(args.data_root)
        mano_binding = verify_frozen_mano_extraction(args.mano_root)
        primary_certificate = build_primary_input_certificate(
            preflight, arctic_binding, mano_binding
        )
        report["primary_input_certificate"] = copy.deepcopy(primary_certificate)
        report["exact_input_binding"] = copy.deepcopy(
            primary_certificate["exact_input_binding"]
        )
        entries = preflight["split_contract"]["entries"]
        selected = [str(entries[0]["sequence_id"]), str(entries[20]["sequence_id"])]
        comparisons: list[dict[str, object]] = []
        models = {
            side: load_mano_model(mano_binding.read_model(side), side)
            for side in ("left", "right")
        }
        for side in ("left", "right"):
                numpy_model = models[side]
                reference = smplx.MANO(
                    model_path="validated-inert-struct",
                    data_struct=_smplx_struct(numpy_model),
                    is_rhand=side == "right",
                    use_pca=False,
                    flat_hand_mean=False,
                    create_transl=False,
                    dtype=torch.float64,
                )
                for sequence_id in selected:
                    label = f"{sequence_id}.mano.npy"
                    mano = _load_scalar_dict(
                        arctic_binding.read("raw_seqs", label), label=label
                    )
                    side_data = mano[side]
                    if not isinstance(side_data, dict):
                        raise NativeValidationError(f"{sequence_id}.{side}: invalid data")
                    frames = int(np.asarray(side_data["rot"]).shape[0])
                    for frame in (0, frames // 2, frames - 1):
                        rotation = np.asarray(side_data["rot"])[frame].astype(np.float64)
                        pose = np.asarray(side_data["pose"])[frame].astype(np.float64)
                        betas = np.asarray(side_data["shape"]).astype(np.float64)
                        translation = np.asarray(side_data["trans"])[frame].astype(np.float64)
                        observed = reconstruct_mano(
                            numpy_model, rotation, pose, betas, translation
                        )
                        with torch.no_grad():
                            expected = reference(
                                global_orient=torch.from_numpy(rotation[None]),
                                hand_pose=torch.from_numpy(pose[None]),
                                betas=torch.from_numpy(betas[None]),
                                transl=torch.from_numpy(translation[None]),
                            ).vertices.detach().cpu().numpy()[0]
                        difference = observed - expected
                        comparisons.append(
                            {
                                "side": side,
                                "sequence_id": sequence_id,
                                "frame": frame,
                                "maximum_absolute_error_m": float(
                                    np.max(np.abs(difference))
                                ),
                                "root_mean_square_error_m": float(
                                    np.sqrt(np.mean(difference * difference))
                                ),
                            }
                        )
        maximum = max(float(item["maximum_absolute_error_m"]) for item in comparisons)
        report["input"] = {
            "selected_sequences": selected,
            "selection": "first frozen calibration group and first frozen evaluation group",
            "frames_per_sequence": "first, midpoint, last",
            "mano_model_sha256": {
                side: models[side].source_sha256
                for side in ("left", "right")
            },
        }
        report["comparison_count"] = len(comparisons)
        report["maximum_observed_vertex_absolute_error_m"] = maximum
        report["comparisons"] = comparisons
        if maximum <= MAX_VERTEX_ABS_ERROR_M:
            verify_resource_report(
                preflight["resources"],
                data_root=args.data_root,
                mano_root=args.mano_root,
                workspace_root=ROOT,
            )
            report["status"] = PASS_STATUS
        else:
            report["status"] = "HOLD_SMPLX_MISMATCH"
    except (Exception, MemoryError) as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        report["status"] = "HOLD_ORACLE_INCOMPLETE"
        report["hold_reason"] = f"{type(exc).__name__}: {exc}"

    report["runtime"] = {
        "cpu_seconds": time.process_time() - started,
        "validated_code_sha256": _code_hashes(),
    }
    if report["status"] == PASS_STATUS:
        if primary_certificate is None:
            report["status"] = "HOLD_ORACLE_PROVENANCE"
            report["hold_reason"] = "primary-input certificate is missing"
        else:
            terminal_seal = terminal_payload_sha256(report)
            report["terminal_seal_sha256"] = terminal_seal
            if not _oracle_pass_contract(
                report,
                expected_certificate=primary_certificate,
                expected_seal=terminal_seal,
            ):
                report.pop("terminal_seal_sha256", None)
                report["status"] = "HOLD_ORACLE_PROVENANCE"
                report["hold_reason"] = "externally pinned oracle contract is incomplete"
    try:
        timestamped, alias = _write_result(args.output_dir, report)
    except (TypeError, ValueError) as exc:
        report = {
            "schema_version": 2,
            "milestone": "QH-E1-M2",
            "status": "HOLD_NONFINITE_OR_UNSERIALIZABLE_ORACLE",
            "engineering_result": "NO_SCIENTIFIC_RESULT",
            "hold_reason": type(exc).__name__,
            "license_attestation": {
                "user_attested_arctic": bool(args.licenses_attested),
                "user_attested_mano": bool(args.licenses_attested),
                "secret_values_recorded": False,
            },
        }
        timestamped, alias = _write_result(args.output_dir, report)
    print(f"[status] {report['status']}")
    print(f"[artifact] {timestamped}")
    print(f"[alias] {alias}")
    return 0 if report["status"] == PASS_STATUS else 3


if __name__ == "__main__":
    raise SystemExit(main())
