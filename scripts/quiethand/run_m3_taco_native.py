#!/usr/bin/env python3
"""Validate frozen TACO native fields without opening model evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quiethand.arctic_native_validation import (  # noqa: E402
    NativeValidationError,
    load_mano_model,
)
from quiethand.safe_torch_pickle import SafeTorchPickleError  # noqa: E402
from quiethand.taco_contract import (  # noqa: E402
    TACO_SELECTION_SHA256,
    TACO_SEQUENCE_LIST,
    canonical_selection_bytes,
    deterministic_taco_split,
    load_frozen_sequence_ids,
)
from quiethand.taco_native import (  # noqa: E402
    TacoNativeError,
    reconstruct_taco_mano,
    validate_native_contact_geometry,
    validate_object_models,
    validate_sequence_native,
    verify_receipt_files,
)


REPOSITORY = ROOT / "external_repos" / "TACO-Instructions"
PROBE_PATH = ROOT / "artifacts" / "quiethand" / "m3" / "QH_M3_TACO_PROBE.json"
RECEIPT_PATH = (
    ROOT / "artifacts" / "quiethand" / "m3" / "QH_M3_TACO_LANDING_RECEIPT.json"
)
DATA_ROOT = ROOT / "external_data" / "taco_v1"
MANO_ROOT = ROOT / "external_data" / "mano_v1_2" / "models"
RESULT_PATH = (
    ROOT / "artifacts" / "quiethand" / "m3" / "QH_M3_TACO_NATIVE_CONTRACT.json"
)
PROTOCOL_V1_1_PATH = (
    ROOT
    / "refine-logs"
    / "QUIETHAND_M3_PROTOCOL_V1_1_20260820_122944.md"
)
PROTOCOL_V1_1_SHA256 = (
    "6f3216350d47b9eee2f029f3e039b15f8929c7b1b9950ce4e163811be673a4cc"
)


def _atomic_json(path: Path, value: dict[str, object]) -> None:
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_payload_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _smplx_struct(model: object) -> object:
    from scipy.sparse import csc_matrix
    from smplx.utils import Struct

    parents = np.asarray(model.parents, dtype=np.int64).copy()
    parents[0] = 0
    return Struct(
        f=np.asarray(model.faces, dtype=np.int64),
        v_template=np.asarray(model.v_template, dtype=np.float64),
        shapedirs=np.asarray(model.shapedirs, dtype=np.float64),
        J_regressor=csc_matrix(np.asarray(model.joint_regressor, dtype=np.float64)),
        posedirs=np.asarray(model.posedirs, dtype=np.float64),
        kintree_table=np.vstack((parents, np.arange(16, dtype=np.int64))),
        weights=np.asarray(model.weights, dtype=np.float64),
        hands_components=np.asarray(model.hand_components, dtype=np.float64),
        hands_mean=np.asarray(model.hand_mean, dtype=np.float64),
    )


def _taco_mano_smplx_oracle() -> dict[str, object]:
    """Independently check TACO's flat-mean, wrist-centred MANO convention."""

    import smplx
    import torch

    rng = np.random.default_rng(20260820)
    maximum_error = 0.0
    sides = {}
    for side in ("left", "right"):
        model = load_mano_model(MANO_ROOT / f"MANO_{side.upper()}.pkl", side)
        reference = smplx.MANO(
            model_path="validated-inert-struct",
            data_struct=_smplx_struct(model),
            is_rhand=side == "right",
            use_pca=False,
            flat_hand_mean=True,
            create_transl=False,
            dtype=torch.float64,
        )
        poses = rng.normal(0.0, 0.2, size=(3, 48))
        shapes = rng.normal(0.0, 0.1, size=(3, 10))
        translations = rng.normal(0.0, 0.3, size=(3, 3))
        with torch.no_grad():
            output = reference(
                global_orient=torch.as_tensor(poses[:, :3], dtype=torch.float64),
                hand_pose=torch.as_tensor(poses[:, 3:], dtype=torch.float64),
                betas=torch.as_tensor(shapes, dtype=torch.float64),
                return_verts=True,
            )
        reference_vertices = output.vertices.detach().cpu().numpy()
        reference_joints = output.joints.detach().cpu().numpy()
        reference_vertices = (
            reference_vertices
            - reference_joints[:, :1, :]
            + translations[:, None, :]
        )
        candidate = np.stack(
            [
                reconstruct_taco_mano(model, poses[index], shapes[index], translations[index])
                for index in range(3)
            ]
        )
        error = float(np.max(np.abs(candidate - reference_vertices)))
        if not np.isfinite(error):
            raise TacoNativeError("TACO MANO oracle produced a non-finite error")
        sides[side] = {"sample_count": 3, "maximum_vertex_absolute_error_m": error}
        maximum_error = max(maximum_error, error)
    if maximum_error > 1e-7:
        raise TacoNativeError("TACO MANO reconstruction differs from the smplx oracle")
    return {
        "closed": True,
        "independent_implementation": "smplx.MANO",
        "flat_hand_mean": True,
        "center_joint": 0,
        "sample_count": 6,
        "maximum_vertex_absolute_error_m": maximum_error,
        "maximum_allowed_error_m": 1e-7,
        "sides": sides,
    }


def main() -> int:
    try:
        if _sha256(PROTOCOL_V1_1_PATH) != PROTOCOL_V1_1_SHA256:
            raise TacoNativeError("approved M3-v1.1 protocol binding changed")
        probe_bytes = PROBE_PATH.read_bytes()
        receipt_bytes = RECEIPT_PATH.read_bytes()
        probe = json.loads(probe_bytes)
        receipt = json.loads(receipt_bytes)
        if (
            probe.get("status") != "READY_FOR_SELECTIVE_LANDING"
            or probe.get("selection_sha256") != TACO_SELECTION_SHA256
            or receipt.get("status") != "READY_FOR_NATIVE_VALIDATION"
            or receipt.get("probe_sha256") != hashlib.sha256(probe_bytes).hexdigest()
            or receipt.get("selected_member_manifest_sha256")
            != probe.get("selected_member_manifest_sha256")
        ):
            raise TacoNativeError("probe/landing provenance does not close")
        split = deterministic_taco_split(
            load_frozen_sequence_ids(REPOSITORY / TACO_SEQUENCE_LIST)
        )
        canonical_selection_bytes(split)
        expected_members = [
            member for catalog in probe["catalogs"] for member in catalog["selected_members"]
        ]
        completed = receipt["completed"]
        print("[verify] 1387 landed member hashes", flush=True)
        verify_receipt_files(DATA_ROOT, expected_members, completed)
        sequences = []
        for entry in split:
            print(f"[native] {entry.rank:02d}/60 {entry.sequence_id}", flush=True)
            sequences.append(validate_sequence_native(DATA_ROOT, entry, completed))
        object_catalog = next(
            item for item in probe["catalogs"] if item["filename"] == "Object_Models.zip"
        )
        print("[native] 67 object meshes", flush=True)
        object_models = validate_object_models(
            DATA_ROOT, object_catalog["selected_members"]
        )
        print("[native] independent TACO MANO convention oracle", flush=True)
        mano_oracle = _taco_mano_smplx_oracle()
        print("[native] 300-event hand/object contact geometry", flush=True)
        contact_geometry = validate_native_contact_geometry(
            DATA_ROOT, split, completed, MANO_ROOT
        )
        result = {
            "schema": "quiethand.m3.taco_native_contract.v1_1",
            "status": "READY_NATIVE",
            "scientific_result": False,
            "created_unix_seconds": int(time.time()),
            "probe_sha256": hashlib.sha256(probe_bytes).hexdigest(),
            "landing_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "selection_sha256": TACO_SELECTION_SHA256,
            "sequence_count": len(sequences),
            "event_count": len(sequences) * 5,
            "all_event_windows_valid": all(
                all(item["event_windows_valid"]) for item in sequences
            ),
            "native_contact_comparator": "<",
            "native_contact_distance_m": 0.003,
            "native_contact_geometry_status": "CLOSED_300_EVENTS",
            "native_contact_geometry": contact_geometry,
            "native_contact_geometry_sha256": _canonical_payload_sha256(
                contact_geometry
            ),
            "taco_mano_smplx_oracle": mano_oracle,
            "protocol_v1_1": {
                "status": "USER_APPROVED_EFFECTIVE",
                "path": "refine-logs/QUIETHAND_M3_PROTOCOL_V1_1_20260820_122944.md",
                "sha256": PROTOCOL_V1_1_SHA256,
                "field": "egocentric_depth_container_header_fps",
                "accepted_rgb_header_fps": "30/1",
                "accepted_depth_header_fps_anomaly": "500000/33333",
                "primary_source": "https://arxiv.org/html/2401.08399#S3.SS1",
                "primary_source_statement": "TACO cameras and mocap system operate at 30 Hz",
                "effective_rule": "retain exact one-to-one frame-index alignment and t=i/30 s; record the exact depth header anomaly; never resample, duplicate, drop, interpolate or replace depth frames",
            },
            "sequences": sequences,
            "object_models": object_models,
            "code_sha256": {
                "native_core": _sha256(ROOT / "quiethand" / "taco_native.py"),
                "safe_torch_pickle": _sha256(
                    ROOT / "quiethand" / "safe_torch_pickle.py"
                ),
                "mano_lbs_core": _sha256(
                    ROOT / "quiethand" / "arctic_native_validation.py"
                ),
                "runner": _sha256(Path(__file__).resolve()),
            },
        }
        _atomic_json(RESULT_PATH, result)
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        SafeTorchPickleError,
        TacoNativeError,
        NativeValidationError,
    ) as exc:
        print(f"[hold] HOLD_ENGINEERING_INCOMPLETE: {exc}", file=sys.stderr)
        return 3
    print(f"[artifact] {RESULT_PATH}", flush=True)
    print("[status] READY_NATIVE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
