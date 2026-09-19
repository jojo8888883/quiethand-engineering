"""Fail-closed ARCTIC array and native-geometry validation for QuietHand M2.

This module is deliberately CPU-only.  It validates the frozen ARCTIC v1.0
raw-array contract and reconstructs a bounded set of MANO/object geometry
sentinels without importing the ARCTIC training stack or calling a model.

The ARCTIC ``.mano.npy`` and ``.egocam.dist.npy`` files are NumPy object
containers.  They are loaded only after the caller has established the exact
frozen archive hashes.  MANO model pickles are handled by an explicit global
allow-list and converted to inert NumPy data before any geometry is evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import pickle
import pickletools
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from quiethand.arctic_archive_binding import (
    FrozenArcticBinding,
    MANO_MODEL_SHA256,
    MANO_MODEL_SIZE,
)

try:  # NumPy 2 public package layout; fallback retains NumPy 1 compatibility.
    from numpy._core.multiarray import _reconstruct as _numpy_reconstruct
except ImportError:  # pragma: no cover - exercised only on NumPy 1.x
    from numpy.core.multiarray import _reconstruct as _numpy_reconstruct


MANO_SIDES = ("left", "right")
MANO_FIELDS = ("rot", "pose", "trans", "shape", "fitting_err")
CAMERA_FIELDS = (
    "R_k_cam_np",
    "T_k_cam_np",
    "intrinsics",
    "ego_markers.ref",
    "ego_markers.label",
    "R0",
    "T0",
    "dist8",
)
MAX_NUMPY_FILE_BYTES = 128 * 1024**2
MAX_MANO_PICKLE_BYTES = 16 * 1024**2
MAX_MANO_PICKLE_OPS = 10_000
MAX_MANO_MEMO_INDEX = 1_000
MAX_MANO_CONTAINER_OPS = 1_000
MAX_CSC_NNZ = 16 * 778
ROTATION_ORTHOGONALITY_TOL = 1e-4
ROTATION_DETERMINANT_TOL = 1e-4
INTRINSICS_TOL = 1e-8
GEOMETRY_FRAME_TOL_M = 2e-6
NATIVE_CONTACT_DISTANCE_M = 0.003


class NativeValidationError(RuntimeError):
    """A sanitized data-contract or native-geometry validation failure."""


class _ChProxy:
    def __new__(cls, *args: object, **kwargs: object) -> "_ChProxy":
        obj = object.__new__(cls)
        obj.newargs = args
        obj.newkwargs = kwargs
        return obj

    def __setstate__(self, state: object) -> None:
        self.state = state


class _SelectProxy(_ChProxy):
    pass


class _SparseProxy(_ChProxy):
    pass


class _RestrictedManoUnpickler(pickle.Unpickler):
    """Load only the seven globals used by the official MANO v1.2 files."""

    def find_class(self, module: str, name: str) -> object:
        if (module, name) in {("__builtin__", "set"), ("builtins", "set")}:
            return set
        if (module, name) in {
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy._core.multiarray", "_reconstruct"),
        }:
            # The frozen MANO files name numpy.core; NumPy 2 moved the callable.
            return _numpy_reconstruct
        if (module, name) == ("numpy", "ndarray"):
            return np.ndarray
        if (module, name) == ("numpy", "dtype"):
            return np.dtype
        if (module, name) == ("scipy.sparse.csc", "csc_matrix"):
            return _SparseProxy
        if (module, name) == ("chumpy.ch", "Ch"):
            return _ChProxy
        if (module, name) == ("chumpy.reordering", "Select"):
            return _SelectProxy
        raise pickle.UnpicklingError(f"forbidden MANO pickle global {module}.{name}")

    def persistent_load(self, pid: object) -> object:
        raise pickle.UnpicklingError("persistent IDs are forbidden in MANO pickle")


_ALLOWED_MANO_GLOBALS = {
    "__builtin__ set",
    "builtins set",
    "chumpy.ch Ch",
    "chumpy.reordering Select",
    "numpy dtype",
    "numpy ndarray",
    "numpy.core.multiarray _reconstruct",
    "numpy._core.multiarray _reconstruct",
    "scipy.sparse.csc csc_matrix",
}
_FORBIDDEN_PICKLE_OPS = {
    "EXT1",
    "EXT2",
    "EXT4",
    "INST",
    "OBJ",
    "PERSID",
    "BINPERSID",
    "STACK_GLOBAL",
}


@dataclass(frozen=True)
class ManoModel:
    side: str
    v_template: np.ndarray
    shapedirs: np.ndarray
    posedirs: np.ndarray
    joint_regressor: np.ndarray
    weights: np.ndarray
    parents: np.ndarray
    faces: np.ndarray
    hand_mean: np.ndarray
    hand_components: np.ndarray
    source_globals: tuple[str, ...]
    source_sha256: str
    source_dtypes: Mapping[str, str]


@dataclass(frozen=True)
class ObjectTemplate:
    name: str
    vertices_m: np.ndarray
    faces: np.ndarray
    top_mask: np.ndarray


def _float_array(value: object, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind != "f":
        raise NativeValidationError(f"{label}: floating-point dtype required")
    if not np.isfinite(array).all():
        raise NativeValidationError(f"{label}: NaN or Inf detected")
    return array


def _integer_array(value: object, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iu":
        raise NativeValidationError(f"{label}: integer dtype required")
    if (
        array.dtype.kind == "u"
        and array.size
        and int(array.max()) > np.iinfo(np.int64).max
    ):
        raise NativeValidationError(f"{label}: unsigned integer exceeds int64 range")
    return array


def _load_numpy_bytes(data: bytes, *, label: str, allow_pickle: bool) -> np.ndarray:
    if not data or len(data) > MAX_NUMPY_FILE_BYTES:
        raise NativeValidationError(f"{label}: file-size contract violated")
    try:
        return np.load(io.BytesIO(data), allow_pickle=allow_pickle)
    except (Exception, MemoryError) as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise NativeValidationError(f"{label}: NumPy load failed") from exc


def _load_scalar_dict(data: bytes, *, label: str) -> dict[str, object]:
    """Load only exact-archive bytes supplied by ``FrozenArcticBinding``."""

    raw = _load_numpy_bytes(data, label=label, allow_pickle=True)
    if raw.shape != () or raw.dtype != object:
        raise NativeValidationError(f"{label}: scalar object dictionary required")
    value = raw.item()
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise NativeValidationError(f"{label}: string-key dictionary required")
    return value


def _shape_is(array: np.ndarray, expected: tuple[int | None, ...]) -> bool:
    return array.ndim == len(expected) and all(
        want is None or got == want for got, want in zip(array.shape, expected)
    )


def _rotation_errors(rotations: np.ndarray) -> tuple[float, float]:
    product = np.matmul(np.swapaxes(rotations, -1, -2), rotations)
    identity = np.eye(3, dtype=np.float64)
    orthogonality = float(np.max(np.abs(product - identity)))
    determinant = float(np.max(np.abs(np.linalg.det(rotations) - 1.0)))
    return orthogonality, determinant


def validate_sequence_arrays(
    binding: FrozenArcticBinding, sequence_id: str
) -> dict[str, object]:
    """Validate one complete ARCTIC raw sequence and return secret-free facts."""

    mano_label = f"{sequence_id}.mano.npy"
    object_label = f"{sequence_id}.object.npy"
    camera_label = f"{sequence_id}.egocam.dist.npy"
    mano = _load_scalar_dict(
        binding.read("raw_seqs", mano_label), label=mano_label
    )
    camera = _load_scalar_dict(
        binding.read("raw_seqs", camera_label), label=camera_label
    )
    object_params = _float_array(
        _load_numpy_bytes(
            binding.read("raw_seqs", object_label),
            label=object_label,
            allow_pickle=False,
        ),
        label=f"{sequence_id}.object",
    )

    if set(mano) != set(MANO_SIDES):
        raise NativeValidationError(f"{sequence_id}: MANO sides must be left/right")
    frame_counts: set[int] = set()
    hand_facts: dict[str, object] = {}
    for side in MANO_SIDES:
        side_data = mano[side]
        if not isinstance(side_data, dict) or set(side_data) != set(MANO_FIELDS):
            raise NativeValidationError(f"{sequence_id}.{side}: MANO field contract violated")
        arrays = {
            key: _float_array(side_data[key], label=f"{sequence_id}.{side}.{key}")
            for key in MANO_FIELDS
        }
        required_shapes = {
            "rot": (None, 3),
            "pose": (None, 45),
            "trans": (None, 3),
            # The release stores one subject/hand beta vector.  Official
            # preprocessing repeats it per frame; data_doc.md says frames x 10.
            "shape": (10,),
            "fitting_err": (None,),
        }
        for key, expected in required_shapes.items():
            if not _shape_is(arrays[key], expected):
                raise NativeValidationError(
                    f"{sequence_id}.{side}.{key}: unexpected shape {arrays[key].shape}"
                )
        dynamic = {arrays[key].shape[0] for key in ("rot", "pose", "trans", "fitting_err")}
        if len(dynamic) != 1:
            raise NativeValidationError(f"{sequence_id}.{side}: MANO frame counts disagree")
        frame_counts.update(dynamic)
        hand_facts[side] = {
            "shape_storage": "constant_10_vector",
            "dtype": {key: str(value.dtype) for key, value in arrays.items()},
        }

    if not _shape_is(object_params, (None, 7)):
        raise NativeValidationError(f"{sequence_id}.object: expected frames x 7")
    frame_counts.add(int(object_params.shape[0]))

    if set(camera) != set(CAMERA_FIELDS):
        raise NativeValidationError(f"{sequence_id}.camera: field contract violated")
    camera_arrays = {key: np.asarray(camera[key]) for key in CAMERA_FIELDS}
    for key in CAMERA_FIELDS:
        if key == "ego_markers.label":
            if camera_arrays[key].shape != (5,) or camera_arrays[key].dtype.kind not in "US":
                raise NativeValidationError(f"{sequence_id}.camera.{key}: invalid labels")
            continue
        camera_arrays[key] = _float_array(
            camera_arrays[key], label=f"{sequence_id}.camera.{key}"
        )
    expected_camera_shapes = {
        "R_k_cam_np": (None, 3, 3),
        "T_k_cam_np": (None, 3, 1),
        "intrinsics": (3, 3),
        "ego_markers.ref": (5, 3),
        "R0": (3, 3),
        "T0": (3, 1),
        "dist8": (8,),
    }
    for key, expected in expected_camera_shapes.items():
        if not _shape_is(camera_arrays[key], expected):
            raise NativeValidationError(
                f"{sequence_id}.camera.{key}: unexpected shape {camera_arrays[key].shape}"
            )
    frame_counts.update(
        {
            int(camera_arrays["R_k_cam_np"].shape[0]),
            int(camera_arrays["T_k_cam_np"].shape[0]),
        }
    )
    if len(frame_counts) != 1:
        raise NativeValidationError(f"{sequence_id}: cross-file frame counts disagree")
    frames = next(iter(frame_counts))
    if frames <= 0:
        raise NativeValidationError(f"{sequence_id}: empty sequence")

    rotation_orthogonality, rotation_determinant = _rotation_errors(
        np.asarray(camera_arrays["R_k_cam_np"], dtype=np.float64)
    )
    r0_orthogonality, r0_determinant = _rotation_errors(
        np.asarray(camera_arrays["R0"], dtype=np.float64)[None]
    )
    if max(rotation_orthogonality, r0_orthogonality) > ROTATION_ORTHOGONALITY_TOL:
        raise NativeValidationError(f"{sequence_id}: camera rotation is not orthogonal")
    if max(rotation_determinant, r0_determinant) > ROTATION_DETERMINANT_TOL:
        raise NativeValidationError(f"{sequence_id}: camera rotation determinant is not +1")
    intrinsics = np.asarray(camera_arrays["intrinsics"], dtype=np.float64)
    if (
        intrinsics[0, 0] <= 0
        or intrinsics[1, 1] <= 0
        or not np.allclose(intrinsics[2], (0.0, 0.0, 1.0), atol=INTRINSICS_TOL, rtol=0.0)
    ):
        raise NativeValidationError(f"{sequence_id}: invalid pinhole intrinsics")

    return {
        "sequence_id": sequence_id,
        "frames": frames,
        "object_id": Path(sequence_id).name.split("_", 1)[0],
        "subject_id": Path(sequence_id).parts[0],
        "object_dtype": str(object_params.dtype),
        "hand_fields": hand_facts,
        "camera_rotation_max_orthogonality_error": rotation_orthogonality,
        "camera_rotation_max_determinant_error": rotation_determinant,
        "r0_orthogonality_error": r0_orthogonality,
        "r0_determinant_error": r0_determinant,
        "camera_dtypes": {
            key: str(value.dtype) for key, value in camera_arrays.items()
        },
    }


def _audit_pickle_program(data: bytes, *, label: str) -> tuple[str, ...]:
    if not data or len(data) > MAX_MANO_PICKLE_BYTES:
        raise NativeValidationError(f"{label}: MANO model size contract violated")
    globals_seen: set[str] = set()
    operation_count = 0
    container_count = 0
    try:
        for opcode, argument, _ in pickletools.genops(data):
            operation_count += 1
            if operation_count > MAX_MANO_PICKLE_OPS:
                raise NativeValidationError(f"{label}: pickle operation budget exceeded")
            if opcode.name in _FORBIDDEN_PICKLE_OPS:
                raise NativeValidationError(
                    f"{label}: forbidden pickle opcode {opcode.name}"
                )
            if opcode.name in {
                "MARK",
                "EMPTY_LIST",
                "EMPTY_DICT",
                "EMPTY_TUPLE",
                "APPENDS",
                "SETITEMS",
            }:
                container_count += 1
                if container_count > MAX_MANO_CONTAINER_OPS:
                    raise NativeValidationError(f"{label}: pickle container budget exceeded")
            if opcode.name in {"BINPUT", "LONG_BINPUT"} and int(argument) > MAX_MANO_MEMO_INDEX:
                raise NativeValidationError(f"{label}: pickle memo budget exceeded")
            if opcode.name == "GLOBAL":
                value = str(argument)
                globals_seen.add(value)
                if value not in _ALLOWED_MANO_GLOBALS:
                    raise NativeValidationError(
                        f"{label}: forbidden pickle global {value}"
                    )
    except (ValueError, pickle.UnpicklingError) as exc:
        if isinstance(exc, NativeValidationError):
            raise
        raise NativeValidationError(f"{label}: malformed pickle program") from exc
    return tuple(sorted(globals_seen))


def _proxy_state(value: object, *, proxy_type: type[_ChProxy], label: str) -> Mapping[str, object]:
    if not isinstance(value, proxy_type) or not isinstance(getattr(value, "state", None), dict):
        raise NativeValidationError(f"{label}: unexpected proxy state")
    return value.state


def _decode_shapedirs(value: object, *, label: str) -> np.ndarray:
    state = _proxy_state(value, proxy_type=_SelectProxy, label=label)
    if set(state) != {"a", "_dirty_vars", "_itr", "_depends_on_deps", "preferred_shape", "idxs"}:
        raise NativeValidationError(f"{label}: unexpected Select fields")
    base_state = _proxy_state(state["a"], proxy_type=_ChProxy, label=f"{label}.a")
    if "x" not in base_state:
        raise NativeValidationError(f"{label}: missing Ch.x")
    base = _float_array(base_state["x"], label=f"{label}.x")
    indices = _integer_array(state["idxs"], label=f"{label}.idxs")
    preferred_shape_array = _integer_array(
        state["preferred_shape"], label=f"{label}.preferred_shape"
    )
    preferred_shape = tuple(int(item) for item in preferred_shape_array)
    if preferred_shape != (778, 3, 10) or indices.shape != (778 * 3 * 10,):
        raise NativeValidationError(f"{label}: shapedirs selection contract violated")
    if indices.min() < 0 or indices.max() >= base.size:
        raise NativeValidationError(f"{label}: shapedirs selection index invalid")
    result = base.reshape(-1)[indices.astype(np.int64)].reshape(preferred_shape)
    return _float_array(result, label=label).astype(np.float64, copy=False)


def _decode_csc(value: object, *, label: str) -> np.ndarray:
    state = _proxy_state(value, proxy_type=_SparseProxy, label=label)
    required = {"indices", "data", "_shape", "indptr"}
    if not required.issubset(state):
        raise NativeValidationError(f"{label}: incomplete CSC state")
    shape_array = _integer_array(state["_shape"], label=f"{label}._shape")
    shape = tuple(int(item) for item in shape_array)
    if shape != (16, 778):
        raise NativeValidationError(f"{label}: unexpected CSC shape {shape}")
    indices = _integer_array(state["indices"], label=f"{label}.indices")
    data = _float_array(state["data"], label=f"{label}.data")
    indptr = _integer_array(state["indptr"], label=f"{label}.indptr")
    indices64 = indices.astype(np.int64, copy=False)
    indptr64 = indptr.astype(np.int64, copy=False)
    if (
        indices.ndim != 1
        or data.ndim != 1
        or indices.shape != data.shape
        or indptr.shape != (shape[1] + 1,)
        or int(indptr[0]) != 0
        or int(indptr[-1]) != data.size
        or data.size > MAX_CSC_NNZ
        or np.any(np.diff(indptr64) < 0)
        or (indices64.size and (indices64.min() < 0 or indices64.max() >= shape[0]))
    ):
        raise NativeValidationError(f"{label}: invalid CSC structure")
    dense = np.zeros(shape, dtype=np.float64)
    for column in range(shape[1]):
        start, end = int(indptr64[column]), int(indptr64[column + 1])
        column_indices = indices64[start:end]
        if np.unique(column_indices).size != column_indices.size:
            raise NativeValidationError(f"{label}: duplicate CSC row index")
        dense[column_indices, column] = data[start:end]
    return dense


def load_mano_model(source: Path | bytes, side: str) -> ManoModel:
    """Load official MANO data through a restricted, inert conversion path."""

    if side not in MANO_SIDES:
        raise ValueError("side must be left or right")
    label = source.name if isinstance(source, Path) else f"MANO_{side.upper()}.pkl"
    try:
        if isinstance(source, Path):
            if source.is_symlink() or not source.is_file():
                raise NativeValidationError(f"{label}: regular non-symlink model required")
            if source.stat().st_size != MANO_MODEL_SIZE[side]:
                raise NativeValidationError(f"{label}: frozen MANO size mismatch")
            with source.open("rb") as handle:
                data = handle.read(MAX_MANO_PICKLE_BYTES + 1)
                if len(data) != MANO_MODEL_SIZE[side] or handle.read(1):
                    raise NativeValidationError(f"{label}: bounded MANO read mismatch")
        else:
            data = bytes(source)
    except OSError as exc:
        raise NativeValidationError(f"{label}: unreadable MANO model") from exc
    if len(data) > MAX_MANO_PICKLE_BYTES:
        raise NativeValidationError(f"{label}: MANO model size contract violated")
    source_sha256 = hashlib.sha256(data).hexdigest()
    if source_sha256 != MANO_MODEL_SHA256[side]:
        raise NativeValidationError(f"{label}: frozen MANO SHA-256 mismatch")
    globals_seen = _audit_pickle_program(data, label=label)
    try:
        payload = _RestrictedManoUnpickler(io.BytesIO(data), encoding="latin1").load()
    except (Exception, MemoryError) as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise NativeValidationError(f"{label}: restricted MANO load failed") from exc
    if not isinstance(payload, dict):
        raise NativeValidationError(f"{label}: MANO dictionary required")
    required = {
        "f",
        "J_regressor",
        "kintree_table",
        "weights",
        "posedirs",
        "hands_mean",
        "v_template",
        "shapedirs",
        "hands_components",
    }
    if not required.issubset(payload):
        raise NativeValidationError(f"{label}: incomplete MANO model")

    shapedirs = _decode_shapedirs(payload["shapedirs"], label=f"{label}.shapedirs")
    joint_regressor = _decode_csc(
        payload["J_regressor"], label=f"{label}.J_regressor"
    )
    arrays = {
        "v_template": _float_array(payload["v_template"], label=f"{label}.v_template"),
        "posedirs": _float_array(payload["posedirs"], label=f"{label}.posedirs"),
        "weights": _float_array(payload["weights"], label=f"{label}.weights"),
        "kintree_table": _integer_array(
            payload["kintree_table"], label=f"{label}.kintree_table"
        ),
        "faces": _integer_array(payload["f"], label=f"{label}.f"),
        "hand_mean": _float_array(payload["hands_mean"], label=f"{label}.hands_mean"),
        "hand_components": _float_array(
            payload["hands_components"], label=f"{label}.hands_components"
        ),
    }
    expected_shapes = {
        "v_template": (778, 3),
        "posedirs": (778, 3, 135),
        "weights": (778, 16),
        "kintree_table": (2, 16),
        "faces": (1538, 3),
        "hand_mean": (45,),
        "hand_components": (45, 45),
    }
    for key, expected in expected_shapes.items():
        if arrays[key].shape != expected:
            raise NativeValidationError(f"{label}.{key}: unexpected shape")
    if arrays["faces"].min() < 0 or arrays["faces"].max() >= 778:
        raise NativeValidationError(f"{label}: invalid face indices")
    weights = arrays["weights"].astype(np.float64, copy=False)
    if np.any(weights < -1e-8) or not np.allclose(weights.sum(axis=1), 1.0, atol=2e-6, rtol=0.0):
        raise NativeValidationError(f"{label}: invalid skinning weights")
    tree = arrays["kintree_table"].astype(np.int64, copy=False)
    if not np.array_equal(tree[1], np.arange(16)):
        raise NativeValidationError(f"{label}: unsupported kinematic tree ordering")
    parents = tree[0].copy()
    parents[0] = -1
    if np.any(parents[1:] < 0) or np.any(parents[1:] >= np.arange(1, 16)):
        raise NativeValidationError(f"{label}: invalid parent topology")

    return ManoModel(
        side=side,
        v_template=arrays["v_template"].astype(np.float64, copy=False),
        shapedirs=shapedirs,
        posedirs=arrays["posedirs"].astype(np.float64, copy=False),
        joint_regressor=joint_regressor,
        weights=weights,
        parents=parents,
        faces=arrays["faces"].astype(np.int64, copy=False),
        hand_mean=arrays["hand_mean"].astype(np.float64, copy=False),
        hand_components=arrays["hand_components"].astype(np.float64, copy=False),
        source_globals=globals_seen,
        source_sha256=source_sha256,
        source_dtypes={
            **{key: str(value.dtype) for key, value in arrays.items()},
            "shapedirs_decoded": str(shapedirs.dtype),
            "joint_regressor_decoded": str(joint_regressor.dtype),
        },
    )


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Stable Rodrigues conversion for arrays ending in dimension three."""

    value = np.asarray(axis_angle, dtype=np.float64)
    if value.shape[-1:] != (3,) or not np.isfinite(value).all():
        raise NativeValidationError("axis-angle input must be finite (..., 3)")
    original_shape = value.shape[:-1]
    flat = value.reshape(-1, 3)
    theta = np.linalg.norm(flat, axis=1)
    skew = np.zeros((flat.shape[0], 3, 3), dtype=np.float64)
    skew[:, 0, 1] = -flat[:, 2]
    skew[:, 0, 2] = flat[:, 1]
    skew[:, 1, 0] = flat[:, 2]
    skew[:, 1, 2] = -flat[:, 0]
    skew[:, 2, 0] = -flat[:, 1]
    skew[:, 2, 1] = flat[:, 0]
    theta2 = theta * theta
    a = np.empty_like(theta)
    b = np.empty_like(theta)
    small = theta < 1e-8
    a[small] = 1.0 - theta2[small] / 6.0 + theta2[small] ** 2 / 120.0
    b[small] = 0.5 - theta2[small] / 24.0 + theta2[small] ** 2 / 720.0
    a[~small] = np.sin(theta[~small]) / theta[~small]
    b[~small] = (1.0 - np.cos(theta[~small])) / theta2[~small]
    identity = np.broadcast_to(np.eye(3), skew.shape).copy()
    rotation = identity + a[:, None, None] * skew + b[:, None, None] * np.matmul(skew, skew)
    return rotation.reshape(original_shape + (3, 3))


def reconstruct_mano(
    model: ManoModel,
    global_orient: np.ndarray,
    hand_pose: np.ndarray,
    betas: np.ndarray,
    translation_m: np.ndarray,
) -> np.ndarray:
    """Reconstruct MANO vertices using the official non-flat hand mean path."""

    orient = _float_array(global_orient, label="MANO global_orient").astype(np.float64)
    pose = _float_array(hand_pose, label="MANO hand_pose").astype(np.float64)
    shape = _float_array(betas, label="MANO betas").astype(np.float64)
    translation = _float_array(translation_m, label="MANO translation").astype(np.float64)
    if orient.shape != (3,) or pose.shape != (45,) or shape.shape != (10,) or translation.shape != (3,):
        raise NativeValidationError("MANO parameter shape contract violated")

    v_shaped = model.v_template + np.tensordot(model.shapedirs, shape, axes=([2], [0]))
    joints = model.joint_regressor @ v_shaped
    full_pose = np.concatenate((orient, pose + model.hand_mean)).reshape(16, 3)
    rotations = axis_angle_to_matrix(full_pose)
    pose_feature = (rotations[1:] - np.eye(3)).reshape(135)
    v_posed = model.v_template + np.tensordot(model.shapedirs, shape, axes=([2], [0]))
    v_posed = v_posed + np.tensordot(model.posedirs, pose_feature, axes=([2], [0]))

    relative_joints = joints.copy()
    relative_joints[1:] -= joints[model.parents[1:]]
    transforms = np.zeros((16, 4, 4), dtype=np.float64)
    transforms[:, 3, 3] = 1.0
    transforms[:, :3, :3] = rotations
    transforms[:, :3, 3] = relative_joints
    for index in range(1, 16):
        transforms[index] = transforms[model.parents[index]] @ transforms[index]
    rest_offset = np.einsum("jab,jb->ja", transforms[:, :3, :3], joints)
    relative_transforms = transforms.copy()
    relative_transforms[:, :3, 3] -= rest_offset
    blended = np.einsum("vj,jab->vab", model.weights, relative_transforms)
    homogeneous = np.concatenate((v_posed, np.ones((v_posed.shape[0], 1))), axis=1)
    vertices = np.einsum("vab,vb->va", blended, homogeneous)[:, :3]
    vertices += translation[None]
    if vertices.shape != (778, 3) or not np.isfinite(vertices).all():
        raise NativeValidationError("MANO reconstruction produced invalid vertices")
    return vertices


def _parse_obj(data: bytes, *, label: str) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    try:
        with io.StringIO(data.decode("utf-8", errors="strict")) as handle:
            for line in handle:
                if line.startswith("v "):
                    fields = line.split()
                    if len(fields) < 4:
                        raise NativeValidationError(f"{label}: malformed vertex")
                    vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
                elif line.startswith("f "):
                    fields = line.split()
                    if len(fields) != 4:
                        raise NativeValidationError(f"{label}: triangular faces required")
                    face = [int(item.split("/", 1)[0]) - 1 for item in fields[1:]]
                    faces.append(face)
    except (UnicodeError, ValueError) as exc:
        raise NativeValidationError(f"{label}: OBJ parse failed") from exc
    vertex_array = _float_array(
        np.asarray(vertices, dtype=np.float64), label=f"{label}.vertices"
    )
    face_array = _integer_array(
        np.asarray(faces, dtype=np.int64), label=f"{label}.faces"
    )
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3 or vertex_array.shape[0] == 0:
        raise NativeValidationError(f"{label}: invalid OBJ vertices")
    if face_array.ndim != 2 or face_array.shape[1] != 3 or face_array.shape[0] == 0:
        raise NativeValidationError(f"{label}: invalid OBJ faces")
    if face_array.min() < 0 or face_array.max() >= vertex_array.shape[0]:
        raise NativeValidationError(f"{label}: OBJ face index out of range")
    return vertex_array, face_array


def load_object_template(
    binding: FrozenArcticBinding, object_id: str
) -> ObjectTemplate:
    relative_root = Path("object_vtemplates") / object_id
    vertices_mm, faces = _parse_obj(
        binding.read("meta", relative_root / "mesh.obj"),
        label=f"{object_id}/mesh.obj",
    )
    try:
        parts = np.asarray(
            json.loads(
                binding.read("meta", relative_root / "parts.json").decode("utf-8")
            )
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise NativeValidationError(f"{object_id}: parts.json load failed") from exc
    if parts.shape != (vertices_mm.shape[0],) or parts.dtype.kind not in "iu":
        raise NativeValidationError(f"{object_id}: parts.json shape/type mismatch")
    if not np.isin(parts, (0, 1)).all():
        raise NativeValidationError(f"{object_id}: parts.json must contain only 0/1")
    # This follows frozen common/object_tensors.py exactly: bool(parts)+1,
    # followed by top_idx == 1, so original value 0 is the articulated top.
    top_mask = parts.astype(bool) == 0
    return ObjectTemplate(
        name=object_id,
        vertices_m=vertices_mm / 1000.0,
        faces=faces,
        top_mask=top_mask,
    )


def reconstruct_object(template: ObjectTemplate, params: np.ndarray) -> np.ndarray:
    row = _float_array(params, label="object params").astype(np.float64)
    if row.shape != (7,):
        raise NativeValidationError("object params must have seven values")
    articulation, global_orient, translation_mm = row[0], row[1:4], row[4:7]
    vertices = template.vertices_m.copy()
    articulation_rotation = axis_angle_to_matrix(
        np.array((0.0, 0.0, -float(articulation)), dtype=np.float64)
    )
    vertices[template.top_mask] = vertices[template.top_mask] @ articulation_rotation.T
    global_rotation = axis_angle_to_matrix(global_orient)
    vertices = vertices @ global_rotation.T + translation_mm[None] / 1000.0
    if not np.isfinite(vertices).all():
        raise NativeValidationError("object reconstruction produced invalid vertices")
    return vertices


def _minimum_vertex_distance(a: np.ndarray, b: np.ndarray) -> float:
    a64 = _float_array(a, label="distance points A").astype(np.float64, copy=False)
    b64 = _float_array(b, label="distance points B").astype(np.float64, copy=False)
    if (
        a64.ndim != 2
        or b64.ndim != 2
        or a64.shape[1:] != (3,)
        or b64.shape[1:] != (3,)
        or not a64.size
        or not b64.size
    ):
        raise NativeValidationError("distance points must be non-empty N x 3 arrays")
    minimum = float("inf")
    for a_start in range(0, a64.shape[0], 256):
        a_chunk = a64[a_start : a_start + 256]
        for b_start in range(0, b64.shape[0], 256):
            b_chunk = b64[b_start : b_start + 256]
            with np.errstate(over="raise", invalid="raise"):
                try:
                    delta = a_chunk[:, None, :] - b_chunk[None, :, :]
                except FloatingPointError as exc:
                    raise NativeValidationError("distance subtraction overflow") from exc
            scale = np.max(np.abs(delta), axis=2)
            normalized = np.zeros_like(delta)
            nonzero = scale > 0
            normalized[nonzero] = delta[nonzero] / scale[nonzero][:, None]
            distances = scale * np.sqrt(np.sum(normalized * normalized, axis=2))
            if not np.isfinite(distances).all():
                raise NativeValidationError("distance computation produced NaN or Inf")
            minimum = min(minimum, float(np.min(distances)))
    if not np.isfinite(minimum):
        raise NativeValidationError("minimum distance is not finite")
    return minimum


def _transform_points(rotation: np.ndarray, translation: np.ndarray, points: np.ndarray) -> np.ndarray:
    transformed = points @ rotation.T + translation.reshape(1, 3)
    if not np.isfinite(transformed).all():
        raise NativeValidationError("point transform produced NaN or Inf")
    return transformed


def _uniform_frame_indices(frames: int, count: int) -> tuple[int, ...]:
    if frames <= 0 or count <= 0:
        return ()
    return tuple(sorted(set(int(round(value)) for value in np.linspace(0, frames - 1, count))))


def validate_geometry_sentinels(
    *,
    binding: FrozenArcticBinding,
    models: Mapping[str, ManoModel],
    sequence_ids: Sequence[str],
    samples_per_sequence: int = 3,
) -> dict[str, object]:
    """Reconstruct bounded native-geometry samples and audit frame transforms."""

    object_cache: dict[str, ObjectTemplate] = {}
    samples: list[dict[str, object]] = []
    maximum_frame_residual = 0.0
    minimum_distance = float("inf")
    strict_contact_samples = 0
    for sequence_id in sequence_ids:
        mano_label = f"{sequence_id}.mano.npy"
        object_label = f"{sequence_id}.object.npy"
        camera_label = f"{sequence_id}.egocam.dist.npy"
        mano = _load_scalar_dict(
            binding.read("raw_seqs", mano_label), label=mano_label
        )
        object_params = _float_array(
            _load_numpy_bytes(
                binding.read("raw_seqs", object_label),
                label=object_label,
                allow_pickle=False,
            ),
            label=f"{sequence_id}.object",
        )
        camera = _load_scalar_dict(
            binding.read("raw_seqs", camera_label), label=camera_label
        )
        frames = int(object_params.shape[0])
        object_id = Path(sequence_id).name.split("_", 1)[0]
        if object_id not in object_cache:
            object_cache[object_id] = load_object_template(binding, object_id)
        template = object_cache[object_id]
        for frame in _uniform_frame_indices(frames, samples_per_sequence):
            object_vertices = reconstruct_object(template, object_params[frame])
            rotation = _float_array(
                np.asarray(camera["R_k_cam_np"])[frame], label=f"{sequence_id}.R[{frame}]"
            ).astype(np.float64)
            translation = _float_array(
                np.asarray(camera["T_k_cam_np"])[frame], label=f"{sequence_id}.T[{frame}]"
            ).astype(np.float64).reshape(3)
            object_ego = _transform_points(rotation, translation, object_vertices)
            distances: dict[str, float] = {}
            residuals: dict[str, float] = {}
            for side in MANO_SIDES:
                side_data = mano[side]
                if not isinstance(side_data, dict):
                    raise NativeValidationError(f"{sequence_id}.{side}: invalid MANO data")
                hand_vertices = reconstruct_mano(
                    models[side],
                    np.asarray(side_data["rot"])[frame],
                    np.asarray(side_data["pose"])[frame],
                    np.asarray(side_data["shape"]),
                    np.asarray(side_data["trans"])[frame],
                )
                world_distance = _minimum_vertex_distance(hand_vertices, object_vertices)
                hand_ego = _transform_points(rotation, translation, hand_vertices)
                ego_distance = _minimum_vertex_distance(hand_ego, object_ego)
                residual = abs(world_distance - ego_distance)
                if not all(np.isfinite(value) for value in (world_distance, ego_distance, residual)):
                    raise NativeValidationError(
                        f"{sequence_id}[{frame}].{side}: non-finite geometry result"
                    )
                if residual > GEOMETRY_FRAME_TOL_M:
                    raise NativeValidationError(
                        f"{sequence_id}[{frame}].{side}: world/ego distance mismatch"
                    )
                distances[side] = world_distance
                residuals[side] = residual
                maximum_frame_residual = max(maximum_frame_residual, residual)
                minimum_distance = min(minimum_distance, world_distance)
                if world_distance < NATIVE_CONTACT_DISTANCE_M:
                    strict_contact_samples += 1
            samples.append(
                {
                    "sequence_id": sequence_id,
                    "frame": frame,
                    "object_id": object_id,
                    "minimum_vertex_distance_m": distances,
                    "world_to_ego_distance_residual_m": residuals,
                }
            )
    expected_sample_count = len(sequence_ids) * samples_per_sequence
    all_finite = all(
        np.isfinite(value)
        for sample in samples
        for mapping in (
            sample["minimum_vertex_distance_m"],
            sample["world_to_ego_distance_residual_m"],
        )
        for value in mapping.values()
    )
    closed = (
        bool(sequence_ids)
        and len(samples) == expected_sample_count
        and len(samples) * 2 == expected_sample_count * 2
        and all_finite
        and np.isfinite(maximum_frame_residual)
        and maximum_frame_residual <= GEOMETRY_FRAME_TOL_M
    )
    return {
        "selected_sequence_count": len(sequence_ids),
        "samples_per_sequence_requested": samples_per_sequence,
        "sample_count": len(samples),
        "hand_sample_count": len(samples) * 2,
        "strict_contact_comparator": "<",
        "strict_contact_distance_m": NATIVE_CONTACT_DISTANCE_M,
        "strict_contact_hand_sample_count": strict_contact_samples,
        "minimum_observed_vertex_distance_m": None if not samples else minimum_distance,
        "maximum_world_to_ego_distance_residual_m": maximum_frame_residual,
        "object_templates_loaded": sorted(object_cache),
        "samples": samples,
        "expected_sample_count": expected_sample_count,
        "all_results_finite": bool(all_finite),
        "closed": bool(closed),
    }


def summarize_array_facts(facts: Iterable[Mapping[str, object]]) -> dict[str, object]:
    values = list(facts)
    if not values:
        raise NativeValidationError("no sequence facts to summarize")
    frames = [int(value["frames"]) for value in values]
    dtype_matrix = {
        "object": sorted({str(value["object_dtype"]) for value in values}),
        "mano": {
            side: {
                field: sorted(
                    {
                        str(value["hand_fields"][side]["dtype"][field])
                        for value in values
                    }
                )
                for field in MANO_FIELDS
            }
            for side in MANO_SIDES
        },
        "camera": {
            field: sorted(
                {str(value["camera_dtypes"][field]) for value in values}
            )
            for field in CAMERA_FIELDS
        },
    }
    return {
        "sequence_count": len(values),
        "total_frames": sum(frames),
        "minimum_frames": min(frames),
        "maximum_frames": max(frames),
        "subject_ids": sorted({str(value["subject_id"]) for value in values}),
        "object_ids": sorted({str(value["object_id"]) for value in values}),
        "maximum_camera_rotation_orthogonality_error": max(
            float(value["camera_rotation_max_orthogonality_error"]) for value in values
        ),
        "maximum_camera_rotation_determinant_error": max(
            float(value["camera_rotation_max_determinant_error"]) for value in values
        ),
        "mano_shape_release_storage": "constant_10_vector_for_both_hands_in_all_sequences",
        "mano_shape_documentation_discrepancy": {
            "data_doc_claim": "frames x 10",
            "release_observation": "10",
            "official_preprocessing_behavior": "repeat constant vector across frames",
            "silent_coercion": False,
        },
        "observed_dtype_matrix": dtype_matrix,
        "closed": True,
    }


def mano_model_facts(model: ManoModel) -> dict[str, object]:
    return {
        "side": model.side,
        "vertices": int(model.v_template.shape[0]),
        "faces": int(model.faces.shape[0]),
        "joints": int(model.weights.shape[1]),
        "shape_dimensions": int(model.shapedirs.shape[2]),
        "pose_blend_dimensions": int(model.posedirs.shape[2]),
        "source_globals": list(model.source_globals),
        "source_sha256": model.source_sha256,
        "source_content_dtypes": dict(model.source_dtypes),
        "validated_inert_dtypes": {
            "v_template": str(model.v_template.dtype),
            "shapedirs": str(model.shapedirs.dtype),
            "posedirs": str(model.posedirs.dtype),
            "joint_regressor": str(model.joint_regressor.dtype),
            "weights": str(model.weights.dtype),
            "parents": str(model.parents.dtype),
            "faces": str(model.faces.dtype),
            "hand_mean": str(model.hand_mean.dtype),
            "hand_components": str(model.hand_components.dtype),
        },
        "maximum_weight_sum_error": float(np.max(np.abs(model.weights.sum(axis=1) - 1.0))),
        "closed": True,
    }


UNIT_COORDINATE_EVIDENCE = {
    "mano_translation": {
        "unit": "meter",
        "source": "official common/body_models.py MANO output path and processing.py world geometry",
        "runtime_conversion": "none",
    },
    "object_template": {
        "stored_unit": "millimeter",
        "runtime_unit": "meter",
        "source": "official common/object_tensors.py divides mesh vertices by 1000",
        "runtime_conversion": "/1000",
    },
    "object_translation": {
        "stored_unit": "millimeter",
        "runtime_unit": "meter",
        "source": "official processing.py passes obj_trans / 1000 to ObjectTensors",
        "runtime_conversion": "/1000",
    },
    "object_articulation": {
        "unit": "radian",
        "source": "official data_doc.md and preprocess_dataset.py",
    },
    "rotations": {
        "representation": "axis-angle radians",
        "source": "official data_doc.md",
    },
    "egocamera": {
        "transform": "world_to_ego: R @ p + T",
        "source": "official preprocess_dataset.py and common/transforms.py",
        "translation_unit": "meter (applied directly to meter world geometry)",
    },
}
