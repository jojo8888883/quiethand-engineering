"""Native TACO field, time, and coordinate validation for QuietHand M3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import BinaryIO, Iterable, Mapping, Sequence

import numpy as np

from .arctic_native_validation import (
    ManoModel,
    NativeValidationError,
    load_mano_model,
    reconstruct_mano,
)
from .safe_torch_pickle import SafeTorchPickleError, load_taco_tensor_pickle
from .taco_contract import TacoSplitEntry


RGB_WIDTH = 1920
RGB_HEIGHT = 1080
SEGMENTATION_WIDTH = 1024
SEGMENTATION_HEIGHT = 750
EXPECTED_FPS = 30.0
RGB_FPS_NUMERATOR = 30
RGB_FPS_DENOMINATOR = 1
DEPTH_HEADER_FPS_NUMERATOR = 500_000
DEPTH_HEADER_FPS_DENOMINATOR = 33_333
MATRIX_TOLERANCE = 2e-3
NATIVE_CONTACT_DISTANCE_M = 0.003


class TacoNativeError(RuntimeError):
    """A selected TACO native field violates the frozen M3 contract."""


@dataclass(frozen=True)
class VideoMetadata:
    container: str
    frame_count: int
    fps_numerator: int
    fps_denominator: int
    width: int
    height: int

    @property
    def fps(self) -> float:
        return self.fps_numerator / self.fps_denominator


def _validate_rgbd_timebase(rgb: VideoMetadata, depth: VideoMetadata) -> None:
    """Enforce the exact approved M3-v1.1 source-header contract."""

    if (
        rgb.container != "mp4"
        or depth.container != "avi"
        or rgb.frame_count != depth.frame_count
        or rgb.width != RGB_WIDTH
        or rgb.height != RGB_HEIGHT
        or depth.width != RGB_WIDTH
        or depth.height != RGB_HEIGHT
        or rgb.fps_numerator != RGB_FPS_NUMERATOR
        or rgb.fps_denominator != RGB_FPS_DENOMINATOR
        or depth.fps_numerator != DEPTH_HEADER_FPS_NUMERATOR
        or depth.fps_denominator != DEPTH_HEADER_FPS_DENOMINATOR
        or rgb.frame_count < 15
    ):
        raise TacoNativeError("RGB-D timebase violates exact M3-v1.1")


def _read_exact(handle: BinaryIO, offset: int, size: int) -> bytes:
    handle.seek(offset)
    payload = handle.read(size)
    if len(payload) != size:
        raise TacoNativeError("video container is truncated")
    return payload


def _mp4_boxes(
    handle: BinaryIO, start: int, end: int
) -> Iterable[tuple[bytes, int, int]]:
    position = start
    while position < end:
        if end - position < 8:
            raise TacoNativeError("MP4 box header is truncated")
        header = _read_exact(handle, position, 8)
        size, kind = struct.unpack(">I4s", header)
        header_bytes = 8
        if size == 1:
            size = struct.unpack(">Q", _read_exact(handle, position + 8, 8))[0]
            header_bytes = 16
        elif size == 0:
            size = end - position
        if size < header_bytes or position + size > end:
            raise TacoNativeError("MP4 box size is invalid")
        yield kind, position + header_bytes, position + size
        position += size


def _single_box(
    handle: BinaryIO, start: int, end: int, kind: bytes
) -> tuple[int, int]:
    matches = [(payload, box_end) for box, payload, box_end in _mp4_boxes(handle, start, end) if box == kind]
    if len(matches) != 1:
        raise TacoNativeError(f"MP4 requires exactly one {kind.decode('ascii')} box")
    return matches[0]


def parse_mp4_metadata(path: Path) -> VideoMetadata:
    path = Path(path)
    size = path.stat().st_size
    if size <= 0:
        raise TacoNativeError("MP4 is empty")
    with path.open("rb") as handle:
        moov_start, moov_end = _single_box(handle, 0, size, b"moov")
        video_tracks: list[tuple[int, int, int, int]] = []
        for box, trak_start, trak_end in _mp4_boxes(handle, moov_start, moov_end):
            if box != b"trak":
                continue
            tkhd_start, tkhd_end = _single_box(handle, trak_start, trak_end, b"tkhd")
            if tkhd_end - tkhd_start < 8:
                raise TacoNativeError("MP4 tkhd box is too short")
            width_fixed, height_fixed = struct.unpack(
                ">II", _read_exact(handle, tkhd_end - 8, 8)
            )
            width = width_fixed >> 16
            height = height_fixed >> 16
            mdia_start, mdia_end = _single_box(handle, trak_start, trak_end, b"mdia")
            hdlr_start, hdlr_end = _single_box(handle, mdia_start, mdia_end, b"hdlr")
            if hdlr_end - hdlr_start < 12:
                raise TacoNativeError("MP4 hdlr box is too short")
            handler = _read_exact(handle, hdlr_start + 8, 4)
            if handler != b"vide":
                continue
            mdhd_start, mdhd_end = _single_box(handle, mdia_start, mdia_end, b"mdhd")
            mdhd = _read_exact(handle, mdhd_start, min(32, mdhd_end - mdhd_start))
            if len(mdhd) < 20:
                raise TacoNativeError("MP4 mdhd box is too short")
            version = mdhd[0]
            if version == 0:
                timescale = struct.unpack(">I", mdhd[12:16])[0]
                duration = struct.unpack(">I", mdhd[16:20])[0]
            elif version == 1 and len(mdhd) >= 32:
                timescale = struct.unpack(">I", mdhd[20:24])[0]
                duration = struct.unpack(">Q", mdhd[24:32])[0]
            else:
                raise TacoNativeError("MP4 mdhd version is unsupported")
            minf_start, minf_end = _single_box(handle, mdia_start, mdia_end, b"minf")
            stbl_start, stbl_end = _single_box(handle, minf_start, minf_end, b"stbl")
            stsz_start, stsz_end = _single_box(handle, stbl_start, stbl_end, b"stsz")
            stsz = _read_exact(handle, stsz_start, min(12, stsz_end - stsz_start))
            if len(stsz) != 12:
                raise TacoNativeError("MP4 stsz box is too short")
            frame_count = struct.unpack(">I", stsz[8:12])[0]
            stts_start, stts_end = _single_box(handle, stbl_start, stbl_end, b"stts")
            stts_header = _read_exact(handle, stts_start, min(8, stts_end - stts_start))
            if len(stts_header) != 8:
                raise TacoNativeError("MP4 stts box is too short")
            entry_count = struct.unpack(">I", stts_header[4:8])[0]
            if entry_count <= 0 or entry_count > 1_000_000:
                raise TacoNativeError("MP4 stts entry count is invalid")
            entries = _read_exact(handle, stts_start + 8, entry_count * 8)
            timing_frames = sum(
                struct.unpack(">I", entries[offset : offset + 4])[0]
                for offset in range(0, len(entries), 8)
            )
            if timing_frames != frame_count or timescale <= 0 or duration <= 0:
                raise TacoNativeError("MP4 time/sample tables disagree")
            video_tracks.append((frame_count, timescale, duration, width << 16 | height))
    if len(video_tracks) != 1:
        raise TacoNativeError("MP4 requires exactly one video track")
    frame_count, timescale, duration, packed_size = video_tracks[0]
    width, height = packed_size >> 16, packed_size & 0xFFFF
    fps = Fraction(frame_count * timescale, duration)
    return VideoMetadata(
        container="mp4",
        frame_count=frame_count,
        fps_numerator=fps.numerator,
        fps_denominator=fps.denominator,
        width=width,
        height=height,
    )


def parse_avi_metadata(path: Path) -> VideoMetadata:
    path = Path(path)
    with path.open("rb") as handle:
        prefix = handle.read(min(path.stat().st_size, 1024 * 1024))
    if len(prefix) < 12 or prefix[:4] != b"RIFF" or prefix[8:12] != b"AVI ":
        raise TacoNativeError("depth video is not an AVI RIFF container")
    offset = prefix.find(b"avih")
    if offset < 0 or offset + 64 > len(prefix):
        raise TacoNativeError("AVI main header is missing")
    chunk_size = struct.unpack("<I", prefix[offset + 4 : offset + 8])[0]
    if chunk_size < 40 or offset + 8 + chunk_size > len(prefix):
        raise TacoNativeError("AVI main header is truncated")
    values = struct.unpack("<10I", prefix[offset + 8 : offset + 48])
    microseconds_per_frame = values[0]
    frame_count = values[4]
    width = values[8]
    height = values[9]
    if microseconds_per_frame <= 0 or frame_count <= 0:
        raise TacoNativeError("AVI timing is invalid")
    fps = Fraction(1_000_000, microseconds_per_frame)
    return VideoMetadata(
        container="avi",
        frame_count=frame_count,
        fps_numerator=fps.numerator,
        fps_denominator=fps.denominator,
        width=width,
        height=height,
    )


def _npy_header(path: Path) -> tuple[tuple[int, ...], bool, np.dtype]:
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version in {(2, 0), (3, 0)}:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise TacoNativeError("NPY version is unsupported")
    return tuple(int(item) for item in shape), bool(fortran), np.dtype(dtype)


def _load_float32_array(path: Path, shape_tail: tuple[int, ...]) -> np.ndarray:
    shape, fortran, dtype = _npy_header(path)
    if fortran or dtype != np.dtype("float32") or shape[1:] != shape_tail or shape[0] <= 0:
        raise TacoNativeError("native float32 array header is outside contract")
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    if not np.isfinite(array).all():
        raise TacoNativeError("native float32 array contains non-finite values")
    return array


def _validate_transform_batch(array: np.ndarray) -> float:
    if not np.allclose(array[:, 3, :], np.array([0, 0, 0, 1], dtype=np.float32), atol=MATRIX_TOLERANCE):
        raise TacoNativeError("homogeneous transform last row is invalid")
    rotations = np.asarray(array[:, :3, :3], dtype=np.float64)
    identities = np.matmul(np.swapaxes(rotations, 1, 2), rotations)
    residual = float(np.max(np.abs(identities - np.eye(3))))
    determinants = np.linalg.det(rotations)
    if residual > MATRIX_TOLERANCE or np.max(np.abs(determinants - 1.0)) > MATRIX_TOLERANCE:
        raise TacoNativeError("native transform rotation is not rigid")
    return max(residual, float(np.max(np.abs(determinants - 1.0))))


def _validate_hand_pickle(
    pose_path: Path, shape_path: Path, frame_count: int
) -> dict[str, object]:
    shape_payload = load_taco_tensor_pickle(shape_path)
    pose_payload = load_taco_tensor_pickle(pose_path)
    if not isinstance(shape_payload, dict) or set(shape_payload) != {"hand_shape"}:
        raise TacoNativeError("hand-shape pickle schema changed")
    hand_shape = shape_payload["hand_shape"]
    if (
        not isinstance(hand_shape, np.ndarray)
        or hand_shape.shape != (10,)
        or hand_shape.dtype != np.float32
        or not np.isfinite(hand_shape).all()
    ):
        raise TacoNativeError("hand-shape tensor is invalid")
    if not isinstance(pose_payload, dict) or len(pose_payload) != frame_count:
        raise TacoNativeError("hand-pose frame count differs from native timebase")
    expected_keys = [f"{index:05d}" for index in range(1, frame_count + 1)]
    if sorted(pose_payload) != expected_keys:
        raise TacoNativeError("hand-pose frame keys are not contiguous")
    max_abs_translation = 0.0
    for key in expected_keys:
        frame = pose_payload[key]
        if not isinstance(frame, dict) or set(frame) != {"hand_pose", "hand_trans"}:
            raise TacoNativeError("hand-pose frame schema changed")
        pose = frame["hand_pose"]
        translation = frame["hand_trans"]
        if (
            not isinstance(pose, np.ndarray)
            or pose.shape != (48,)
            or pose.dtype != np.float32
            or not isinstance(translation, np.ndarray)
            or translation.shape != (3,)
            or translation.dtype != np.float32
            or not np.isfinite(pose).all()
            or not np.isfinite(translation).all()
        ):
            raise TacoNativeError("hand-pose tensor is invalid")
        max_abs_translation = max(max_abs_translation, float(np.max(np.abs(translation))))
    return {
        "frame_count": frame_count,
        "shape_l2": float(np.linalg.norm(hand_shape.astype(np.float64))),
        "max_abs_translation_m": max_abs_translation,
    }


def _load_obj_vertices(path: Path) -> tuple[np.ndarray, int]:
    vertices: list[list[float]] = []
    face_indices: list[list[int]] = []
    face_count = 0
    with path.open("r", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if line.startswith("v "):
                fields = line.split()
                if len(fields) < 4:
                    raise TacoNativeError("OBJ vertex is malformed")
                vertex = [float(fields[1]), float(fields[2]), float(fields[3])]
                if not np.isfinite(vertex).all():
                    raise TacoNativeError("OBJ vertex is non-finite")
                vertices.append(vertex)
            elif line.startswith("f "):
                fields = line.split()[1:]
                if len(fields) < 3:
                    raise TacoNativeError("OBJ face is malformed")
                indices = []
                for field in fields:
                    index_text = field.split("/", 1)[0]
                    if not index_text or int(index_text) == 0:
                        raise TacoNativeError("OBJ face index is invalid")
                    indices.append(int(index_text))
                face_indices.append(indices)
                face_count += 1
    vertex_array = np.asarray(vertices, dtype=np.float64)
    if vertex_array.shape[0] < 3 or vertex_array.shape[1:] != (3,) or face_count < 1:
        raise TacoNativeError("OBJ mesh is empty")
    for face in face_indices:
        for index in face:
            resolved = index - 1 if index > 0 else vertex_array.shape[0] + index
            if resolved < 0 or resolved >= vertex_array.shape[0]:
                raise TacoNativeError("OBJ face index is out of range")
    return vertex_array, face_count


def _validate_obj(path: Path) -> dict[str, object]:
    vertices, face_count = _load_obj_vertices(path)
    return {
        "vertex_count": int(vertices.shape[0]),
        "face_count": face_count,
        "bounds_cm": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
    }


def _event_centers(frame_count: int) -> list[int]:
    return [math.floor(value * (frame_count - 1) + 0.5) for value in (0.1, 0.3, 0.5, 0.7, 0.9)]


def reconstruct_taco_mano(
    model: ManoModel,
    pose_48: np.ndarray,
    betas: np.ndarray,
    translation_m: np.ndarray,
) -> np.ndarray:
    """Reconstruct one TACO hand with the exact official ManoLayer semantics.

    TACO calls MANO with ``flat_hand_mean=True`` and ``center_idx=0``, then
    converts the layer's millimetre return value back to metres and adds the
    recorded world-space hand translation.  The MANO source model itself is
    metrically scaled, so the two explicit 1000 factors cancel.
    """

    pose = np.asarray(pose_48)
    shape = np.asarray(betas)
    translation = np.asarray(translation_m)
    if (
        pose.dtype.kind != "f"
        or pose.shape != (48,)
        or shape.dtype.kind != "f"
        or shape.shape != (10,)
        or translation.dtype.kind != "f"
        or translation.shape != (3,)
        or not np.isfinite(pose).all()
        or not np.isfinite(shape).all()
        or not np.isfinite(translation).all()
    ):
        raise TacoNativeError("TACO MANO parameter contract violated")
    pose64 = pose.astype(np.float64, copy=False)
    shape64 = shape.astype(np.float64, copy=False)
    translation64 = translation.astype(np.float64, copy=False)

    # ``reconstruct_mano`` implements the canonical MANO LBS and adds the
    # stored hand mean.  TACO requests a flat mean, hence the explicit
    # cancellation below before evaluating the same LBS.
    uncentered = reconstruct_mano(
        model,
        pose64[:3],
        pose64[3:] - model.hand_mean,
        shape64,
        np.zeros(3, dtype=np.float64),
    )
    shaped = model.v_template + np.tensordot(
        model.shapedirs, shape64, axes=([2], [0])
    )
    wrist = model.joint_regressor[0] @ shaped
    vertices = uncentered - wrist[None, :] + translation64[None, :]
    if vertices.shape != (778, 3) or not np.isfinite(vertices).all():
        raise TacoNativeError("TACO MANO reconstruction produced invalid vertices")
    return vertices


def _sequence_paths(
    data_root: Path,
    entry: TacoSplitEntry,
    receipt: Mapping[str, Mapping[str, object]],
) -> dict[str, Path]:
    prefix = f"{entry.triplet}/{entry.sequence_name}"
    return {
        key: data_root / key
        for key in receipt
        if f"/{prefix}/" in key
    }


def _only_suffix(paths: Mapping[str, Path], suffix: str) -> Path:
    matches = [path for key, path in paths.items() if key.endswith("/" + suffix)]
    if len(matches) != 1:
        raise TacoNativeError(f"expected exactly one landed {suffix}")
    return matches[0]


def _hand_parameters(
    pose_path: Path, shape_path: Path, frame_count: int
) -> tuple[np.ndarray, np.ndarray]:
    shape_payload = load_taco_tensor_pickle(shape_path)
    pose_payload = load_taco_tensor_pickle(pose_path)
    if not isinstance(shape_payload, dict) or set(shape_payload) != {"hand_shape"}:
        raise TacoNativeError("hand-shape pickle schema changed during geometry pass")
    shape = shape_payload["hand_shape"]
    expected_keys = [f"{index:05d}" for index in range(1, frame_count + 1)]
    if not isinstance(pose_payload, dict) or sorted(pose_payload) != expected_keys:
        raise TacoNativeError("hand-pose keys changed during geometry pass")
    poses = []
    translations = []
    for key in expected_keys:
        frame = pose_payload[key]
        if not isinstance(frame, dict) or set(frame) != {"hand_pose", "hand_trans"}:
            raise TacoNativeError("hand-pose frame schema changed during geometry pass")
        poses.append(frame["hand_pose"])
        translations.append(frame["hand_trans"])
    pose_array = np.asarray(poses)
    translation_array = np.asarray(translations)
    if (
        not isinstance(shape, np.ndarray)
        or shape.shape != (10,)
        or shape.dtype != np.float32
        or pose_array.shape != (frame_count, 48)
        or pose_array.dtype != np.float32
        or translation_array.shape != (frame_count, 3)
        or translation_array.dtype != np.float32
        or not np.isfinite(shape).all()
        or not np.isfinite(pose_array).all()
        or not np.isfinite(translation_array).all()
    ):
        raise TacoNativeError("hand parameters changed during geometry pass")
    return shape, np.concatenate((pose_array, translation_array), axis=1)


def validate_native_contact_geometry(
    data_root: Path,
    entries: Sequence[TacoSplitEntry],
    receipt: Mapping[str, Mapping[str, object]],
    mano_root: Path,
) -> dict[str, object]:
    """Evaluate the frozen 300-event native hand/object contact reference."""

    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:  # pragma: no cover - runner environment contract
        raise TacoNativeError("SciPy is required for exact vertex nearest-neighbour search") from exc

    models = {
        side: load_mano_model(mano_root / f"MANO_{side.upper()}.pkl", side)
        for side in ("left", "right")
    }
    object_cache: dict[str, np.ndarray] = {}
    events: list[dict[str, object]] = []
    pairing_frame_count = 0
    strict_contact_pairing_frames = 0
    global_minimum = float("inf")

    for entry in entries:
        paths = _sequence_paths(data_root, entry, receipt)
        object_pose_paths = {
            path.stem.split("_", 1)[0]: path
            for key, path in paths.items()
            if "/Object_Poses/" in "/" + key and key.endswith(".npy")
        }
        if set(object_pose_paths) != {"tool", "target"}:
            raise TacoNativeError(f"{entry.sequence_id}: tool/target pose files changed")
        pose_arrays = {
            role: _load_float32_array(path, (4, 4))
            for role, path in object_pose_paths.items()
        }
        frame_count = int(pose_arrays["tool"].shape[0])
        if pose_arrays["target"].shape[0] != frame_count:
            raise TacoNativeError(f"{entry.sequence_id}: object pose frame counts differ")

        object_ids: dict[str, str] = {}
        for role, path in object_pose_paths.items():
            object_id = path.stem.split("_", 1)[1]
            object_ids[role] = object_id
            if object_id not in object_cache:
                model_matches = [
                    data_root / key
                    for key in receipt
                    if key.endswith("/" + f"{object_id}_cm.obj")
                ]
                if len(model_matches) != 1:
                    raise TacoNativeError(
                        f"{entry.sequence_id}: object model {object_id} is ambiguous"
                    )
                model_path = model_matches[0]
                vertices_cm, _ = _load_obj_vertices(model_path)
                object_cache[object_id] = vertices_cm * 0.01

        hand_data = {}
        for side in ("left", "right"):
            hand_data[side] = _hand_parameters(
                _only_suffix(paths, f"{side}_hand.pkl"),
                _only_suffix(paths, f"{side}_hand_shape.pkl"),
                frame_count,
            )

        centers = _event_centers(frame_count)
        selected_frames = sorted(
            {frame for center in centers for frame in range(center - 7, center + 8)}
        )
        if selected_frames[0] < 0 or selected_frames[-1] >= frame_count:
            raise TacoNativeError(f"{entry.sequence_id}: event window is out of range")
        distances: dict[tuple[str, str], dict[int, float]] = {
            (side, role): {}
            for side in ("left", "right")
            for role in ("tool", "target")
        }
        for frame in selected_frames:
            hand_vertices = {}
            for side in ("left", "right"):
                shape, parameters = hand_data[side]
                hand_vertices[side] = reconstruct_taco_mano(
                    models[side],
                    parameters[frame, :48],
                    shape,
                    parameters[frame, 48:],
                )
            for role in ("tool", "target"):
                transform = np.asarray(pose_arrays[role][frame], dtype=np.float64)
                rotation = transform[:3, :3]
                translation = transform[:3, 3]
                object_world = (
                    object_cache[object_ids[role]] @ rotation.T
                    + translation[None, :]
                )
                tree = cKDTree(object_world)
                for side in ("left", "right"):
                    nearest, _ = tree.query(hand_vertices[side], k=1, workers=1)
                    minimum = float(np.min(nearest))
                    if not np.isfinite(minimum) or minimum < 0:
                        raise TacoNativeError("native vertex distance is invalid")
                    distances[(side, role)][frame] = minimum
                    pairing_frame_count += 1
                    global_minimum = min(global_minimum, minimum)
                    if minimum < NATIVE_CONTACT_DISTANCE_M:
                        strict_contact_pairing_frames += 1

        for event_index, center in enumerate(centers, start=1):
            frames = list(range(center - 7, center + 8))
            pairings = {}
            for side in ("left", "right"):
                for role in ("tool", "target"):
                    values = [distances[(side, role)][frame] for frame in frames]
                    contacts = [value < NATIVE_CONTACT_DISTANCE_M for value in values]
                    pairings[f"{side}_{role}"] = {
                        "minimum_vertex_distance_m": min(values),
                        "strict_contact_any": any(contacts),
                        "strict_contact_frame_count": sum(contacts),
                        "per_frame_minimum_vertex_distance_m": values,
                    }
            events.append(
                {
                    "rank": entry.rank,
                    "split": entry.split,
                    "sequence_id": entry.sequence_id,
                    "event_index": event_index,
                    "center_frame": center,
                    "frame_indices": frames,
                    "object_ids": object_ids,
                    "pairings": pairings,
                    "provenance": "derived_from_taco_native_mano_object_geometry",
                }
            )

    expected_pairing_frames = len(entries) * 5 * 15 * 2 * 2
    closed = (
        len(entries) == 60
        and len(events) == 300
        and pairing_frame_count == expected_pairing_frames
        and np.isfinite(global_minimum)
    )
    if not closed:
        raise TacoNativeError("native contact geometry coverage did not close")
    return {
        "closed": True,
        "event_count": len(events),
        "pairing_frame_count": pairing_frame_count,
        "strict_contact_pairing_frame_count": strict_contact_pairing_frames,
        "strict_contact_comparator": "<",
        "strict_contact_distance_m": NATIVE_CONTACT_DISTANCE_M,
        "minimum_observed_vertex_distance_m": global_minimum,
        "object_model_count": len(object_cache),
        "mano_model_sha256": {
            side: models[side].source_sha256 for side in ("left", "right")
        },
        "events": events,
    }


def validate_sequence_native(
    data_root: Path,
    entry: TacoSplitEntry,
    receipt: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    paths = _sequence_paths(data_root, entry, receipt)
    required_suffixes = {
        "color.mp4": 1,
        "egocentric_depth.avi": 1,
        "egocentric_frame_extrinsic.npy": 1,
        "egocentric_intrinsic.txt": 1,
        "left_hand.pkl": 1,
        "left_hand_shape.pkl": 1,
        "right_hand.pkl": 1,
        "right_hand_shape.pkl": 1,
    }
    selected: dict[str, Path] = {}
    for suffix, expected_count in required_suffixes.items():
        matches = [path for key, path in paths.items() if key.endswith("/" + suffix)]
        if len(matches) != expected_count:
            raise TacoNativeError(f"{entry.sequence_id}: missing {suffix}")
        selected[suffix] = matches[0]
    object_pose_paths = [
        path
        for key, path in paths.items()
        if "/Object_Poses/" in "/" + key and key.endswith(".npy")
    ]
    segmentation_paths = [
        path
        for key, path in paths.items()
        if key.startswith("2D_Segmentation/") and key.endswith("_masks.npy")
    ]
    if len(object_pose_paths) != 2 or len(segmentation_paths) != 12:
        raise TacoNativeError(f"{entry.sequence_id}: native pose/segmentation count changed")

    rgb = parse_mp4_metadata(selected["color.mp4"])
    depth = parse_avi_metadata(selected["egocentric_depth.avi"])
    try:
        _validate_rgbd_timebase(rgb, depth)
    except TacoNativeError as exc:
        raise TacoNativeError(f"{entry.sequence_id}: {exc}") from exc
    frame_count = rgb.frame_count

    camera = _load_float32_array(
        selected["egocentric_frame_extrinsic.npy"], (4, 4)
    )
    object_poses = [
        _load_float32_array(path, (4, 4)) for path in sorted(object_pose_paths)
    ]
    if camera.shape[0] != frame_count or any(item.shape[0] != frame_count for item in object_poses):
        raise TacoNativeError(f"{entry.sequence_id}: pose timebase differs from RGB-D")
    max_transform_residual = max(
        [_validate_transform_batch(camera)]
        + [_validate_transform_batch(item) for item in object_poses]
    )
    intrinsic = np.loadtxt(selected["egocentric_intrinsic.txt"], dtype=np.float64)
    if (
        intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
        or intrinsic[0, 0] <= 0
        or intrinsic[1, 1] <= 0
        or not np.allclose(intrinsic[2], [0, 0, 1], atol=1e-8)
    ):
        raise TacoNativeError(f"{entry.sequence_id}: camera intrinsic is invalid")

    left = _validate_hand_pickle(
        selected["left_hand.pkl"], selected["left_hand_shape.pkl"], frame_count
    )
    right = _validate_hand_pickle(
        selected["right_hand.pkl"], selected["right_hand_shape.pkl"], frame_count
    )
    segmentation_counts = []
    segmentation_cameras = []
    for path in sorted(segmentation_paths):
        shape, fortran, dtype = _npy_header(path)
        if (
            fortran
            or dtype != np.dtype("uint8")
            or len(shape) != 3
            or shape[0] <= 0
            or shape[1:] != (SEGMENTATION_HEIGHT, SEGMENTATION_WIDTH)
        ):
            raise TacoNativeError(f"{entry.sequence_id}: segmentation header is invalid")
        segmentation_counts.append(shape[0])
        segmentation_cameras.append(path.name.removesuffix("_masks.npy"))

    centers = _event_centers(frame_count)
    event_validity = [center - 7 >= 0 and center + 7 < frame_count for center in centers]
    return {
        "rank": entry.rank,
        "split": entry.split,
        "sequence_id": entry.sequence_id,
        "frame_count": frame_count,
        "rgb": asdict(rgb),
        "depth": asdict(depth),
        "effective_acquisition_fps": EXPECTED_FPS,
        "depth_container_header_fps_anomaly": True,
        "timebase_basis": "M3-v1.1 exact frame-index alignment; t=i/30 s; depth AVI header 500000/33333 is recorded and never resampled",
        "event_centers": centers,
        "event_windows_valid": event_validity,
        "camera_intrinsic": intrinsic.tolist(),
        "coordinate_convention": "camera_from_world @ object_from_local",
        "max_transform_residual": max_transform_residual,
        "left_hand": left,
        "right_hand": right,
        "object_pose_files": sorted(path.name for path in object_pose_paths),
        "segmentation_camera_ids": segmentation_cameras,
        "segmentation_frame_counts": segmentation_counts,
        "segmentation_timebase_note": "official allocentric masks; not substituted for egocentric RGB-D alignment",
    }


def verify_receipt_files(
    data_root: Path,
    expected_members: Sequence[Mapping[str, object]],
    completed: Mapping[str, Mapping[str, object]],
) -> None:
    expected_paths = {str(member["path"]) for member in expected_members}
    if set(completed) != expected_paths:
        raise TacoNativeError("landing receipt file set differs from the frozen probe")
    for relative in sorted(expected_paths):
        record = completed[relative]
        path = data_root.joinpath(*Path(relative).parts)
        if not path.is_file() or path.is_symlink():
            raise TacoNativeError("landed TACO member is missing or non-regular")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        if size != record.get("bytes") or digest.hexdigest() != record.get("sha256"):
            raise TacoNativeError("landed TACO member SHA-256 changed")


def validate_object_models(
    data_root: Path, object_members: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    summaries = {}
    for member in object_members:
        path = data_root.joinpath(*Path(str(member["path"])).parts)
        summaries[path.name] = _validate_obj(path)
    return summaries
