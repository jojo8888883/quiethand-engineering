"""Restricted decoder for the exact tensor-only pickle format used by TACO."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import io
import math
from pathlib import Path
import pickle
import struct
from typing import Any

import numpy as np


TORCH_MAGIC_NUMBER = 0x1950A86A20F9469CFC6C
TORCH_PROTOCOL_VERSION = 1001
MAX_TACO_PICKLE_BYTES = 64 * 1024**2
MAX_TENSOR_ELEMENTS = 10_000_000


class SafeTorchPickleError(RuntimeError):
    """A TACO pickle is outside the frozen tensor-only decoding contract."""


class _FloatStorageKind:
    dtype = np.dtype("<f4")


@dataclass
class _SafeStorage:
    key: str
    elements: int
    array: np.ndarray | None = None


class _NestedTorchUnpickler(pickle.Unpickler):
    def __init__(self, stream: io.BytesIO, storages: dict[str, _SafeStorage]):
        super().__init__(stream)
        self._storages = storages

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("torch", "FloatStorage"):
            return _FloatStorageKind
        raise SafeTorchPickleError(f"forbidden nested pickle global: {module}.{name}")

    def persistent_load(self, persistent_id: object) -> _SafeStorage:
        if not isinstance(persistent_id, tuple) or len(persistent_id) != 6:
            raise SafeTorchPickleError("malformed torch storage persistent ID")
        kind, storage_type, key, location, elements, view = persistent_id
        if (
            kind != "storage"
            or storage_type is not _FloatStorageKind
            or not isinstance(key, str)
            or location != "cpu"
            or not isinstance(elements, int)
            or elements < 0
            or elements > MAX_TENSOR_ELEMENTS
            or view is not None
        ):
            raise SafeTorchPickleError("unsupported torch storage descriptor")
        storage = self._storages.get(key)
        if storage is None:
            storage = _SafeStorage(key=key, elements=elements)
            self._storages[key] = storage
        elif storage.elements != elements:
            raise SafeTorchPickleError("torch storage size changed within one payload")
        return storage


def _nested_load(stream: io.BytesIO, storages: dict[str, _SafeStorage]) -> object:
    return _NestedTorchUnpickler(stream, storages).load()


def _safe_load_storage(payload: bytes) -> _SafeStorage:
    if not payload or len(payload) > MAX_TACO_PICKLE_BYTES:
        raise SafeTorchPickleError("embedded torch storage payload exceeds its cap")
    stream = io.BytesIO(payload)
    storages: dict[str, _SafeStorage] = {}
    magic = _nested_load(stream, storages)
    protocol = _nested_load(stream, storages)
    system_info = _nested_load(stream, storages)
    root = _nested_load(stream, storages)
    keys = _nested_load(stream, storages)
    if magic != TORCH_MAGIC_NUMBER or protocol != TORCH_PROTOCOL_VERSION:
        raise SafeTorchPickleError("embedded torch storage header changed")
    if not isinstance(system_info, dict) or system_info.get("little_endian") is not True:
        raise SafeTorchPickleError("embedded torch storage endian contract changed")
    if system_info.get("type_sizes") != {"short": 2, "int": 4, "long": 4}:
        raise SafeTorchPickleError("embedded torch storage type sizes changed")
    if not isinstance(keys, list) or any(not isinstance(key, str) for key in keys):
        raise SafeTorchPickleError("embedded torch storage key list is malformed")
    if set(keys) != set(storages) or len(keys) != len(storages):
        raise SafeTorchPickleError("embedded torch storage keys are incomplete")
    for key in keys:
        storage = storages[key]
        count_bytes = stream.read(8)
        if len(count_bytes) != 8:
            raise SafeTorchPickleError("embedded torch storage count is truncated")
        observed_count = struct.unpack("<Q", count_bytes)[0]
        if observed_count != storage.elements:
            raise SafeTorchPickleError("embedded torch storage count changed")
        byte_count = storage.elements * _FloatStorageKind.dtype.itemsize
        raw = stream.read(byte_count)
        if len(raw) != byte_count:
            raise SafeTorchPickleError("embedded torch storage data is truncated")
        storage.array = np.frombuffer(raw, dtype=_FloatStorageKind.dtype).copy()
    if stream.read(1) != b"":
        raise SafeTorchPickleError("embedded torch storage has trailing data")
    if not isinstance(root, _SafeStorage) or root.array is None:
        raise SafeTorchPickleError("embedded torch storage root is invalid")
    return root


def _safe_rebuild_tensor_v2(
    storage: _SafeStorage,
    storage_offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    requires_grad: bool,
    backward_hooks: OrderedDict,
) -> np.ndarray:
    if (
        not isinstance(storage, _SafeStorage)
        or storage.array is None
        or not isinstance(storage_offset, int)
        or storage_offset < 0
        or not isinstance(size, tuple)
        or not isinstance(stride, tuple)
        or len(size) != len(stride)
        or len(size) > 4
        or any(not isinstance(item, int) or item < 0 for item in size)
        or any(not isinstance(item, int) or item < 0 for item in stride)
        or not isinstance(requires_grad, bool)
        or not isinstance(backward_hooks, OrderedDict)
        or backward_hooks
    ):
        raise SafeTorchPickleError("tensor rebuild arguments are outside contract")
    elements = math.prod(size)
    if elements > MAX_TENSOR_ELEMENTS:
        raise SafeTorchPickleError("tensor element count exceeds its cap")
    maximum = storage_offset
    for dimension, step in zip(size, stride):
        if dimension:
            maximum += (dimension - 1) * step
    if elements and maximum >= storage.array.size:
        raise SafeTorchPickleError("tensor view exceeds its storage")
    if not elements:
        return np.empty(size, dtype=np.float32)
    view = np.ndarray(
        shape=size,
        dtype=storage.array.dtype,
        buffer=storage.array.data,
        offset=storage_offset * storage.array.dtype.itemsize,
        strides=tuple(item * storage.array.dtype.itemsize for item in stride),
    )
    return np.array(view, dtype=np.float32, copy=True)


class _OuterTacoUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        allowed = {
            ("torch._utils", "_rebuild_tensor_v2"): _safe_rebuild_tensor_v2,
            ("torch.storage", "_load_from_bytes"): _safe_load_storage,
            ("collections", "OrderedDict"): OrderedDict,
        }
        value = allowed.get((module, name))
        if value is None:
            raise SafeTorchPickleError(f"forbidden TACO pickle global: {module}.{name}")
        return value

    def persistent_load(self, persistent_id: object) -> object:
        del persistent_id
        raise SafeTorchPickleError("outer TACO pickle may not use persistent IDs")


def load_taco_tensor_pickle(path: Path) -> object:
    """Load one pre-bound official TACO tensor pickle without importing torch."""

    path = Path(path)
    size = path.stat().st_size
    if size <= 0 or size > MAX_TACO_PICKLE_BYTES:
        raise SafeTorchPickleError("TACO pickle size is outside contract")
    with path.open("rb") as handle:
        value = _OuterTacoUnpickler(handle).load()
        if handle.read(1) != b"":
            raise SafeTorchPickleError("TACO pickle contains trailing data")
    return value
