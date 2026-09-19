"""Frozen TACO V1 selection and source contract for QuietHand M3."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Sequence


TACO_RELEASE_ID = "TACO_V1"
TACO_REPOSITORY = "https://github.com/leolyliu/TACO-Instructions"
TACO_SOURCE_COMMIT = "b385ac1e89f35214bfbbef0d63e6df0ac8dac02e"
TACO_SEQUENCE_LIST = "data_lists/v1_egocentric_data_available_sequences.txt"
TACO_SEQUENCE_LIST_BYTES = 79_460
TACO_SEQUENCE_LIST_SHA256 = (
    "8c6b15a272e93efa3af0d0d58f79a9fb0e4aeffc2b4ff8c09ef9278aced63354"
)
TACO_SELECTION_SHA256 = (
    "532f013f3c00a3ae48868ce99aa4304e76e9ddae164ef5864a74b798bfa423e1"
)
TACO_CALIBRATION_IDS_SHA256 = (
    "b44138fb33bf5a3ba16639c43a0e79c4e6202c335e1e8bd3d0708912438e221d"
)
TACO_EVALUATION_IDS_SHA256 = (
    "d52327d3c30c1744b66e0792655072ec5a13a61ceec74157dd3c54637d324813"
)

TACO_DROPBOX_LINK = (
    "https://www.dropbox.com/scl/fo/8w7xir110nbcnq8uo1845/"
    "AOaHUxGEcR0sWvfmZRQQk9g?rlkey=xnhajvn71ua5i23w75la1nidx&dl=0"
)
TACO_DROPBOX_LIST_ENDPOINT = (
    "https://www.dropbox.com/list_shared_link_folder_entries"
)
TACO_DROPBOX_LINK_KEY = "8w7xir110nbcnq8uo1845"
TACO_DROPBOX_SECURE_HASH = "AOaHUxGEcR0sWvfmZRQQk9g"
TACO_DROPBOX_RLKEY = "xnhajvn71ua5i23w75la1nidx"

TACO_DATA_CAP_BYTES = 80 * 1024**3
TACO_REQUIRED_ARCHIVES = (
    "2D_Segmentation.zip",
    "Egocentric_Camera_Parameters.zip",
    "Egocentric_Depth_Videos.zip",
    "Egocentric_RGB_Videos.zip",
    "Hand_Poses.zip",
    "Object_Models.zip",
    "Object_Poses.zip",
)

_SEQUENCE_PATTERN = re.compile(
    r"^(?P<triplet>\(.+\)) (?P<sequence>\d{8}_\d+)$"
)


class TacoContractError(RuntimeError):
    """The frozen TACO source or selection does not satisfy the M3 contract."""


@dataclass(frozen=True)
class TacoSplitEntry:
    rank: int
    split: str
    rank_hash: str
    sequence_id: str
    triplet: str
    sequence_name: str


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_frozen_sequence_ids(path: Path) -> tuple[str, ...]:
    """Read and byte-bind the exact official full-modality sequence list."""

    path = Path(path)
    payload = path.read_bytes()
    if len(payload) != TACO_SEQUENCE_LIST_BYTES:
        raise TacoContractError("TACO sequence-list byte length changed")
    if sha256_bytes(payload) != TACO_SEQUENCE_LIST_SHA256:
        raise TacoContractError("TACO sequence-list SHA-256 changed")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TacoContractError("TACO sequence list is not UTF-8") from exc
    sequence_ids = tuple(line.strip() for line in text.splitlines() if line.strip())
    if len(sequence_ids) != 2_212:
        raise TacoContractError("TACO eligible sequence count changed")
    if len(set(sequence_ids)) != len(sequence_ids):
        raise TacoContractError("TACO sequence list contains duplicates")
    for sequence_id in sequence_ids:
        if _SEQUENCE_PATTERN.fullmatch(sequence_id) is None:
            raise TacoContractError(f"invalid TACO sequence id: {sequence_id!r}")
    return sequence_ids


def deterministic_taco_split(
    sequence_ids: Sequence[str],
) -> tuple[TacoSplitEntry, ...]:
    """Implement the frozen sha256('TACO_V1' || sequence_id) prefix split."""

    if len(set(sequence_ids)) != len(sequence_ids):
        raise TacoContractError("duplicate sequence ids would leak across splits")
    ranked = sorted(
        (
            hashlib.sha256((TACO_RELEASE_ID + item).encode("utf-8")).hexdigest(),
            item,
        )
        for item in sequence_ids
    )[:60]
    result: list[TacoSplitEntry] = []
    for zero_rank, (rank_hash, sequence_id) in enumerate(ranked):
        match = _SEQUENCE_PATTERN.fullmatch(sequence_id)
        if match is None:
            raise TacoContractError(f"invalid TACO sequence id: {sequence_id!r}")
        result.append(
            TacoSplitEntry(
                rank=zero_rank + 1,
                split="calibration" if zero_rank < 30 else "evaluation",
                rank_hash=rank_hash,
                sequence_id=sequence_id,
                triplet=match.group("triplet"),
                sequence_name=match.group("sequence"),
            )
        )
    if len(result) != 60:
        raise TacoContractError("fewer than 60 TACO sequences are available")
    return tuple(result)


def canonical_selection_bytes(entries: Sequence[TacoSplitEntry]) -> bytes:
    payload = "".join(
        f"{entry.rank}\t{entry.split}\t{entry.rank_hash}\t{entry.sequence_id}\n"
        for entry in entries
    ).encode("utf-8")
    if sha256_bytes(payload) != TACO_SELECTION_SHA256:
        raise TacoContractError("TACO 60-clip canonical manifest SHA-256 changed")
    calibration = "".join(
        f"{entry.sequence_id}\n" for entry in entries if entry.split == "calibration"
    ).encode("utf-8")
    evaluation = "".join(
        f"{entry.sequence_id}\n" for entry in entries if entry.split == "evaluation"
    ).encode("utf-8")
    if sha256_bytes(calibration) != TACO_CALIBRATION_IDS_SHA256:
        raise TacoContractError("TACO calibration ID-list SHA-256 changed")
    if sha256_bytes(evaluation) != TACO_EVALUATION_IDS_SHA256:
        raise TacoContractError("TACO evaluation ID-list SHA-256 changed")
    return payload
