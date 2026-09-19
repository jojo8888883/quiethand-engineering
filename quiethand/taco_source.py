"""Selective, archive-bound TACO source discovery for QuietHand M3."""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
import stat
from typing import Callable, Iterable, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import zipfile

from .http_range_zip import HTTPRangeReader
from .taco_contract import (
    TACO_DATA_CAP_BYTES,
    TACO_DROPBOX_LINK_KEY,
    TACO_DROPBOX_LIST_ENDPOINT,
    TACO_DROPBOX_RLKEY,
    TACO_DROPBOX_SECURE_HASH,
    TACO_REQUIRED_ARCHIVES,
    TacoContractError,
    TacoSplitEntry,
)


MAX_LISTING_BYTES = 2 * 1024**2
MAX_ARCHIVE_MEMBERS = 100_000
MAX_SELECTED_MEMBER_BYTES = 4 * 1024**3
MAX_SELECTED_COMPRESSION_RATIO = 2_000.0
SEQUENCE_ARCHIVES = (
    "2D_Segmentation.zip",
    "Egocentric_Camera_Parameters.zip",
    "Egocentric_Depth_Videos.zip",
    "Egocentric_RGB_Videos.zip",
    "Hand_Poses.zip",
    "Object_Poses.zip",
)

_EXPECTED_SEQUENCE_FILES = {
    "2D_Segmentation.zip": 12,
    "Egocentric_Camera_Parameters.zip": 2,
    "Egocentric_Depth_Videos.zip": 1,
    "Egocentric_RGB_Videos.zip": 1,
    "Hand_Poses.zip": 4,
    "Object_Poses.zip": 2,
}


class TacoSourceError(RuntimeError):
    """The remote TACO source is incomplete, ambiguous, or over budget."""


def fetch_dropbox_listing(
    *, opener: Callable[..., object] = urlopen
) -> tuple[bytes, Mapping[str, object]]:
    """Fetch the exact public shared-folder listing used by Dropbox's page."""

    form = urlencode(
        {
            "t": "T",
            "link_type": "c",
            "link_key": TACO_DROPBOX_LINK_KEY,
            "secure_hash": TACO_DROPBOX_SECURE_HASH,
            "sub_path": "",
            "rlkey": TACO_DROPBOX_RLKEY,
        }
    ).encode("ascii")
    request = Request(
        TACO_DROPBOX_LIST_ENDPOINT,
        data=form,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Cookie": "t=T",
            "User-Agent": "QuietHand-M3/1.0",
        },
    )
    with opener(request, timeout=60) as response:
        if getattr(response, "status", None) != 200:
            raise TacoSourceError("Dropbox listing request was not HTTP 200")
        content_type = response.headers.get_content_type().lower()
        if content_type != "application/json":
            raise TacoSourceError("Dropbox listing response is not JSON")
        declared = response.headers.get("Content-Length")
        if declared and declared.isdigit() and int(declared) > MAX_LISTING_BYTES:
            raise TacoSourceError("Dropbox listing exceeds its response cap")
        payload = response.read(MAX_LISTING_BYTES + 1)
    if not payload or len(payload) > MAX_LISTING_BYTES:
        raise TacoSourceError("Dropbox listing is empty or over its response cap")
    try:
        parsed = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TacoSourceError("Dropbox listing JSON is malformed") from exc
    if not isinstance(parsed, dict):
        raise TacoSourceError("Dropbox listing root is not an object")
    return payload, parsed


def bind_required_archives(
    listing: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    entries = listing.get("entries")
    if not isinstance(entries, list) or listing.get("has_more_entries") is not False:
        raise TacoSourceError("Dropbox top-level listing is incomplete")
    by_name: dict[str, dict[str, object]] = {}
    for raw in entries:
        if not isinstance(raw, dict) or not isinstance(raw.get("filename"), str):
            raise TacoSourceError("Dropbox listing contains an invalid entry")
        name = raw["filename"]
        if name in by_name:
            raise TacoSourceError("Dropbox listing contains duplicate names")
        by_name[name] = raw

    bound: dict[str, dict[str, object]] = {}
    for name in TACO_REQUIRED_ARCHIVES:
        raw = by_name.get(name)
        if raw is None or raw.get("is_dir") is not False:
            raise TacoSourceError(f"required TACO archive is missing: {name}")
        if not isinstance(raw.get("bytes"), int) or raw["bytes"] <= 0:
            raise TacoSourceError(f"required TACO archive has invalid size: {name}")
        if not isinstance(raw.get("href"), str) or not raw["href"].startswith(
            "https://www.dropbox.com/scl/fo/"
        ):
            raise TacoSourceError(f"required TACO archive has invalid href: {name}")
        if not isinstance(raw.get("file_id"), str) or not isinstance(
            raw.get("revision_id"), str
        ):
            raise TacoSourceError(f"required TACO archive lacks immutable identity: {name}")
        bound[name] = {
            "filename": name,
            "bytes": raw["bytes"],
            "href": raw["href"],
            "file_id": raw["file_id"],
            "revision_id": raw["revision_id"],
        }
    return bound


def _safe_file_info(info: zipfile.ZipInfo) -> bool:
    name = info.filename
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or any(ord(character) < 32 for character in name)
        or path.is_absolute()
        or ".." in path.parts
    ):
        return False
    mode = info.external_attr >> 16
    if stat.S_IFMT(mode) == stat.S_IFLNK:
        return False
    return not info.is_dir()


def _sequence_key_in_path(
    path: PurePosixPath, entries: Sequence[TacoSplitEntry]
) -> TacoSplitEntry | None:
    parts = path.parts
    for entry in entries:
        for index in range(len(parts) - 1):
            if parts[index] == entry.triplet and parts[index + 1] == entry.sequence_name:
                return entry
    return None


def catalog_line(info: zipfile.ZipInfo) -> bytes:
    return (
        json.dumps(
            [
                info.filename,
                info.CRC,
                info.compress_size,
                info.file_size,
                info.header_offset,
                info.compress_type,
                info.flag_bits,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def catalog_sequence_archive(
    archive: Mapping[str, object],
    entries: Sequence[TacoSplitEntry],
) -> dict[str, object]:
    """Byte-range read one ZIP catalog and bind only the frozen 60 sequences."""

    name = str(archive["filename"])
    reader = HTTPRangeReader(
        str(archive["href"]).replace("dl=0", "dl=1"),
        int(archive["bytes"]),
    )
    digest = hashlib.sha256()
    seen: set[str] = set()
    selected: list[dict[str, object]] = []
    selected_by_rank = {entry.rank: 0 for entry in entries}
    total_compressed = 0
    total_uncompressed = 0
    with zipfile.ZipFile(reader) as handle:
        infos = handle.infolist()
        if len(infos) > MAX_ARCHIVE_MEMBERS:
            raise TacoSourceError(f"{name}: archive member count exceeds cap")
        for info in infos:
            if info.filename in seen:
                raise TacoSourceError(f"{name}: duplicate ZIP member")
            seen.add(info.filename)
            digest.update(catalog_line(info))
            if info.file_size < 0 or info.compress_size < 0:
                raise TacoSourceError(f"{name}: negative member size")
            total_compressed += info.compress_size
            total_uncompressed += info.file_size
            if info.is_dir():
                continue
            if not _safe_file_info(info):
                raise TacoSourceError(f"{name}: unsafe ZIP member")
            split_entry = _sequence_key_in_path(PurePosixPath(info.filename), entries)
            if split_entry is None:
                continue
            if info.flag_bits & 0x1:
                raise TacoSourceError(f"{name}: encrypted selected member")
            if info.file_size > MAX_SELECTED_MEMBER_BYTES:
                raise TacoSourceError(f"{name}: selected member exceeds size cap")
            ratio = info.file_size / max(1, info.compress_size)
            if ratio > MAX_SELECTED_COMPRESSION_RATIO:
                raise TacoSourceError(f"{name}: selected member compression ratio is unsafe")
            selected_by_rank[split_entry.rank] += 1
            selected.append(
                {
                    "rank": split_entry.rank,
                    "split": split_entry.split,
                    "sequence_id": split_entry.sequence_id,
                    "path": info.filename,
                    "crc32": f"{info.CRC:08x}",
                    "compressed_bytes": info.compress_size,
                    "uncompressed_bytes": info.file_size,
                    "header_offset": info.header_offset,
                    "compression": info.compress_type,
                    "flag_bits": info.flag_bits,
                }
            )

    expected = _EXPECTED_SEQUENCE_FILES[name]
    bad = [rank for rank, count in selected_by_rank.items() if count != expected]
    if bad:
        raise TacoSourceError(
            f"{name}: ranks with unexpected selected-file count: {bad[:8]}"
        )
    selected.sort(key=lambda item: (int(item["rank"]), str(item["path"])))
    return {
        **archive,
        "member_count": len(seen),
        "catalog_sha256": digest.hexdigest(),
        "catalog_total_compressed_bytes": total_compressed,
        "catalog_total_uncompressed_bytes": total_uncompressed,
        "catalog_range_requests": reader.requests_made,
        "catalog_range_bytes": reader.bytes_fetched,
        "selected_members": selected,
    }


def object_ids_from_pose_catalog(pose_catalog: Mapping[str, object]) -> tuple[str, ...]:
    identifiers: set[str] = set()
    members = pose_catalog.get("selected_members")
    if not isinstance(members, list):
        raise TacoSourceError("object-pose catalog lacks selected members")
    for member in members:
        filename = PurePosixPath(str(member["path"])).name
        if not filename.endswith(".npy") or "_" not in filename:
            raise TacoSourceError("object-pose member name is malformed")
        role, identifier = filename[:-4].split("_", 1)
        if role not in {"tool", "target"} or not identifier.isdigit():
            raise TacoSourceError("object-pose member identity is malformed")
        identifiers.add(identifier)
    if not identifiers:
        raise TacoSourceError("no object identities were selected")
    return tuple(sorted(identifiers))


def catalog_object_models(
    archive: Mapping[str, object], object_ids: Iterable[str]
) -> dict[str, object]:
    name = str(archive["filename"])
    expected_names = {f"{identifier}_cm.obj" for identifier in object_ids}
    reader = HTTPRangeReader(
        str(archive["href"]).replace("dl=0", "dl=1"),
        int(archive["bytes"]),
    )
    digest = hashlib.sha256()
    seen: set[str] = set()
    selected: list[dict[str, object]] = []
    total_compressed = 0
    total_uncompressed = 0
    with zipfile.ZipFile(reader) as handle:
        infos = handle.infolist()
        if len(infos) > MAX_ARCHIVE_MEMBERS:
            raise TacoSourceError(f"{name}: archive member count exceeds cap")
        for info in infos:
            if info.filename in seen:
                raise TacoSourceError(f"{name}: duplicate ZIP member")
            seen.add(info.filename)
            digest.update(catalog_line(info))
            total_compressed += info.compress_size
            total_uncompressed += info.file_size
            if info.is_dir():
                continue
            if not _safe_file_info(info):
                raise TacoSourceError(f"{name}: unsafe ZIP member")
            if PurePosixPath(info.filename).name not in expected_names:
                continue
            selected.append(
                {
                    "path": info.filename,
                    "crc32": f"{info.CRC:08x}",
                    "compressed_bytes": info.compress_size,
                    "uncompressed_bytes": info.file_size,
                    "header_offset": info.header_offset,
                    "compression": info.compress_type,
                    "flag_bits": info.flag_bits,
                }
            )
    observed_names = {PurePosixPath(str(item["path"])).name for item in selected}
    if observed_names != expected_names or len(selected) != len(expected_names):
        raise TacoSourceError("Object_Models.zip does not close selected object IDs")
    selected.sort(key=lambda item: str(item["path"]))
    return {
        **archive,
        "member_count": len(seen),
        "catalog_sha256": digest.hexdigest(),
        "catalog_total_compressed_bytes": total_compressed,
        "catalog_total_uncompressed_bytes": total_uncompressed,
        "catalog_range_requests": reader.requests_made,
        "catalog_range_bytes": reader.bytes_fetched,
        "selected_members": selected,
    }


def selected_member_manifest_sha256(catalogs: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for catalog in sorted(catalogs, key=lambda item: str(item["filename"])):
        for member in catalog["selected_members"]:
            digest.update(
                (
                    json.dumps(
                        [catalog["filename"], member],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
            )
    return digest.hexdigest()


def validate_selected_budget(catalogs: Sequence[Mapping[str, object]]) -> int:
    selected_bytes = sum(
        int(member["uncompressed_bytes"])
        for catalog in catalogs
        for member in catalog["selected_members"]
    )
    if selected_bytes <= 0 or selected_bytes > TACO_DATA_CAP_BYTES:
        raise TacoContractError("selected TACO members violate the 80 GiB cap")
    return selected_bytes
