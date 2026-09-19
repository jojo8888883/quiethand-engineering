"""Frozen ARCTIC v1.0 contract and credential-safe M2 preflight helpers.

This module intentionally stops before loading NumPy arrays or reconstructing meshes.
It establishes only the release, file, split, license-attestation, and resource
boundary needed before the data-dependent M2 adapter can run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Mapping, Sequence


ARCTIC_REPOSITORY = "https://github.com/zc-alexfan/arctic"
ARCTIC_SOURCE_COMMIT = "49f3eb4b7f01663172576357bb3d95695d7a7689"
ARCTIC_RELEASE_VERSION = "v1_0"
ARCTIC_RELEASE_ID = f"arctic-{ARCTIC_RELEASE_VERSION}@{ARCTIC_SOURCE_COMMIT}"
ARCTIC_LICENSE_URL = (
    f"{ARCTIC_REPOSITORY}/blob/{ARCTIC_SOURCE_COMMIT}/LICENSE"
)
ARCTIC_REGISTRATION_URL = "https://arctic.is.tue.mpg.de/register.php"
ARCTIC_DATA_DOC_URL = (
    f"{ARCTIC_REPOSITORY}/blob/{ARCTIC_SOURCE_COMMIT}/docs/data/data_doc.md"
)
ARCTIC_PAPER_URL = "https://download.is.tue.mpg.de/arctic/arctic_april_24.pdf"

M2_DATA_CAP_BYTES = 80 * 1024**3
WORKSPACE_WARNING_BYTES = 225 * 1024**3
WORKSPACE_HARD_CAP_BYTES = 250 * 1024**3
CALIBRATION_GROUPS = 20
EVALUATION_GROUPS = 20
WINDOWS_PER_GROUP_CAP = 3

# ARCTIC's contact-distance evaluation uses strict distance < 3 mm.  The raw
# release supplies fitted MANO/object geometry, not a semantic support-hand role.
NATIVE_CONTACT_DISTANCE_M = 0.003
NATIVE_CONTACT_COMPARATOR = "<"
NATIVE_CONTACT_PROVENANCE = "derived_from_arctic_native_mano_object_geometry"
NATIVE_SEMANTIC_ROLE_LABEL_AVAILABLE = False

REQUIRED_SEQUENCE_SUFFIXES = (
    ".mano.npy",
    ".object.npy",
    ".egocam.dist.npy",
)
OPTIONAL_SEQUENCE_SUFFIXES = (".smplx.npy",)
LICENSE_ATTESTATION_ENV = "ARCTIC_LICENSE_ACCEPTED"
MANO_LICENSE_ATTESTATION_ENV = "MANO_LICENSE_ACCEPTED"
ARCTIC_CREDENTIAL_ENVS = ("ARCTIC_USERNAME", "ARCTIC_PASSWORD")


class ResourceAccountingError(RuntimeError):
    """The filesystem usage total could not be established fail-closed."""


@dataclass(frozen=True)
class AssetSpec:
    name: str
    relative_path: str
    url: str
    sha256: str
    size_bytes: int
    documented_size: str
    required_for_m2: bool


_DOWNLOAD_BASE = (
    "https://download.is.tue.mpg.de/download.php?domain=arctic&resume=1&"
    "sfile=arctic_release/"
    "c7216c3b205186106a1f8326ed7b948f838e4907e69b21c8b3c87bb69d87206e/"
    f"{ARCTIC_RELEASE_VERSION}/data"
)

M2_ASSETS = (
    AssetSpec(
        name="splits_json",
        relative_path="data/splits_json.zip",
        url=f"{_DOWNLOAD_BASE}/splits_json.zip",
        sha256="99af5f6759c727df07ef897db6d293cc866d465e6b3d6095579fcb62ac831b7d",
        size_bytes=2_734,
        documented_size="40K",
        required_for_m2=True,
    ),
    AssetSpec(
        name="raw_seqs",
        relative_path="data/raw_seqs.zip",
        url=f"{_DOWNLOAD_BASE}/raw_seqs.zip",
        sha256="3c74f8cdb5fb4f521d99132faf0471432bab5db97c7653493b01920d2ad48535",
        size_bytes=225_334_351,
        documented_size="215M",
        required_for_m2=True,
    ),
    AssetSpec(
        name="meta",
        relative_path="data/meta.zip",
        url=f"{_DOWNLOAD_BASE}/meta.zip",
        sha256="2ec627bcb8f17be33defc985a79d1dd744ee44b1f1b5732ed163ecef217c0c6e",
        size_bytes=95_270_963,
        documented_size="91M",
        required_for_m2=True,
    ),
)

EXCLUDED_M2_ASSETS = (
    "RGB/cropped-images/background archives",
    "SMPL-X body model and sequence fields",
    "ARCTIC pretrained model checkpoints",
    "all perception-model checkpoints",
)


@dataclass(frozen=True)
class SequenceRecord:
    sequence_id: str
    subject_id: str
    object_id: str
    files: dict[str, str]
    optional_files: dict[str, str]
    missing_or_empty_fields: tuple[str, ...]
    eligible: bool


@dataclass(frozen=True)
class SplitEntry:
    sequence_id: str
    sha256: str
    split: str
    rank: int


def _present_nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _resolve_release_dir(data_root: Path, name: str) -> Path:
    """Return the only permitted extraction root for the frozen release."""

    return data_root / "data" / name


def discover_sequence_records(raw_root: Path) -> tuple[SequenceRecord, ...]:
    """Audit the required per-sequence files without importing or trusting NumPy."""

    raw_root = Path(raw_root)
    stems: set[str] = set()
    for suffix in REQUIRED_SEQUENCE_SUFFIXES + OPTIONAL_SEQUENCE_SUFFIXES:
        for path in raw_root.rglob(f"*{suffix}") if raw_root.is_dir() else ():
            relative = path.relative_to(raw_root).as_posix()
            stems.add(relative[: -len(suffix)])

    records: list[SequenceRecord] = []
    for stem in sorted(stems):
        parts = Path(stem).parts
        subject_id = parts[0] if len(parts) > 1 else "unknown"
        object_id = Path(stem).name.split("_", 1)[0]
        files: dict[str, str] = {}
        optional: dict[str, str] = {}
        missing: list[str] = []
        for suffix in REQUIRED_SEQUENCE_SUFFIXES:
            path = raw_root / f"{stem}{suffix}"
            files[suffix] = path.as_posix()
            if not _present_nonempty(path):
                missing.append(suffix)
        for suffix in OPTIONAL_SEQUENCE_SUFFIXES:
            path = raw_root / f"{stem}{suffix}"
            if _present_nonempty(path):
                optional[suffix] = path.as_posix()
        records.append(
            SequenceRecord(
                sequence_id=stem,
                subject_id=subject_id,
                object_id=object_id,
                files=files,
                optional_files=optional,
                missing_or_empty_fields=tuple(missing),
                eligible=not missing,
            )
        )
    return tuple(records)


def split_hash(sequence_id: str, release_id: str = ARCTIC_RELEASE_ID) -> str:
    """Implement the frozen sha256(release_id || sequence_id) ordering exactly."""

    if not isinstance(sequence_id, str) or not sequence_id:
        raise ValueError("sequence_id must be a non-empty string")
    return hashlib.sha256((release_id + sequence_id).encode("utf-8")).hexdigest()


def deterministic_group_split(
    sequence_ids: Sequence[str],
    release_id: str = ARCTIC_RELEASE_ID,
    calibration_groups: int = CALIBRATION_GROUPS,
    evaluation_groups: int = EVALUATION_GROUPS,
) -> tuple[SplitEntry, ...]:
    """Return a stable, leakage-free prefix split; never backfill by data quality."""

    if calibration_groups < 0 or evaluation_groups < 0:
        raise ValueError("group counts must be non-negative")
    normalized = list(sequence_ids)
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate sequence_id would create group leakage")
    ranked = sorted((split_hash(item, release_id), item) for item in normalized)
    limit = calibration_groups + evaluation_groups
    entries: list[SplitEntry] = []
    for rank, (digest, sequence_id) in enumerate(ranked[:limit]):
        split = "calibration" if rank < calibration_groups else "evaluation"
        entries.append(SplitEntry(sequence_id, digest, split, rank))
    return tuple(entries)


def _truthy_attestation(value: str | None) -> bool:
    return value is not None and value.strip().casefold() in {"1", "true", "yes"}


@dataclass(frozen=True)
class TreeSnapshot:
    label: str
    root: Path
    usage_bytes: int
    entry_count: int
    regular_file_count: int
    manifest_sha256: str
    reject_symlinks: bool


@dataclass(frozen=True)
class ResourceSnapshot:
    data_root: Path
    mano_root: Path | None
    workspace_root: Path
    m2_trees: tuple[TreeSnapshot, ...]
    workspace_tree: TreeSnapshot
    m2_usage_bytes: int
    m2_manifest_sha256: str


def _snapshot_tree(
    path: Path,
    *,
    label: str,
    reject_symlinks: bool,
) -> TreeSnapshot:
    """Build a metadata-only logical-byte manifest without following symlinks."""

    root = Path(path).absolute()
    records: list[tuple[object, ...]] = []
    usage_bytes = 0
    entry_count = 0
    regular_file_count = 0
    try:
        root_stat = root.lstat()
    except FileNotFoundError:
        digest = hashlib.sha256(b"missing-root\n").hexdigest()
        return TreeSnapshot(label, root, 0, 0, 0, digest, reject_symlinks)
    except OSError as exc:
        raise ResourceAccountingError(f"cannot inspect {label} root") from exc
    if stat.S_ISLNK(root_stat.st_mode):
        raise ResourceAccountingError(f"symlinked {label} root")
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ResourceAccountingError(f"{label} root is not a directory")

    def visit(directory: Path, relative: Path) -> None:
        nonlocal usage_bytes, entry_count, regular_file_count
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError as exc:
            raise ResourceAccountingError(f"cannot traverse {label}") from exc
        for entry in entries:
            child_relative = relative / entry.name
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ResourceAccountingError(f"cannot stat {label} entry") from exc
            mode = metadata.st_mode
            common = (
                child_relative.as_posix(),
                int(metadata.st_size),
                int(metadata.st_mtime_ns),
                int(metadata.st_dev),
                int(metadata.st_ino),
            )
            entry_count += 1
            if stat.S_ISLNK(mode):
                if reject_symlinks:
                    raise ResourceAccountingError(f"symlink inside {label}")
                usage_bytes += int(metadata.st_size)
                records.append(("L", *common))
            elif stat.S_ISDIR(mode):
                records.append(("D", *common))
                visit(Path(entry.path), child_relative)
            elif stat.S_ISREG(mode):
                usage_bytes += int(metadata.st_size)
                regular_file_count += 1
                records.append(("F", *common))
            else:
                if reject_symlinks:
                    raise ResourceAccountingError(f"special file inside {label}")
                usage_bytes += int(metadata.st_size)
                records.append(("S", *common))

    visit(root, Path())
    serialized = json.dumps(
        records, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return TreeSnapshot(
        label=label,
        root=root,
        usage_bytes=usage_bytes,
        entry_count=entry_count,
        regular_file_count=regular_file_count,
        manifest_sha256=hashlib.sha256(serialized).hexdigest(),
        reject_symlinks=reject_symlinks,
    )


def build_resource_snapshot(
    data_root: Path,
    mano_root: Path | None,
    workspace_root: Path,
) -> ResourceSnapshot:
    """Create the single pre-read source of truth for M2 and workspace usage."""

    workspace = Path(workspace_root).absolute()
    roots = [
        ("arctic", Path(data_root).absolute()),
        ("m2_artifacts", workspace / "artifacts" / "quiethand" / "m2"),
    ]
    if mano_root is not None:
        roots.append(("mano", Path(mano_root).absolute()))
    for index, (_, left) in enumerate(roots):
        for _, right in roots[index + 1 :]:
            if left == right or left.is_relative_to(right) or right.is_relative_to(left):
                raise ResourceAccountingError("overlapping M2 accounting roots")
    m2_trees = tuple(
        _snapshot_tree(path, label=label, reject_symlinks=True)
        for label, path in roots
    )
    combined = [
        (
            tree.label,
            tree.usage_bytes,
            tree.entry_count,
            tree.regular_file_count,
            tree.manifest_sha256,
        )
        for tree in m2_trees
    ]
    m2_manifest_sha256 = hashlib.sha256(
        json.dumps(combined, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    workspace_tree = _snapshot_tree(
        workspace, label="workspace", reject_symlinks=False
    )
    return ResourceSnapshot(
        data_root=Path(data_root).absolute(),
        mano_root=Path(mano_root).absolute() if mano_root is not None else None,
        workspace_root=workspace,
        m2_trees=m2_trees,
        workspace_tree=workspace_tree,
        m2_usage_bytes=sum(tree.usage_bytes for tree in m2_trees),
        m2_manifest_sha256=m2_manifest_sha256,
    )


def resource_snapshot_report(snapshot: ResourceSnapshot) -> dict[str, object]:
    return {
        "m2_data_cap_bytes": M2_DATA_CAP_BYTES,
        "data_usage_bytes": snapshot.m2_usage_bytes,
        "data_usage_scope": "ARCTIC+MANO+partials+extract+rollback+M2-derived-artifacts",
        "m2_manifest_sha256": snapshot.m2_manifest_sha256,
        "m2_entry_count": sum(tree.entry_count for tree in snapshot.m2_trees),
        "m2_regular_file_count": sum(
            tree.regular_file_count for tree in snapshot.m2_trees
        ),
        "workspace_warning_bytes": WORKSPACE_WARNING_BYTES,
        "workspace_hard_cap_bytes": WORKSPACE_HARD_CAP_BYTES,
        "workspace_usage_bytes": snapshot.workspace_tree.usage_bytes,
        "workspace_manifest_sha256": snapshot.workspace_tree.manifest_sha256,
        "workspace_entry_count": snapshot.workspace_tree.entry_count,
        "accounting_complete": True,
        "snapshot_stable": True,
        "accounting_errors": [],
        "gpu_required": False,
        "model_call_required": False,
    }


def verify_resource_snapshot(snapshot: ResourceSnapshot) -> None:
    current = build_resource_snapshot(
        snapshot.data_root, snapshot.mano_root, snapshot.workspace_root
    )
    if current != snapshot:
        raise ResourceAccountingError("resource snapshot changed during validation")


def verify_resource_report(
    report: Mapping[str, object],
    *,
    data_root: Path,
    mano_root: Path | None,
    workspace_root: Path,
) -> None:
    expected = resource_snapshot_report(
        build_resource_snapshot(data_root, mano_root, workspace_root)
    )
    observed = {key: report.get(key) for key in expected}
    if observed != expected:
        raise ResourceAccountingError("terminal resource snapshot mismatch")


def sha256_exact_file(path: Path, *, root: Path, expected_size: int) -> str:
    """Hash exactly one pinned regular file with no unbounded read or symlink."""

    path = Path(path)
    root = Path(root)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ResourceAccountingError("pinned file escaped its root") from exc
    current = path
    try:
        while current != root:
            if current.is_symlink():
                raise ResourceAccountingError("symlink in pinned file path")
            current = current.parent
        if root.is_symlink():
            raise ResourceAccountingError("symlinked pinned file root")
        metadata = path.lstat()
    except OSError as exc:
        raise ResourceAccountingError("cannot inspect pinned file") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_size:
        raise ResourceAccountingError("pinned file size/type mismatch")

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    digest = hashlib.sha256()
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            opened = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_size != expected_size
                or opened.st_dev != metadata.st_dev
                or opened.st_ino != metadata.st_ino
            ):
                raise ResourceAccountingError("pinned file changed before read")
            remaining = expected_size
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ResourceAccountingError("short pinned file read")
                digest.update(chunk)
                remaining -= len(chunk)
            if handle.read(1):
                raise ResourceAccountingError("oversized pinned file read")
            closed = os.fstat(handle.fileno())
            if (
                closed.st_size != expected_size
                or closed.st_dev != opened.st_dev
                or closed.st_ino != opened.st_ino
                or closed.st_mtime_ns != opened.st_mtime_ns
            ):
                raise ResourceAccountingError("pinned file changed during read")
    except ResourceAccountingError:
        raise
    except OSError as exc:
        raise ResourceAccountingError("cannot read pinned file") from exc
    return digest.hexdigest()


def _asset_integrity_audit(data_root: Path) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for asset in M2_ASSETS:
        path = data_root / asset.relative_path
        present = _present_nonempty(path)
        actual_sha256 = (
            sha256_exact_file(
                path,
                root=data_root / "data",
                expected_size=asset.size_bytes,
            )
            if present
            else None
        )
        entries.append(
            {
                "name": asset.name,
                "relative_path": asset.relative_path,
                "present_nonempty": present,
                "expected_sha256": asset.sha256,
                "actual_sha256": actual_sha256,
                "sha256_match": actual_sha256 == asset.sha256,
            }
        )
    required = [
        entry
        for asset, entry in zip(M2_ASSETS, entries)
        if asset.required_for_m2
    ]
    return {
        "entries": entries,
        "closed": bool(required) and all(item["sha256_match"] for item in required),
    }


def _meta_audit(meta_root: Path, required_object_ids: Sequence[str]) -> dict[str, object]:
    misc_path = meta_root / "misc.json"
    unique_objects = tuple(sorted(set(required_object_ids)))
    complete_objects: list[str] = []
    missing_objects: list[str] = []
    for object_id in unique_objects:
        template = meta_root / "object_vtemplates" / object_id
        if all(
            _present_nonempty(template / name)
            for name in ("mesh.obj", "parts.json", "object_params.json")
        ):
            complete_objects.append(object_id)
        else:
            missing_objects.append(object_id)
    return {
        "root": meta_root.as_posix(),
        "misc_json_path": misc_path.as_posix(),
        "misc_json_present_nonempty": _present_nonempty(misc_path),
        "required_object_ids": list(unique_objects),
        "complete_object_ids": complete_objects,
        "missing_object_ids": missing_objects,
        "closed": bool(unique_objects)
        and _present_nonempty(misc_path)
        and not missing_objects,
    }


def _mano_audit(mano_root: Path | None) -> dict[str, object]:
    expected = ("MANO_LEFT.pkl", "MANO_RIGHT.pkl")
    if mano_root is None:
        return {"root": None, "files_present": {name: False for name in expected}, "closed": False}
    model_root = mano_root / "models"
    found = {name: _present_nonempty(model_root / name) for name in expected}
    return {"root": mano_root.as_posix(), "files_present": found, "closed": all(found.values())}


def native_reference_contract() -> dict[str, object]:
    return {
        "contact_distance_m": NATIVE_CONTACT_DISTANCE_M,
        "contact_comparator": NATIVE_CONTACT_COMPARATOR,
        "contact_provenance": NATIVE_CONTACT_PROVENANCE,
        "semantic_role_label_available": NATIVE_SEMANTIC_ROLE_LABEL_AVAILABLE,
        "data_doc_url": ARCTIC_DATA_DOC_URL,
        "contact_source_url": ARCTIC_PAPER_URL,
        "scope": "geometry/reference validation only; no support-role semantic validation",
    }


def build_m2_preflight(
    data_root: Path,
    workspace_root: Path,
    mano_root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Build a secret-free M2 readiness report without downloading or loading data."""

    data_root = Path(data_root)
    workspace_root = Path(workspace_root)
    mano_root = Path(mano_root) if mano_root is not None else None
    environment = os.environ if env is None else env
    license_attested = _truthy_attestation(environment.get(LICENSE_ATTESTATION_ENV))
    mano_license_attested = _truthy_attestation(
        environment.get(MANO_LICENSE_ATTESTATION_ENV)
    )
    credentials_present = {
        name: bool(environment.get(name, "").strip()) for name in ARCTIC_CREDENTIAL_ENVS
    }
    credentials_complete = all(credentials_present.values())

    raw_root = _resolve_release_dir(data_root, "raw_seqs")
    meta_root = _resolve_release_dir(data_root, "meta")
    accounting_errors: list[str] = []
    snapshot: ResourceSnapshot | None = None
    try:
        snapshot = build_resource_snapshot(data_root, mano_root, workspace_root)
        resources = resource_snapshot_report(snapshot)
    except ResourceAccountingError as exc:
        accounting_errors.append(str(exc))
        resources = {
            "m2_data_cap_bytes": M2_DATA_CAP_BYTES,
            "data_usage_bytes": None,
            "data_usage_scope": "ARCTIC+MANO+partials+extract+rollback+M2-derived-artifacts",
            "m2_manifest_sha256": None,
            "m2_entry_count": None,
            "m2_regular_file_count": None,
            "workspace_warning_bytes": WORKSPACE_WARNING_BYTES,
            "workspace_hard_cap_bytes": WORKSPACE_HARD_CAP_BYTES,
            "workspace_usage_bytes": None,
            "workspace_manifest_sha256": None,
            "workspace_entry_count": None,
            "accounting_complete": False,
            "snapshot_stable": False,
            "accounting_errors": list(accounting_errors),
            "gpu_required": False,
            "model_call_required": False,
        }

    data_usage = resources["data_usage_bytes"]
    workspace_usage = resources["workspace_usage_bytes"]
    resource_blocked = bool(
        accounting_errors
        or (
            isinstance(workspace_usage, int)
            and workspace_usage >= WORKSPACE_WARNING_BYTES
        )
        or (
            isinstance(data_usage, int)
            and data_usage > M2_DATA_CAP_BYTES
        )
    )

    sequence_records: tuple[SequenceRecord, ...] = ()
    eligible_ids: list[str] = []
    eligible_object_ids: list[str] = []
    split: tuple[SplitEntry, ...] = ()
    calibration_count = 0
    evaluation_count = 0
    asset_integrity: dict[str, object] = {"entries": [], "closed": False}
    meta: dict[str, object] = {
        "root": meta_root.as_posix(),
        "misc_json_path": (meta_root / "misc.json").as_posix(),
        "misc_json_present_nonempty": False,
        "required_object_ids": [],
        "complete_object_ids": [],
        "missing_object_ids": [],
        "closed": False,
    }
    mano: dict[str, object] = _mano_audit(None)
    alternate_roots_present = False
    dataset_present = False

    if not resource_blocked and snapshot is not None:
        try:
            sequence_records = discover_sequence_records(raw_root)
            eligible_ids = [
                record.sequence_id for record in sequence_records if record.eligible
            ]
            eligible_object_ids = [
                record.object_id for record in sequence_records if record.eligible
            ]
            split = deterministic_group_split(eligible_ids)
            calibration_count = sum(item.split == "calibration" for item in split)
            evaluation_count = sum(item.split == "evaluation" for item in split)
            asset_integrity = _asset_integrity_audit(data_root)
            meta = _meta_audit(meta_root, eligible_object_ids)
            mano = _mano_audit(mano_root)
            alternate_roots_present = any(
                (data_root / name).exists() or (data_root / name).is_symlink()
                for name in ("raw_seqs", "meta", "splits_json")
            )
            dataset_present = raw_root.is_dir() or meta_root.is_dir()
            verify_resource_snapshot(snapshot)
        except ResourceAccountingError as exc:
            accounting_errors.append(str(exc))
            resources["accounting_complete"] = False
            resources["snapshot_stable"] = False
            resources["accounting_errors"] = list(accounting_errors)

    try:
        workspace_free = shutil.disk_usage(workspace_root).free
    except OSError:
        workspace_free = None
    resources["workspace_free_bytes"] = workspace_free
    warnings: list[str] = []
    if not NATIVE_SEMANTIC_ROLE_LABEL_AVAILABLE:
        warnings.append("ARCTIC has no native active-hand/support-hand semantic role label")
    warnings.append(
        "contact is derived from native fitted hand/object geometry using distance < 0.003 m"
    )
    if isinstance(workspace_usage, int) and workspace_usage >= WORKSPACE_WARNING_BYTES:
        warnings.append("workspace usage reached the frozen 225 GiB warning threshold")

    if accounting_errors:
        status = "HOLD_RESOURCE_ACCOUNTING_INCOMPLETE"
    elif isinstance(workspace_usage, int) and workspace_usage >= WORKSPACE_HARD_CAP_BYTES:
        status = "HOLD_WORKSPACE_HARD_CAP"
    elif isinstance(data_usage, int) and data_usage > M2_DATA_CAP_BYTES:
        status = "HOLD_M2_DATA_CAP"
    elif isinstance(workspace_usage, int) and workspace_usage >= WORKSPACE_WARNING_BYTES:
        status = "HOLD_WORKSPACE_WARNING_USER_ACTION"
    elif not license_attested or not mano_license_attested:
        status = "HOLD_DATA_ACCESS_AUTH_REQUIRED"
    elif not dataset_present and not credentials_complete:
        status = "HOLD_DATA_ACCESS_AUTH_REQUIRED"
    elif not dataset_present:
        status = "READY_FOR_MINIMAL_DOWNLOAD"
    elif alternate_roots_present:
        status = "HOLD_DATA_CONTRACT"
    elif not bool(asset_integrity["closed"]):
        status = "HOLD_ASSET_INTEGRITY"
    elif not raw_root.is_dir() or not bool(meta["closed"]):
        status = "HOLD_DATA_CONTRACT"
    elif not bool(mano["closed"]):
        status = "HOLD_MANO_MODEL_REQUIRED"
    elif calibration_count < CALIBRATION_GROUPS:
        status = "HOLD_CALIBRATION_COVERAGE"
    elif evaluation_count < EVALUATION_GROUPS:
        status = "HOLD_GROUP_COVERAGE"
    else:
        status = "READY_FOR_ARRAY_AND_GEOMETRY_VALIDATION"

    return {
        "schema_version": 2,
        "milestone": "QH-E1-M2",
        "status": status,
        "release": {
            "repository": ARCTIC_REPOSITORY,
            "source_commit": ARCTIC_SOURCE_COMMIT,
            "release_version": ARCTIC_RELEASE_VERSION,
            "release_id": ARCTIC_RELEASE_ID,
            "assets": [asdict(asset) for asset in M2_ASSETS],
            "excluded_assets": list(EXCLUDED_M2_ASSETS),
        },
        "license_and_access": {
            "license_url": ARCTIC_LICENSE_URL,
            "registration_url": ARCTIC_REGISTRATION_URL,
            "user_attestation_env": LICENSE_ATTESTATION_ENV,
            "user_attested": license_attested,
            "mano_license_attestation_env": MANO_LICENSE_ATTESTATION_ENV,
            "mano_license_attested": mano_license_attested,
            "credential_env_presence": credentials_present,
            "credentials_complete": credentials_complete,
            "secret_values_recorded": False,
        },
        "native_reference_contract": native_reference_contract(),
        "paths": {
            "data_root": data_root.as_posix(),
            "raw_root": raw_root.as_posix(),
            "meta_root": meta_root.as_posix(),
            "canonical_roots_only": True,
            "alternate_roots_present": alternate_roots_present,
        },
        "field_audit": {
            "required_sequence_suffixes": list(REQUIRED_SEQUENCE_SUFFIXES),
            "optional_sequence_suffixes": list(OPTIONAL_SEQUENCE_SUFFIXES),
            "candidate_sequence_count": len(sequence_records),
            "eligible_sequence_count": len(eligible_ids),
            "ineligible_sequences": [
                {
                    "sequence_id": record.sequence_id,
                    "missing_or_empty_fields": list(record.missing_or_empty_fields),
                }
                for record in sequence_records
                if not record.eligible
            ],
            "asset_integrity": asset_integrity,
            "meta": meta,
            "mano": mano,
            "array_contents_validated": False,
        },
        "split_contract": {
            "hash_expression": "sha256(utf8(release_id || sequence_id))",
            "calibration_groups_required": CALIBRATION_GROUPS,
            "evaluation_groups_required": EVALUATION_GROUPS,
            "windows_per_group_cap": WINDOWS_PER_GROUP_CAP,
            "calibration_groups_selected": calibration_count,
            "evaluation_groups_selected": evaluation_count,
            "entries": [asdict(item) for item in split],
            "subject_object_distribution_used_for_recut": False,
        },
        "resources": resources,
        "warnings": warnings,
    }
