#!/usr/bin/env python3
"""Validate two blinded calibration annotations and compile a human reference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile


FIELDS = ("left_role", "right_role", "left_contact", "right_contact", "evidence")
COLUMNS = ("audit_index", "event_id", "video_relative_path", *FIELDS, "comment_optional")
ENUMS = {
    "left_role": {"active", "support", "both", "neither", "unknown"},
    "right_role": {"active", "support", "both", "neither", "unknown"},
    "left_contact": {"tool", "target", "both", "neither", "unknown"},
    "right_contact": {"tool", "target", "both", "neither", "unknown"},
    "evidence": {"visible", "ambiguous", "not_visible"},
}


class HumanAuditCompileError(RuntimeError):
    pass


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--package-manifest-sha256", type=str, required=True)
    parser.add_argument("--annotator-a", type=Path, required=True)
    parser.add_argument("--annotator-a-id", type=str, required=True)
    parser.add_argument("--annotator-b", type=Path, required=True)
    parser.add_argument("--annotator-b-id", type=str, required=True)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--adjudicator-id", type=str)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path, expected_ids: set[str]) -> dict[str, dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise HumanAuditCompileError("annotation sheet is missing or symlinked")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, dialect="excel-tab")
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise HumanAuditCompileError("annotation sheet columns changed")
        rows = list(reader)
    if len(rows) != len(expected_ids):
        raise HumanAuditCompileError("annotation sheet coverage changed")
    result = {}
    for position, row in enumerate(rows, start=1):
        event_id = row["event_id"]
        if row["audit_index"] != str(position) or event_id in result or event_id not in expected_ids:
            raise HumanAuditCompileError("annotation row index or identity is invalid")
        if row["video_relative_path"] != f"videos/{event_id}.mp4":
            raise HumanAuditCompileError("annotation video binding changed")
        for field in FIELDS:
            if row[field] not in ENUMS[field]:
                raise HumanAuditCompileError(f"annotation enum is invalid: {field}")
        result[event_id] = row
    if set(result) != expected_ids:
        raise HumanAuditCompileError("annotation sheet event set changed")
    return result


def kappa(rows_a: dict[str, dict[str, str]], rows_b: dict[str, dict[str, str]], field: str) -> float:
    event_ids = sorted(rows_a)
    observed = sum(rows_a[event][field] == rows_b[event][field] for event in event_ids) / len(event_ids)
    expected = 0.0
    for value in ENUMS[field]:
        pa = sum(rows_a[event][field] == value for event in event_ids) / len(event_ids)
        pb = sum(rows_b[event][field] == value for event in event_ids) / len(event_ids)
        expected += pa * pb
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1.0 - expected)


def write_adjudication_template(path: Path, event_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, dialect="excel-tab", lineterminator="\n")
        writer.writerow(COLUMNS)
        for index, event_id in enumerate(event_ids, start=1):
            writer.writerow([index, event_id, f"videos/{event_id}.mp4", "", "", "", "", "", ""])


def atomic_json(path: Path, value: dict[str, object]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    args = arguments()
    temporary: Path | None = None
    try:
        if args.output.exists():
            raise HumanAuditCompileError("human audit compile destination already exists")
        ids = [args.annotator_a_id.strip(), args.annotator_b_id.strip()]
        if not all(ids) or len(set(ids)) != 2:
            raise HumanAuditCompileError("two distinct non-empty annotator IDs are required")
        if sha256_file(args.package_manifest) != args.package_manifest_sha256:
            raise HumanAuditCompileError("blinded package manifest changed")
        package = json.loads(args.package_manifest.read_text(encoding="utf-8"))
        videos = package.get("videos")
        if (
            package.get("status") != "READY_FOR_TWO_INDEPENDENT_ANNOTATORS"
            or package.get("event_count") != 150
            or package.get("evaluation_event_count") != 0
            or package.get("model_or_native_outputs_included") is not False
            or not isinstance(videos, list)
            or len(videos) != 150
        ):
            raise HumanAuditCompileError("blinded package envelope is invalid")
        event_ids = {row.get("event_id") for row in videos if isinstance(row, dict)}
        if None in event_ids or len(event_ids) != 150:
            raise HumanAuditCompileError("blinded package event identity is invalid")
        rows_a = read_rows(args.annotator_a, event_ids)
        rows_b = read_rows(args.annotator_b, event_ids)
        disagreements = sorted(
            event_id for event_id in event_ids if any(rows_a[event_id][field] != rows_b[event_id][field] for field in FIELDS)
        )
        agreement = {
            field: {
                "agreement": sum(rows_a[event][field] == rows_b[event][field] for event in event_ids) / 150.0,
                "cohen_kappa": kappa(rows_a, rows_b, field),
            }
            for field in FIELDS
        }
        temporary = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
        audit = {
            "schema": "quiethand.m3.human_audit_compile.v1",
            "package_manifest_sha256": args.package_manifest_sha256,
            "annotator_a": {"id": ids[0], "sha256": sha256_file(args.annotator_a)},
            "annotator_b": {"id": ids[1], "sha256": sha256_file(args.annotator_b)},
            "event_count": 150,
            "disagreement_event_count": len(disagreements),
            "agreement": agreement,
            "scientific_result": False,
        }
        if disagreements and args.adjudication is None:
            write_adjudication_template(temporary / "adjudication_blind.tsv", disagreements)
            audit["status"] = "HOLD_HUMAN_AUDIT_ADJUDICATION_REQUIRED"
            atomic_json(temporary / "audit.json", audit)
            os.replace(temporary, args.output)
            temporary = None
            print(f"[artifact] {args.output / 'adjudication_blind.tsv'}")
            print(f"[hold] HOLD_HUMAN_AUDIT_ADJUDICATION_REQUIRED: {len(disagreements)} events", file=sys.stderr)
            return 3
        adjudicated: dict[str, dict[str, str]] = {}
        if disagreements:
            adjudicator = (args.adjudicator_id or "").strip()
            if not adjudicator or adjudicator in ids:
                raise HumanAuditCompileError("a distinct non-empty adjudicator ID is required")
            adjudicated = read_rows(args.adjudication, set(disagreements))
            audit["adjudicator"] = {"id": adjudicator, "sha256": sha256_file(args.adjudication)}
        elif args.adjudication is not None or args.adjudicator_id is not None:
            raise HumanAuditCompileError("adjudication was supplied although no event disagrees")
        reference = []
        for event_id in sorted(event_ids):
            source = "adjudicated" if event_id in adjudicated else "annotator_agreement"
            row = adjudicated.get(event_id, rows_a[event_id])
            reference.append(
                {
                    "event_id": event_id,
                    "source": source,
                    **{field: row[field] for field in FIELDS},
                }
            )
        human_reference = {
            "schema": "quiethand.m3.human_semantic_reference.v1",
            "status": "READY_HUMAN_REFERENCE",
            "event_count": 150,
            "evaluation_event_count": 0,
            "labels": reference,
        }
        atomic_json(temporary / "human_reference.json", human_reference)
        audit["status"] = "READY_HUMAN_REFERENCE"
        audit["human_reference_sha256"] = sha256_file(temporary / "human_reference.json")
        atomic_json(temporary / "audit.json", audit)
        os.replace(temporary, args.output)
        temporary = None
    except (OSError, ValueError, KeyError, json.JSONDecodeError, HumanAuditCompileError) as exc:
        print(f"[hold] HOLD_HUMAN_AUDIT_INCOMPLETE: {exc}", file=sys.stderr)
        return 3
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
    print(f"[artifact] {args.output / 'human_reference.json'}")
    print("[status] READY_HUMAN_REFERENCE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
