#!/usr/bin/env python3
"""Seal the two preregistered evaluation rankings before target review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_EVENTS = 30


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-results", type=Path, required=True)
    parser.add_argument("--fusion", type=Path, required=True)
    parser.add_argument("--rule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_uncertainty(document: dict[str, Any], rule: dict[str, Any]) -> int:
    if document.get("status") != "observed" or not isinstance(document.get("candidate"), dict):
        return int(rule["vlm_only"]["semantic_invalid_score"])
    candidate = document["candidate"]
    score = sum(candidate.get(field) is None or candidate.get(field) == "" for field in rule["vlm_only"]["null_or_empty_fields"])
    for path in rule["vlm_only"]["unknown_fields"]:
        side, field = path.split(".")
        score += (candidate.get(side) or {}).get(field) == "unknown"
    score += int(rule["vlm_only"]["evidence_weight"][candidate["evidence"]])
    return int(score)


def claimed_evidence(candidate: dict[str, Any], hand: dict[str, Any], side: str) -> str:
    contact = (candidate.get(side) or {}).get("contact")
    if contact not in {"tool", "target"}:
        return "missing"
    pair = (hand.get("pairs") or {}).get(contact)
    if not isinstance(pair, dict) or pair.get("status") != "observed":
        return "missing"
    state = (pair.get("contact") or {}).get("state")
    if state in {"maintained_close", "boundary_near"}:
        return "supported"
    if state == "not_close":
        return "contradicted"
    return "missing"


def forbidden_paths(value: Any, prefix: str = "$") -> list[str]:
    forbidden = {"review_needed", "review_decision", "human_review", "correction_text", "target_label"}
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in forbidden:
                found.append(f"{prefix}.{key}")
            found.extend(forbidden_paths(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(forbidden_paths(child, f"{prefix}[{index}]"))
    return found


def main() -> int:
    cfg = arguments()
    rule = load(cfg.rule)
    fusion = load(cfg.fusion)
    records = fusion.get("records")
    semantic_audit = load(cfg.semantic_results / "audit.json")
    if (
        rule.get("evaluation_event_count") != EXPECTED_EVENTS
        or rule.get("human_review_budget") != 6
        or rule.get("retuning_allowed") is not False
        or fusion.get("status") != "COMPLETE"
        or fusion.get("event_count") != EXPECTED_EVENTS
        or not isinstance(records, list)
        or len(records) != EXPECTED_EVENTS
        or semantic_audit.get("status") != "COMPLETE"
        or semantic_audit.get("event_count") != EXPECTED_EVENTS
        or semantic_audit.get("evaluation_results_opened") is not True
    ):
        raise SystemExit("ranking dependencies do not match the frozen evaluation rule")
    rows = []
    for record in records:
        event_id = record["event_id"]
        semantic_document = load(cfg.semantic_results / "events" / f"{event_id}.json")
        uncertainty = semantic_uncertainty(semantic_document, rule)
        candidate = semantic_document.get("candidate") or {}
        evidence = {side: claimed_evidence(candidate, record["hands"][side], side) for side in ("left", "right")}
        rows.append({
            "event_id": event_id,
            "semantic_uncertainty_count": uncertainty,
            "claimed_contact_evidence": evidence,
            "contradicted_hand_count": sum(value == "contradicted" for value in evidence.values()),
            "missing_hand_count": sum(value == "missing" for value in evidence.values()),
        })
    if len({row["event_id"] for row in rows}) != EXPECTED_EVENTS:
        raise SystemExit("ranking event ids are not unique")
    vlm_only = sorted(rows, key=lambda row: (-row["semantic_uncertainty_count"], row["event_id"]))
    geometry = sorted(rows, key=lambda row: (-row["contradicted_hand_count"], -row["semantic_uncertainty_count"], -row["missing_hand_count"], row["event_id"]))
    output = {
        "schema": "quiethand.m3_5.evaluation_rankings.v1",
        "status": "SEALED_BEFORE_TARGET_REVIEW",
        "event_count": EXPECTED_EVENTS,
        "human_review_budget_each": 6,
        "rule_sha256": sha(cfg.rule),
        "training_performed": False,
        "human_targets_used": False,
        "vlm_only": [{**row, "rank": rank, "selected_for_review": rank <= 6} for rank, row in enumerate(vlm_only, start=1)],
        "vlm_plus_geometry": [{**row, "rank": rank, "selected_for_review": rank <= 6} for rank, row in enumerate(geometry, start=1)],
    }
    leaks = forbidden_paths(output)
    if leaks or sum(row["selected_for_review"] for row in output["vlm_only"]) != 6 or sum(row["selected_for_review"] for row in output["vlm_plus_geometry"]) != 6:
        raise SystemExit(f"ranking seal failed: leaks={leaks}")
    if cfg.output.exists():
        raise SystemExit(f"ranking output already exists: {cfg.output}")
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "event_count": EXPECTED_EVENTS, "budget_each": 6, "rule_sha256": output["rule_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
