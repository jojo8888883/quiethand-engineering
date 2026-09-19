"""Validated and deterministically serializable QuietHand records."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any, Mapping, Sequence


AVAILABLE_STATUSES = frozenset({"observed", "inferred", "human", "derived"})
MISSING_STATUSES = frozenset({"missing", "unobservable", "ambiguous", "invalid"})
SOURCE_KINDS = frozenset({"oracle", "model", "human", "infill", "derived"})
HANDS = frozenset({"left", "right", "ambiguous"})
TAU_RESTRICTED = 0.20
TAU_FREE = 0.80


def _finite(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _vec3(value: Sequence[float], label: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{label} must contain exactly three values")
    return tuple(_finite(item, label) for item in value)  # type: ignore[return-value]


def _require_exact_keys(value: Mapping[str, Any], required: set[str], label: str) -> None:
    actual = set(value)
    if actual != required:
        raise ValueError(
            f"{label} fields must exactly match the schema; missing={sorted(required-actual)}, extra={sorted(actual-required)}"
        )


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a JSON string")
    return value


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a JSON number, not bool or string")
    return _finite(value, label)


def _validate_json_value(value: Any, label: str, *, allow_null: bool) -> None:
    if value is None:
        if allow_null:
            return
        raise ValueError(f"{label} contains a silent null")
    if isinstance(value, bool) or isinstance(value, int) or isinstance(value, str):
        return
    if isinstance(value, float):
        _finite(value, label)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{label} contains a non-string or blank object key")
            _validate_json_value(child, f"{label}.{key}", allow_null=allow_null)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_value(child, f"{label}[{index}]", allow_null=allow_null)
        return
    raise ValueError(f"{label} contains a non-JSON value of type {type(value).__name__}")


@dataclass(frozen=True)
class Provenance:
    """Origin of one evidence value; empty or anonymous origins are rejected."""

    source_kind: str
    source_name: str
    source_version: str
    run_id: str

    def __post_init__(self) -> None:
        if self.source_kind not in SOURCE_KINDS:
            raise ValueError(f"unsupported source_kind: {self.source_kind}")
        for name in ("source_name", "source_version", "run_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"provenance.{name} cannot be empty")

    def to_dict(self) -> dict[str, str]:
        return {
            "source_kind": self.source_kind,
            "source_name": self.source_name,
            "source_version": self.source_version,
            "run_id": self.run_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Provenance":
        if not isinstance(value, Mapping):
            raise ValueError("provenance must be a JSON object")
        _require_exact_keys(
            value,
            {"source_kind", "source_name", "source_version", "run_id"},
            "provenance",
        )
        return cls(
            source_kind=_require_string(value["source_kind"], "provenance.source_kind"),
            source_name=_require_string(value["source_name"], "provenance.source_name"),
            source_version=_require_string(value["source_version"], "provenance.source_version"),
            run_id=_require_string(value["run_id"], "provenance.run_id"),
        )


@dataclass(frozen=True)
class EvidenceRecord:
    """One timestamped item with explicit missingness and dependencies."""

    key: str
    status: str
    value: Any
    unit: str
    frame_id: str
    timestamp_s: float
    provenance: Provenance
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    failure_reason: str | None = None
    error_set_ref: str | None = None

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("evidence key cannot be empty")
        object.__setattr__(self, "timestamp_s", _finite(self.timestamp_s, "timestamp_s"))
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        if any(not isinstance(item, str) or not item.strip() for item in self.depends_on):
            raise ValueError("depends_on entries must be nonblank strings")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("depends_on cannot contain duplicates")
        if self.key in self.depends_on:
            raise ValueError("an evidence item cannot depend on itself")
        if self.status in AVAILABLE_STATUSES:
            if self.value is None:
                raise ValueError("available evidence cannot have a null value")
            _validate_json_value(self.value, f"evidence[{self.key}].value", allow_null=False)
            if self.failure_reason is not None:
                raise ValueError("available evidence cannot carry a failure_reason")
        elif self.status in MISSING_STATUSES:
            if self.value is not None:
                raise ValueError("missing/unusable evidence must have a null value")
            if not self.failure_reason or not self.failure_reason.strip():
                raise ValueError("missing/unusable evidence requires a failure_reason")
        else:
            raise ValueError(f"unsupported evidence status: {self.status}")
        if self.error_set_ref is not None and not self.error_set_ref.strip():
            raise ValueError("error_set_ref must be null or a nonblank reference")

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "status": self.status,
            "value": self.value,
            "unit": self.unit,
            "frame_id": self.frame_id,
            "timestamp_s": self.timestamp_s,
            "provenance": self.provenance.to_dict(),
            "depends_on": list(self.depends_on),
            "failure_reason": self.failure_reason,
            "error_set_ref": self.error_set_ref,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceRecord":
        if not isinstance(value, Mapping):
            raise ValueError("evidence record must be a JSON object")
        _require_exact_keys(
            value,
            {
                "key",
                "status",
                "value",
                "unit",
                "frame_id",
                "timestamp_s",
                "provenance",
                "depends_on",
                "failure_reason",
                "error_set_ref",
            },
            "evidence record",
        )
        if not isinstance(value["depends_on"], list):
            raise ValueError("evidence record depends_on must be a JSON array")
        if value["failure_reason"] is not None and not isinstance(
            value["failure_reason"], str
        ):
            raise ValueError("failure_reason must be a JSON string or null")
        if value["error_set_ref"] is not None and not isinstance(
            value["error_set_ref"], str
        ):
            raise ValueError("error_set_ref must be a JSON string or null")
        return cls(
            key=_require_string(value["key"], "evidence.key"),
            status=_require_string(value["status"], "evidence.status"),
            value=value.get("value"),
            unit=_require_string(value["unit"], "evidence.unit"),
            frame_id=_require_string(value["frame_id"], "evidence.frame_id"),
            timestamp_s=_require_number(value["timestamp_s"], "evidence.timestamp_s"),
            provenance=Provenance.from_dict(value["provenance"]),
            depends_on=tuple(
                _require_string(item, "evidence.depends_on[]")
                for item in value["depends_on"]
            ),
            failure_reason=value["failure_reason"],
            error_set_ref=value["error_set_ref"],
        )


@dataclass(frozen=True)
class EvidenceEvent:
    """A bimanual event packet with a closed, acyclic local dependency graph."""

    event_id: str
    dataset: str
    sequence_id: str
    start_s: float
    end_s: float
    active_hand: str
    support_hand: str
    supported_object_id: str
    support_frame_id: str
    evidence: tuple[EvidenceRecord, ...]
    schema_version: str = "qh.evidence_event.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "qh.evidence_event.v1":
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        for name in (
            "event_id",
            "dataset",
            "sequence_id",
            "supported_object_id",
            "support_frame_id",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be empty")
        start = _finite(self.start_s, "start_s")
        end = _finite(self.end_s, "end_s")
        object.__setattr__(self, "start_s", start)
        object.__setattr__(self, "end_s", end)
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if end <= start:
            raise ValueError("event end_s must be greater than start_s")
        if self.active_hand not in HANDS or self.support_hand not in HANDS:
            raise ValueError("hand roles must be left, right, or ambiguous")
        if self.active_hand == self.support_hand and self.active_hand != "ambiguous":
            raise ValueError("active and support hands must differ when both are definite")
        keys = [item.key for item in self.evidence]
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("event evidence must contain unique nonempty keys")
        key_set = set(keys)
        for item in self.evidence:
            if not self.start_s <= item.timestamp_s <= self.end_s:
                raise ValueError(f"evidence {item.key} lies outside the event interval")
            unknown = set(item.depends_on) - key_set
            if unknown:
                raise ValueError(f"evidence {item.key} has unknown dependencies: {unknown}")
        self._validate_acyclic_dependencies()

    def _validate_acyclic_dependencies(self) -> None:
        graph = {item.key: item.depends_on for item in self.evidence}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(key: str) -> None:
            if key in visiting:
                raise ValueError("evidence dependency graph contains a cycle")
            if key in visited:
                return
            visiting.add(key)
            for parent in graph[key]:
                visit(parent)
            visiting.remove(key)
            visited.add(key)

        for key in graph:
            visit(key)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "dataset": self.dataset,
            "sequence_id": self.sequence_id,
            "interval_s": {"start": self.start_s, "end": self.end_s},
            "roles": {"active_hand": self.active_hand, "support_hand": self.support_hand},
            "supported_object_id": self.supported_object_id,
            "support_frame_id": self.support_frame_id,
            "evidence": [item.to_dict() for item in sorted(self.evidence, key=lambda x: x.key)],
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceEvent":
        if not isinstance(value, Mapping):
            raise ValueError("EvidenceEvent must be a JSON object")
        _require_exact_keys(
            value,
            {
                "schema_version",
                "event_id",
                "dataset",
                "sequence_id",
                "interval_s",
                "roles",
                "supported_object_id",
                "support_frame_id",
                "evidence",
            },
            "EvidenceEvent",
        )
        interval = value["interval_s"]
        roles = value["roles"]
        if not isinstance(interval, Mapping) or not isinstance(roles, Mapping):
            raise ValueError("interval_s and roles must be objects")
        if not isinstance(value["evidence"], list):
            raise ValueError("evidence must be a JSON array")
        _require_exact_keys(interval, {"start", "end"}, "interval_s")
        _require_exact_keys(roles, {"active_hand", "support_hand"}, "roles")
        return cls(
            schema_version=_require_string(value["schema_version"], "schema_version"),
            event_id=_require_string(value["event_id"], "event_id"),
            dataset=_require_string(value["dataset"], "dataset"),
            sequence_id=_require_string(value["sequence_id"], "sequence_id"),
            start_s=_require_number(interval["start"], "interval_s.start"),
            end_s=_require_number(interval["end"], "interval_s.end"),
            active_hand=_require_string(roles["active_hand"], "roles.active_hand"),
            support_hand=_require_string(roles["support_hand"], "roles.support_hand"),
            supported_object_id=_require_string(
                value["supported_object_id"], "supported_object_id"
            ),
            support_frame_id=_require_string(value["support_frame_id"], "support_frame_id"),
            evidence=tuple(EvidenceRecord.from_dict(item) for item in value["evidence"]),
        )

    @classmethod
    def from_json(cls, value: str) -> "EvidenceEvent":
        def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON object key: {key}")
                result[key] = item
            return result

        def reject_constant(token: str) -> None:
            raise ValueError(f"non-standard JSON constant: {token}")

        decoded = json.loads(
            value, object_pairs_hook=strict_object, parse_constant=reject_constant
        )
        if not isinstance(decoded, dict):
            raise ValueError("EvidenceEvent JSON root must be an object")
        return cls.from_dict(decoded)


@dataclass(frozen=True)
class PatchConstraint:
    """One deduplicated hand-link/object-surface support patch."""

    patch_id: str
    hand_link: str
    surface_component: str
    point_m: tuple[float, float, float]
    normal: tuple[float, float, float]
    weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("patch_id", "hand_link", "surface_component"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be empty")
        point = _vec3(self.point_m, "point_m")
        normal = _vec3(self.normal, "normal")
        norm = math.sqrt(sum(item * item for item in normal))
        if norm <= 1e-15:
            raise ValueError("patch normal must be nonzero")
        weight = _finite(self.weight, "weight")
        if abs(weight - 1.0) > 1e-12:
            raise ValueError("QH-E1 freezes unit weight per deduplicated patch")
        object.__setattr__(self, "point_m", point)
        object.__setattr__(self, "normal", tuple(item / norm for item in normal))
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True)
class MobilityInterval:
    lower: float
    upper: float
    label: str

    def __post_init__(self) -> None:
        lower = _finite(self.lower, "lower")
        upper = _finite(self.upper, "upper")
        if lower < -1e-12 or upper > 1.0 + 1e-12 or lower > upper + 1e-12:
            raise ValueError("mobility interval must satisfy 0 <= lower <= upper <= 1")
        lower = min(1.0, max(0.0, lower))
        upper = min(1.0, max(0.0, upper))
        expected_label = (
            "restricted"
            if upper <= TAU_RESTRICTED
            else "free"
            if lower >= TAU_FREE
            else "undetermined"
        )
        if self.label != expected_label:
            raise ValueError(
                f"mobility label {self.label!r} contradicts interval; expected {expected_label!r}"
            )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def to_dict(self) -> dict[str, Any]:
        return {"lower": self.lower, "upper": self.upper, "label": self.label}


@dataclass(frozen=True)
class MobilityCertificate:
    directions: Mapping[str, MobilityInterval]
    status: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
    parent_ids: tuple[str, ...] = field(default_factory=tuple)
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"exact_outer", "mode_point", "finite_hull", "failure_safe"}:
            raise ValueError(f"unsupported certificate status: {self.status}")
        if not self.directions:
            raise ValueError("certificate must contain at least one direction")
        if any(not isinstance(key, str) or not key.strip() for key in self.directions):
            raise ValueError("certificate direction names must be nonblank strings")
        object.__setattr__(self, "directions", dict(self.directions))
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "parent_ids", tuple(self.parent_ids))
        object.__setattr__(self, "parameters", dict(self.parameters))
        _validate_json_value(self.parameters, "certificate.parameters", allow_null=True)
        if self.status == "failure_safe":
            if not self.reasons or any(not reason.strip() for reason in self.reasons):
                raise ValueError("failure_safe certificate requires a nonblank reason")
            if any(
                interval.lower != 0.0
                or interval.upper != 1.0
                or interval.label != "undetermined"
                for interval in self.directions.values()
            ):
                raise ValueError(
                    "failure_safe certificate must be [0,1]/undetermined in every direction"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "parent_ids": list(self.parent_ids),
            "parameters": dict(sorted(self.parameters.items())),
            "directions": {
                key: self.directions[key].to_dict() for key in sorted(self.directions)
            },
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
