"""Verified interval enclosure for the CPU-only QuietHand M1 gate.

The supported continuous uncertainty is deliberately small and explicit:
only the three coordinates of each contact point may vary over finite closed
intervals.  Contact normals, support frames, modes, and the characteristic
length are fixed inside a box.  Every arithmetic primitive rounds outwards;
any unsupported or unresolved operation fails the whole event to [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
import time
from typing import Callable, Mapping, Sequence

from .compiler import DEFAULT_DIRECTIONS, LAMBDA, classify_interval
from .geometry import characteristic_length
from .linalg import inverse
from .models import MobilityInterval, PatchConstraint


IntervalMatrix = list[list["Interval"]]
Clock = Callable[[], float]


@dataclass
class _EventBudget:
    """One event-wide CPU deadline; successful paths check it repeatedly."""

    deadline: float
    clock: Clock

    def check(self, stage: str) -> None:
        now = _finite(self.clock(), "clock")
        if now >= self.deadline:
            raise TimeoutError(f"verified interval event timed out during {stage}")


def _finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _down(value: float) -> float:
    value = _finite(value, "interval result")
    result = math.nextafter(value, -math.inf)
    if not math.isfinite(result):
        raise OverflowError("outward rounding overflowed")
    return result


def _up(value: float) -> float:
    value = _finite(value, "interval result")
    result = math.nextafter(value, math.inf)
    if not math.isfinite(result):
        raise OverflowError("outward rounding overflowed")
    return result


@dataclass(frozen=True)
class Interval:
    """A finite closed binary64 interval with outward-rounded operations."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        lower = _finite(self.lower, "interval.lower")
        upper = _finite(self.upper, "interval.upper")
        if lower > upper:
            raise ValueError("interval lower bound exceeds upper bound")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @classmethod
    def point(cls, value: float) -> "Interval":
        value = _finite(value, "interval point")
        return cls(value, value)

    @property
    def midpoint(self) -> float:
        # This form avoids overflow for large same-sign endpoints.
        return self.lower + (self.upper - self.lower) * 0.5

    @property
    def width(self) -> float:
        return _up(self.upper - self.lower) if self.upper != self.lower else 0.0

    @property
    def max_abs(self) -> float:
        return max(abs(self.lower), abs(self.upper))

    def contains(self, value: float, tolerance: float = 0.0) -> bool:
        value = float(value)
        return self.lower - tolerance <= value <= self.upper + tolerance

    def contains_zero(self) -> bool:
        return self.lower <= 0.0 <= self.upper

    def intersect(self, other: "Interval") -> "Interval":
        lower = max(self.lower, other.lower)
        upper = min(self.upper, other.upper)
        if lower > upper:
            raise ValueError("interval intersection is empty")
        return Interval(lower, upper)

    def hull(self, other: "Interval") -> "Interval":
        return Interval(min(self.lower, other.lower), max(self.upper, other.upper))

    def __neg__(self) -> "Interval":
        return Interval(-self.upper, -self.lower)

    def __add__(self, other: object) -> "Interval":
        rhs = as_interval(other)
        return Interval(_down(self.lower + rhs.lower), _up(self.upper + rhs.upper))

    def __radd__(self, other: object) -> "Interval":
        return self + other

    def __sub__(self, other: object) -> "Interval":
        return self + (-as_interval(other))

    def __rsub__(self, other: object) -> "Interval":
        return as_interval(other) - self

    def __mul__(self, other: object) -> "Interval":
        rhs = as_interval(other)
        products = (
            self.lower * rhs.lower,
            self.lower * rhs.upper,
            self.upper * rhs.lower,
            self.upper * rhs.upper,
        )
        if not all(math.isfinite(value) for value in products):
            raise OverflowError("interval multiplication overflowed")
        return Interval(_down(min(products)), _up(max(products)))

    def __rmul__(self, other: object) -> "Interval":
        return self * other

    def reciprocal(self) -> "Interval":
        if self.contains_zero():
            raise ZeroDivisionError("interval divisor contains zero")
        values = (1.0 / self.lower, 1.0 / self.upper)
        return Interval(_down(min(values)), _up(max(values)))

    def __truediv__(self, other: object) -> "Interval":
        return self * as_interval(other).reciprocal()

    def __rtruediv__(self, other: object) -> "Interval":
        return as_interval(other) / self

    def square(self) -> "Interval":
        if self.contains_zero():
            lower = 0.0
        else:
            lower = min(self.lower * self.lower, self.upper * self.upper)
        upper = max(self.lower * self.lower, self.upper * self.upper)
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise OverflowError("interval square overflowed")
        return Interval(0.0 if lower == 0.0 else _down(lower), _up(upper))

    def sqrt(self) -> "Interval":
        if self.lower < 0.0:
            raise ValueError("interval square root requires a nonnegative interval")
        lower = math.sqrt(self.lower)
        upper = math.sqrt(self.upper)
        return Interval(0.0 if lower == 0.0 else _down(lower), _up(upper))

    def to_list(self) -> list[float]:
        return [self.lower, self.upper]


def as_interval(value: object) -> Interval:
    if isinstance(value, Interval):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"cannot convert {type(value).__name__} to Interval")
    return Interval.point(float(value))


@dataclass(frozen=True)
class UncertainPatch:
    """One deduplicated patch with interval-valued contact point only."""

    patch_id: str
    hand_link: str
    surface_component: str
    point_m: tuple[Interval, Interval, Interval]
    normal: tuple[float, float, float]
    weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("patch_id", "hand_link", "surface_component"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} cannot be empty")
        if len(self.point_m) != 3 or any(not isinstance(item, Interval) for item in self.point_m):
            raise ValueError("point_m must contain exactly three Interval values")
        if len(self.normal) != 3:
            raise ValueError("normal must contain exactly three fixed values")
        normal = tuple(_finite(item, "normal") for item in self.normal)
        norm = math.hypot(*normal)
        if norm <= 1e-15:
            raise ValueError("patch normal must be nonzero")
        weight = _finite(self.weight, "weight")
        if abs(weight - 1.0) > 1e-12:
            raise ValueError("M1 freezes unit weight per deduplicated patch")
        object.__setattr__(self, "point_m", tuple(self.point_m))
        object.__setattr__(self, "normal", tuple(item / norm for item in normal))
        object.__setattr__(self, "weight", weight)

    @classmethod
    def singleton(cls, patch: PatchConstraint) -> "UncertainPatch":
        return cls(
            patch.patch_id,
            patch.hand_link,
            patch.surface_component,
            tuple(Interval.point(value) for value in patch.point_m),  # type: ignore[arg-type]
            patch.normal,
            patch.weight,
        )

    def sample(self, coordinates: Sequence[float]) -> PatchConstraint:
        if len(coordinates) != 3 or any(
            not interval.contains(float(value))
            for interval, value in zip(self.point_m, coordinates)
        ):
            raise ValueError("sample lies outside uncertain patch")
        return PatchConstraint(
            self.patch_id,
            self.hand_link,
            self.surface_component,
            tuple(map(float, coordinates)),
            self.normal,
            self.weight,
        )


@dataclass(frozen=True)
class VerifiedCertificate:
    """Deterministically serializable M1 interval certificate."""

    directions: Mapping[str, MobilityInterval]
    status: str
    reasons: tuple[str, ...] = field(default_factory=tuple)
    parent_ids: tuple[str, ...] = field(default_factory=tuple)
    parameters: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed = {"verified_interval", "finite_hull", "parent_hull", "failure_safe"}
        if self.status not in allowed:
            raise ValueError(f"unsupported verified certificate status: {self.status}")
        if not self.directions or any(
            not isinstance(name, str) or not name.strip() for name in self.directions
        ):
            raise ValueError("certificate requires nonblank direction names")
        object.__setattr__(self, "directions", dict(self.directions))
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "parent_ids", tuple(self.parent_ids))
        object.__setattr__(self, "parameters", dict(self.parameters))
        if self.status == "failure_safe":
            if not self.reasons or any(not reason.strip() for reason in self.reasons):
                raise ValueError("failure_safe certificate requires a reason")
            if any(
                (item.lower, item.upper, item.label) != (0.0, 1.0, "undetermined")
                for item in self.directions.values()
            ):
                raise ValueError("failure_safe certificate must be full [0,1]")
        # JSON encoding is also the strict no-NaN/no-Inf validation.
        json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "reasons": list(self.reasons),
            "parent_ids": list(self.parent_ids),
            "parameters": dict(sorted(self.parameters.items())),
            "directions": {
                name: self.directions[name].to_dict() for name in sorted(self.directions)
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


def _validate_directions(
    directions: Mapping[str, Sequence[float]],
) -> dict[str, tuple[float, ...]]:
    if not isinstance(directions, Mapping) or not directions:
        raise ValueError("direction namespace must be a nonempty mapping")
    result: dict[str, tuple[float, ...]] = {}
    for name, raw in directions.items():
        if not isinstance(name, str) or not name.strip() or len(raw) != 6:
            raise ValueError("directions require nonblank names and six values")
        values = tuple(_finite(value, f"direction[{name}]") for value in raw)
        norm = math.hypot(*values)
        if norm <= 1e-15:
            raise ValueError(f"direction[{name}] must be nonzero")
        result[name] = tuple(value / norm for value in values)
    return result


def failure_safe_certificate(
    reason: str,
    directions: Mapping[str, Sequence[float]] = DEFAULT_DIRECTIONS,
) -> VerifiedCertificate:
    try:
        names = tuple(_validate_directions(directions))
    except Exception:
        names = tuple(DEFAULT_DIRECTIONS)
        reason = f"{reason}; invalid direction namespace, used frozen defaults"
    return VerifiedCertificate(
        directions={
            name: MobilityInterval(0.0, 1.0, "undetermined") for name in names
        },
        status="failure_safe",
        reasons=(str(reason) or "unspecified verified-interval failure",),
        parameters={"lambda": LAMBDA},
    )


def unsupported_continuous_uncertainty(
    reason: str,
    directions: Mapping[str, Sequence[float]] = DEFAULT_DIRECTIONS,
) -> VerifiedCertificate:
    return failure_safe_certificate(
        f"unsupported_continuous_uncertainty: {reason}", directions
    )


def _zeros(rows: int, cols: int) -> IntervalMatrix:
    return [[Interval.point(0.0) for _ in range(cols)] for _ in range(rows)]


def _identity(size: int) -> IntervalMatrix:
    return [
        [Interval.point(1.0 if row == col else 0.0) for col in range(size)]
        for row in range(size)
    ]


def _transpose(matrix: IntervalMatrix) -> IntervalMatrix:
    if not matrix or any(len(row) != len(matrix[0]) for row in matrix):
        raise ValueError("interval transpose requires a nonempty rectangular matrix")
    return [[matrix[row][col] for row in range(len(matrix))] for col in range(len(matrix[0]))]


def _matmul(
    left: IntervalMatrix,
    right: IntervalMatrix,
    budget: _EventBudget | None = None,
) -> IntervalMatrix:
    if not left or not right or len(left[0]) != len(right):
        raise ValueError("incompatible interval matrix shapes")
    if any(len(row) != len(left[0]) for row in left) or any(
        len(row) != len(right[0]) for row in right
    ):
        raise ValueError("interval matrices must be rectangular")
    result = _zeros(len(left), len(right[0]))
    for row in range(len(left)):
        if budget is not None:
            budget.check("interval matrix multiplication")
        for col in range(len(right[0])):
            total = Interval.point(0.0)
            for inner in range(len(right)):
                total = total + left[row][inner] * right[inner][col]
            result[row][col] = total
    if budget is not None:
        budget.check("interval matrix multiplication completion")
    return result


def _gram(rows: IntervalMatrix) -> IntervalMatrix:
    """B B^T with interval-square tightening on its diagonal."""

    if not rows or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Gram matrix requires rectangular interval rows")
    result = _zeros(len(rows), len(rows))
    for i in range(len(rows)):
        for j in range(len(rows)):
            total = Interval.point(0.0)
            for k in range(len(rows[0])):
                term = rows[i][k].square() if i == j else rows[i][k] * rows[j][k]
                total = total + term
            result[i][j] = total
    return result


def _interval_inverse(
    matrix: IntervalMatrix, budget: _EventBudget | None = None
) -> IntervalMatrix:
    """Natural interval Gauss-Jordan enclosure with deterministic pivots."""

    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("interval inverse requires a nonempty square matrix")
    augmented = [list(row) + list(_identity(size)[index]) for index, row in enumerate(matrix)]
    for col in range(size):
        if budget is not None:
            budget.check("interval Gauss-Jordan pivot")
        pivot = max(
            range(col, size),
            key=lambda row: (abs(augmented[row][col].midpoint), -row),
        )
        if augmented[pivot][col].contains_zero():
            raise ValueError(f"interval inverse pivot {col} contains zero")
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        pivot_value = augmented[col][col]
        augmented[col] = [item / pivot_value for item in augmented[col]]
        for row in range(size):
            if row == col:
                continue
            if budget is not None:
                budget.check("interval Gauss-Jordan elimination")
            factor = augmented[row][col]
            augmented[row] = [
                current - factor * reference
                for current, reference in zip(augmented[row], augmented[col])
            ]
    if budget is not None:
        budget.check("interval Gauss-Jordan completion")
    return [row[size:] for row in augmented]


def _point_velocity_rows(point: tuple[Interval, Interval, Interval], ell: float) -> IntervalMatrix:
    x, y, z = point
    inv_ell = 1.0 / ell
    return [
        [as_interval(1.0), as_interval(0.0), as_interval(0.0), as_interval(0.0), z * inv_ell, -y * inv_ell],
        [as_interval(0.0), as_interval(1.0), as_interval(0.0), -z * inv_ell, as_interval(0.0), x * inv_ell],
        [as_interval(0.0), as_interval(0.0), as_interval(1.0), y * inv_ell, -x * inv_ell, as_interval(0.0)],
    ]


def _normalized_rows(patch: UncertainPatch, mode: str, ell: float) -> IntervalMatrix:
    sticking = _point_velocity_rows(patch.point_m, ell)
    if mode == "sticking":
        return sticking
    if mode != "normal":
        raise ValueError(f"unsupported contact mode: {mode}")
    return [[
        sum((sticking[row][col] * patch.normal[row] for row in range(3)), Interval.point(0.0))
        for col in range(6)
    ]]


def _patch_projector(
    patch: UncertainPatch, mode: str, ell: float, budget: _EventBudget
) -> IntervalMatrix:
    rows = _normalized_rows(patch, mode, ell)
    budget.check("constraint row construction")
    gram = _gram(rows)
    budget.check("Gram construction")
    gram_inverse = _interval_inverse(gram, budget)
    left = _matmul(_transpose(rows), gram_inverse, budget)
    projector = _matmul(left, rows, budget)
    budget.check("patch projector completion")
    return projector


def _validate_patches(patches: Sequence[UncertainPatch]) -> tuple[UncertainPatch, ...]:
    items = tuple(patches)
    if not items or any(not isinstance(item, UncertainPatch) for item in items):
        raise ValueError("at least one UncertainPatch is required")
    ids = [item.patch_id for item in items]
    components = [(item.hand_link, item.surface_component) for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("patch_id values must be unique")
    if len(components) != len(set(components)):
        raise ValueError("patches must be deduplicated by hand link and surface component")
    return items


def _operator(
    patches: Sequence[UncertainPatch],
    modes: Mapping[str, str],
    ell: float,
    budget: _EventBudget,
) -> IntervalMatrix:
    items = _validate_patches(patches)
    if set(modes) != {item.patch_id for item in items}:
        raise ValueError("mode keys must exactly match patch IDs")
    result = _zeros(6, 6)
    for patch in items:
        budget.check("patch operator")
        block = _patch_projector(patch, modes[patch.patch_id], ell, budget)
        result = [
            [result[i][j] + block[i][j] for j in range(6)] for i in range(6)
        ]
    budget.check("constraint operator completion")
    return result


def _fixed_left_matmul(
    left: Sequence[Sequence[float]],
    right: IntervalMatrix,
    budget: _EventBudget | None = None,
) -> IntervalMatrix:
    fixed = [[Interval.point(value) for value in row] for row in left]
    return _matmul(fixed, right, budget)


def _matvec(matrix: IntervalMatrix, vector: Sequence[float]) -> list[Interval]:
    if any(len(row) != len(vector) for row in matrix):
        raise ValueError("incompatible interval matrix/vector shapes")
    result: list[Interval] = []
    for row in matrix:
        total = Interval.point(0.0)
        for coefficient, value in zip(row, vector):
            total = total + coefficient * value
        result.append(total)
    return result


def _fixed_matvec(matrix: Sequence[Sequence[float]], vector: Sequence[Interval]) -> list[Interval]:
    if any(len(row) != len(vector) for row in matrix):
        raise ValueError("incompatible fixed matrix/interval vector shapes")
    result: list[Interval] = []
    for row in matrix:
        total = Interval.point(0.0)
        for coefficient, value in zip(row, vector):
            total = total + value * coefficient
        result.append(total)
    return result


def _upward_sum(values: Sequence[float]) -> float:
    """Upper-bound a sum of finite nonnegative binary64 values."""

    total = 0.0
    for raw in values:
        value = _finite(raw, "nonnegative sum term")
        if value < 0.0:
            raise ValueError("directed upward sum requires nonnegative terms")
        total = _up(total + value)
    return total


def _contraction_radius(eta: float, error: IntervalMatrix) -> tuple[float, float]:
    """Return a proof-safe radius and outward upper bound on ||error||_inf."""

    eta = _finite(eta, "contraction eta")
    if eta < 0.0 or not error:
        raise ValueError("contraction eta/error is invalid")
    q = max(_upward_sum([item.max_abs for item in row]) for row in error)
    if not math.isfinite(q) or q >= 1.0:
        raise ValueError(f"interval residual contraction failed: q={q!r}")
    denominator = Interval.point(1.0) - Interval.point(q)
    if denominator.lower <= 0.0:
        raise ValueError("interval residual contraction denominator is not positive")
    radius = (Interval.point(eta) / denominator).upper
    return radius, q


def _verified_mobility(
    direction: Sequence[float],
    operator: IntervalMatrix,
    budget: _EventBudget,
    lambda_: float = LAMBDA,
) -> tuple[Interval, float]:
    if len(operator) != 6 or any(len(row) != 6 for row in operator):
        raise ValueError("mobility operator must be 6x6")
    unit = _validate_directions({"query": direction})["query"]
    system = [
        [
            Interval.point(1.0 if i == j else 0.0) + operator[i][j] * lambda_
            for j in range(6)
        ]
        for i in range(6)
    ]
    budget.check("mobility system construction")
    midpoint = [[item.midpoint for item in row] for row in system]
    preconditioner = inverse(midpoint)
    budget.check("mobility midpoint inverse")
    x0 = [sum(preconditioner[i][j] * unit[j] for j in range(6)) for i in range(6)]

    residual_ax0 = _matvec(system, x0)
    residual = [Interval.point(unit[i]) - residual_ax0[i] for i in range(6)]
    z = _fixed_matvec(preconditioner, residual)
    eta = max(item.max_abs for item in z)
    budget.check("mobility residual")

    ra = _fixed_left_matmul(preconditioner, system, budget)
    error = [
        [Interval.point(1.0 if i == j else 0.0) - ra[i][j] for j in range(6)]
        for i in range(6)
    ]
    radius, q = _contraction_radius(eta, error)
    budget.check("mobility contraction proof")
    enclosure = [Interval(_down(value - radius), _up(value + radius)) for value in x0]
    mobility = Interval.point(0.0)
    for coefficient, component in zip(unit, enclosure):
        mobility = mobility + component * coefficient
    result = mobility.intersect(Interval(0.0, 1.0))
    budget.check("mobility direction completion")
    return result, q


def _split_patch(patch: UncertainPatch, axis: int) -> tuple[UncertainPatch, UncertainPatch]:
    interval = patch.point_m[axis]
    midpoint = interval.midpoint
    if midpoint <= interval.lower or midpoint >= interval.upper:
        raise ValueError("selected interval cannot be split in binary64")
    left_point = list(patch.point_m)
    right_point = list(patch.point_m)
    left_point[axis] = Interval(interval.lower, midpoint)
    right_point[axis] = Interval(midpoint, interval.upper)
    return (
        replace(patch, point_m=tuple(left_point)),  # type: ignore[arg-type]
        replace(patch, point_m=tuple(right_point)),  # type: ignore[arg-type]
    )


def _subdivide(
    patches: tuple[UncertainPatch, ...],
    max_boxes: int,
    ell: float,
    budget: _EventBudget,
) -> tuple[list[tuple[UncertainPatch, ...]], list[str]]:
    leaves = [patches]
    trace: list[str] = []
    while len(leaves) < max_boxes:
        budget.check("subdivision")
        candidates: list[tuple[float, int, str, int, int]] = []
        for leaf_index, leaf in enumerate(leaves):
            for patch_index, patch in enumerate(leaf):
                for axis, interval in enumerate(patch.point_m):
                    width = interval.width / ell
                    if width > 0.0 and interval.lower < interval.midpoint < interval.upper:
                        candidates.append((-width, leaf_index, patch.patch_id, axis, patch_index))
        budget.check("subdivision candidate scan")
        if not candidates:
            break
        _, leaf_index, _, axis, patch_index = min(candidates)
        leaf = leaves[leaf_index]
        trace.append(f"{leaf_index}:{leaf[patch_index].patch_id}:{axis}")
        left_patch, right_patch = _split_patch(leaf[patch_index], axis)
        left_leaf = list(leaf)
        right_leaf = list(leaf)
        left_leaf[patch_index] = left_patch
        right_leaf[patch_index] = right_patch
        leaves[leaf_index : leaf_index + 1] = [tuple(left_leaf), tuple(right_leaf)]
        budget.check("subdivision split completion")
    budget.check("subdivision completion")
    return leaves, trace


def _hull_certificates(
    certificates: Sequence[VerifiedCertificate],
    status: str,
    parent_ids: Sequence[str] = (),
    parameters: Mapping[str, object] | None = None,
) -> VerifiedCertificate:
    children = tuple(certificates)
    if not children:
        raise ValueError("certificate hull requires at least one child")
    names = set(children[0].directions)
    if any(set(child.directions) != names for child in children):
        raise ValueError("certificate hull direction namespaces disagree")
    directions: dict[str, MobilityInterval] = {}
    for name in sorted(names):
        lower = min(child.directions[name].lower for child in children)
        upper = max(child.directions[name].upper for child in children)
        directions[name] = MobilityInterval(lower, upper, classify_interval(lower, upper))
    if any(child.status == "failure_safe" for child in children):
        return failure_safe_certificate(
            "; ".join(reason for child in children for reason in child.reasons)
            or "child hull contains a failure-safe certificate",
            {name: DEFAULT_DIRECTIONS.get(name, (1, 0, 0, 0, 0, 0)) for name in names},
        )
    merged_parameters = dict(parameters or {})
    merged_parameters["child_count"] = len(children)
    return VerifiedCertificate(
        directions=directions,
        status=status,
        parent_ids=tuple(parent_ids),
        parameters=merged_parameters,
    )


def _apply_parent_hull(
    child: VerifiedCertificate,
    parents: Sequence[VerifiedCertificate],
    parent_ids: Sequence[str],
) -> VerifiedCertificate:
    if not parents:
        return child
    ids = tuple(parent_ids) if parent_ids else tuple(f"parent-{i}" for i in range(len(parents)))
    if len(ids) != len(parents):
        raise ValueError("parent_ids length must match parent certificates")
    return _hull_certificates(
        (child, *parents),
        "parent_hull",
        ids,
        {**dict(child.parameters), "parent_count": len(parents)},
    )


def _compile_mode_over_leaves(
    leaves: Sequence[tuple[UncertainPatch, ...]],
    modes: Mapping[str, str],
    ell: float,
    directions: Mapping[str, tuple[float, ...]],
    budget: _EventBudget,
    split_trace: Sequence[str],
) -> VerifiedCertificate:
    if not leaves:
        raise ValueError("mode evaluation requires at least one leaf")
    hulls: dict[str, Interval] = {}
    max_q = 0.0
    for leaf in leaves:
        budget.check("leaf evaluation")
        operator = _operator(leaf, modes, ell, budget)
        for name, direction in directions.items():
            budget.check(f"direction {name}")
            interval, q = _verified_mobility(direction, operator, budget)
            max_q = max(max_q, q)
            hulls[name] = interval if name not in hulls else hulls[name].hull(interval)
            budget.check(f"direction {name} hull")
        budget.check("leaf completion")
    mobility = {
        name: MobilityInterval(
            hulls[name].lower,
            hulls[name].upper,
            classify_interval(hulls[name].lower, hulls[name].upper),
        )
        for name in sorted(hulls)
    }
    return VerifiedCertificate(
        directions=mobility,
        status="verified_interval",
        parameters={
            "ell_m": ell,
            "lambda": LAMBDA,
            "leaf_count": len(leaves),
            "event_box_count": len(leaves),
            "split_trace": list(split_trace),
            "max_contraction_q": max_q,
        },
    )


def _start_budget(clock: Clock, timeout_s: float) -> tuple[_EventBudget, float]:
    start = _finite(clock(), "clock")
    timeout = _finite(timeout_s, "timeout_s")
    if timeout <= 0.0 or timeout > 60.0:
        raise ValueError("timeout_s must lie in (0,60]")
    budget = _EventBudget(start + timeout, clock)
    budget.check("event start")
    return budget, timeout


def _validate_max_boxes(max_boxes: int) -> int:
    if (
        isinstance(max_boxes, bool)
        or not isinstance(max_boxes, int)
        or not 1 <= max_boxes <= 128
    ):
        raise ValueError("max_boxes must be an integer in [1,128]")
    return max_boxes


def compile_verified_mode_certificate(
    patches: Sequence[UncertainPatch],
    modes: Mapping[str, str],
    ell_m: float,
    directions: Mapping[str, Sequence[float]] = DEFAULT_DIRECTIONS,
    *,
    max_boxes: int = 128,
    timeout_s: float = 60.0,
    parent_certificates: Sequence[VerifiedCertificate] = (),
    parent_ids: Sequence[str] = (),
    clock: Clock = time.process_time,
) -> VerifiedCertificate:
    """Enclose one fixed mode assignment over a joint contact-point box."""

    try:
        budget, _ = _start_budget(clock, timeout_s)
        items = _validate_patches(patches)
        budget.check("patch validation")
        ell = characteristic_length(ell_m)
        max_boxes = _validate_max_boxes(max_boxes)
        normalized_directions = _validate_directions(directions)
        budget.check("direction validation")
        if set(modes) != {patch.patch_id for patch in items} or any(
            mode not in {"normal", "sticking"} for mode in modes.values()
        ):
            raise ValueError("modes must exactly assign normal/sticking to every patch")
        budget.check("mode validation")
        leaves, trace = _subdivide(items, max_boxes, ell, budget)
        result = _compile_mode_over_leaves(
            leaves,
            dict(modes),
            ell,
            normalized_directions,
            budget,
            trace,
        )
        result = _apply_parent_hull(result, parent_certificates, parent_ids)
        budget.check("mode certificate return")
        return result
    except Exception as exc:
        return failure_safe_certificate(f"{type(exc).__name__}: {exc}", directions)


def compile_verified_outer_certificate(
    patches: Sequence[UncertainPatch],
    ell_m: float,
    directions: Mapping[str, Sequence[float]] = DEFAULT_DIRECTIONS,
    *,
    max_boxes: int = 128,
    timeout_s: float = 60.0,
    parent_certificates: Sequence[VerifiedCertificate] = (),
    parent_ids: Sequence[str] = (),
    clock: Clock = time.process_time,
) -> VerifiedCertificate:
    """Enclose all normal/sticking assignments and contact points.

    All-normal and all-sticking extremes are independently verified and then
    hulled.  The Loewner argument from M0 places every hybrid mode between
    those two endpoints for each fixed geometry.
    """

    try:
        budget, _ = _start_budget(clock, timeout_s)
        items = _validate_patches(patches)
        budget.check("patch validation")
        ell = characteristic_length(ell_m)
        max_boxes = _validate_max_boxes(max_boxes)
        normalized_directions = _validate_directions(directions)
        budget.check("direction validation")
        leaves, trace = _subdivide(items, max_boxes, ell, budget)
        all_normal = {patch.patch_id: "normal" for patch in items}
        all_sticking = {patch.patch_id: "sticking" for patch in items}
        normal = _compile_mode_over_leaves(
            leaves, all_normal, ell, normalized_directions, budget, trace
        )
        sticking = _compile_mode_over_leaves(
            leaves, all_sticking, ell, normalized_directions, budget, trace
        )
        result = _hull_certificates(
            (normal, sticking),
            "finite_hull",
            parameters={
                "mode_extremes": ["all_normal", "all_sticking"],
                "event_box_count": len(leaves),
                "leaf_mode_evaluations": 2 * len(leaves),
                "split_trace": list(trace),
            },
        )
        result = _apply_parent_hull(result, parent_certificates, parent_ids)
        budget.check("outer certificate return")
        return result
    except Exception as exc:
        return failure_safe_certificate(f"{type(exc).__name__}: {exc}", directions)


def compile_verified_finite_hypotheses(
    hypotheses: Sequence[tuple[Sequence[UncertainPatch], Mapping[str, str]]],
    ell_m: float,
    directions: Mapping[str, Sequence[float]] = DEFAULT_DIRECTIONS,
    *,
    max_boxes: int = 128,
    timeout_s: float = 60.0,
    parent_certificates: Sequence[VerifiedCertificate] = (),
    parent_ids: Sequence[str] = (),
    clock: Clock = time.process_time,
) -> VerifiedCertificate:
    """Explicit finite symmetry/mode hull; sampling is never a certificate."""

    try:
        budget, _ = _start_budget(clock, timeout_s)
        if not hypotheses:
            raise ValueError("finite hypothesis set cannot be empty")
        max_boxes = _validate_max_boxes(max_boxes)
        if len(hypotheses) > max_boxes:
            raise ValueError("finite hypotheses exceed the event-wide box budget")
        normalized_directions = _validate_directions(directions)
        ell = characteristic_length(ell_m)
        budget.check("finite hypothesis validation")
        base, remainder = divmod(max_boxes, len(hypotheses))
        allocations = [base + int(index < remainder) for index in range(len(hypotheses))]
        children: list[VerifiedCertificate] = []
        event_box_count = 0
        split_traces: list[list[str]] = []
        for index, ((raw_patches, modes), allocation) in enumerate(zip(hypotheses, allocations)):
            budget.check(f"finite hypothesis {index}")
            items = _validate_patches(raw_patches)
            if set(modes) != {patch.patch_id for patch in items} or any(
                mode not in {"normal", "sticking"} for mode in modes.values()
            ):
                raise ValueError("finite hypothesis has an invalid mode assignment")
            leaves, trace = _subdivide(items, allocation, ell, budget)
            child = _compile_mode_over_leaves(
                leaves,
                dict(modes),
                ell,
                normalized_directions,
                budget,
                trace,
            )
            children.append(child)
            event_box_count += len(leaves)
            split_traces.append(list(trace))
        if event_box_count > max_boxes:
            raise AssertionError("finite hypotheses exceeded the event-wide box budget")
        result = _hull_certificates(
            children,
            "finite_hull",
            parameters={
                "hypothesis_count": len(children),
                "event_box_count": event_box_count,
                "box_allocation": allocations,
                "split_traces": split_traces,
            },
        )
        result = _apply_parent_hull(result, parent_certificates, parent_ids)
        budget.check("finite hypothesis certificate return")
        return result
    except Exception as exc:
        return failure_safe_certificate(f"{type(exc).__name__}: {exc}", directions)
