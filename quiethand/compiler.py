"""Exact finite-mode QuietHand C-OAME compiler for the M0 engineering gate."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .linalg import Matrix, add, identity, is_finite_matrix, is_finite_vector, scale, solve, symmetric_eigenvalues
from .geometry import characteristic_length, patch_projector
from .models import MobilityCertificate, MobilityInterval, PatchConstraint

LAMBDA = 9.0
TAU_RESTRICTED = 0.20
TAU_FREE = 0.80
DEFAULT_DIRECTIONS: dict[str, tuple[float, ...]] = {
    "v_x": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "v_y": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    "v_z": (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
    "omega_x": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
    "omega_y": (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    "omega_z": (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
}


def _validate_patch_set(patches: Sequence[PatchConstraint]) -> tuple[PatchConstraint, ...]:
    items = tuple(patches)
    if not items:
        raise ValueError("at least one support patch is required")
    ids = [patch.patch_id for patch in items]
    components = [(patch.hand_link, patch.surface_component) for patch in items]
    if len(ids) != len(set(ids)):
        raise ValueError("patch_id values must be unique")
    if len(components) != len(set(components)):
        raise ValueError("patches must be deduplicated by hand_link and surface_component")
    return items


def constraint_operator(
    patches: Sequence[PatchConstraint], modes: Mapping[str, str], ell_m: float
) -> Matrix:
    items = _validate_patch_set(patches)
    ell = characteristic_length(ell_m)
    expected_ids = {patch.patch_id for patch in items}
    if set(modes) != expected_ids:
        raise ValueError(
            f"mode keys must exactly match patch IDs; expected {sorted(expected_ids)}, got {sorted(modes)}"
        )
    blocks = []
    for patch in items:
        blocks.append(patch_projector(patch, modes[patch.patch_id], ell))
    operator = add(*blocks)
    if not is_finite_matrix(operator):
        raise ValueError("constraint operator contains a non-finite value")
    return operator


def directional_mobility(
    direction: Sequence[float], operator: Sequence[Sequence[float]], lambda_: float = LAMBDA
) -> float:
    if len(direction) != 6 or len(operator) != 6 or any(len(row) != 6 for row in operator):
        raise ValueError("mobility requires a 6D direction and 6x6 operator")
    if not is_finite_vector(direction) or not is_finite_matrix(operator):
        raise ValueError("mobility inputs must be finite")
    norm = math.hypot(*(float(value) for value in direction))
    if not math.isfinite(norm) or norm <= 1e-15:
        raise ValueError("direction must be finite and nonzero")
    if not math.isfinite(float(lambda_)) or lambda_ <= 0.0:
        raise ValueError("lambda must be finite and positive")
    unit_direction = [float(value) / norm for value in direction]
    system = add(identity(6), scale(operator, lambda_))
    solution = solve(system, unit_direction)
    value = sum(a * b for a, b in zip(unit_direction, solution))
    if not math.isfinite(value):
        raise ValueError("mobility solve produced a non-finite value")
    if value < -1e-10 or value > 1.0 + 1e-10:
        raise ValueError(f"mobility escaped [0,1]: {value}")
    return min(1.0, max(0.0, value))


def classify_interval(lower: float, upper: float) -> str:
    if upper <= TAU_RESTRICTED:
        return "restricted"
    if lower >= TAU_FREE:
        return "free"
    return "undetermined"


def compile_mode_certificate(
    patches: Sequence[PatchConstraint],
    modes: Mapping[str, str],
    ell_m: float,
    directions: Mapping[str, Sequence[float]] = DEFAULT_DIRECTIONS,
) -> MobilityCertificate:
    try:
        operator = constraint_operator(patches, modes, ell_m)
        intervals: dict[str, MobilityInterval] = {}
        for name, direction in directions.items():
            value = directional_mobility(direction, operator)
            intervals[name] = MobilityInterval(value, value, classify_interval(value, value))
        return MobilityCertificate(
            directions=intervals,
            status="mode_point",
            parameters={"ell_m": characteristic_length(ell_m), "lambda": LAMBDA},
        )
    except Exception as exc:
        return failure_safe_certificate(f"{type(exc).__name__}: {exc}", directions)


def failure_safe_certificate(
    reason: str,
    directions: Mapping[str, Sequence[float]] = DEFAULT_DIRECTIONS,
) -> MobilityCertificate:
    try:
        names = tuple(directions)
    except (TypeError, AttributeError):
        names = ()
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        names = tuple(DEFAULT_DIRECTIONS)
        reason = f"{reason}; invalid or empty direction namespace, used frozen defaults"
    return MobilityCertificate(
        directions={
            name: MobilityInterval(0.0, 1.0, "undetermined") for name in names
        },
        status="failure_safe",
        reasons=(reason,),
        parameters={"lambda": LAMBDA},
    )


def compile_exact_outer_certificate(
    patches: Sequence[PatchConstraint],
    ell_m: float,
    directions: Mapping[str, Sequence[float]] = DEFAULT_DIRECTIONS,
) -> MobilityCertificate:
    """Hull all-normal/all-sticking endpoints using Loewner monotonicity.

    Every admissible per-patch mode is between its normal and sticking
    projector.  The regularized inverse reverses that PSD order, so the two
    endpoints are an exact outer interval for finite normal/sticking mode
    uncertainty.  Any invalid input returns a full failure-safe certificate.
    """

    try:
        items = _validate_patch_set(patches)
        all_normal = {patch.patch_id: "normal" for patch in items}
        all_sticking = {patch.patch_id: "sticking" for patch in items}
        normal_operator = constraint_operator(items, all_normal, ell_m)
        sticking_operator = constraint_operator(items, all_sticking, ell_m)
        difference = add(sticking_operator, scale(normal_operator, -1.0))
        minimum_order_eigenvalue = min(symmetric_eigenvalues(difference))
        if minimum_order_eigenvalue < -1e-10:
            raise ValueError(
                f"sticking operator does not PSD-dominate normal operator: {minimum_order_eigenvalue}"
            )
        intervals: dict[str, MobilityInterval] = {}
        for name, direction in directions.items():
            lower = directional_mobility(direction, sticking_operator)
            upper = directional_mobility(direction, normal_operator)
            if lower > upper + 1e-10:
                raise ValueError(f"mode monotonicity failed for {name}")
            intervals[name] = MobilityInterval(lower, upper, classify_interval(lower, upper))
        return MobilityCertificate(
            directions=intervals,
            status="exact_outer",
            parameters={
                "ell_m": characteristic_length(ell_m),
                "lambda": LAMBDA,
                "minimum_order_eigenvalue": minimum_order_eigenvalue,
                "tau_free": TAU_FREE,
                "tau_restricted": TAU_RESTRICTED,
            },
        )
    except Exception as exc:  # failure isolation is part of the public contract
        return failure_safe_certificate(f"{type(exc).__name__}: {exc}", directions)
