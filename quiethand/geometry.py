"""Constraint-row and coordinate-transform construction."""

from __future__ import annotations

import math
from typing import Sequence

from .linalg import Matrix, inverse, matmul, matvec, row_space_projector, transpose
from .models import PatchConstraint


def characteristic_length(native_aabb_diagonal_m: float) -> float:
    value = float(native_aabb_diagonal_m)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("native AABB diagonal must be finite and nonnegative")
    return max(value, 1e-3)


def point_velocity_rows(point_m: Sequence[float]) -> Matrix:
    """Rows for v_point = v + omega x r with twist [v, omega]."""

    if len(point_m) != 3:
        raise ValueError("point must have three coordinates")
    x, y, z = map(float, point_m)
    if not all(math.isfinite(value) for value in (x, y, z)):
        raise ValueError("point coordinates must be finite")
    return [
        [1.0, 0.0, 0.0, 0.0, z, -y],
        [0.0, 1.0, 0.0, -z, 0.0, x],
        [0.0, 0.0, 1.0, y, -x, 0.0],
    ]


def normalized_constraint_rows(patch: PatchConstraint, mode: str, ell_m: float) -> Matrix:
    """Build J M^-1/2 in normalized screw coordinates."""

    ell = characteristic_length(ell_m)
    stick = point_velocity_rows(patch.point_m)
    if mode == "normal":
        rows = [[sum(patch.normal[i] * stick[i][j] for i in range(3)) for j in range(6)]]
    elif mode == "sticking":
        rows = stick
    else:
        raise ValueError(f"unsupported contact mode: {mode}")
    inverse_sqrt_metric = [1.0, 1.0, 1.0, 1.0 / ell, 1.0 / ell, 1.0 / ell]
    return [[value * inverse_sqrt_metric[col] for col, value in enumerate(row)] for row in rows]


def patch_projector(patch: PatchConstraint, mode: str, ell_m: float) -> Matrix:
    rows = normalized_constraint_rows(patch, mode, ell_m)
    expected_rank = 1 if mode == "normal" else 3
    return row_space_projector(rows, expected_rank)


def rotate_patch(patch: PatchConstraint, rotation: Sequence[Sequence[float]]) -> PatchConstraint:
    validate_rotation(rotation)
    return PatchConstraint(
        patch_id=patch.patch_id,
        hand_link=patch.hand_link,
        surface_component=patch.surface_component,
        point_m=tuple(matvec(rotation, patch.point_m)),
        normal=tuple(matvec(rotation, patch.normal)),
    )


def twist_rotation(rotation: Sequence[Sequence[float]]) -> Matrix:
    validate_rotation(rotation)
    return [
        [float(rotation[i % 3][j % 3]) if i // 3 == j // 3 else 0.0 for j in range(6)]
        for i in range(6)
    ]


def rotate_operator(operator: Sequence[Sequence[float]], rotation: Sequence[Sequence[float]]) -> Matrix:
    transform = twist_rotation(rotation)
    return matmul(matmul(transform, operator), transpose(transform))


def validate_rotation(rotation: Sequence[Sequence[float]], tol: float = 1e-10) -> None:
    if len(rotation) != 3 or any(len(row) != 3 for row in rotation):
        raise ValueError("rotation must be 3x3")
    values = [[float(value) for value in row] for row in rotation]
    if not all(math.isfinite(value) for row in values for value in row):
        raise ValueError("rotation must be finite")
    gram = matmul(transpose(values), values)
    error = max(
        abs(gram[i][j] - (1.0 if i == j else 0.0)) for i in range(3) for j in range(3)
    )
    determinant = (
        values[0][0] * (values[1][1] * values[2][2] - values[1][2] * values[2][1])
        - values[0][1] * (values[1][0] * values[2][2] - values[1][2] * values[2][0])
        + values[0][2] * (values[1][0] * values[2][1] - values[1][1] * values[2][0])
    )
    if error > tol or abs(determinant - 1.0) > tol:
        raise ValueError("rotation must lie in SO(3)")


def transform_patch(
    patch: PatchConstraint,
    rotation: Sequence[Sequence[float]],
    translation_m: Sequence[float],
) -> PatchConstraint:
    """Express a support patch in x' = R x + t coordinates."""

    validate_rotation(rotation)
    if len(translation_m) != 3 or not all(math.isfinite(float(v)) for v in translation_m):
        raise ValueError("translation must be a finite 3-vector")
    rotated_point = matvec(rotation, patch.point_m)
    point = tuple(rotated_point[i] + float(translation_m[i]) for i in range(3))
    return PatchConstraint(
        patch_id=patch.patch_id,
        hand_link=patch.hand_link,
        surface_component=patch.surface_component,
        point_m=point,
        normal=tuple(matvec(rotation, patch.normal)),
    )


def twist_adjoint(
    rotation: Sequence[Sequence[float]], translation_m: Sequence[float]
) -> Matrix:
    """Q for xi'=[R v + t x R omega, R omega]."""

    validate_rotation(rotation)
    if len(translation_m) != 3 or not all(math.isfinite(float(v)) for v in translation_m):
        raise ValueError("translation must be a finite 3-vector")
    x, y, z = map(float, translation_m)
    skew_translation = [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]]
    coupling = matmul(skew_translation, rotation)
    return [
        [
            float(rotation[i][j]) if i < 3 and j < 3
            else coupling[i][j - 3] if i < 3 and j >= 3
            else float(rotation[i - 3][j - 3]) if i >= 3 and j >= 3
            else 0.0
            for j in range(6)
        ]
        for i in range(6)
    ]


def normalized_rows_with_whitener(
    raw_rows: Sequence[Sequence[float]], whitener: Sequence[Sequence[float]]
) -> Matrix:
    """Return J W^-1 for M=W^T W in arbitrary screw coordinates."""

    if len(whitener) != 6 or any(len(row) != 6 for row in whitener):
        raise ValueError("whitener must be 6x6")
    return matmul(raw_rows, inverse(whitener))
