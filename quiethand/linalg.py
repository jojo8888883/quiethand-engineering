"""Small deterministic dense linear algebra for the CPU-only M0 gate.

The matrices are at most 6x6.  Keeping this module dependency-free lets M0 run
in a clean Python installation; later verified interval work is a separate M1.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

Matrix = list[list[float]]
Vector = list[float]


def is_finite_vector(vector: Sequence[float]) -> bool:
    return all(math.isfinite(float(value)) for value in vector)


def is_finite_matrix(matrix: Sequence[Sequence[float]]) -> bool:
    return bool(matrix) and all(is_finite_vector(row) for row in matrix)


def identity(size: int) -> Matrix:
    return [[1.0 if row == col else 0.0 for col in range(size)] for row in range(size)]


def transpose(matrix: Sequence[Sequence[float]]) -> Matrix:
    if not matrix:
        return []
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("matrix rows have inconsistent widths")
    return [[float(matrix[row][col]) for row in range(len(matrix))] for col in range(width)]


def matmul(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> Matrix:
    if not left or not right:
        raise ValueError("matmul requires nonempty matrices")
    inner = len(left[0])
    if any(len(row) != inner for row in left) or len(right) != inner:
        raise ValueError("incompatible matrix shapes")
    width = len(right[0])
    if any(len(row) != width for row in right):
        raise ValueError("matrix rows have inconsistent widths")
    return [
        [sum(float(left[i][k]) * float(right[k][j]) for k in range(inner)) for j in range(width)]
        for i in range(len(left))
    ]


def matvec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Vector:
    if any(len(row) != len(vector) for row in matrix):
        raise ValueError("incompatible matrix/vector shapes")
    return [sum(float(a) * float(b) for a, b in zip(row, vector)) for row in matrix]


def add(*matrices: Sequence[Sequence[float]]) -> Matrix:
    if not matrices:
        raise ValueError("add requires at least one matrix")
    rows, cols = len(matrices[0]), len(matrices[0][0])
    if any(len(matrix) != rows or any(len(row) != cols for row in matrix) for matrix in matrices):
        raise ValueError("matrix shapes must match")
    return [[sum(float(matrix[i][j]) for matrix in matrices) for j in range(cols)] for i in range(rows)]


def scale(matrix: Sequence[Sequence[float]], scalar: float) -> Matrix:
    return [[float(scalar) * float(item) for item in row] for row in matrix]


def symmetrize(matrix: Sequence[Sequence[float]]) -> Matrix:
    if len(matrix) != len(matrix[0]) or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("symmetrize requires a square matrix")
    return scale(add(matrix, transpose(matrix)), 0.5)


def max_abs_difference(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float:
    if len(left) != len(right) or any(len(a) != len(b) for a, b in zip(left, right)):
        raise ValueError("matrix shapes must match")
    return max(abs(float(a) - float(b)) for row_a, row_b in zip(left, right) for a, b in zip(row_a, row_b))


def solve(matrix: Sequence[Sequence[float]], rhs: Sequence[float], tol: float = 1e-14) -> Vector:
    """Solve a square system using deterministic partial-pivot Gauss-Jordan."""

    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix) or len(rhs) != size:
        raise ValueError("solve requires a nonempty square system")
    if not is_finite_matrix(matrix) or not is_finite_vector(rhs):
        raise ValueError("solve requires finite inputs")
    augmented = [list(map(float, row)) + [float(rhs[i])] for i, row in enumerate(matrix)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: (abs(augmented[row][col]), -row))
        if abs(augmented[pivot][col]) <= tol:
            raise ValueError("singular or rank-deficient system")
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        pivot_value = augmented[col][col]
        augmented[col] = [item / pivot_value for item in augmented[col]]
        for row in range(size):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor:
                augmented[row] = [
                    current - factor * reference
                    for current, reference in zip(augmented[row], augmented[col])
                ]
    result = [augmented[row][-1] for row in range(size)]
    if not is_finite_vector(result):
        raise ValueError("linear solve produced a non-finite result")
    return result


def inverse(matrix: Sequence[Sequence[float]], tol: float = 1e-14) -> Matrix:
    size = len(matrix)
    columns = [solve(matrix, [1.0 if i == j else 0.0 for i in range(size)], tol) for j in range(size)]
    return transpose(columns)


def row_rank(matrix: Sequence[Sequence[float]], tol: float = 1e-12) -> int:
    if not matrix:
        return 0
    work = [list(map(float, row)) for row in matrix]
    width = len(work[0])
    rank = 0
    for col in range(width):
        pivot = max(range(rank, len(work)), key=lambda row: (abs(work[row][col]), -row))
        if abs(work[pivot][col]) <= tol:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        pivot_value = work[rank][col]
        work[rank] = [item / pivot_value for item in work[rank]]
        for row in range(len(work)):
            if row == rank:
                continue
            factor = work[row][col]
            work[row] = [a - factor * b for a, b in zip(work[row], work[rank])]
        rank += 1
        if rank == len(work):
            break
    return rank


def row_space_projector(rows: Sequence[Sequence[float]], expected_rank: int) -> Matrix:
    """Return a row-space projector from a scale-stable orthonormal basis.

    Modified Gram-Schmidt is applied to individually normalized rows with a
    second re-orthogonalization pass.  Unlike the normal-equation formula, a
    harmless invertible rescaling of the input rows does not square the scale
    or condition number.
    """

    if not rows or any(len(row) != 6 for row in rows):
        raise ValueError("constraint rows must be a nonempty matrix with six columns")
    if len(rows) != expected_rank or not is_finite_matrix(rows):
        raise ValueError("constraint block does not have its required row rank")
    basis: list[Vector] = []
    for raw_row in rows:
        norm = math.hypot(*(float(value) for value in raw_row))
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError("constraint block contains a zero or non-finite row")
        vector = [float(value) / norm for value in raw_row]
        for _ in range(2):
            for unit in basis:
                coefficient = sum(a * b for a, b in zip(vector, unit))
                vector = [a - coefficient * b for a, b in zip(vector, unit)]
        residual = math.hypot(*vector)
        if not math.isfinite(residual) or residual <= 1e-12:
            raise ValueError("constraint block does not have its required row rank")
        basis.append([value / residual for value in vector])
    projector = [
        [sum(unit[i] * unit[j] for unit in basis) for j in range(6)]
        for i in range(6)
    ]
    return symmetrize(projector)


def quadratic(vector: Sequence[float], matrix: Sequence[Sequence[float]]) -> float:
    transformed = matvec(matrix, vector)
    return sum(float(a) * float(b) for a, b in zip(vector, transformed))


def symmetric_eigenvalues(matrix: Sequence[Sequence[float]], tol: float = 1e-15, max_sweeps: int = 100) -> Vector:
    """Jacobi eigenvalues for tiny symmetric matrices, used only as an M0 check."""

    if not is_finite_matrix(matrix):
        raise ValueError("eigendecomposition requires a finite matrix")
    work = symmetrize(matrix)
    size = len(work)
    for _ in range(max_sweeps * size * size):
        p, q = max(
            ((i, j) for i in range(size) for j in range(i + 1, size)),
            key=lambda pair: abs(work[pair[0]][pair[1]]),
            default=(0, 0),
        )
        if p == q or abs(work[p][q]) <= tol:
            break
        app, aqq, apq = work[p][p], work[q][q], work[p][q]
        angle = 0.5 * math.atan2(2.0 * apq, aqq - app)
        cosine, sine = math.cos(angle), math.sin(angle)
        for k in range(size):
            if k in (p, q):
                continue
            wkp, wkq = work[k][p], work[k][q]
            work[k][p] = work[p][k] = cosine * wkp - sine * wkq
            work[k][q] = work[q][k] = sine * wkp + cosine * wkq
        work[p][p] = cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq
        work[q][q] = sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq
        work[p][q] = work[q][p] = 0.0
    return sorted(work[i][i] for i in range(size))


def diagonal(values: Iterable[float]) -> Matrix:
    items = list(map(float, values))
    return [[items[i] if i == j else 0.0 for j in range(len(items))] for i in range(len(items))]
