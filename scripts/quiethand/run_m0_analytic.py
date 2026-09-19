#!/usr/bin/env python3
"""Run the six preregistered QuietHand M0 engineering checks."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quiethand.compiler import compile_exact_outer_certificate, compile_mode_certificate, constraint_operator, directional_mobility
from quiethand.geometry import normalized_constraint_rows, normalized_rows_with_whitener, point_velocity_rows, rotate_operator, rotate_patch, transform_patch, twist_adjoint, twist_rotation
from quiethand.linalg import add, diagonal, inverse, matmul, matvec, max_abs_difference, row_space_projector, scale, symmetric_eigenvalues
from quiethand.models import EvidenceEvent, EvidenceRecord, PatchConstraint, Provenance
from quiethand.outer import finite_hypothesis_hull, unresolved_continuous_uncertainty


SOURCE_PATHS = (
    "quiethand/__init__.py",
    "quiethand/models.py",
    "quiethand/linalg.py",
    "quiethand/geometry.py",
    "quiethand/compiler.py",
    "quiethand/outer.py",
    "quiethand/schema/evidence_event.schema.json",
    "scripts/quiethand/run_m0_analytic.py",
    "tests/test_quiethand_m0.py",
)


def _rotation(seed: int):
    rng = random.Random(seed)
    axis = [rng.uniform(-1.0, 1.0) for _ in range(3)]
    norm = math.sqrt(sum(value * value for value in axis))
    x, y, z = (value / norm for value in axis)
    angle = rng.uniform(-math.pi, math.pi)
    c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]


def _source_hashes() -> dict[str, str]:
    return {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in SOURCE_PATHS
    }


def _count_nulls(value) -> int:
    if value is None:
        return 1
    if isinstance(value, dict):
        return sum(_count_nulls(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count_nulls(item) for item in value)
    return 0


def build_single_process_report() -> dict:
    """Rebuild every M0 case once; the parent CLI invokes this in three processes."""

    provenance = Provenance("oracle", "synthetic-m0", "1", "QH-M0-001")
    event = EvidenceEvent(
        "synthetic:seq:0-1",
        "synthetic",
        "seq",
        0.0,
        1.0,
        "right",
        "left",
        "object",
        "object",
        (
            EvidenceRecord("roles", "observed", {"active": "right", "support": "left"}, "semantic", "ego", 0.5, provenance),
            EvidenceRecord("force", "unobservable", None, "N", "object", 0.5, provenance, failure_reason="sensor absent"),
        ),
    )
    event_roundtrip = EvidenceEvent.from_json(event.canonical_json())
    silent_null_count = sum(
        _count_nulls(record.value)
        for record in event_roundtrip.evidence
        if record.status in {"observed", "inferred", "human", "derived"}
    )
    invalid_payloads = []
    payload = json.loads(event.canonical_json())
    payload["event_id"] = 123
    invalid_payloads.append(payload)
    payload = json.loads(event.canonical_json())
    payload["evidence"][0]["provenance"]["source_name"] = 123
    invalid_payloads.append(payload)
    payload = json.loads(event.canonical_json())
    payload["evidence"][0]["timestamp_s"] = True
    invalid_payloads.append(payload)
    payload = json.loads(event.canonical_json())
    payload["evidence"][0]["depends_on"] = "roles"
    invalid_payloads.append(payload)
    payload = json.loads(event.canonical_json())
    payload["evidence"] = {"not": "an array"}
    invalid_payloads.append(payload)
    strict_type_rejections = 0
    for payload in invalid_payloads:
        try:
            EvidenceEvent.from_dict(payload)
        except ValueError:
            strict_type_rejections += 1

    point = (0.2, -0.1, 0.3)
    patch = PatchConstraint("p0", "left_index", "surface_0", point, (1.0, 2.0, -0.5))
    ell = 0.8
    certificate = compile_exact_outer_certificate([patch], ell)

    velocity, omega = [0.5, -0.7, 0.2], [1.1, -0.4, 0.9]
    observed_velocity = matvec(point_velocity_rows(point), velocity + omega)
    expected_cross = [
        omega[1] * point[2] - omega[2] * point[1],
        omega[2] * point[0] - omega[0] * point[2],
        omega[0] * point[1] - omega[1] * point[0],
    ]
    cross_product_error = max(
        abs(observed_velocity[i] - velocity[i] - expected_cross[i]) for i in range(3)
    )
    nx, ny, nz = patch.normal
    x, y, z = point
    covector = [
        nx,
        ny,
        nz,
        (-ny * z + nz * y) / ell,
        (nx * z - nz * x) / ell,
        (-nx * y + ny * x) / ell,
    ]
    direction = [0.3, -0.1, 0.8, 0.4, -0.5, 0.2]
    dot = sum(a * b for a, b in zip(direction, covector))
    sherman_morrison = 1.0 - 0.9 * dot * dot / (
        sum(value * value for value in direction) * sum(value * value for value in covector)
    )
    normal = constraint_operator([patch], {"p0": "normal"}, ell)
    sticking = constraint_operator([patch], {"p0": "sticking"}, ell)
    off_center_oracle_error = abs(directional_mobility(direction, normal) - sherman_morrison)

    rows = normalized_constraint_rows(patch, "sticking", ell)
    reference_projector = row_space_projector(rows, 3)
    basis_errors = []
    changed_rows = matmul(
        [[2.0, -1.0, 0.5], [0.0, 1.5, -0.25], [0.0, 0.0, 0.75]], rows
    )
    basis_errors.append(max_abs_difference(reference_projector, row_space_projector(changed_rows, 3)))
    for exponent in (-100, -8, 8, 100):
        scaled_rows = [[(10.0**exponent) * value for value in row] for row in rows]
        basis_errors.append(max_abs_difference(reference_projector, row_space_projector(scaled_rows, 3)))
    basis_error = max(basis_errors)
    minimum_order_eigenvalue = min(
        symmetric_eigenvalues(add(sticking, scale(normal, -1.0)))
    )
    mode_margin = directional_mobility(direction, normal) - directional_mobility(direction, sticking)

    second = PatchConstraint("p1", "left_thumb", "surface_1", (-0.1, 0.3, 0.2), (0.0, 1.0, 1.0))
    patches = (patch, second)
    finite_outer = compile_exact_outer_certificate(patches, ell)
    finite_mode_violations = 0
    for assignment in itertools.product(("normal", "sticking"), repeat=2):
        child = compile_mode_certificate(patches, dict(zip(("p0", "p1"), assignment)), ell)
        for name, interval in child.directions.items():
            parent = finite_outer.directions[name]
            finite_mode_violations += int(
                interval.lower < parent.lower - 1e-10
                or interval.upper > parent.upper + 1e-10
            )

    equivariance_errors = []
    for seed in (0, 1, 2):
        rotation = _rotation(seed)
        rotated = constraint_operator([rotate_patch(patch, rotation)], {"p0": "normal"}, ell)
        equivariance_errors.append(max_abs_difference(rotated, rotate_operator(normal, rotation)))
        rotated_direction = matvec(twist_rotation(rotation), direction)
        equivariance_errors.append(
            abs(
                directional_mobility(direction, normal)
                - directional_mobility(rotated_direction, rotated)
            )
        )

    rotation = _rotation(7)
    translation = (0.6, -0.2, 0.3)
    transformed_patch = transform_patch(patch, rotation, translation)
    q = twist_adjoint(rotation, translation)
    q_inverse = inverse(q)
    source_raw = point_velocity_rows(patch.point_m)
    target_raw = point_velocity_rows(transformed_patch.point_m)
    expected_target_raw = matmul(matmul(rotation, source_raw), q_inverse)
    source_whitener = diagonal([1.0, 1.0, 1.0, ell, ell, ell])
    target_whitener = matmul(source_whitener, q_inverse)
    source_normalized = normalized_rows_with_whitener(source_raw, source_whitener)
    target_normalized = normalized_rows_with_whitener(target_raw, target_whitener)
    se3_raw_error = max_abs_difference(target_raw, expected_target_raw)
    se3_projector_error = max_abs_difference(
        row_space_projector(source_normalized, 3),
        row_space_projector(target_normalized, 3),
    )

    failure = unresolved_continuous_uncertainty("M1 interval solver absent")
    hull = finite_hypothesis_hull([certificate, failure], parent_ids=("finite", "continuous"))
    nonfinite_failures = 0
    for bad_ell in (float("nan"), float("inf"), float("-inf")):
        candidate = compile_exact_outer_certificate([patch], bad_ell)
        nonfinite_failures += int(
            candidate.status == "failure_safe"
            and all(
                (item.lower, item.upper, item.label) == (0.0, 1.0, "undetermined")
                for item in candidate.directions.values()
            )
        )
    for bad_direction in (float("nan"), float("inf"), float("-inf")):
        candidate = compile_exact_outer_certificate(
            [patch], ell, {"bad": (bad_direction, 0.0, 0.0, 0.0, 0.0, 0.0)}
        )
        nonfinite_failures += int(
            candidate.status == "failure_safe"
            and (candidate.directions["bad"].lower, candidate.directions["bad"].upper)
            == (0.0, 1.0)
        )
    empty_direction_fallbacks = sum(
        certificate.status == "failure_safe"
        and len(certificate.directions) == 6
        and all(
            (item.lower, item.upper, item.label) == (0.0, 1.0, "undetermined")
            for item in certificate.directions.values()
        )
        for certificate in (
            compile_exact_outer_certificate([patch], ell, {}),
            compile_mode_certificate([patch], {"p0": "normal"}, ell, {}),
        )
    )

    runs = {
        "QH-M0-001": {
            "event_roundtrip_identical": event.canonical_json() == event_roundtrip.canonical_json(),
            "silent_null_count_measured": silent_null_count,
            "strict_raw_type_rejections": strict_type_rejections,
            "strict_raw_type_cases_total": len(invalid_payloads),
            "schema_version": event.schema_version,
        },
        "QH-M0-002": {
            "certificate_status": certificate.status,
            "cross_product_oracle_max_abs_error": cross_product_error,
            "off_center_sherman_morrison_error": off_center_oracle_error,
            "intervals": certificate.to_dict()["directions"],
        },
        "QH-M0-003": {
            "basis_invariance_max_abs_error": basis_error,
            "minimum_mode_order_eigenvalue": minimum_order_eigenvalue,
            "mode_mobility_margin": mode_margin,
            "finite_mode_outer_violations": finite_mode_violations,
        },
        "QH-M0-004": {
            "so3_frame_equivariance_max_abs_error": max(equivariance_errors),
            "se3_adjoint_raw_row_error": se3_raw_error,
            "se3_transformed_metric_projector_error": se3_projector_error,
        },
        "QH-M0-005": {
            "continuous_status": failure.status,
            "parent_hull_status": hull.status,
            "full_interval_count": sum(
                (value.lower, value.upper) == (0.0, 1.0)
                for value in failure.directions.values()
            ),
            "nonfinite_failures_caught": nonfinite_failures,
            "nonfinite_cases_total": 6,
            "empty_direction_fallbacks": empty_direction_fallbacks,
            "empty_direction_cases_total": 2,
        },
    }
    checks = [
        event.canonical_json() == event_roundtrip.canonical_json(),
        silent_null_count == 0,
        strict_type_rejections == len(invalid_payloads),
        certificate.status == "exact_outer",
        cross_product_error <= 1e-10,
        off_center_oracle_error <= 1e-10,
        basis_error <= 1e-10,
        minimum_order_eigenvalue >= -1e-10,
        mode_margin >= -1e-10,
        finite_mode_violations == 0,
        max(equivariance_errors) <= 1e-10,
        se3_raw_error <= 1e-10,
        se3_projector_error <= 1e-10,
        failure.status == "failure_safe",
        all(
            (value.lower, value.upper, value.label) == (0.0, 1.0, "undetermined")
            for value in failure.directions.values()
        ),
        nonfinite_failures == 6,
        empty_direction_fallbacks == 2,
    ]
    return {
        "gate": "quiethand-qh-e1-m0",
        "claim_ceiling": "finite_mode_schema_and_compiler_engineering_correctness_only",
        "environment": {
            "python": sys.version.split()[0],
            "dependencies": "stdlib_only",
            "source_sha256": _source_hashes(),
        },
        "runs": runs,
        "thresholds": {
            "analytic_tolerance": 1e-10,
            "lambda": 9.0,
            "tau_restricted": 0.2,
            "tau_free": 0.8,
        },
        "summary": {
            "checks_passed": sum(checks),
            "checks_total": len(checks),
            "status": "PASS" if all(checks) else "FAIL",
        },
    }


def _canonical_bytes(report: dict) -> bytes:
    return (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def build_three_process_report() -> dict:
    with tempfile.TemporaryDirectory(prefix="quiethand-m0-replay-") as directory:
        outputs = []
        for index in range(3):
            output = Path(directory) / f"run{index + 1}.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--single-run",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"independent replay {index + 1} failed: {completed.stderr or completed.stdout}"
                )
            outputs.append(output.read_bytes())
    hashes = [hashlib.sha256(payload).hexdigest() for payload in outputs]
    report = json.loads(outputs[0].decode("utf-8"))
    byte_identical = len(set(outputs)) == 1
    report["runs"]["QH-M0-006"] = {
        "process_count": 3,
        "command_template": "{python} scripts/quiethand/run_m0_analytic.py --single-run --output runN.json",
        "payload_sha256": hashes,
        "byte_identical": byte_identical,
    }
    report["summary"]["checks_total"] += 1
    report["summary"]["checks_passed"] += int(byte_identical)
    report["summary"]["status"] = (
        "PASS"
        if report["summary"]["checks_passed"] == report["summary"]["checks_total"]
        else "FAIL"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--single-run", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    report = build_single_process_report() if args.single_run else build_three_process_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical_bytes(report))
    print(json.dumps(report["summary"], sort_keys=True))
    return 0 if report["summary"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
