#!/usr/bin/env python3
"""Run the four preregistered QuietHand M1 verified-interval checks."""

from __future__ import annotations

import argparse
from decimal import Decimal, getcontext
import hashlib
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quiethand.compiler import compile_exact_outer_certificate, compile_mode_certificate
from quiethand.models import PatchConstraint
from quiethand.verified_interval import (
    Interval,
    UncertainPatch,
    _contraction_radius,
    _interval_inverse,
    compile_verified_finite_hypotheses,
    compile_verified_mode_certificate,
    compile_verified_outer_certificate,
    unsupported_continuous_uncertainty,
)


TOLERANCE = 1e-10
MAX_BOXES_MATRIX = (1, 7, 16, 31, 128)
SEEDS = (0, 1, 2)
SAMPLES_PER_CASE = 256
SOURCE_PATHS = (
    "quiethand/__init__.py",
    "quiethand/models.py",
    "quiethand/linalg.py",
    "quiethand/geometry.py",
    "quiethand/compiler.py",
    "quiethand/verified_interval.py",
    "scripts/quiethand/run_m1_verified.py",
    "tests/test_quiethand_m1.py",
    "refine-logs/EXPERIMENT_PLAN_20260819_115901.md",
)


def _source_hashes() -> dict[str, str]:
    return {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in SOURCE_PATHS
    }


def _patch(
    patch_id: str,
    point: tuple[float, float, float],
    normal: tuple[float, float, float],
) -> PatchConstraint:
    return PatchConstraint(
        patch_id,
        f"left_{patch_id}",
        f"surface_{patch_id}",
        point,
        normal,
    )


def _uncertain(
    patch_id: str,
    center: tuple[float, float, float],
    radius: tuple[float, float, float],
    normal: tuple[float, float, float],
) -> UncertainPatch:
    return UncertainPatch(
        patch_id,
        f"left_{patch_id}",
        f"surface_{patch_id}",
        tuple(
            Interval(value - width, value + width)
            for value, width in zip(center, radius)
        ),
        normal,
    )


def _contains(parent, child, tolerance: float = 0.0) -> bool:
    return all(
        parent.directions[name].lower <= item.lower + tolerance
        and item.upper <= parent.directions[name].upper + tolerance
        for name, item in child.directions.items()
    )


def _full(certificate) -> bool:
    return certificate.status == "failure_safe" and all(
        (item.lower, item.upper, item.label) == (0.0, 1.0, "undetermined")
        for item in certificate.directions.values()
    )


class _AdvancingClock:
    def __init__(self) -> None:
        self.value = -1.0

    def __call__(self) -> float:
        self.value += 1.0
        return self.value


class _FineClock:
    def __init__(self) -> None:
        self.value = -0.01

    def __call__(self) -> float:
        self.value += 0.01
        return self.value


def _arithmetic_oracle_violations() -> int:
    getcontext().prec = 100
    violations = 0
    cases = (
        (Interval(-0.3, 0.7), Interval(0.2, 1.1)),
        (Interval(-3.0, -0.25), Interval(-2.0, 0.4)),
        (Interval(1e-100, 2e-100), Interval(3e99, 4e99)),
    )
    for left, right in cases:
        additions = [
            Decimal.from_float(a) + Decimal.from_float(b)
            for a in (left.lower, left.upper)
            for b in (right.lower, right.upper)
        ]
        products = [
            Decimal.from_float(a) * Decimal.from_float(b)
            for a in (left.lower, left.upper)
            for b in (right.lower, right.upper)
        ]
        observed_add = left + right
        observed_mul = left * right
        violations += int(Decimal.from_float(observed_add.lower) > min(additions))
        violations += int(Decimal.from_float(observed_add.upper) < max(additions))
        violations += int(Decimal.from_float(observed_mul.lower) > min(products))
        violations += int(Decimal.from_float(observed_mul.upper) < max(products))
    quotient = Interval(-0.7, 0.3) / Interval(0.2, 1.1)
    exact_quotients = [
        Decimal.from_float(a) / Decimal.from_float(b)
        for a in (-0.7, 0.3)
        for b in (0.2, 1.1)
    ]
    violations += int(Decimal.from_float(quotient.lower) > min(exact_quotients))
    violations += int(Decimal.from_float(quotient.upper) < max(exact_quotients))
    square_root = Interval(0.2, 1.1).sqrt()
    violations += int(
        Decimal.from_float(square_root.lower) > Decimal.from_float(0.2).sqrt()
    )
    violations += int(
        Decimal.from_float(square_root.upper) < Decimal.from_float(1.1).sqrt()
    )
    return violations


def _inverse_oracle_violations() -> int:
    matrix = [
        [Interval(1.9, 2.1), Interval(0.09, 0.11)],
        [Interval(0.19, 0.21), Interval(1.4, 1.6)],
    ]
    enclosure = _interval_inverse(matrix)
    endpoints = [
        (matrix[row][col].lower, matrix[row][col].upper)
        for row in range(2)
        for col in range(2)
    ]
    violations = 0
    for choices in itertools.product((0, 1), repeat=4):
        a, b, c, d = [
            Decimal.from_float(endpoints[index][choice])
            for index, choice in enumerate(choices)
        ]
        determinant = a * d - b * c
        oracle = ((d / determinant, -b / determinant), (-c / determinant, a / determinant))
        for row in range(2):
            for col in range(2):
                violations += int(
                    Decimal.from_float(enclosure[row][col].lower) > oracle[row][col]
                    or Decimal.from_float(enclosure[row][col].upper) < oracle[row][col]
                )
    return violations


def _contraction_oracle_metrics() -> tuple[int, int]:
    almost_one = float.fromhex("0x1.fffffffffffffp-1")
    dangerous = [[Interval.point(value) for value in (
        almost_one, 5e-17, 5e-17, 5e-17, 5e-17, 5e-17
    )]]
    unsafe_accepts = 0
    try:
        _contraction_radius(0.1, dangerous)
        unsafe_accepts += 1
    except ValueError:
        pass

    values = (0.1, 0.2, 0.05, 0.03, 0.02, 0.01)
    radius, q = _contraction_radius(
        0.125, [[Interval.point(value) for value in values]]
    )
    exact_q = sum(Decimal.from_float(value) for value in values)
    exact_radius = Decimal.from_float(0.125) / (Decimal(1) - exact_q)
    radius_underbounds = int(
        Decimal.from_float(q) < exact_q
        or Decimal.from_float(radius) < exact_radius
    )
    return unsafe_accepts, radius_underbounds


def build_single_process_report() -> dict:
    """Rebuild every M1 case once; the parent invokes this in three processes."""

    arithmetic_violations = _arithmetic_oracle_violations()
    inverse_oracle_violations = _inverse_oracle_violations()
    contraction_unsafe_accepts, contraction_radius_underbounds = (
        _contraction_oracle_metrics()
    )

    exact_patches = (
        _patch("p0", (0.2, -0.1, 0.3), (1.0, 2.0, -0.5)),
        _patch("p1", (-0.15, 0.25, 0.1), (0.0, 1.0, 1.0)),
    )
    singleton_patches = tuple(UncertainPatch.singleton(patch) for patch in exact_patches)
    singleton_max_error = 0.0
    singleton_containment_violations = 0
    finite_mode_outer_violations = 0
    for assignment in itertools.product(("normal", "sticking"), repeat=2):
        modes = dict(zip(("p0", "p1"), assignment))
        exact = compile_mode_certificate(exact_patches, modes, 0.8)
        verified = compile_verified_mode_certificate(
            singleton_patches, modes, 0.8, max_boxes=1
        )
        singleton_containment_violations += int(not _contains(verified, exact, TOLERANCE))
        for name, item in exact.directions.items():
            singleton_max_error = max(
                singleton_max_error,
                abs(verified.directions[name].lower - item.lower),
                abs(verified.directions[name].upper - item.upper),
            )
    exact_outer = compile_exact_outer_certificate(exact_patches, 0.8)
    verified_singleton_outer = compile_verified_outer_certificate(
        singleton_patches, 0.8, max_boxes=1
    )
    singleton_containment_violations += int(
        not _contains(verified_singleton_outer, exact_outer, TOLERANCE)
    )
    for assignment in itertools.product(("normal", "sticking"), repeat=2):
        exact = compile_mode_certificate(
            exact_patches, dict(zip(("p0", "p1"), assignment)), 0.8
        )
        finite_mode_outer_violations += int(
            not _contains(verified_singleton_outer, exact, TOLERANCE)
        )

    uncertain = (
        _uncertain(
            "p0", (0.2, -0.1, 0.3), (0.01, 0.01, 0.01), (1.0, 2.0, -0.5)
        ),
    )
    sampled_outer_violations = 0
    sampled_points = 0
    coverage_widths: dict[str, float] = {}
    statuses: dict[str, str] = {}
    event_box_counts: dict[str, int] = {}
    for max_boxes in MAX_BOXES_MATRIX:
        verified = compile_verified_outer_certificate(
            uncertain, 0.8, max_boxes=max_boxes
        )
        statuses[str(max_boxes)] = verified.status
        event_box_counts[str(max_boxes)] = int(
            verified.parameters.get("event_box_count", max_boxes + 1)
        )
        coverage_widths[str(max_boxes)] = max(
            item.upper - item.lower for item in verified.directions.values()
        )
        for seed in SEEDS:
            rng = random.Random(seed)
            for _ in range(SAMPLES_PER_CASE):
                point = tuple(
                    rng.uniform(interval.lower, interval.upper)
                    for interval in uncertain[0].point_m
                )
                patch = uncertain[0].sample(point)
                sampled_points += 1
                for mode in ("normal", "sticking"):
                    exact = compile_mode_certificate(
                        (patch,), {"p0": mode}, 0.8
                    )
                    sampled_outer_violations += int(
                        not _contains(verified, exact, TOLERANCE)
                    )

    first = _uncertain(
        "p0", (0.2, -0.1, 0.3), (0.002, 0.003, 0.002), (1.0, 2.0, -0.5)
    )
    second = _uncertain(
        "p0", (-0.2, 0.15, 0.25), (0.002, 0.002, 0.003), (0.0, 1.0, 1.0)
    )
    hypotheses = (
        ((first,), {"p0": "normal"}),
        ((second,), {"p0": "sticking"}),
    )
    hypothesis_hull = compile_verified_finite_hypotheses(
        hypotheses, 0.8, max_boxes=7
    )
    finite_hypothesis_violations = 0
    finite_hypothesis_sampled_points = 0
    for patches, modes in hypotheses:
        child = compile_verified_mode_certificate(
            patches, modes, 0.8, max_boxes=7
        )
        finite_hypothesis_violations += int(
            not _contains(hypothesis_hull, child, TOLERANCE)
        )
        for seed in SEEDS:
            rng = random.Random(seed)
            for _ in range(SAMPLES_PER_CASE):
                point = tuple(
                    rng.uniform(interval.lower, interval.upper)
                    for interval in patches[0].point_m
                )
                exact = compile_mode_certificate(
                    (patches[0].sample(point),), modes, 0.8
                )
                finite_hypothesis_sampled_points += 1
                finite_hypothesis_violations += int(
                    not _contains(hypothesis_hull, exact, TOLERANCE)
                )

    inner = (
        _uncertain(
            "p0", (0.2, -0.1, 0.3), (0.002, 0.003, 0.002), (1.0, 2.0, -0.5)
        ),
    )
    outer = (
        _uncertain(
            "p0", (0.2, -0.1, 0.3), (0.01, 0.012, 0.008), (1.0, 2.0, -0.5)
        ),
    )
    parent = compile_verified_outer_certificate(inner, 0.8, max_boxes=31)
    without_parent = compile_verified_outer_certificate(outer, 0.8, max_boxes=7)
    with_parent = compile_verified_outer_certificate(
        outer,
        0.8,
        max_boxes=7,
        parent_certificates=(parent,),
        parent_ids=("inner",),
    )
    parent_shrink_violations = int(not _contains(with_parent, parent)) + int(
        not _contains(with_parent, without_parent)
    )
    nested_sampled_outer_violations = 0
    nested_sampled_points = 0
    for box in (inner, outer):
        for seed in SEEDS:
            rng = random.Random(seed)
            for _ in range(SAMPLES_PER_CASE):
                point = tuple(
                    rng.uniform(interval.lower, interval.upper)
                    for interval in box[0].point_m
                )
                patch = box[0].sample(point)
                nested_sampled_points += 1
                for mode in ("normal", "sticking"):
                    exact = compile_mode_certificate((patch,), {"p0": mode}, 0.8)
                    nested_sampled_outer_violations += int(
                        not _contains(with_parent, exact, TOLERANCE)
                    )

    equal = UncertainPatch(
        "p0",
        "left_p0",
        "surface_p0",
        (Interval(-0.0009765625, 0.0009765625),) * 3,
        (1, 2, 3),
    )
    tie = compile_verified_outer_certificate((equal,), 1.0, max_boxes=7)
    tie_trace = tie.parameters.get("split_trace", [])
    deterministic_tie_violations = int(
        tie.status != "finite_hull"
        or tie_trace[:3] != ["0:p0:0", "0:p0:1", "0:p0:2"]
    )
    tie_sampled_outer_violations = 0
    tie_sampled_points = 0
    for seed in SEEDS:
        rng = random.Random(seed)
        for _ in range(SAMPLES_PER_CASE):
            point = tuple(
                rng.uniform(interval.lower, interval.upper)
                for interval in equal.point_m
            )
            patch = equal.sample(point)
            tie_sampled_points += 1
            for mode in ("normal", "sticking"):
                exact = compile_mode_certificate((patch,), {"p0": mode}, 1.0)
                tie_sampled_outer_violations += int(
                    not _contains(tie, exact, TOLERANCE)
                )
    event_box_budget_violations = sum(
        event_box_counts[str(limit)] > limit for limit in MAX_BOXES_MATRIX
    )
    event_box_budget_violations += int(
        hypothesis_hull.parameters.get("event_box_count", 8) > 7
        or tie.parameters.get("event_box_count", 8) > 7
        or with_parent.parameters.get("event_box_count", 8) > 7
    )

    timeout = compile_verified_outer_certificate(
        uncertain,
        0.8,
        max_boxes=128,
        timeout_s=0.5,
        clock=_AdvancingClock(),
    )
    late_timeout = compile_verified_outer_certificate(
        (equal,),
        1.0,
        {f"d{index}": (1, 0, 0, 0, 0, 0) for index in range(100)},
        max_boxes=1,
        timeout_s=0.5,
        clock=_FineClock(),
    )
    late_hypothesis_timeout = compile_verified_finite_hypotheses(
        (((equal,), {"p0": "normal"}),),
        1.0,
        {f"d{index}": (1, 0, 0, 0, 0, 0) for index in range(100)},
        max_boxes=1,
        timeout_s=0.5,
        clock=_FineClock(),
    )
    hypothesis_budget_failure = compile_verified_finite_hypotheses(
        tuple((((equal,), {"p0": "normal"})) for _ in range(8)),
        1.0,
        max_boxes=7,
    )
    pivot = compile_verified_mode_certificate(
        (
            UncertainPatch(
                "p0",
                "left_p0",
                "surface_p0",
                (Interval(-1, 1),) * 3,
                (1, 2, 3),
            ),
        ),
        {"p0": "sticking"},
        1.0,
        max_boxes=1,
    )
    contraction = compile_verified_outer_certificate(
        (
            UncertainPatch(
                "p0",
                "left_p0",
                "surface_p0",
                (Interval(-0.05, 0.05),) * 3,
                (1, 2, 3),
            ),
        ),
        1.0,
        max_boxes=1,
    )
    overflow = compile_verified_outer_certificate(
        (
            UncertainPatch(
                "p0",
                "left_p0",
                "surface_p0",
                (Interval.point(1e308), Interval.point(0), Interval.point(0)),
                (1, 2, 3),
            ),
        ),
        0.8,
        max_boxes=1,
    )
    failures = (
        timeout,
        late_timeout,
        late_hypothesis_timeout,
        hypothesis_budget_failure,
        unsupported_continuous_uncertainty("normal interval"),
        compile_verified_outer_certificate(uncertain, float("nan")),
        compile_verified_outer_certificate(uncertain, 0.8, max_boxes=0),
        compile_verified_outer_certificate(
            uncertain, 0.8, {"zero": (0, 0, 0, 0, 0, 0)}
        ),
        pivot,
        contraction,
        overflow,
    )
    unsafe_failure_count = sum(not _full(certificate) for certificate in failures)

    runs = {
        "QH-M1-001": {
            "directed_arithmetic_oracle_violations": arithmetic_violations,
            "interval_inverse_decimal_corner_violations": inverse_oracle_violations,
            "contraction_unsafe_accepts": contraction_unsafe_accepts,
            "contraction_radius_underbounds": contraction_radius_underbounds,
            "singleton_max_abs_error": singleton_max_error,
            "singleton_containment_violations": singleton_containment_violations,
            "singleton_assignment_count": 4,
        },
        "QH-M1-002": {
            "max_boxes_matrix": list(MAX_BOXES_MATRIX),
            "seeds": list(SEEDS),
            "samples_per_seed_per_case": SAMPLES_PER_CASE,
            "sampled_points_total": sampled_points,
            "sampled_outer_violations": sampled_outer_violations,
            "finite_mode_outer_violations": finite_mode_outer_violations,
            "certificate_status_by_max_boxes": statuses,
            "event_box_count_by_max_boxes": event_box_counts,
            "maximum_direction_width_by_max_boxes": coverage_widths,
        },
        "QH-M1-003": {
            "finite_hypothesis_outer_violations": finite_hypothesis_violations,
            "finite_hypothesis_sampled_points": finite_hypothesis_sampled_points,
            "parent_or_nested_shrink_violations": parent_shrink_violations,
            "nested_sampled_outer_violations": nested_sampled_outer_violations,
            "nested_sampled_points": nested_sampled_points,
            "event_box_budget_violations": event_box_budget_violations,
            "deterministic_split_tie_violations": deterministic_tie_violations,
            "tie_case_sampled_outer_violations": tie_sampled_outer_violations,
            "tie_case_sampled_points": tie_sampled_points,
            "unsafe_failure_or_timeout_count": unsafe_failure_count,
            "failure_cases_total": len(failures),
            "timeout_status": timeout.status,
            "late_timeout_status": late_timeout.status,
            "late_hypothesis_timeout_status": late_hypothesis_timeout.status,
            "finite_hypothesis_status": hypothesis_hull.status,
            "parent_hull_status": with_parent.status,
        },
    }
    checks = (
        arithmetic_violations == 0,
        inverse_oracle_violations == 0,
        contraction_unsafe_accepts == 0,
        contraction_radius_underbounds == 0,
        singleton_max_error <= TOLERANCE,
        singleton_containment_violations == 0,
        sampled_outer_violations == 0,
        finite_mode_outer_violations == 0,
        finite_hypothesis_violations == 0,
        parent_shrink_violations == 0,
        nested_sampled_outer_violations == 0,
        event_box_budget_violations == 0,
        deterministic_tie_violations == 0,
        tie_sampled_outer_violations == 0,
        unsafe_failure_count == 0,
        all(status != "failure_safe" for status in statuses.values()),
        hypothesis_hull.status == "finite_hull",
        with_parent.status == "parent_hull",
    )
    return {
        "gate": "quiethand-qh-e1-m1",
        "claim_ceiling": "verified_interval_engineering_correctness_on_frozen_synthetic_contract_only",
        "environment": {
            "python": sys.version.split()[0],
            "dependencies": "stdlib_only",
            "source_sha256": _source_hashes(),
        },
        "runs": runs,
        "thresholds": {
            "analytic_tolerance": TOLERANCE,
            "lambda": 9.0,
            "max_boxes_hard_cap": 128,
            "timeout_seconds_hard_cap": 60.0,
            "samples_are_diagnostics_not_certificates": True,
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
    with tempfile.TemporaryDirectory(prefix="quiethand-m1-replay-") as directory:
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
    report["runs"]["QH-M1-004"] = {
        "process_count": 3,
        "command_template": "{python} scripts/quiethand/run_m1_verified.py --single-run --output runN.json",
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
