from __future__ import annotations

from decimal import Decimal, getcontext
import itertools
import random
import unittest

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
getcontext().prec = 100


def _patch(
    point=(0.2, -0.1, 0.3),
    normal=(1.0, 2.0, -0.5),
    patch_id="p0",
    link=None,
    surface=None,
):
    return PatchConstraint(
        patch_id,
        link or f"left_{patch_id}",
        surface or f"surface_{patch_id}",
        point,
        normal,
    )


def _uncertain(
    patch_id="p0",
    center=(0.2, -0.1, 0.3),
    radius=(0.01, 0.01, 0.01),
    normal=(1.0, 2.0, -0.5),
):
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


def _is_full(certificate) -> bool:
    return certificate.status == "failure_safe" and all(
        (item.lower, item.upper, item.label) == (0.0, 1.0, "undetermined")
        for item in certificate.directions.values()
    )


def _contains(parent, child, tolerance=0.0) -> bool:
    return all(
        parent.directions[name].lower <= item.lower + tolerance
        and item.upper <= parent.directions[name].upper + tolerance
        for name, item in child.directions.items()
    )


class DirectedIntervalArithmeticTests(unittest.TestCase):
    def test_operations_contain_high_precision_endpoint_oracles(self):
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
            self.assertLessEqual(Decimal.from_float(observed_add.lower), min(additions))
            self.assertGreaterEqual(Decimal.from_float(observed_add.upper), max(additions))
            self.assertLessEqual(Decimal.from_float(observed_mul.lower), min(products))
            self.assertGreaterEqual(Decimal.from_float(observed_mul.upper), max(products))

        quotient = Interval(-0.7, 0.3) / Interval(0.2, 1.1)
        exact = [
            Decimal.from_float(a) / Decimal.from_float(b)
            for a in (-0.7, 0.3)
            for b in (0.2, 1.1)
        ]
        self.assertLessEqual(Decimal.from_float(quotient.lower), min(exact))
        self.assertGreaterEqual(Decimal.from_float(quotient.upper), max(exact))

    def test_invalid_arithmetic_fails_explicitly(self):
        with self.assertRaises(ZeroDivisionError):
            _ = Interval(1.0, 2.0) / Interval(-1.0, 1.0)
        with self.assertRaises(ValueError):
            Interval(-1.0, 1.0).sqrt()
        with self.assertRaises(ValueError):
            Interval(2.0, 1.0)

    def test_interval_inverse_contains_independent_decimal_corner_oracle(self):
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
        for choices in itertools.product((0, 1), repeat=4):
            a, b, c, d = [
                Decimal.from_float(endpoints[index][choice])
                for index, choice in enumerate(choices)
            ]
            determinant = a * d - b * c
            oracle = ((d / determinant, -b / determinant), (-c / determinant, a / determinant))
            for row in range(2):
                for col in range(2):
                    self.assertLessEqual(
                        Decimal.from_float(enclosure[row][col].lower), oracle[row][col]
                    )
                    self.assertGreaterEqual(
                        Decimal.from_float(enclosure[row][col].upper), oracle[row][col]
                    )

    def test_contraction_norm_and_radius_are_directed_outward(self):
        almost_one = float.fromhex("0x1.fffffffffffffp-1")
        dangerous = [[Interval.point(value) for value in (
            almost_one, 5e-17, 5e-17, 5e-17, 5e-17, 5e-17
        )]]
        with self.assertRaisesRegex(ValueError, "contraction failed"):
            _contraction_radius(0.1, dangerous)

        values = (0.1, 0.2, 0.05, 0.03, 0.02, 0.01)
        error = [[Interval.point(value) for value in values]]
        radius, q = _contraction_radius(0.125, error)
        exact_q = sum(Decimal.from_float(value) for value in values)
        exact_radius = Decimal.from_float(0.125) / (Decimal(1) - exact_q)
        self.assertGreaterEqual(Decimal.from_float(q), exact_q)
        self.assertGreaterEqual(Decimal.from_float(radius), exact_radius)


class VerifiedCompilerTests(unittest.TestCase):
    def test_singleton_mode_and_outer_agree_with_m0(self):
        patches = (
            _patch(),
            _patch((-0.15, 0.25, 0.1), (0.0, 1.0, 1.0), "p1"),
        )
        uncertain = tuple(UncertainPatch.singleton(patch) for patch in patches)
        for assignment in itertools.product(("normal", "sticking"), repeat=2):
            modes = dict(zip(("p0", "p1"), assignment))
            exact = compile_mode_certificate(patches, modes, 0.8)
            verified = compile_verified_mode_certificate(
                uncertain, modes, 0.8, max_boxes=1
            )
            self.assertEqual(verified.status, "verified_interval")
            self.assertTrue(_contains(verified, exact, TOLERANCE))
            self.assertLessEqual(
                max(
                    max(
                        abs(verified.directions[name].lower - exact.directions[name].lower),
                        abs(verified.directions[name].upper - exact.directions[name].upper),
                    )
                    for name in exact.directions
                ),
                TOLERANCE,
            )

        exact_outer = compile_exact_outer_certificate(patches, 0.8)
        verified_outer = compile_verified_outer_certificate(
            uncertain, 0.8, max_boxes=1
        )
        self.assertEqual(verified_outer.status, "finite_hull")
        self.assertTrue(_contains(verified_outer, exact_outer, TOLERANCE))

    def test_non_singleton_encloses_fixed_seed_point_samples(self):
        uncertain = (_uncertain(),)
        for max_boxes in (1, 7, 16, 31, 128):
            verified = compile_verified_outer_certificate(
                uncertain, 0.8, max_boxes=max_boxes
            )
            self.assertEqual(verified.status, "finite_hull")
            for seed in (0, 1, 2):
                rng = random.Random(seed)
                for _ in range(256):
                    point = tuple(
                        rng.uniform(interval.lower, interval.upper)
                        for interval in uncertain[0].point_m
                    )
                    patch = uncertain[0].sample(point)
                    for mode in ("normal", "sticking"):
                        exact = compile_mode_certificate(
                            (patch,), {"p0": mode}, 0.8
                        )
                        self.assertTrue(_contains(verified, exact, TOLERANCE))

    def test_finite_mode_and_symmetry_hull_is_explicit(self):
        first = _uncertain(radius=(0.002, 0.003, 0.002))
        second = _uncertain(
            center=(-0.2, 0.15, 0.25),
            radius=(0.002, 0.002, 0.003),
            normal=(0.0, 1.0, 1.0),
        )
        hypotheses = (
            ((first,), {"p0": "normal"}),
            ((second,), {"p0": "sticking"}),
        )
        hull = compile_verified_finite_hypotheses(
            hypotheses, 0.8, max_boxes=7
        )
        self.assertEqual(hull.status, "finite_hull")
        self.assertLessEqual(hull.parameters["event_box_count"], 7)
        self.assertEqual(hull.parameters["box_allocation"], [4, 3])
        for patches, modes in hypotheses:
            child = compile_verified_mode_certificate(
                patches, modes, 0.8, max_boxes=7
            )
            self.assertTrue(_contains(hull, child, TOLERANCE))
            for seed in (0, 1, 2):
                rng = random.Random(seed)
                for _ in range(256):
                    sampled = tuple(
                        rng.uniform(interval.lower, interval.upper)
                        for interval in patches[0].point_m
                    )
                    exact = compile_mode_certificate(
                        (patches[0].sample(sampled),), modes, 0.8
                    )
                    self.assertTrue(_contains(hull, exact, TOLERANCE))

    def test_parent_hull_prevents_partition_or_nested_set_shrink(self):
        inner = (_uncertain(radius=(0.002, 0.003, 0.002)),)
        outer = (_uncertain(radius=(0.01, 0.012, 0.008)),)
        parent = compile_verified_outer_certificate(inner, 0.8, max_boxes=31)
        child_without_parent = compile_verified_outer_certificate(
            outer, 0.8, max_boxes=7
        )
        child = compile_verified_outer_certificate(
            outer,
            0.8,
            max_boxes=7,
            parent_certificates=(parent,),
            parent_ids=("inner",),
        )
        self.assertNotEqual(parent.status, "failure_safe")
        self.assertNotEqual(child_without_parent.status, "failure_safe")
        self.assertEqual(child.status, "parent_hull")
        self.assertLessEqual(child.parameters["event_box_count"], 7)
        self.assertTrue(_contains(child, parent))
        self.assertTrue(_contains(child, child_without_parent))
        for box in (inner, outer):
            for seed in (0, 1, 2):
                rng = random.Random(seed)
                for _ in range(256):
                    sampled = tuple(
                        rng.uniform(interval.lower, interval.upper)
                        for interval in box[0].point_m
                    )
                    patch = box[0].sample(sampled)
                    for mode in ("normal", "sticking"):
                        exact = compile_mode_certificate(
                            (patch,), {"p0": mode}, 0.8
                        )
                        self.assertTrue(_contains(child, exact, TOLERANCE))

    def test_timeout_unsupported_and_invalid_inputs_are_whole_event_full(self):
        class AdvancingClock:
            def __init__(self):
                self.value = -1.0

            def __call__(self):
                self.value += 1.0
                return self.value

        timeout = compile_verified_outer_certificate(
            (_uncertain(),),
            0.8,
            max_boxes=128,
            timeout_s=0.5,
            clock=AdvancingClock(),
        )
        unsupported = unsupported_continuous_uncertainty(
            "normal intervals are not supported in M1"
        )
        invalid = compile_verified_outer_certificate(
            (_uncertain(),), float("nan")
        )
        singular_direction = compile_verified_outer_certificate(
            (_uncertain(),), 0.8, {"zero": (0, 0, 0, 0, 0, 0)}
        )
        pivot = compile_verified_mode_certificate(
            (
                UncertainPatch(
                    "p0",
                    "left_p0",
                    "surface_p0",
                    (Interval(-1, 1), Interval(-1, 1), Interval(-1, 1)),
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
        for certificate in (
            timeout,
            unsupported,
            invalid,
            singular_direction,
            pivot,
            contraction,
            overflow,
        ):
            self.assertTrue(_is_full(certificate))

    def test_event_wide_box_cap_hard_timeout_and_split_tie(self):
        equal = UncertainPatch(
            "p0",
            "left_p0",
            "surface_p0",
            (Interval(-0.0009765625, 0.0009765625),) * 3,
            (1, 2, 3),
        )
        outer = compile_verified_outer_certificate((equal,), 1.0, max_boxes=7)
        self.assertEqual(outer.status, "finite_hull")
        self.assertEqual(outer.parameters["event_box_count"], 7)
        self.assertEqual(outer.parameters["split_trace"][:3], ["0:p0:0", "0:p0:1", "0:p0:2"])
        for seed in (0, 1, 2):
            rng = random.Random(seed)
            for _ in range(256):
                sampled = tuple(
                    rng.uniform(interval.lower, interval.upper)
                    for interval in equal.point_m
                )
                patch = equal.sample(sampled)
                for mode in ("normal", "sticking"):
                    exact = compile_mode_certificate(
                        (patch,), {"p0": mode}, 1.0
                    )
                    self.assertTrue(_contains(outer, exact, TOLERANCE))

        class FineClock:
            def __init__(self):
                self.value = -0.01

            def __call__(self):
                self.value += 0.01
                return self.value

        many_directions = {
            f"d{index}": (1, 0, 0, 0, 0, 0) for index in range(100)
        }
        late = compile_verified_outer_certificate(
            (equal,),
            1.0,
            many_directions,
            max_boxes=1,
            timeout_s=0.5,
            clock=FineClock(),
        )
        self.assertTrue(_is_full(late))
        late_hypothesis = compile_verified_finite_hypotheses(
            (((equal,), {"p0": "normal"}),),
            1.0,
            many_directions,
            max_boxes=1,
            timeout_s=0.5,
            clock=FineClock(),
        )
        over_budget = compile_verified_finite_hypotheses(
            tuple((((equal,), {"p0": "normal"})) for _ in range(8)),
            1.0,
            max_boxes=7,
        )
        self.assertTrue(_is_full(late_hypothesis))
        self.assertTrue(_is_full(over_budget))

    def test_replay_is_byte_identical(self):
        patch = (_uncertain(radius=(0.004, 0.003, 0.002)),)
        payloads = [
            compile_verified_outer_certificate(
                patch, 0.8, max_boxes=16
            ).canonical_json()
            for _ in range(3)
        ]
        self.assertEqual(len(set(payloads)), 1)


if __name__ == "__main__":
    unittest.main()
