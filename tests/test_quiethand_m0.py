from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import unittest

from quiethand.compiler import (
    compile_exact_outer_certificate,
    compile_mode_certificate,
    constraint_operator,
    directional_mobility,
)
from quiethand.geometry import (
    normalized_rows_with_whitener,
    normalized_constraint_rows,
    point_velocity_rows,
    rotate_operator,
    rotate_patch,
    transform_patch,
    twist_adjoint,
    twist_rotation,
)
from quiethand.linalg import (
    add,
    diagonal,
    inverse,
    matmul,
    matvec,
    max_abs_difference,
    row_space_projector,
    scale,
    symmetric_eigenvalues,
)
from quiethand.models import (
    EvidenceEvent,
    EvidenceRecord,
    MobilityCertificate,
    MobilityInterval,
    PatchConstraint,
    Provenance,
)
from quiethand.outer import finite_hypothesis_hull, unresolved_continuous_uncertainty


def _provenance(kind: str = "oracle") -> Provenance:
    return Provenance(kind, "synthetic-m0", "1", "QH-M0-001")


def _event() -> EvidenceEvent:
    records = (
        EvidenceRecord(
            key="hand_roles",
            status="human",
            value={"active": "right", "support": "left"},
            unit="semantic",
            frame_id="ego",
            timestamp_s=1.25,
            provenance=_provenance("human"),
        ),
        EvidenceRecord(
            key="support_patch",
            status="derived",
            value={"point_m": [0.0, 0.0, 0.0], "normal": [1.0, 0.0, 0.0]},
            unit="m",
            frame_id="object",
            timestamp_s=1.25,
            provenance=_provenance("derived"),
            depends_on=("hand_roles",),
        ),
        EvidenceRecord(
            key="force_proxy",
            status="unobservable",
            value=None,
            unit="N",
            frame_id="object",
            timestamp_s=1.25,
            provenance=_provenance("model"),
            failure_reason="no force sensor in ego RGB",
        ),
    )
    return EvidenceEvent(
        event_id="synthetic:seq-1:1.0-1.5",
        dataset="synthetic",
        sequence_id="seq-1",
        start_s=1.0,
        end_s=1.5,
        active_hand="right",
        support_hand="left",
        supported_object_id="object-1",
        support_frame_id="object",
        evidence=records,
    )


def _patch(point=(0.0, 0.0, 0.0), normal=(1.0, 0.0, 0.0), patch_id="p0"):
    return PatchConstraint(patch_id, f"left_index_{patch_id}", f"surface_{patch_id}", point, normal)


def _rotation_from_seed(seed: int):
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


class EvidenceEventTests(unittest.TestCase):
    def test_round_trip_is_byte_stable_and_schema_is_parseable(self):
        event = _event()
        encoded = event.canonical_json()
        self.assertEqual(encoded, EvidenceEvent.from_json(encoded).canonical_json())
        schema_path = Path("quiethand/schema/evidence_event.schema.json")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["properties"]["schema_version"]["const"], event.schema_version)
        self.assertEqual(set(schema["required"]), set(event.to_dict()))

    def test_silent_null_and_dependency_cycles_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "null value"):
            EvidenceRecord("bad", "observed", None, "", "ego", 1.0, _provenance())
        with self.assertRaisesRegex(ValueError, "failure_reason"):
            EvidenceRecord("bad", "missing", None, "", "ego", 1.0, _provenance())
        a = EvidenceRecord("a", "derived", 1, "", "ego", 1.0, _provenance("derived"), ("b",))
        b = EvidenceRecord("b", "derived", 2, "", "ego", 1.0, _provenance("derived"), ("a",))
        with self.assertRaisesRegex(ValueError, "cycle"):
            EvidenceEvent("e", "d", "s", 0.5, 1.5, "left", "right", "o", "f", (a, b))

    def test_raw_json_entry_is_strict_and_rejects_nested_null_or_nonfinite(self):
        payload = _event().to_dict()
        del payload["evidence"][0]["unit"]
        with self.assertRaisesRegex(ValueError, "exactly match"):
            EvidenceEvent.from_dict(payload)

        payload = _event().to_dict()
        payload["schema_version"] = "qh.evidence_event.v0"
        with self.assertRaisesRegex(ValueError, "schema_version"):
            EvidenceEvent.from_dict(payload)

        payload = _event().to_dict()
        available = next(item for item in payload["evidence"] if item["key"] == "hand_roles")
        available["value"]["active"] = None
        with self.assertRaisesRegex(ValueError, "silent null"):
            EvidenceEvent.from_dict(payload)

        encoded = _event().canonical_json().replace('"right"', "NaN", 1)
        with self.assertRaisesRegex(ValueError, "non-standard JSON"):
            EvidenceEvent.from_json(encoded)

        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            EvidenceEvent.from_json('{"schema_version":"qh.evidence_event.v1","schema_version":"qh.evidence_event.v1"}')

    def test_raw_json_types_are_never_silently_coerced(self):
        mutations = []
        payload = _event().to_dict()
        payload["event_id"] = 123
        mutations.append(payload)
        payload = _event().to_dict()
        payload["evidence"][0]["provenance"]["source_name"] = 123
        mutations.append(payload)
        payload = _event().to_dict()
        payload["evidence"][0]["timestamp_s"] = True
        mutations.append(payload)
        payload = _event().to_dict()
        payload["evidence"][0]["depends_on"] = "hand_roles"
        mutations.append(payload)
        payload = _event().to_dict()
        payload["evidence"] = {"not": "an array"}
        mutations.append(payload)
        for payload in mutations:
            with self.assertRaises(ValueError):
                EvidenceEvent.from_dict(copy.deepcopy(payload))


class CompilerAnalyticTests(unittest.TestCase):
    def test_plane_origin_has_closed_form_mobility(self):
        certificate = compile_exact_outer_certificate([_patch()], ell_m=1.0)
        self.assertEqual(certificate.status, "exact_outer")
        self.assertAlmostEqual(certificate.directions["v_x"].lower, 0.1, places=12)
        self.assertAlmostEqual(certificate.directions["v_x"].upper, 0.1, places=12)
        self.assertEqual(certificate.directions["v_x"].label, "restricted")
        self.assertAlmostEqual(certificate.directions["v_y"].lower, 0.1, places=12)
        self.assertAlmostEqual(certificate.directions["v_y"].upper, 1.0, places=12)
        self.assertEqual(certificate.directions["v_y"].label, "undetermined")
        self.assertEqual(certificate.directions["omega_x"].label, "free")

    def test_row_basis_invariance(self):
        rows = normalized_constraint_rows(_patch((0.2, -0.1, 0.4)), "sticking", 0.7)
        basis_change = [[2.0, -1.0, 0.5], [0.0, 1.5, -0.25], [0.0, 0.0, 0.75]]
        changed_rows = matmul(basis_change, rows)
        original = row_space_projector(rows, 3)
        changed = row_space_projector(changed_rows, 3)
        self.assertLessEqual(max_abs_difference(original, changed), 1e-12)
        for exponent in (-100, -8, 8, 100):
            scaled = [[(10.0**exponent) * value for value in row] for row in rows]
            self.assertLessEqual(
                max_abs_difference(original, row_space_projector(scaled, 3)), 1e-12
            )

    def test_off_center_cross_product_and_sherman_morrison_oracle(self):
        point = (0.2, -0.3, 0.4)
        velocity = [0.5, -0.7, 0.2]
        omega = [1.1, -0.4, 0.9]
        observed_velocity = matvec(point_velocity_rows(point), velocity + omega)
        expected_cross = [
            omega[1] * point[2] - omega[2] * point[1],
            omega[2] * point[0] - omega[0] * point[2],
            omega[0] * point[1] - omega[1] * point[0],
        ]
        self.assertLessEqual(
            max(abs(observed_velocity[i] - velocity[i] - expected_cross[i]) for i in range(3)),
            1e-15,
        )

        ell = 0.7
        patch = _patch(point, (1.0, 2.0, -1.0))
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
        expected = 1.0 - 0.9 * dot * dot / (
            sum(value * value for value in direction)
            * sum(value * value for value in covector)
        )
        normal_operator = constraint_operator([patch], {"p0": "normal"}, ell)
        self.assertAlmostEqual(directional_mobility(direction, normal_operator), expected, places=12)

    def test_mode_order_is_psd_and_mobility_monotone(self):
        patches = [_patch((0.2, 0.1, -0.3), (1.0, 2.0, 0.5), "p0"), _patch((-0.1, 0.3, 0.2), (0.0, 1.0, 1.0), "p1")]
        normal = constraint_operator(patches, {"p0": "normal", "p1": "normal"}, 0.8)
        sticking = constraint_operator(patches, {"p0": "sticking", "p1": "sticking"}, 0.8)
        minimum_eigenvalue = min(symmetric_eigenvalues(add(sticking, scale(normal, -1.0))))
        self.assertGreaterEqual(minimum_eigenvalue, -1e-10)
        for direction in ([1, 0, 0, 0, 0, 0], [0, 1, 2, -1, 0.5, 3]):
            self.assertLessEqual(
                directional_mobility(direction, sticking),
                directional_mobility(direction, normal) + 1e-10,
            )

    def test_frame_equivariance_for_seeded_rotations(self):
        patch = _patch((0.2, -0.4, 0.1), (1.0, 2.0, -0.5))
        base = constraint_operator([patch], {"p0": "normal"}, 0.9)
        direction = [0.3, -0.1, 0.8, 0.4, -0.5, 0.2]
        for seed in (0, 1, 2):
            rotation = _rotation_from_seed(seed)
            rotated_patch = rotate_patch(patch, rotation)
            observed = constraint_operator([rotated_patch], {"p0": "normal"}, 0.9)
            expected = rotate_operator(base, rotation)
            self.assertLessEqual(max_abs_difference(observed, expected), 1e-10)
            rotated_direction = matvec(twist_rotation(rotation), direction)
            self.assertAlmostEqual(
                directional_mobility(direction, base),
                directional_mobility(rotated_direction, observed),
                places=10,
            )

    def test_se3_adjoint_with_transformed_metric_whitener(self):
        patch = _patch((0.2, -0.4, 0.1), (1.0, 2.0, -0.5))
        rotation = _rotation_from_seed(7)
        translation = (0.6, -0.2, 0.3)
        transformed = transform_patch(patch, rotation, translation)
        q = twist_adjoint(rotation, translation)
        q_inverse = inverse(q)
        source_raw = point_velocity_rows(patch.point_m)
        target_raw = point_velocity_rows(transformed.point_m)
        independent_target = matmul(matmul(rotation, source_raw), q_inverse)
        self.assertLessEqual(max_abs_difference(target_raw, independent_target), 1e-10)

        ell = 0.9
        source_whitener = diagonal([1.0, 1.0, 1.0, ell, ell, ell])
        target_whitener = matmul(source_whitener, q_inverse)
        source_normalized = normalized_rows_with_whitener(source_raw, source_whitener)
        target_normalized = normalized_rows_with_whitener(target_raw, target_whitener)
        source_projector = row_space_projector(source_normalized, 3)
        target_projector = row_space_projector(target_normalized, 3)
        self.assertLessEqual(max_abs_difference(source_projector, target_projector), 1e-10)

    def test_rotation_validation_and_mode_key_validation_fail_closed(self):
        patch = _patch()
        with self.assertRaisesRegex(ValueError, "SO\(3\)"):
            rotate_patch(patch, [[2, 0, 0], [0, 1, 0], [0, 0, 1]])
        certificate = compile_mode_certificate(
            [patch], {"p0": "normal", "stale": "normal"}, 1.0
        )
        self.assertEqual(certificate.status, "failure_safe")

    def test_invalid_patch_set_fails_closed(self):
        first = _patch(patch_id="a")
        duplicate_component = PatchConstraint("b", first.hand_link, first.surface_component, (0, 0, 0), (0, 1, 0))
        certificate = compile_exact_outer_certificate([first, duplicate_component], 1.0)
        self.assertEqual(certificate.status, "failure_safe")
        self.assertTrue(all((item.lower, item.upper) == (0.0, 1.0) for item in certificate.directions.values()))

    def test_nonfinite_inputs_fail_closed(self):
        patch = _patch()
        for ell in (float("nan"), float("inf"), float("-inf")):
            certificate = compile_exact_outer_certificate([patch], ell)
            self.assertEqual(certificate.status, "failure_safe")
            self.assertTrue(
                all(
                    (item.lower, item.upper, item.label)
                    == (0.0, 1.0, "undetermined")
                    for item in certificate.directions.values()
                )
            )
        for bad in (float("nan"), float("inf"), float("-inf")):
            certificate = compile_exact_outer_certificate(
                [patch], 1.0, {"bad": (bad, 0.0, 0.0, 0.0, 0.0, 0.0)}
            )
            self.assertEqual(certificate.status, "failure_safe")
            self.assertEqual(
                (certificate.directions["bad"].lower, certificate.directions["bad"].upper),
                (0.0, 1.0),
            )

    def test_empty_direction_namespace_uses_default_failure_safe_axes(self):
        patch = _patch()
        for certificate in (
            compile_exact_outer_certificate([patch], 1.0, {}),
            compile_mode_certificate([patch], {"p0": "normal"}, 1.0, {}),
        ):
            self.assertEqual(certificate.status, "failure_safe")
            self.assertEqual(len(certificate.directions), 6)
            self.assertTrue(
                all(
                    (item.lower, item.upper, item.label)
                    == (0.0, 1.0, "undetermined")
                    for item in certificate.directions.values()
                )
            )

    def test_exact_outer_contains_every_finite_mode_assignment(self):
        patches = [
            _patch((0.2, 0.1, -0.3), (1.0, 2.0, 0.5), "p0"),
            _patch((-0.1, 0.3, 0.2), (0.0, 1.0, 1.0), "p1"),
        ]
        outer = compile_exact_outer_certificate(patches, 0.8)
        for assignment in itertools.product(("normal", "sticking"), repeat=2):
            child = compile_mode_certificate(
                patches, dict(zip(("p0", "p1"), assignment)), 0.8
            )
            self.assertEqual(child.status, "mode_point")
            for name, interval in child.directions.items():
                self.assertLessEqual(outer.directions[name].lower, interval.lower + 1e-12)
                self.assertGreaterEqual(outer.directions[name].upper + 1e-12, interval.upper)


class OuterCertificateTests(unittest.TestCase):
    def test_finite_parent_hull_contains_children(self):
        patch = _patch((0.2, 0.0, 0.0))
        normal = compile_mode_certificate([patch], {"p0": "normal"}, 1.0)
        sticking = compile_mode_certificate([patch], {"p0": "sticking"}, 1.0)
        hull = finite_hypothesis_hull([normal, sticking], parent_ids=("normal", "sticking"))
        self.assertEqual(hull.status, "finite_hull")
        self.assertEqual(hull.parent_ids, ("normal", "sticking"))
        for child in (normal, sticking):
            for name, interval in child.directions.items():
                self.assertLessEqual(hull.directions[name].lower, interval.lower)
                self.assertGreaterEqual(hull.directions[name].upper, interval.upper)

    def test_unimplemented_continuous_uncertainty_fails_closed(self):
        certificate = unresolved_continuous_uncertainty("M1 interval solver absent")
        self.assertEqual(certificate.status, "failure_safe")
        self.assertTrue(all((item.lower, item.upper) == (0.0, 1.0) for item in certificate.directions.values()))

    def test_failure_safe_and_interval_labels_are_object_invariants(self):
        with self.assertRaisesRegex(ValueError, "contradicts interval"):
            MobilityInterval(0.9, 0.9, "restricted")
        with self.assertRaisesRegex(ValueError, "failure_safe"):
            MobilityCertificate(
                {"v_x": MobilityInterval(0.9, 0.9, "free")},
                "failure_safe",
                reasons=("claimed failure",),
            )
        with self.assertRaisesRegex(ValueError, "reason"):
            MobilityCertificate(
                {"v_x": MobilityInterval(0.0, 1.0, "undetermined")},
                "failure_safe",
            )

    def test_certificate_replay_is_byte_identical(self):
        payloads = [compile_exact_outer_certificate([_patch((0.1, -0.2, 0.3))], 0.6).canonical_json() for _ in range(3)]
        self.assertEqual(len(set(payloads)), 1)
        hashes = [hashlib.sha256(payload.encode("utf-8")).hexdigest() for payload in payloads]
        self.assertEqual(len(set(hashes)), 1)


if __name__ == "__main__":
    unittest.main()
