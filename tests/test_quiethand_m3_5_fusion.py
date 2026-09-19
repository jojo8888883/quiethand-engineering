import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/quiethand/m3_5_fusion/build_fusion_preview.py"
SPEC = importlib.util.spec_from_file_location("m3_5_fusion", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FusionRulesTest(unittest.TestCase):
    def test_contact_requires_count_and_consecutive_run(self):
        values = [0.02] * 8 + [0.08] * 7
        result = MODULE.classify_contact(values)
        self.assertEqual(result["state"], "maintained_close")
        broken = [0.02, 0.08] * 7 + [0.02]
        result = MODULE.classify_contact(broken)
        self.assertEqual(result["state"], "boundary_near")

    def test_relative_envelope_uses_object_coordinates(self):
        hand = np.zeros((15, 778, 3), dtype=np.float32)
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], 15, axis=0)
        shift = np.linspace(0.0, 0.02, 15)
        hand[:, :, 0] = shift[:, None]
        envelope = MODULE.relative_envelope(hand, poses)
        self.assertIsNotNone(envelope)
        self.assertTrue(envelope["stable_with_object"])
        self.assertLess(envelope["diagonal_span_m"], 0.05)

    def test_conflicting_geometry_is_not_forced(self):
        maintained = {"contact": {"state": "maintained_close"}, "relative_motion": {"stable_with_object": True}}
        absent = {"contact": {"state": "not_close"}, "relative_motion": {"stable_with_object": False}}
        result = MODULE.role_fusion("support", maintained, absent)
        self.assertEqual(result["geometry_role"], "active_candidate")
        self.assertEqual(result["relation"], "conflict")
        self.assertEqual(result["final_candidate"], "ambiguous_conflict")

    def test_preview_contains_persistent_review_controls(self):
        html = MODULE.html_document([], {})
        self.assertIn("data-choice=\"ok\"", html)
        self.assertIn("data-choice=\"bad\"", html)
        self.assertIn("data-choice=\"other\"", html)
        self.assertIn("quiethand-m3-5-fusion-review-v1", html)
        self.assertIn("quiethand-m3-5-fusion-review", html)
        self.assertIn("导出本轮评审", html)


if __name__ == "__main__":
    unittest.main()
