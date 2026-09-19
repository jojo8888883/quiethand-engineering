import unittest

from examples.reuse_demo import build_demo
from quiethand.models import EvidenceRecord


class ReuseDemoTest(unittest.TestCase):
    def test_explicit_binding_selects_mesh_instead_of_list_order(self):
        result = build_demo()
        self.assertEqual(result["selected_mesh"]["mesh_path"], "meshes/bowl.obj")
        self.assertEqual(result["role_to_entity"]["tool"], "bowl")

    def test_uncertainty_and_provenance_survive_serialization(self):
        result = build_demo()
        self.assertIsNone(result["unresolved_target"])
        missing = EvidenceRecord.from_dict(result["evidence"][1])
        self.assertEqual(missing.status, "ambiguous")
        self.assertIsNone(missing.value)
        self.assertTrue(missing.failure_reason)
        self.assertEqual(missing.provenance.source_name, "synthetic-interface-demo")

    def test_fixture_never_claims_video_inference(self):
        result = build_demo()
        self.assertFalse(result["model_called"])
        self.assertEqual(result["example_kind"], "synthetic_interface_only")
        self.assertNotEqual(result["synthetic_constraint"]["status"], "failure_safe")


if __name__ == "__main__":
    unittest.main()
