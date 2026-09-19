import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from quiethand.object_identity import ObjectIdentityError, parse_visual_binding, resolve_role_entity


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("identity_repair", ROOT / "scripts/quiethand/m3_5_fusion/repair_object_identity.py")
REPAIR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPAIR)


class ObjectIdentityTest(unittest.TestCase):
    def setUp(self):
        self.event = {"event_id": "fixture", "object_state": {"entities": [
            {"entity_id": "cad_020", "mesh_path": "plate.obj"},
            {"entity_id": "cad_146", "mesh_path": "bowl.obj"}]}}
        self.raw = json.dumps({"region_0": {"entity_id": "cad_146", "reason": "round deep bowl"},
                               "region_1": {"entity_id": "cad_020", "reason": "shallow rectangular plate"}})

    def test_swapped_roles_select_physical_mesh_not_role_name(self):
        binding = parse_visual_binding(self.raw, self.event)
        self.assertEqual(resolve_role_entity(self.event, binding, "tool")["mesh_path"], "bowl.obj")
        self.assertEqual(resolve_role_entity(self.event, binding, "target")["mesh_path"], "plate.obj")
        self.event["object_state"]["entities"].reverse()
        self.assertEqual(resolve_role_entity(self.event, binding, "tool")["mesh_path"], "bowl.obj")

    def test_unknown_and_duplicate_entities_are_not_implicitly_repaired(self):
        for bad in [self.raw.replace("cad_146", "cad_unknown"), self.raw.replace("cad_020", "cad_146")]:
            with self.assertRaises(ObjectIdentityError):
                parse_visual_binding(bad, self.event)

    def test_unknown_region_stays_unresolved(self):
        raw = self.raw.replace('"cad_146"', 'null')
        binding = parse_visual_binding(raw, self.event)
        self.assertIsNone(resolve_role_entity(self.event, binding, "tool"))
        self.assertEqual(resolve_role_entity(self.event, binding, "target")["entity_id"], "cad_020")

    def test_other_event_cannot_supply_binding(self):
        binding = parse_visual_binding(self.raw, self.event)
        binding["event_id"] = "other"
        with self.assertRaises(ObjectIdentityError):
            resolve_role_entity(self.event, binding, "tool")

    def test_old_role_named_mesh_contract_is_not_accepted(self):
        event = {"event_id": "fixture", "object_state": {"tool_mesh_path": "plate.obj", "target_mesh_path": "bowl.obj"}}
        with self.assertRaises(KeyError):
            parse_visual_binding(self.raw, event)

    def test_rerun_passes_correct_mesh_mask_to_estimator_and_preserves_old_outputs(self):
        self._exercise_rerun(unresolved=False)

    def test_unresolved_identity_does_not_call_estimator_but_keeps_segmentation(self):
        self._exercise_rerun(unresolved=True)

    def test_unchanged_identity_reuses_pose_without_loading_model(self):
        self._exercise_rerun(unresolved=False, unchanged=True)

    def _exercise_rerun(self, unresolved, unchanged=False):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            event = copy.deepcopy(self.event)
            event["previous_role_to_entity"] = {"tool": "cad_146" if unchanged else "cad_020", "target": "cad_020" if unchanged else "cad_146"}
            Image.new("RGB", (4, 3)).save(workspace / "rgb.png")
            np.save(workspace / "depth.npy", np.ones((3, 4), np.float32))
            np.save(workspace / "K.npy", np.eye(3))
            event["segmentation"] = {"ordered_rgb_frames": ["rgb.png"] * 15}
            event["object_state"].update(ordered_depth_frames=["depth.npy"] * 15, intrinsic_path="K.npy")
            binding = parse_visual_binding(self.raw.replace('"cad_146"', 'null') if unresolved else self.raw, event)
            REPAIR.atomic_json(workspace / REPAIR.REPAIR / "binding/events/fixture.json", binding)
            old = workspace / "results_v1_2/perception"
            folder = old / "events/fixture"
            folder.mkdir(parents=True)
            items = {}
            for index, role in enumerate(REPAIR.ROLES):
                mask = np.zeros((15, 3, 4), bool)
                mask[:, :, index] = True
                np.save(folder / f"{role}_mask.npy", mask)
                np.save(folder / f"{role}_pose.npy", np.full((15, 4, 4), index))
                items[role] = {"status": "observed", "artifact": {"relative_path": f"events/fixture/{role}_mask.npy"}, "failure_reason": None}
            REPAIR.atomic_json(folder / "segmentation.json", {"items": items})
            REPAIR.atomic_json(folder / "object_state.json", {"items": {role: {"status": "observed", "artifact": {"relative_path": f"events/fixture/{role}_pose.npy"}, "failure_reason": None} for role in REPAIR.ROLES}})
            original = {p.name: p.read_bytes() for p in folder.iterdir()}
            calls = []
            def estimate(*args):
                mesh, rgbs, depths, intrinsic, masks = args[4:]
                calls.append((mesh.name, int(np.nonzero(masks[0])[1][0])))
                return np.repeat(np.eye(4, dtype=np.float32)[None], 15, axis=0)
            fake = types.ModuleType("run_perception")
            fake._load_foundation = lambda _: (None,) * 4
            fake._poses = estimate
            fake._artifact = lambda root, path, **kwargs: {"relative_path": path.relative_to(root).as_posix(), **kwargs}
            with patch.dict("sys.modules", {"run_perception": fake}):
                REPAIR.rerun(types.SimpleNamespace(workspace=workspace, output_root=REPAIR.REPAIR, reuse_root=None), {"events": [event]})
            self.assertEqual(calls, [] if unchanged else [("plate.obj", 1)] if unresolved else [("bowl.obj", 0), ("plate.obj", 1)])
            self.assertEqual(original, {p.name: p.read_bytes() for p in folder.iterdir()})
            output = workspace / REPAIR.REPAIR / "perception"
            objects = REPAIR.read(output / "events/fixture/object_state.json")["items"]
            self.assertEqual(objects["target"]["entity_id"], "cad_020")
            self.assertEqual(objects["tool"]["status"], "abstain" if unresolved else "observed")
            self.assertEqual(objects["target"]["pose_origin"], "reused_existing_pose" if unchanged else "rerun_pose")
            if unchanged:
                for role in REPAIR.ROLES:
                    self.assertTrue((output / objects[role]["artifact"]["relative_path"]).samefile(folder / f"{role}_pose.npy"))
            segments = REPAIR.read(output / "events/fixture/segmentation.json")["items"]
            for role in REPAIR.ROLES:
                self.assertTrue((output / segments[role]["artifact"]["relative_path"]).samefile(folder / f"{role}_mask.npy"))


if __name__ == "__main__":
    unittest.main()
