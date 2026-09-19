import importlib.util
from pathlib import Path
import unittest

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "scripts/quiethand/m3_5_fusion/diagnose_trajectories.py"
SPEC = importlib.util.spec_from_file_location("trajectory_diagnosis", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DiagnosisTest(unittest.TestCase):
    def test_real_motion_is_not_counted_as_estimation_error(self):
        reference = np.array([[0., 0, 0], [0, 0, 1], [1, 0, 1]])
        result = MODULE.trajectory_errors(reference, reference)
        self.assertEqual(result["mean_position_error_m"], 0.)
        self.assertEqual(result["mean_displacement_error_m"], 0.)
        self.assertGreater(MODULE.span(reference), .05)

    def test_constant_offset_does_not_become_jitter(self):
        reference = np.array([[0., 0, 0], [1, 0, 0], [2, 1, 0]])
        result = MODULE.trajectory_errors(reference + [0, 0, 2], reference)
        self.assertEqual(result["mean_position_error_m"], 2.)
        self.assertEqual(result["mean_displacement_error_m"], 0.)

    def test_world_to_camera_rotation_and_translation(self):
        pose = np.eye(4)[None]
        pose[0, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        pose[0, :3, 3] = [2, 3, 4]
        np.testing.assert_allclose(MODULE.transform_points(pose, np.array([[[1., 0, 0]]])), [[[2, 4, 4]]])

    def test_relative_motion_cancels_shared_rigid_motion(self):
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[:, :3, 3] = [[0, 0, 0], [1, 2, 0], [2, 3, 0]]
        local = np.array([.1, .2, .3])
        centers = poses[:, :3, 3] + local
        np.testing.assert_allclose(MODULE.relative_positions(centers, poses), np.tile(local, (3, 1)))


if __name__ == "__main__":
    unittest.main()
