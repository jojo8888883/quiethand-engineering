import importlib.util
from pathlib import Path
import unittest

import numpy as np

PATH = Path(__file__).resolve().parents[1] / 'scripts/quiethand/m3_5_fusion/mask_depth_se3_icp.py'
SPEC = importlib.util.spec_from_file_location('mask_depth_se3_icp', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def rotation_y(degrees):
    angle = np.radians(degrees)
    return np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])


class MaskDepthSE3ICPTest(unittest.TestCase):
    def test_kabsch_recovers_proper_rigid_motion(self):
        rng = np.random.default_rng(4)
        source = rng.normal(size=(300, 3))
        expected_rotation = rotation_y(13)
        expected_translation = np.array([0.03, -0.02, 0.08])
        target = source @ expected_rotation.T + expected_translation
        rotation, translation, singular_values = MODULE._proper_rigid_alignment(source, target)
        self.assertTrue(np.allclose(rotation, expected_rotation, atol=1e-10))
        self.assertTrue(np.allclose(translation, expected_translation, atol=1e-10))
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0, places=10)
        self.assertGreater(singular_values[1], 0)

    def test_projective_icp_reduces_rotation_and_translation_error(self):
        x = np.linspace(-0.10, 0.10, 31)
        y = np.linspace(-0.035, 0.035, 15)
        vertices = np.array([(a, b, 0.006 * np.sin(18 * a) + 0.002 * b)
                             for a in x for b in y], dtype=np.float64)
        intrinsic = np.array([[520, 0, 320], [0, 520, 180], [0, 0, 1]], dtype=np.float64)
        target_pose = np.eye(4)
        target_pose[:3, :3] = rotation_y(7)
        target_pose[:3, 3] = [0.01, -0.005, 0.70]
        camera = vertices @ target_pose[:3, :3].T + target_pose[:3, 3]
        projected = camera @ intrinsic.T
        pixels = np.rint(projected[:, :2] / projected[:, 2:3]).astype(int)
        depth = np.zeros((360, 640), dtype=np.float64)
        for point, (u, v) in zip(camera, pixels, strict=True):
            if 0 <= u < 640 and 0 <= v < 360 and (depth[v, u] == 0 or point[2] < depth[v, u]):
                depth[v, u] = point[2]
        mask = np.ones_like(depth, dtype=bool)
        initial = np.eye(4)
        # Production poses are persisted as float32 before being read as float64.
        initial[:3, :3] = rotation_y(2).astype(np.float32)
        initial[:3, 3] = [0.015, -0.001, 0.708]
        refined, record = MODULE.refine_se3_projective_icp(
            vertices, initial, intrinsic, depth, mask,
            max_iterations=8, min_inliers=20)
        before_rotation = MODULE._rotation_angle_deg(initial[:3, :3] @ target_pose[:3, :3].T)
        after_rotation = MODULE._rotation_angle_deg(refined[:3, :3] @ target_pose[:3, :3].T)
        before_translation = np.linalg.norm(initial[:3, 3] - target_pose[:3, 3])
        after_translation = np.linalg.norm(refined[:3, 3] - target_pose[:3, 3])
        self.assertLess(after_rotation, before_rotation)
        self.assertLess(after_translation, before_translation)
        self.assertGreaterEqual(record['iteration_records'][0]['inlier_count'], 20)

    def test_missing_mask_depth_is_explicit(self):
        with self.assertRaises(ValueError):
            MODULE.refine_se3_projective_icp(
                np.ones((100, 3)), np.eye(4), np.eye(3),
                np.ones((16, 16)), np.zeros((16, 16), dtype=bool))


if __name__ == '__main__':
    unittest.main()
