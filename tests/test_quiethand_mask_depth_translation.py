import importlib.util
from pathlib import Path
import unittest

import numpy as np

PATH = Path(__file__).resolve().parents[1] / 'scripts/quiethand/m3_5_fusion/mask_depth_translation.py'
SPEC = importlib.util.spec_from_file_location('mask_depth_translation', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MaskDepthTranslationTest(unittest.TestCase):
    def test_same_pixel_depth_residual_corrects_camera_z_once(self):
        grid = np.linspace(-0.4, 0.4, 9)
        vertices = np.array([(x, y, 0) for x in grid for y in grid], dtype=float)
        pose = np.eye(4); pose[2, 3] = 2
        intrinsic = np.array([[40, 0, 40], [0, 40, 40], [0, 0, 1]], dtype=float)
        depth = np.full((80, 80), 2.5, dtype=float)
        mask = np.ones((80, 80), dtype=bool)
        refined, record = MODULE.refine_translation_once(
            vertices, pose, intrinsic, depth, mask)
        self.assertEqual(record['correspondence_count'], 81)
        self.assertAlmostEqual(refined[2, 3], 2.5, places=7)
        self.assertAlmostEqual(record['translation_delta_m'][2], 0.5, places=7)

    def test_missing_mask_correspondence_is_explicit(self):
        vertices = np.zeros((30, 3)); vertices[:, 2] = 1
        with self.assertRaises(ValueError):
            MODULE.refine_translation_once(
                vertices, np.eye(4), np.eye(3), np.ones((8, 8)),
                np.zeros((8, 8), dtype=bool))


if __name__ == '__main__':
    unittest.main()
