import importlib.util
from pathlib import Path
import unittest

import numpy as np

PATH = Path(__file__).resolve().parents[1] / 'scripts/quiethand/m3_5_fusion/mask_pose_consistency.py'
SPEC = importlib.util.spec_from_file_location('mask_pose_consistency', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MaskPoseConsistencyTest(unittest.TestCase):
    def test_projected_hull_iou_prefers_pose_on_observed_mask(self):
        vertices = np.array([
            [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
        ], dtype=float)
        intrinsic = np.array([[10, 0, 20], [0, 10, 20], [0, 0, 1]], dtype=float)
        observed = np.zeros((40, 40), dtype=bool)
        observed[15:26, 15:26] = True
        aligned = np.eye(4); aligned[2, 3] = 2
        shifted = aligned.copy(); shifted[0, 3] = 3
        self.assertGreater(
            MODULE.projected_mask_iou(vertices, aligned, intrinsic, observed),
            MODULE.projected_mask_iou(vertices, shifted, intrinsic, observed))

    def test_equal_mask_support_prefers_temporal_track(self):
        self.assertEqual(MODULE.select_source(0.4, 0.4), 'track')
        self.assertEqual(MODULE.select_source(0.3, 0.4), 'register')


if __name__ == '__main__':
    unittest.main()
