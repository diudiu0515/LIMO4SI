import math
import unittest

import numpy as np

from limo4si.egobody_task4 import pv_face_axes, smplx_forward


class EgoBodyTask4Tests(unittest.TestCase):
    def test_identity_orientation_faces_canonical_positive_z(self):
        forward = smplx_forward([0.0, 0.0, 0.0])
        self.assertAlmostEqual(forward[0], 0.0)
        self.assertAlmostEqual(forward[2], 1.0)

    def test_yaw_rotates_forward_on_ground_plane(self):
        forward = smplx_forward([0.0, math.pi / 2.0, 0.0])
        self.assertAlmostEqual(forward[0], 1.0, places=6)
        self.assertAlmostEqual(forward[2], 0.0, places=6)


    def test_pv_axes_use_face_negative_z_and_human_right_positive_x(self):
        forward, right = pv_face_axes(np.eye(4), np.eye(4))
        self.assertAlmostEqual(forward[0], 0.0)
        self.assertAlmostEqual(forward[2], -1.0)
        self.assertAlmostEqual(right[0], 1.0)
        self.assertAlmostEqual(right[2], 0.0)

    def test_pv_axes_applies_official_calibration(self):
        calibration = np.eye(4)
        calibration[:3, :3] = np.array(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        )
        forward, right = pv_face_axes(np.eye(4), calibration)
        self.assertAlmostEqual(forward[0], -1.0)
        self.assertAlmostEqual(forward[2], 0.0)
        self.assertAlmostEqual(right[0], 0.0)
        self.assertAlmostEqual(right[2], -1.0)


if __name__ == "__main__":
    unittest.main()
