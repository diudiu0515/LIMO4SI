import math
import unittest

from limo4si.egobody_task4 import smplx_forward


class EgoBodyTask4Tests(unittest.TestCase):
    def test_identity_orientation_faces_canonical_positive_z(self):
        forward = smplx_forward([0.0, 0.0, 0.0])
        self.assertAlmostEqual(forward[0], 0.0)
        self.assertAlmostEqual(forward[2], 1.0)

    def test_yaw_rotates_forward_on_ground_plane(self):
        forward = smplx_forward([0.0, math.pi / 2.0, 0.0])
        self.assertAlmostEqual(forward[0], 1.0, places=6)
        self.assertAlmostEqual(forward[2], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
