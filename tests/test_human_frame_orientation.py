import unittest

from limo4si.human_frame import HumanFrame, apply_forward_sign


class HumanFrameOrientationTests(unittest.TestCase):
    def test_audited_forward_flip_changes_only_forward_axis(self):
        frame = HumanFrame(
            origin=(1.0, 2.0, 3.0),
            right=(1.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            forward=(0.0, 0.0, 1.0),
        )
        flipped = apply_forward_sign(frame, -1)
        self.assertEqual(flipped.origin, frame.origin)
        self.assertEqual(flipped.right, frame.right)
        self.assertEqual(flipped.up, frame.up)
        self.assertEqual(flipped.forward, (0.0, 0.0, -1.0))
        self.assertEqual(flipped.world_to_human((1.0, 2.0, 4.0))[2], -1.0)

    def test_forward_sign_rejects_invalid_calibration(self):
        frame = HumanFrame((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
        with self.assertRaises(ValueError):
            apply_forward_sign(frame, 0)


if __name__ == "__main__":
    unittest.main()
