import unittest

from limo4si.panoptic_task4 import _body, _projected_inside, _stable_person_ids


class PanopticTask4Tests(unittest.TestCase):
    def test_body_uses_confident_coco19_head_and_midhip_in_meters(self):
        joints = [0.0] * (19 * 4)
        joints[1 * 4:1 * 4 + 4] = [100.0, 200.0, 300.0, 0.8]
        joints[2 * 4:2 * 4 + 4] = [50.0, 100.0, 150.0, 0.9]
        body = _body({"id": 7, "joints19": joints}, 0.2)
        self.assertEqual(body["track_id"], 7)
        self.assertEqual(body["head"], [1.0, 2.0, 3.0])
        self.assertEqual(body["pelvis"], [0.5, 1.0, 1.5])

    def test_stable_ids_require_window_coverage(self):
        frames = [
            {1: {}, 2: {}, 3: {}, **({4: {}} if index == 0 else {})}
            for index in range(10)
        ]
        self.assertEqual(_stable_person_ids(frames, 0.9), [1, 2, 3])

    def test_camera_projection_rejects_points_outside_the_image(self):
        camera = {
            "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "t": [[0], [0], [0]],
            "K": [[100, 0, 50], [0, 100, 50], [0, 0, 1]],
            "distCoef": [0, 0, 0, 0, 0],
            "resolution": [100, 100],
        }
        self.assertTrue(_projected_inside([0, 0, 1], camera))
        self.assertFalse(_projected_inside([2, 0, 1], camera))


if __name__ == "__main__":
    unittest.main()
