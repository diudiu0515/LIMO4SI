import unittest

from limo4si.task5_human_state import (
    body_centric_relation,
    quaternion_rotation_xyzw,
    ray_aabb_distance,
    supported_gaze_events,
)


class Task5HumanStateTests(unittest.TestCase):
    def test_ray_hits_axis_aligned_box(self):
        distance = ray_aabb_distance([0, 0, 0], [0, 0, 1], [[-1, 1], [-1, 1], [2, 3]])
        self.assertAlmostEqual(distance, 2.0)
        self.assertIsNone(ray_aabb_distance([0, 0, 0], [1, 0, 0], [[-1, 1], [-1, 1], [2, 3]]))

    def test_body_relation_uses_metric_wearer_axes(self):
        relation = body_centric_relation([0, 0, 0], [-1, 0, 2], [1, 0, 0], [0, 0, 1])
        self.assertEqual(relation["label"], "left-front")
        self.assertAlmostEqual(relation["distance_m"], 5 ** 0.5)

    def test_quaternion_identity(self):
        self.assertEqual(quaternion_rotation_xyzw([0, 0, 0, 1]), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def test_gaze_event_requires_consecutive_support(self):
        states = [
            {"frame_index": i, "time_s": i / 30, "gazed_object_id": value, "gazed_object_name": "cup" if value else None}
            for i, value in enumerate([None, "cup", "cup", "cup", None, "cup", "cup", "cup", "cup"])
        ]
        events = supported_gaze_events(states, minimum_run=4, maximum_internal_gap=0)
        self.assertEqual([(event["start_index"], event["end_index"]) for event in events], [(5, 8)])
        self.assertEqual(events[0]["direct_hit_count"], 4)
        self.assertEqual(events[0]["hit_support_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
