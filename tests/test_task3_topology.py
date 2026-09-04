import copy
import unittest

from limo4si.scale_quality import TASK3_ID, validate_release
from limo4si.task3_topology import trajectory_topology


class Task3TopologyTests(unittest.TestCase):
    def setUp(self):
        self.timeline = [
            {"frame": index, "t": float(index), "origin_world_m": [index * 0.5, 0.0, 0.0], "up_world_unit": [0.0, 1.0, 0.0]}
            for index in range(9)
        ]
        grounding = {
            "manual_static_review": True,
            "centroid_reprojects_inside_box": True,
            "robust_inlier_points": 20,
        }
        self.landmarks = [
            {"object_id": "left landmark_0", "object_xyz_world_m": [1.5, 0.0, 0.6], "static_scene_landmark": True, "grounding": grounding},
            {"object_id": "right landmark_0", "object_xyz_world_m": [3.0, 0.0, -0.8], "static_scene_landmark": True, "grounding": grounding},
            {"object_id": "far landmark_0", "object_xyz_world_m": [2.0, 0.0, 2.0], "static_scene_landmark": True, "grounding": grounding},
        ]

    def test_uses_local_path_tangent_for_side_and_order(self):
        result = trajectory_topology(self.timeline, self.landmarks)
        self.assertEqual(result["status"], "ok")
        by_id = {row["landmark_id"]: row for row in result["landmarks"]}
        self.assertEqual(by_id["left landmark_0"]["pass_side"], "left")
        self.assertEqual(by_id["right landmark_0"]["pass_side"], "right")
        self.assertTrue(by_id["left landmark_0"]["valid_local_pass"] )
        self.assertTrue(any(row["first_landmark"] == "left landmark_0" and row["second_landmark"] == "right landmark_0" for row in result["proximity_order_pairs"]))

    def test_release_gate_rejects_small_side_margin(self):
        topology = trajectory_topology(self.timeline, self.landmarks)
        event = copy.deepcopy(next(row for row in topology["landmarks"] if row["landmark_id"] == "left landmark_0"))
        correct = "The landmark stays on the left side around closest approach."
        question = {
            "task_id": TASK3_ID, "question_type": "local_landmark_pass_side",
            "question": "Which side is the landmark on around closest approach?",
            "options": [
                {"label": "A", "text": correct},
                {"label": "B", "text": "The landmark stays on the right side around closest approach."},
                {"label": "C", "text": "The landmark changes from left to right around closest approach."},
                {"label": "D", "text": "The landmark changes from right to left around closest approach."},
            ],
            "correct_option": "A", "correct_answer": correct, "answer": correct,
            "result_json": {"status": "ok", "answer_type": "local_landmark_pass_side", "T_Q": True, "H_Q": True, "S_Q": True, "topology": topology, "pass_event": event},
        }
        group = {"name": "task3_test", "video_window": {"duration_sec": 8.0}, "qa": [question]}
        self.assertEqual(validate_release({"groups": [group]})["status"], "ok")
        group["qa"][0]["result_json"]["pass_event"]["signed_lateral_m"] = 0.1
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("side margin" in error for error in errors))

    def test_release_gate_rejects_unreviewed_landmark(self):
        landmarks = copy.deepcopy(self.landmarks)
        landmarks[0]["static_scene_landmark"] = False
        topology = trajectory_topology(self.timeline, landmarks)
        event = next(row for row in topology["landmarks"] if row["landmark_id"] == "left landmark_0")
        correct = "The landmark stays on the left side around closest approach."
        question = {
            "task_id": TASK3_ID, "question_type": "local_landmark_pass_side",
            "question": "Which side is the landmark on around closest approach?",
            "options": [
                {"label": "A", "text": correct},
                {"label": "B", "text": "The landmark stays on the right side around closest approach."},
                {"label": "C", "text": "The landmark changes from left to right around closest approach."},
                {"label": "D", "text": "The landmark changes from right to left around closest approach."},
            ],
            "correct_option": "A", "correct_answer": correct, "answer": correct,
            "result_json": {"status": "ok", "answer_type": "local_landmark_pass_side", "T_Q": True, "H_Q": True, "S_Q": True, "topology": topology, "pass_event": event},
        }
        group = {"name": "task3_unreviewed", "video_window": {"duration_sec": 8.0}, "qa": [question]}
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("not audited as scene-fixed" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
