import unittest

from limo4si.egoexo_task4 import audit_task4_evidence


class EgoExoTask4EvidenceTests(unittest.TestCase):
    def test_single_aria_and_single_subject_fail_closed(self):
        result = audit_task4_evidence(
            capture={
                "capture_uid": "capture-1",
                "capture_name": "example",
                "cameras": [
                    {"cam_id": "aria01", "is_ego": True, "device_type": "aria"},
                    {"cam_id": "gp05", "is_ego": True, "device_type": "hero10"},
                ],
            },
            trajectory_session_uids=["session-1"],
            body_subject_counts={"example_1": 1},
            relations_take_names=[],
            available_files=["open_loop_trajectory.csv"],
        )
        self.assertFalse(result["release_eligible"])
        self.assertEqual(result["tracked_ego_cameras"], ["aria01"])
        self.assertEqual(
            result["capabilities"]["passing_side_and_final_position"]["status"],
            "rejected",
        )
        self.assertEqual(
            result["capabilities"]["physical_visibility_occlusion_timeline"]["status"],
            "rejected",
        )

    def test_relations_match_is_scoped_to_capture(self):
        result = audit_task4_evidence(
            capture={
                "capture_uid": "capture-2",
                "capture_name": "scene",
                "cameras": [
                    {"cam_id": "aria01", "is_ego": True, "device_type": "aria"},
                    {"cam_id": "aria02", "is_ego": True, "device_type": "aria"},
                ],
            },
            trajectory_session_uids=["one", "two"],
            body_subject_counts={"scene_1": 2},
            relations_take_names=["other_1", "scene_1"],
            available_files=["semidense_points.csv.gz"],
        )
        self.assertEqual(result["relations_takes"], ["scene_1"])
        # Even these inputs are insufficient: body-forward and 3D semantic blocker
        # geometry remain mandatory rather than being guessed by a language model.
        self.assertFalse(result["release_eligible"])


if __name__ == "__main__":
    unittest.main()
