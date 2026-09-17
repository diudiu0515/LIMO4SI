import unittest

from limo4si.task5_dynamics import (
    gaze_onset_side_change,
    last_gaze_object_relation_change,
    relation_change_between_gazes,
)


def state(index, label, right, gaze=None, forward=None):
    return {
        "frame_index": index, "gazed_object_id": gaze,
        "forward_world": forward or [0.0, 0.0, 1.0],
        "object_relations": {"cup": {"label": label, "right_m": right}},
    }


class Task5DynamicsTests(unittest.TestCase):
    def test_relation_change_uses_two_supported_gazes(self):
        rows = [state(i, "left-front" if i < 10 else "right-front", -0.4 if i < 10 else 0.5) for i in range(20)]
        events = [
            {"object_id": "cup", "start_index": 1, "end_index": 4, "start_time_s": 0.1, "end_time_s": 0.4},
            {"object_id": "cup", "start_index": 14, "end_index": 18, "start_time_s": 3.0, "end_time_s": 3.4},
        ]
        result = relation_change_between_gazes(rows, events, "cup")
        self.assertEqual((result["start_relation"], result["end_relation"]), ("left-front", "right-front"))

    def test_gaze_onset_requires_turn_and_pre_state(self):
        rows = [state(i, "left-front", -0.4) for i in range(12)]
        rows[6] = state(6, "right-front", 0.4, gaze="cup", forward=[1.0, 0.0, 0.0])
        result = gaze_onset_side_change(rows, {"start_index": 6, "end_index": 9}, "cup")
        self.assertGreater(result["wearer_turn_deg"], 8.0)

    def test_last_object_is_selected_by_final_event_not_name(self):
        rows = [state(i, "left-front" if i < 6 else "right-front", -0.3 if i < 6 else 0.4) for i in range(12)]
        events = [{"object_id": "cup", "start_index": 8, "end_index": 11}]
        result = last_gaze_object_relation_change(rows, events)
        self.assertEqual(result["object_id"], "cup")
        self.assertEqual(result["relation_sequence"], ["left-front", "right-front"])


if __name__ == "__main__":
    unittest.main()
