import copy
import unittest

from limo4si.scale_quality import TASK1_ID, TASK4_ID, validate_release


def question(task_id, qtype, result, answer="Correct answer"):
    result = {**result, "answer_type": qtype, "T_Q": True, "H_Q": True, "S_Q": True}
    return {
        "task_id": task_id,
        "question_type": qtype,
        "question": "What happens over time?",
        "options": [
            {"label": "A", "text": answer},
            {"label": "B", "text": "Wrong one"},
            {"label": "C", "text": "Wrong two"},
            {"label": "D", "text": "Wrong three"},
        ],
        "correct_option": "A",
        "correct_answer": answer,
        "answer": answer,
        "result_json": result,
    }


def metric_group(qtype="dominant_facing_relation_over_video", counts=None, coverage=1.0):
    states = []
    for index in range(8):
        states.append({
            "t": index * 15.0 / 7.0,
            "frame_id": index,
            "distance_m": 2.0 - index * 0.05,
            "facing_score": 0.8 if index < 7 else 0.0,
            "facing_state": "facing_each_other" if index < 7 else "side_by_side_or_oblique",
            "b_relative_to_a": "right_front",
            "a_relative_to_b": "left_front",
            "body_forward_field": {"state": "mutual_body_forward_field"},
            "evidence": {
                "person_a": {"pelvis_xyz_m": [0.0, 0.0, 0.0], "forward_unit": [0.0, 0.0, 1.0]},
                "person_b": {"pelvis_xyz_m": [2.0 - index * 0.05, 0.0, 0.0], "forward_unit": [0.0, 0.0, -1.0]},
            },
        })
    result = {
        "pair_timeline": {"status": "ok", "pair": ["A", "B"], "states": states},
        "facing_counts": counts or {"facing_each_other": 7, "side_by_side_or_oblique": 1},
    }
    audit = {
        "status": "complete_and_identity_aligned",
        "metric_identity_alignment": {"mapping": {"A": "V1", "B": "V2"}, "margin": 0.5},
        # The calibrator intentionally relabels aligned V tracks to A/B.
        "visible_2d_tracks": [{"id": "A", "coverage": coverage}, {"id": "B", "coverage": 1.0}],
    }
    return {
        "name": "metric_case",
        "video_window": {"duration_sec": 15.0},
        "visual_person_audit": audit,
        "qa": [question(TASK4_ID, qtype, result)],
    }


class ScaleQualityTests(unittest.TestCase):
    def test_accepts_identity_aligned_temporal_case(self):
        report = validate_release({"groups": [metric_group()]})
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["accepted_count"], 1)

    def test_rejects_stale_correct_option(self):
        group = metric_group()
        group["qa"][0]["options"][0]["text"] = "Old answer"
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("stale" in error for error in errors))

    def test_rejects_partial_visual_identity_coverage(self):
        errors = validate_release({"groups": [metric_group(coverage=0.47)]})["cases"][0]["errors"]
        self.assertTrue(any("coverage" in error for error in errors))

    def test_rejects_ambiguous_dominant_relation(self):
        group = metric_group(counts={"facing_each_other": 4, "side_by_side_or_oblique": 4})
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("ambiguous" in error for error in errors))

    def test_front_behind_task1_requires_consistent_orientation_audit(self):
        states = [
            {"frame": index, "t_sec_from_center": -7.5 + index * 15 / 7, "relation": {"distance_m": 1.0}}
            for index in range(8)
        ]
        group = {
            "name": "task1_front_case",
            "video_window": {"duration_sec": 15.0},
            "qa": [question(TASK1_ID, "relation_change_over_video", {"object_track": {"states": states}}, "It moves from behind to front.")],
        }
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("orientation sign" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
