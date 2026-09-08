import copy
import json
import unittest
from pathlib import Path

from limo4si.scale_quality import validate_release


ROOT = Path(__file__).resolve().parents[1]


class Task5QualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = [json.loads(line) for line in (ROOT / "outputs/qa/task5_scaled_qa.jsonl").read_text().splitlines()]
        cls.group = {
            "name": rows[0]["case_id"],
            "video_window": {"duration_sec": rows[0]["result_json"]["timeline"][-1]["time_s"] - rows[0]["result_json"]["timeline"][0]["time_s"]},
            "qa": [{key: value for key, value in rows[0].items() if key not in {"case_index", "case_id", "video_clip"}}],
        }

    def test_accepts_annotation_derived_task5_case(self):
        self.assertEqual(validate_release({"groups": [self.group]})["status"], "ok")

    def test_rejects_head_direction_disguised_as_gaze(self):
        group = copy.deepcopy(self.group)
        group["qa"][0]["result_json"]["gaze_grounding_method"] = "head_forward_proxy"
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("direction+depth" in error for error in errors))

    def test_rejects_gaze_depth_outside_target_box(self):
        group = copy.deepcopy(self.group)
        state = group["qa"][0]["result_json"]["timeline"][0]
        state["gaze_depth_m"] = state["gaze_hit_exit_distance_m"] + 0.5
        state["gaze_depth_obb_residual_m"] = 0.5
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("fixation depth" in error for error in errors))

    def test_rejects_stale_relation_label(self):
        group = copy.deepcopy(self.group)
        group["qa"][0]["result_json"]["timeline"][0]["relation"]["label"] = "behind"
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("geometrically inconsistent" in error for error in errors))

    def test_rejects_unsustained_gaze_event(self):
        group = copy.deepcopy(self.group)
        group["qa"][0]["result_json"]["gaze_events"][0]["state_count"] = 1
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("consecutive support" in error for error in errors))

    def test_accepts_one_internal_missing_gaze_state(self):
        group = copy.deepcopy(self.group)
        result = group["qa"][0]["result_json"]
        event = result["gaze_events"][0]
        middle_frame = (int(event["start_index"]) + int(event["end_index"])) // 2
        state = next(state for state in result["timeline"] if int(state["frame"]) == middle_frame)
        self.assertEqual(state["gazed_object_id"], result["object_id"])
        state["gazed_object_id"] = None
        event["direct_hit_count"] -= 1
        event["hit_support_ratio"] = event["direct_hit_count"] / event["state_count"]
        self.assertEqual(validate_release({"groups": [group]})["status"], "ok")

    def test_rejects_hidden_third_repeated_gaze(self):
        group = copy.deepcopy(self.group)
        result = group["qa"][0]["result_json"]
        timeline = result["timeline"]
        first_end = int(result["gaze_events"][0]["end_index"])
        injected = [state for state in timeline if first_end < int(state["frame"]) <= first_end + 4]
        self.assertEqual(len(injected), 4)
        for state in injected:
            state["gazed_object_id"] = result["object_id"]
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("unreported target gaze" in error for error in errors))

    def test_rejects_stale_timestamp_alignment(self):
        group = copy.deepcopy(self.group)
        group["qa"][0]["result_json"]["alignment_diagnostics"]["maximum_observed_wearer_skew_ms"] = 25.0
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("timestamp alignment" in error for error in errors))

    def test_rejects_low_direct_gaze_hit_support(self):
        group = copy.deepcopy(self.group)
        event = group["qa"][0]["result_json"]["gaze_events"][0]
        event["direct_hit_count"] = 1
        event["hit_support_ratio"] = 0.1
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("direct ray-hit support" in error for error in errors))

    def test_rejects_stale_target_object_pose(self):
        group = copy.deepcopy(self.group)
        group["qa"][0]["result_json"]["timeline"][0]["object_pose_skew_ms"] = 75.0
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("target-object timestamp alignment" in error for error in errors))

    def test_rejects_release_level_correct_option_bias(self):
        groups = [copy.deepcopy(self.group) for _ in range(4)]
        for index, group in enumerate(groups):
            group["name"] = f"biased_{index}"
        report = validate_release({"groups": groups})
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("positions are imbalanced" in error for error in report["cases"][0]["errors"]))

    def test_rejects_unequal_or_overprecise_numeric_option(self):
        group = copy.deepcopy(self.group)
        group["qa"][0]["options"][0]["text"] += " It is 1.23 m away."
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("numeric detail" in error or "precision" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
