import copy
import json
import unittest
from pathlib import Path

from limo4si.scale_quality import validate_release
from scripts.build_task5_scaled import build_question


ROOT = Path(__file__).resolve().parents[1]


class Task5QualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # ADT remains a compatibility path, but the main Task 5 JSONL is now
        # EgoExo4D. Build the legacy 3D-ray fixture directly from its retained
        # analysis/config instead of assuming ownership of the main output.
        analysis = json.loads((ROOT / "outputs/qa/task5_adt_analysis.json").read_text())
        config = json.loads((ROOT / "configs/task5_release_cases.json").read_text())
        spec = dict(config["cases"][0], _correct_option_index=0)
        question, _ = build_question(spec, analysis)
        lo, hi = (int(value) for value in spec["window_frames"])
        cls.group = {
            "name": spec["id"],
            "video_window": {
                "duration_sec": (
                    analysis["states"][hi]["time_s"] - analysis["states"][lo]["time_s"]
                )
            },
            "qa": [question],
        }
        repeated_question, _ = build_question({
            "id": "test_repeated_kitchen_island",
            "question_type": "relation_change_between_gazes",
            "object_name": "KitchIsland",
            "event_start_frames": [114, 182],
            "window_frames": [0, 256],
            "_correct_option_index": 0,
        }, analysis)
        cls.repeated_group = {
            "name": "test_repeated_kitchen_island",
            "video_window": {"duration_sec": analysis["states"][256]["time_s"] - analysis["states"][0]["time_s"]},
            "qa": [repeated_question],
        }
        onset_question, _ = build_question({
            "id": "test_onset_wooden_bowl",
            "question_type": "gaze_onset_side_change",
            "object_name": "WoodenBowl",
            "event_start_frames": [84],
            "pre_frame": 58,
            "window_frames": [0, 270],
            "_correct_option_index": 0,
        }, analysis)
        cls.onset_group = {
            "name": "test_onset_wooden_bowl",
            "video_window": {
                "duration_sec": analysis["states"][270]["time_s"] - analysis["states"][0]["time_s"]
            },
            "qa": [onset_question],
        }

    def validate_current(self, data):
        return validate_release(data)

    def test_accepts_annotation_derived_task5_case(self):
        self.assertEqual(self.validate_current({"groups": [self.group]})["status"], "ok")

    def test_default_gate_rejects_too_short_public_video(self):
        group = copy.deepcopy(self.group)
        group["video_window"]["duration_sec"] = 2.3
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("about 9 seconds" in error for error in errors))

    def test_default_gate_rejects_short_repeated_gaze_gap(self):
        group = copy.deepcopy(self.repeated_group)
        events = group["qa"][0]["result_json"]["gaze_events"]
        events[1]["start_time_s"] = events[0]["end_time_s"] + 0.5
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("at least 2 seconds" in error for error in errors))

    def test_rejects_large_support_target(self):
        group = copy.deepcopy(self.group)
        group["qa"][0]["result_json"]["object_category"] = "table"
        errors = self.validate_current({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("large scene/support object" in error for error in errors))

    def test_rejects_short_gaze_onset_event(self):
        group = copy.deepcopy(self.onset_group)
        event = group["qa"][0]["result_json"]["gaze_events"][0]
        event["end_time_s"] = event["start_time_s"] + 0.1
        errors = self.validate_current({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("duration is below" in error for error in errors))

    def test_rejects_head_direction_disguised_as_gaze(self):
        group = copy.deepcopy(self.group)
        group["qa"][0]["result_json"]["gaze_grounding_method"] = "head_forward_proxy"
        errors = self.validate_current({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("direction+depth" in error for error in errors))

    def test_rejects_gaze_depth_outside_target_box(self):
        group = copy.deepcopy(self.group)
        result = group["qa"][0]["result_json"]
        state = next(
            state for state in result["timeline"]
            if str(state.get("gazed_object_id")) == str(result["object_id"])
        )
        state["gaze_depth_m"] = state["gaze_hit_exit_distance_m"] + 0.5
        state["gaze_depth_obb_residual_m"] = 0.5
        errors = self.validate_current({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("fixation depth" in error for error in errors))

    def test_rejects_stale_relation_label(self):
        group = copy.deepcopy(self.group)
        group["qa"][0]["result_json"]["timeline"][0]["relation"]["label"] = "behind"
        errors = self.validate_current({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("geometrically inconsistent" in error for error in errors))

    def test_rejects_unsustained_gaze_event(self):
        group = copy.deepcopy(self.group)
        group["qa"][0]["result_json"]["gaze_events"][0]["state_count"] = 1
        errors = self.validate_current({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("consecutive support" in error for error in errors))

    def test_rejects_post_release_internal_gaze_mutation(self):
        group = copy.deepcopy(self.group)
        result = group["qa"][0]["result_json"]
        event = result["gaze_events"][0]
        middle_frame = (int(event["start_index"]) + int(event["end_index"])) // 2
        state = next(state for state in result["timeline"] if int(state["frame"]) == middle_frame)
        self.assertEqual(state["gazed_object_id"], result["object_id"])
        state["gazed_object_id"] = None
        event["direct_hit_count"] -= 1
        event["hit_support_ratio"] = event["direct_hit_count"] / event["state_count"]
        report = self.validate_current({"groups": [group]})
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any(
            "evidence signature is stale" in error
            for error in report["cases"][0]["errors"]
        ))

    def test_rejects_hidden_third_repeated_gaze(self):
        group = copy.deepcopy(self.repeated_group)
        result = group["qa"][0]["result_json"]
        timeline = result["timeline"]
        first_end = int(result["gaze_events"][0]["end_index"])
        injected = [state for state in timeline if first_end < int(state["frame"]) <= first_end + 4]
        self.assertEqual(len(injected), 4)
        for state in injected:
            state["gazed_object_id"] = result["object_id"]
        errors = self.validate_current({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("unreported target gaze" in error for error in errors))

    def test_rejects_stale_timestamp_alignment(self):
        group = copy.deepcopy(self.group)
        group["qa"][0]["result_json"]["alignment_diagnostics"]["maximum_observed_wearer_skew_ms"] = 25.0
        errors = self.validate_current({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("timestamp alignment" in error for error in errors))

    def test_rejects_low_direct_gaze_hit_support(self):
        group = copy.deepcopy(self.group)
        event = group["qa"][0]["result_json"]["gaze_events"][0]
        event["direct_hit_count"] = 1
        event["hit_support_ratio"] = 0.1
        errors = self.validate_current({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("direct ray-hit support" in error for error in errors))

    def test_rejects_stale_target_object_pose(self):
        group = copy.deepcopy(self.group)
        group["qa"][0]["result_json"]["timeline"][0]["object_pose_skew_ms"] = 75.0
        errors = self.validate_current({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("target-object timestamp alignment" in error for error in errors))

    def test_rejects_release_level_correct_option_bias(self):
        groups = [copy.deepcopy(self.group) for _ in range(4)]
        for index, group in enumerate(groups):
            group["name"] = f"biased_{index}"
        report = self.validate_current({"groups": groups})
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any("positions are imbalanced" in error for error in report["cases"][0]["errors"]))

    def test_rejects_unequal_or_overprecise_numeric_option(self):
        group = copy.deepcopy(self.group)
        group["qa"][0]["options"][0]["text"] += " It is 1.23 m away."
        errors = self.validate_current({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("numeric detail" in error or "precision" in error for error in errors))

    def test_rejects_tampered_result_signature(self):
        group = copy.deepcopy(self.group)
        group["qa"][0]["result_json"]["answer_signature"] = "sha256:tampered"
        errors = self.validate_current({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("result_json answer signature" in error for error in errors))

    def test_rejects_tampered_semantic_fact(self):
        group = copy.deepcopy(self.group)
        group["qa"][0]["semantic_gt"]["semantic_facts"][0]["value"] = "tampered"
        errors = self.validate_current({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("answer signature is stale" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
