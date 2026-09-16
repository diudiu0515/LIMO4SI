import copy
import unittest

from limo4si.scale_quality import TASK1_ID, TASK4_ID, validate_release
from limo4si.multihuman import derive_task4_answer_semantics
from limo4si.semantic_gt import seal_deterministic_question


def question(task_id, qtype, result, answer="Correct answer"):
    result = {**result, "answer_type": qtype, "T_Q": True, "H_Q": True, "S_Q": True}
    raw = {
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
        "explanation": "The deterministic evidence supports the recorded answer.",
        "result_json": result,
    }
    return seal_deterministic_question(raw, case_id=f"test::{task_id}::{qtype}")


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
        "pair_timeline": {
            "status": "ok", "pair": ["A", "B"], "states": states,
            "coordinate_frame": {
                "forward_axis": "projected face/body-forward direction",
                "right_axis": "scene-up cross forward", "right_sign": 1,
            },
        },
        "facing_counts": counts or {"facing_each_other": 7, "side_by_side_or_oblique": 1},
    }
    result["answer_semantics"] = derive_task4_answer_semantics(qtype, result)
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
        "person_display_aliases": {
            "A": "the man in a dark shirt",
            "B": "the man in a light shirt",
        },
        "person_display_alias_status": {
            "status": "complete",
            "schema_version": "limo4si.person_attributes.v1",
            "source": "manual_visual_review",
            "reviewed_attributes": ["gender_term", "upper_body", "lower_body"],
            "audit_status": "verified_from_original_and_localized_six_frame_sheets",
            "reviewed_at": "2026-09-15",
            "evidence_refs": [
                {"path": "outputs/audit/case_original.jpg", "sha256": "a" * 64},
                {"path": "outputs/audit/case_localized.jpg", "sha256": "b" * 64},
            ],
            "configured_descriptors": {
                "A": "the man in a dark shirt",
                "B": "the man in a light shirt",
            },
            "metric_to_visible": {"A": "V1", "B": "V2"},
        },
        "qa": [question(TASK4_ID, qtype, result)],
    }


class ScaleQualityTests(unittest.TestCase):
    def test_accepts_identity_aligned_temporal_case(self):
        report = validate_release({"groups": [metric_group()]})
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["accepted_count"], 1)

    def test_rejects_explicit_legacy_non_release_question(self):
        group = metric_group()
        group["qa"][0]["release_eligible"] = False
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertIn("question is explicitly marked non-release", errors)

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

    def test_rejects_option_with_uniquely_precise_numeric_detail(self):
        group = metric_group()
        group["qa"][0]["options"][0]["text"] = "They approach from 2.73 m to 1.14 m while facing each other."
        group["qa"][0]["correct_answer"] = group["qa"][0]["options"][0]["text"]
        group["qa"][0]["answer"] = group["qa"][0]["options"][0]["text"]
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("numeric detail" in error or "precision" in error for error in errors))

    def test_rejects_stale_task4_answer_semantics(self):
        group = metric_group()
        group["qa"][0]["result_json"]["answer_semantics"]["dominant_facing"] = "back_to_back_or_away"
        group["qa"][0] = seal_deterministic_question(group["qa"][0], case_id="metric_case")
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("answer semantics are stale" in error for error in errors))

    def test_rejects_unreviewed_person_attributes(self):
        group = metric_group()
        group["person_display_alias_status"]["source"] = "automatic_guess"
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("manual visual review" in error for error in errors))

    def test_rejects_person_attributes_without_visual_evidence(self):
        group = metric_group()
        group["person_display_alias_status"]["evidence_refs"] = []
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("original and localized visual evidence" in error for error in errors))

    def test_rejects_invalid_person_attribute_evidence_hash(self):
        group = metric_group()
        group["person_display_alias_status"]["evidence_refs"][0]["sha256"] = "not-a-hash"
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("invalid evidence hash" in error for error in errors))

    def test_rejects_descriptor_that_differs_from_attribute_audit(self):
        group = metric_group()
        group["person_display_aliases"]["B"] = "the woman in a light shirt"
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("differs from its reviewed person attributes" in error for error in errors))

    def test_rejects_duplicate_public_person_descriptions(self):
        group = metric_group()
        group["person_display_aliases"]["B"] = group["person_display_aliases"]["A"]
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("pairwise distinct" in error for error in errors))

    def test_unannotated_third_person_requires_explicit_pair_names(self):
        group = metric_group()
        group["visual_person_audit"].update({
            "persistent_visible_person_count": 2,
            "max_visible_person_count": 3,
            "metric_3d_track_count": 2,
        })
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("unnamed metric pair" in error for error in errors))

        group["qa"][0]["question"] = (
            "What happens between the man in a dark shirt and the man in a light shirt over time?"
        )
        group["qa"][0] = seal_deterministic_question(
            group["qa"][0], case_id="metric_case",
        )
        report = validate_release({"groups": [group]})
        self.assertEqual(report["status"], "ok")
        self.assertTrue(any("intermittent extra person" in warning for warning in report["cases"][0]["warnings"]))

    def test_rejects_option_information_outlier_without_numbers(self):
        group = metric_group()
        group["qa"][0]["options"][0]["text"] = (
            "Correct answer with uniquely detailed temporal and relational explanation throughout the entire clip"
        )
        group["qa"][0]["correct_answer"] = group["qa"][0]["options"][0]["text"]
        group["qa"][0]["answer"] = group["qa"][0]["options"][0]["text"]
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("information ratio" in error or "word-count ratio" in error for error in errors))

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
