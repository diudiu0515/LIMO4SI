import unittest

import numpy as np

from limo4si.scale_quality import TASK5_ID, validate_release
from limo4si.semantic_gt import seal_deterministic_question
from limo4si.task5_egoexo import (
    GAZE_GROUNDING_METHOD,
    anchor_options,
    display_object_name,
    gaze_row_for_video_frame,
    unique_encoded_mask_hit,
    unique_mask_hit,
    validate_review_result,
)


class Task5EgoExoTests(unittest.TestCase):
    def test_frame_alignment_is_exact_and_fail_closed(self):
        rows = [{"frame_num": str(i), "x": "5.0", "y": "6.0"} for i in range(5)]
        row, skew = gaze_row_for_video_frame(rows, 9)
        self.assertEqual(row["frame_num"], "3")
        self.assertAlmostEqual(skew, 0.0)
        with self.assertRaisesRegex(ValueError, "alignment skew"):
            gaze_row_for_video_frame(rows, 1)

    def test_unique_mask_hit_requires_one_deep_containment(self):
        first = np.zeros((60, 60), np.uint8)
        first[10:50, 10:50] = 1
        second = np.zeros((60, 60), np.uint8)
        second[2:8, 2:8] = 1
        hit = unique_mask_hit({"bowl_0": first, "spoon_0": second}, 30, 30, minimum_boundary_margin_px=10)
        self.assertEqual(hit["object_id"], "bowl_0")
        self.assertGreaterEqual(hit["boundary_margin_px"], 10)
        self.assertEqual(hit["annotated_mask_count"], 2)

    def test_overlapping_masks_are_rejected(self):
        mask = np.ones((20, 20), np.uint8)
        with self.assertRaisesRegex(ValueError, "2 annotated masks"):
            unique_mask_hit({"a_0": mask, "b_0": mask}, 10, 10, minimum_boundary_margin_px=1)

    def test_encoded_mask_prefilter_keeps_exact_evidence(self):
        from unittest.mock import patch
        from pycocotools import mask as mask_utils

        def record(mask):
            return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))

        target = np.zeros((80, 80), np.uint8)
        target[10:70, 10:70] = 1
        far = np.zeros((80, 80), np.uint8)
        far[1:8, 1:8] = 1
        with patch("limo4si.task5_egoexo._coco_rle", side_effect=lambda value: value):
            hit = unique_encoded_mask_hit(
                {"bowl_0": record(target), "spoon_0": record(far)},
                40,
                40,
                minimum_boundary_margin_px=10,
            )
        self.assertEqual(hit["object_id"], "bowl_0")
        self.assertEqual(hit["annotated_mask_count"], 2)
        self.assertEqual(hit["bbox_prefilter_candidate_count"], 1)

    def test_stored_review_result_is_recomputed(self):
        anchors = []
        for index, object_id in enumerate(("plate_0", "bowl_0", "salt_0")):
            anchors.append({
                "video_frame": index * 150,
                "time_s": 100.0 + index * 5.0,
                "object_id": object_id,
                "unique_annotated_mask_hit": True,
                "boundary_margin_px": 12.0,
                "alignment_skew_ms": 0.0,
                "mask_width": 1408,
                "mask_height": 1408,
                "mask_area_px": 100,
            })
        result = {
            "annotation_direct": True,
            "gaze_grounding_method": GAZE_GROUNDING_METHOD,
            "anchors": anchors,
            "target_object_id": "bowl_0",
            "target_anchor_index": 1,
            "correct_semantic_option_id": "anchor_2",
        }
        validate_review_result(result)
        result["target_anchor_index"] = 0
        with self.assertRaisesRegex(ValueError, "stored anchor index"):
            validate_review_result(result)

    def test_display_name_preserves_words_and_drops_instance_suffix(self):
        self.assertEqual(display_object_name("soy_sauce_bottle_0"), "soy sauce bottle")
        self.assertEqual(display_object_name("stir_fried_spagetti_0"), "stir-fried spaghetti")


    def test_release_gate_accepts_signed_15_second_egoexo_case(self):
        anchors = []
        for index, object_id in enumerate(("plate_0", "bowl_0", "salt_0")):
            anchors.append({
                "video_frame": 3000 + index * 150,
                "time_s": 100.0 + index * 5.0,
                "object_id": object_id,
                "unique_annotated_mask_hit": True,
                "boundary_margin_px": 12.0,
                "alignment_skew_ms": 0.0,
                "mask_width": 1408,
                "mask_height": 1408,
                "mask_area_px": 100,
            })
        result = {
            "status": "ok",
            "answer_type": "gaze_point_inside_relation_mask_at_anchor",
            "T_Q": True,
            "H_Q": True,
            "S_Q": True,
            "annotation_direct": True,
            "release_status": "signed_semantic_gt_egoexo_primary",
            "annotation_source": "Ego-Exo4D v2 take_eye_gaze + Relations masks",
            "gaze_grounding_method": GAZE_GROUNDING_METHOD,
            "coordinate_frame": (
                "frame-aligned ego RGB annotation plane; containment only, "
                "no directional claim"
            ),
            "target_object_id": "bowl_0",
            "target_anchor_index": 1,
            "correct_semantic_option_id": "anchor_2",
            "anchors": anchors,
            "source_window": {
                "start_sec": 97.5,
                "end_sec": 112.5,
                "duration_sec": 15.0,
            },
            "alignment_diagnostics": {"maximum_anchor_skew_ms": 0.0},
            "source_evidence": {
                "relations": "annotations/relations_val.json",
                "gaze": "takes/example/eye_gaze/general_eye_gaze_2d.csv",
                "video": "takes/example/frame_aligned_videos/aria.mp4",
            },
        }
        question = seal_deterministic_question(
            {
                "task_id": TASK5_ID,
                "task_name": "Task 5",
                "question_type": "gaze_point_inside_relation_mask_at_anchor",
                "question": "At which marked anchor does the gaze point enter the bowl mask?",
                "options": [
                    {"label": "A", "text": "Only at the first marked anchor."},
                    {"label": "B", "text": "Only at the second marked anchor."},
                    {"label": "C", "text": "Only at the third marked anchor."},
                    {"label": "D", "text": "At none of the three marked anchors."},
                ],
                "correct_option": "B",
                "correct_answer": "Only at the second marked anchor.",
                "answer": "Only at the second marked anchor.",
                "explanation": "The synchronized gaze point is inside the bowl mask only at anchor two.",
                "status": "ok",
                "release_eligible": True,
                "result_json": result,
            },
            case_id="task5_egoexo_release_fixture",
        )
        group = {
            "name": "task5_egoexo_release_fixture",
            "video_window": {"start_sec": 97.5, "duration_sec": 15.0},
            "qa": [question],
        }
        self.assertEqual(validate_release({"groups": [group]})["status"], "ok")
        import copy
        bad = copy.deepcopy(group)
        bad_question = bad["qa"][0]
        bad_question["correct_option"] = "A"
        bad_question["correct_answer"] = bad_question["answer"] = bad_question["options"][0]["text"]
        bad["qa"][0] = seal_deterministic_question(bad_question, case_id="task5_egoexo_release_fixture")
        errors = validate_release({"groups": [bad]})["cases"][0]["errors"]
        self.assertTrue(any("locked correct option" in error for error in errors))
        group["video_window"]["duration_sec"] = 9.0
        errors = validate_release({"groups": [group]})["cases"][0]["errors"]
        self.assertTrue(any("about 15 seconds" in error for error in errors))



    def test_clip_time_options_are_explicit_and_equal_detail(self):
        options = anchor_options([2.0, 5.0, 13.0])
        self.assertEqual([row["id"] for row in options], ["anchor_1", "anchor_2", "anchor_3", "no_anchor"])
        for option in options:
            self.assertIn("2.0s", option["statement"])
            self.assertIn("5.0s", option["statement"])
            self.assertIn("13.0s", option["statement"])


if __name__ == "__main__":
    unittest.main()
