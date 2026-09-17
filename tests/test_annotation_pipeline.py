import unittest

from limo4si.scale_quality import validate_release
from limo4si.task4_annotation import (
    AnnotationEvidenceError,
    generate_task4_group,
    generate_task4_release,
)


def scene(*, calibrated=True, ambiguous=False):
    frames = []
    for index in range(8):
        a_forward = [0.0, 0.0, 1.0]
        b_forward = [0.0, 0.0, -1.0]
        if ambiguous and index >= 4:
            b_forward = [0.0, 0.0, 1.0]
        frames.append({
            "t": index * 15.0 / 7.0,
            "frame_id": index,
            "people": [
                {"id": "A", "pelvis": [0.0, 0.0, 0.0], "head": [0.0, 1.6, 0.0], "forward": a_forward},
                {"id": "B", "pelvis": [0.0, 0.0, 1.0], "head": [0.0, 1.6, 1.0], "forward": b_forward},
            ],
        })
    frame = {
        "forward_axis": "projected face/body-forward direction",
        "right_axis": "scene-up cross forward",
        "right_sign": 1,
        "orientation_calibration": {"source": "annotation schema fixture"},
    }
    return {
        "scene_id": "annotation_case_01",
        "dataset": "fixture",
        "duration_sec": 15.0,
        "human_coordinate_frame": frame if calibrated else None,
        "frames": frames,
    }


class Task4AnnotationGenerationTests(unittest.TestCase):
    def test_generates_release_grade_question_without_visual_traits(self):
        group = generate_task4_group(scene())
        question = group["qa"][0]
        self.assertNotIn("man", question["question"].lower())
        self.assertNotIn("woman", question["question"].lower())
        report = validate_release({"groups": [group]})
        self.assertEqual(report["status"], "ok", report)

    def test_annotation_only_pipeline_generates_balanced_passing_question(self):
        value = scene()
        xs = [-3.0, -2.0, -1.0, -0.3, 0.3, 1.0, 2.0, 3.0]
        for frame, x in zip(value["frames"], xs):
            frame["people"][0]["pelvis"] = [x, 0.0, 0.3]
            frame["people"][0]["head"] = [x, 1.6, 0.3]
            frame["people"][1]["pelvis"] = [0.0, 0.0, 0.0]
            frame["people"][1]["head"] = [0.0, 1.6, 0.0]
        group = generate_task4_group(value)
        question = group["qa"][0]
        self.assertEqual(question["question_type"], "passing_side_and_final_position")
        parts = question["result_json"]["compound_option_parts"]
        self.assertEqual(sorted([row[0] for row in parts.values()]), ["left", "left", "right", "right"])
        self.assertEqual(validate_release({"groups": [group]})["status"], "ok")

    def test_missing_coordinate_calibration_fails_closed(self):
        with self.assertRaisesRegex(AnnotationEvidenceError, "human_coordinate_frame"):
            generate_task4_group(scene(calibrated=False))

    def test_ambiguous_relation_is_rejected_with_reason(self):
        data, audit = generate_task4_release([scene(ambiguous=True)])
        self.assertEqual(data["groups"], [])
        self.assertEqual(audit["rejected_count"], 1)
        self.assertIn("unambiguous", audit["rejected"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
