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
        self.assertTrue(any("ray/3D-box" in error for error in errors))

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


if __name__ == "__main__":
    unittest.main()
