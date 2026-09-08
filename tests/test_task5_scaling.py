import json
import unittest
from collections import Counter
from pathlib import Path

from limo4si.scale_quality import option_information_profile
from limo4si.task5_scaling import (
    CATEGORY_BY_TYPE,
    generate_task5_candidates,
    select_balanced_candidates,
)

ROOT = Path(__file__).resolve().parents[1]


class Task5ScalingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = json.loads((ROOT / "outputs/qa/task5_adt_analysis.json").read_text())

    def test_automatic_miner_finds_two_per_category(self):
        candidates, diagnostics = generate_task5_candidates(self.analysis)
        selected, report = select_balanced_candidates(candidates, target_per_category=2)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(Counter(row["category"] for row in selected), Counter({value: 2 for value in CATEGORY_BY_TYPE.values()}))
        self.assertGreaterEqual(diagnostics["candidate_count"], 6)
        self.assertEqual(len({(row["sequence_name"], *row["window_frames"]) for row in selected}), 6)

    def test_published_task5_options_have_equal_information_slots(self):
        for line in (ROOT / "outputs/qa/task5_scaled_qa.jsonl").read_text().splitlines():
            question = json.loads(line)
            profiles = [option_information_profile(option["text"]) for option in question["options"]]
            self.assertEqual(len({profile["numbers"] for profile in profiles}), 1)
            self.assertEqual(len({profile["temporal_markers"] for profile in profiles}), 1)
            self.assertLessEqual(max(p["information_units"] for p in profiles) / min(p["information_units"] for p in profiles), 1.25)

    def test_published_correct_option_positions_are_balanced(self):
        rows = [json.loads(line) for line in (ROOT / "outputs/qa/task5_scaled_qa.jsonl").read_text().splitlines()]
        counts = Counter(row["correct_option"] for row in rows)
        self.assertGreaterEqual(len(counts), 3)
        self.assertLessEqual(max(counts.values()) / len(rows), 0.5)

    def test_selection_reports_category_deficits(self):
        candidates, _ = generate_task5_candidates(self.analysis)
        _, report = select_balanced_candidates(
            [row for row in candidates if row["category"] != "last_gaze_annotated_object"],
            target_per_category=2,
        )
        self.assertEqual(report["status"], "insufficient_candidates")
        self.assertEqual(report["deficits"]["last_gaze_annotated_object"], 2)


if __name__ == "__main__":
    unittest.main()
