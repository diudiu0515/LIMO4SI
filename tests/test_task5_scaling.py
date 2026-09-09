import json
import unittest
from collections import Counter
from pathlib import Path

from limo4si.scale_quality import option_information_profile
from limo4si.task5_scaling import (
    CATEGORY_BY_TYPE,
    Task5CandidatePolicy,
    _expanded_window,
    generate_task5_candidates,
    select_balanced_candidates,
)

ROOT = Path(__file__).resolve().parents[1]


class Task5ScalingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = json.loads((ROOT / "outputs/qa/task5_adt_analysis.json").read_text())
        cls.sample_only_policy = Task5CandidatePolicy(
            min_event_gap_sec=0.30, max_repeated_window_sec=6.0,
            target_window_sec=4.0, min_window_sec=0.5, max_window_sec=6.0,
        )

    def test_automatic_miner_finds_two_per_category(self):
        candidates, diagnostics = generate_task5_candidates(self.analysis, self.sample_only_policy)
        selected, report = select_balanced_candidates(candidates, target_per_category=2)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(Counter(row["category"] for row in selected), Counter({value: 2 for value in CATEGORY_BY_TYPE.values()}))
        self.assertGreaterEqual(diagnostics["candidate_count"], 6)
        self.assertEqual(len({(row["sequence_name"], *row["window_frames"]) for row in selected}), 6)

    def test_window_expansion_uses_real_nine_second_annotation_span(self):
        states = {
            frame: {"frame_index": frame, "time_s": frame / 10.0}
            for frame in range(121)
        }
        policy = Task5CandidatePolicy()
        centered = _expanded_window(states, 40, 60, policy)
        self.assertIsNotNone(centered)
        start, end = centered
        self.assertLessEqual(start, 40)
        self.assertGreaterEqual(end, 60)
        self.assertAlmostEqual(states[end]["time_s"] - states[start]["time_s"], 9.0, places=6)

        end_aligned = _expanded_window(states, 90, 110, policy, align_end=True)
        self.assertIsNotNone(end_aligned)
        start, end = end_aligned
        self.assertEqual(end, 110)
        self.assertAlmostEqual(states[end]["time_s"] - states[start]["time_s"], 9.0, places=6)

    def test_default_policy_uses_real_nine_second_source_coverage(self):
        candidates, diagnostics = generate_task5_candidates(self.analysis)
        self.assertGreater(len(candidates), 0)
        times = {int(state["frame_index"]): float(state["time_s"]) for state in self.analysis["states"]}
        for candidate in candidates:
            start, end = candidate["window_frames"]
            self.assertGreaterEqual(times[end] - times[start], 8.5)
            self.assertLessEqual(times[end] - times[start], 10.0)
        self.assertEqual(diagnostics["policy"]["min_event_gap_sec"], 2.0)

    def test_default_policy_requires_two_second_repeated_gaze_gap(self):
        policy = Task5CandidatePolicy()
        self.assertEqual(policy.min_event_gap_sec, 2.0)
        self.assertEqual((policy.min_window_sec, policy.target_window_sec, policy.max_window_sec), (8.5, 9.0, 10.0))

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
        candidates, _ = generate_task5_candidates(self.analysis, self.sample_only_policy)
        _, report = select_balanced_candidates(
            [row for row in candidates if row["category"] != "last_gaze_annotated_object"],
            target_per_category=2,
        )
        self.assertEqual(report["status"], "insufficient_candidates")
        self.assertEqual(report["deficits"]["last_gaze_annotated_object"], 2)


if __name__ == "__main__":
    unittest.main()
