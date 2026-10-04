import json
import unittest
from collections import Counter
from pathlib import Path

from limo4si.scale_quality import option_information_profile
from limo4si.task5_scaling import (
    CATEGORY_BY_TYPE,
    REQUESTED_CATEGORIES,
    Task5CandidatePolicy,
    _expanded_window,
    balanced_category_targets,
    generate_task5_candidates,
    select_balanced_candidates,
    sustained_relation_sequence,
)

ROOT = Path(__file__).resolve().parents[1]


class Task5ScalingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = json.loads((ROOT / "outputs/qa/task5_adt_analysis.json").read_text())
        cls.sample_only_policy = Task5CandidatePolicy(
            min_event_gap_sec=0.30, min_repeated_event_direct_hits=4, min_onset_event_direct_hits=4,
            min_repeated_event_duration_sec=0.0, min_onset_event_duration_sec=0.0,
            min_last_event_duration_sec=0.0, min_lateral_shift_m=0.05, min_relation_run_states=1,
            excluded_target_categories=(), max_repeated_window_sec=6.0,
            target_window_sec=4.0, min_window_sec=0.5, max_window_sec=6.0,
        )

    def test_single_short_source_reports_nonoverlap_deficits(self):
        candidates, diagnostics = generate_task5_candidates(self.analysis, self.sample_only_policy)
        selected, report = select_balanced_candidates(candidates, target_per_category=2)
        expected = 2 * len(REQUESTED_CATEGORIES)
        self.assertGreaterEqual(diagnostics["candidate_count"], expected)
        self.assertEqual(report["status"], "insufficient_candidates")
        self.assertLess(len(selected), expected)
        for index, left in enumerate(selected):
            for right in selected[index + 1:]:
                if left["sequence_name"] != right["sequence_name"]:
                    continue
                self.assertLess(
                    min(left["window_frames"][1], right["window_frames"][1]),
                    max(left["window_frames"][0], right["window_frames"][0]),
                )

    def test_balanced_selector_reaches_total_with_distinct_sources(self):
        candidates = []
        for category in REQUESTED_CATEGORIES:
            for index in range(2):
                candidates.append({
                    "id": f"{category}_{index}",
                    "category": category,
                    "sequence_name": f"sequence_{category}_{index}",
                    "object_id": f"object_{index}",
                    "window_frames": [0, 150],
                    "candidate_score": 1.0,
                })
        selected, report = select_balanced_candidates(candidates, target_per_category=2)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(Counter(row["category"] for row in selected), Counter({
            value: 2 for value in REQUESTED_CATEGORIES
        }))
        self.assertEqual(balanced_category_targets(40), {
            REQUESTED_CATEGORIES[0]: 14,
            REQUESTED_CATEGORIES[1]: 13,
            REQUESTED_CATEGORIES[2]: 13,
        })

    def test_window_expansion_uses_real_fifteen_second_annotation_span(self):
        states = {
            frame: {"frame_index": frame, "time_s": frame / 10.0}
            for frame in range(201)
        }
        policy = Task5CandidatePolicy()
        centered = _expanded_window(states, 40, 60, policy)
        self.assertIsNotNone(centered)
        start, end = centered
        self.assertLessEqual(start, 40)
        self.assertGreaterEqual(end, 60)
        self.assertAlmostEqual(states[end]["time_s"] - states[start]["time_s"], 15.0, places=6)

        end_aligned = _expanded_window(states, 160, 180, policy, align_end=True)
        self.assertIsNotNone(end_aligned)
        start, end = end_aligned
        self.assertEqual(end, 180)
        self.assertAlmostEqual(states[end]["time_s"] - states[start]["time_s"], 15.0, places=6)

    def test_default_policy_rejects_fixture_without_fifteen_second_coverage(self):
        candidates, diagnostics = generate_task5_candidates(self.analysis)
        self.assertEqual(candidates, [])
        self.assertEqual(diagnostics["policy"]["min_event_gap_sec"], 2.0)

    def test_default_policy_requires_two_second_repeated_gaze_gap(self):
        policy = Task5CandidatePolicy()
        self.assertEqual(policy.min_event_gap_sec, 2.0)
        self.assertEqual((policy.min_window_sec, policy.target_window_sec, policy.max_window_sec), (14.5, 15.0, 15.5))

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
        self.assertEqual(len(counts), min(len(rows), 4))
        self.assertLessEqual(max(counts.values()) / len(rows), 0.5)

    def test_relation_sequence_ignores_brief_boundary_flicker(self):
        labels = ["left-front"] * 10 + ["front"] * 2 + ["left-front"] * 8 + ["front"] * 6
        self.assertEqual(sustained_relation_sequence(labels, minimum_run=6), ["left-front", "front"])

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
