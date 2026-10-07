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
    _target_event_pair_closes_window,
    balanced_category_targets,
    generate_task5_candidates,
    select_balanced_candidates,
    sustained_relation_sequence,
)
from build_task5_scaled import sequence_options

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

    def test_selector_does_not_starve_a_category_with_one_unique_source(self):
        first, second, third = REQUESTED_CATEGORIES
        candidates = [
            {"id": "first_flexible", "category": first, "sequence_name": "shared", "object_id": "a", "window_frames": [0, 150], "candidate_score": 10.0},
            {"id": "first_exclusive", "category": first, "sequence_name": "first_only", "object_id": "b", "window_frames": [0, 150], "candidate_score": 1.0},
            {"id": "second_exclusive", "category": second, "sequence_name": "second_only", "object_id": "c", "window_frames": [0, 150], "candidate_score": 1.0},
            {"id": "third_only", "category": third, "sequence_name": "shared", "object_id": "d", "window_frames": [0, 150], "candidate_score": 1.0},
        ]
        selected, report = select_balanced_candidates(candidates, target_per_category=1)
        self.assertEqual(report["status"], "ok")
        self.assertEqual({row["id"] for row in selected}, {
            "first_exclusive", "second_exclusive", "third_only",
        })

    def test_release_selector_rejects_more_than_one_case_per_source_video(self):
        with self.assertRaisesRegex(ValueError, "one question per source sequence"):
            select_balanced_candidates([], target_per_category=1, max_cases_per_sequence=2)

    def test_selector_accepts_public_question_type_target_names(self):
        candidates = [{
            "id": "one", "category": "between_repeated_gaze_events",
            "sequence_name": "sequence_one", "object_id": "object_one",
            "window_frames": [0, 150], "candidate_score": 1.0,
        }]
        selected, report = select_balanced_candidates(
            candidates,
            target_per_category=1,
            category_targets={"relation_change_between_gazes": 1},
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(report["target_by_category"]["between_repeated_gaze_events"], 1)

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

    def test_repeated_gaze_pair_must_be_the_only_target_runs_overlapping_clip(self):
        leading = {"start_index": 0, "end_index": 12}
        first = {"start_index": 30, "end_index": 42}
        second = {"start_index": 90, "end_index": 102}
        events = [leading, first, second]
        self.assertFalse(_target_event_pair_closes_window(events, first, second, (8, 110)))
        self.assertTrue(_target_event_pair_closes_window(events, first, second, (20, 110)))
        middle = {"start_index": 60, "end_index": 66}
        self.assertTrue(_target_event_pair_closes_window([first, middle, second], first, second, (20, 110)))

    def test_sequence_options_have_equal_word_counts(self):
        options, _, _ = sequence_options(
            "wooden bowl", ["front", "left-front", "right"], "case", 0,
        )
        profiles = [option_information_profile(option["text"]) for option in options]
        self.assertEqual(len({profile["words"] for profile in profiles}), 1)
        self.assertEqual(len({profile["information_units"] for profile in profiles}), 1)

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
