import unittest

from limo4si.scale_quality import validate_release
from limo4si.semantic_gt import TemplateLanguageRealizer
from limo4si.task4_contract import TASK4_CAPABILITIES
from limo4si.task4_scaling import (
    AnnotationEvidenceError,
    balanced_category_targets,
    generate_task4_candidates,
    generate_task4_scale_release,
    realize_task4_release,
    select_task4_candidates,
)


def _person(person_id, position):
    return {
        "id": person_id,
        "pelvis": position,
        "head": [position[0], 1.6, position[2]],
        "forward": [0.0, 0.0, 1.0],
    }


def _scene(name, a_positions, b_positions, c_positions=None, blockers=None):
    frames = []
    for index, (a_position, b_position) in enumerate(zip(a_positions, b_positions)):
        people = [_person("A", a_position), _person("B", b_position)]
        if c_positions is not None:
            people.append(_person("C", c_positions[index]))
        frames.append({
            "t": index * 15.0 / (len(a_positions) - 1),
            "frame_id": index,
            "people": people,
        })
    value = {
        "scene_id": name,
        "duration_sec": 15.0,
        "video_clip": f"{name}.mp4",
        "frames": frames,
        "person_identities": {
            "A": "the person in red",
            "B": "the person in blue",
            "C": "the person in green",
        },
        "human_coordinate_frame": {
            "forward_axis": "projected face/body-forward direction",
            "right_axis": "scene-up cross forward",
            "right_sign": 1,
            "orientation_calibration": {"source": "test fixture"},
        },
    }
    if c_positions is not None:
        value["metric_person_ids"] = ["A", "B", "C"]
    if blockers is not None:
        value["blockers"] = blockers
    return value


def _scenes():
    zeros = [[0.0, 0.0, 0.0] for _ in range(8)]
    passing = _scene(
        "passing",
        [[x, 0.0, 1.0] for x in (-3.0, -2.0, -1.0, -0.2, 0.4, 1.2, 2.2, 3.0)],
        zeros,
    )
    reunion_distances = (1.0, 1.2, 2.0, 3.0, 2.5, 1.5, 1.1, 1.0)
    reunion = _scene(
        "reunion",
        zeros,
        [[-distance if index < 4 else distance, 0.0, 0.0]
         for index, distance in enumerate(reunion_distances)],
    )
    group = _scene(
        "group",
        zeros,
        [[0.6 + index * 2.4 / 7, 0.0, 0.0] for index in range(8)],
        [[3.0 - index * 2.4 / 7, 0.0, 0.0] for index in range(8)],
    )
    visibility = _scene(
        "visibility",
        zeros,
        [[2.0 - index * 2.0 / 7, 0.0, 2.0] for index in range(8)],
        blockers=[{"id": "partition", "center": [0.0, 1.6, 1.0], "radius": 0.3}],
    )
    return passing, reunion, group, visibility


class Task4ScalingTests(unittest.TestCase):
    def test_forty_case_target_is_balanced_across_all_seven_categories(self):
        targets = balanced_category_targets(40)
        self.assertEqual(tuple(targets), TASK4_CAPABILITIES)
        self.assertEqual(sum(targets.values()), 40)
        self.assertLessEqual(max(targets.values()) - min(targets.values()), 1)
        self.assertTrue(all(value >= 5 for value in targets.values()))

    def test_scale_release_accepts_a_single_pass_scene_iterator(self):
        scenes = _scenes()
        data, audit = generate_task4_scale_release(
            (scene for scene in scenes), target_count=7, allow_partial=True,
        )
        self.assertEqual(audit["input_scene_count"], len(scenes))
        self.assertTrue(data["groups"])

    def test_synthetic_annotations_cover_all_seven_release_categories(self):
        accepted = {}
        for scene in _scenes():
            candidates, _ = generate_task4_candidates(scene)
            for candidate in candidates:
                group = {
                    key: value for key, value in candidate.items()
                    if not key.startswith("_")
                }
                report = validate_release({"groups": [group]})
                self.assertEqual(
                    report["status"],
                    "ok",
                    (candidate["_capability"], report["cases"][0]["errors"]),
                )
                accepted.setdefault(candidate["_capability"], candidate)
        self.assertEqual(set(accepted), set(TASK4_CAPABILITIES))

    def test_transient_physical_occlusion_is_a_complete_timeline_event(self):
        zeros = [[0.0, 0.0, 0.0] for _ in range(8)]
        scene = _scene(
            "transient_visibility",
            zeros,
            [[2.0, 0.0, z] for z in (2.0, 2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0)],
            blockers=[{"id": "partition", "center": [1.0, 1.6, 0.0], "radius": 0.3}],
        )
        candidates, _ = generate_task4_candidates(scene)
        visibility = next(
            row for row in candidates
            if row["_capability"] == "physical_visibility_occlusion_timeline"
        )
        question = visibility["qa"][0]
        self.assertEqual(
            question["result_json"]["answer_semantics"]["stable_visibility_states"],
            ["clear", "blocked", "clear"],
        )
        group = {key: value for key, value in visibility.items() if not key.startswith("_")}
        self.assertEqual(validate_release({"groups": [group]})["status"], "ok")

    def test_selection_uses_one_question_per_source_window(self):
        candidates = [
            {
                "name": f"scene_{index}",
                "_capability": category,
                "_candidate_score": 1.0,
                "qa": [],
            }
            for index, category in enumerate(TASK4_CAPABILITIES)
        ]
        selected, report = select_task4_candidates(
            candidates, {category: 1 for category in TASK4_CAPABILITIES},
        )
        self.assertEqual(report["status"], "ok")
        self.assertEqual(len(selected), len(TASK4_CAPABILITIES))

        self.assertEqual(len({row["name"] for row in selected}), len(selected))

    def test_selection_uses_one_question_per_source_video(self):
        candidates = []
        for index, category in enumerate(TASK4_CAPABILITIES):
            candidates.append({
                "name": f"scene_{index}",
                "video_window": {"source": "shared_long_video.mp4"},
                "_capability": category,
                "_candidate_score": 1.0,
                "qa": [],
            })
        selected, report = select_task4_candidates(
            candidates, {category: 1 for category in TASK4_CAPABILITIES},
        )
        self.assertEqual(len(selected), 1)
        self.assertTrue(report["one_question_per_source_video"])
        self.assertEqual(sum(report["deficits"].values()), len(TASK4_CAPABILITIES) - 1)

    def test_missing_body_axis_allows_only_evidence_closed_candidates(self):
        scene = _scenes()[0]
        scene.pop("human_coordinate_frame")
        candidates, rejected = generate_task4_candidates(scene)
        by_category = {
            candidate["_capability"]: candidate for candidate in candidates
        }
        self.assertIn("distance_evolution", by_category)
        self.assertNotIn("passing_side_and_final_position", by_category)
        self.assertIn("passing", rejected)
        distance_group = {
            key: value for key, value in by_category["distance_evolution"].items()
            if not key.startswith("_")
        }
        self.assertEqual(
            validate_release({"groups": [distance_group]})["status"], "ok",
        )

    def test_source_capability_whitelist_prevents_unsupported_direction_claims(self):
        scene = _scenes()[0]
        scene["supported_task4_capabilities"] = ["distance_evolution"]
        candidates, rejected = generate_task4_candidates(scene)
        self.assertEqual(
            [candidate["_capability"] for candidate in candidates],
            ["distance_evolution"],
        )
        self.assertEqual(
            rejected["passing"], "source annotation capability is not declared",
        )

    def test_body_centric_dominant_mode_needs_only_the_anchor_face_frame(self):
        scene = _scene(
            "anchor_only_dominant",
            [[0.0, 0.0, 0.0] for _ in range(8)],
            [[1.0, 0.0, 2.0] for _ in range(8)],
        )
        scene["dominant_relation_mode"] = "body_centric_position"
        scene["face_forward_person_ids"] = ["A"]
        candidates, _ = generate_task4_candidates(scene)
        dominant = [
            row for row in candidates
            if row["_capability"] == "dominant_interaction_relation"
        ]
        self.assertEqual(len(dominant), 1)
        self.assertEqual(
            dominant[0]["qa"][0]["question_type"],
            "dominant_body_centric_position",
        )

    def test_language_realization_runs_only_after_deterministic_selection(self):
        class CountingRealizer:
            name = "counting-test-realizer"

            def __init__(self):
                self.calls = 0
                self.template = TemplateLanguageRealizer()

            def realize(self, request):
                self.calls += 1
                return self.template.realize(request)

        data, _ = generate_task4_scale_release(
            _scenes(), target_count=7, allow_partial=True,
        )
        realizer = CountingRealizer()
        self.assertEqual(realizer.calls, 0)
        realize_task4_release(data, realizer)
        self.assertEqual(realizer.calls, len(data["groups"]))
        self.assertEqual(validate_release(data)["status"], "ok")

    def test_short_public_window_is_rejected_before_candidate_generation(self):

        scene = _scenes()[0]
        scene["duration_sec"] = 10.0
        with self.assertRaisesRegex(AnnotationEvidenceError, "outside 14.5"):
            generate_task4_candidates(scene)


if __name__ == "__main__":
    unittest.main()
