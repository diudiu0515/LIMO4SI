import unittest
from copy import deepcopy
import numpy as np

from limo4si.multihuman_release import (
    render_person_descriptor,
    recompute_hoim3_pair_timelines,
    resolve_person_aliases,
)
from limo4si.visual_tracking import associate


class MultihumanReleaseTests(unittest.TestCase):
    def test_reviewed_attributes_bind_metric_and_visible_ids(self):
        config = {
            "schema_version": "limo4si.person_attributes.v1",
            "entries": {
                "scene": {
                    "source": "manual_visual_review",
                    "reviewed_attributes": ["gender_term", "upper_body", "lower_body"],
                    "audit_status": "verified_from_original_and_localized_six_frame_sheets",
                    "people": {
                        "A": {"gender_term": "man", "upper_body": "navy shirt", "lower_body": "gray trousers"},
                        "B": {"gender_term": "woman", "upper_body": "printed shirt", "lower_body": "blue jeans"},
                    },
                }
            },
        }
        audit = {"metric_identity_alignment": {"mapping": {"A": "V2", "B": "V1"}}}
        aliases, status = resolve_person_aliases(
            group_name="scene", visual_person_audit=audit,
            attribute_config=config, config_path="attributes.json",
        )
        self.assertEqual(aliases["V2"], aliases["A"])
        self.assertEqual(aliases["V1"], aliases["B"])
        self.assertEqual(status["status"], "complete")

    def test_descriptor_rejects_unstructured_gender(self):
        with self.assertRaises(ValueError):
            render_person_descriptor({"gender_term": "unknown", "upper_body": "shirt"})

    def test_frame_override_is_applied_before_relation_computation(self):
        scene = {
            "scene_id": "hoi_m3_demo_win01",
            "frames": [{
                "t": 0.0,
                "people": [
                    {"id": "A", "pelvis": [0, 0, 0], "forward": [0, 0, 1]},
                    {"id": "B", "pelvis": [1, 0, 1], "forward": [0, 0, -1]},
                ],
            }],
        }
        data = {"groups": [{
            "name": scene["scene_id"],
            "qa": [{"result_json": {"pair_timeline": {"states": [{}]}}}],
        }]}
        overrides = {
            "hoi_m3_demo": {
                "source": "test",
                "right_sign": -1,
                "expected_relations": {
                    scene["scene_id"]: {"start": "left_front", "end": "left_front"}
                },
            }
        }
        recompute_hoim3_pair_timelines(
            data, {"scenes": [scene]}, overrides, compact_timeline=deepcopy,
        )
        state = data["groups"][0]["qa"][0]["result_json"]["pair_timeline"]["states"][0]
        self.assertEqual(state["b_relative_to_a"], "left_front")


class VisualTrackingTests(unittest.TestCase):
    @staticmethod
    def signature(value_bin: int) -> np.ndarray:
        hist = np.zeros(1536, dtype=np.float32)
        for offset in (0, 768):
            hist[offset + value_bin] = 1.0
        return hist

    def test_crossing_tracks_keep_appearance_identity(self):
        dark, light = self.signature(0), self.signature(7)
        detected = []
        for t, dark_x, light_x in [(0.0, 10, 90), (0.5, 35, 65), (1.0, 65, 35), (1.5, 90, 10)]:
            detected.append({
                "t": t,
                "detections": [
                    {"box": np.array([dark_x, 0, dark_x + 10, 20], np.float32), "score": 0.9, "hist": dark},
                    {"box": np.array([light_x, 0, light_x + 10, 20], np.float32), "score": 0.9, "hist": light},
                ],
            })
        tracks = associate(detected, 120, 80)
        self.assertEqual(len(tracks), 2)
        self.assertGreater(float(tracks[0]["obs"][-1]["box"][0]), 80)
        self.assertLess(float(tracks[1]["obs"][-1]["box"][0]), 20)


if __name__ == "__main__":
    unittest.main()
