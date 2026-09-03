import importlib.util
import unittest
from pathlib import Path

from limo4si.multihuman import multi_person_metric_timeline


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_panoptic_multihuman.py"
SPEC = importlib.util.spec_from_file_location("convert_panoptic_multihuman", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PanopticConverterTests(unittest.TestCase):
    def test_converts_centimetres_and_builds_nose_disambiguated_forward(self):
        joints = [0.0] * (19 * 4)
        def put(index, xyz, confidence=1.0):
            joints[index*4:index*4+4] = [*xyz, confidence]
        put(0, [0, 100, 0])
        put(1, [0, 100, 10])
        put(2, [0, 0, 0])
        put(3, [-50, 100, 0])
        put(9, [50, 100, 0])
        person = MODULE.convert_body({"id": 7, "joints19": joints}, 0.1)
        self.assertEqual(person["id"], "7")
        self.assertEqual(person["pelvis"], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(person["forward"][2], 1.0)

    def test_rejects_body_without_required_confident_joints(self):
        self.assertIsNone(MODULE.convert_body({"id": 1, "joints19": [0.0] * 76}, 0.1))

    def test_maps_stable_source_ids_to_engine_ids(self):
        self.assertEqual(MODULE.public_identity_map({"12", "2", "7"}), {"2": "A", "7": "B", "12": "C"})

    def test_all_pair_metric_timeline_requires_and_uses_three_people(self):
        people = [
            {"id": "A", "pelvis": [0.0, 0.0, 0.0]},
            {"id": "B", "pelvis": [1.0, 0.0, 0.0]},
            {"id": "C", "pelvis": [3.0, 0.0, 0.0]},
        ]
        scene = {
            "metric_person_ids": ["A", "B", "C"],
            "frames": [{"t": 0.0, "frame_id": 0, "people": people}],
        }
        timeline = multi_person_metric_timeline(scene)
        self.assertEqual(timeline["status"], "ok")
        self.assertEqual(timeline["person_count"], 3)
        self.assertEqual(len(timeline["states"][0]["pair_distances_m"]), 3)
        self.assertEqual(timeline["states"][0]["closest_pair"], "A–B")

        scene["metric_person_ids"] = ["A", "B"]
        self.assertEqual(multi_person_metric_timeline(scene)["status"], "missing_evidence")


if __name__ == "__main__":
    unittest.main()
