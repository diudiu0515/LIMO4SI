import json
import tempfile
import unittest
from pathlib import Path

from limo4si.adt_task4 import (
    AdtTask4Policy, _quat_wxyz_to_matrix, _skeleton, _wearer,
    _wearer_skeleton_names,
)
from limo4si.multihuman import line_blocked


class AdtTask4Tests(unittest.TestCase):
    def test_cpf_axes_define_face_forward_and_explicit_human_right(self):
        person = _wearer(
            {
                "translation": [1.0, 1.7, 2.0],
                "quaternion": [0.0, 0.0, 0.0, 1.0],
                "gravity": [0.0, -9.81, 0.0],
                "quality": 1.0,
            },
            AdtTask4Policy(),
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        self.assertEqual(person["forward"], [0.0, 0.0, 1.0])
        self.assertEqual(person["right"], [-1.0, 0.0, 0.0])
        self.assertAlmostEqual(person["pelvis"][1], 1.05)

    def test_official_labeled_shoulders_determine_torso_forward(self):
        joints = [[0.0, 0.0, 0.0] for _ in range(51)]
        joints[0] = [0.0, 0.0, 0.0]
        joints[2] = [0.0, 1.0, 0.0]
        joints[4] = [0.0, 1.7, 0.0]
        joints[6] = [-0.5, 1.0, 0.0]
        joints[25] = [0.5, 1.0, 0.0]
        person = _skeleton({"joints": joints}, "B", "Skeleton_T")
        self.assertEqual(person["forward"], [0.0, 0.0, 1.0])
        self.assertEqual(person["right"], [1.0, 0.0, 0.0])
        self.assertEqual(person["orientation_scope"], "torso-forward only")

    def test_oriented_box_blocks_only_an_intersecting_head_segment(self):
        blocker = {
            "id": "wall",
            "local_min": [-0.1, -1.0, -1.0],
            "local_max": [0.1, 1.0, 1.0],
            "translation": [1.0, 0.0, 0.0],
            "rotation_world_from_local": _quat_wxyz_to_matrix([1.0, 0.0, 0.0, 0.0]),
        }
        hit = line_blocked([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [blocker])
        miss = line_blocked([0.0, 2.0, 0.0], [2.0, 2.0, 0.0], [blocker])
        self.assertTrue(hit["blocked"])
        self.assertEqual(hit["blocker"]["geometry"], "oriented_box")
        self.assertFalse(miss["blocked"])

    def test_oriented_box_containing_an_endpoint_is_not_valid_occluder_evidence(self):
        blocker = {
            "id": "bad_enclosing_box",
            "local_min": [-0.2, -0.2, -0.2],
            "local_max": [0.2, 0.2, 0.2],
            "translation": [0.0, 0.0, 0.0],
            "rotation_world_from_local": _quat_wxyz_to_matrix([1.0, 0.0, 0.0, 0.0]),
        }
        result = line_blocked([0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [blocker])
        self.assertFalse(result["blocked"])

    def test_multiskeleton_association_excludes_the_current_wearer_track(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "skeleton_aria_association.json").write_text(json.dumps({
                "SkeletonMetadata": [
                    {"AssociatedDeviceSerial": "M1292", "SkeletonName": "Skeleton_T"},
                    {"AssociatedDeviceSerial": "71292", "SkeletonName": "Skeleton_C"},
                ],
            }))
            self.assertEqual(
                _wearer_skeleton_names(root, "M1292", require_association=True),
                {"Skeleton_T"},
            )


if __name__ == "__main__":
    unittest.main()
