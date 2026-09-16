import unittest

from limo4si.identity.annotation_ids import neutral_public_names, stable_person_ids
from limo4si.identity.global_assignment import AssignmentPolicy, assign_global_identities
from limo4si.identity.quality import validate_identity_result


class GlobalIdentityAssignmentTests(unittest.TestCase):
    def test_three_people_are_assigned_globally(self):
        metric = {
            "P1": {"points": [[0, 0], [1, 0], [2, 0], [3, 0]]},
            "P2": {"points": [[0, 0], [0, 1], [0, 3], [0, 6]]},
            "P3": {"points": [[0, 0], [1, 1], [1, 2], [2, 4]]},
        }
        visible = {
            "V1": {"points": [[10, 10], [11, 11], [11, 12], [12, 14]], "coverage": 1.0},
            "V2": {"points": [[20, 20], [20, 21], [20, 23], [20, 26]], "coverage": 1.0},
            "V3": {"points": [[30, 30], [31, 30], [32, 30], [33, 30]], "coverage": 1.0},
        }
        result = assign_global_identities(
            metric, visible, policy=AssignmentPolicy(min_margin=0.05),
        )
        self.assertEqual(result.status, "aligned")
        self.assertEqual(result.mapping, {"P1": "V3", "P2": "V2", "P3": "V1"})
        self.assertGreater(result.alternative_cost, result.best_cost)
        validate_identity_result(result.as_dict())

    def test_low_coverage_fails_closed(self):
        metric = {
            "P1": {"points": [[0, 0], [1, 0], [2, 0]]},
            "P2": {"points": [[0, 0], [0, 1], [0, 2]]},
        }
        visible = {
            "V1": {"points": [[0, 0], [1, 0], [2, 0]], "coverage": 0.5},
            "V2": {"points": [[0, 0], [0, 1], [0, 2]], "coverage": 1.0},
        }
        result = assign_global_identities(metric, visible)
        self.assertEqual(result.status, "unresolved")
        self.assertIn("coverage", result.reason)

    def test_ambiguous_assignment_does_not_guess(self):
        line = {"points": [[0, 0], [1, 0], [2, 0]]}
        metric = {"P1": line, "P2": line}
        visible = {
            "V1": {**line, "coverage": 1.0},
            "V2": {**line, "coverage": 1.0},
        }
        result = assign_global_identities(metric, visible)
        self.assertEqual(result.status, "unresolved")
        self.assertIsNone(result.mapping)
        with self.assertRaises(ValueError):
            validate_identity_result(result.as_dict())

    def test_internal_ids_and_public_names_are_separate(self):
        stable = stable_person_ids(["raw_b", "raw_a"])
        self.assertEqual(stable, {"raw_a": "P0001", "raw_b": "P0002"})
        names = neutral_public_names(stable.values())
        self.assertEqual(names["P0001"], "annotated person 1")


if __name__ == "__main__":
    unittest.main()
