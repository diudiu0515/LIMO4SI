import unittest
from limo4si.task4_dynamics import passing_side_and_final_position, reunion_relation_restoration

class Task4DynamicsTests(unittest.TestCase):
    def test_passing_requires_interior_approach_and_side_crossing(self):
        ds = [3.0, 2.0, 1.0, 0.5, 1.0, 2.0, 3.0, 3.5]
        rel = ["left_front"] * 4 + ["right_front"] * 4
        rows = [{"distance_m": d, "a_relative_to_b": r} for d, r in zip(ds, rel)]
        result = passing_side_and_final_position(rows)
        self.assertEqual(result["passing_side"], "left")
        self.assertEqual(result["final_relation"], "right_front")

    def test_reunion_compares_signed_start_and_end_relation(self):
        ds = [1.0, 1.2, 2.0, 3.0, 2.5, 1.5, 1.1, 1.0]
        rows = [{"distance_m": d, "b_relative_to_a": "left_front" if i < 4 else "right_front"} for i, d in enumerate(ds)]
        result = reunion_relation_restoration(rows)
        self.assertFalse(result["restored"])
        self.assertEqual(result["end_relation"], "right_front")
