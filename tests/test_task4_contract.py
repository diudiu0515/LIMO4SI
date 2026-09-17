import unittest
from limo4si.task4_contract import TASK4_CAPABILITIES, validate_compound_option_parts

class Task4ContractTests(unittest.TestCase):
    def test_contract_has_exactly_seven_requested_capabilities(self):
        self.assertEqual(len(TASK4_CAPABILITIES), 7)
    def test_compound_grid_requires_both_parts(self):
        options = [{"label": label} for label in "ABCD"]
        balanced = {"A": ["near", "left"], "B": ["near", "right"], "C": ["far", "left"], "D": ["far", "right"]}
        self.assertEqual(validate_compound_option_parts(options, "A", balanced), [])
        leaking = {"A": ["near", "left"], "B": ["far", "right"], "C": ["far", "left"], "D": ["far", "center"]}
        self.assertTrue(validate_compound_option_parts(options, "A", leaking))
