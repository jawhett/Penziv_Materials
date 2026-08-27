"""Tests for Born Mechanical Stability criteria."""

import unittest
import numpy as np
from penziv_materials.validation.born_stability import BornStabilityValidator
from penziv_materials.core.models import ValidationStatus


class TestBornStability(unittest.TestCase):
    def test_born_stability_stable_cubic(self):
        c11, c12, c44 = 260.0, 160.0, 110.0
        stable, details = BornStabilityValidator.validate_cubic(c11, c12, c44)
        self.assertTrue(stable)
        self.assertTrue(details["conditions_met"]["shear_tetragonal"])
        self.assertTrue(details["conditions_met"]["bulk_stability"])
        self.assertTrue(details["conditions_met"]["shear_trigonal"])
        self.assertGreater(details["lambda_min"], 0.0)

    def test_born_stability_unstable_shear(self):
        c11, c12, c44 = 150.0, 160.0, 110.0
        stable, details = BornStabilityValidator.validate_cubic(c11, c12, c44)
        self.assertFalse(stable)
        self.assertFalse(details["conditions_met"]["shear_tetragonal"])
        self.assertLess(details["lambda_min"], 0.0)

    def test_born_stability_full_tensor(self):
        C = np.eye(6) * 100.0
        receipt = BornStabilityValidator.validate(C)
        self.assertEqual(receipt.status, ValidationStatus.PASSED)
        self.assertEqual(receipt.metric_value, 100.0)


if __name__ == "__main__":
    unittest.main()
