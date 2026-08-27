"""Unit tests for the recursive chemical formula parser on nested polyanions."""

import unittest
from penziv_materials.core.formula_parser import parse_chemical_formula, compute_element_mass_fractions


class TestFormulaParser(unittest.TestCase):
    def test_simple_binary_formula(self):
        counts = parse_chemical_formula("MgS")
        self.assertEqual(counts["Mg"], 1.0)
        self.assertEqual(counts["S"], 1.0)

    def test_polyanion_nested_group(self):
        # Mg1.10Sc0.20Zr1.80(PS4)3 -> P should be 3.0, S should be 12.0
        counts = parse_chemical_formula("Mg1.10Sc0.20Zr1.80(PS4)3")
        self.assertAlmostEqual(counts["Mg"], 1.10)
        self.assertAlmostEqual(counts["Sc"], 0.20)
        self.assertAlmostEqual(counts["Zr"], 1.80)
        self.assertAlmostEqual(counts["P"], 3.0)
        self.assertAlmostEqual(counts["S"], 12.0)

    def test_multiple_polyanion_groups(self):
        # Na3Zr2(SiO4)2(PO4) -> Na: 3, Zr: 2, Si: 2, O: 8 + 4 = 12, P: 1
        counts = parse_chemical_formula("Na3Zr2(SiO4)2(PO4)")
        self.assertAlmostEqual(counts["Na"], 3.0)
        self.assertAlmostEqual(counts["Zr"], 2.0)
        self.assertAlmostEqual(counts["Si"], 2.0)
        self.assertAlmostEqual(counts["P"], 1.0)
        self.assertAlmostEqual(counts["O"], 12.0)

    def test_mass_fractions_sum_to_one(self):
        mass_fracs = compute_element_mass_fractions("Mg1.10Sc0.20Zr1.80(PS4)3")
        total = sum(mass_fracs.values())
        self.assertAlmostEqual(total, 1.0, places=5)
        self.assertGreater(mass_fracs["S"], 0.40)  # S is major mass contributor


if __name__ == "__main__":
    unittest.main()
