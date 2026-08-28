"""Unit tests for the Zero-Parameter Chemical Formula Benchmark Suite."""

import unittest
from penziv_materials.benchmarks.formula_prediction_benchmark import FormulaPredictionBenchmarkSuite


class TestFormulaBenchmark(unittest.TestCase):
    def setUp(self):
        self.suite = FormulaPredictionBenchmarkSuite()

    def test_single_element_copper_prediction(self):
        rep = self.suite.predict_material_from_formula("Cu", temperature_k=300.0)
        self.assertEqual(rep.formula, "Cu")
        self.assertEqual(rep.predicted_space_group, "Fm-3m")
        self.assertTrue(rep.born_mechanical_stability)
        self.assertGreater(rep.youngs_modulus_gpa, 50.0)
        self.assertGreater(rep.theoretical_density_g_cm3, 7.0)

    def test_single_element_aluminum_prediction(self):
        rep = self.suite.predict_material_from_formula("Al", temperature_k=300.0)
        self.assertEqual(rep.formula, "Al")
        self.assertEqual(rep.predicted_space_group, "Fm-3m")
        self.assertTrue(rep.born_mechanical_stability)
        self.assertGreater(rep.youngs_modulus_gpa, 40.0)
        self.assertLess(rep.theoretical_density_g_cm3, 4.0)

    def test_ceramic_calcium_oxide_prediction(self):
        rep = self.suite.predict_material_from_formula("CaO", temperature_k=300.0)
        self.assertEqual(rep.formula, "CaO")
        self.assertEqual(rep.predicted_space_group, "Fm-3m")
        self.assertTrue(rep.born_mechanical_stability)
        self.assertIn("Ca", rep.parsed_composition)
        self.assertIn("O", rep.parsed_composition)

    def test_multicomponent_austenitic_alloy_prediction(self):
        rep = self.suite.predict_material_from_formula("Fe0.70Cr0.18Ni0.10Mo0.02", temperature_k=300.0)
        self.assertEqual(rep.formula, "Fe0.70Cr0.18Ni0.10Mo0.02")
        self.assertEqual(rep.predicted_space_group, "Fm-3m")
        self.assertTrue(rep.born_mechanical_stability)
        self.assertGreater(rep.yield_strength_mpa, 200.0)
        self.assertEqual(rep.handshake_receipts_passed, rep.total_handshake_receipts)

    def test_full_chemical_benchmark_suite(self):
        res = self.suite.run_full_chemical_benchmark(
            benchmark_formulas=["Cu", "Al", "CaO", "Fe0.70Cr0.18Ni0.10Mo0.02"],
            temperature_k=300.0,
        )
        self.assertEqual(res["total_materials_benchmarked"], 4)
        self.assertTrue(res["all_born_stable"])
        self.assertEqual(len(res["reports"]), 4)


if __name__ == "__main__":
    unittest.main()
