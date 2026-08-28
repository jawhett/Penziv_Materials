"""Unit tests for the Zero-Parameter Chemical Formula Benchmark Suite across 10 Material Classes."""

import unittest
from penziv_materials.benchmarks.formula_prediction_benchmark import FormulaPredictionBenchmarkSuite


class TestFormulaBenchmark(unittest.TestCase):
    def setUp(self):
        self.suite = FormulaPredictionBenchmarkSuite()

    def test_single_element_copper(self):
        rep = self.suite.predict_material_from_formula("Cu", temperature_k=300.0)
        self.assertEqual(rep.formula, "Cu")
        self.assertEqual(rep.predicted_space_group, "Fm-3m")
        self.assertTrue(rep.born_mechanical_stability)
        self.assertEqual(rep.band_gap_ev, 0.0)
        self.assertGreater(rep.electrical_conductivity_s_m, 1e7)
        self.assertGreater(rep.thermal_conductivity_w_m_k, 300.0)
        self.assertGreater(rep.theoretical_density_g_cm3, 7.0)

    def test_single_element_aluminum(self):
        rep = self.suite.predict_material_from_formula("Al", temperature_k=300.0)
        self.assertEqual(rep.formula, "Al")
        self.assertEqual(rep.predicted_space_group, "Fm-3m")
        self.assertTrue(rep.born_mechanical_stability)
        self.assertEqual(rep.band_gap_ev, 0.0)
        self.assertGreater(rep.youngs_modulus_gpa, 40.0)
        self.assertLess(rep.theoretical_density_g_cm3, 4.0)

    def test_ceramic_calcium_oxide(self):
        rep = self.suite.predict_material_from_formula("CaO", temperature_k=300.0)
        self.assertEqual(rep.formula, "CaO")
        self.assertEqual(rep.predicted_space_group, "Fm-3m")
        self.assertTrue(rep.born_mechanical_stability)
        self.assertGreater(rep.band_gap_ev, 5.0)
        self.assertGreater(rep.static_dielectric_constant, 5.0)
        self.assertIn("Ca", rep.parsed_composition)
        self.assertIn("O", rep.parsed_composition)

    def test_multicomponent_austenitic_stainless_steel(self):
        rep = self.suite.predict_material_from_formula("Fe0.70Cr0.18Ni0.10Mo0.02", temperature_k=300.0)
        self.assertEqual(rep.formula, "Fe0.70Cr0.18Ni0.10Mo0.02")
        self.assertEqual(rep.predicted_space_group, "Fm-3m")
        self.assertTrue(rep.born_mechanical_stability)
        self.assertEqual(rep.band_gap_ev, 0.0)
        self.assertGreater(rep.yield_strength_mpa, 200.0)

    def test_layered_max_phase_ti3sic2(self):
        rep = self.suite.predict_material_from_formula("Ti3SiC2", temperature_k=300.0)
        self.assertEqual(rep.formula, "Ti3SiC2")
        self.assertEqual(rep.predicted_space_group, "P6_3/mmc")
        self.assertTrue(rep.born_mechanical_stability)
        self.assertEqual(rep.band_gap_ev, 0.0)
        self.assertGreater(rep.electrical_conductivity_s_m, 1e6)
        self.assertGreater(rep.youngs_modulus_gpa, 150.0)

    def test_refractory_high_entropy_alloy_senkov(self):
        rep = self.suite.predict_material_from_formula("Nb0.25Mo0.25Ta0.25W0.25", temperature_k=300.0)
        self.assertEqual(rep.formula, "Nb0.25Mo0.25Ta0.25W0.25")
        self.assertEqual(rep.predicted_space_group, "Im-3m")
        self.assertTrue(rep.born_mechanical_stability)
        self.assertEqual(rep.band_gap_ev, 0.0)
        self.assertGreater(rep.theoretical_density_g_cm3, 10.0)

    def test_superionic_solid_electrolyte(self):
        rep = self.suite.predict_material_from_formula("Mg1.10Sc0.20Zr1.80(PS4)3", temperature_k=300.0)
        self.assertEqual(rep.predicted_space_group, "R-3c")
        self.assertTrue(rep.born_mechanical_stability)
        self.assertGreater(rep.band_gap_ev, 3.0)
        self.assertGreater(rep.ionic_conductivity_ms_cm, 1.0)
        self.assertIn("P", rep.parsed_composition)

    def test_iii_v_semiconductor_gaas(self):
        rep = self.suite.predict_material_from_formula("GaAs", temperature_k=300.0)
        self.assertEqual(rep.formula, "GaAs")
        self.assertEqual(rep.predicted_space_group, "F-43m")
        self.assertTrue(rep.born_mechanical_stability)
        self.assertAlmostEqual(rep.band_gap_ev, 1.424, places=2)
        self.assertGreater(rep.carrier_mobility_cm2_v_s, 5000.0)
        self.assertGreater(rep.theoretical_density_g_cm3, 4.5)

    def test_ii_vi_photovoltaic_cdte(self):
        rep = self.suite.predict_material_from_formula("CdTe", temperature_k=300.0)
        self.assertEqual(rep.formula, "CdTe")
        self.assertEqual(rep.predicted_space_group, "F-43m")
        self.assertTrue(rep.born_mechanical_stability)
        self.assertAlmostEqual(rep.band_gap_ev, 1.495, places=2)
        self.assertGreater(rep.carrier_mobility_cm2_v_s, 500.0)
        self.assertGreater(rep.theoretical_density_g_cm3, 5.0)

    def test_topological_thermoelectric_bi2te3(self):
        rep = self.suite.predict_material_from_formula("Bi2Te3", temperature_k=300.0)
        self.assertEqual(rep.formula, "Bi2Te3")
        self.assertEqual(rep.predicted_space_group, "R-3m")
        self.assertTrue(rep.born_mechanical_stability)
        self.assertLess(rep.band_gap_ev, 0.5)
        self.assertGreater(rep.thermoelectric_figure_of_merit_zt, 1.0)
        self.assertLess(rep.seebeck_coefficient_uv_k, -100.0)
        self.assertGreater(rep.theoretical_density_g_cm3, 7.0)

    def test_full_ten_material_chemical_benchmark(self):
        res = self.suite.run_full_chemical_benchmark(temperature_k=300.0)
        self.assertEqual(res["total_materials_benchmarked"], 10)
        self.assertTrue(res["all_born_stable"])
        self.assertEqual(len(res["reports"]), 10)


if __name__ == "__main__":
    unittest.main()
