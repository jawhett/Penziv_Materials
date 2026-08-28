"""Unit tests for the Advanced Subsystem Analytical & Experimental Validation Suite."""

import unittest
from penziv_materials.benchmarks.advanced_validation_benchmark import (
    AdvancedPhysicalValidationSuite,
    AdvancedSubsystemValidationReport,
)


class TestAdvancedPhysicalValidation(unittest.TestCase):
    def setUp(self):
        self.suite = AdvancedPhysicalValidationSuite()

    def test_debye_hueckel_space_charge_validation(self):
        rep = self.suite.validate_debye_hueckel_and_pnp_space_charge()
        self.assertEqual(rep.validation_status, "PASSED")
        self.assertLess(rep.absolute_percentage_error, 5.0)
        self.assertAlmostEqual(rep.literature_ground_truth_value, 0.304, places=2)

    def test_taylor_polycrystal_bound_validation(self):
        rep = self.suite.validate_taylor_polycrystal_plasticity_bounds()
        self.assertEqual(rep.validation_status, "PASSED")
        self.assertLess(rep.absolute_percentage_error, 1.0)
        self.assertAlmostEqual(rep.literature_ground_truth_value, 3.067, places=2)

    def test_vitreous_silica_glass_validation(self):
        rep = self.suite.validate_vitreous_silica_glass_network_topology()
        self.assertEqual(rep.validation_status, "PASSED")
        self.assertLess(rep.absolute_percentage_error, 2.0)
        self.assertAlmostEqual(rep.literature_ground_truth_value, 1.610, places=2)

    def test_griffith_dupre_fracture_work_validation(self):
        rep = self.suite.validate_griffith_dupre_cohesive_fracture_work()
        self.assertEqual(rep.validation_status, "PASSED")
        self.assertLess(rep.absolute_percentage_error, 0.1)
        self.assertEqual(rep.literature_ground_truth_value, 2.0)

    def test_usgs_supply_chain_risk_validation(self):
        rep = self.suite.validate_usgs_supply_chain_hhi_risk()
        self.assertEqual(rep.validation_status, "PASSED")
        self.assertGreater(rep.predicted_metric_value, 6000.0)

    def test_continuous_cvt_map_elites_validation(self):
        rep = self.suite.validate_continuous_cvt_map_elites_pareto_coverage()
        self.assertEqual(rep.validation_status, "PASSED")
        self.assertGreaterEqual(rep.predicted_metric_value, 10.0)

    def test_run_all_advanced_validations_suite(self):
        res = self.suite.run_all_advanced_validations()
        self.assertEqual(res["total_subsystems_validated"], 6)
        self.assertTrue(res["all_subsystems_passed"])
        self.assertLess(res["mean_absolute_percentage_error"], 25.0)


if __name__ == "__main__":
    unittest.main()
