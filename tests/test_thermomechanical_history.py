"""Unit tests for the Thermomechanical History & Fatigue/Fracture Variation Engine."""

import unittest
from penziv_materials.scale1_process.thermomechanical_history import (
    ThermomechanicalHistoryEngine,
    ThermomechanicalHistoryParameters,
    ProcessingRoute,
)


class TestThermomechanicalHistory(unittest.TestCase):
    def setUp(self):
        self.engine = ThermomechanicalHistoryEngine(shear_modulus_gpa=80.0, poisson_ratio=0.30)
        self.base_yield_mpa = 300.0
        self.base_youngs_gpa = 200.0

    def test_annealed_recrystallized_state(self):
        params = ThermomechanicalHistoryParameters(route=ProcessingRoute.ANNEALED_RECRYSTALLIZED)
        res = self.engine.predict_properties_from_history(self.base_yield_mpa, self.base_youngs_gpa, params)

        self.assertGreater(res.total_elongation_to_failure_percent, 40.0)
        self.assertGreater(res.fracture_toughness_k_ic_mpa_sqrt_m, 100.0)
        self.assertLess(res.dislocation_density_m2, 1e13)
        self.assertGreater(res.transition_fatigue_life_cycles_nt, 1000.0)

    def test_cold_worked_50pct_state(self):
        params = ThermomechanicalHistoryParameters(route=ProcessingRoute.COLD_WORKED_50PCT)
        res = self.engine.predict_properties_from_history(self.base_yield_mpa, self.base_youngs_gpa, params)

        # Cold work increases yield strength via Taylor forest hardening
        self.assertGreater(res.yield_strength_mpa, 450.0)
        self.assertGreater(res.dislocation_density_m2, 1e15)
        # Cold work reduces elongation and fracture toughness
        self.assertLess(res.total_elongation_to_failure_percent, 20.0)
        self.assertLess(res.strain_hardening_exponent_n, 0.12)

    def test_peak_aged_t6_precipitation(self):
        params = ThermomechanicalHistoryParameters(route=ProcessingRoute.SOLUTION_TREATED_PEAK_AGED_T6)
        res = self.engine.predict_properties_from_history(self.base_yield_mpa, self.base_youngs_gpa, params)

        self.assertGreater(res.precipitate_volume_fraction, 0.03)
        self.assertGreater(res.yield_strength_mpa, 500.0)
        self.assertGreater(res.fatigue_endurance_limit_sigma_e_mpa, 200.0)

    def test_additive_lpbf_as_printed_vs_hip_treated(self):
        as_printed_params = ThermomechanicalHistoryParameters(route=ProcessingRoute.ADDITIVE_LPBF_AS_PRINTED)
        hip_params = ThermomechanicalHistoryParameters(route=ProcessingRoute.ADDITIVE_LPBF_HIP_AGED)

        res_print = self.engine.predict_properties_from_history(self.base_yield_mpa, self.base_youngs_gpa, as_printed_params)
        res_hip = self.engine.predict_properties_from_history(self.base_yield_mpa, self.base_youngs_gpa, hip_params)

        # As-printed has high fine-cell strength but reduced fatigue endurance due to residual stress
        self.assertGreater(res_print.yield_strength_mpa, 350.0)
        # HIP treatment relieves residual stress, increasing ductility and fatigue endurance
        self.assertGreater(res_hip.fatigue_endurance_limit_sigma_e_mpa, res_print.fatigue_endurance_limit_sigma_e_mpa)
        self.assertGreater(res_hip.fracture_toughness_k_ic_mpa_sqrt_m, res_print.fracture_toughness_k_ic_mpa_sqrt_m)
        self.assertGreater(res_hip.total_elongation_to_failure_percent, res_print.total_elongation_to_failure_percent)

    def test_fatigue_and_paris_law_exponents_validity(self):
        params = ThermomechanicalHistoryParameters(route=ProcessingRoute.ANNEALED_RECRYSTALLIZED)
        res = self.engine.predict_properties_from_history(self.base_yield_mpa, self.base_youngs_gpa, params)

        # Basquin exponent b should be in [-0.16, -0.06]
        self.assertGreaterEqual(res.basquin_exponent_b, -0.16)
        self.assertLessEqual(res.basquin_exponent_b, -0.06)

        # Coffin-Manson exponent c should be in [-0.75, -0.45]
        self.assertGreaterEqual(res.coffin_manson_exponent_c, -0.75)
        self.assertLessEqual(res.coffin_manson_exponent_c, -0.45)

        # Paris law exponent m should be in [2.5, 4.5]
        self.assertGreaterEqual(res.paris_law_m, 2.5)
        self.assertLessEqual(res.paris_law_m, 4.5)
        self.assertGreater(res.fatigue_threshold_delta_k_th_mpa_sqrt_m, 2.0)


if __name__ == "__main__":
    unittest.main()
