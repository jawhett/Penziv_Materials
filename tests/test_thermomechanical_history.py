"""Unit tests for the Thermomechanical History & Fatigue/Fracture Variation Engine."""

import unittest
import numpy as np
from penziv_materials.scale1_process.thermomechanical_history import (
    ThermomechanicalHistoryEngine,
    ThermomechanicalHistoryParameters,
    ProcessingRoute,
)


class TestThermomechanicalHistory(unittest.TestCase):
    def setUp(self):
        self.engine = ThermomechanicalHistoryEngine(shear_modulus_gpa=77.0, poisson_ratio=0.30)
        self.base_yield_mpa = 300.0
        self.base_youngs_gpa = 200.0

    def test_annealed_recrystallized_state(self):
        params = ThermomechanicalHistoryParameters(route=ProcessingRoute.ANNEALED_RECRYSTALLIZED)
        res = self.engine.predict_properties_from_history(self.base_yield_mpa, self.base_youngs_gpa, params)

        self.assertGreater(res.total_elongation_to_failure_percent, 40.0)
        self.assertGreater(res.fracture_toughness_k_ic_mpa_sqrt_m, 60.0)
        self.assertLess(res.dislocation_density_m2, 1e13)
        self.assertGreater(res.transition_fatigue_life_cycles_nt, 1000.0)

    def test_cold_worked_50pct_state(self):
        params = ThermomechanicalHistoryParameters(route=ProcessingRoute.COLD_WORKED_50PCT)
        res = self.engine.predict_properties_from_history(self.base_yield_mpa, self.base_youngs_gpa, params)

        # Cold work increases yield strength via Taylor forest hardening
        self.assertGreater(res.yield_strength_mpa, 450.0)
        self.assertGreater(res.dislocation_density_m2, 5e14)
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

        # As-printed has high cellular strength but reduced fatigue endurance due to surface roughness and residual stress
        self.assertGreater(res_print.yield_strength_mpa, 350.0)
        # HIP treatment relieves residual stress and closes pores, increasing ductility and fatigue endurance
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

    def test_continuous_isv_trajectory_cold_work_and_recovery(self):
        # Piecewise time-temperature-strain history: 10s cold deformation followed by 300s isothermal annealing
        t_arr = np.linspace(0.0, 310.0, 100)
        temp_arr = np.where(t_arr < 10.0, 298.15, 1173.15)
        strain_rate_arr = np.where(t_arr < 10.0, 0.05, 0.0)

        res = self.engine.integrate_continuous_isv_trajectory(
            time_series_s=t_arr,
            temperature_series_k=temp_arr,
            strain_rate_series_s_inv=strain_rate_arr,
            base_yield_strength_mpa=self.base_yield_mpa,
            base_youngs_modulus_gpa=self.base_youngs_gpa,
        )

        self.assertIn("final_isv", res)
        final_isv = res["final_isv"]
        self.assertGreater(final_isv.dislocation_density_m2, 1e11)
        self.assertGreater(len(res["yield_strength_trajectory_mpa"]), 50)
        self.assertGreater(res["final_yield_strength_mpa"], 250.0)

    def test_lsw_precipitate_coarsening_and_shearing_orowan_transition(self):
        # Under-aged state (r_p = 1.5 nm < r_crit = 4.0 nm): shearing dominated
        tau_shear = self.engine.compute_precipitate_strengthening(
            mean_radius_nm=1.5,
            volume_fraction=0.03,
            g_shear_mpa=77000.0,
            critical_radius_nm=4.0,
        )
        # Peak / Over-aged state (r_p = 12.0 nm > r_crit): Orowan looping dominated
        tau_orowan = self.engine.compute_precipitate_strengthening(
            mean_radius_nm=12.0,
            volume_fraction=0.03,
            g_shear_mpa=77000.0,
            critical_radius_nm=4.0,
        )

        self.assertGreater(tau_shear, 50.0)
        self.assertGreater(tau_orowan, 50.0)

        # LSW coarsening rate increases with temperature
        r_rate_low = self.engine.compute_lsw_precipitate_coarsening_rate(radius_nm=2.0, temperature_k=500.0)
        r_rate_high = self.engine.compute_lsw_precipitate_coarsening_rate(radius_nm=2.0, temperature_k=900.0)
        self.assertGreater(r_rate_high, r_rate_low)

    def test_master_sintering_path_integration(self):
        from penziv_materials.synthesis.retrosynthesis_planner import RetrosynthesisAssemblyPlanner
        planner = RetrosynthesisAssemblyPlanner()

        # Non-isothermal profile: ramp from 300K to 1100K in 1800s, dwell 3600s, cool in 1800s
        t_arr = np.linspace(0.0, 7200.0, 100)
        t_ramp_up = np.linspace(300.0, 1100.0, 25)
        t_dwell = np.full(50, 1100.0)
        t_cool = np.linspace(1100.0, 300.0, 25)
        temp_arr = np.concatenate([t_ramp_up, t_dwell, t_cool])

        msc = planner.integrate_master_sintering_path(
            time_series_s=t_arr,
            temperature_series_k=temp_arr,
            q_diff_j_mol=250000.0,
            green_density_pct=70.0,
            theoretical_density_pct=99.5,
        )

        self.assertGreater(msc["theta_msc_path_integral_s_k"], 0.0)
        self.assertGreater(msc["relative_density_percent"], 70.0)
        self.assertEqual(len(msc["density_trajectory_percent"]), len(t_arr))

    def test_anharmonic_quantum_elastic_softening(self):
        from penziv_materials.scale5_quantum.q_elec import QElecAgent
        q_agent = QElecAgent()
        c_base = np.eye(6) * 200.0

        # Anharmonic softening at elevated temperature with strain history and dislocation density
        c_softened = q_agent.evaluate_elastic_constants_temperature_dependent(
            c_base_gpa=c_base,
            temperature_k=1000.0,
            melting_point_k=1800.0,
            internal_strain_tensor=np.eye(3) * 0.02,
            dislocation_density_m2=1.0e14,
            thermal_expansion_coeff=1.5e-5,
        )

        self.assertLess(c_softened[0, 0], c_base[0, 0])
        self.assertGreater(c_softened[0, 0], 50.0)

    def test_dynamic_defect_kinetics_and_solute_drag(self):
        from penziv_materials.scale4_atomistic.atom_dyn import AtomDynAgent
        atom_agent = AtomDynAgent()

        # Check solute drag barrier shift during dynamic deformation
        shift = atom_agent.compute_solute_drag_barrier_shift(
            temperature_k=800.0,
            strain_rate_s_inv=1e-2,
            solute_concentration=0.05,
            dislocation_density_m2=1e13,
        )
        self.assertGreater(shift, 0.0)

        # Path-dependent vacancy supersaturation trajectory
        t_arr = np.linspace(0.0, 100.0, 50)
        temp_arr = np.linspace(1200.0, 300.0, 50)  # Quenching
        res = atom_agent.integrate_path_dependent_defect_kinetics(
            time_series_s=t_arr,
            temperature_series_k=temp_arr,
            strain_rate_series_s_inv=np.full(50, 1e-3),
        )
        self.assertIn("vacancy_concentration_trajectory", res)
        self.assertGreater(len(res["kinetic_flux_trajectory_s_inv"]), 10)

    def test_isv_coupled_stz_plasticity(self):
        from penziv_materials.scale3_mesoscale.calphad_grand_potential import CALPHADGrandPotentialPhaseFieldEngine
        pf_engine = CALPHADGrandPotentialPhaseFieldEngine()

        stz_res = pf_engine.compute_isv_coupled_stz_plastic_strain_rate(
            deviatoric_shear_stress_mpa=600.0,
            grain_size_um=15.0,
            dislocation_density_m2=5.0e13,
            precipitate_radius_nm=6.0,
            precipitate_volume_fraction=0.02,
        )

        self.assertGreater(stz_res["dynamic_characteristic_yield_stress_mpa"], 200.0)
        self.assertGreater(stz_res["plastic_shear_strain_rate_s_inv"], 0.0)


if __name__ == "__main__":
    unittest.main()
