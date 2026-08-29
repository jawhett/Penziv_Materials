"""Unit tests for the Advanced Physics Fundamentals: Matthiessen Transport, GB Segregation, Residual Stress, and Wagner Oxidation."""

import unittest
import numpy as np

from penziv_materials.physics.matthiessen_transport import MatthiessenTransportEngine
from penziv_materials.scale4_atomistic.gb_segregation import GrainBoundarySegregationEngine
from penziv_materials.scale1_process.thermal_residual_stress import ThermalResidualStressEngine
from penziv_materials.physics.wagner_oxidation import WagnerOxidationEngine


class TestAdvancedPhysicsFundamentals(unittest.TestCase):
    def test_matthiessen_electronic_relaxation_rates(self):
        engine = MatthiessenTransportEngine(temperature_k=300.0)
        res = engine.compute_electronic_relaxation_rates(
            effective_mass_ratio=1.0,
            deformation_potential_ev=8.5,
            static_dielectric_constant=12.0,
            high_freq_dielectric_constant=10.0,
            longitudinal_sound_velocity_m_s=5000.0,
            density_kg_m3=7800.0,
            ionized_impurity_density_m3=1.0e22,
            dislocation_density_m2=1.0e12,
            grain_size_um=30.0,
        )

        self.assertGreater(res["rate_acoustic_phonon_s_inv"], 0.0)
        self.assertGreater(res["rate_total_s_inv"], 1e10)
        self.assertGreater(res["relaxation_time_tau_s"], 1e-16)
        self.assertGreater(res["carrier_mobility_cm2_v_s"], 10.0)

    def test_matthiessen_multichannel_thermal_conductivity(self):
        engine = MatthiessenTransportEngine(temperature_k=300.0)
        # Copper-like metallic material
        res_metal = engine.compute_coupled_multichannel_thermal_conductivity(
            average_atomic_mass_amu=63.55,
            debye_temperature_k=343.0,
            unit_cell_volume_ang3=47.2,
            sound_velocity_m_s=4700.0,
            carrier_concentration_m3=8.5e28,
            carrier_mobility_cm2_v_s=43.5,
            dislocation_density_m2=1.0e11,
            grain_size_um=45.0,
        )

        self.assertGreater(res_metal["total_thermal_conductivity_w_m_k"], 100.0)
        self.assertGreater(res_metal["electronic_thermal_conductivity_w_m_k"], 50.0)
        self.assertGreater(res_metal["lattice_thermal_conductivity_w_m_k"], 5.0)

        # High-entropy alloy with intense point-defect scattering
        res_hea = engine.compute_coupled_multichannel_thermal_conductivity(
            average_atomic_mass_amu=90.0,
            debye_temperature_k=380.0,
            unit_cell_volume_ang3=50.0,
            sound_velocity_m_s=4000.0,
            carrier_concentration_m3=5.0e28,
            carrier_mobility_cm2_v_s=8.0,
            solute_fraction=0.25,
            solute_mass_difference_ratio=0.5,
            dislocation_density_m2=5.0e13,
            grain_size_um=10.0,
        )
        self.assertLess(res_hea["total_thermal_conductivity_w_m_k"], res_metal["total_thermal_conductivity_w_m_k"])

    def test_grain_boundary_segregation_and_embrittlement(self):
        engine = GrainBoundarySegregationEngine(temperature_k=800.0)
        
        # Test elastic misfit segregation enthalpy
        delta_h = engine.compute_elastic_strain_segregation_enthalpy(
            matrix_shear_modulus_gpa=80.0,
            solute_bulk_modulus_gpa=160.0,
            matrix_covalent_radius_ang=1.24,
            solute_covalent_radius_ang=1.70,
        )
        self.assertLess(delta_h, -10000.0)

        # Multi-component segregation: steel with P, S (embrittlers) and Mo (cohesion enhancer)
        bulk_comp = {"Fe": 0.97, "P": 0.005, "S": 0.002, "Mo": 0.015, "Cr": 0.008}
        gb_comp = engine.solve_multicomponent_mclean_segregation(
            bulk_concentrations=bulk_comp,
            matrix_element="Fe",
        )
        # S and P should be significantly enriched at GB compared to bulk
        self.assertGreater(gb_comp["P"], bulk_comp["P"])
        self.assertGreater(gb_comp["S"], bulk_comp["S"])

        # Interfacial embrittlement evaluation
        rice_wang = engine.evaluate_rice_wang_interfacial_embrittlement(
            gb_solute_concentrations=gb_comp,
            clean_gb_surface_energy_j_m2=0.85,
            clean_free_surface_energy_j_m2=2.20,
        )
        self.assertIn("interfacial_cohesion_ratio", rice_wang)
        self.assertGreater(rice_wang["effective_work_of_adhesion_j_m2"], 1.0)

    def test_thermal_residual_stress_and_dislocation_cells(self):
        engine = ThermalResidualStressEngine(
            youngs_modulus_gpa=200.0,
            poisson_ratio=0.30,
            thermal_expansion_coeff_ppm_k=16.0,
        )
        res = engine.compute_1d_through_thickness_residual_stress(
            thickness_um=3000.0,
            surface_temp_k=300.0,
            center_temp_k=1100.0,
            yield_strength_mpa=500.0,
        )

        self.assertEqual(len(res.depth_profile_z_um), 50)
        self.assertEqual(len(res.stress_profile_sigma_xx_mpa), 50)
        self.assertGreater(res.cell_wall_dislocation_density_m2, res.cell_interior_dislocation_density_m2)
        self.assertGreater(res.kinematic_back_stress_mpa, 0.0)
        self.assertLess(res.bauschinger_effect_ratio, 1.0)

    def test_wagner_oxidation_kinetics_and_fatigue(self):
        engine = WagnerOxidationEngine(temperature_k=1100.0)
        ox_res = engine.compute_parabolic_oxidation_kinetics(
            exposure_time_hours=500.0,
            pre_exponential_kp0_m2_s=1.5e-4,
            activation_energy_q_j_mol=250000.0,
            base_fatigue_endurance_mpa=450.0,
        )

        self.assertGreater(ox_res["oxide_scale_thickness_um"], 0.01)
        self.assertGreater(ox_res["subsurface_solute_depletion_depth_um"], 0.01)
        self.assertGreaterEqual(ox_res["oxide_notch_stress_concentration_factor_kt"], 1.0)
        self.assertLess(ox_res["environmentally_degraded_fatigue_endurance_mpa"], 450.0)


if __name__ == "__main__":
    unittest.main()
