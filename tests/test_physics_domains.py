"""Unit tests for Semiconductor Electronics, Thermal Transport, and Generative Crystallography."""

import unittest
import numpy as np

from penziv_materials.physics.semiconductor_electronics import SemiconductorElectronicEngine
from penziv_materials.physics.thermal_extreme_transport import ThermalExtremeTransportEngine
from penziv_materials.generative.crystal_generator import GenerativeCrystalSynthesizer
from penziv_materials.scale2_continuum.cont_micro import ContMicroAgent


class TestPhysicsDomains(unittest.TestCase):
    def setUp(self):
        self.semi = SemiconductorElectronicEngine()
        self.thermal = ThermalExtremeTransportEngine()
        self.synth = GenerativeCrystalSynthesizer(target_carrier_cation="Mg")
        self.cont = ContMicroAgent()

    def test_effective_mass_and_mobility(self):
        curvature = np.diag([12.5, 12.5, 15.0])
        m_tensor, m_scalar = self.semi.compute_effective_mass_tensor(curvature)
        self.assertEqual(m_tensor.shape, (3, 3))
        self.assertGreater(m_scalar, 0.0)

        mob_res = self.semi.compute_carrier_mobility(m_scalar, deformation_potential_ev=6.5)
        self.assertIn("electron_mobility_cm2_v_s", mob_res)
        self.assertGreater(mob_res["electron_mobility_cm2_v_s"], 1.0)

    def test_dielectric_breakdown(self):
        break_res = self.semi.compute_dielectric_tensor_and_breakdown_field(band_gap_ev=3.8)
        self.assertTrue(break_res["is_ultra_wide_bandgap"])
        self.assertGreater(break_res["dielectric_breakdown_field_mv_cm"], 5.0)

    def test_lattice_thermal_conductivity(self):
        k_res = self.thermal.compute_lattice_thermal_conductivity_slack(
            average_atomic_mass_amu=35.0,
            debye_temperature_k=650.0,
            volume_per_atom_ang3=15.0,
        )
        self.assertIn("lattice_thermal_conductivity_w_m_k", k_res)
        self.assertGreater(k_res["lattice_thermal_conductivity_w_m_k"], 10.0)

    def test_thermal_shock_and_radiation(self):
        shock_res = self.thermal.compute_thermal_shock_resistance_parameter(
            fracture_strength_mpa=450.0,
            youngs_modulus_gpa=220.0,
            poisson_ratio=0.25,
            thermal_conductivity_w_m_k=45.0,
            thermal_expansion_coeff_1_k=8.5e-6,
        )
        self.assertIn("thermal_shock_critical_delta_t_k", shock_res)
        self.assertGreater(shock_res["thermal_shock_critical_delta_t_k"], 100.0)

        rad_res = self.thermal.compute_radiation_displacement_threshold(
            cohesive_energy_ev_atom=5.8,
            shear_modulus_gpa=95.0,
        )
        self.assertIn("threshold_displacement_energy_ed_ev", rad_res)
        self.assertGreater(rad_res["threshold_displacement_energy_ed_ev"], 20.0)

    def test_generative_crystal_systems_and_percolation(self):
        crystal_hex = self.synth.synthesize_unconstrained_crystal_structure(archetype="Hexagonal_Wurtzite")
        self.assertEqual(crystal_hex.space_group, "P6_3mc")

        pathway = self.synth.find_percolation_pathways_dijkstra(crystal_hex)
        self.assertIn("minimum_bottleneck_radius_angstrom", pathway)
        self.assertGreater(pathway["minimum_bottleneck_radius_angstrom"], 0.0)

    def test_anisotropic_fracture_and_taylor_factor(self):
        k_ic = self.cont.compute_anisotropic_fracture_toughness(youngs_modulus_gpa=210.0, poisson_ratio=0.28)
        self.assertGreater(k_ic, 10.0)


if __name__ == "__main__":
    unittest.main()
