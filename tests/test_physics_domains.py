"""Unit tests for Universal Physics Domains (Semiconductors, Thermals, Space Groups, Amorphous, Multiphase, SPS)."""

import unittest
import numpy as np

from penziv_materials.physics.semiconductor_electronics import SemiconductorElectronicEngine
from penziv_materials.physics.thermal_extreme_transport import ThermalExtremeTransportEngine
from penziv_materials.generative.crystal_generator import GenerativeCrystalSynthesizer
from penziv_materials.scale2_continuum.cont_micro import ContMicroAgent
from penziv_materials.structure.space_groups import SpaceGroupSymmetryEngine
from penziv_materials.structure.amorphous_topologies import AmorphousTopologyEngine
from penziv_materials.scale3_mesoscale.multiphase_grand_potential import MultiPhaseGrandPotentialEngine
from penziv_materials.scale1_process.multimodal_synthesizability import MultiModalSynthesizabilityEngine


class TestPhysicsDomains(unittest.TestCase):
    def setUp(self):
        self.semi = SemiconductorElectronicEngine()
        self.thermal = ThermalExtremeTransportEngine()
        self.synth = GenerativeCrystalSynthesizer(target_carrier_cation="Mg")
        self.cont = ContMicroAgent()
        self.sg_engine = SpaceGroupSymmetryEngine()
        self.amorphous = AmorphousTopologyEngine()
        self.multiphase = MultiPhaseGrandPotentialEngine(num_phases=3, grid_shape=(16, 16))
        self.proc = MultiModalSynthesizabilityEngine()

    def test_effective_mass_and_mobility(self):
        curvature = np.diag([12.5, 12.5, 15.0])
        m_tensor, m_scalar = self.semi.compute_effective_mass_tensor(curvature)
        self.assertEqual(m_tensor.shape, (3, 3))
        self.assertGreater(m_scalar, 0.0)

        mob_res = self.semi.compute_carrier_mobility(m_scalar, deformation_potential_ev=6.5)
        self.assertIn("electron_mobility_cm2_v_s", mob_res)
        self.assertGreater(mob_res["electron_mobility_cm2_v_s"], 1.0)

        # Wannier BTE tensor
        v_tensor = np.eye(3) * 1.5e5
        bte_res = self.semi.compute_wannier_bte_mobility_tensor(v_tensor, relaxation_time_fs=150.0)
        self.assertIn("isotropic_mobility_cm2_v_s", bte_res)
        self.assertGreater(bte_res["isotropic_mobility_cm2_v_s"], 10.0)

    def test_dielectric_breakdown(self):
        break_res = self.semi.compute_dielectric_tensor_and_breakdown_field(band_gap_ev=3.8)
        self.assertTrue(break_res["is_ultra_wide_bandgap"])
        self.assertGreater(break_res["dielectric_breakdown_field_mv_cm"], 5.0)

    def test_lattice_thermal_conductivity_and_hkl(self):
        k_res = self.thermal.compute_lattice_thermal_conductivity_slack(
            average_atomic_mass_amu=35.0,
            debye_temperature_k=650.0,
            volume_per_atom_ang3=15.0,
        )
        self.assertIn("lattice_thermal_conductivity_w_m_k", k_res)
        self.assertGreater(k_res["lattice_thermal_conductivity_w_m_k"], 10.0)

        # Space vacuum outgassing
        hkl_res = self.thermal.compute_space_vacuum_outgassing_rate_hkl(
            molecular_weight_g_mol=80.0,
            vapor_pressure_pa=1.0e-8,
        )
        self.assertIn("sublimation_mass_flux_kg_m2_s", hkl_res)
        self.assertTrue(hkl_res["is_space_vacuum_stable"])

    def test_irreducible_born_stability_and_slip(self):
        c_matrix = np.eye(6) * 150.0
        c_matrix[0, 1] = c_matrix[0, 2] = c_matrix[1, 0] = c_matrix[1, 2] = c_matrix[2, 0] = c_matrix[2, 1] = 60.0
        born_res = self.sg_engine.evaluate_irreducible_born_stability(c_matrix, crystal_system="cubic")
        self.assertTrue(born_res["is_mechanically_stable"])

        lattice = np.eye(3) * 4.0
        slip_res = self.sg_engine.generate_anisotropic_slip_and_twinning_systems(lattice)
        self.assertGreater(slip_res["num_active_slip_systems"], 0)

    def test_amorphous_rdf_stz_and_vrh(self):
        np.random.seed(42)
        pos = np.random.uniform(0, 20.0, (50, 3))
        rdf_res = self.amorphous.compute_radial_distribution_function(pos, box_length_angstrom=20.0)
        self.assertIn("g_r", rdf_res)

        stz_res = self.amorphous.compute_shear_transformation_zone_plasticity(applied_shear_stress_mpa=350.0)
        self.assertIn("stz_plastic_shear_rate_s_inv", stz_res)

        vrh_res = self.amorphous.compute_variable_range_hopping_transport(regime="Mott")
        self.assertIn("vrh_conductivity_s_cm", vrh_res)

    def test_multiphase_grand_potential_stepping(self):
        phi_init = np.ones((3, 16, 16)) / 3.0
        chem_pot = np.zeros(3)
        phi_new = self.multiphase.step_forward_multiphase_field(phi_init, chem_pot, dt_s=0.005)
        self.assertEqual(phi_new.shape, (3, 16, 16))
        # Verify partition of unity
        sum_phi = np.sum(phi_new, axis=0)
        np.testing.assert_allclose(sum_phi, 1.0, atol=1e-5)

    def test_multimodal_synthesizability(self):
        cvd_res = self.proc.evaluate_chemical_vapor_deposition()
        self.assertTrue(cvd_res["is_synthetically_feasible"])

        sps_res = self.proc.evaluate_spark_plasma_sintering()
        self.assertGreater(sps_res["final_relative_density"], 0.55)

        melt_res = self.proc.evaluate_melt_spinning_glass_formation(wheel_speed_m_s=40.0)
        self.assertIn("is_vitrified_amorphous_ribbon", melt_res)


if __name__ == "__main__":
    unittest.main()
