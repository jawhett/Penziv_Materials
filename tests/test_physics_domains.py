"""Unit tests for Universal Physics Domains, Lippmann-Schwinger 3D, and Seitz Symmetry."""

import unittest
import numpy as np

from penziv_materials.physics.semiconductor_electronics import SemiconductorElectronicEngine
from penziv_materials.physics.thermal_extreme_transport import ThermalExtremeTransportEngine
from penziv_materials.generative.crystal_generator import GenerativeCrystalSynthesizer
from penziv_materials.scale2_continuum.cont_micro import ContMicroAgent
from penziv_materials.scale2_continuum.lippmann_schwinger_solver import LippmannSchwinger3DSolver
from penziv_materials.scale2_continuum.damage_mechanics import NonLocalDamageMechanics
from penziv_materials.scale3_mesoscale.phase_field import PhaseFieldEngine
from penziv_materials.scale5_quantum.dft_engine import DFTEngine
from penziv_materials.structure.space_groups import SpaceGroupSymmetryEngine
from penziv_materials.structure.space_group_builder import UniversalCrystalBuilder
from penziv_materials.structure.crystal_structure import PeriodicLattice, Site
from penziv_materials.structure.amorphous_topologies import AmorphousTopologyEngine
from penziv_materials.scale3_mesoscale.multiphase_grand_potential import MultiPhaseGrandPotentialEngine
from penziv_materials.scale1_process.multimodal_synthesizability import MultiModalSynthesizabilityEngine
from penziv_materials.thermodynamics.dynamic_hull import UniversalConvexHullSolver
from penziv_materials.scale2_continuum.generalized_slip import UniversalSlipGenerator


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
        self.ls_solver = LippmannSchwinger3DSolver(grid_shape=(8, 8, 8))
        self.pf_3d = PhaseFieldEngine(grid_size=(8, 8, 8))
        self.damage_3d = NonLocalDamageMechanics(grid_shape=(8, 8, 8))
        self.dft = DFTEngine()

    def test_dft_crpa_and_gsfe(self):
        chi0 = np.eye(4) * -0.05
        crpa_res = self.dft.compute_crpa_screened_coulomb_u(electronic_polarizability_matrix=chi0)
        self.assertIn("screened_coulomb_u_ev", crpa_res)
        self.assertGreater(crpa_res["screened_coulomb_u_ev"], 1.0)

        gamma_sfe = self.dft.compute_generalized_stacking_fault_energy(0.5)
        self.assertGreater(gamma_sfe, 10.0)

    def test_3d_phase_field_fracture(self):
        d_init = np.zeros((8, 8, 8))
        eps_field = np.zeros((8, 8, 8, 3, 3))
        eps_field[3:5, 3:5, 3:5] = np.diag([0.05, -0.01, -0.01])  # Tensile localized zone
        frac_res = self.damage_3d.solve_3d_phase_field_fracture_step(d_init, eps_field, dt=0.02)
        self.assertIn("max_damage_parameter", frac_res)
        self.assertGreater(frac_res["max_damage_parameter"], 0.0)

    def test_lippmann_schwinger_3d_solver(self):
        stiffness_field = np.ones((8, 8, 8)) * 160.0
        stiffness_field[2:6, 2:6, 2:6] = 240.0  # Hard inclusion
        macro_eps = np.diag([0.01, -0.003, -0.003])
        res = self.ls_solver.solve_heterogeneous_elastic_equilibrium(stiffness_field, macro_eps, max_iter=15)
        self.assertIn("homogenized_stress_gpa", res)
        self.assertGreater(res["max_von_mises_stress_gpa"], 0.0)

    def test_3d_phase_field_stepping(self):
        c_init = np.ones((8, 8, 8)) * 0.50
        eta_init = np.zeros((8, 8, 8))
        eta_init[3:5, 3:5, 3:5] = 1.0
        c_new, eta_new = self.pf_3d.step_forward_semi_implicit(c_init, eta_init, dt=0.01, n_steps=2)
        self.assertEqual(c_new.shape, (8, 8, 8))
        self.assertEqual(eta_new.shape, (8, 8, 8))

    def test_seitz_screw_and_glide_operations(self):
        ops_screw = UniversalCrystalBuilder.generate_standard_symmetry_operations("P2_1/c")
        self.assertGreaterEqual(len(ops_screw), 2)
        translations = [t for R, t in ops_screw if np.linalg.norm(t) > 0]
        self.assertGreater(len(translations), 0)

    def test_universal_crystal_builder_wyckoff(self):
        lat = PeriodicLattice.from_parameters(4.0, 4.0, 4.0, 90.0, 90.0, 90.0)
        sym_ops = UniversalCrystalBuilder.generate_standard_symmetry_operations("Fm-3m")
        asym = [("Ni", np.array([0.0, 0.0, 0.0]))]
        crystal = UniversalCrystalBuilder.expand_wyckoff_sites(lat, sym_ops, asym, space_group="Fm-3m", space_group_number=225)
        self.assertGreaterEqual(len(crystal.sites), 4)

    def test_universal_slip_and_gsfe(self):
        crystal = self.synth.synthesize_unconstrained_crystal_structure(archetype="Cubic_Spinel")
        systems = UniversalSlipGenerator.generate_systems_from_structure(crystal)
        self.assertGreater(len(systems), 0)
        self.assertIn("schmid_tensor", systems[0])

        tau_crss = UniversalSlipGenerator.compute_gsfe_critical_resolved_shear_stress(
            shear_modulus_gpa=80.0,
            burgers_magnitude_angstrom=2.54,
            interplanar_spacing_angstrom=2.07,
        )
        self.assertGreater(tau_crss, 0.01)

    def test_dynamic_convex_hull_solver(self):
        comp = {"Mg": 1.0, "S": 1.0}
        ref_db = [
            {"composition": {"Mg": 1.0}, "energy_per_atom": 0.0, "formula": "Mg"},
            {"composition": {"S": 1.0}, "energy_per_atom": 0.0, "formula": "S"},
            {"composition": {"Mg": 1.0, "S": 1.0}, "energy_per_atom": -1.82, "formula": "MgS"},
        ]
        res = UniversalConvexHullSolver.solve_stability(comp, -1.82, ref_db)
        self.assertTrue(res["is_thermodynamically_stable"])
        self.assertAlmostEqual(res["energy_above_hull_ev_atom"], 0.0, places=2)

        g_t = UniversalConvexHullSolver.compute_temperature_dependent_gibbs_energy(-1.82, temperature_k=500.0, composition=comp)
        self.assertIsInstance(g_t, float)

    def test_amorphous_dense_random_packing(self):
        drp_res = self.amorphous.generate_stochastic_dense_random_packing(num_atoms=32, box_length_angstrom=10.0)
        self.assertGreater(drp_res["num_atoms_packed"], 10)
        self.assertGreater(drp_res["packing_fraction"], 0.05)

    def test_effective_mass_and_mobility(self):
        curvature = np.diag([12.5, 12.5, 15.0])
        m_tensor, m_scalar = self.semi.compute_effective_mass_tensor(curvature)
        self.assertEqual(m_tensor.shape, (3, 3))
        self.assertGreater(m_scalar, 0.0)

        mob_res = self.semi.compute_carrier_mobility(m_scalar, deformation_potential_ev=6.5)
        self.assertIn("electron_mobility_cm2_v_s", mob_res)
        self.assertGreater(mob_res["electron_mobility_cm2_v_s"], 1.0)

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

    def test_multiphase_grand_potential_stepping(self):
        phi_init = np.ones((3, 16, 16)) / 3.0
        chem_pot = np.zeros(3)
        phi_new = self.multiphase.step_forward_multiphase_field(phi_init, chem_pot, dt_s=0.005)
        self.assertEqual(phi_new.shape, (3, 16, 16))
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
