"""Unit tests for Universal Physics Domains, Lippmann-Schwinger 3D, and Seitz Symmetry."""

import unittest
import numpy as np

from penziv_materials.physics.semiconductor_electronics import SemiconductorElectronicEngine
from penziv_materials.physics.thermal_extreme_transport import ThermalExtremeTransportEngine
from penziv_materials.physics.electro_chemo_mechanics import CoupledPNPMechanicsSolver
from penziv_materials.physics.radiation_damage import RadiationDamageEngine
from penziv_materials.physics.boltzmann_transport import AbInitioBoltzmannTransportEngine
from penziv_materials.generative.crystal_generator import GenerativeCrystalSynthesizer
from penziv_materials.scale2_continuum.cont_micro import ContMicroAgent
from penziv_materials.scale2_continuum.lippmann_schwinger_solver import LippmannSchwinger3DSolver
from penziv_materials.scale2_continuum.unified_spectral_solver import Unified3DSpectralMultiphysicsSolver
from penziv_materials.scale2_continuum.spectral_multiphase_homogenizer import SpectralMultiphaseHomogenizer
from penziv_materials.scale2_continuum.eyre_milton_homogenizer import AcceleratedEyreMiltonSpectralHomogenizer
from penziv_materials.scale2_continuum.damage_mechanics import NonLocalDamageMechanics
from penziv_materials.scale3_mesoscale.phase_field import PhaseFieldEngine
from penziv_materials.scale4_atomistic.equivariant_mlip import EquivariantMLIPEngine
from penziv_materials.scale5_quantum.dft_engine import DFTEngine
from penziv_materials.structure.space_groups import SpaceGroupSymmetryEngine
from penziv_materials.structure.shubnikov_symmetry import ShubnikovMagneticSymmetryEngine
from penziv_materials.structure.universal_crystallography import UniversalCrystallographicTensorEngine
from penziv_materials.structure.crystal_structure import PeriodicLattice, Site, CrystalStructure
from penziv_materials.structure.amorphous_topologies import AmorphousTopologyEngine, AmorphousMeltQuenchEngine
from penziv_materials.scale3_mesoscale.multiphase_grand_potential import MultiPhaseGrandPotentialEngine
from penziv_materials.scale1_process.multimodal_synthesizability import MultiModalSynthesizabilityEngine
from penziv_materials.scale2_continuum.generalized_slip import UniversalSlipGenerator
from penziv_materials.orchestration.dag_orchestrator import DynamicDAGOrchestrator, MultiscaleDAGNode
from penziv_materials.orchestration.qd_discovery_engine import BayesianQualityDiversityDiscoveryEngine


class TestPhysicsDomains(unittest.TestCase):
    def setUp(self):
        self.semi = SemiconductorElectronicEngine()
        self.thermal = ThermalExtremeTransportEngine()
        self.synth = GenerativeCrystalSynthesizer(target_carrier_cation="Mg")
        self.cont = ContMicroAgent()
        self.sg_engine = SpaceGroupSymmetryEngine()
        self.amorphous = AmorphousTopologyEngine()
        self.melt_quench = AmorphousMeltQuenchEngine()
        self.spectral = Unified3DSpectralMultiphysicsSolver(grid_shape=(8, 8, 8))
        self.homogenizer = SpectralMultiphaseHomogenizer(grid_shape=(8, 8, 8))
        self.multiphase = MultiPhaseGrandPotentialEngine(num_phases=3, grid_shape=(16, 16))
        self.proc = MultiModalSynthesizabilityEngine()
        self.slip_gen = UniversalSlipGenerator()
        self.dag = DynamicDAGOrchestrator()
        self.pnp = CoupledPNPMechanicsSolver(grid_shape=(8, 8, 8))
        self.rad = RadiationDamageEngine()
        self.dft = DFTEngine()

    def test_frohlich_and_brooks_herring_mobility(self):
        pop_res = self.semi.compute_frohlich_pop_mobility(
            effective_mass_relative=0.20,
            eps_static=12.5,
            eps_high_freq=10.0,
            lo_phonon_energy_mev=92.0,
        )
        self.assertIn("frohlich_pop_mobility_cm2_v_s", pop_res)
        self.assertIn("frohlich_coupling_constant_alpha", pop_res)
        self.assertGreater(pop_res["frohlich_coupling_constant_alpha"], 0.0)

        bh_res = self.semi.compute_ionized_impurity_mobility_brooks_herring(
            effective_mass_relative=0.20,
            donor_density_cm3=1.0e17,
            eps_static=12.5,
        )
        self.assertIn("ionized_impurity_mobility_cm2_v_s", bh_res)
        self.assertGreater(bh_res["ionized_impurity_mobility_cm2_v_s"], 0.0)

    def test_radiation_damage_nrt_and_cascades(self):
        res = self.rad.compute_nrt_displacements_per_atom(
            damage_energy_t_dam_kev=15.0,
            threshold_displacement_energy_e_d_ev=35.0,
            ion_fluence_ions_cm2=1.0e16,
            atomic_density_atoms_cm3=8.5e22,
        )
        self.assertIn("nrt_frenkel_pairs_per_pka", res)
        self.assertIn("total_displacements_per_atom_dpa", res)
        self.assertGreater(res["nrt_frenkel_pairs_per_pka"], 0.0)

    def test_universal_crystallography_neumann_and_frank_bilby(self):
        C_init = np.random.uniform(10.0, 150.0, (3, 3, 3, 3))
        point_ops = np.array([np.eye(3), -np.eye(3)])
        C_sym = UniversalCrystallographicTensorEngine.enforce_neumann_symmetry(C_init, point_ops)
        self.assertEqual(C_sym.shape, (3, 3, 3, 3))

        lat_A = np.eye(3) * 3.60
        lat_B = np.eye(3) * 3.65
        n_plane = np.array([0.0, 0.0, 1.0])
        misfit_res = UniversalCrystallographicTensorEngine.evaluate_interphase_misfit_energy_density(
            lat_A, lat_B, n_plane, shear_modulus_gpa=80.0
        )
        self.assertIn("net_misfit_dislocation_density", misfit_res)
        self.assertGreater(misfit_res["net_misfit_dislocation_density"], 0.0)

    def test_ab_initio_boltzmann_transport(self):
        freqs = np.linspace(1.0, 12.0, 20)
        vels = np.ones((20, 3)) * 3000.0
        scatt = np.ones(20) * 0.5
        kappa_lat = AbInitioBoltzmannTransportEngine.solve_phonon_bte_tensor(
            frequencies_thz=freqs,
            group_velocities_m_s=vels,
            scattering_rates_thz=scatt,
            cell_volume_ang3=120.0,
            temperature_k=300.0,
        )
        self.assertEqual(kappa_lat.shape, (3, 3))
        self.assertGreater(kappa_lat[0, 0], 0.0)

        e_bins = np.linspace(-1.0, 1.0, 30)
        dos = np.ones(30) * 2.0
        e_vels = np.ones((30, 3)) * 1.0e5
        tau_e = np.ones(30) * 50.0
        el_bte = AbInitioBoltzmannTransportEngine.solve_electron_bte_tensor(
            energies_ev=e_bins,
            dos_states_ev=dos,
            group_velocities_m_s=e_vels,
            relaxation_times_fs=tau_e,
            cell_volume_ang3=120.0,
            temperature_k=300.0,
            fermi_energy_ev=0.0,
        )
        self.assertIn("electrical_conductivity_tensor_s_m", el_bte)
        self.assertIn("seebeck_tensor_uv_k", el_bte)

    def test_accelerated_eyre_milton_homogenizer(self):
        homog = AcceleratedEyreMiltonSpectralHomogenizer(grid_shape=(8, 8, 8))
        c_field = np.ones((8, 8, 8)) * 160.0
        c_field[4:, :, :] = 1.0
        macro_eps = np.diag([0.001, -0.0005, -0.0005])
        res = homog.homogenize_extreme_contrast_elasticity(c_field, macro_eps, max_iter=15)
        self.assertTrue(res["converged"])
        self.assertTrue(res["is_eyre_milton_accelerated"])

    def test_bayesian_quality_diversity_search(self):
        qd_engine = BayesianQualityDiversityDiscoveryEngine()
        qd_res = qd_engine.execute_quality_diversity_search(
            base_elements=["Ni", "Cr", "Al", "Ti"],
            n_iterations=2,
            batch_size=2,
        )
        self.assertIn("archive_size", qd_res)
        self.assertGreater(qd_res["archive_size"], 0)

    def test_space_vacuum_outgassing_hkl(self):
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

        c_mono = np.eye(6) * 120.0
        c_mono[0, 1] = c_mono[1, 0] = 40.0
        c_mono[0, 5] = c_mono[5, 0] = 10.0
        born_mono = self.sg_engine.evaluate_irreducible_born_stability(c_mono, crystal_system="monoclinic")
        self.assertTrue(born_mono["is_mechanically_stable"])

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

    def test_universal_cauchy_born_and_vrh_aggregates(self):
        from penziv_materials.core.tensors import compute_universal_cauchy_born_stiffness, compute_voigt_reuss_hill_aggregates
        lat_mono = np.array([[4.2, 0.0, 0.0], [0.0, 4.0, 0.0], [0.5, 0.0, 4.5]])
        coords_mono = np.array([[0.0, 0.0, 0.0], [2.1, 2.0, 2.2]])
        species = ["Fe", "Ni"]

        def dummy_pes(lat, coords, spec):
            r = np.linalg.norm(coords[0] - coords[1])
            return float(100.0 * (r - 2.5)**2)

        c_voigt = compute_universal_cauchy_born_stiffness(dummy_pes, lat_mono, coords_mono, species, strain_magnitude=0.005)
        self.assertEqual(c_voigt.shape, (6, 6))
        np.testing.assert_allclose(c_voigt, c_voigt.T, atol=1e-3)

        c_pos = c_voigt + np.eye(6) * 150.0
        vrh = compute_voigt_reuss_hill_aggregates(c_pos)
        self.assertGreater(vrh["bulk_modulus_hill_gpa"], 0.0)
        self.assertGreater(vrh["shear_modulus_hill_gpa"], 0.0)
        self.assertIn("pugh_ductility_ratio", vrh)

    def test_relaxed_2d_gamma_surface(self):
        from penziv_materials.scale5_quantum.gamma_surface import TwoDimensionalGammaSurfaceEngine
        engine = TwoDimensionalGammaSurfaceEngine(grid_resolution=5, use_mlip=False)
        res = engine.evaluate_2d_gamma_surface_grid(relax_z=True, relax_z_steps=5)
        self.assertIn("gamma_surface_grid_mj_m2", res)
        self.assertIn("unstable_stacking_fault_energy_gamma_usf_mj_m2", res)
        self.assertGreater(res["unstable_stacking_fault_energy_gamma_usf_mj_m2"], 0.0)

    def test_grand_canonical_charged_defect_energetics(self):
        defect_res = self.semi.compute_charged_defect_formation_energy(
            e_defect_dft_ev=-150.2,
            e_bulk_dft_ev=-155.0,
            chemical_potentials_ev={"Ga": -3.5, "As": -4.2},
            stoichiometry_change_delta_n={"Ga": -1},  # Ga vacancy
            charge_state_q=-1,
            fermi_energy_ev=0.7,
            e_vbm_ev=0.0,
            dielectric_constant_eps_r=12.9,
            cell_volume_ang3=180.0,
        )
        self.assertIn("formation_energy_ev", defect_res)
        self.assertIn("equilibrium_concentration_cm3", defect_res)
        self.assertIn("fnv_image_charge_correction_ev", defect_res)

    def test_cahill_pohl_minimum_thermal_conductivity(self):
        cahill_res = self.thermal.compute_cahill_pohl_minimum_thermal_conductivity(
            number_density_atoms_m3=6.5e28,
            longitudinal_sound_velocity_m_s=5500.0,
            transverse_sound_velocity_m_s=3200.0,
        )
        self.assertIn("cahill_pohl_kappa_min_w_m_k", cahill_res)
        self.assertGreater(cahill_res["cahill_pohl_kappa_min_w_m_k"], 0.0)

    def test_reactive_interdiffusion_stefan_growth(self):
        from penziv_materials.physics.cohesive_interface import CohesiveZoneInterfaceEngine
        engine = CohesiveZoneInterfaceEngine(temperature_k=1100.0)
        growth_res = engine.solve_reactive_interdiffusion_stefan_growth(time_seconds=36000.0)
        self.assertIn("layer_thickness_microns", growth_res)
        self.assertIn("growth_rate_microns_per_hour", growth_res)
        self.assertGreater(growth_res["layer_thickness_microns"], 0.0)

    def test_dynamic_active_convex_hull_and_multivalent_redox(self):
        from penziv_materials.adapters.standard_adapters import PhaseDiagramAdapter
        res = PhaseDiagramAdapter.compute_energy_above_hull(
            target_formula="Al2O3",
            target_formation_energy_ev_atom=-6.80,
        )
        self.assertEqual(res["backend"], "pymatgen")
        self.assertIn("energy_above_hull_ev_atom", res)
        self.assertTrue(res["is_thermodynamically_stable"])

    def test_amorphous_melt_quench_molecular_dynamics(self):
        melt_res = self.melt_quench.generate_melt_quenched_glass(
            num_atoms=32,
            t_melt_k=2200.0,
            box_length_angstrom=10.0,
            species_ratio={"Si": 0.8, "O": 0.2},
        )
        self.assertEqual(melt_res["num_atoms"], 32)
        self.assertTrue(melt_res["is_amorphous_glass"])
        self.assertEqual(melt_res["t_target_k"], 300.0)
        coords = np.array(melt_res["vitrified_coordinates_angstrom"])
        self.assertEqual(coords.shape, (32, 3))
        self.assertTrue(np.all(coords >= 0.0) and np.all(coords < 10.0))
        self.assertGreater(melt_res["kinetic_temperature_k"], 0.0)


if __name__ == "__main__":
    unittest.main()

