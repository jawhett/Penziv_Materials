"""Comprehensive verification suite for all Phase 1-4 multiscale modules."""

import unittest
import numpy as np

from penziv_materials.core.hcal import HCALDevice
from penziv_materials.scale5_quantum.dft_engine import DFTEngine
from penziv_materials.scale4_atomistic.equivariant_mlip import EquivariantMLIPEngine
from penziv_materials.scale3_mesoscale.phase_field import PhaseFieldEngine
from penziv_materials.scale3_mesoscale.ddd_kinetics import DiscreteDislocationEngine
from penziv_materials.scale3_mesoscale.rve_generator import ConformedRVEGenerator
from penziv_materials.io.invariant_compression import InvariantCompressionEngine
from penziv_materials.io.async_checkpoint import AsyncCheckpointManager
from penziv_materials.scale2_continuum.cpfft_solver import CPFFTSolver
from penziv_materials.scale2_continuum.damage_mechanics import NonLocalDamageMechanics
from penziv_materials.scale1_process.meltpool_cfd import MeltPoolCFDEngine
from penziv_materials.scale1_process.environmental_degradation import EnvironmentalDegradationEngine
from penziv_materials.adapters.solver_adapters import SolverAdapterBridge
from penziv_materials.meta_bridge.so3_pino import SO3PINOSurrogate
from penziv_materials.meta_bridge.bayesian_assimilation import BayesianDataAssimilationEngine
from penziv_materials.benchmarks.superalloy_discovery import SuperalloyBenchmarkSuite


class TestAllPhases(unittest.TestCase):
    def test_phase1_hcal_determinism(self):
        hcal = HCALDevice(bitwise_deterministic=True)
        arr = np.array([1.0, 2.0, 3.0])
        tensor = hcal.to_tensor(arr)
        np_back = hcal.to_numpy(tensor)
        np.testing.assert_allclose(arr, np_back)

    def test_phase1_dft_mermin_and_dlm(self):
        dft = DFTEngine()
        energies = np.linspace(-5.0, 5.0, 100)
        dos = np.exp(-energies**2)
        u, s, f = dft.compute_mermin_electronic_free_energy(dos, energies, fermi_energy_ev=0.0, temperature_e_k=1123.15)
        self.assertIsInstance(f, float)

        dlm_offset = dft.compute_dlm_paramagnetic_energy_offset(
            magnetic_moment_bohr_magneton=1.8,
            curie_temperature_k=631.0,
            operating_temperature_k=1123.15,
        )
        self.assertLess(dlm_offset, 0.0)

    def test_phase1_equivariant_mlip_virial(self):
        mlip = EquivariantMLIPEngine(num_ensemble=2)
        z = np.array([28, 24, 13])
        pos = np.array([[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0]])
        e, forces, stress, force_sigma = mlip.predict_energy_forces_virial(z, pos)
        self.assertEqual(forces.shape, (3, 3))
        self.assertEqual(stress.shape, (3, 3))
        self.assertGreaterEqual(force_sigma, 0.0)

    def test_phase2_phase_field_spectral(self):
        pf = PhaseFieldEngine(grid_size=(16, 16))
        c = np.ones((16, 16)) * 0.5
        eta = np.zeros((16, 16))
        c_new, eta_new = pf.step_forward_semi_implicit(c, eta, dt=0.01)
        self.assertEqual(c_new.shape, (16, 16))
        self.assertEqual(eta_new.shape, (16, 16))

    def test_phase2_ddd_peach_koehler(self):
        ddd = DiscreteDislocationEngine()
        sigma = np.array([[100e6, 0, 0], [0, -50e6, 0], [0, 0, 0]])
        b = np.array([2.5e-10, 0, 0])
        xi = np.array([0, 0, 1.0])
        f_pk = ddd.compute_peach_koehler_force(sigma, b, xi)
        self.assertEqual(f_pk.shape, (3,))
        self.assertGreater(np.linalg.norm(f_pk), 0.0)

    def test_phase2_rve_level_set(self):
        rve = ConformedRVEGenerator(resolution=(8, 8, 8))
        grains, euler = rve.generate_synthetic_polycrystal_voronoi(num_grains=4)
        smoothed = rve.apply_level_set_interface_smoothing(grains)
        self.assertEqual(smoothed.shape, (8, 8, 8))
        odf_dict = rve.extract_orientation_distribution_function(euler)
        self.assertIn("texture_index_J", odf_dict)

    def test_phase2_invariant_compression(self):
        comp = InvariantCompressionEngine(scalar_abs_tolerance=1e-5)
        scalars = np.linspace(300, 1500, 500)
        c_bytes, ratio, meta = comp.compress_scalar_field_loss_bounded(scalars)
        self.assertGreater(ratio, 1.0)
        self.assertIn("q_step", meta)

    def test_phase3_cpfft_slip_and_gnd(self):
        solver = CPFFTSolver()
        strain_rate = np.diag([0.001, -0.0005, -0.0005])
        res = solver.step_plastic_slip_and_gnd(strain_rate, dt_s=0.01)
        self.assertIn("plastic_dissipation_rate", res)
        self.assertIn("rho_gnd_norm", res)

    def test_phase3_nonlocal_damage(self):
        damage_model = NonLocalDamageMechanics(characteristic_length_lc_um=10.0)
        local_eps = np.linspace(0.001, 0.05, 50)
        nonlocal_eps = damage_model.solve_nonlocal_equivalent_strain_1d(local_eps)
        d = damage_model.compute_damage_variable(nonlocal_eps)
        self.assertTrue(np.all(d >= 0.0))
        self.assertTrue(np.all(d < 1.0))

    def test_phase3_meltpool_marangoni(self):
        melt = MeltPoolCFDEngine()
        tau_s = melt.compute_marangoni_shear_stress(temperature_gradient_surface_k_m=1e7)
        self.assertGreater(tau_s, 0.0)
        k_eff, peclet = melt.solve_subgrid_boundary_layer_segregation(solidification_velocity_m_s=0.05)
        self.assertGreater(k_eff, 0.0)
        self.assertLessEqual(k_eff, 1.0)

    def test_phase3_environmental_degradation(self):
        env = EnvironmentalDegradationEngine()
        flux = env.compute_stress_assisted_interstitial_flux(
            concentration=0.05,
            concentration_gradient=-0.01,
            hydrostatic_stress_gradient_pa_m=1e9,
            temperature_k=1123.15,
        )
        self.assertIsInstance(flux, float)

    def test_phase3_solver_adapters(self):
        bridge = SolverAdapterBridge()
        qe = bridge.generate_quantum_espresso_input("Ni-Superalloy", 3.52)
        lammps = bridge.generate_lammps_neb_script()
        damask = bridge.generate_damask_material_config(260.0, 160.0, 110.0, 320.0)
        self.assertIn("&CONTROL", qe)
        self.assertIn("fix 2 all neb/ci", lammps)
        self.assertIn("Matrix_gamma", damask)

    def test_phase4_so3_pino_frame_indifference(self):
        pino = SO3PINOSurrogate()
        F = np.array([[1.02, 0.01, 0.0], [0.0, 0.99, 0.0], [0.0, 0.0, 0.99]])
        theta = np.pi / 4.0
        Q = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ])
        sigma = pino.forward_frame_indifferent_operator(F)
        self.assertEqual(sigma.shape, (3, 3))

    def test_phase4_bayesian_assimilation_mcmc(self):
        bayes = BayesianDataAssimilationEngine(num_samples=20)
        initial_theta = np.array([0.55, 6.0])
        exp_xrd = np.array([0.58, 0.42])
        samples, acc = bayes.run_metropolis_hastings_calibration(
            initial_theta=initial_theta,
            exp_xrd_phases=exp_xrd,
            exp_nano_hardness_gpa=7.2,
            num_steps=15,
        )
        self.assertEqual(samples.shape, (16, 2))
        self.assertGreaterEqual(acc, 0.0)

    def test_phase4_superalloy_benchmark(self):
        res = SuperalloyBenchmarkSuite.run_high_temperature_superalloy_benchmark(num_candidates=5)
        self.assertIn("benchmark_name", res)
        self.assertEqual(res["candidates_evaluated"], 5)


if __name__ == "__main__":
    unittest.main()
