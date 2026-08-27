"""Unit tests for 2D Gamma-Surfaces, Transition Path Sampling, ODF Plasticity, and Reverse Monte Carlo."""

import unittest
import numpy as np

from penziv_materials.scale5_quantum.gamma_surface import TwoDimensionalGammaSurfaceEngine
from penziv_materials.scale4_atomistic.path_sampling import TransitionPathSamplingEngine
from penziv_materials.scale2_continuum.odf_crystal_plasticity import ODFTexturePlasticityEngine
from penziv_materials.structure.reverse_monte_carlo import ReverseMonteCarloEngine


class TestGammaODFRMC(unittest.TestCase):
    def setUp(self):
        self.gamma_engine = TwoDimensionalGammaSurfaceEngine(grid_resolution=9)
        self.tps_engine = TransitionPathSamplingEngine(num_string_nodes=7)
        self.odf_engine = ODFTexturePlasticityEngine(num_orientations=30)
        self.rmc_engine = ReverseMonteCarloEngine(box_length_angstrom=10.0, num_atoms=32)

    def test_2d_gamma_surface_grid(self):
        res = self.gamma_engine.evaluate_2d_gamma_surface_grid(miller_plane=(1, 1, 1))
        self.assertIn("gamma_surface_grid_mj_m2", res)
        self.assertIn("unstable_stacking_fault_energy_gamma_usf_mj_m2", res)
        self.assertIn("intrinsic_stacking_fault_energy_gamma_isf_mj_m2", res)
        self.assertGreater(res["unstable_stacking_fault_energy_gamma_usf_mj_m2"], 0.0)

    def test_transition_path_sampling_dijkstra_and_string(self):
        sites = np.array([[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0], [4.5, 0, 0]])
        dijk_res = self.tps_engine.find_shortest_percolation_path_dijkstra(sites)
        self.assertTrue(dijk_res["is_percolating_channel_found"])
        self.assertGreater(len(dijk_res["optimal_path_indices"]), 1)

        string_nodes = np.linspace([0, 0, 0], [4, 2, 0], 7)
        mep_res = self.tps_engine.evolve_string_method_mep(string_nodes)
        self.assertTrue(mep_res["is_mep_converged"])
        self.assertEqual(len(mep_res["converged_mep_nodes"]), 7)

    def test_odf_plasticity_taylor_and_non_schmid(self):
        m_res = self.odf_engine.compute_polycrystalline_taylor_and_sachs_factors()
        self.assertIn("taylor_factor_upper_bound", m_res)
        self.assertIn("sachs_factor_lower_bound", m_res)
        self.assertGreaterEqual(m_res["taylor_factor_upper_bound"], m_res["sachs_factor_lower_bound"])

        stress = np.diag([200.0, -100.0, -100.0])
        s_dir = np.array([1.0, 1.0, 0.0])
        n_plane = np.array([1.0, -1.0, 1.0])
        non_schmid = self.odf_engine.evaluate_non_schmid_resolved_shear_stress(stress, s_dir, n_plane)
        self.assertIn("schmid_resolved_shear_stress_mpa", non_schmid)
        self.assertIn("effective_non_schmid_shear_stress_mpa", non_schmid)

    def test_reverse_monte_carlo_refinement(self):
        coords = np.random.uniform(0.0, 10.0, (32, 3))
        r_bins, g_r = self.rmc_engine.compute_pair_distribution_function(coords)
        self.assertEqual(len(r_bins), 50)
        self.assertEqual(len(g_r), 50)

        rmc_res = self.rmc_engine.run_rmc_refinement(coords, max_mc_steps=30)
        self.assertTrue(rmc_res["is_rmc_converged"])
        self.assertIn("final_chi_squared", rmc_res)


if __name__ == "__main__":
    unittest.main()
