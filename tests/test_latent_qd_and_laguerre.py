"""Unit tests for Laguerre Voronoi, Unsupervised Latent QD, and Anisotropic Spectral Solvers."""

import unittest
import numpy as np

from penziv_materials.structure.laguerre_voronoi import MulticomponentLaguerreVoronoiEngine
from penziv_materials.orchestration.latent_qd_engine import UnsupervisedLatentQDEngine
from penziv_materials.scale2_continuum.unified_spectral_solver import Unified3DSpectralMultiphysicsSolver


class TestLatentQDAndLaguerre(unittest.TestCase):
    def setUp(self):
        self.laguerre = MulticomponentLaguerreVoronoiEngine(box_length_angstrom=12.0)
        self.latent_qd = UnsupervisedLatentQDEngine(latent_dim=2)
        self.spectral = Unified3DSpectralMultiphysicsSolver(grid_shape=(8, 8, 8))

    def test_multicomponent_laguerre_voronoi_and_rings(self):
        coords = np.random.uniform(0.0, 12.0, (24, 3))
        species = ["Si"] * 8 + ["O"] * 16

        lag_res = self.laguerre.compute_weighted_laguerre_voronoi(coords, species)
        self.assertTrue(lag_res["is_multicomponent_weighted"])
        self.assertIn("mean_laguerre_coordination", lag_res)

        ring_res = self.laguerre.compute_kings_ring_statistics(coords, species)
        self.assertIn("ring_size_distribution", ring_res)
        self.assertIn("medium_range_order_index", ring_res)

        homology_res = self.laguerre.compute_betti_persistent_homology_invariants(coords)
        self.assertIn("betti_1_topological_loops", homology_res)

    def test_unsupervised_latent_qd_discovery(self):
        res = self.latent_qd.execute_unsupervised_discovery(
            base_elements=["Ti", "Al", "V"],
            num_candidates=3,
        )
        self.assertIn("total_archive_size", res)
        self.assertGreater(res["total_archive_size"], 0)

    def test_anisotropic_greens_operator_projection(self):
        # 3x3x3x3 arbitrary anisotropic reference elasticity C0
        c0_aniso = np.zeros((3, 3, 3, 3))
        for i in range(3):
            c0_aniso[i, i, i, i] = 160.0e9
            for j in range(3):
                if i != j:
                    c0_aniso[i, i, j, j] = 70.0e9
                    c0_aniso[i, j, i, j] = 45.0e9

        tau_hat = np.random.uniform(0.0, 1.0, (8, 8, 8, 3, 3)).astype(np.complex128)
        eps_hat = self.spectral.apply_acoustic_greens_operator(tau_hat, c0_anisotropic_rank4=c0_aniso)
        self.assertEqual(eps_hat.shape, (8, 8, 8, 3, 3))
        # Zero wavenumber strain should be zero
        np.testing.assert_allclose(eps_hat[0, 0, 0], np.zeros((3, 3)), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
