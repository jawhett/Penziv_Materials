"""Unit tests for Persistent Topology, Unsupervised Latent QD, and Anisotropic Spectral Solvers."""

import unittest
import numpy as np

from penziv_materials.adapters.standard_adapters import TopologyAdapter
from penziv_materials.orchestration.latent_qd_engine import UnsupervisedLatentQDEngine
from penziv_materials.scale2_continuum.unified_spectral_solver import Unified3DSpectralMultiphysicsSolver


class TestLatentQDAndLaguerre(unittest.TestCase):
    def setUp(self):
        self.latent_qd = UnsupervisedLatentQDEngine(latent_dim=2)
        self.spectral = Unified3DSpectralMultiphysicsSolver(grid_shape=(8, 8, 8))

    def test_persistent_homology_topology(self):
        coords = np.random.uniform(0.0, 12.0, (24, 3))
        betti = TopologyAdapter.compute_persistent_betti_numbers(coords, max_edge_length=3.0)
        self.assertEqual(betti["backend"], "gudhi")
        self.assertIn("betti_0", betti)
        self.assertIn("betti_1", betti)
        self.assertIn("betti_2", betti)

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
