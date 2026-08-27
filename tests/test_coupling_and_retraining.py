"""Unit tests for Multiscale Coupling, Online Active Retraining, and CVT-MAP-Elites Pareto QD."""

import unittest
import numpy as np

from penziv_materials.scale2_continuum.multiscale_coupling import UniversalMultiscaleCouplingEngine
from penziv_materials.meta_bridge.online_active_retraining import OnlineActiveRetrainingWorkflow
from penziv_materials.orchestration.differentiable_pareto_qd import DifferentiableContinuousParetoQDEngine


class TestCouplingAndRetraining(unittest.TestCase):
    def setUp(self):
        self.coupling = UniversalMultiscaleCouplingEngine()
        self.retraining = OnlineActiveRetrainingWorkflow(force_variance_threshold=0.03, nll_threshold=10.0)
        self.cvt_qd = DifferentiableContinuousParetoQDEngine(latent_dim=6, num_centroids=20)

    def test_monolithic_chemo_mechanics_3d(self):
        stiffness = np.zeros((8, 8, 8, 3, 3, 3, 3))
        for i in range(3):
            stiffness[..., i, i, i, i] = 160.0e9
            for j in range(3):
                if i != j:
                    stiffness[..., i, i, j, j] = 70.0e9
                    stiffness[..., i, j, i, j] = 45.0e9

        c_field = np.ones((8, 8, 8)) * 1000.0
        c_field[4:, :, :] = 1200.0  # Inhomogeneous solute gradient

        res = self.coupling.solve_monolithic_chemo_mechanics_3d(
            stiffness_field_c4=stiffness,
            concentration_field=c_field,
            max_iter=15,
        )
        self.assertTrue(res["is_converged"])
        self.assertIn("stress_chemical_potential_j_mol", res)
        self.assertIn("max_von_mises_stress_mpa", res)
        self.assertGreater(res["max_von_mises_stress_mpa"], 0.0)

    def test_online_active_retraining_workflow(self):
        lat = np.eye(3) * 3.60
        species = ["Ni", "Al"]
        # Intentionally distorted coordinates to trigger OOD active learning dispatch
        fracs = np.array([[0.0, 0.0, 0.0], [0.15, 0.15, 0.15]])

        res = self.retraining.execute_active_learning_cycle(
            system_formula="NiAl_distorted",
            atomic_species=species,
            fractional_coords=fracs,
            lattice_matrix=lat,
            simulated_dft_ground_truth_energy=-12.50,
        )
        self.assertTrue(res["is_retraining_loop_active"])
        self.assertTrue(res["first_principles_dispatched"])
        self.assertIsNotNone(res["dispatch_and_ingestion_event"])
        self.assertGreater(res["total_active_learning_cycles_executed"], 0)

    def test_cvt_map_elites_search(self):
        res = self.cvt_qd.execute_cvt_map_elites_search(
            base_elements=["Ni", "Cr", "Al"],
            num_evaluations=4,
        )
        self.assertTrue(res["is_cvt_search_successful"])
        self.assertGreater(res["occupied_cvt_cells"], 0)
        self.assertIn("archive_coverage_percent", res)


if __name__ == "__main__":
    unittest.main()
