"""Tests for handshake validation gates, compression, and Meta-Orchestrator."""

import unittest
import numpy as np
from penziv_materials.validation.handshake_gates import HandshakeGatekeeper
from penziv_materials.orchestration.meta_orchestrator import MetaOrchestrator
from penziv_materials.core.models import ValidationStatus
from penziv_materials.io.tiered_storage import TieredStorageManager


class TestHandshakeAndOrchestration(unittest.TestCase):
    def test_handshake_force_residual(self):
        rec_pass = HandshakeGatekeeper.validate_force_residual(5.0e-5)
        self.assertEqual(rec_pass.status, ValidationStatus.PASSED)

        rec_fail = HandshakeGatekeeper.validate_force_residual(2.0e-3)
        self.assertEqual(rec_fail.status, ValidationStatus.FAILED)

    def test_handshake_dissipation(self):
        rec_pass = HandshakeGatekeeper.validate_clausius_duhem_dissipation(1.5e5)
        self.assertEqual(rec_pass.status, ValidationStatus.PASSED)

        rec_fail = HandshakeGatekeeper.validate_clausius_duhem_dissipation(-10.0)
        self.assertEqual(rec_fail.status, ValidationStatus.FAILED)

    def test_meta_orchestrator_forward_loop(self):
        orch = MetaOrchestrator()
        comp = {"Ni": 0.55, "Cr": 0.20, "Al": 0.10, "Ti": 0.05, "Mo": 0.05, "Fe": 0.05}
        cand = orch.run_forward_multiscale_prediction(
            candidate_name="Penziv-Alloy-Test1",
            composition=comp,
            target_temperature_k=1123.15,
        )

        self.assertIsNotNone(cand.quantum)
        self.assertIsNotNone(cand.atomistic)
        self.assertIsNotNone(cand.mesoscale)
        self.assertIsNotNone(cand.continuum)
        self.assertIsNotNone(cand.process)
        self.assertIsNotNone(cand.assimilation)
        self.assertGreaterEqual(len(cand.validation_receipts), 5)

    def test_tiered_compression_lossless_and_bounded(self):
        storage = TieredStorageManager(checkpoint_dir="test_checkpoints")

        # Scalar field
        temp_field = np.linspace(300.0, 1800.0, 1000)
        comp_scalar, ratio_scalar = storage.compress_scalar_field_loss_bounded(temp_field, absolute_error_bound=1e-6)
        self.assertGreater(len(comp_scalar), 0)
        self.assertGreater(ratio_scalar, 1.0)

        # Differential tensor
        tensor_field = np.random.rand(3, 3, 100)
        comp_tensor, ratio_tensor = storage.compress_differential_tensor_lossless(tensor_field)
        self.assertGreater(len(comp_tensor), 0)

    def test_handshake_distribution_matching(self):
        rng = np.random.default_rng(42)
        exp_samples = rng.normal(loc=550.0, scale=30.0, size=100)
        
        # Good matching distribution
        pred_good = rng.normal(loc=552.0, scale=29.0, size=100)
        rec_pass = HandshakeGatekeeper.validate_distribution_matching(
            predicted_samples=pred_good,
            experimental_samples=exp_samples,
            max_wasserstein_distance=0.15,
            property_name="Yield Strength",
        )
        self.assertEqual(rec_pass.status, ValidationStatus.PASSED)

        # Discrepant distribution (large shift in mean/variance)
        pred_bad = rng.normal(loc=750.0, scale=80.0, size=100)
        rec_fail = HandshakeGatekeeper.validate_distribution_matching(
            predicted_samples=pred_bad,
            experimental_samples=exp_samples,
            max_wasserstein_distance=0.10,
            property_name="Yield Strength",
        )
        self.assertEqual(rec_fail.status, ValidationStatus.FAILED)

    def test_process_uncertainty_monte_carlo(self):
        t_arr = np.linspace(0.0, 60.0, 20)
        mean_t = np.linspace(300.0, 900.0, 20)
        std_t = np.full(20, 15.0)
        mean_eps = np.full(20, 1e-3)
        std_eps = np.full(20, 1e-4)

        mc_res = HandshakeGatekeeper.sample_process_uncertainty_monte_carlo(
            mean_temperature_history_k=mean_t,
            std_temperature_history_k=std_t,
            mean_strain_rate_s_inv=mean_eps,
            std_strain_rate_s_inv=std_eps,
            time_series_s=t_arr,
            num_samples=15,
            random_seed=42,
        )

        self.assertEqual(mc_res["num_samples"], 15)
        self.assertEqual(len(mc_res["sampled_property_values"]), 15)
        self.assertGreater(mc_res["mean"], 0.0)
        self.assertGreater(mc_res["std"], 0.0)


if __name__ == "__main__":
    unittest.main()
