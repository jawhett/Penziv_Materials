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


if __name__ == "__main__":
    unittest.main()
