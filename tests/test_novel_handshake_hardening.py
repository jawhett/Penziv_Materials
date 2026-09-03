"""Unit tests for Novel Multiscale Handshake Gates, Variational Stability, QD MAP-Elites & Retrosynthesis."""

import unittest
import numpy as np

from penziv_materials.validation.handshake_gates import HandshakeGatekeeper
from penziv_materials.core.models import ValidationStatus
from penziv_materials.swarm.holistic_stability import HolisticStabilityRelaxationEngine
from penziv_materials.swarm.map_elites import MAPElitesSwarmEngine
from penziv_materials.synthesis.retrosynthesis_planner import RetrosynthesisAssemblyPlanner


class TestNovelHandshakeAndSynthesis(unittest.TestCase):
    def test_scale_5_4_force_residual_gate(self):
        receipt_pass = HandshakeGatekeeper.validate_force_residual(5e-5)
        self.assertEqual(receipt_pass.status, ValidationStatus.PASSED)

        receipt_fail = HandshakeGatekeeper.validate_force_residual(5e-3)
        self.assertEqual(receipt_fail.status, ValidationStatus.FAILED)

    def test_scale_5_4_ood_density_gate(self):
        receipt_in = HandshakeGatekeeper.validate_ood_density(2.5, threshold=5.0)
        self.assertEqual(receipt_in.status, ValidationStatus.PASSED)

        receipt_ood = HandshakeGatekeeper.validate_ood_density(8.5, threshold=5.0)
        self.assertEqual(receipt_ood.status, ValidationStatus.ROUTED_TO_HIGH_FIDELITY)

    def test_scale_4_3_stacking_fault_positivity_and_trip_twip(self):
        # Stable dislocation slip
        rec_stable = HandshakeGatekeeper.validate_stacking_fault_positivity(55.0)
        self.assertEqual(rec_stable.status, ValidationStatus.PASSED)

        # TRIP/TWIP driver regime (low positive SFE: 0 to 45 mJ/m2)
        rec_trip = HandshakeGatekeeper.validate_stacking_fault_positivity(15.0)
        self.assertEqual(rec_trip.status, ValidationStatus.PASSED)
        self.assertIn("TRIP/TWIP", rec_trip.details)

        # Spontaneous barrierless shear instability (negative SFE <= 0)
        rec_unstable = HandshakeGatekeeper.validate_stacking_fault_positivity(-15.0)
        self.assertEqual(rec_unstable.status, ValidationStatus.FAILED)

    def test_scale_4_3_lognormal_rate_variance(self):
        rec_pass = HandshakeGatekeeper.validate_lognormal_rate_variance(0.12)
        self.assertEqual(rec_pass.status, ValidationStatus.PASSED)

        rec_fail = HandshakeGatekeeper.validate_lognormal_rate_variance(0.75)
        self.assertEqual(rec_fail.status, ValidationStatus.FAILED)

    def test_scale_3_2_rve_convergence(self):
        rec_pass = HandshakeGatekeeper.validate_rve_convergence(0.008)
        self.assertEqual(rec_pass.status, ValidationStatus.PASSED)

        rec_fail = HandshakeGatekeeper.validate_rve_convergence(0.025)
        self.assertEqual(rec_fail.status, ValidationStatus.FAILED)

    def test_scale_2_1_clausius_duhem_dissipation(self):
        rec_pass = HandshakeGatekeeper.validate_clausius_duhem_dissipation(1.2e4)
        self.assertEqual(rec_pass.status, ValidationStatus.PASSED)

        rec_fail = HandshakeGatekeeper.validate_clausius_duhem_dissipation(-1.0)
        self.assertEqual(rec_fail.status, ValidationStatus.FAILED)

    def test_meta_compound_variance_bound(self):
        rec_pass = HandshakeGatekeeper.validate_compound_variance_bound(0.08)
        self.assertEqual(rec_pass.status, ValidationStatus.PASSED)

        rec_fail = HandshakeGatekeeper.validate_compound_variance_bound(0.55)
        self.assertEqual(rec_fail.status, ValidationStatus.FAILED)

    def test_toxicity_and_encapsulated_electronic_exemption(self):
        # Toxic heavy metal in regular application -> Fails
        rec_open = HandshakeGatekeeper.validate_toxicity_and_banned_species(
            banned_elements=["Pb"],
            epa_hazard_score=5.5,
            is_encapsulated_electronic=False,
        )
        self.assertEqual(rec_open.status, ValidationStatus.FAILED)

        # Encapsulated electronic exemption -> Passes
        rec_encapsulated = HandshakeGatekeeper.validate_toxicity_and_banned_species(
            banned_elements=[],
            epa_hazard_score=5.5,
            is_encapsulated_electronic=True,
        )
        self.assertEqual(rec_encapsulated.status, ValidationStatus.PASSED)

    def test_distribution_matching_wasserstein_and_ks(self):
        rng = np.random.default_rng(42)
        exp_dist = rng.normal(500.0, 25.0, 100)
        pred_dist = exp_dist + rng.normal(0.0, 0.5, 100)
        rec = HandshakeGatekeeper.validate_distribution_matching(
            predicted_samples=pred_dist,
            experimental_samples=exp_dist,
            property_name="Yield Strength",
        )
        self.assertEqual(rec.status, ValidationStatus.PASSED)

    def test_holistic_variational_composite_hamiltonian(self):
        engine = HolisticStabilityRelaxationEngine(biot_coefficient_alpha=0.8)
        res = engine.evaluate_multiphase_hamiltonian(
            phase_volume_fractions={"matrix": 0.7, "reinforcement": 0.3},
            phase_strain_energy_densities_mj_m3={"matrix": 40.0, "reinforcement": 90.0},
            phase_critical_strain_energies_mj_m3={"matrix": 60.0, "reinforcement": 120.0},
            fluid_pressure_work_mj_m3=10.0,
            fluid_volume_fraction=0.05,
        )
        self.assertIn("total_system_free_energy_mj_m3", res)
        self.assertIn("composite_co_design_stabilized", res)
        self.assertTrue(res["composite_co_design_stabilized"])

    def test_map_elites_qd_swarm(self):
        engine = MAPElitesSwarmEngine(grid_dimensions=(5, 5, 5))
        # Add elite
        placed = engine.add_candidate(
            candidate_data={"formula": "Ti3AlC2", "fitness": 95.0},
            fitness_score=95.0,
            descriptors=[0.5, 0.2, 0.1],
        )
        self.assertTrue(placed)
        stats = engine.get_archive_statistics()
        self.assertGreater(stats["archive_coverage"], 0.0)

    def test_retrosynthesis_msc_and_robotic_export(self):
        planner = RetrosynthesisAssemblyPlanner()
        # Test MSC Integration
        times = np.linspace(0, 3600, 100)
        temps = np.linspace(300, 1200, 100)
        msc_res = planner.integrate_master_sintering_path(times, temps, q_diff_j_mol=150000.0)
        self.assertIn("theta_msc_path_integral_s_k", msc_res)
        self.assertGreater(msc_res["relative_density_percent"], 80.0)

        # Test Opentrons Script
        ot2_script = planner.export_opentrons_ot2_script("Li7La3Zr2O12", {"Li_precursor": 150.0})
        self.assertIn("from opentrons import protocol_api", ot2_script)

        # Test Robotic JSON Recipe
        recipe_json = planner.export_robotic_synthesis_recipe_json("Li7La3Zr2O12")
        self.assertEqual(recipe_json["target_compound"], "Li7La3Zr2O12")
        self.assertGreaterEqual(len(recipe_json["automation_execution_sequence"]), 5)


if __name__ == "__main__":
    unittest.main()
