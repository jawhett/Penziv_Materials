"""Unit tests for State-Driven Dynamic DAG Orchestrator."""

import unittest
from penziv_materials.orchestration.state_driven_orchestrator import StateDrivenDAGOrchestrator, MaterialDomainTarget


class TestStateDrivenOrchestrator(unittest.TestCase):
    def setUp(self):
        self.orch = StateDrivenDAGOrchestrator()

    def test_thermoelectric_domain_pipeline(self):
        tgt = MaterialDomainTarget(
            domain_type="thermoelectric",
            target_temperature_k=600.0,
            applied_stress_mpa=50.0,
        )
        res = self.orch.execute_state_driven_pipeline(
            candidate_name="Bi2Te3",
            composition={"Bi": 0.4, "Te": 0.6},
            target=tgt,
        )
        self.assertTrue(res["is_state_driven_pipeline_successful"])
        self.assertIn("lattice_thermal_conductivity_w_m_k", res)
        self.assertIn("seebeck_coefficient_uv_k", res)
        self.assertIn("thermoelectric_power_factor_uw_m_k2", res)
        self.assertGreater(res["lattice_thermal_conductivity_w_m_k"], 0.0)

    def test_superalloy_structural_pipeline(self):
        tgt = MaterialDomainTarget(
            domain_type="structural_alloy",
            target_temperature_k=1123.15,
            applied_stress_mpa=350.0,
        )
        res = self.orch.execute_state_driven_pipeline(
            candidate_name="NiCoCrAlTi",
            composition={"Ni": 0.45, "Co": 0.20, "Cr": 0.15, "Al": 0.10, "Ti": 0.10},
            target=tgt,
        )
        self.assertTrue(res["is_state_driven_pipeline_successful"])
        self.assertIn("symmetric_stiffness_tensor", res)
        self.assertIn("phase_field_fractions", res)
        self.assertIn("work_of_separation_j_m2", res)


if __name__ == "__main__":
    unittest.main()
