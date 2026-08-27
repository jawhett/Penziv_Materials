"""Tests for Alloy Discovery Engine and Pareto Optimization."""

import unittest
from penziv_materials.orchestration.discovery_engine import (
    AlloyDiscoveryEngine,
    DiscoveryTargetConstraints,
    ParetoDiscoveryResult,
)


class TestDiscoveryEngine(unittest.TestCase):
    def setUp(self):
        self.engine = AlloyDiscoveryEngine()

    def test_generate_random_compositions(self):
        elements = ["Ni", "Cr", "Al", "Ti", "Nb", "Mo", "B"]
        compositions = self.engine.generate_random_compositions(
            base_elements=elements,
            n_samples=10,
            primary_element="Ni",
            random_seed=123,
        )
        self.assertEqual(len(compositions), 10)
        for comp in compositions:
            self.assertIn("Ni", comp)
            total = sum(comp.values())
            self.assertAlmostEqual(total, 1.0, places=2)
            self.assertGreaterEqual(comp["Ni"], 0.40)

    def test_discover_optimal_alloys_workflow(self):
        elements = ["Ni", "Cr", "Al", "Ti", "Nb", "B"]
        constraints = DiscoveryTargetConstraints(
            min_yield_strength_mpa=900.0,
            max_steady_state_creep_rate_s_inv=1.0e-10,
            min_fracture_toughness_k_ic=50.0,
            max_crustal_exergy_mj_kg=120.0,
            target_temperature_k=1123.15,
        )

        result = self.engine.discover_optimal_alloys(
            base_elements=elements,
            constraints=constraints,
            n_samples=6,
            prefix_name="TestAlloy",
        )

        self.assertEqual(result.total_screened, 6)
        self.assertGreaterEqual(result.physically_stable_count, 1)
        self.assertIsInstance(result, ParetoDiscoveryResult)
        if result.pareto_optimal_candidates:
            top = result.top_candidate
            self.assertIsNotNone(top)
            self.assertEqual(top.pareto_rank, 1)


if __name__ == "__main__":
    unittest.main()
