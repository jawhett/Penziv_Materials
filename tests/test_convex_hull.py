"""Unit tests for Grand Canonical Convex Hull and Phase Stability."""

import unittest
from penziv_materials.thermodynamics.convex_hull import GrandCanonicalConvexHull
from penziv_materials.electrochem.phase_stability import ElectrochemicalPhaseStabilityEngine


class TestConvexHull(unittest.TestCase):
    def setUp(self):
        self.hull = GrandCanonicalConvexHull()
        self.phase_engine = ElectrochemicalPhaseStabilityEngine(metal_reference="Mg")

    def test_energy_above_convex_hull(self):
        res = self.hull.compute_energy_above_convex_hull(
            candidate_formula="MgSc2S4",
            candidate_energy_per_atom_ev=-2.20,
        )
        self.assertIn("energy_above_hull_mev_atom", res)
        self.assertTrue(res["is_thermodynamically_stable"])

    def test_electrochemical_window_vs_reference(self):
        v_red, v_ox = self.hull.compute_electrochemical_window_vs_reference_metal(
            candidate_formula="MgSc2S4",
            candidate_formation_energy_ev_atom=-2.20,
            reference_metal="Mg",
        )
        self.assertLess(v_red, 0.5)
        self.assertGreater(v_ox, 2.5)

    def test_phase_stability_engine_integration(self):
        stab = self.phase_engine.evaluate_electrochemical_stability_window("MgSc2S4")
        self.assertIn("stability_window_width_v", stab)
        self.assertGreater(stab["stability_window_width_v"], 1.5)


if __name__ == "__main__":
    unittest.main()
