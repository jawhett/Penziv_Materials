"""Unit tests for Grand Canonical Convex Hull and Phase Stability using pymatgen PhaseDiagramAdapter."""

import unittest
from penziv_materials.adapters.standard_adapters import PhaseDiagramAdapter
from penziv_materials.electrochem.phase_stability import ElectrochemicalPhaseStabilityEngine


class TestConvexHull(unittest.TestCase):
    def setUp(self):
        self.phase_engine = ElectrochemicalPhaseStabilityEngine(metal_reference="Mg")

    def test_energy_above_convex_hull(self):
        res = PhaseDiagramAdapter.compute_energy_above_hull(
            target_formula="TiO2",
            target_formation_energy_ev_atom=-8.70,
        )
        self.assertEqual(res["backend"], "pymatgen")
        self.assertIn("energy_above_hull_ev_atom", res)
        self.assertTrue(res["is_thermodynamically_stable"])

    def test_phase_stability_engine_integration(self):
        stab = self.phase_engine.evaluate_electrochemical_stability_window("MgSc2S4")
        self.assertIn("stability_window_width_v", stab)
        self.assertGreater(stab["stability_window_width_v"], 1.5)


if __name__ == "__main__":
    unittest.main()

