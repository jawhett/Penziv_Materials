"""Unit tests for Active Learning Loop, XRD Rietveld simulation, and Robotic Script Generation."""

import unittest
import numpy as np

from penziv_materials.structure.crystal_structure import PeriodicLattice, Site, CrystalStructure
from penziv_materials.scale4_atomistic.equivariant_mlip import EquivariantMLIPEngine
from penziv_materials.orchestration.active_learning_loop import ActiveLearningHPCDispatchLoop
from penziv_materials.meta_bridge.bayesian_assimilation import BayesianDataAssimilationEngine
from penziv_materials.synthesis.retrosynthesis_planner import RetrosynthesisAssemblyPlanner


class TestActiveLearningAndRobotics(unittest.TestCase):
    def setUp(self):
        lattice = PeriodicLattice.from_parameters(a=5.0, b=5.0, c=5.0)
        sites = [
            Site(species="Ni", fractional_coords=np.array([0.0, 0.0, 0.0])),
            Site(species="Al", fractional_coords=np.array([0.5, 0.5, 0.5])),
        ]
        self.crystal = CrystalStructure(lattice, sites)
        self.mlip = EquivariantMLIPEngine()
        self.al_loop = ActiveLearningHPCDispatchLoop()

    def test_active_learning_uncertainty_evaluation(self):
        res = self.al_loop.evaluate_configuration_uncertainty(self.crystal, self.mlip)
        self.assertIn("max_force_variance_sigma_f", res)
        self.assertIn("status", res)

    def test_xrd_pseudo_voigt_simulation(self):
        bayes = BayesianDataAssimilationEngine()
        two_theta = np.linspace(20.0, 80.0, 300)
        peaks = [38.5, 44.7, 65.1]
        intensities = [1000.0, 500.0, 300.0]
        y_sim = bayes.simulate_xrd_pseudo_voigt_pattern(two_theta, peaks, intensities)
        self.assertEqual(len(y_sim), 300)
        r_wp = bayes.compute_rietveld_residual_rwp(y_sim, y_sim * 1.02)
        self.assertLess(r_wp, 0.05)

    def test_csm_nanoindentation_inversion(self):
        bayes = BayesianDataAssimilationEngine()
        h_depth = np.linspace(50.0, 1000.0, 50)
        stiffness = np.linspace(1e5, 2e6, 50)
        csm_res = bayes.invert_nanoindentation_csm_curve(h_depth, stiffness)
        self.assertIn("inferred_effective_modulus_gpa", csm_res)
        self.assertGreater(csm_res["inferred_effective_modulus_gpa"], 0.0)

    def test_opentrons_ot2_export(self):
        synth = RetrosynthesisAssemblyPlanner()
        ot2_code = synth.export_opentrons_ot2_script(
            candidate_formula="Na3Zr2(SiO4)2(PO4)",
            liquid_precursors_ul={"Zr_precursor": 150.0, "Na_precursor": 220.0, "P_precursor": 100.0},
        )
        self.assertIn("from opentrons import protocol_api", ot2_code)
        self.assertIn("p300.aspirate(150.0", ot2_code)


if __name__ == "__main__":
    unittest.main()
