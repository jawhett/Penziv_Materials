"""Unit tests for Active Learning Loop, 3D TPMS PNP solver, and Robotic Script Generation."""

import unittest
import numpy as np

from penziv_materials.structure.crystal_structure import PeriodicLattice, Site, CrystalStructure
from penziv_materials.scale4_atomistic.equivariant_mlip import EquivariantMLIPEngine
from penziv_materials.orchestration.active_learning_loop import ActiveLearningHPCDispatchLoop
from penziv_materials.meta_bridge.bayesian_assimilation import BayesianDataAssimilationEngine
from penziv_materials.synthesis.retrosynthesis_planner import RetrosynthesisAssemblyPlanner
from penziv_materials.multiphysics.coupled_pnp_mechanics import CoupledPNPMechanicsSolver
from penziv_materials.adapters.solver_adapters import SolverAdapterBridge


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

    def test_automated_dft_dispatch(self):
        retrain_res = self.al_loop.dispatch_automated_dft_and_retrain(self.crystal, self.mlip)
        self.assertTrue(retrain_res["slurm_job_submitted"])
        self.assertTrue(retrain_res["dft_scf_converged"])
        self.assertGreater(retrain_res["active_learning_pool_size"], 0)

    def test_dynamic_qe_deck_generation(self):
        bridge = SolverAdapterBridge()
        qe_deck = bridge.generate_quantum_espresso_input(formula="NiAl", crystal_structure=self.crystal)
        self.assertIn("ATOMIC_SPECIES", qe_deck)
        self.assertIn("Ni", qe_deck)
        self.assertIn("Al", qe_deck)
        self.assertIn("K_POINTS (automatic)", qe_deck)

    def test_3d_pnp_tpms_space_charge(self):
        pnp = CoupledPNPMechanicsSolver()
        tpms_grid = np.random.uniform(0, 1, (16, 16, 16))
        pnp_3d = pnp.solve_space_charge_potential_3d(tpms_grid, applied_voltage_v=0.05)
        self.assertIn("peak_electric_field_v_m", pnp_3d)
        self.assertIn("triple_phase_current_crowding_factor", pnp_3d)
        self.assertGreater(pnp_3d["peak_electric_field_v_m"], 0.0)

    def test_xrd_pseudo_voigt_simulation(self):
        bayes = BayesianDataAssimilationEngine()
        two_theta = np.linspace(20.0, 80.0, 300)
        peaks = [38.5, 44.7, 65.1]
        intensities = [1000.0, 500.0, 300.0]
        y_sim = bayes.simulate_xrd_pseudo_voigt_pattern(two_theta, peaks, intensities)
        self.assertEqual(len(y_sim), 300)
        r_wp = bayes.compute_rietveld_residual_rwp(y_sim, y_sim * 1.02)
        self.assertLess(r_wp, 0.05)

    def test_raw_xrd_and_eis_parsers(self):
        bayes = BayesianDataAssimilationEngine()
        xy_data = "20.5  120.0\n21.0  350.0\n21.5  850.0\n22.0  200.0"
        two_theta, ints = bayes.parse_raw_xrd_xy_file(xy_data)
        self.assertEqual(len(two_theta), 4)

        mpt_data = "1000000.0  12.5  -1.2\n10000.0  45.0  -25.0\n100.0  145.0  -5.0"
        eis_res = bayes.parse_and_fit_biologic_eis_mpt(mpt_data)
        self.assertIn("extracted_ionic_conductivity_ms_cm", eis_res)
        self.assertGreater(eis_res["extracted_ionic_conductivity_ms_cm"], 0.0)

    def test_multistep_synthesis_path(self):
        synth = RetrosynthesisAssemblyPlanner()
        path = synth.find_optimal_multistep_synthesis_path("MgSc2S4", ["MgS", "Sc2S3"])
        self.assertTrue(path["is_kinetically_feasible"])
        self.assertLess(path["cumulative_delta_g_kj_mol"], 0.0)

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
