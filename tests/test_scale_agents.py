"""Tests for specialized scale agents."""

import unittest
import numpy as np
from penziv_materials.scale5_quantum.q_elec import QElectAgent
from penziv_materials.scale4_atomistic.atom_dyn import AtomDynAgent
from penziv_materials.scale3_mesoscale.meso_kinetic import MesoKineticAgent
from penziv_materials.scale2_continuum.cont_micro import ContMicroAgent
from penziv_materials.scale1_process.proc_mfg import ProcMfgAgent
from penziv_materials.meta_bridge.uq_bridge import UqBridgeAgent


class TestScaleAgents(unittest.TestCase):
    def test_q_elec_tdep_vibrational_energy(self):
        agent = QElectAgent()
        phonons = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        f_vib_300 = agent.compute_tdep_vibrational_free_energy(phonons, 300.0)
        f_vib_800 = agent.compute_tdep_vibrational_free_energy(phonons, 800.0)
        self.assertIsInstance(f_vib_300, float)
        # As T increases, -T*S_vib term drives Helmholtz free energy more negative
        self.assertLess(f_vib_800, f_vib_300)

    def test_atom_dyn_gmm_ood_detection(self):
        agent = AtomDynAgent()
        nll_in, is_ood_in = agent.evaluate_gmm_ood(np.array([0.0, 0.0]))
        self.assertFalse(is_ood_in)

        nll_out, is_ood_out = agent.evaluate_gmm_ood(np.array([25.0, 30.0]))
        self.assertTrue(is_ood_out)
        self.assertGreater(nll_out, nll_in)

    def test_meso_kinetic_cgm_solute_trapping(self):
        agent = MesoKineticAgent()
        k_slow = agent.compute_cgm_solute_partitioning(equilibrium_partition_k0=0.5, solidification_velocity_m_s=0.001)
        k_fast = agent.compute_cgm_solute_partitioning(equilibrium_partition_k0=0.5, solidification_velocity_m_s=50.0)
        self.assertLess(k_slow, k_fast)
        self.assertGreater(k_fast, 0.90)
        self.assertLessEqual(k_fast, 1.0)

    def test_cont_micro_creep_and_yield(self):
        agent = ContMicroAgent()
        yield_mpa = agent.compute_taylor_homogenized_yield(crss_gpa=0.35)
        self.assertGreater(yield_mpa, 1000.0)

        creep_rate = agent.compute_high_temperature_creep_rate(applied_stress_mpa=200.0, temperature_k=1123.15)
        self.assertGreater(creep_rate, 0.0)
        self.assertLess(creep_rate, 1.0e-5)

    def test_proc_mfg_exergy(self):
        agent = ProcMfgAgent()
        comp = {"Ni": 0.60, "Al": 0.10, "Cr": 0.20, "Ti": 0.10}
        exergy = agent.compute_minimum_ore_extraction_exergy(comp)
        self.assertGreater(exergy, 30.0)
        self.assertLess(exergy, 150.0)

    def test_uq_bridge_nix_gao(self):
        agent = UqBridgeAgent()
        measured_h = 8.5
        depth = 200.0
        h0 = agent.evaluate_nix_gao_depth_correction(measured_h, depth, characteristic_length_h_star_nm=180.0)
        self.assertLess(h0, measured_h)


if __name__ == "__main__":
    unittest.main()
