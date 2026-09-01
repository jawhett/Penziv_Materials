"""Unit tests for PyCALPHAD Adapter & Grand-Potential Phase-Field Coupling."""

import unittest
import numpy as np
from penziv_materials.adapters.standard_adapters import CalphadAdapter
from penziv_materials.scale3_mesoscale.calphad_grand_potential import CALPHADGrandPotentialPhaseFieldEngine


SAMPLE_TDB_CONTENT = """
ELEMENT /- ELECTRON_GAS 0.0000E+00 0.0000E+00 0.0000E+00!
ELEMENT VA VACUUM 0.0000E+00 0.0000E+00 0.0000E+00!
ELEMENT FE BCC_A2 5.58470E+01 4.48900E+03 2.72800E+01 !
ELEMENT CR BCC_A2 5.19960E+01 4.05000E+03 2.36000E+01 !
SPECIES FE FE1!
SPECIES CR CR1!
PHASE BCC_A2 % 1 1.0 !
CONSTITUENT BCC_A2 : FE, CR : !
PARAMETER G(BCC_A2,FE;0) 298.15 +1224.7 + 124.134*T - 23.5143*T*LN(T) ; 6000.0 N !
PARAMETER G(BCC_A2,CR;0) 298.15 -8856.94 + 157.48*T - 26.908*T*LN(T) ; 6000.0 N !
"""


class TestSymbolicCALPHAD(unittest.TestCase):
    def test_calphad_adapter_equilibrium(self):
        eq_res = CalphadAdapter.evaluate_gibbs_equilibrium(
            elements=["FE", "CR"],
            phases=["BCC_A2"],
            temperature_k=1000.0,
            tdb_file_content=SAMPLE_TDB_CONTENT,
        )
        self.assertEqual(eq_res["backend"], "pycalphad")
        self.assertIn("stable_phases", eq_res)
        self.assertIn("gibbs_energy_j_mol", eq_res)

    def test_grand_potential_phase_field_coupling(self):
        pf_engine = CALPHADGrandPotentialPhaseFieldEngine(
            num_phases=2,
            grid_shape=(8, 8, 8),
            temperature_k=1000.0,
        )
        mu_vec = np.array([-50000.0, -45000.0])
        omega = pf_engine.compute_calphad_grand_potentials(mu_vec)
        self.assertEqual(len(omega), 2)
        self.assertIsInstance(float(omega[0]), float)
        self.assertIsInstance(float(omega[1]), float)


if __name__ == "__main__":
    unittest.main()

