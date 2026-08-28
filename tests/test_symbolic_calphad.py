"""Unit tests for the Symbolic CALPHAD AST Engine & Grand-Potential Phase-Field Coupling."""

import unittest
import numpy as np
from penziv_materials.thermodynamics.opencalphad_tdb import OpenCALPHADTDBEngine, CALPHADFunctionAST
from penziv_materials.scale3_mesoscale.calphad_grand_potential import CALPHADGrandPotentialPhaseFieldEngine


SAMPLE_TDB_CONTENT = """
$ Sample Thermodynamic Database for Fe-Cr
ELEMENT FE BCC_A2 5.58470E+01 4.48900E+03 2.72800E+01 !
ELEMENT CR BCC_A2 5.19960E+01 4.05000E+03 2.36000E+01 !

FUNCTION GHSERFE 298.15 +1224.7 + 124.134*T - 23.5143*T*LN(T) - 0.00439*T**2 + 77358.5/T ; 1811.0 Y !
FUNCTION GHSERCR 298.15 -8856.94 + 157.48*T - 26.908*T*LN(T) + 0.00189*T**2 + 10000.0/T ; 2180.0 Y !

PHASE BCC_A2 % 1 1.0 !
CONSTITUENT BCC_A2 : FE, CR : !

PARAMETER G(BCC_A2,FE;0) 298.15 +1224.7 + 124.134*T - 23.5143*T*LN(T) ; 6000.0 N !
PARAMETER G(BCC_A2,CR;0) 298.15 -8856.94 + 157.48*T - 26.908*T*LN(T) ; 6000.0 N !
PARAMETER L(BCC_A2,FE,CR;0) 298.15 +20500 - 9.68*T ; 6000.0 N !
PARAMETER TC(BCC_A2,FE;0) 298.15 1043.0 ; 6000.0 N !
PARAMETER BMAGN(BCC_A2,FE;0) 298.15 2.22 ; 6000.0 N !

PHASE FCC_A1 % 1 1.0 !
CONSTITUENT FCC_A1 : FE, CR : !
PARAMETER G(FCC_A1,FE;0) 298.15 -1462.4 + 129.2*T - 24.6*T*LN(T) ; 6000.0 N !
PARAMETER G(FCC_A1,CR;0) 298.15 +7200.0 + 150.0*T - 25.0*T*LN(T) ; 6000.0 N !
PARAMETER L(FCC_A1,FE,CR;0) 298.15 +10800 - 4.5*T ; 6000.0 N !
"""


class TestSymbolicCALPHAD(unittest.TestCase):
    def setUp(self):
        self.engine = OpenCALPHADTDBEngine()
        self.parse_info = self.engine.parse_tdb_content(SAMPLE_TDB_CONTENT)

    def test_ast_evaluation(self):
        fn = CALPHADFunctionAST("+1224.7 + 124.134*T - 23.5143*T*LN(T)")
        val_300 = fn.evaluate(300.0)
        # 1224.7 + 124.134*300 - 23.5143*300*ln(300) = 1224.7 + 37240.2 - 23.5143*300*5.70378 = 38464.9 - 40236.4 = -1771.5
        self.assertAlmostEqual(val_300, -1771.5, delta=50.0)

    def test_tdb_parsing_and_sublattices(self):
        self.assertTrue(self.parse_info["is_tdb_valid"])
        self.assertIn("FE", self.parse_info["parsed_elements"])
        self.assertIn("CR", self.parse_info["parsed_elements"])
        self.assertIn("BCC_A2", self.parse_info["parsed_phases"])
        self.assertIn("FCC_A1", self.parse_info["parsed_phases"])

    def test_evaluate_phase_gibbs_energy(self):
        g_bcc = self.engine.evaluate_phase_gibbs_energy("BCC_A2", {"FE": 0.80, "CR": 0.20}, temperature_k=1000.0)
        g_fcc = self.engine.evaluate_phase_gibbs_energy("FCC_A1", {"FE": 0.80, "CR": 0.20}, temperature_k=1000.0)
        self.assertIsInstance(g_bcc, float)
        self.assertIsInstance(g_fcc, float)
        # Gibbs energy should be negative at 1000K due to -T*S terms
        self.assertLess(g_bcc, 0.0)

    def test_chemical_potentials_and_hessian(self):
        mu_dict, H = self.engine.evaluate_chemical_potentials_and_hessian("BCC_A2", {"FE": 0.50, "CR": 0.50}, temperature_k=1000.0)
        self.assertEqual(len(mu_dict), 2)
        self.assertEqual(H.shape, (2, 2))
        # Hessian diagonal must be positive for thermodynamic stability
        self.assertGreater(H[0, 0], 0.0)
        self.assertGreater(H[1, 1], 0.0)

    def test_grand_potential_phase_field_coupling(self):
        pf_engine = CALPHADGrandPotentialPhaseFieldEngine(
            num_phases=2,
            grid_shape=(8, 8, 8),
            temperature_k=1000.0,
            calphad_engine=self.engine,
        )
        mu_dict = {"FE": -50000.0, "CR": -45000.0}
        omega = pf_engine.compute_calphad_grand_potentials(mu_dict, ["BCC_A2", "FCC_A1"])
        self.assertEqual(len(omega), 2)
        self.assertIsInstance(float(omega[0]), float)
        self.assertIsInstance(float(omega[1]), float)


if __name__ == "__main__":
    unittest.main()
