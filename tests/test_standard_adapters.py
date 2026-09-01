"""Unit tests for Standard Library Adapter Layer (spglib, pymatgen, pycalphad, gudhi, pyvoro)."""

import unittest
import numpy as np

from penziv_materials.adapters.standard_adapters import (
    SymmetryAdapter,
    PhaseDiagramAdapter,
    CalphadAdapter,
    TopologyAdapter,
    ElasticityAdapter,
)


MINIMAL_NI_AL_TDB = """
ELEMENT /- ELECTRON_GAS 0.0000E+00 0.0000E+00 0.0000E+00!
ELEMENT VA VACUUM 0.0000E+00 0.0000E+00 0.0000E+00!
ELEMENT NI FCC_A1 5.8693E+01 4.7870E+03 2.9796E+01!
ELEMENT AL FCC_A1 2.6982E+01 4.5400E+03 2.8300E+01!
SPECIES NI NI1!
SPECIES AL AL1!
PHASE FCC_A1 % 1 1.0 !
CONSTITUENT FCC_A1 :AL,NI: !
PARAMETER G(FCC_A1,AL;0) 298.15 -7976.15+137.093038*T-24.3671976*T*LN(T)-.001884662*T**2-8.77664E-07*T**3+74092*T**(-1); 6000.0 N !
PARAMETER G(FCC_A1,NI;0) 298.15 -5179.15+117.854*T-22.096*T*LN(T)-.0048407*T**2; 6000.0 N !
"""


class TestStandardAdapters(unittest.TestCase):
    def test_symmetry_adapter_operations(self):
        # Test cubic space group 225 (Fm-3m)
        ops_225 = SymmetryAdapter.get_symmetry_operations(225)
        self.assertGreaterEqual(len(ops_225), 8)
        for R, t in ops_225:
            self.assertEqual(R.shape, (3, 3))
            self.assertEqual(t.shape, (3,))

        # Test space group 1 (P1)
        ops_1 = SymmetryAdapter.get_symmetry_operations(1)
        self.assertGreaterEqual(len(ops_1), 1)

    def test_symmetry_adapter_space_group_info(self):
        lattice = np.eye(3) * 4.0
        positions = np.array([[0.0, 0.0, 0.0]])
        numbers = [14]  # Silicon
        info = SymmetryAdapter.get_space_group_info(lattice, positions, numbers)
        self.assertEqual(info["backend"], "spglib")
        self.assertEqual(info["space_group_number"], 221)  # Simple cubic lattice with 1 atom = Pm-3m

    def test_symmetry_adapter_wyckoff_expansion(self):
        # Expand 1a in SG 221 (Pm-3m)
        orbit = SymmetryAdapter.expand_wyckoff_orbit(221, "a", (0.0, 0.0, 0.0))
        self.assertGreaterEqual(len(orbit), 1)
        np.testing.assert_allclose(orbit[0], [0.0, 0.0, 0.0], atol=1e-4)

    def test_phase_diagram_adapter_convex_hull(self):
        # Stable compound test: Li2O
        res = PhaseDiagramAdapter.compute_energy_above_hull("TiO2", -8.70)
        self.assertEqual(res["backend"], "pymatgen")
        self.assertIn("energy_above_hull_ev_atom", res)
        self.assertIn("is_thermodynamically_stable", res)

    def test_calphad_adapter_equilibrium(self):
        res = CalphadAdapter.evaluate_gibbs_equilibrium(
            elements=["NI", "AL"],
            phases=["FCC_A1"],
            temperature_k=1273.15,
            tdb_file_content=MINIMAL_NI_AL_TDB,
        )
        self.assertEqual(res["backend"], "pycalphad")
        self.assertIn("stable_phases", res)
        self.assertIn("gibbs_energy_j_mol", res)
        self.assertEqual(res["temperature_k"], 1273.15)

    def test_topology_adapter_betti_numbers(self):
        # Create a simple point cloud (cube corners)
        pts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ], dtype=np.float64)
        betti = TopologyAdapter.compute_persistent_betti_numbers(pts, max_edge_length=2.0)
        self.assertEqual(betti["backend"], "gudhi")
        self.assertIn("betti_0", betti)
        self.assertIn("betti_1", betti)
        self.assertIn("betti_2", betti)
        self.assertGreaterEqual(betti["betti_0"], 1)

    def test_elasticity_adapter_analysis(self):
        # Cubic Silicon stiffness matrix in GPa
        c_si = np.zeros((6, 6))
        c_si[0, 0] = c_si[1, 1] = c_si[2, 2] = 165.7
        c_si[0, 1] = c_si[0, 2] = c_si[1, 2] = 63.9
        c_si[1, 0] = c_si[2, 0] = c_si[2, 1] = 63.9
        c_si[3, 3] = c_si[4, 4] = c_si[5, 5] = 79.6

        analysis = ElasticityAdapter.analyze_elastic_tensor_6x6(c_si)
        self.assertIn("bulk_modulus_gpa", analysis)
        self.assertIn("shear_modulus_gpa", analysis)
        self.assertIn("youngs_modulus_gpa", analysis)
        self.assertIn("poissons_ratio", analysis)
        self.assertIn("born_stable", analysis)
        self.assertTrue(analysis["born_stable"])
        self.assertAlmostEqual(analysis["bulk_modulus_gpa"], (165.7 + 2 * 63.9) / 3.0, places=1)


if __name__ == "__main__":
    unittest.main()
