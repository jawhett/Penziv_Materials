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


class TestStandardAdapters(unittest.TestCase):
    def test_symmetry_adapter_operations(self):
        # Test cubic space group 225 (Fm-3m)
        ops_225 = SymmetryAdapter.get_symmetry_operations(225)
        self.assertGreaterEqual(len(ops_225), 24)
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
        self.assertIn("space_group_number", info)
        self.assertIn("backend", info)
        self.assertGreater(info["space_group_number"], 0)

    def test_symmetry_adapter_wyckoff_expansion(self):
        # Expand 1a in SG 221 (Pm-3m)
        orbit = SymmetryAdapter.expand_wyckoff_orbit(221, "a", (0.0, 0.0, 0.0))
        self.assertGreaterEqual(len(orbit), 1)
        np.testing.assert_allclose(orbit[0], [0.0, 0.0, 0.0], atol=1e-4)

    def test_phase_diagram_adapter_convex_hull(self):
        # Stable compound test: Li2O
        res = PhaseDiagramAdapter.compute_energy_above_hull("Li2O", -2.05)
        self.assertIn("energy_above_hull_ev_atom", res)
        self.assertIn("is_thermodynamically_stable", res)
        self.assertIn("backend", res)

    def test_calphad_adapter_equilibrium(self):
        res = CalphadAdapter.evaluate_gibbs_equilibrium(
            elements=["Ni", "Al"],
            phases=["FCC_A1", "GAMMA_PRIME"],
            temperature_k=1273.15,
        )
        self.assertIn("stable_phases", res)
        self.assertIn("temperature_k", res)
        self.assertEqual(res["temperature_k"], 1273.15)
        self.assertIn("backend", res)

    def test_topology_adapter_betti_numbers(self):
        # Create a simple point cloud (cube corners)
        pts = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
        ], dtype=np.float64)
        betti = TopologyAdapter.compute_persistent_betti_numbers(pts, max_edge_length=2.0)
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
