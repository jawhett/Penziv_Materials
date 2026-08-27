"""Unit tests for crystallographic structure container, lattice transforms, and CIF export."""

import unittest
import numpy as np

from penziv_materials.structure.crystal_structure import PeriodicLattice, Site, CrystalStructure
from penziv_materials.scale4_atomistic.equivariant_mlip import EquivariantMLIPEngine


class TestCrystalStructure(unittest.TestCase):
    def setUp(self):
        self.lattice = PeriodicLattice.from_parameters(a=5.20, b=5.20, c=5.20, alpha_deg=90.0, beta_deg=90.0, gamma_deg=90.0)
        self.sites = [
            Site(species="Mg", fractional_coords=np.array([0.0, 0.0, 0.0])),
            Site(species="Sc", fractional_coords=np.array([0.5, 0.5, 0.5])),
            Site(species="S",  fractional_coords=np.array([0.25, 0.25, 0.25])),
            Site(species="S",  fractional_coords=np.array([0.75, 0.75, 0.75])),
        ]
        self.crystal = CrystalStructure(self.lattice, self.sites, space_group="Fm-3m", space_group_number=225)

    def test_fractional_to_cartesian(self):
        cart = self.crystal.cartesian_coords
        self.assertEqual(cart.shape, (4, 3))
        self.assertAlmostEqual(cart[1, 0], 2.60)

    def test_voronoi_bottleneck_radius(self):
        r_bottleneck = self.crystal.compute_voronoi_bottleneck_radius(mobile_carrier_species="Mg")
        self.assertGreater(r_bottleneck, 0.0)

    def test_cif_export(self):
        cif_str = self.crystal.to_cif_string("Mg-Sc-S-Spinel")
        self.assertIn("data_Mg-Sc-S-Spinel", cif_str)
        self.assertIn("_cell_length_a                   5.200000", cif_str)
        self.assertIn("Mg", cif_str)

    def test_mlip_relaxation_and_elastic_tensor(self):
        mlip = EquivariantMLIPEngine()
        relaxed_crystal, energy, conv = mlip.relax_crystal_structure(self.crystal, max_steps=10)
        self.assertEqual(relaxed_crystal.num_sites, 4)

        c_ij = mlip.compute_elastic_stiffness_tensor(relaxed_crystal)
        self.assertEqual(c_ij.shape, (6, 6))
        self.assertGreater(c_ij[0, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
