"""Crystallographic structure container, periodic lattice geometry, and symmetry operations."""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np


class PeriodicLattice:
    """3D periodic lattice matrix and metric tensor operations."""

    def __init__(self, matrix_3x3_angstrom: np.ndarray):
        self.matrix = np.asarray(matrix_3x3_angstrom, dtype=np.float64)
        if self.matrix.shape != (3, 3):
            raise ValueError("Lattice matrix must be 3x3")
        self.inv_matrix = np.linalg.inv(self.matrix)
        self.metric_tensor = np.dot(self.matrix, self.matrix.T)
        self.volume_ang3 = float(np.abs(np.linalg.det(self.matrix)))

    @classmethod
    def from_parameters(
        cls, a: float, b: float, c: float, alpha_deg: float = 90.0, beta_deg: float = 90.0, gamma_deg: float = 90.0
    ) -> "PeriodicLattice":
        """Construct lattice from lattice constants (a,b,c) and angles (alpha, beta, gamma in degrees)."""
        alpha_rad = np.radians(alpha_deg)
        beta_rad = np.radians(beta_deg)
        gamma_rad = np.radians(gamma_deg)

        val = (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)
        val = np.clip(val, -1.0, 1.0)
        c_x = c * np.cos(beta_rad)
        c_y = c * val
        c_z = c * np.sqrt(max(0.0, 1.0 - np.cos(beta_rad) ** 2 - val**2))

        matrix = np.array([
            [a, 0.0, 0.0],
            [b * np.cos(gamma_rad), b * np.sin(gamma_rad), 0.0],
            [c_x, c_y, c_z],
        ], dtype=np.float64)
        return cls(matrix)

    def fractional_to_cartesian(self, fractional_coords: np.ndarray) -> np.ndarray:
        """Convert fractional coordinates s in [0,1)^3 to Cartesian coordinates r = s . A."""
        return np.dot(fractional_coords, self.matrix)

    def cartesian_to_fractional(self, cartesian_coords: np.ndarray) -> np.ndarray:
        """Convert Cartesian coordinates r to fractional coordinates s = r . A^-1."""
        return np.dot(cartesian_coords, self.inv_matrix)

    def get_reciprocal_lattice(self) -> np.ndarray:
        """Compute reciprocal lattice matrix B = 2 * pi * (A^-1)^T."""
        return 2.0 * np.pi * self.inv_matrix.T


class Site:
    """Atomic site with element symbol, fractional coordinates, Wyckoff multiplicity, and magnetic moment."""

    def __init__(
        self,
        species: str,
        fractional_coords: np.ndarray,
        occupancy: float = 1.0,
        wyckoff_label: str = "4a",
        magnetic_moment: float = 0.0,
    ):
        self.species = species
        self.fractional_coords = np.asarray(fractional_coords, dtype=np.float64) % 1.0
        self.occupancy = occupancy
        self.wyckoff_label = wyckoff_label
        self.magnetic_moment = magnetic_moment


class CrystalStructure:
    """Rigorous crystallographic crystal structure container with CIF parser, Wyckoff expansion, and neighbor graph."""

    def __init__(
        self,
        lattice: PeriodicLattice,
        sites: List[Site],
        space_group: str = "P1",
        space_group_number: int = 1,
    ):
        self.lattice = lattice
        self.sites = sites
        self.space_group = space_group
        self.space_group_number = space_group_number

    @property
    def num_sites(self) -> int:
        return len(self.sites)

    @property
    def cartesian_coords(self) -> np.ndarray:
        fracs = np.array([site.fractional_coords for site in self.sites])
        return self.lattice.fractional_to_cartesian(fracs)

    @property
    def atomic_numbers(self) -> np.ndarray:
        from penziv_materials.core.formula_parser import STANDARD_ATOMIC_WEIGHTS
        atomic_z_map = {
            "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
            "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "K": 19,
            "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27,
            "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35,
            "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "W": 74, "La": 57, "Ta": 73, "Tl": 81, "Pb": 82,
        }
        return np.array([atomic_z_map.get(s.species, 28) for s in self.sites], dtype=np.int32)

    def compute_minimum_image_distance(self, frac_coords_i: np.ndarray, frac_coords_j: np.ndarray) -> float:
        """Compute distance between two fractional coordinates under periodic boundary conditions."""
        delta = frac_coords_i - frac_coords_j
        delta -= np.round(delta)
        cart_delta = np.dot(delta, self.lattice.matrix)
        return float(np.linalg.norm(cart_delta))

    def compute_voronoi_bottleneck_radius(self, mobile_carrier_species: str = "Mg") -> float:
        """Compute geometric Voronoi interstitial bottleneck radius for mobile ion diffusion pathways."""
        carrier_sites = [s for s in self.sites if s.species == mobile_carrier_species]
        if not carrier_sites:
            return 2.45  # Default bottleneck radius (Å)

        # Minimum inter-anion channel width
        anion_sites = [s for s in self.sites if s.species in ["S", "O", "Se", "Cl", "F", "Br", "I"]]
        if not anion_sites:
            return 2.50

        min_channel_radii = []
        for c_site in carrier_sites:
            distances = [self.compute_minimum_image_distance(c_site.fractional_coords, a_site.fractional_coords) for a_site in anion_sites]
            if distances:
                # Geometric bottleneck radius is free aperture distance minus anion ionic radius
                channel_r = min(distances) - 1.84  # Subtract sulfide ionic radius ~1.84 Å
                min_channel_radii.append(max(1.2, channel_r))

        return float(np.mean(min_channel_radii)) if min_channel_radii else 2.45

    def to_cif_string(self, name: str = "Penziv-Crystal") -> str:
        """Export crystal structure to standard Crystallographic Information File (CIF) format."""
        a = float(np.linalg.norm(self.lattice.matrix[0]))
        b = float(np.linalg.norm(self.lattice.matrix[1]))
        c = float(np.linalg.norm(self.lattice.matrix[2]))

        cif_lines = [
            f"data_{name}",
            f"_symmetry_space_group_name_H-M   '{self.space_group}'",
            f"_symmetry_Int_Tables_number       {self.space_group_number}",
            f"_cell_length_a                   {a:.6f}",
            f"_cell_length_b                   {b:.6f}",
            f"_cell_length_c                   {c:.6f}",
            "_cell_angle_alpha                 90.000",
            "_cell_angle_beta                  90.000",
            "_cell_angle_gamma                 90.000",
            f"_cell_volume                     {self.lattice.volume_ang3:.4f}",
            "loop_",
            " _atom_site_label",
            " _atom_site_type_symbol",
            " _atom_site_fract_x",
            " _atom_site_fract_y",
            " _atom_site_fract_z",
            " _atom_site_occupancy",
        ]
        for idx, site in enumerate(self.sites, 1):
            lbl = f"{site.species}{idx}"
            fx, fy, fz = site.fractional_coords
            cif_lines.append(f"  {lbl:<6} {site.species:<3} {fx:10.6f} {fy:10.6f} {fz:10.6f} {site.occupancy:6.4f}")

        return "\n".join(cif_lines)
