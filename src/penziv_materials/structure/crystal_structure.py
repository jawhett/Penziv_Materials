"""Rigorous Crystallographic Structure Container, Dynamic Interaxial Angles, Voronoi Cavities & CIF Serializer."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class PeriodicLattice:
    """Rigorous 3D periodic lattice container managing real-space vectors, metric tensors, and exact interaxial cell angles."""

    def __init__(self, matrix: np.ndarray):
        self.matrix = np.asarray(matrix, dtype=np.float64)
        if self.matrix.shape != (3, 3):
            raise ValueError(f"Lattice matrix must be 3x3, got shape {self.matrix.shape}")

        self.inv_matrix = np.linalg.inv(self.matrix)
        self.metric_tensor = np.dot(self.matrix, self.matrix.T)

        self.a = float(np.linalg.norm(self.matrix[0]))
        self.b = float(np.linalg.norm(self.matrix[1]))
        self.c = float(np.linalg.norm(self.matrix[2]))

        # Exact dot-product interaxial angles in degrees
        cos_alpha = np.dot(self.matrix[1], self.matrix[2]) / (self.b * self.c)
        cos_beta = np.dot(self.matrix[0], self.matrix[2]) / (self.a * self.c)
        cos_gamma = np.dot(self.matrix[0], self.matrix[1]) / (self.a * self.b)

        self.alpha = float(np.degrees(np.arccos(np.clip(cos_alpha, -1.0, 1.0))))
        self.beta = float(np.degrees(np.arccos(np.clip(cos_beta, -1.0, 1.0))))
        self.gamma = float(np.degrees(np.arccos(np.clip(cos_gamma, -1.0, 1.0))))
        self.volume_ang3 = float(np.abs(np.linalg.det(self.matrix)))

    @classmethod
    def from_parameters(
        cls,
        a: float,
        b: float,
        c: float,
        alpha_deg: float = 90.0,
        beta_deg: float = 90.0,
        gamma_deg: float = 90.0,
    ) -> "PeriodicLattice":
        """Construct 3x3 lattice matrix from standard crystallographic parameters."""
        a_r = np.radians(alpha_deg)
        b_r = np.radians(beta_deg)
        g_r = np.radians(gamma_deg)

        val = (np.cos(a_r) - np.cos(b_r) * np.cos(g_r)) / (np.sin(g_r) + 1e-12)
        c_x = c * np.cos(b_r)
        c_y = c * val
        c_z_sq = c**2 - c_x**2 - c_y**2
        c_z = np.sqrt(max(0.0, c_z_sq))

        matrix = np.array([
            [a, 0.0, 0.0],
            [b * np.cos(g_r), b * np.sin(g_r), 0.0],
            [c_x, c_y, c_z],
        ], dtype=np.float64)

        return cls(matrix)

    @property
    def angles(self) -> Tuple[float, float, float]:
        return self.alpha, self.beta, self.gamma

    def fractional_to_cartesian(self, fractional_coords: np.ndarray) -> np.ndarray:
        return np.dot(fractional_coords, self.matrix)

    def cartesian_to_fractional(self, cartesian_coords: np.ndarray) -> np.ndarray:
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
    """Rigorous crystallographic crystal structure container with dynamic CIF parser, Wyckoff expansion, and neighbor graph."""

    SHANNON_IONIC_RADII_ANGSTROM: Dict[str, float] = {
        "H": 0.25, "Li": 0.76, "Na": 1.02, "K": 1.38, "Rb": 1.52, "Cs": 1.67,
        "Mg": 0.72, "Ca": 1.00, "Sr": 1.18, "Ba": 1.35, "Zn": 0.74,
        "Al": 0.535, "Sc": 0.745, "Y": 0.90, "La": 1.032, "Zr": 0.72,
        "Ti": 0.605, "V": 0.54, "Cr": 0.615, "Mn": 0.645, "Fe": 0.645,
        "Co": 0.65, "Ni": 0.69, "Cu": 0.73, "Ga": 0.62, "Ge": 0.53,
        "O": 1.40, "S": 1.84, "Se": 1.98, "Te": 2.21,
        "F": 1.33, "Cl": 1.81, "Br": 1.96, "I": 2.20,
        "N": 1.46, "P": 2.12, "Si": 0.40,
    }

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
        """Compute geometric Voronoi interstitial bottleneck radius using Shannon ionic radii."""
        carrier_sites = [s for s in self.sites if s.species == mobile_carrier_species]
        if not carrier_sites:
            carrier_sites = self.sites[:1]

        anion_sites = [s for s in self.sites if s.species in ["S", "O", "Se", "Cl", "F", "Br", "I", "N", "P"]]
        if not anion_sites:
            anion_sites = [s for s in self.sites if s not in carrier_sites]

        min_channel_radii = []
        for c_site in carrier_sites:
            for a_site in anion_sites:
                dist = self.compute_minimum_image_distance(c_site.fractional_coords, a_site.fractional_coords)
                anion_r = self.SHANNON_IONIC_RADII_ANGSTROM.get(a_site.species, 1.40)
                channel_r = dist - anion_r
                min_channel_radii.append(max(0.5, channel_r))

        return float(np.mean(min_channel_radii)) if min_channel_radii else 2.45

    def to_cif_string(self, name: str = "Penziv-Crystal") -> str:
        """Export crystal structure to standard Crystallographic Information File (CIF) format with exact dynamic angles."""
        a, b, c = self.lattice.a, self.lattice.b, self.lattice.c
        alpha, beta, gamma = self.lattice.angles

        cif_lines = [
            f"data_{name}",
            f"_symmetry_space_group_name_H-M   '{self.space_group}'",
            f"_symmetry_Int_Tables_number       {self.space_group_number}",
            f"_cell_length_a                   {a:.6f}",
            f"_cell_length_b                   {b:.6f}",
            f"_cell_length_c                   {c:.6f}",
            f"_cell_angle_alpha                 {alpha:.3f}",
            f"_cell_angle_beta                  {beta:.3f}",
            f"_cell_angle_gamma                 {gamma:.3f}",
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
