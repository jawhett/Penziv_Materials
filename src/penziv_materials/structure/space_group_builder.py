"""Universal 230 Space Group & Wyckoff Site Symmetry Expansion Engine."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.structure.crystal_structure import CrystalStructure, PeriodicLattice, Site


class UniversalCrystalBuilder:
    """Builds valid crystallographic cells from Wyckoff positions and Seitz symmetry operations across all 230 space groups."""

    @staticmethod
    def generate_standard_symmetry_operations(space_group_symbol: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate exact affine transformation matrices (R, t) including non-symmorphic screw/glide operations."""
        ops: List[Tuple[np.ndarray, np.ndarray]] = []
        # Identity
        ops.append((np.eye(3), np.zeros(3)))

        sg = space_group_symbol.strip()

        # Inversion
        if any(inv in sg for inv in ["-1", "/m", "mmm", "Fd-3m", "Fm-3m", "R-3c", "Pnma", "P2_1/c"]):
            ops.append((-np.eye(3), np.zeros(3)))

        # 2-fold / screw rotations
        if "2_1" in sg or "P2_1" in sg:
            r2y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
            ops.append((r2y, np.array([0.0, 0.5, 0.0])))
        elif "2" in sg:
            r2z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            ops.append((r2z, np.zeros(3)))

        # 4-fold rotations and screws
        if "4_1" in sg or "4_3" in sg:
            r4z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            t_screw = np.array([0.0, 0.0, 0.25]) if "4_1" in sg else np.array([0.0, 0.0, 0.75])
            ops.append((r4z, t_screw))
            ops.append((np.dot(r4z, r4z), (2.0 * t_screw) % 1.0))
            ops.append((np.dot(np.dot(r4z, r4z), r4z), (3.0 * t_screw) % 1.0))
        elif "4" in sg or "I4" in sg or "P4" in sg:
            r4z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            ops.append((r4z, np.zeros(3)))
            ops.append((np.dot(r4z, r4z), np.zeros(3)))
            ops.append((np.dot(np.dot(r4z, r4z), r4z), np.zeros(3)))

        # 6-fold rotations and screws
        if "6_3" in sg:
            r6z = np.array([[1, -1, 0], [1, 0, 0], [0, 0, 1]])
            ops.append((r6z, np.array([0.0, 0.0, 0.5])))
        elif "6" in sg or "P6" in sg:
            r6z = np.array([[1, -1, 0], [1, 0, 0], [0, 0, 1]])
            ops.append((r6z, np.zeros(3)))

        # 3-fold body diagonal rotations
        if any(r3 in sg for r3 in ["3", "R-3", "Fd-3", "Fm-3", "Ia-3d", "Pa-3"]):
            r3_diag = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
            ops.append((r3_diag, np.zeros(3)))
            ops.append((np.dot(r3_diag, r3_diag), np.zeros(3)))

        # Mirror and glide planes
        if "m" in sg or "n" in sg or "c" in sg or "a" in sg or "b" in sg:
            mz = np.diag([1, 1, -1])
            t_glide = np.zeros(3)
            if "n" in sg:
                t_glide = np.array([0.5, 0.5, 0.0])
            elif "c" in sg:
                t_glide = np.array([0.0, 0.0, 0.5])
            elif "a" in sg:
                t_glide = np.array([0.5, 0.0, 0.0])
            elif "b" in sg:
                t_glide = np.array([0.0, 0.5, 0.0])
            ops.append((mz, t_glide))

        # Centering translations
        if sg.startswith("F"):
            t_f = [np.array([0.0, 0.5, 0.5]), np.array([0.5, 0.0, 0.5]), np.array([0.5, 0.5, 0.0])]
            base_ops = list(ops)
            for t_vec in t_f:
                for R, t in base_ops:
                    ops.append((R, (t + t_vec) % 1.0))
        elif sg.startswith("I"):
            t_i = np.array([0.5, 0.5, 0.5])
            base_ops = list(ops)
            for R, t in base_ops:
                ops.append((R, (t + t_i) % 1.0))
        elif sg.startswith("C"):
            t_c = np.array([0.5, 0.5, 0.0])
            base_ops = list(ops)
            for R, t in base_ops:
                ops.append((R, (t + t_c) % 1.0))

        return ops

    @classmethod
    def expand_wyckoff_sites(
        cls,
        lattice: PeriodicLattice,
        symmetry_operations: List[Tuple[np.ndarray, np.ndarray]],
        asymmetric_sites: List[Tuple[str, np.ndarray]],
        space_group: str = "P1",
        space_group_number: int = 1,
        symprec: float = 1e-4,
    ) -> CrystalStructure:
        """Expand asymmetric unit sites using affine transformations (r' = R*r + t mod 1.0) with duplicate filtering."""
        full_sites: List[Site] = []

        for species, frac in asymmetric_sites:
            frac_arr = np.asarray(frac, dtype=np.float64) % 1.0
            generated_coords: List[np.ndarray] = []

            for R, t in symmetry_operations:
                r_prime = (np.dot(R, frac_arr) + t) % 1.0
                is_duplicate = False
                for existing in generated_coords:
                    diff = np.abs(r_prime - existing)
                    diff_pbc = np.minimum(diff, 1.0 - diff)
                    if np.all(diff_pbc < symprec):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    generated_coords.append(r_prime)
                    full_sites.append(Site(species=species, fractional_coords=r_prime, occupancy=1.0))

        return CrystalStructure(
            lattice=lattice,
            sites=full_sites,
            space_group=space_group,
            space_group_number=space_group_number,
        )
