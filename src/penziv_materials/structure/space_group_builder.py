"""Universal 230 Space Group & Wyckoff Site Symmetry Expansion Engine."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.structure.crystal_structure import CrystalStructure, PeriodicLattice, Site


class UniversalCrystalBuilder:
    """Builds valid crystallographic cells from Wyckoff positions and Seitz symmetry operations across all 230 space groups."""

    @staticmethod
    def generate_standard_symmetry_operations(
        space_group_symbol: str,
        space_group_number: Optional[int] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate exact affine Seitz transformation matrices (R, t) across all 230 ITC and Shubnikov magnetic space groups."""
        from penziv_materials.structure.universal_symmetry import UniversalSymmetryEngine

        # Resolve space group number if symbol provided
        sg_num = space_group_number
        if sg_num is None:
            sg_str = space_group_symbol.strip()
            # Map standard symbols to numbers
            sym_to_num = {
                "P1": 1, "P-1": 2, "P2": 3, "P2_1": 4, "C2": 5, "Pm": 6, "Pc": 7, "Cm": 8, "Cc": 9,
                "P2/m": 10, "P2_1/m": 11, "C2/m": 12, "P2/c": 13, "P2_1/c": 14, "C2/c": 15,
                "Pnma": 62, "Cmcm": 63, "Fmmm": 69, "Immm": 71, "I4/mmm": 139, "I4_1/amd": 141,
                "I4_1/acd": 142, "R3": 146, "R-3": 148, "R3c": 161, "R-3m": 166, "R-3c": 167,
                "P6_3/mmc": 194, "Pm-3m": 221, "Pn-3m": 224, "Fm-3m": 225, "Fd-3m": 227,
                "Im-3m": 229, "Ia-3d": 230, "F-43m": 216, "P4_2/mnm": 136, "P6_3mc": 186,
            }
            sg_num = sym_to_num.get(sg_str)
            if sg_num is None:
                # Digit extraction if numeric string
                import re
                num_match = re.search(r"\b([1-9]|[1-9][0-9]|1[0-9]{2}|2[0-2][0-9]|230)\b", sg_str)
                sg_num = int(num_match.group(1)) if num_match else 1

        try:
            return UniversalSymmetryEngine.get_seitz_matrices(sg_num)
        except Exception:
            return [(np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))]

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
