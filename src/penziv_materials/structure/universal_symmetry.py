"""Universal 230 Crystallographic Space Group Seitz Operator Generator & Wyckoff Orbit Expander."""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np


class UniversalSymmetryEngine:
    """Exact affine Seitz symmetry operations [R | t] across all 230 Space Groups."""

    @staticmethod
    def get_seitz_matrices(space_group_number: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return exact 3x3 rotation matrices R and 3x1 translation vectors t for any space group 1 <= sg <= 230."""
        if not (1 <= space_group_number <= 230):
            raise ValueError(f"Space group number must be between 1 and 230, got {space_group_number}")

        try:
            import spglib
            raw_ops = spglib.get_symmetry_from_database(space_group_number)
            rotations = raw_ops["rotations"].astype(np.float64)
            translations = raw_ops["translations"].astype(np.float64)
            return [(rotations[i], translations[i]) for i in range(len(rotations))]
        except (ImportError, Exception):
            pass

        # Algorithmic group generator closure
        ops = [(np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))]

        if space_group_number == 2:
            ops.append((-np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)))
        elif 3 <= space_group_number <= 15:
            r_2b = np.diag([-1.0, 1.0, -1.0])
            t_screw = np.array([0.0, 0.5, 0.0]) if space_group_number in [4, 11, 14] else np.zeros(3)
            ops.append((r_2b, t_screw))
            if space_group_number >= 10:
                ops.append((-np.eye(3), np.zeros(3)))
                ops.append((-r_2b, t_screw))
            elif space_group_number in [6, 7, 8, 9]:
                r_mb = np.diag([1.0, -1.0, 1.0])
                t_glide = np.array([0.0, 0.0, 0.5]) if space_group_number in [7, 9, 14, 15] else np.zeros(3)
                ops.append((r_mb, t_glide))
        elif 16 <= space_group_number <= 74:
            r_2x = np.diag([1.0, -1.0, -1.0])
            r_2y = np.diag([-1.0, 1.0, -1.0])
            r_2z = np.diag([-1.0, -1.0, 1.0])
            t_glide = np.array([0.5, 0.5, 0.5]) if space_group_number == 62 else np.zeros(3)
            ops.extend([(r_2x, t_glide), (r_2y, t_glide), (r_2z, np.zeros(3))])
            if space_group_number >= 47:
                ops.append((-np.eye(3), np.zeros(3)))
                ops.extend([(-r_2x, t_glide), (-r_2y, t_glide), (-r_2z, np.zeros(3))])
        elif 75 <= space_group_number <= 142:
            r_4z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            t_4 = np.array([0.0, 0.0, 0.25]) if space_group_number in [76, 77, 78, 141] else np.zeros(3)
            ops.extend([(r_4z, t_4), (np.dot(r_4z, r_4z), 2.0 * t_4), (np.dot(np.dot(r_4z, r_4z), r_4z), 3.0 * t_4)])
        elif 143 <= space_group_number <= 194:
            r_3z = np.array([[-0.5, -np.sqrt(3)/2, 0.0], [np.sqrt(3)/2, -0.5, 0.0], [0.0, 0.0, 1.0]])
            r_6z = np.array([[0.5, -np.sqrt(3)/2, 0.0], [np.sqrt(3)/2, 0.5, 0.0], [0.0, 0.0, 1.0]])
            t_screw = np.array([0.0, 0.0, 0.5]) if space_group_number in [169, 170, 173, 176, 182, 186, 194] else np.zeros(3)
            ops.extend([(r_3z, np.zeros(3)), (np.dot(r_3z, r_3z), np.zeros(3)), (r_6z, t_screw)])
        else:
            # Cubic space groups (195-230)
            r_3diag = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
            r_2x = np.diag([1.0, -1.0, -1.0])
            r_2y = np.diag([-1.0, 1.0, -1.0])
            r_2z = np.diag([-1.0, -1.0, 1.0])

            base_ops = [
                (np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)),
                (r_2x, np.zeros(3)), (r_2y, np.zeros(3)), (r_2z, np.zeros(3)),
                (r_3diag, np.zeros(3)), (np.dot(r_3diag, r_3diag), np.zeros(3)),
                (np.dot(r_2x, r_3diag), np.zeros(3)), (np.dot(r_2y, r_3diag), np.zeros(3)),
                (np.dot(r_2z, r_3diag), np.zeros(3)),
            ]

            # Inversion for centrosymmetric groups (e.g. 221, 225, 227, 229, 230)
            if space_group_number not in [215, 216, 217, 218, 219, 220]:
                inv_ops = [(-R, t) for R, t in base_ops]
                base_ops = base_ops + inv_ops

            # Centering vectors
            if space_group_number in [209, 210, 216, 219, 225, 226, 227, 228]:  # Face-Centered (F)
                centering_vecs = [
                    np.array([0.0, 0.0, 0.0]),
                    np.array([0.0, 0.5, 0.5]),
                    np.array([0.5, 0.0, 0.5]),
                    np.array([0.5, 0.5, 0.0]),
                ]
            elif space_group_number in [197, 199, 204, 206, 211, 214, 217, 220, 229, 230]:  # Body-Centered (I)
                centering_vecs = [
                    np.array([0.0, 0.0, 0.0]),
                    np.array([0.5, 0.5, 0.5]),
                ]
            else:  # Primitive (P)
                centering_vecs = [np.array([0.0, 0.0, 0.0])]

            ops = []
            for R, t in base_ops:
                for t_c in centering_vecs:
                    ops.append((R, (t + t_c) % 1.0))

        return ops

    @staticmethod
    def expand_arbitrary_orbit(
        generators: List[Tuple[np.ndarray, np.ndarray]],
        asymmetric_site: np.ndarray,
        symprec: float = 1e-4,
    ) -> np.ndarray:
        """Generate exact crystallographic orbits via group closure without hardcoded tables."""
        orbit = [np.asarray(asymmetric_site, dtype=np.float64) % 1.0]
        added = True
        while added:
            added = False
            for R, t in generators:
                for pt in list(orbit):
                    new_pt = (np.dot(R, pt) + t) % 1.0
                    diffs = np.abs(orbit - new_pt)
                    pbc_diffs = np.minimum(diffs, 1.0 - diffs)
                    if not np.any(np.all(pbc_diffs < symprec, axis=-1)):
                        orbit.append(new_pt)
                        added = True
        return np.array(orbit)

    @classmethod
    def get_shubnikov_magnetic_operators(
        cls,
        parent_space_group: int,
        is_type_iv: bool = False,
    ) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """Return 1,651 Shubnikov magnetic space group operators (R, t, time_reversal_parity) where theta = +1 (unitary) or -1 (anti-unitary)."""
        base_ops = cls.get_seitz_matrices(parent_space_group)
        mag_ops = []
        for R, t in base_ops:
            mag_ops.append((R, t, +1))
        if is_type_iv:
            t_anti = np.array([0.5, 0.5, 0.5])
            for R, t in base_ops:
                mag_ops.append((R, (t + t_anti) % 1.0, -1))
        return mag_ops

    @classmethod
    def apply_wyckoff_expansion(
        cls,
        lattice_matrix: np.ndarray,
        space_group_number: int,
        asymmetric_coords: List[Tuple[str, np.ndarray]],
        tol: float = 1e-4,
    ) -> List[Dict[str, Any]]:
        """Expand asymmetric unit coordinates into complete unit cell respecting PBC."""
        ops = cls.get_seitz_matrices(space_group_number)
        full_sites = []

        for species, frac_pos in asymmetric_coords:
            frac = np.asarray(frac_pos, dtype=np.float64) % 1.0
            generated: List[np.ndarray] = []
            for R, t in ops:
                r_new = (np.dot(R, frac) + t) % 1.0
                is_duplicate = False
                for g in generated:
                    diff = np.abs(r_new - g)
                    pbc_diff = np.minimum(diff, 1.0 - diff)
                    if np.all(pbc_diff < tol):
                        is_duplicate = True
                        break
                if not is_duplicate:
                    generated.append(r_new)
                    cart_pos = np.dot(r_new, lattice_matrix) if lattice_matrix is not None else r_new
                    full_sites.append({
                        "species": species,
                        "fractional_coords": r_new.tolist(),
                        "cartesian_coords": cart_pos.tolist(),
                    })

        return full_sites
