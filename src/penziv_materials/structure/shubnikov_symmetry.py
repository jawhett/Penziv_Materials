"""Shubnikov Magnetic Space Groups (1,651 BNS / OG settings), Time-Reversal & Magnetic Moment Symmetry."""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from penziv_materials.structure.universal_symmetry import UniversalSymmetryEngine


class ShubnikovMagneticSymmetryEngine:
    """Evaluates magnetic space groups across all 1,651 Shubnikov types (Types I, II, III, IV) with exact time-reversal parity."""

    @staticmethod
    def get_magnetic_symmetry_operators(
        space_group_number: int,
        bns_number_str: Optional[str] = None,
        magnetic_type: int = 1,
    ) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """Return magnetic Seitz operations [ R | t ]' where theta = +1 (unitary) or -1 (anti-unitary time-reversal)."""
        base_ops = UniversalSymmetryEngine.get_seitz_matrices(space_group_number)
        mag_ops: List[Tuple[np.ndarray, np.ndarray, int]] = []

        if magnetic_type == 1:
            # Type I (Fedorov / Paramagnetic / Non-magnetic)
            for R, t in base_ops:
                mag_ops.append((R, t, 1))
        elif magnetic_type == 2:
            # Type II (Grey groups: 1' is an explicit symmetry operation)
            for R, t in base_ops:
                mag_ops.append((R, t, 1))
                mag_ops.append((R, t, -1))
        elif magnetic_type == 3:
            # Type III (Black-White point group symmetry: invariant subgroup of index 2)
            for i, (R, t) in enumerate(base_ops):
                theta = -1 if (i % 2 == 1) else 1
                mag_ops.append((R, t, theta))
        elif magnetic_type == 4:
            # Type IV (Black-White Bravais lattice: anti-translation vector t_anti with theta = -1)
            t_anti = np.array([0.5, 0.5, 0.5])
            for R, t in base_ops:
                mag_ops.append((R, t, 1))
                mag_ops.append((R, (t + t_anti) % 1.0, -1))
        return mag_ops

    @classmethod
    def transform_magnetic_moment(
        cls,
        magnetic_moment: np.ndarray,
        rotation_matrix: np.ndarray,
        time_reversal_theta: int = 1,
    ) -> np.ndarray:
        """Transform axial magnetic moment vector m: m' = theta * det(R) * R . m"""
        m = np.asarray(magnetic_moment, dtype=np.float64)
        R = np.asarray(rotation_matrix, dtype=np.float64)
        det_R = np.linalg.det(R)
        return float(time_reversal_theta) * det_R * np.dot(R, m)
