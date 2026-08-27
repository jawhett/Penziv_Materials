"""Universal Coordinate-Free Rank-N Neumann Tensor Projection Engine."""

from typing import List, Tuple, Dict, Optional, Any
import numpy as np


class UniversalNeumannTensorEngine:
    """Enforces Neumann's Principle on arbitrary rank-N physical property tensors
    using exact group projection operators over 3D crystallographic point groups.
    """

    @staticmethod
    def project_tensor(
        tensor: np.ndarray,
        point_group_rotations: List[np.ndarray],
        rank: Optional[int] = None,
    ) -> np.ndarray:
        """Coordinate-free projection of a rank-N tensor onto the invariant point group subspace:

        T_{i1...iN} = (1 / |G|) * sum_{R in G} R_{i1 j1} ... R_{iN jN} * T_{j1...jN}
        """
        arr = np.asarray(tensor, dtype=np.float64)
        actual_rank = arr.ndim if rank is None else rank

        if arr.ndim != actual_rank or any(d != 3 for d in arr.shape):
            raise ValueError(f"Tensor must have shape {(3,) * actual_rank}, got {arr.shape}")

        n_ops = len(point_group_rotations)
        if n_ops == 0:
            return arr

        # Dynamic einsum notation construction for arbitrary rank
        in_indices = [chr(97 + i) for i in range(actual_rank)]          # ['a', 'b', 'c', ...]
        out_indices = [chr(105 + i) for i in range(actual_rank)]        # ['i', 'j', 'k', ...]

        # Build einsum string: "ia,jb,kc,abcd->ijk"
        r_terms = [f"{out_indices[k]}{in_indices[k]}" for k in range(actual_rank)]
        einsum_str = f"{','.join(r_terms)},{''.join(in_indices)}->{''.join(out_indices)}"

        sym_tensor = np.zeros_like(arr, dtype=np.float64)
        for R in point_group_rotations:
            r_mat = np.asarray(R, dtype=np.float64)
            r_mats = [r_mat] * actual_rank
            sym_tensor += np.einsum(einsum_str, *r_mats, arr)

        return sym_tensor / float(n_ops)

    @classmethod
    def project_elastic_stiffness_rank4(
        cls,
        stiffness_rank4: np.ndarray,
        point_group_rotations: List[np.ndarray],
    ) -> np.ndarray:
        """Project rank-4 elastic stiffness tensor C_ijkl and enforce Voigt minor and major symmetries."""
        c_proj = cls.project_tensor(stiffness_rank4, point_group_rotations, rank=4)
        # Minor symmetries: C_ijkl = C_jikl = C_ijlk
        c_sym = 0.25 * (c_proj + np.swapaxes(c_proj, 0, 1) + np.swapaxes(c_proj, 2, 3) + np.swapaxes(np.swapaxes(c_proj, 0, 1), 2, 3))
        # Major symmetry: C_ijkl = C_klij
        c_sym = 0.5 * (c_sym + np.swapaxes(np.swapaxes(c_sym, 0, 2), 1, 3))
        return c_sym

    @classmethod
    def project_piezoelectric_rank3(
        cls,
        piezo_rank3: np.ndarray,
        point_group_rotations: List[np.ndarray],
    ) -> np.ndarray:
        """Project rank-3 piezoelectric tensor d_ijk and enforce Voigt minor symmetry d_ijk = d_ikj."""
        d_proj = cls.project_tensor(piezo_rank3, point_group_rotations, rank=3)
        d_sym = 0.5 * (d_proj + np.swapaxes(d_proj, 1, 2))
        return d_sym
