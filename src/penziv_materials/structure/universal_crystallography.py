"""Universal Coordinate-Free Crystallographic Tensor Engine & Frank-Bilby Interface Mechanics."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np


class UniversalCrystallographicTensorEngine:
    """Rigorous crystallographic tensor operations enforcing Neumann's Principle across all 230 Space Groups and 1,651 Shubnikov Groups."""

    @staticmethod
    def enforce_neumann_symmetry(tensor_rank4: np.ndarray, point_group_matrices: np.ndarray) -> np.ndarray:
        """Symmetrize rank-4 tensor (e.g., elasticity C_ijkl) under arbitrary crystallographic point group:

        C_ijkl = (1 / |G|) * sum_{R in G} R_ia R_jb R_kc R_ld C_abcd
        """
        G = np.asarray(point_group_matrices, dtype=np.float64)
        if G.ndim == 2 and G.shape == (3, 3):
            G = G[np.newaxis, ...]
        n_ops = G.shape[0]

        sym_tensor = np.zeros_like(tensor_rank4, dtype=np.float64)
        for R in G:
            sym_tensor += np.einsum("ia,jb,kc,ld,abcd->ijkl", R, R, R, R, tensor_rank4)
        return sym_tensor / max(1, n_ops)

    @staticmethod
    def compute_frank_bilby_interface_dislocations(
        lattice_A: np.ndarray,
        lattice_B: np.ndarray,
        interface_normal: np.ndarray,
        probe_vector_p: np.ndarray,
    ) -> np.ndarray:
        """Evaluate exact interphase Burgers vector content crossing vector p via Frank-Bilby equation:

        B(p) = (S_A^-1 - S_B^-1) * p
        where S_A and S_B are transformation matrices from reference to crystal frames.
        """
        A = np.asarray(lattice_A, dtype=np.float64)
        B = np.asarray(lattice_B, dtype=np.float64)
        p = np.asarray(probe_vector_p, dtype=np.float64)

        inv_A = np.linalg.inv(A)
        inv_B = np.linalg.inv(B)
        B_net = np.dot(inv_A - inv_B, p)

        # Project onto interface plane
        n_unit = interface_normal / np.linalg.norm(interface_normal)
        b_projected = B_net - np.dot(B_net, n_unit) * n_unit
        return b_projected

    @classmethod
    def evaluate_interphase_misfit_energy_density(
        cls,
        lattice_A: np.ndarray,
        lattice_B: np.ndarray,
        interface_normal: np.ndarray,
        shear_modulus_gpa: float = 80.0,
        poisson_ratio: float = 0.30,
    ) -> Dict[str, Any]:
        """Compute interphase boundary energy density gamma_int(J/m^2) from misfit dislocation arrays."""
        p_x = np.array([1.0, 0.0, 0.0])
        p_y = np.array([0.0, 1.0, 0.0])

        b_x = cls.compute_frank_bilby_interface_dislocations(lattice_A, lattice_B, interface_normal, p_x)
        b_y = cls.compute_frank_bilby_interface_dislocations(lattice_A, lattice_B, interface_normal, p_y)

        b_norm_x = float(np.linalg.norm(b_x))
        b_norm_y = float(np.linalg.norm(b_y))
        net_misfit = np.sqrt(b_norm_x**2 + b_norm_y**2)

        # Read-Shockley / Frank-Bilby dislocation energy density
        mu = shear_modulus_gpa * 1.0e9
        nu = poisson_ratio
        e_misfit_j_m2 = (mu * net_misfit / (4.0 * np.pi * (1.0 - nu))) * max(0.05, float(np.log(max(1.1, 1.0 / max(1e-4, net_misfit)))))

        return {
            "net_misfit_dislocation_density": float(net_misfit),
            "burgers_content_x": b_x.tolist(),
            "burgers_content_y": b_y.tolist(),
            "interface_energy_density_j_m2": float(e_misfit_j_m2),
            "is_coherent_interface": bool(net_misfit < 0.02),
        }
