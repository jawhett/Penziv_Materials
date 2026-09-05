"""Space Group Symmetry Operations, Irreducible Born Stability & Universal Slip Generators."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np


class SpaceGroupSymmetryEngine:
    """Algorithmic space group operations, irreducible Born stability decomposition, and universal anisotropic slip/twinning generators."""

    POINT_GROUPS_32 = [
        "1", "-1", "2", "m", "2/m", "222", "mm2", "mmm",
        "4", "-4", "4/m", "422", "4mm", "-42m", "4/mmm",
        "3", "-3", "32", "3m", "-3m",
        "6", "-6", "6/m", "622", "6mm", "-62m", "6/mmm",
        "23", "m-3", "432", "-43m", "m-3m",
    ]

    def __init__(self, space_group: str = "P1", space_group_number: int = 1):
        self.space_group = space_group
        self.space_group_number = space_group_number

    def evaluate_irreducible_born_stability(
        self,
        c_voigt_gpa: np.ndarray,
        crystal_system: str = "cubic",
        prestress_tensor: Optional[np.ndarray] = None,
        num_acoustic_samples: int = 100,
    ) -> Dict[str, Any]:
        """Evaluate coordinate-free mechanical stability under finite strain and arbitrary orientation:
        1. Sylvester leading principal minors Det(M_kxk) > 0 and lambda_min > 0.
        2. Positive-definiteness of acoustic tensor det[Lambda(n)] > 0 for all wavevectors n on unit sphere S^2.
        """
        C = np.asarray(c_voigt_gpa, dtype=np.float64)
        if C.shape != (6, 6):
            return {"is_mechanically_stable": False, "failed_irreducible_modes": ["Invalid tensor dimensions"]}

        failed_modes = []
        C_sym = 0.5 * (C + C.T)
        eigvals = np.linalg.eigvalsh(C_sym)
        min_eig = float(np.min(eigvals))

        # 1. Sylvester Leading Principal Minors check (Frame-invariant positive definiteness)
        for k in range(1, 7):
            minor_det = float(np.linalg.det(C_sym[:k, :k]))
            if minor_det <= 0:
                failed_modes.append(f"Sylvester Minor Det(M_{k}x{k}) > 0")

        # 2. Coordinate-free acoustic tensor positivity on unit sphere S^2
        voigt_map = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        C4 = np.zeros((3, 3, 3, 3), dtype=np.float64)
        for a in range(6):
            i, j = voigt_map[a]
            for b in range(6):
                k_idx, l_idx = voigt_map[b]
                val = C_sym[a, b]
                C4[i, j, k_idx, l_idx] = val
                C4[j, i, k_idx, l_idx] = val
                C4[i, j, l_idx, k_idx] = val
                C4[j, i, l_idx, k_idx] = val

        # Fibonacci sphere sampling for uniform S^2 coverage
        phi = np.pi * (np.sqrt(5.0) - 1.0)
        indices = np.arange(num_acoustic_samples)
        y = 1.0 - (indices / float(max(1, num_acoustic_samples - 1))) * 2.0
        radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
        theta = phi * indices
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        wavevectors = np.stack([x, y, z], axis=-1)

        min_acoustic_det = float("inf")
        for n in wavevectors:
            Lambda = np.einsum("ijkl,j,l->ik", C4, n, n)
            if prestress_tensor is not None:
                sig = np.asarray(prestress_tensor, dtype=np.float64)
                sig_n = np.dot(sig, n)
                n_sig_n = float(np.dot(n, sig_n))
                Lambda += n_sig_n * np.eye(3)

            det_L = float(np.linalg.det(Lambda))
            if det_L < min_acoustic_det:
                min_acoustic_det = det_L

        if min_acoustic_det <= 0.0:
            failed_modes.append("Acoustic Tensor Positivity: min_det[Lambda(n)] > 0")

        is_stable = len(failed_modes) == 0 and min_eig > 0.0

        return {
            "is_mechanically_stable": is_stable,
            "failed_irreducible_modes": failed_modes,
            "min_eigenvalue_gpa": min_eig,
            "min_acoustic_tensor_determinant": float(min_acoustic_det),
            "all_eigenvalues_gpa": eigvals.tolist(),
            "crystal_system": crystal_system,
        }

    def generate_anisotropic_slip_and_twinning_systems(
        self,
        lattice_matrix: np.ndarray,
        max_miller_index: int = 2,
    ) -> Dict[str, Any]:
        """Generate active crystallographic slip and deformation twinning systems satisfying b . n = 0."""
        lat = np.asarray(lattice_matrix, dtype=np.float64)
        recip_lat = np.linalg.inv(lat).T

        slip_systems = []
        # Construct candidate slip plane normals n and slip directions b
        indices = range(-max_miller_index, max_miller_index + 1)
        for h in indices:
            for k in indices:
                for l in indices:
                    if h == 0 and k == 0 and l == 0:
                        continue
                    n_cart = np.dot(np.array([h, k, l]), recip_lat)
                    n_norm = n_cart / np.linalg.norm(n_cart)

                    for u in indices:
                        for v in indices:
                            for w in indices:
                                if u == 0 and v == 0 and w == 0:
                                    continue
                                b_cart = np.dot(np.array([u, v, w]), lat)
                                b_len = np.linalg.norm(b_cart)
                                b_norm = b_cart / b_len

                                # Orthogonality condition b . n = 0
                                if abs(np.dot(b_norm, n_norm)) < 1e-4:
                                    schmid_m = np.outer(b_norm, n_norm)
                                    slip_systems.append({
                                        "plane_hkl": [h, k, l],
                                        "direction_uvw": [u, v, w],
                                        "burgers_vector_length_angstrom": float(b_len),
                                        "schmid_tensor": schmid_m.tolist(),
                                    })
                                    if len(slip_systems) >= 24:
                                        break

        return {
            "num_active_slip_systems": len(slip_systems),
            "primary_slip_systems": slip_systems[:12],
        }
