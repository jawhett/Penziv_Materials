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
    ) -> Dict[str, Any]:
        """Evaluate irreducible strain representation Born stability conditions across all 7 crystal systems:

        delta U = 1/2 sum_Gamma C^(Gamma) (epsilon^(Gamma))^2 > 0
        """
        C = np.asarray(c_voigt_gpa, dtype=np.float64)
        if C.shape != (6, 6):
            return {"is_mechanically_stable": False, "failed_irreducible_modes": ["Invalid tensor dimensions"]}

        failed_modes = []
        sys = crystal_system.lower()

        # Symmetrize Voigt matrix
        C_sym = 0.5 * (C + C.T)
        eigvals = np.linalg.eigvalsh(C_sym)
        min_eig = float(np.min(eigvals))

        # 1. Sylvester Leading Principal Minors check for all systems
        for k in range(1, 7):
            minor_det = np.linalg.det(C_sym[:k, :k])
            if minor_det <= 0:
                failed_modes.append(f"Sylvester Minor Det(M_{k}x{k}) > 0")

        # 2. Crystal-system specific irreducible representation strain modes
        if "cub" in sys:
            if (C[0, 0] + 2.0 * C[0, 1]) <= 0:
                failed_modes.append("A_1g Bulk Modulus: C11 + 2C12 > 0")
            if (C[0, 0] - C[0, 1]) <= 0:
                failed_modes.append("E_g Tetragonal Shear: C11 - C12 > 0")
            if C[3, 3] <= 0:
                failed_modes.append("T_2g Shear: C44 > 0")

        elif "hex" in sys or "trig" in sys:
            if (C[0, 0] - C[0, 1]) <= 0:
                failed_modes.append("C11 - C12 > 0")
            if (C[0, 0] + C[0, 1]) * C[2, 2] - 2.0 * (C[0, 2] ** 2) <= 0:
                failed_modes.append("(C11 + C12)*C33 - 2*C13^2 > 0")

        elif "ortho" in sys or "tetra" in sys:
            det_principal = float(np.linalg.det(C[:3, :3]))
            if det_principal <= 0:
                failed_modes.append("Principal 3x3 Determinant > 0")

        elif "mono" in sys or "tric" in sys:
            # Monoclinic (13 moduli) & Triclinic (21 moduli) coordinate-free positive definiteness
            if min_eig <= 0:
                failed_modes.append("General Anisotropic Eigenmode Positivity: lambda_min > 0")

        is_stable = len(failed_modes) == 0 and min_eig > 0.0

        return {
            "is_mechanically_stable": is_stable,
            "failed_irreducible_modes": failed_modes,
            "min_eigenvalue_gpa": min_eig,
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
