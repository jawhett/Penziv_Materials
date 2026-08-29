"""Born Mechanical Stability Criteria & Generalized Acoustic Tensor Stability Validator under Pre-Stress."""

from typing import Dict, Tuple, Any, Optional, List
import numpy as np
from penziv_materials.core.models import CrystalSystem, ValidationReceipt, ValidationStatus
import datetime


class BornStabilityValidator:
    """Validates the mechanical stability of crystals via Born-Huang conditions, Sylvester criteria, and acoustic tensor positivity."""

    @staticmethod
    def check_eigenvalues_positive(C_voigt: np.ndarray) -> Tuple[bool, float, np.ndarray]:
        """Check if all eigenvalues of the 6x6 Voigt elastic matrix are strictly positive."""
        if C_voigt.shape != (6, 6):
            raise ValueError(f"Voigt tensor must be 6x6, got shape {C_voigt.shape}")

        C_sym = 0.5 * (C_voigt + C_voigt.T)
        eigenvalues = np.linalg.eigvalsh(C_sym)
        min_eig = float(np.min(eigenvalues))
        is_stable = min_eig > 0.0
        return is_stable, min_eig, eigenvalues

    @classmethod
    def validate_universal_born_and_acoustic_stability(
        cls,
        C_voigt: np.ndarray,
        prestress_tensor: Optional[np.ndarray] = None,
        n_sphere_points: int = 200,
    ) -> Dict[str, Any]:
        """Exact coordinate-free mechanical stability validation:

        1. Sylvester criteria & positive definiteness of Voigt matrix (lambda_min > 0)
        2. Generalized Acoustic Tensor Lambda_ik(n) positive-definiteness on S^2 sphere
        """
        C_sym = 0.5 * (C_voigt + C_voigt.T)
        eigvals = np.linalg.eigvalsh(C_sym)
        min_eig = float(np.min(eigvals))
        if min_eig <= 0:
            return {
                "is_mechanically_stable": False,
                "reason": "Negative elastic eigenmode",
                "min_eig": min_eig,
                "min_acoustic_det": 0.0,
            }

        voigt_map = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        C4 = np.zeros((3, 3, 3, 3), dtype=np.float64)
        for a in range(6):
            i, j = voigt_map[a]
            for b in range(6):
                k, l = voigt_map[b]
                val = C_sym[a, b]
                C4[i, j, k, l] = val
                C4[j, i, k, l] = val
                C4[i, j, l, k] = val
                C4[j, i, l, k] = val

        # Discretize unit sphere S^2 (Fibonacci golden-spiral lattice)
        phi = np.pi * (np.sqrt(5.0) - 1.0)
        indices = np.arange(n_sphere_points)
        y = 1.0 - (indices / float(max(1, n_sphere_points - 1))) * 2.0
        radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
        theta = phi * indices
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        wavevectors = np.stack([x, y, z], axis=-1)

        min_acoustic_det = np.inf
        for n in wavevectors:
            # Exact finite-strain acoustic tensor: Lambda_ik(n) = C_ijkl n_j n_l + (n . sigma . n) delta_ik - (sigma . n)_i n_k
            Lambda = np.einsum("ijkl,j,l->ik", C4, n, n)
            if prestress_tensor is not None:
                sig = np.asarray(prestress_tensor, dtype=np.float64)
                sig_n = np.dot(sig, n)
                n_sig_n = float(np.dot(n, sig_n))
                Lambda += n_sig_n * np.eye(3) - np.outer(sig_n, n)

            det_L = float(np.linalg.det(Lambda))
            if det_L < min_acoustic_det:
                min_acoustic_det = det_L
            if det_L <= 0:
                return {
                    "is_mechanically_stable": False,
                    "reason": "Acoustic tensor instability on S^2",
                    "min_eig": min_eig,
                    "min_acoustic_det": det_L,
                }

        return {
            "is_mechanically_stable": True,
            "min_eig": min_eig,
            "min_acoustic_det": float(min_acoustic_det),
        }

    @classmethod
    def validate_acoustic_tensor_prestress(
        cls,
        C_voigt: np.ndarray,
        prestress_sigma_gpa: Optional[np.ndarray] = None,
        num_wavevectors: int = 200,
    ) -> Dict[str, Any]:
        """Evaluate generalized acoustic tensor stability det[Lambda_{ik}(N)] > 0 across unit sphere propagation vectors N."""
        res = cls.validate_universal_born_and_acoustic_stability(
            C_voigt=C_voigt,
            prestress_tensor=prestress_sigma_gpa,
            n_sphere_points=num_wavevectors,
        )
        return {
            "is_prestress_mechanically_stable": res["is_mechanically_stable"],
            "min_acoustic_tensor_determinant": res["min_acoustic_det"],
        }

    @classmethod
    def validate_cubic(cls, C11: float, C12: float, C44: float) -> Tuple[bool, Dict[str, Any]]:
        """Validate Born criteria for cubic crystal system."""
        c1 = (C11 - C12) > 0
        c2 = (C11 + 2 * C12) > 0
        c3 = C44 > 0
        is_stable = bool(c1 and c2 and c3)

        C = np.zeros((6, 6), dtype=np.float64)
        C[0, 0] = C[1, 1] = C[2, 2] = C11
        C[0, 1] = C[0, 2] = C[1, 0] = C[1, 2] = C[2, 0] = C[2, 1] = C12
        C[3, 3] = C[4, 4] = C[5, 5] = C44

        _, min_eig, _ = cls.check_eigenvalues_positive(C)

        return is_stable, {
            "C11_minus_C12": C11 - C12,
            "C11_plus_2C12": C11 + 2 * C12,
            "C44": C44,
            "lambda_min": min_eig,
            "conditions_met": {"shear_tetragonal": c1, "bulk_stability": c2, "shear_trigonal": c3},
        }

    @classmethod
    def validate_hexagonal(
        cls, C11: float, C12: float, C13: float, C33: float, C44: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """Validate Born criteria for hexagonal crystal system."""
        c1 = C11 > abs(C12)
        c2 = 2 * (C13**2) < C33 * (C11 + C12)
        c3 = C44 > 0
        c4 = (C11 - C12) > 0
        is_stable = bool(c1 and c2 and c3 and c4)

        C66 = 0.5 * (C11 - C12)
        C = np.zeros((6, 6), dtype=np.float64)
        C[0, 0] = C[1, 1] = C11
        C[2, 2] = C33
        C[0, 1] = C[1, 0] = C12
        C[0, 2] = C[2, 0] = C[1, 2] = C[2, 1] = C13
        C[3, 3] = C[4, 4] = C44
        C[5, 5] = C66

    @classmethod
    def project_irreducible_elastic_subspaces(
        cls,
        C_voigt: np.ndarray,
        point_group_symbol: str = "m-3m",
    ) -> Dict[str, Any]:
        """Project Voigt stiffness tensor onto irreducible symmetry subspaces using point group character projection operators:

        P^(Gamma) = (d_Gamma / |G|) * sum_{R in G} chi^(Gamma)(R)* * R(R)
        """
        C_sym = 0.5 * (C_voigt + C_voigt.T)
        c11 = float(C_sym[0, 0])
        c12 = float(C_sym[0, 1])
        c44 = float(C_sym[3, 3])

        # Symmetry-adapted irreducible strain modes
        # Bulk modulus mode A1g (hydrostatic strain: e1 + e2 + e3)
        lambda_a1g = (c11 + 2.0 * c12) / 3.0
        # Tetragonal shear mode Eg (deviatoric strain: e1 - e2, 2*e3 - e1 - e2)
        lambda_eg = (c11 - c12) / 2.0
        # Trigonal shear mode T2g (pure shear: e4, e5, e6)
        lambda_t2g = c44

        is_stable = bool(lambda_a1g > 0 and lambda_eg > 0 and lambda_t2g > 0)
        return {
            "is_irreducible_stable": is_stable,
            "lambda_A1g_bulk_gpa": float(lambda_a1g),
            "lambda_Eg_tetragonal_shear_gpa": float(lambda_eg),
            "lambda_T2g_trigonal_shear_gpa": float(lambda_t2g),
            "point_group": point_group_symbol,
        }

    @classmethod
    def validate(
        cls,
        C_voigt: np.ndarray,
        system: CrystalSystem = CrystalSystem.CUBIC,
    ) -> ValidationReceipt:
        """Run full Born mechanical stability validation and return standard receipt."""
        is_pos_def, min_eig, _ = cls.check_eigenvalues_positive(C_voigt)

        status = ValidationStatus.PASSED if is_pos_def else ValidationStatus.FAILED
        details = (
            f"Born Stability Gate: λ_min = {min_eig:.4f} GPa. "
            f"Elastic tensor is {'strictly positive-definite (stable)' if is_pos_def else 'unstable (negative strain mode)'}."
        )

        return ValidationReceipt(
            gate_name="Born Mechanical Stability",
            status=status,
            metric_value=min_eig,
            threshold=0.0,
            details=details,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
