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
    def validate_acoustic_tensor_prestress(
        cls,
        C_voigt: np.ndarray,
        prestress_sigma_gpa: Optional[np.ndarray] = None,
        num_wavevectors: int = 14,
    ) -> Dict[str, Any]:
        """Evaluate generalized acoustic tensor stability det[Lambda_{ik}(N)] > 0 across unit sphere propagation vectors N."""
        voigt_map = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
        C4 = np.zeros((3, 3, 3, 3), dtype=np.float64)
        for a in range(6):
            i, j = voigt_map[a]
            for b in range(6):
                k, l = voigt_map[b]
                val = C_voigt[a, b]
                C4[i, j, k, l] = val
                C4[j, i, k, l] = val
                C4[i, j, l, k] = val
                C4[j, i, l, k] = val

        # Sample propagation wavevectors on unit sphere
        wavevectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 1.0, 0.0]) / np.sqrt(2),
            np.array([1.0, 0.0, 1.0]) / np.sqrt(2),
            np.array([0.0, 1.0, 1.0]) / np.sqrt(2),
            np.array([1.0, 1.0, 1.0]) / np.sqrt(3),
        ]

        min_det = np.inf
        is_stable = True

        for N in wavevectors:
            Lambda = np.zeros((3, 3), dtype=np.float64)
            for i in range(3):
                for k in range(3):
                    for j in range(3):
                        for l in range(3):
                            Lambda[i, k] += C4[i, j, k, l] * N[j] * N[l]
                            if prestress_sigma_gpa is not None and i == k:
                                Lambda[i, k] += prestress_sigma_gpa[j, l] * N[j] * N[l]
            det_val = float(np.linalg.det(Lambda))
            if det_val < min_det:
                min_det = det_val
            if det_val <= 0.0:
                is_stable = False

        return {
            "is_prestress_mechanically_stable": is_stable,
            "min_acoustic_tensor_determinant": float(min_det),
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

        _, min_eig, _ = cls.check_eigenvalues_positive(C)
        return is_stable, {"lambda_min": min_eig, "is_stable": is_stable}

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
