"""Born Mechanical Stability Criteria and Elastic Tensor Positive Definiteness Validator."""

from typing import Dict, Tuple, Any
import numpy as np
from penziv_materials.core.models import CrystalSystem, ValidationReceipt, ValidationStatus
import datetime


class BornStabilityValidator:
    """Validates the mechanical stability of crystals via Born-Huang conditions and positive definiteness."""

    @staticmethod
    def check_eigenvalues_positive(C_voigt: np.ndarray) -> Tuple[bool, float, np.ndarray]:
        """Check if all eigenvalues of the 6x6 Voigt elastic matrix are strictly positive."""
        if C_voigt.shape != (6, 6):
            raise ValueError(f"Voigt tensor must be 6x6, got shape {C_voigt.shape}")

        # Symmetrize in case of numerical noise
        C_sym = 0.5 * (C_voigt + C_voigt.T)
        eigenvalues = np.linalg.eigvalsh(C_sym)
        min_eig = float(np.min(eigenvalues))
        is_stable = min_eig > 0.0
        return is_stable, min_eig, eigenvalues

    @classmethod
    def validate_cubic(cls, C11: float, C12: float, C44: float) -> Tuple[bool, Dict[str, Any]]:
        """Validate Born criteria for cubic crystal system:

        1. C11 - C12 > 0
        2. C11 + 2*C12 > 0 (Bulk modulus K = (C11 + 2*C12)/3 > 0)
        3. C44 > 0
        """
        c1 = (C11 - C12) > 0
        c2 = (C11 + 2 * C12) > 0
        c3 = C44 > 0
        is_stable = bool(c1 and c2 and c3)

        # Build 6x6 matrix
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
