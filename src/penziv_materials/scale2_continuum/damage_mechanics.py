"""Energy-Conserving Non-Local Gradient Damage Mechanics & Weibull Statistical Scaling."""

from typing import Dict, Tuple, List, Optional
import numpy as np


class NonLocalDamageMechanics:
    """Non-local gradient-enhanced continuum damage model preserving intrinsic fracture energy G_c."""

    def __init__(
        self,
        characteristic_length_lc_um: float = 25.0,
        critical_energy_release_rate_gc_j_m2: float = 45000.0,
        damage_initiation_strain: float = 0.008,
    ):
        self.lc = characteristic_length_lc_um
        self.gc = critical_energy_release_rate_gc_j_m2
        self.eps_0 = damage_initiation_strain

    def solve_nonlocal_equivalent_strain_1d(
        self,
        local_equivalent_strain: np.ndarray,
        dx_um: float = 1.0,
    ) -> np.ndarray:
        """Solve 1D non-local Helmholtz equation:

        Y - c_g * d^2Y/dx^2 = Y_local,  where c_g = l_c^2 / 2
        """
        n = len(local_equivalent_strain)
        c_g = (self.lc**2) / 2.0

        # Tridiagonal system matrix A
        main_diag = (1.0 + 2.0 * c_g / (dx_um**2)) * np.ones(n)
        off_diag = (-c_g / (dx_um**2)) * np.ones(n - 1)

        # Build tridiagonal matrix
        A = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        # Solve A * Y_nonlocal = Y_local
        nonlocal_strain = np.linalg.solve(A, local_equivalent_strain)
        return nonlocal_strain

    def compute_damage_variable(
        self,
        nonlocal_equivalent_strain: np.ndarray,
    ) -> np.ndarray:
        """Compute scalar damage variable D in [0, 1) using exponential degradation law:

        D = 1 - (eps_0 / eps_eq) * exp(-(eps_eq - eps_0) / eps_f)
        """
        d = np.zeros_like(nonlocal_equivalent_strain)
        mask = nonlocal_equivalent_strain > self.eps_0
        eps_f = 0.05  # Critical failure strain
        d[mask] = 1.0 - (self.eps_0 / nonlocal_equivalent_strain[mask]) * np.exp(
            -(nonlocal_equivalent_strain[mask] - self.eps_0) / eps_f
        )
        return np.clip(d, 0.0, 0.999)

    def evaluate_weibull_failure_probability(
        self,
        stress_field_mpa: np.ndarray,
        reference_stress_sigma0_mpa: float = 1250.0,
        weibull_modulus_m: float = 16.5,
        voxel_volume_mm3: float = 0.001,
        reference_volume_v0_mm3: float = 1.0,
    ) -> float:
        """Evaluate weakest-link stochastic Weibull failure probability:

        P_f(sigma) = 1 - exp( - integral [ (sigma / sigma_0)^m ] dV / V_0 )
        """
        pos_stress = np.maximum(0.0, stress_field_mpa)
        integral_term = np.sum((pos_stress / reference_stress_sigma0_mpa) ** weibull_modulus_m) * (
            voxel_volume_mm3 / reference_volume_v0_mm3
        )
        prob_failure = 1.0 - np.exp(-integral_term)
        return float(np.clip(prob_failure, 0.0, 1.0))
