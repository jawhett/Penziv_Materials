"""Tensor algebra, stress-strain transformations, Voigt/Mandel notation, and invariant calculations."""

from typing import Tuple
import numpy as np

# Voigt index mapping: 11, 22, 33, 23, 13, 12 -> 0, 1, 2, 3, 4, 5
VOIGT_MAP = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def tensor4_to_voigt66(C_ijkl: np.ndarray) -> np.ndarray:
    """Convert 4th-order elasticity tensor C_ijkl (3x3x3x3) to 6x6 Voigt stiffness matrix."""
    if C_ijkl.shape != (3, 3, 3, 3):
        raise ValueError(f"Expected shape (3,3,3,3), got {C_ijkl.shape}")
    C_voigt = np.zeros((6, 6), dtype=np.float64)
    for i, (i1, i2) in enumerate(VOIGT_MAP):
        for j, (j1, j2) in enumerate(VOIGT_MAP):
            C_voigt[i, j] = C_ijkl[i1, i2, j1, j2]
    return C_voigt


def voigt66_to_tensor4(C_voigt: np.ndarray) -> np.ndarray:
    """Convert 6x6 Voigt stiffness matrix to 4th-order elasticity tensor C_ijkl (3x3x3x3)."""
    if C_voigt.shape != (6, 6):
        raise ValueError(f"Expected shape (6,6), got {C_voigt.shape}")
    C_ijkl = np.zeros((3, 3, 3, 3), dtype=np.float64)
    for i, (i1, i2) in enumerate(VOIGT_MAP):
        for j, (j1, j2) in enumerate(VOIGT_MAP):
            val = C_voigt[i, j]
            # Symmetrize across minor pairs
            C_ijkl[i1, i2, j1, j2] = val
            C_ijkl[i2, i1, j1, j2] = val
            C_ijkl[i1, i2, j2, j1] = val
            C_ijkl[i2, i1, j2, j1] = val
    return C_ijkl


def compute_mandel_stress(C_elastic: np.ndarray, S_pk2: np.ndarray) -> np.ndarray:
    """Compute Mandel stress tensor M_bar = C_e · S_bar.

    M_bar = C^e · S_pk2
    Where C^e is the right Cauchy-Green elastic deformation tensor and S_pk2 is 2nd Piola-Kirchhoff stress.
    """
    return np.matmul(C_elastic, S_pk2)


def compute_nye_dislocation_tensor(grad_Fp: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute Nye dislocation density tensor alpha_Nye = curl(F^p) and GND density rho_GND.

    grad_Fp is the spatial gradient of plastic deformation gradient (shape: 3x3x3, where grad_Fp[i, j, k] = d(F^p_ij)/dx_k).
    Returns (alpha_Nye (3x3), rho_gnd (scalar in 1/m^2)).
    """
    alpha = np.zeros((3, 3), dtype=np.float64)
    # alpha_ij = eps_jkl * d(F^p_ik)/dx_l
    for i in range(3):
        alpha[i, 0] = grad_Fp[i, 2, 1] - grad_Fp[i, 1, 2]
        alpha[i, 1] = grad_Fp[i, 0, 2] - grad_Fp[i, 2, 0]
        alpha[i, 2] = grad_Fp[i, 1, 0] - grad_Fp[i, 0, 1]

    # L1 norm of Nye tensor represents total GND density
    rho_gnd_norm = np.sum(np.abs(alpha))
    return alpha, float(rho_gnd_norm)


def evaluate_clausius_duhem(
    cauchy_stress: np.ndarray,
    plastic_strain_rate: np.ndarray,
    d_isv_dt: float,
) -> float:
    """Evaluate Clausius-Duhem dissipation rate D_int = sigma : dot(eps_p) - d(psi_isv)/dt >= 0."""
    stress_power = np.sum(cauchy_stress * plastic_strain_rate)
    d_int = stress_power - d_isv_dt
    return float(d_int)
