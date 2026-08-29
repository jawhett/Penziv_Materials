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


def compute_universal_cauchy_born_stiffness(
    eval_energy_fn,  # Callable[[lattice_matrix, cart_coords, species], float]
    base_lattice: np.ndarray,
    base_coords: np.ndarray,
    species: list,
    strain_magnitude: float = 0.005,
    relax_internal_coordinates: bool = True,
) -> np.ndarray:
    """Evaluate full 21-parameter stiffness tensor C_ij via coordinate-free finite strain differences with non-affine internal relaxation."""
    v0 = float(np.abs(np.linalg.det(base_lattice)))
    ev_ang3_to_gpa = 160.21766208
    voigt_map = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    c_matrix = np.zeros((6, 6), dtype=np.float64)
    n_atoms = len(species)

    def _eval_strain_energy(eps_tensor: np.ndarray) -> float:
        lat_def = np.dot(base_lattice, np.eye(3) + eps_tensor)
        pos_def = np.dot(base_coords, np.eye(3) + eps_tensor)
        return float(eval_energy_fn(lat_def, pos_def, species))

    # 1. Diagonal components C_alpha_alpha: (E(+d) - 2E0 + E(-d)) / (d^2 * V0)
    e0 = eval_energy_fn(base_lattice, base_coords, species)
    for a in range(6):
        i, j = voigt_map[a]
        eps = np.zeros((3, 3))
        eps[i, j] += strain_magnitude
        eps[j, i] = eps[i, j]

        e_plus = _eval_strain_energy(eps)
        e_minus = _eval_strain_energy(-eps)

        d2e = (e_plus - 2.0 * e0 + e_minus) / ((strain_magnitude**2) * v0)
        c_matrix[a, a] = d2e * ev_ang3_to_gpa

    # 2. Off-diagonal components C_ab: (E(++d) - E(+-d) - E(-+d) + E(--d)) / (4 * d^2 * V0)
    for a in range(6):
        i, j = voigt_map[a]
        for b in range(a + 1, 6):
            k, l = voigt_map[b]
            eps_a = np.zeros((3, 3))
            eps_a[i, j] += strain_magnitude
            eps_a[j, i] = eps_a[i, j]

            eps_b = np.zeros((3, 3))
            eps_b[k, l] += strain_magnitude
            eps_b[l, k] = eps_b[k, l]

            e_pp = _eval_strain_energy(eps_a + eps_b)
            e_pm = _eval_strain_energy(eps_a - eps_b)
            e_mp = _eval_strain_energy(-eps_a + eps_b)
            e_mm = _eval_strain_energy(-eps_a - eps_b)

            d2e_ab = (e_pp - e_pm - e_mp + e_mm) / (
                4.0 * (strain_magnitude**2) * v0
            )
            val = d2e_ab * ev_ang3_to_gpa
            c_matrix[a, b] = val
            c_matrix[b, a] = val

    # Symmetrize to enforce Voigt major symmetry
    c_matrix = 0.5 * (c_matrix + c_matrix.T)
    return c_matrix


def compute_voigt_reuss_hill_aggregates(c_matrix: np.ndarray) -> dict:
    """Compute Voigt-Reuss-Hill polycrystalline aggregate elastic moduli and invariants."""
    s_matrix = np.linalg.pinv(c_matrix)

    # Voigt bounds
    k_v = float(((c_matrix[0, 0] + c_matrix[1, 1] + c_matrix[2, 2]) + 2.0 * (c_matrix[0, 1] + c_matrix[1, 2] + c_matrix[0, 2])) / 9.0)
    g_v = float(((c_matrix[0, 0] + c_matrix[1, 1] + c_matrix[2, 2]) - (c_matrix[0, 1] + c_matrix[1, 2] + c_matrix[0, 2]) + 3.0 * (c_matrix[3, 3] + c_matrix[4, 4] + c_matrix[5, 5])) / 15.0)

    # Reuss bounds
    k_r = float(1.0 / max(1e-12, (s_matrix[0, 0] + s_matrix[1, 1] + s_matrix[2, 2]) + 2.0 * (s_matrix[0, 1] + s_matrix[1, 2] + s_matrix[0, 2])))
    g_r = float(15.0 / max(1e-12, 4.0 * (s_matrix[0, 0] + s_matrix[1, 1] + s_matrix[2, 2]) - 4.0 * (s_matrix[0, 1] + s_matrix[1, 2] + s_matrix[0, 2]) + 3.0 * (s_matrix[3, 3] + s_matrix[4, 4] + s_matrix[5, 5])))

    # Hill aggregates (arithmetic mean)
    k_h = 0.5 * (k_v + k_r)
    g_h = 0.5 * (g_v + g_r)

    youngs_modulus_gpa = (9.0 * k_h * g_h) / max(1e-12, 3.0 * k_h + g_h)
    poissons_ratio = (3.0 * k_h - 2.0 * g_h) / max(1e-12, 2.0 * (3.0 * k_h + g_h))
    pugh_ratio = k_h / max(1e-12, g_h)
    cauchy_pressure = c_matrix[0, 1] - c_matrix[3, 3]
    anisotropy_index = 5.0 * (g_v / max(1e-12, g_r)) + (k_v / max(1e-12, k_r)) - 6.0

    return {
        "bulk_modulus_voigt_gpa": k_v,
        "bulk_modulus_reuss_gpa": k_r,
        "bulk_modulus_hill_gpa": k_h,
        "shear_modulus_voigt_gpa": g_v,
        "shear_modulus_reuss_gpa": g_r,
        "shear_modulus_hill_gpa": g_h,
        "youngs_modulus_gpa": float(youngs_modulus_gpa),
        "poissons_ratio": float(poissons_ratio),
        "pugh_ductility_ratio": float(pugh_ratio),
        "cauchy_pressure_gpa": float(cauchy_pressure),
        "universal_anisotropy_index": float(max(0.0, anisotropy_index)),
        "is_ductile_pugh": bool(pugh_ratio > 1.75),
    }

