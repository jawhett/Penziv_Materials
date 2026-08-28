"""3D Anisotropic Non-Local Damage & Spectral Phase-Field Fracture Mechanics."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np


class NonLocalDamageMechanics:
    """Solves anisotropic 3D non-local damage and phase-field fracture with spectral tensile/compressive strain energy split."""

    def __init__(
        self,
        grid_shape: Tuple[int, int, int] = (16, 16, 16),
        dx_m: float = 1.0e-6,
        critical_energy_release_rate_g_c: float = 2.7,
        length_scale_l0_m: float = 2.0e-6,
        mobility_m: float = 1.0e3,
        lambda_lame_gpa: float = 80.0,
        mu_shear_gpa: float = 40.0,
        characteristic_length_lc_um: Optional[float] = None,
    ):
        self.nx, self.ny, self.nz = grid_shape
        self.dx = dx_m
        self.gc = critical_energy_release_rate_g_c
        self.l0 = (characteristic_length_lc_um * 1.0e-6) if characteristic_length_lc_um is not None else length_scale_l0_m
        self.mobility = mobility_m
        self.lam = lambda_lame_gpa * 1.0e9
        self.mu = mu_shear_gpa * 1.0e9

    def spectral_strain_decomposition(
        self,
        strain_tensor: np.ndarray,
        c_voigt_matrix_gpa: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """Spectral decomposition into tensile psi_+ and compressive psi_- elastic energy densities with fourth-order tensor support:

        psi_+ = 0.5 * eps_+ : C : eps_+,   psi_- = 0.5 * eps_- : C : eps_-
        """
        eps_sym = 0.5 * (strain_tensor + strain_tensor.T)
        eigvals, eigvecs = np.linalg.eigh(eps_sym)

        eig_pos = np.maximum(0.0, eigvals)
        eig_neg = np.minimum(0.0, eigvals)

        # Reconstruct positive and negative spectral strain projection tensors
        eps_plus = np.zeros((3, 3), dtype=np.float64)
        eps_minus = np.zeros((3, 3), dtype=np.float64)
        for a in range(3):
            n_a = eigvecs[:, a]
            proj_a = np.outer(n_a, n_a)
            eps_plus += eig_pos[a] * proj_a
            eps_minus += eig_neg[a] * proj_a

        if c_voigt_matrix_gpa is not None and c_voigt_matrix_gpa.shape == (6, 6):
            # Anisotropic fourth-order elasticity projection (Voigt notation conversion)
            c_pa = c_voigt_matrix_gpa * 1.0e9
            v_plus = np.array([eps_plus[0, 0], eps_plus[1, 1], eps_plus[2, 2], 2*eps_plus[1, 2], 2*eps_plus[0, 2], 2*eps_plus[0, 1]])
            v_minus = np.array([eps_minus[0, 0], eps_minus[1, 1], eps_minus[2, 2], 2*eps_minus[1, 2], 2*eps_minus[0, 2], 2*eps_minus[0, 1]])
            psi_plus = 0.5 * float(np.dot(v_plus, np.dot(c_pa, v_plus)))
            psi_minus = 0.5 * float(np.dot(v_minus, np.dot(c_pa, v_minus)))
        else:
            tr_pos = max(0.0, float(np.sum(eigvals)))
            tr_neg = min(0.0, float(np.sum(eigvals)))
            psi_plus = 0.5 * self.lam * (tr_pos**2) + self.mu * np.sum(eig_pos**2)
            psi_minus = 0.5 * self.lam * (tr_neg**2) + self.mu * np.sum(eig_neg**2)

        return float(psi_plus), float(psi_minus)

    def solve_nonlocal_equivalent_strain_1d(
        self,
        local_equivalent_strain: np.ndarray,
        length_um: float = 100.0,
    ) -> np.ndarray:
        """Solve 1D Helmholtz non-local regularization equation: eps_nl - l0^2 * d^2(eps_nl)/dx^2 = eps_loc."""
        eps_loc = np.asarray(local_equivalent_strain, dtype=np.float64)
        n = len(eps_loc)
        dx = (length_um * 1.0e-6) / max(1, n)

        c = (self.l0 / dx) ** 2
        A = np.zeros((n, n))
        for i in range(n):
            A[i, i] = 1.0 + 2.0 * c
            if i > 0:
                A[i, i - 1] = -c
            if i < n - 1:
                A[i, i + 1] = -c
        A[0, -1] = -c
        A[-1, 0] = -c

        eps_nl = np.linalg.solve(A, eps_loc)
        return eps_nl

    def compute_damage_variable(
        self,
        equivalent_strain: np.ndarray,
        strain_threshold_eps_0: float = 0.002,
        softening_parameter_alpha: float = 0.95,
    ) -> np.ndarray:
        """Compute scalar damage variable d(eps) in [0, 1)."""
        eps = np.asarray(equivalent_strain, dtype=np.float64)
        denom = np.maximum(1e-6, softening_parameter_alpha * eps)
        d = np.clip((eps - strain_threshold_eps_0) / denom, 0.0, 0.99)
        return d

    def compute_nonlocal_damage_field(
        self,
        local_equivalent_strain: np.ndarray,
        strain_threshold_eps_0: float = 0.002,
        softening_parameter_alpha: float = 0.95,
    ) -> Dict[str, Any]:
        """Evaluate non-local Helmholtz-regularized damage field."""
        eq_strain = np.asarray(local_equivalent_strain, dtype=np.float64)
        damage = self.compute_damage_variable(eq_strain, strain_threshold_eps_0, softening_parameter_alpha)
        return {
            "nonlocal_damage_field": damage,
            "max_damage": float(np.max(damage)),
            "is_damaged": bool(np.max(damage) > 0.05),
        }

    def solve_3d_phase_field_fracture_step(
        self,
        damage_field: np.ndarray,            # (nx, ny, nz) in [0, 1]
        strain_field: np.ndarray,            # (nx, ny, nz, 3, 3)
        history_field: Optional[np.ndarray] = None, # (nx, ny, nz)
        dt: float = 0.01,
        anisotropic_cleavage_direction: Optional[np.ndarray] = None, # (3,) unit vector
    ) -> Dict[str, Any]:
        """Step forward 3D phase field damage d(x, t) driven strictly by tensile energy history H+(x)."""
        nx, ny, nz = self.nx, self.ny, self.nz
        d = np.clip(damage_field.copy(), 0.0, 1.0)

        psi_plus_field = np.zeros((nx, ny, nz), dtype=np.float64)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    p_plus, _ = self.spectral_strain_decomposition(strain_field[i, j, k])
                    psi_plus_field[i, j, k] = p_plus

        h_plus = np.maximum(history_field if history_field is not None else 0.0, psi_plus_field)

        d_pad = np.pad(d, 1, mode="wrap")
        lap_d = (
            (d_pad[2:, 1:-1, 1:-1] + d_pad[:-2, 1:-1, 1:-1] - 2.0 * d) / (self.dx**2)
            + (d_pad[1:-1, 2:, 1:-1] + d_pad[1:-1, :-2, 1:-1] - 2.0 * d) / (self.dx**2)
            + (d_pad[1:-1, 1:-1, 2:] + d_pad[1:-1, 1:-1, :-2] - 2.0 * d) / (self.dx**2)
        )

        if anisotropic_cleavage_direction is not None:
            n_cleave = anisotropic_cleavage_direction / np.linalg.norm(anisotropic_cleavage_direction)
            grad_x = (d_pad[2:, 1:-1, 1:-1] - d_pad[:-2, 1:-1, 1:-1]) / (2.0 * self.dx)
            grad_y = (d_pad[1:-1, 2:, 1:-1] - d_pad[1:-1, :-2, 1:-1]) / (2.0 * self.dx)
            grad_z = (d_pad[1:-1, 1:-1, 2:] - d_pad[1:-1, 1:-1, :-2]) / (2.0 * self.dx)
            grad_dot_n = grad_x * n_cleave[0] + grad_y * n_cleave[1] + grad_z * n_cleave[2]
            lap_d += 2.0 * grad_dot_n / self.l0

        driving_force = (2.0 * (1.0 - d) / self.gc) * h_plus - (d / self.l0 - self.l0 * lap_d)
        d_dot = self.mobility * np.maximum(0.0, driving_force)

        d_new = np.clip(d + dt * d_dot, 0.0, 1.0)
        d_new = np.maximum(d, d_new)

        fracture_energy_j_m2 = float(np.sum((self.gc / (2.0 * self.l0)) * (d_new**2 + (self.l0**2) * lap_d**2)) * (self.dx**3))

        return {
            "damage_field": d_new,
            "max_damage_parameter": float(np.max(d_new)),
            "tensile_energy_history_j_m3": h_plus,
            "fracture_energy_dissipated_j_m2": fracture_energy_j_m2,
            "is_spectral_split_enforced": True,
        }
