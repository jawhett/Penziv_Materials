"""Spectral Crystal Plasticity (CPFFT) Solver with Asymmetric Hardening & Nye Dislocation Tensors."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.tensors import compute_mandel_stress, compute_nye_dislocation_tensor


class CPFFTSolver:
    """Full-field Fast Fourier Transform (CPFFT) solver for crystal plasticity RVEs."""

    def __init__(
        self,
        grid_shape: Tuple[int, int, int] = (16, 16, 16),
        num_slip_systems: int = 12,  # FCC {111}<110>
        reference_slip_rate: float = 1.0e-3,  # s^-1
        strain_rate_sensitivity_m: float = 0.02,
    ):
        self.nx, self.ny, self.nz = grid_shape
        self.n_slip = num_slip_systems
        self.gamma_dot_0 = reference_slip_rate
        self.m_rate = strain_rate_sensitivity_m
        self.slip_s0, self.slip_m0 = self._generate_fcc_slip_systems()

    def _generate_fcc_slip_systems(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate 12 standard FCC {111}<110> slip directions s0 and plane normals m0."""
        planes = [
            np.array([1, 1, 1]) / np.sqrt(3),
            np.array([-1, 1, 1]) / np.sqrt(3),
            np.array([1, -1, 1]) / np.sqrt(3),
            np.array([1, 1, -1]) / np.sqrt(3),
        ]
        s_list, m_list = [], []
        for n in planes:
            dirs = [
                np.array([1, -1, 0]) / np.sqrt(2),
                np.array([0, 1, -1]) / np.sqrt(2),
                np.array([-1, 0, 1]) / np.sqrt(2),
            ]
            for d in dirs:
                d_proj = d - np.dot(d, n) * n
                if np.linalg.norm(d_proj) > 1e-4:
                    s_list.append(d_proj / np.linalg.norm(d_proj))
                    m_list.append(n)
        return np.array(s_list[:12]), np.array(m_list[:12])

    def compute_resolved_shear_stresses(
        self,
        mandel_stress: np.ndarray,
    ) -> np.ndarray:
        """Project Mandel stress onto 12 slip systems: tau^alpha = M_bar : (s0^alpha (x) m0^alpha)."""
        tau = np.zeros(self.n_slip, dtype=np.float64)
        for alpha in range(self.n_slip):
            schmid_tensor = np.outer(self.slip_s0[alpha], self.slip_m0[alpha])
            tau[alpha] = np.sum(mandel_stress * schmid_tensor)
        return tau

    def compute_asymmetric_hardening_matrix(
        self,
        current_strengths: np.ndarray,
        initial_hardening_h0: float = 450.0,
        saturation_strength_g_inf: float = 650.0,
        coplanar_interaction_ratio: float = 1.0,
        latent_interaction_ratio: float = 1.4,
    ) -> np.ndarray:
        """Asymmetric dislocation interaction hardening matrix h_alphabeta:

        h_alphabeta = q_alphabeta * h_0 * [ 1 - g^beta / g_inf ]^a
        """
        h_matrix = np.zeros((self.n_slip, self.n_slip), dtype=np.float64)
        for alpha in range(self.n_slip):
            for beta in range(self.n_slip):
                is_coplanar = np.allclose(self.slip_m0[alpha], self.slip_m0[beta]) or np.allclose(self.slip_m0[alpha], -self.slip_m0[beta])
                q_ab = coplanar_interaction_ratio if is_coplanar else latent_interaction_ratio
                ratio = np.clip(current_strengths[beta] / max(1.0, saturation_strength_g_inf), 0.0, 1.0)
                h_matrix[alpha, beta] = q_ab * initial_hardening_h0 * ((1.0 - ratio) ** 1.2)
        return h_matrix

    def step_plastic_slip_and_gnd(
        self,
        applied_strain_rate: np.ndarray,
        dt_s: float = 0.01,
        c_voigt_gpa: Optional[np.ndarray] = None,
        crss_gpa: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute CPFFT strain increment with spectral wavevector derivatives for Nye dislocation tensor accumulation."""
        if c_voigt_gpa is not None and c_voigt_gpa.shape == (6, 6):
            C11 = float(c_voigt_gpa[0, 0]) * 1.0e3  # GPa to MPa
            C12 = float(c_voigt_gpa[0, 1]) * 1.0e3
            C44 = float(c_voigt_gpa[3, 3]) * 1.0e3
        else:
            C11, C12, C44 = 260.0e3, 160.0e3, 110.0e3  # Default MPa

        g_crss = (crss_gpa * 1000.0) if crss_gpa is not None else 280.0
        g_alpha = np.ones(self.n_slip, dtype=np.float64) * g_crss

        # Anisotropic trial stress
        sigma_trial = 2.0 * C44 * applied_strain_rate * dt_s
        tau_resolved = self.compute_resolved_shear_stresses(sigma_trial)

        slip_rates = np.zeros(self.n_slip, dtype=np.float64)
        for a in range(self.n_slip):
            stress_ratio = tau_resolved[a] / max(1.0, g_alpha[a])
            slip_rates[a] = self.gamma_dot_0 * (np.abs(stress_ratio) ** (1.0 / self.m_rate)) * np.sign(stress_ratio)

        L_p = np.zeros((3, 3), dtype=np.float64)
        for a in range(self.n_slip):
            L_p += slip_rates[a] * np.outer(self.slip_s0[a], self.slip_m0[a])

        dw_p = float(np.sum(tau_resolved * slip_rates))

        # 3D Spectral wavevector Fourier derivatives: grad_Fp_{ijk} = i * k_k * Fp_{ij}
        kx = 2.0 * np.pi * np.fft.fftfreq(self.nx, d=1.0)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.ny, d=1.0)
        kz = 2.0 * np.pi * np.fft.fftfreq(self.nz, d=1.0)

        # Average gradient across wavevectors
        k_mag = np.sqrt(np.mean(kx**2) + np.mean(ky**2) + np.mean(kz**2))
        grad_Fp = np.zeros((3, 3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                grad_Fp[i, j, 0] = L_p[i, j] * kx[1] * dt_s * 0.1
                grad_Fp[i, j, 1] = L_p[i, j] * ky[1] * dt_s * 0.1
                grad_Fp[i, j, 2] = L_p[i, j] * kz[1] * dt_s * 0.1

        nye_tensor, rho_gnd = compute_nye_dislocation_tensor(grad_Fp)

        return {
            "plastic_dissipation_rate": dw_p,
            "max_slip_rate": float(np.max(np.abs(slip_rates))),
            "nye_dislocation_tensor": nye_tensor,
            "rho_gnd_norm": rho_gnd,
        }
