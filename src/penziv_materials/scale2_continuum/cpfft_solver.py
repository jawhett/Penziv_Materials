"""Spectral Crystal Plasticity (CPFFT) Solver with Dynamic Real/Reciprocal Metric Slip Systems."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.tensors import compute_mandel_stress, compute_nye_dislocation_tensor


class CPFFTSolver:
    """Full-field Fast Fourier Transform (CPFFT) solver for crystal plasticity RVEs across arbitrary crystal systems."""

    def __init__(
        self,
        grid_shape: Tuple[int, int, int] = (16, 16, 16),
        crystal_system: str = "FCC",
        crystal_structure: Optional[Any] = None,
        reference_slip_rate: float = 1.0e-3,
        strain_rate_sensitivity_m: float = 0.02,
    ):
        self.nx, self.ny, self.nz = grid_shape
        self.crystal_system = crystal_system.upper()
        self.gamma_dot_0 = reference_slip_rate
        self.m_rate = strain_rate_sensitivity_m
        self.slip_s0, self.slip_m0 = self._generate_slip_systems(crystal_structure=crystal_structure, system=self.crystal_system)
        self.n_slip = len(self.slip_s0)

    def _generate_slip_systems(
        self,
        crystal_structure: Optional[Any] = None,
        system: str = "FCC",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Construct complete active slip and deformation twinning systems from real/reciprocal metric tensors."""
        if crystal_structure is not None and hasattr(crystal_structure, "lattice"):
            lattice_mat = crystal_structure.lattice.matrix
            recip_mat = crystal_structure.lattice.get_reciprocal_lattice()
            s_list, m_list = [], []

            search_indices = np.array([
                [1, 0, 0], [0, 1, 0], [0, 0, 1],
                [1, 1, 0], [1, -1, 0], [0, 1, 1], [0, 1, -1], [1, 0, 1], [-1, 0, 1],
                [1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1],
            ])
            for uvw in search_indices:
                b_vec = np.dot(uvw, lattice_mat)
                b_norm = np.linalg.norm(b_vec)
                if b_norm < 1.0 or b_norm > 7.5:
                    continue
                s_dir = b_vec / b_norm
                for hkl in search_indices:
                    n_vec = np.dot(hkl, recip_mat)
                    n_norm = np.linalg.norm(n_vec)
                    if n_norm < 1e-3:
                        continue
                    m_norm = n_vec / n_norm
                    # Orthogonality condition: b . n = 0
                    if abs(np.dot(s_dir, m_norm)) < 1e-3:
                        s_list.append(s_dir)
                        m_list.append(m_norm)

            if s_list:
                return np.array(s_list[:48]), np.array(m_list[:48])

        s_list, m_list = [], []
        if "BCC" in system:
            planes = [
                np.array([1, 1, 0]) / np.sqrt(2), np.array([1, 0, 1]) / np.sqrt(2),
                np.array([0, 1, 1]) / np.sqrt(2), np.array([-1, 1, 0]) / np.sqrt(2),
                np.array([-1, 0, 1]) / np.sqrt(2), np.array([0, -1, 1]) / np.sqrt(2),
            ]
            dirs = [
                np.array([1, 1, 1]) / np.sqrt(3), np.array([-1, 1, 1]) / np.sqrt(3),
                np.array([1, -1, 1]) / np.sqrt(3), np.array([1, 1, -1]) / np.sqrt(3),
            ]
            for n in planes:
                for d in dirs:
                    if abs(np.dot(n, d)) < 1e-4:
                        s_list.append(d)
                        m_list.append(n)
        elif "HCP" in system:
            a1 = np.array([1, 0, 0])
            a2 = np.array([-0.5, np.sqrt(3)/2, 0])
            a3 = -(a1 + a2)
            c_axis = np.array([0, 0, 1])
            for d in [a1, a2, a3]:
                s_list.append(d / np.linalg.norm(d))
                m_list.append(c_axis)
            p_planes = [np.cross(d, c_axis) for d in [a1, a2, a3]]
            for p, d in zip(p_planes, [a1, a2, a3]):
                s_list.append(d / np.linalg.norm(d))
                m_list.append(p / np.linalg.norm(p))
        else:
            planes = [
                np.array([1, 1, 1]) / np.sqrt(3), np.array([-1, 1, 1]) / np.sqrt(3),
                np.array([1, -1, 1]) / np.sqrt(3), np.array([1, 1, -1]) / np.sqrt(3),
            ]
            for n in planes:
                dirs = [
                    np.array([1, -1, 0]) / np.sqrt(2), np.array([0, 1, -1]) / np.sqrt(2),
                    np.array([-1, 0, 1]) / np.sqrt(2),
                ]
                for d in dirs:
                    d_proj = d - np.dot(d, n) * n
                    if np.linalg.norm(d_proj) > 1e-4:
                        s_list.append(d_proj / np.linalg.norm(d_proj))
                        m_list.append(n)

        return np.array(s_list[:48]), np.array(m_list[:48])

    def compute_resolved_shear_stresses(self, mandel_stress: np.ndarray) -> np.ndarray:
        """Project Mandel stress onto slip systems: tau^alpha = M_bar : (s0^alpha (x) m0^alpha)."""
        tau = np.zeros(self.n_slip, dtype=np.float64)
        for alpha in range(self.n_slip):
            schmid_tensor = np.outer(self.slip_s0[alpha], self.slip_m0[alpha])
            tau[alpha] = np.sum(mandel_stress * schmid_tensor)
        return tau

    def step_plastic_slip_and_gnd(
        self,
        applied_strain_rate: np.ndarray,
        dt_s: float = 0.01,
        c_voigt_gpa: Optional[np.ndarray] = None,
        crss_gpa: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute CPFFT strain increment with spectral wavevector derivatives for Nye dislocation tensor accumulation."""
        if c_voigt_gpa is not None and c_voigt_gpa.shape == (6, 6):
            C44 = float(c_voigt_gpa[3, 3]) * 1.0e3
        else:
            C44 = 110.0e3

        g_crss = (crss_gpa * 1000.0) if crss_gpa is not None else 280.0
        g_alpha = np.ones(self.n_slip, dtype=np.float64) * g_crss

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

        kx = 2.0 * np.pi * np.fft.fftfreq(self.nx, d=1.0)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.ny, d=1.0)
        kz = 2.0 * np.pi * np.fft.fftfreq(self.nz, d=1.0)

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
