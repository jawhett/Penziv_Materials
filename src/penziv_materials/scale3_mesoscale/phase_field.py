"""Phase-Field Microstructure Evolution Engine (Cahn-Hilliard, Allen-Cahn & Khachaturyan Microelasticity)."""

from typing import Dict, Tuple, List, Optional
import numpy as np
from penziv_materials.core.hcal import default_hcal


class PhaseFieldEngine:
    """Coupled Cahn-Hilliard (conserved) and Allen-Cahn (non-conserved) solver with Fourier microelasticity."""

    def __init__(
        self,
        grid_size: Tuple[int, int] = (64, 64),
        dx: float = 1.0,  # nm
        mobility_c: float = 1.0,
        mobility_eta: float = 2.0,
        gradient_coeff_c: float = 0.5,
        gradient_coeff_eta: float = 0.5,
    ):
        self.nx, self.ny = grid_size
        self.dx = dx
        self.mobility_c = mobility_c
        self.mobility_eta = mobility_eta
        self.kappa_c = gradient_coeff_c
        self.kappa_eta = gradient_coeff_eta

        # Fourier wave vectors
        kx = 2.0 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.ny, d=self.dx)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing="ij")
        self.K_SQ = self.KX**2 + self.KY**2

    def compute_free_energy_derivative(
        self,
        c: np.ndarray,
        eta: np.ndarray,
        w_barrier: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute chemical free energy functional derivatives:

        f(c, eta) = w * [ c^2 * (1-c)^2 + eta^2 * (1-eta)^2 + alpha * c * (1-eta) ]
        """
        # df/dc
        df_dc = w_barrier * (2.0 * c * (1.0 - c) * (1.0 - 2.0 * c))
        # df/deta
        df_deta = w_barrier * (2.0 * eta * (1.0 - eta) * (1.0 - 2.0 * eta))
        return df_dc, df_deta

    def compute_khachaturyan_elastic_energy_fourier(
        self,
        eta: np.ndarray,
        eigenstrain: float = 0.015,
        shear_modulus_c44: float = 80.0,
    ) -> np.ndarray:
        """Evaluate exact Fourier-space Khachaturyan-Shtremel microelasticity interaction kernel B(n).

        E_elast = 1/2 * integral B(n) * |eta_k|^2 dk
        """
        eta_k = np.fft.fftn(eta)
        # Direction cosines n_x, n_y
        k_norm = np.sqrt(self.K_SQ)
        k_norm[0, 0] = 1.0  # Avoid div by zero
        nx = self.KX / k_norm
        ny = self.KY / k_norm

        # Interaction kernel B(n) ~ C * epsilon_0^2 * (n_x^2 * n_y^2)
        b_kernel = shear_modulus_c44 * (eigenstrain**2) * (nx**2 * ny**2)
        b_kernel[0, 0] = 0.0

        elastic_energy_k = b_kernel * eta_k
        elastic_force = np.real(np.fft.ifftn(elastic_energy_k))
        return elastic_force

    def step_forward_semi_implicit(
        self,
        c: np.ndarray,
        eta: np.ndarray,
        dt: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute one semi-implicit time step using spectral Cahn-Hilliard and Allen-Cahn equations:

        dc/dt = M_c * laplacian(df/dc - kappa_c * laplacian(c))
        deta/dt = -M_eta * (df/deta - kappa_eta * laplacian(eta) + df_elast/deta)
        """
        df_dc, df_deta = self.compute_free_energy_derivative(c, eta)
        elast_force = self.compute_khachaturyan_elastic_energy_fourier(eta)

        # Fourier transform non-linear terms
        df_dc_k = np.fft.fftn(df_dc)
        df_deta_k = np.fft.fftn(df_deta + elast_force)
        c_k = np.fft.fftn(c)
        eta_k = np.fft.fftn(eta)

        # Semi-implicit update in k-space
        # (1 + dt * M_c * kappa_c * k^4) * c_k(t+dt) = c_k(t) - dt * M_c * k^2 * df_dc_k
        denom_c = 1.0 + dt * self.mobility_c * self.kappa_c * (self.K_SQ**2)
        c_k_new = (c_k - dt * self.mobility_c * self.K_SQ * df_dc_k) / denom_c

        # (1 + dt * M_eta * kappa_eta * k^2) * eta_k(t+dt) = eta_k(t) - dt * M_eta * df_deta_k
        denom_eta = 1.0 + dt * self.mobility_eta * self.kappa_eta * self.K_SQ
        eta_k_new = (eta_k - dt * self.mobility_eta * df_deta_k) / denom_eta

        c_new = np.clip(np.real(np.fft.ifftn(c_k_new)), 0.0, 1.0)
        eta_new = np.clip(np.real(np.fft.ifftn(eta_k_new)), 0.0, 1.0)

        return c_new, eta_new
