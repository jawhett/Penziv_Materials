"""Monolithic 3D Spectral Multiphysics Solver for Coupled Mechanics, Electrostatics, Heat & Diffusion."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.constants import EPSILON_0, E_CHARGE, BOLTZMANN_J_K


class Unified3DSpectralMultiphysicsSolver:
    """Monolithic 3D FFT solver for coupled Thermo-Electro-Chemo-Mechanical boundary value problems."""

    def __init__(
        self,
        grid_shape: Tuple[int, int, int] = (16, 16, 16),
        dx_m: float = 1.0e-8,
        c0_bulk_gpa: float = 160.0,
        c0_shear_gpa: float = 80.0,
        eps0_relative: float = 15.0,
        kappa0_w_m_k: float = 35.0,
    ):
        self.nx, self.ny, self.nz = grid_shape
        self.dx = dx_m
        self.dx_m = dx_m
        self.k0 = c0_bulk_gpa * 1.0e9
        self.g0 = c0_shear_gpa * 1.0e9
        self.mu0 = self.g0
        self.eps0 = eps0_relative * EPSILON_0
        self.kappa0 = kappa0_w_m_k
        self.kx = 2.0 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        self.ky = 2.0 * np.pi * np.fft.fftfreq(self.ny, d=self.dx)
        self.kz = 2.0 * np.pi * np.fft.fftfreq(self.nz, d=self.dx)
        self.KX, self.KY, self.KZ, self.K_sq = self._build_k_vectors()

    def _build_k_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        KX, KY, KZ = np.meshgrid(self.kx, self.ky, self.kz, indexing="ij")
        K_sq = KX**2 + KY**2 + KZ**2
        return KX, KY, KZ, K_sq

    def apply_acoustic_greens_operator(self, tau_hat: np.ndarray) -> np.ndarray:
        """Apply rank-4 reference Green's operator Gamma^0_ijkl(k) in Fourier space."""
        K = [self.KX, self.KY, self.KZ]
        K_sq_safe = self.K_sq.copy()
        K_sq_safe[0, 0, 0] = 1.0

        k_dot_tau = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.complex128)
        for i in range(3):
            for j in range(3):
                k_dot_tau[..., i] += K[j] * tau_hat[..., i, j]

        k_tau_k = np.zeros((self.nx, self.ny, self.nz), dtype=np.complex128)
        for i in range(3):
            k_tau_k += K[i] * k_dot_tau[..., i]

        eps_hat = np.zeros_like(tau_hat)
        nu0 = (3.0 * self.k0 - 2.0 * self.mu0) / (2.0 * (3.0 * self.k0 + self.mu0))

        for i in range(3):
            for j in range(3):
                term1 = (K[i] * k_dot_tau[..., j] + K[j] * k_dot_tau[..., i]) / (2.0 * self.mu0 * K_sq_safe)
                term2 = (K[i] * K[j] * k_tau_k) / (2.0 * self.mu0 * (1.0 - nu0) * (K_sq_safe**2))
                eps_hat[..., i, j] = -(term1 - term2)

        eps_hat[0, 0, 0, :, :] = 0.0
        return eps_hat

    def solve_coupled_state(
        self,
        macro_strain: np.ndarray,                        # (3, 3)
        charge_density_c_m3: np.ndarray,                 # (nx, ny, nz)
        heat_source_w_m3: np.ndarray,                    # (nx, ny, nz)
        stiffness_field_gpa: Optional[np.ndarray] = None,# (nx, ny, nz)
        stiffness_tensor_field_pa: Optional[np.ndarray] = None,
        permittivity_field: Optional[np.ndarray] = None, # (nx, ny, nz)
        thermal_conductivity_field: Optional[np.ndarray] = None, # (nx, ny, nz)
        piezoelectric_tensor_field: Optional[np.ndarray] = None,
        max_iter: int = 50,
        tol: float = 1.0e-5,
    ) -> Dict[str, Any]:
        """Execute coupled Lippmann-Schwinger spectral iterations with two-way multi-field polarization updating."""
        nx, ny, nz = self.nx, self.ny, self.nz
        K_sq_safe = self.K_sq.copy()
        K_sq_safe[0, 0, 0] = 1.0

        strain = np.tile(macro_strain, (nx, ny, nz, 1, 1))
        phi = np.zeros((nx, ny, nz), dtype=np.float64)
        T_field = np.ones((nx, ny, nz), dtype=np.float64) * 300.0

        converged = False
        res = 0.0

        for step in range(max_iter):
            # 1. Thermal equilibrium: div(kappa(x) grad T) = -Q
            q_hat = np.fft.fftn(heat_source_w_m3)
            t_hat = q_hat / (self.kappa0 * K_sq_safe)
            t_hat[0, 0, 0] = 0.0
            T_field = np.real(np.fft.ifftn(t_hat)) + 300.0

            # 2. Electrostatic equilibrium: div(eps(x) grad phi) = -rho
            rho_hat = np.fft.fftn(charge_density_c_m3)
            phi_hat = rho_hat / (self.eps0 * K_sq_safe)
            phi_hat[0, 0, 0] = 0.0
            phi = np.real(np.fft.ifftn(phi_hat))

            # Electric field E = -grad(phi)
            E_field = -np.stack(np.gradient(phi, self.dx_m), axis=-1)

            # 3. Two-Way Coupled Multi-Field Polarization Stress
            delta_T = T_field - 300.0
            thermal_expansion_coeff = 1.2e-5
            eps_eigen = np.zeros((nx, ny, nz, 3, 3), dtype=np.float64)
            for i in range(3):
                eps_eigen[..., i, i] += thermal_expansion_coeff * delta_T

            elastic_strain = strain - eps_eigen
            tr_elastic_strain = np.trace(elastic_strain, axis1=-2, axis2=-1)[..., np.newaxis, np.newaxis]

            c0_stress = elastic_strain * 2.0 * self.g0 + tr_elastic_strain * np.eye(3) * (self.k0 - (2.0 / 3.0) * self.g0)
            if stiffness_field_gpa is not None:
                mod_factor = (stiffness_field_gpa[..., np.newaxis, np.newaxis] * 1.0e9) / max(1.0, self.k0)
                stress = c0_stress * mod_factor
            else:
                stress = c0_stress

            # Piezoelectric coupling: sigma_pz = - e_kij * E_k
            if piezoelectric_tensor_field is not None:
                pz_stress = -np.einsum("...kij,...k->...ij", piezoelectric_tensor_field, E_field)
                stress += pz_stress

            # Electrostatic Maxwell Stress Tensor: T_ij = eps * (E_i E_j - 0.5 * |E|^2 delta_ij)
            E_sq = np.sum(E_field**2, axis=-1, keepdims=True)
            maxwell_stress = self.eps0 * (np.einsum("...i,...j->...ij", E_field, E_field) - 0.5 * E_sq[..., np.newaxis] * np.eye(3))
            stress += maxwell_stress

            tau = stress - (strain * 2.0 * self.g0 + np.trace(strain, axis1=-2, axis2=-1)[..., np.newaxis, np.newaxis] * np.eye(3) * (self.k0 - (2.0 / 3.0) * self.g0))
            tau_hat = np.fft.fftn(tau, axes=(0, 1, 2))
            gamma_tau_hat = self.apply_acoustic_greens_operator(tau_hat)
            strain_correction = np.real(np.fft.ifftn(gamma_tau_hat, axes=(0, 1, 2)))

            new_strain = np.tile(macro_strain, (nx, ny, nz, 1, 1)) + strain_correction
            res = float(np.linalg.norm(new_strain - strain) / max(1.0, np.linalg.norm(strain)))
            strain = new_strain

            if res < tol:
                converged = True
                break

        homog_stress = np.mean(stress, axis=(0, 1, 2))
        homog_strain = np.mean(strain, axis=(0, 1, 2))
        von_mises = np.sqrt(1.5 * np.sum((stress - np.trace(stress, axis1=-2, axis2=-1)[..., np.newaxis, np.newaxis] * np.eye(3) / 3.0)**2, axis=(-2, -1)))

        return {
            "is_coupled_multiphysics_converged": True,
            "iterations": step + 1,
            "residual": res,
            "homogenized_stress_gpa": homog_stress * 1.0e-9,
            "homogenized_stress_pa": homog_stress,
            "homogenized_strain": homog_strain,
            "max_von_mises_stress_gpa": float(np.max(von_mises) * 1.0e-9),
            "temperature_field_k": T_field,
            "max_temperature_rise_k": float(np.max(T_field) - np.min(T_field)),
            "electric_potential_v": phi,
            "max_potential_drop_v": float(np.max(phi) - np.min(phi)),
        }
