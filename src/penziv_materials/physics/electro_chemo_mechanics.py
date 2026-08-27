"""Coupled Poisson-Nernst-Planck (PNP) Electro-Chemo-Mechanics with Heterogeneous Dielectric Tensors."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.constants import E_CHARGE, BOLTZMANN_J_K, EPSILON_0


class CoupledPNPMechanicsSolver:
    """3D Finite-Difference / Spectral solver for coupled Poisson-Nernst-Planck and electro-chemo-mechanics in solid-state electrolytes."""

    def __init__(
        self,
        grid_shape: Tuple[int, int, int] = (16, 16, 16),
        dx_nm: float = 0.5,
        temperature_k: float = 300.0,
    ):
        self.nx, self.ny, self.nz = grid_shape
        self.dx_m = dx_nm * 1.0e-9
        self.T = temperature_k
        self.kbt_j = BOLTZMANN_J_K * self.T

    def solve_space_charge_potential_3d(
        self,
        charge_density_c_m3: np.ndarray,
        relative_permittivity_field: Optional[np.ndarray] = None,
        relative_permittivity: float = 25.0,
        max_iter: int = 30,
        tol: float = 1e-4,
    ) -> Dict[str, Any]:
        """Solve heterogeneous Poisson equation:

        nabla . ( eps(r) * nabla phi(r) ) = - rho(r)
        via iterative spectral polarization updates:
        eps_0 * nabla^2 phi^(k+1) = - rho(r) + nabla . P_pol^(k)(r)
        where P_pol(r) = (eps(r) - eps_0) * (-nabla phi(r)).
        """
        nx, ny, nz = self.nx, self.ny, self.nz
        rho = np.asarray(charge_density_c_m3, dtype=np.float64)

        if relative_permittivity_field is not None:
            eps_r_field = np.asarray(relative_permittivity_field, dtype=np.float64)
        else:
            eps_r_field = np.ones((nx, ny, nz), dtype=np.float64) * relative_permittivity

        eps_0_scalar = float(np.mean(eps_r_field)) * EPSILON_0

        kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=self.dx_m)
        ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=self.dx_m)
        kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=self.dx_m)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        K_sq = KX**2 + KY**2 + KZ**2
        K_sq[0, 0, 0] = 1.0  # Avoid division by zero at k=0

        rho_hat = np.fft.fftn(rho)
        phi = np.real(np.fft.ifftn(rho_hat / (eps_0_scalar * K_sq)))
        phi -= np.mean(phi)

        converged = False
        for _ in range(max_iter):
            # Compute electric field E = -nabla phi
            grad_x = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2.0 * self.dx_m)
            grad_y = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * self.dx_m)
            grad_z = (np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)) / (2.0 * self.dx_m)

            # Polarization P_pol = (eps(r) - eps_0) * (-grad phi)
            delta_eps = (eps_r_field * EPSILON_0) - eps_0_scalar
            Px = -delta_eps * grad_x
            Py = -delta_eps * grad_y
            Pz = -delta_eps * grad_z

            # Divergence of polarization in Fourier space: i k . P_hat
            Px_hat = np.fft.fftn(Px)
            Py_hat = np.fft.fftn(Py)
            Pz_hat = np.fft.fftn(Pz)
            div_P_hat = 1j * (KX * Px_hat + KY * Py_hat + KZ * Pz_hat)

            # Updated potential in Fourier space
            phi_hat_new = (rho_hat - div_P_hat) / (eps_0_scalar * K_sq)
            phi_hat_new[0, 0, 0] = 0.0

            phi_new = np.real(np.fft.ifftn(phi_hat_new))
            phi_new -= np.mean(phi_new)

            err = float(np.linalg.norm(phi_new - phi) / max(1e-12, np.linalg.norm(phi)))
            phi = phi_new
            if err < tol:
                converged = True
                break

        e_field_max = float(np.max(np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)) * 1.0e-2)  # V/cm -> MV/cm scaling

        return {
            "electric_potential_v": phi,
            "max_potential_drop_v": float(np.max(phi) - np.min(phi)),
            "max_electric_field_v_m": float(np.max(np.sqrt(grad_x**2 + grad_y**2 + grad_z**2))),
            "converged": converged,
        }

    def compute_interfacial_chemomechanical_stress(
        self,
        applied_current_density_ma_cm2: float,
        overpotential_v: float = 0.05,
        shear_modulus_gpa: float = 35.0,
        molar_volume_cm3_mol: float = 13.0,
    ) -> Dict[str, float]:
        """Monroe-Newman / Butler-Volmer stability of electrodeposited metal/electrolyte interface."""
        j_a_m2 = applied_current_density_ma_cm2 * 10.0
        omega_m3_mol = molar_volume_cm3_mol * 1.0e-6
        omega_per_atom = omega_m3_mol / 6.02214076e23

        # Electro-chemo-mechanical overpotential shift: Delta mu_mech = 2 * G * eps_trans * Omega
        sigma_hydro_pa = (shear_modulus_gpa * 1.0e9) * 0.015
        delta_v_mech = (sigma_hydro_pa * omega_per_atom) / E_CHARGE

        eff_overpotential = overpotential_v + delta_v_mech
        # Monroe-Newman criterion: G_electrolyte / G_Li > 1.8 for planar electrodeposition stability
        is_dendrite_suppressed = bool(shear_modulus_gpa >= 8.5)

        return {
            "hydrostatic_interfacial_stress_mpa": float(sigma_hydro_pa * 1.0e-6),
            "mechano_electrochemical_overpotential_v": float(eff_overpotential),
            "is_electrodeposition_mechanically_stable": is_dendrite_suppressed,
        }
