"""Exact FFT-based Khachaturyan-Shatalov Microelasticity Solver for Anisotropic Stiffness & Multi-Variant Eigenstrains."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np


class SpectralMicroelasticitySolver:
    """Exact FFT-based Khachaturyan-Shatalov microelasticity solver for arbitrary anisotropic stiffness and eigenstrain fields."""

    @staticmethod
    def solve_elastic_fields(
        eigenstrain_field: np.ndarray,          # Shape: (nx, ny, nz, 3, 3)
        C_ijkl: np.ndarray,                      # Shape: (3, 3, 3, 3) or (6, 6) Voigt matrix
        dx_nm: float = 1.0,
        applied_strain: Optional[np.ndarray] = None, # Shape: (3, 3) macroscopic applied strain
    ) -> Dict[str, np.ndarray]:
        """Solve mechanical equilibrium nabla . sigma = 0 in Fourier space.

        Equations:
            Omega_ik(n) = C_ijkl n_j n_l
            G_ik(n) = [Omega_ik(n)]^-1
            sigma0_ij(k) = C_ijkl * eps0_kl(k)
            u_i(k) = -i * G_ik(n) * sigma0_kj(k) * n_j / |k|
            eps_tot_ij(k) = 0.5 * (k_j u_i + k_i u_j)
            eps_el(r) = eps_tot(r) - eps0(r)
            sigma(r) = C_ijkl : eps_el(r)
        """
        nx, ny, nz, _, _ = eigenstrain_field.shape

        # Convert Voigt 6x6 to 4th order if needed
        if C_ijkl.shape == (6, 6):
            v_map = {0: (0, 0), 1: (1, 1), 2: (2, 2), 3: (1, 2), 4: (0, 2), 5: (0, 1)}
            C4 = np.zeros((3, 3, 3, 3), dtype=np.float64)
            for a in range(6):
                i, j = v_map[a]
                for b in range(6):
                    k, l = v_map[b]
                    val = C_ijkl[a, b]
                    C4[i, j, k, l] = val
                    C4[j, i, k, l] = val
                    C4[i, j, l, k] = val
                    C4[j, i, l, k] = val
        else:
            C4 = C_ijkl.astype(np.float64)

        kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx_nm)
        ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dx_nm)
        kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=dx_nm)
        Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing="ij")
        K = np.stack([Kx, Ky, Kz], axis=-1)  # (nx, ny, nz, 3)
        k_norm = np.linalg.norm(K, axis=-1, keepdims=True)
        k_unit = np.zeros_like(K)
        non_zero = (k_norm[..., 0] > 1e-8)
        k_unit[non_zero] = K[non_zero] / k_norm[non_zero]

        # Fourier transform of eigenstrain field
        eps0_k = np.fft.fftn(eigenstrain_field, axes=(0, 1, 2))  # (nx, ny, nz, 3, 3)

        # Polarization stress tensor in k-space: sigma0_ij(k) = C_ijkl * eps0_kl(k)
        sigma0_k = np.einsum("ijkl,...kl->...ij", C4, eps0_k)

        # Acoustic tensor in Fourier space: Omega_ik(n) = C_ijkl * n_j * n_l
        Omega = np.einsum("ijkl,...j,...l->...ik", C4, k_unit, k_unit)  # (nx, ny, nz, 3, 3)

        # Acoustic Green's tensor G_ik(n) = Omega_ik^-1
        G_k = np.zeros_like(Omega)
        G_k[non_zero] = np.linalg.inv(Omega[non_zero])

        # Displacement field in Fourier space: u_i(k) = -i * G_ik(n) * sigma0_kj(k) * n_j / |k|
        sigma0_k_n = np.einsum("...ij,...j->...i", sigma0_k, k_unit)
        u_k = np.zeros((nx, ny, nz, 3), dtype=np.complex128)
        u_k[non_zero] = -1j * np.einsum("...ik,...k->...i", G_k[non_zero], sigma0_k_n[non_zero]) / k_norm[non_zero, 0]

        # Total strain field in Fourier space: eps_tot_ij(k) = 0.5 * (k_j u_i + k_i u_j)
        eps_tot_k = 0.5 * (
            np.einsum("...j,...i->...ij", 1j * K, u_k)
            + np.einsum("...i,...j->...ij", 1j * K, u_k)
        )

        # Average strain compatibility (k = 0)
        if applied_strain is not None:
            eps_tot_k[0, 0, 0] = applied_strain * float(nx * ny * nz)
        else:
            eps_tot_k[0, 0, 0] = 0.0

        total_strain = np.real(np.fft.ifftn(eps_tot_k, axes=(0, 1, 2)))
        elastic_strain = total_strain - eigenstrain_field
        stress_field = np.einsum("ijkl,...kl->...ij", C4, elastic_strain)
        elastic_energy_density = 0.5 * np.einsum("...ij,...ij->...", stress_field, elastic_strain)

        return {
            "total_strain": total_strain,
            "elastic_strain": elastic_strain,
            "stress_field": stress_field,
            "elastic_energy_density": elastic_energy_density,
            "displacement_field": np.real(np.fft.ifftn(u_k, axes=(0, 1, 2))),
        }
