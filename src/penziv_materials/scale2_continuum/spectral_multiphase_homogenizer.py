"""Full-Field Multi-Phase Spectral FFT Homogenizer for Conductivity, Elasticity & Permittivity."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np


class SpectralMultiphaseHomogenizer:
    """3D Lippmann-Schwinger FFT homogenizer for arbitrary N-phase heterogeneous microstructures."""

    def __init__(self, grid_shape: Tuple[int, int, int] = (16, 16, 16), dx_m: float = 1.0e-6):
        self.nx, self.ny, self.nz = grid_shape
        self.dx = dx_m
        self.KX, self.KY, self.KZ, self.K_sq = self._build_k_vectors()

    def _build_k_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        kx = 2.0 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.ny, d=self.dx)
        kz = 2.0 * np.pi * np.fft.fftfreq(self.nz, d=self.dx)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        K_sq = KX**2 + KY**2 + KZ**2
        K_sq[0, 0, 0] = 1.0
        return KX, KY, KZ, K_sq

    def homogenize_conductivity_tensor(
        self,
        local_conductivity_field: np.ndarray,  # (nx, ny, nz, 3, 3) or (nx, ny, nz)
        applied_macro_gradient: np.ndarray = np.array([1.0, 0.0, 0.0]),
        max_iter: int = 40,
        tol: float = 1e-5,
    ) -> Dict[str, Any]:
        """Solve div(kappa(r) . grad(T(r))) = 0 under PBC to compute effective conductivity tensor kappa_eff."""
        nx, ny, nz = self.nx, self.ny, self.nz

        if local_conductivity_field.ndim == 3:
            cond_3x3 = np.zeros((nx, ny, nz, 3, 3), dtype=np.float64)
            for i in range(3):
                cond_3x3[..., i, i] = local_conductivity_field
        else:
            cond_3x3 = np.asarray(local_conductivity_field, dtype=np.float64)

        k0_scalar = float(np.mean(cond_3x3[..., 0, 0]))
        grad_T = np.tile(applied_macro_gradient, (nx, ny, nz, 1))

        flux = np.zeros_like(grad_T)
        for _ in range(max_iter):
            # Flux J(r) = -kappa(r) . grad_T(r)
            flux = -np.einsum("abcij,abcj->abci", cond_3x3, grad_T)
            # Polarization field P(r) = -J(r) - kappa_0 * grad_T(r)
            pol = -flux - k0_scalar * grad_T

            pol_hat = np.fft.fftn(pol, axes=(0, 1, 2))
            k_dot_p = self.KX * pol_hat[..., 0] + self.KY * pol_hat[..., 1] + self.KZ * pol_hat[..., 2]

            grad_hat = np.zeros_like(pol_hat)
            grad_hat[..., 0] = -(self.KX * k_dot_p) / (k0_scalar * self.K_sq)
            grad_hat[..., 1] = -(self.KY * k_dot_p) / (k0_scalar * self.K_sq)
            grad_hat[..., 2] = -(self.KZ * k_dot_p) / (k0_scalar * self.K_sq)
            grad_hat[0, 0, 0, :] = 0.0

            grad_correction = np.real(np.fft.ifftn(grad_hat, axes=(0, 1, 2)))
            new_grad_T = np.tile(applied_macro_gradient, (nx, ny, nz, 1)) + grad_correction

            err = float(np.linalg.norm(new_grad_T - grad_T) / max(1e-12, np.linalg.norm(grad_T)))
            grad_T = new_grad_T
            if err < tol:
                break

        macro_flux = np.mean(flux, axis=(0, 1, 2))
        macro_grad = np.mean(grad_T, axis=(0, 1, 2))
        kappa_eff = -macro_flux / np.maximum(1e-12, macro_grad)

        return {
            "effective_conductivity_w_m_k": kappa_eff,
            "isotropic_effective_conductivity": float(np.mean(kappa_eff)),
            "macro_flux": macro_flux,
            "macro_gradient": macro_grad,
            "iterations": _ + 1,
        }
