"""3D Full-Field Lippmann-Schwinger Spectral Homogenization Solver."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np


class LippmannSchwinger3DSolver:
    """Solves static mechanical equilibrium nabla . sigma(x) = 0 and transport flux continuity nabla . J(x) = 0 via 3D FFT Green's operator."""

    def __init__(
        self,
        grid_shape: Tuple[int, int, int] = (16, 16, 16),
        c0_bulk_gpa: float = 160.0,
        c0_shear_gpa: float = 80.0,
    ):
        self.nx, self.ny, self.nz = grid_shape
        self.k0 = c0_bulk_gpa
        self.g0 = c0_shear_gpa
        # Reference isotropic stiffness tensor C^0
        self.C0_3x3x3x3 = self._build_isotropic_elastic_tensor(c0_bulk_gpa, c0_shear_gpa)
        self.k_vectors = self._build_wavevectors()

    def _build_isotropic_elastic_tensor(self, bulk_gpa: float, shear_gpa: float) -> np.ndarray:
        """Construct rank-4 isotropic elasticity tensor C^0_{ijkl} = lambda * delta_ij * delta_kl + mu * (delta_ik delta_jl + delta_il delta_jk)."""
        lam = bulk_gpa - (2.0 / 3.0) * shear_gpa
        mu = shear_gpa
        C = np.zeros((3, 3, 3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        term1 = lam * (1.0 if i == j else 0.0) * (1.0 if k == l else 0.0)
                        term2 = mu * ((1.0 if i == k else 0.0) * (1.0 if j == l else 0.0) + (1.0 if i == l else 0.0) * (1.0 if j == k else 0.0))
                        C[i, j, k, l] = term1 + term2
        return C

    def _build_wavevectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        kx = 2.0 * np.pi * np.fft.fftfreq(self.nx, d=1.0)
        ky = 2.0 * np.pi * np.fft.fftfreq(self.ny, d=1.0)
        kz = 2.0 * np.pi * np.fft.fftfreq(self.nz, d=1.0)
        return np.meshgrid(kx, ky, kz, indexing="ij")

    def _compute_acoustic_tensor_inverse(self, xi: np.ndarray) -> np.ndarray:
        """Compute K^0_{ik} = C^0_{ijkl} * xi_j * xi_l and its inverse (K^0)^-1."""
        K = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            for k in range(3):
                for j in range(3):
                    for l in range(3):
                        K[i, k] += self.C0_3x3x3x3[i, j, k, l] * xi[j] * xi[l]
        return np.linalg.pinv(K)

    def solve_heterogeneous_elastic_equilibrium(
        self,
        local_stiffness_field_gpa: np.ndarray,  # (nx, ny, nz, 6, 6) in Voigt or (nx, ny, nz) scalar scale
        macro_strain: np.ndarray,              # (3, 3)
        eigenstrain_field: Optional[np.ndarray] = None,  # (nx, ny, nz, 3, 3)
        max_iter: int = 40,
        tol: float = 1e-4,
    ) -> Dict[str, Any]:
        """Iterative Lippmann-Schwinger fixed-point resolution:

        epsilon^(k+1)(xi) = - Gamma^0(xi) : tau^k(xi),  xi != 0
        """
        nx, ny, nz = self.nx, self.ny, self.nz
        # Initialize strain field with macro strain
        strain = np.tile(macro_strain, (nx, ny, nz, 1, 1))

        KX, KY, KZ = self.k_vectors
        converged = False
        res_history = []

        for step in range(max_iter):
            # 1. Compute local Cauchy stress sigma(x) = C(x) : (epsilon(x) - epsilon^*(x))
            stress = np.zeros((nx, ny, nz, 3, 3), dtype=np.float64)
            # Simplified tensor contraction
            if local_stiffness_field_gpa.ndim == 3:
                c_scale = local_stiffness_field_gpa[..., np.newaxis, np.newaxis]
                c0_tensor = self.C0_3x3x3x3
                eff_strain = strain - (eigenstrain_field if eigenstrain_field is not None else 0.0)
                for i in range(3):
                    for j in range(3):
                        for k in range(3):
                            for l in range(3):
                                stress[..., i, j] += (c_scale[..., 0, 0] / 160.0) * c0_tensor[i, j, k, l] * eff_strain[..., k, l]
            else:
                stress = strain * 2.0 * self.g0 + np.trace(strain, axis1=-2, axis2=-1)[..., np.newaxis, np.newaxis] * np.eye(3) * (self.k0 - 2.0/3.0 * self.g0)

            # Polarization stress tau(x) = sigma(x) - C^0 : epsilon(x)
            c0_strain = np.zeros_like(stress)
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for l in range(3):
                            c0_strain[..., i, j] += self.C0_3x3x3x3[i, j, k, l] * strain[..., k, l]
            tau = stress - c0_strain

            # 2. Fourier Transform Polarization tau_hat(k)
            tau_hat = np.fft.fftn(tau, axes=(0, 1, 2))

            # 3. Apply Green operator Gamma^0 in Fourier space
            strain_hat = np.zeros_like(tau_hat)

            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        if ix == 0 and iy == 0 and iz == 0:
                            # Mean field strain preserved at k = 0
                            strain_hat[0, 0, 0] = macro_strain * (nx * ny * nz)
                            continue

                        k_vec = np.array([KX[ix, iy, iz], KY[ix, iy, iz], KZ[ix, iy, iz]])
                        k_norm = np.linalg.norm(k_vec)
                        xi = k_vec / max(1e-12, k_norm)

                        inv_K = self._compute_acoustic_tensor_inverse(xi)

                        # Gamma^0_{ikjl} = 1/4 * (inv_K_ik * xi_j * xi_l + inv_K_il * xi_j * xi_k + inv_K_jk * xi_i * xi_l + inv_K_jl * xi_i * xi_k)
                        for i in range(3):
                            for j in range(3):
                                for k in range(3):
                                    for l in range(3):
                                        gamma_0 = 0.25 * (
                                            inv_K[i, k] * xi[j] * xi[l]
                                            + inv_K[i, l] * xi[j] * xi[k]
                                            + inv_K[j, k] * xi[i] * xi[l]
                                            + inv_K[j, l] * xi[i] * xi[k]
                                        )
                                        strain_hat[ix, iy, iz, i, j] -= gamma_0 * tau_hat[ix, iy, iz, k, l]

            # 4. Inverse Fourier Transform to update real-space strain
            new_strain = np.real(np.fft.ifftn(strain_hat, axes=(0, 1, 2)))

            # 5. Convergence check on strain increment L2 norm
            rel_error = float(np.linalg.norm(new_strain - strain) / max(1e-12, np.linalg.norm(strain)))
            res_history.append(rel_error)
            strain = new_strain

            if rel_error < tol:
                converged = True
                break

        # Homogenized effective macro stress <sigma> and macro stiffness C_eff
        homog_stress = np.mean(stress, axis=(0, 1, 2))
        homog_strain = np.mean(strain, axis=(0, 1, 2))

        return {
            "converged": converged,
            "iterations": step + 1,
            "residual": float(res_history[-1]) if res_history else 0.0,
            "homogenized_stress_gpa": homog_stress,
            "homogenized_strain": homog_strain,
            "max_von_mises_stress_gpa": float(np.max(np.sqrt(1.5 * np.sum((stress - np.trace(stress, axis1=-2, axis2=-1)[..., np.newaxis, np.newaxis] * np.eye(3) / 3.0)**2, axis=(-2, -1))))),
            "strain_field_shape": strain.shape,
        }
