"""Full-Field Multi-Phase Spectral FFT Homogenizer for Conductivity, Elasticity & Permittivity across Arbitrary N-Phase Microstructures."""

from typing import Dict, Tuple, List, Optional, Any, Union
import numpy as np


class SpectralMultiphaseHomogenizer:
    """3D Lippmann-Schwinger FFT homogenizer for arbitrary N-phase heterogeneous microstructures and anisotropic property fields."""

    def __init__(self, grid_shape: Tuple[int, int, int] = (16, 16, 16), dx_m: float = 1.0e-6):
        self.grid_shape = grid_shape
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

    def assemble_n_phase_tensor_field(
        self,
        phase_map: np.ndarray,                                     # (nx, ny, nz) integer phase IDs or (nx, ny, nz, N)
        phase_tensors: Dict[Union[int, str], np.ndarray],           # phase_id -> tensor (e.g. (3,3) or (3,3,3,3))
    ) -> np.ndarray:
        """Construct local anisotropic tensor field T(x) = sum_alpha phi_alpha(x) * T^alpha."""
        nx, ny, nz = self.nx, self.ny, self.nz
        # Sample tensor shape
        sample_tensor = next(iter(phase_tensors.values()))
        tensor_shape = sample_tensor.shape

        field = np.zeros((nx, ny, nz) + tensor_shape, dtype=np.float64)

        if phase_map.ndim == 3:
            # Discrete integer phase map
            unique_ids = np.unique(phase_map)
            for pid in unique_ids:
                mask = (phase_map == pid)
                tensor_val = phase_tensors.get(pid, phase_tensors.get(str(pid), sample_tensor))
                field[mask] = tensor_val
        elif phase_map.ndim == 4:
            # Continuous order parameter / volume fraction fields phi_alpha(x)
            for pid_idx, (pid, tensor_val) in enumerate(phase_tensors.items()):
                if pid_idx < phase_map.shape[-1]:
                    phi_alpha = phase_map[..., pid_idx, np.newaxis, np.newaxis]
                    field += phi_alpha * tensor_val

        return field

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
        for step in range(max_iter):
            # Flux J(r) = -kappa(r) . grad_T(r)
            flux = -np.einsum("...ij,...j->...i", cond_3x3, grad_T)
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
            "iterations": step + 1,
        }

    def homogenize_elastic_stiffness_tensor(
        self,
        local_stiffness_field_c4: np.ndarray,      # (nx, ny, nz, 3, 3, 3, 3)
        eigenstrain_field: Optional[np.ndarray] = None, # (nx, ny, nz, 3, 3)
        max_iter: int = 50,
        tol: float = 1e-5,
    ) -> Dict[str, Any]:
        """Compute full 4th-order effective homogenized elastic stiffness tensor C_eff_ijkl under 6 orthogonal unit strain states."""
        nx, ny, nz = self.nx, self.ny, self.nz
        c_eff_6x6 = np.zeros((6, 6))

        # Reference isotropic moduli from spatial average
        c_mean = np.mean(local_stiffness_field_c4, axis=(0, 1, 2))
        c0_bulk = float(c_mean[0, 0, 0, 0] + 2.0 * c_mean[0, 0, 1, 1]) / 3.0
        c0_shear = float(c_mean[0, 1, 0, 1])

        # 6 unit Voigt strain modes
        voigt_map = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

        for col, (i_idx, j_idx) in enumerate(voigt_map):
            macro_strain = np.zeros((3, 3))
            if i_idx == j_idx:
                macro_strain[i_idx, j_idx] = 1.0
            else:
                macro_strain[i_idx, j_idx] = 0.5
                macro_strain[j_idx, i_idx] = 0.5

            strain = np.tile(macro_strain, (nx, ny, nz, 1, 1))

            for _ in range(max_iter):
                elastic_strain = strain if eigenstrain_field is None else (strain - eigenstrain_field)
                stress = np.einsum("...ijkl,...kl->...ij", local_stiffness_field_c4, elastic_strain)
                
                tr_e = np.trace(strain, axis1=-2, axis2=-1)[..., np.newaxis, np.newaxis]
                ref_stress = strain * 2.0 * c0_shear + tr_e * np.eye(3) * (c0_bulk - (2.0 / 3.0) * c0_shear)
                tau = stress - ref_stress

                tau_hat = np.fft.fftn(tau, axes=(0, 1, 2))
                K = [self.KX, self.KY, self.KZ]
                k_dot_tau = np.zeros((nx, ny, nz, 3), dtype=np.complex128)
                for i in range(3):
                    for j in range(3):
                        k_dot_tau[..., i] += K[j] * tau_hat[..., i, j]

                k_tau_k = np.zeros((nx, ny, nz), dtype=np.complex128)
                for i in range(3):
                    k_tau_k += K[i] * k_dot_tau[..., i]

                eps_hat = np.zeros_like(tau_hat)
                nu0 = (3.0 * c0_bulk - 2.0 * c0_shear) / (2.0 * (3.0 * c0_bulk + c0_shear))

                for i in range(3):
                    for j in range(3):
                        t1 = (K[i] * k_dot_tau[..., j] + K[j] * k_dot_tau[..., i]) / (2.0 * c0_shear * self.K_sq)
                        t2 = (K[i] * K[j] * k_tau_k) / (2.0 * c0_shear * (1.0 - nu0) * (self.K_sq**2))
                        eps_hat[..., i, j] = -(t1 - t2)
                eps_hat[0, 0, 0, :, :] = 0.0

                strain_corr = np.real(np.fft.ifftn(eps_hat, axes=(0, 1, 2)))
                new_strain = np.tile(macro_strain, (nx, ny, nz, 1, 1)) + strain_corr

                res = float(np.linalg.norm(new_strain - strain) / max(1.0, np.linalg.norm(strain)))
                strain = new_strain
                if res < tol:
                    break

            homog_stress = np.mean(stress, axis=(0, 1, 2))
            for row, (r_i, r_j) in enumerate(voigt_map):
                c_eff_6x6[row, col] = homog_stress[r_i, r_j]

        return {
            "effective_c_voigt_matrix": c_eff_6x6,
            "effective_c11": float(c_eff_6x6[0, 0]),
            "effective_c12": float(c_eff_6x6[0, 1]),
            "effective_c44": float(c_eff_6x6[3, 3]),
            "effective_bulk_modulus_gpa": float((c_eff_6x6[0, 0] + 2.0 * c_eff_6x6[0, 1]) / 3.0),
        }
