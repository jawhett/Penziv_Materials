"""3D Multi-Variant Phase-Field Solidification, Coarsening & Khachaturyan-Shatalov Microelasticity Engine."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np


class PhaseFieldEngine:
    """3D Phase-Field engine solving coupled Cahn-Hilliard (conserved solute c), Allen-Cahn (structural order parameters eta_p), and Khachaturyan microelasticity."""

    def __init__(
        self,
        grid_size: Tuple[int, ...] = (16, 16, 16),
        dx_nm: float = 1.0,
        mobility_c: float = 1.0,
        mobility_eta: float = 2.5,
        gradient_coeff_kappa_c: float = 0.5,
        gradient_coeff_kappa_eta: float = 1.0,
    ):
        self.grid_size = grid_size
        self.dim = len(grid_size)
        self.dx = dx_nm
        self.M_c = mobility_c
        self.L_eta = mobility_eta
        self.kappa_c = gradient_coeff_kappa_c
        self.kappa_eta = gradient_coeff_kappa_eta

    def compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute periodic finite difference Laplacian nabla^2 in 2D or 3D."""
        lap = np.zeros_like(field)
        dx2 = self.dx**2

        if field.ndim == 2:
            lap = (
                np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0)
                + np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1)
                - 4.0 * field
            ) / dx2
        elif field.ndim == 3:
            lap = (
                np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0)
                + np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1)
                + np.roll(field, 1, axis=2) + np.roll(field, -1, axis=2)
                - 6.0 * field
            ) / dx2
        return lap

    def compute_chemical_free_energy_derivative(
        self,
        c: np.ndarray,
        eta: np.ndarray,
        w_barrier: float = 1.0,
        c_alpha_eq: float = 0.05,
        c_beta_eq: float = 0.95,
        curvature_alpha: float = 1.0,
        curvature_beta: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate exact variational derivatives delta F_chem / delta c and delta F_chem / delta eta from phase equilibria."""
        dg_deta = 2.0 * eta * (1.0 - eta) * (1.0 - 2.0 * eta)
        h_eta = eta**3 * (10.0 - 15.0 * eta + 6.0 * eta**2)
        dh_deta = 30.0 * (eta**2) * ((1.0 - eta) ** 2)

        f_alpha = 0.5 * curvature_alpha * (c - c_alpha_eq) ** 2
        f_beta = 0.5 * curvature_beta * (c - c_beta_eq) ** 2

        df_dc = (1.0 - h_eta) * curvature_alpha * (c - c_alpha_eq) + h_eta * curvature_beta * (c - c_beta_eq)
        df_deta = w_barrier * dg_deta + (f_beta - f_alpha) * dh_deta

        return df_dc, df_deta


    def compute_anisotropic_gradient_operator(
        self,
        field: np.ndarray,
        kappa_tensor: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute directional interfacial gradient operator sum_{i,j} kappa_{ij} d^2 phi / (dx_i dx_j).

        If kappa_tensor is None, falls back to isotropic Laplacian scaled by kappa_eta.
        """
        if kappa_tensor is None:
            return self.compute_laplacian(field) * self.kappa_eta

        ndim = field.ndim
        if ndim == 2:
            nx, ny = field.shape
            kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=self.dx)
            ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=self.dx)
            KX, KY = np.meshgrid(kx, ky, indexing="ij")
            K_vec = np.stack([KX, KY], axis=-1)
            k_kappa_k = np.einsum("...i,ij,...j->...", K_vec, kappa_tensor[:2, :2], K_vec)
            phi_hat = np.fft.fftn(field)
            return np.real(np.fft.ifftn(-k_kappa_k * phi_hat))
        elif ndim == 3:
            nx, ny, nz = field.shape
            kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=self.dx)
            ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=self.dx)
            kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=self.dx)
            KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
            K_vec = np.stack([KX, KY, KZ], axis=-1)
            k_kappa_k = np.einsum("...i,ij,...j->...", K_vec, kappa_tensor[:3, :3], K_vec)
            phi_hat = np.fft.fftn(field)
            return np.real(np.fft.ifftn(-k_kappa_k * phi_hat))
        else:
            return self.compute_laplacian(field) * self.kappa_eta

    def solve_khachaturyan_elastic_equilibrium_fft(
        self,
        order_parameters: np.ndarray,                     # (num_variants, nx, ny, nz) or (nx, ny, nz)
        eigenstrain_tensors: List[np.ndarray],            # list of (3, 3) symmetric eigenstrain tensors per variant
        stiffness_tensor_4th_order: np.ndarray,           # (3, 3, 3, 3) or (6, 6) reference stiffness C^0
        applied_strain: Optional[np.ndarray] = None,      # (3, 3) macroscopic applied strain
    ) -> Dict[str, Any]:
        """Solve exact periodic 3D mechanical equilibrium nabla . sigma = 0 via spectral Green's tensor.

        Returns total strain field, elastic strain field, Cauchy stress, and variational elastic driving forces:
            delta F_elast / delta eta_p(r) = -sigma_0^(p) : eps(r) + 1/2 sigma_0^(p) : eps_0^(p)
        """
        # Ensure 4D shape (P, nx, ny, nz)
        if order_parameters.ndim == 2:
            phi_4d = order_parameters[np.newaxis, :, :, np.newaxis]
        elif order_parameters.ndim == 3:
            phi_4d = order_parameters[np.newaxis, ...]
        else:
            phi_4d = order_parameters

        num_variants, nx, ny, nz = phi_4d.shape

        # Convert Voigt 6x6 to 4th order (3, 3, 3, 3) if needed
        if stiffness_tensor_4th_order.shape == (6, 6):
            v_map = {0: (0, 0), 1: (1, 1), 2: (2, 2), 3: (1, 2), 4: (0, 2), 5: (0, 1)}
            C4 = np.zeros((3, 3, 3, 3), dtype=np.float64)
            for a in range(6):
                i, j = v_map[a]
                for b in range(6):
                    k, l = v_map[b]
                    val = stiffness_tensor_4th_order[a, b]
                    C4[i, j, k, l] = val
                    C4[j, i, k, l] = val
                    C4[i, j, l, k] = val
                    C4[j, i, l, k] = val
        else:
            C4 = stiffness_tensor_4th_order.astype(np.float64)

        # Spectral grid
        kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=self.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=self.dx)
        kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=self.dx)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        K_sq = KX**2 + KY**2 + KZ**2
        K_norm = np.sqrt(K_sq)

        # Unit wavevectors n_i = k_i / |k|
        nx_vec = np.where(K_norm > 1e-10, KX / np.maximum(1e-10, K_norm), 0.0)
        ny_vec = np.where(K_norm > 1e-10, KY / np.maximum(1e-10, K_norm), 0.0)
        nz_vec = np.where(K_norm > 1e-10, KZ / np.maximum(1e-10, K_norm), 0.0)
        n_vec = np.stack([nx_vec, ny_vec, nz_vec], axis=-1)  # (nx, ny, nz, 3)

        # Transformation stress sigma_0_p = C4 : eps_0_p
        sigma_0_list = []
        for p in range(num_variants):
            eps_0 = eigenstrain_tensors[p] if p < len(eigenstrain_tensors) else np.zeros((3, 3))
            sig_0 = np.einsum("ijkl,kl->ij", C4, eps_0)
            sigma_0_list.append(sig_0)
        sigma_0_arr = np.array(sigma_0_list, dtype=np.float64)  # (P, 3, 3)

        # FFT of order parameters
        phi_hat = np.fft.fftn(phi_4d, axes=(1, 2, 3))  # (P, nx, ny, nz)

        # Transformation stress in Fourier space: sigma_0_hat(k) = sum_p sigma_0_p * phi_hat_p(k)
        sigma_0_hat = np.einsum("pij,pxyz->xyzij", sigma_0_arr, phi_hat)  # (nx, ny, nz, 3, 3)

        # Khachaturyan acoustic tensor A_ik(n) = C_ijkl n_j n_l
        A = np.einsum("ijkl,...j,...l->...ik", C4, n_vec, n_vec)  # (nx, ny, nz, 3, 3)

        zero_mask = (K_norm <= 1e-10)
        A_reg = A.copy()
        A_reg[zero_mask] = np.eye(3)
        Omega = np.linalg.inv(A_reg)  # (nx, ny, nz, 3, 3)
        Omega[zero_mask] = 0.0

        # eps_hat_ij(k) = 1/2 (n_i Omega_jk + n_j Omega_ik) sigma_0_hat_kl n_l
        sig_n = np.einsum("...kl,...l->...k", sigma_0_hat, n_vec)  # (nx, ny, nz, 3)
        omega_sig_n = np.einsum("...jk,...k->...j", Omega, sig_n)  # (nx, ny, nz, 3)
        eps_hat = 0.5 * (np.einsum("...i,...j->...ij", n_vec, omega_sig_n) + np.einsum("...j,...i->...ij", n_vec, omega_sig_n))

        if applied_strain is not None:
            eps_hat[zero_mask] = applied_strain * float(nx * ny * nz)
        else:
            eps_hat[zero_mask] = 0.0

        total_strain = np.real(np.fft.ifftn(eps_hat, axes=(0, 1, 2)))  # (nx, ny, nz, 3, 3)

        # Heterogeneous eigenstrain field: sum_p phi_p(r) * eps_0_p
        eigenstrains_arr = np.array([eigenstrain_tensors[p] if p < len(eigenstrain_tensors) else np.zeros((3, 3)) for p in range(num_variants)], dtype=np.float64)
        eigenstrain_field = np.einsum("pxyz,pij->xyzij", phi_4d, eigenstrains_arr)
        elastic_strain = total_strain - eigenstrain_field
        stress_field = np.einsum("ijkl,...kl->...ij", C4, elastic_strain)

        # Variational driving forces
        elastic_driving_forces = np.zeros((num_variants, nx, ny, nz), dtype=np.float64)
        for p in range(num_variants):
            sig_0_p = sigma_0_arr[p]
            eps_0_p = eigenstrains_arr[p]
            self_e = 0.5 * np.sum(sig_0_p * eps_0_p)
            int_e = np.einsum("ij,...ij->...", sig_0_p, total_strain)
            elastic_driving_forces[p] = -int_e + self_e

        total_elastic_energy = 0.5 * np.sum(stress_field * elastic_strain) * (self.dx ** 3)

        return {
            "total_strain_field": total_strain,
            "elastic_strain_field": elastic_strain,
            "cauchy_stress_field": stress_field,
            "elastic_driving_forces": elastic_driving_forces if order_parameters.ndim >= 3 else elastic_driving_forces.squeeze(),
            "total_elastic_energy": float(total_elastic_energy),
        }

    def compute_khachaturyan_elastic_driving_force(
        self,
        strain_field: np.ndarray,                         # (nx, ny, nz, 3, 3)
        eigenstrain_tensors: List[np.ndarray],            # list of (3, 3) tensors per phase/variant
        stiffness_tensors: List[np.ndarray],              # list of (3, 3, 3, 3) tensors per phase/variant
        phi_fields: np.ndarray,                           # (num_phases, nx, ny, nz)
    ) -> np.ndarray:
        """Compute variational elastic driving force -delta F_elast / delta phi_alpha."""
        # Use spectral solver if stiffness and eigenstrain are provided
        if len(stiffness_tensors) > 0 and len(eigenstrain_tensors) > 0:
            res = self.solve_khachaturyan_elastic_equilibrium_fft(
                order_parameters=phi_fields,
                eigenstrain_tensors=eigenstrain_tensors,
                stiffness_tensor_4th_order=stiffness_tensors[0],
            )
            return res["elastic_driving_forces"]

        num_phases = phi_fields.shape[0]
        elastic_driving_forces = np.zeros_like(phi_fields)
        for a in range(num_phases):
            C_a = stiffness_tensors[a] if a < len(stiffness_tensors) else stiffness_tensors[0]
            eps_0_a = eigenstrain_tensors[a] if a < len(eigenstrain_tensors) else eigenstrain_tensors[0]
            elastic_strain = strain_field - eps_0_a
            energy_density = 0.5 * np.einsum("...ij,ijkl,...kl->...", elastic_strain, C_a, elastic_strain)
            elastic_driving_forces[a] = -energy_density

        return elastic_driving_forces

    def step_forward_semi_implicit(
        self,
        c_field: np.ndarray,
        eta_field: np.ndarray,
        dt: float = 0.01,
        n_steps: int = 1,
        elastic_strain_field: Optional[np.ndarray] = None,
        eigenstrain_tensors: Optional[List[np.ndarray]] = None,
        stiffness_tensors: Optional[List[np.ndarray]] = None,
        anisotropic_kappa_eta: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute semi-implicit temporal update for coupled Cahn-Hilliard, Allen-Cahn, and Khachaturyan microelasticity."""
        c = c_field.copy()
        eta = eta_field.copy()

        for _ in range(n_steps):
            df_dc, df_deta = self.compute_chemical_free_energy_derivative(c, eta)

            if eigenstrain_tensors is not None and stiffness_tensors is not None:
                phi_stack = np.stack([1.0 - eta, eta], axis=0) if eta.ndim == 3 else np.stack([1.0 - eta, eta], axis=0)
                dF_elast = self.compute_khachaturyan_elastic_driving_force(
                    strain_field=elastic_strain_field if elastic_strain_field is not None else np.zeros(eta.shape + (3, 3)),
                    eigenstrain_tensors=eigenstrain_tensors,
                    stiffness_tensors=stiffness_tensors,
                    phi_fields=phi_stack,
                )
                df_deta += (dF_elast[1] - dF_elast[0])

            lap_c = self.compute_laplacian(c)
            mu_chem = df_dc - self.kappa_c * lap_c

            lap_mu = self.compute_laplacian(mu_chem)
            c += dt * self.M_c * lap_mu

            grad_eta_term = self.compute_anisotropic_gradient_operator(eta, anisotropic_kappa_eta)
            eta -= dt * self.L_eta * (df_deta - grad_eta_term)

            c = np.clip(c, 0.0, 1.0)
            eta = np.clip(eta, 0.0, 1.0)

        return c, eta
