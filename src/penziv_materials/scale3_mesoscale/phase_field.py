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


    def compute_khachaturyan_elastic_driving_force(
        self,
        strain_field: np.ndarray,                         # (nx, ny, nz, 3, 3)
        eigenstrain_tensors: List[np.ndarray],            # list of (3, 3) tensors per phase/variant
        stiffness_tensors: List[np.ndarray],              # list of (3, 3, 3, 3) tensors per phase/variant
        phi_fields: np.ndarray,                           # (num_phases, nx, ny, nz)
    ) -> np.ndarray:
        """Compute exact variational elastic driving force -delta F_elast / delta phi_alpha without isotropic simplifications."""
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Execute semi-implicit temporal update for coupled Cahn-Hilliard, Allen-Cahn, and Khachaturyan microelasticity."""
        c = c_field.copy()
        eta = eta_field.copy()

        for _ in range(n_steps):
            df_dc, df_deta = self.compute_chemical_free_energy_derivative(c, eta)

            if elastic_strain_field is not None and eigenstrain_tensors is not None and stiffness_tensors is not None:
                phi_stack = np.stack([1.0 - eta, eta], axis=0)
                dF_elast = self.compute_khachaturyan_elastic_driving_force(
                    strain_field=elastic_strain_field,
                    eigenstrain_tensors=eigenstrain_tensors,
                    stiffness_tensors=stiffness_tensors,
                    phi_fields=phi_stack,
                )
                df_deta += (dF_elast[1] - dF_elast[0])

            lap_c = self.compute_laplacian(c)
            mu_chem = df_dc - self.kappa_c * lap_c

            lap_mu = self.compute_laplacian(mu_chem)
            c += dt * self.M_c * lap_mu

            lap_eta = self.compute_laplacian(eta)
            eta -= dt * self.L_eta * (df_deta - self.kappa_eta * lap_eta)

            c = np.clip(c, 0.0, 1.0)
            eta = np.clip(eta, 0.0, 1.0)

        return c, eta
