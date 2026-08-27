"""Multi-Phase Grand Potential Phase-Field Dynamics & Anisotropic Read-Shockley Grain Boundaries."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np


class MultiPhaseGrandPotentialEngine:
    """Multi-phase order parameter field dynamics for N coexisting phases and anisotropic Read-Shockley GB energy tensors."""

    def __init__(
        self,
        num_phases: int = 3,
        grid_shape: Tuple[int, int] = (32, 32),
        interfacial_mobility: float = 1.2e-4,
        gradient_energy_coeff: float = 0.85,
    ):
        self.num_phases = num_phases
        self.nx, self.ny = grid_shape
        self.L_mob = interfacial_mobility
        self.kappa_grad = gradient_energy_coeff

    def step_forward_multiphase_field(
        self,
        phi_fields: np.ndarray,
        chemical_potentials_ev: np.ndarray,
        free_energy_densities: Optional[np.ndarray] = None,
        dt_s: float = 0.005,
    ) -> np.ndarray:
        """Step forward multi-phase order parameters coupled to thermodynamic driving forces:

        d phi_alpha / dt = - sum_beta (L_{alpha beta} / N) [ delta F / delta phi_alpha - delta F / delta phi_beta ]
        with Lagrange constraint sum_alpha phi_alpha = 1.
        """
        phi = np.asarray(phi_fields, dtype=np.float64)  # Shape (num_phases, nx, ny)
        n_p, nx, ny = phi.shape
        f_bulk = free_energy_densities if free_energy_densities is not None else np.zeros(n_p)

        new_phi = phi.copy()

        for a in range(n_p):
            lap_phi_a = (
                np.roll(phi[a], 1, axis=0) + np.roll(phi[a], -1, axis=0)
                + np.roll(phi[a], 1, axis=1) + np.roll(phi[a], -1, axis=1)
                - 4.0 * phi[a]
            )

            d_barrier_a = 4.0 * phi[a] * (1.0 - phi[a]) * (1.0 - 2.0 * phi[a])
            dF_dphi_a = d_barrier_a + f_bulk[a] - self.kappa_grad * lap_phi_a

            dphi_dt = np.zeros((nx, ny))
            for b in range(n_p):
                if a == b:
                    continue
                lap_phi_b = (
                    np.roll(phi[b], 1, axis=0) + np.roll(phi[b], -1, axis=0)
                    + np.roll(phi[b], 1, axis=1) + np.roll(phi[b], -1, axis=1)
                    - 4.0 * phi[b]
                )
                d_barrier_b = 4.0 * phi[b] * (1.0 - phi[b]) * (1.0 - 2.0 * phi[b])
                dF_dphi_b = d_barrier_b + f_bulk[b] - self.kappa_grad * lap_phi_b
                dphi_dt -= (self.L_mob / n_p) * (dF_dphi_a - dF_dphi_b)

            new_phi[a] += dt_s * dphi_dt

        # Enforce positivity and partition of unity sum_alpha phi_alpha = 1
        new_phi = np.maximum(0.0, new_phi)
        sum_phi = np.sum(new_phi, axis=0, keepdims=True)
        new_phi = new_phi / np.maximum(1e-8, sum_phi)

        return new_phi

    def compute_read_shockley_grain_boundary_energy(
        self,
        misorientation_angle_deg: float,
        sigma_max_j_m2: float = 0.85,
        theta_limit_deg: float = 15.0,
    ) -> float:
        """Read-Shockley dislocation model for low-angle grain boundary energy sigma_GB(theta)."""
        theta = max(0.01, misorientation_angle_deg)
        theta_m = theta_limit_deg

        if theta >= theta_m:
            return float(sigma_max_j_m2)

        ratio = theta / theta_m
        sigma_gb = sigma_max_j_m2 * ratio * (1.0 - np.log(ratio))
        return float(np.clip(sigma_gb, 0.05, sigma_max_j_m2))

    def compute_interfacial_energy_matrix(
        self,
        misorientations_deg: np.ndarray,
    ) -> np.ndarray:
        """Compute full N x N anisotropic grain boundary interfacial energy matrix."""
        n = len(misorientations_deg)
        sigma_matrix = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                delta_theta = abs(misorientations_deg[i] - misorientations_deg[j])
                e_gb = self.compute_read_shockley_grain_boundary_energy(delta_theta)
                sigma_matrix[i, j] = sigma_matrix[j, i] = e_gb

        return sigma_matrix
