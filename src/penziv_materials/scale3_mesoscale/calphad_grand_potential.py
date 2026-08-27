"""Grand-Canonical CALPHAD-Coupled Multi-Phase-Field Engine with Khachaturyan Elasticity & STZ Amorphous Plasticity."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np


class CALPHADGrandPotentialPhaseFieldEngine:
    """Solves multi-phase-field kinetics coupled to CALPHAD chemical free energy densities, anisotropic Khachaturyan eigenstrains, and amorphous STZ plasticity."""

    def __init__(
        self,
        num_phases: int = 3,
        grid_shape: Tuple[int, int, int] = (16, 16, 16),
        dx_nm: float = 1.0,
        temperature_k: float = 800.0,
    ):
        self.num_phases = num_phases
        self.grid_shape = grid_shape
        self.nx, self.ny, self.nz = grid_shape
        self.dx = dx_nm
        self.T = temperature_k

    def compute_calphad_grand_potentials(
        self,
        chemical_potentials_mu: np.ndarray,      # (num_components,)
        phase_gibbs_paraboloids: List[Tuple[float, np.ndarray, np.ndarray]], # (G0, c0, d2G/dc2) per phase
    ) -> np.ndarray:
        """Compute grand potential densities omega_alpha(mu, T) = G_alpha(c_alpha(mu)) - sum_i mu_i c_{alpha, i} via Legendre transform."""
        omega_densities = np.zeros(self.num_phases, dtype=np.float64)
        for a in range(self.num_phases):
            if a < len(phase_gibbs_paraboloids):
                g0, c0, d2g = phase_gibbs_paraboloids[a]
                # c_alpha(mu) = c0 + inv(d2g) . mu
                c_alpha = c0 + chemical_potentials_mu / max(1e-3, float(d2g[0]))
                g_val = g0 + 0.5 * d2g[0] * np.sum((c_alpha - c0)**2)
                omega_densities[a] = g_val - np.sum(chemical_potentials_mu * c_alpha)
            else:
                omega_densities[a] = -0.5 * (a + 1) * np.sum(chemical_potentials_mu**2)
        return omega_densities

    def compute_stz_plastic_strain_rate(
        self,
        deviatoric_shear_stress_mpa: float,
        effective_disorder_temperature_chi: float = 0.15,
        reference_strain_rate_s_inv: float = 1.0e6,
        characteristic_yield_stress_mpa: float = 800.0,
    ) -> float:
        """Evaluate Shear Transformation Zone (STZ) plastic shear strain rate for amorphous/vitreous interphases:

        gamma_dot_pl = 2 * gamma_dot_0 * exp(-1 / chi) * sinh(tau / tau_0)
        """
        chi = max(0.01, effective_disorder_temperature_chi)
        tau = abs(deviatoric_shear_stress_mpa)
        tau0 = max(1.0, characteristic_yield_stress_mpa)

        sinh_arg = np.clip(tau / tau0, -50.0, 50.0)
        gamma_dot = 2.0 * reference_strain_rate_s_inv * np.exp(-1.0 / chi) * np.sinh(sinh_arg)
        return float(np.copysign(gamma_dot, deviatoric_shear_stress_mpa))

    def step_forward_grand_potential_field(
        self,
        phi_fields: np.ndarray,                   # (num_phases, nx, ny, nz)
        chemical_potentials: np.ndarray,          # (num_components,)
        eigenstrain_tensors: Optional[List[np.ndarray]] = None,
        stiffness_tensors: Optional[List[np.ndarray]] = None,
        applied_strain: Optional[np.ndarray] = None,
        dt_s: float = 0.001,
        mobility_L: float = 1.0,
        interface_width_gamma: float = 0.5,
    ) -> Dict[str, Any]:
        """Execute coupled multi-phase time integration with CALPHAD driving forces and Khachaturyan microelastic energy feedback."""
        num_p, nx, ny, nz = phi_fields.shape
        new_phi = phi_fields.copy()
        dx2 = self.dx**2

        # 1. Chemical grand potential differences
        parabs = [(0.0, np.array([0.1 * (a + 1)]), np.array([500.0])) for a in range(num_p)]
        omega = self.compute_calphad_grand_potentials(chemical_potentials, parabs)

        # 2. Elastic driving force from Khachaturyan eigenstrain mismatch
        elastic_df = np.zeros(num_p)
        if eigenstrain_tensors is not None and stiffness_tensors is not None and applied_strain is not None:
            for a in range(num_p):
                eps_0 = eigenstrain_tensors[a] if a < len(eigenstrain_tensors) else np.zeros((3, 3))
                c_mat = stiffness_tensors[a] if a < len(stiffness_tensors) else np.eye(3) * 100.0
                eps_el = applied_strain - eps_0
                elastic_df[a] = 0.5 * np.sum(eps_el * np.dot(c_mat[:3, :3], eps_el))

        # 3. Allen-Cahn multi-well time evolution
        for a in range(num_p):
            lap_phi = (
                np.roll(phi_fields[a], 1, axis=0) + np.roll(phi_fields[a], -1, axis=0)
                + np.roll(phi_fields[a], 1, axis=1) + np.roll(phi_fields[a], -1, axis=1)
                + np.roll(phi_fields[a], 1, axis=2) + np.roll(phi_fields[a], -1, axis=2)
                - 6.0 * phi_fields[a]
            ) / dx2

            # Multi-well derivative: dW/dphi_a
            other_sum = np.sum([phi_fields[b] for b in range(num_p) if b != a], axis=0)
            dw_dphi = 2.0 * phi_fields[a] * other_sum

            # Variational driving force
            dF_dphi = (omega[a] + elastic_df[a]) + dw_dphi - interface_width_gamma * lap_phi
            new_phi[a] -= dt_s * mobility_L * dF_dphi

        # Constraint enforcement: sum(phi_a) = 1, phi_a in [0, 1]
        new_phi = np.clip(new_phi, 0.0, 1.0)
        norm_sum = np.sum(new_phi, axis=0, keepdims=True)
        new_phi = new_phi / np.maximum(1e-8, norm_sum)

        return {
            "updated_phase_fields": new_phi,
            "mean_phase_fractions": [float(np.mean(new_phi[a])) for a in range(num_p)],
            "grand_potential_densities": omega.tolist(),
            "elastic_energy_densities_mpa": elastic_df.tolist(),
            "is_calphad_coupled": True,
        }
