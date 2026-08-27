"""Coupled Poisson-Nernst-Planck (PNP) Electro-Chemo-Mechanics & Butler-Volmer Engine."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.constants import ELEMENTARY_CHARGE_C, BOLTZMANN_J_K, FARADAY_C_MOL, R_GAS


class CoupledPNPMechanicsSolver:
    """Solves coupled Poisson-Nernst-Planck (PNP) electro-chemo-mechanics and Butler-Volmer charge transfer."""

    def __init__(
        self,
        grid_points: int = 100,
        domain_length_nm: float = 20.0,
        dielectric_constant_eps_r: float = 14.0,
        cation_charge_z: int = 2,
    ):
        self.nx = grid_points
        self.l_x = domain_length_nm * 1.0e-9  # m
        self.dx = self.l_x / max(1, self.nx - 1)
        self.eps_r = dielectric_constant_eps_r
        self.z = cation_charge_z
        self.eps_0 = 8.8541878128e-12

    def solve_space_charge_potential_1d(
        self,
        cation_concentration_m3: np.ndarray,
        anion_background_concentration_m3: float,
        boundary_potential_left_v: float = 0.0,
        boundary_potential_right_v: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Solve 1D Poisson equation for space-charge electrostatic potential phi(x):

        d^2 phi / dx^2 = - rho(x) / (eps_r * eps_0)
        where rho(x) = z * e * (c_cation(x) - c_anion)
        """
        # Net charge density rho (C/m3)
        rho = self.z * ELEMENTARY_CHARGE_C * (cation_concentration_m3 - anion_background_concentration_m3)

        # Tridiagonal Poisson matrix
        A = np.zeros((self.nx, self.nx), dtype=np.float64)
        rhs = np.zeros(self.nx, dtype=np.float64)

        for i in range(1, self.nx - 1):
            A[i, i - 1] = 1.0 / (self.dx**2)
            A[i, i] = -2.0 / (self.dx**2)
            A[i, i + 1] = 1.0 / (self.dx**2)
            rhs[i] = -rho[i] / (self.eps_r * self.eps_0)

        # Dirichlet boundary conditions
        A[0, 0] = 1.0
        rhs[0] = boundary_potential_left_v
        A[-1, -1] = 1.0
        rhs[-1] = boundary_potential_right_v

        phi = np.linalg.solve(A, rhs)
        electric_field = -np.gradient(phi, self.dx)

        # Debye screening length lambda_D = sqrt(eps_r * eps_0 * k_B * T / (2 * z^2 * e^2 * c_0))
        kbt = BOLTZMANN_J_K * 300.0
        lambda_debye_nm = 1.0e9 * np.sqrt(
            (self.eps_r * self.eps_0 * kbt)
            / (2.0 * (self.z * ELEMENTARY_CHARGE_C) ** 2 * max(1.0, anion_background_concentration_m3))
        )

        return phi, electric_field, float(lambda_debye_nm)

    def evaluate_butler_volmer_current_density(
        self,
        overpotential_eta_v: float,
        exchange_current_density_j0_a_m2: float = 10.0,
        charge_transfer_alpha: float = 0.5,
        temperature_k: float = 300.0,
    ) -> float:
        """Evaluate non-linear Butler-Volmer interfacial charge transfer current density:

        J_BV = J_0 * [ exp(alpha * z * F * eta / (R * T)) - exp(-(1 - alpha) * z * F * eta / (R * T)) ]
        """
        rt = R_GAS * temperature_k
        f_factor = (self.z * FARADAY_C_MOL) / rt

        anodic = np.exp(np.clip(charge_transfer_alpha * f_factor * overpotential_eta_v, -40.0, 40.0))
        cathodic = np.exp(np.clip(-(1.0 - charge_transfer_alpha) * f_factor * overpotential_eta_v, -40.0, 40.0))

        j_bv = exchange_current_density_j0_a_m2 * (anodic - cathodic)
        return float(j_bv)

    def compute_chemo_mechanical_stress_coupling(
        self,
        elastic_strain: np.ndarray,
        concentration_change_mol_m3: float,
        electric_field_v_m: float,
        youngs_modulus_pa: float = 40.0e9,
        partial_molar_volume_m3_mol: float = 1.2e-5,
        piezo_coupling_coeff: float = 0.05,
    ) -> Dict[str, float]:
        """Coupled stress tensor: sigma = E * eps - (E * Omega / 3) * Delta c - gamma * E_field."""
        # Vegard chemo-elastic eigenstrain
        vegard_stress_pa = (youngs_modulus_pa * partial_molar_volume_m3_mol / 3.0) * concentration_change_mol_m3
        maxwell_stress_pa = piezo_coupling_coeff * electric_field_v_m
        elastic_stress_pa = youngs_modulus_pa * np.trace(elastic_strain)

        total_stress_pa = elastic_stress_pa - vegard_stress_pa - maxwell_stress_pa
        return {
            "total_stress_mpa": float(total_stress_pa * 1.0e-6),
            "vegard_chemo_stress_mpa": float(vegard_stress_pa * 1.0e-6),
            "electrostatic_coupling_stress_mpa": float(maxwell_stress_pa * 1.0e-6),
        }
