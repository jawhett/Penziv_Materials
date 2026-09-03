"""Thermal Gradient Residual Stress Integrator & Cellular Dislocation Substructure Engine."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pydantic import BaseModel, Field


class ResidualStressState(BaseModel):
    """3D/1D residual stress and cellular dislocation state summary."""
    surface_residual_stress_mpa: float
    centerline_residual_stress_mpa: float
    max_tensile_residual_stress_mpa: float
    depth_profile_z_um: List[float]
    stress_profile_sigma_xx_mpa: List[float]
    cell_wall_dislocation_density_m2: float
    cell_interior_dislocation_density_m2: float
    kinematic_back_stress_mpa: float
    bauschinger_effect_ratio: float
    is_yield_exceeded_in_solidification: bool


class ThermalResidualStressEngine:
    """Integrates transient thermal gradients during quenching/AM to compute residual stress profiles and cellular dislocation partitioning."""

    def __init__(
        self,
        youngs_modulus_gpa: float = 200.0,
        poisson_ratio: float = 0.30,
        thermal_expansion_coeff_ppm_k: float = 15.0,
    ):
        self.E = youngs_modulus_gpa * 1.0e3  # MPa
        self.nu = poisson_ratio
        self.alpha = thermal_expansion_coeff_ppm_k * 1.0e-6  # 1/K

    def compute_1d_through_thickness_residual_stress(
        self,
        thickness_um: float = 2000.0,
        surface_temp_k: float = 300.0,
        center_temp_k: float = 1200.0,
        yield_strength_mpa: float = 400.0,
        num_grid_points: int = 50,
    ) -> ResidualStressState:
        """Solve 1D thermo-elastic-plastic residual stress equilibrium under severe thermal gradients:

        sigma_xx(z) = - [E / (1 - nu)] * alpha * [T(z) - T_mean]
        clamped to [-sigma_y, +sigma_y] plastic yield criteria.
        """
        n = max(10, num_grid_points)
        z_grid = np.linspace(-thickness_um / 2.0, thickness_um / 2.0, n)
        z_norm = z_grid / (thickness_um / 2.0)

        # Parabolic thermal gradient profile across plate / melt track
        t_profile = center_temp_k - (center_temp_k - surface_temp_k) * (z_norm**2)
        t_mean = float(np.mean(t_profile))

        # Elastic stress increment
        stiffness_factor = self.E / max(0.1, 1.0 - self.nu)
        elastic_stress = -stiffness_factor * self.alpha * (t_profile - t_mean)

        # Plastic relaxation & self-equilibrating residual stress
        sigma_y = max(50.0, yield_strength_mpa)
        sigma_plastic = np.clip(elastic_stress, -sigma_y, sigma_y)

        # Balance force equilibrium integral(sigma * dz) = 0
        mean_offset = float(np.mean(sigma_plastic))
        sigma_residual = sigma_plastic - mean_offset

        surf_stress = float(sigma_residual[0])
        center_stress = float(sigma_residual[n // 2])
        max_tensile = float(np.max(sigma_residual))

        # Cellular dislocation substructure partitioning
        # Local plastic strain drives cell formation (Mughrabi composite model)
        f_wall = 0.20  # Volume fraction of dislocation cell walls
        f_cell = 1.0 - f_wall
        mean_rho = 1.0e14 * (1.0 + 3.0 * (abs(surf_stress) / sigma_y))
        rho_wall = float(mean_rho * 4.5 / f_wall)
        rho_interior = float(mean_rho * 0.2 / f_cell)

        # Kinematic back stress tau_back = 0.5 * mu * b * sqrt(rho_wall - rho_int)
        mu_mpa = self.E / (2.0 * (1.0 + self.nu))
        b_m = 2.54e-10
        tau_back = float(0.35 * mu_mpa * b_m * np.sqrt(max(0.0, rho_wall - rho_interior)))
        bauschinger_ratio = float(max(0.05, min(1.0, 1.0 - (tau_back / (0.5 * sigma_y)))))

        return ResidualStressState(
            surface_residual_stress_mpa=surf_stress,
            centerline_residual_stress_mpa=center_stress,
            max_tensile_residual_stress_mpa=max_tensile,
            depth_profile_z_um=z_grid.tolist(),
            stress_profile_sigma_xx_mpa=sigma_residual.tolist(),
            cell_wall_dislocation_density_m2=rho_wall,
            cell_interior_dislocation_density_m2=rho_interior,
            kinematic_back_stress_mpa=tau_back * 2.0,
            bauschinger_effect_ratio=bauschinger_ratio,
            is_yield_exceeded_in_solidification=bool(np.any(np.abs(elastic_stress) >= sigma_y)),
        )
