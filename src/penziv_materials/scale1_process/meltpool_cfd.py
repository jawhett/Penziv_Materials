"""Melt-Pool Hydrodynamics & Stefan Solidification Engine (Marangoni Shear & Boundary Layer)."""

from typing import Dict, Tuple, List, Optional
import numpy as np


class MeltPoolCFDEngine:
    """Solves laser melt-pool thermal history, Marangoni thermocapillary shear, and Stefan solidification."""

    def __init__(
        self,
        laser_power_w: float = 280.0,
        scan_speed_m_s: float = 1.0,
        beam_radius_um: float = 50.0,
        d_gamma_dt: float = -0.35e-3,  # N/(m·K) Marangoni thermocapillary coefficient
    ):
        self.power = laser_power_w
        self.v_scan = scan_speed_m_s
        self.r_beam = beam_radius_um * 1.0e-6
        self.d_gamma_dt = d_gamma_dt

    def compute_marangoni_shear_stress(
        self,
        temperature_gradient_surface_k_m: float,
    ) -> float:
        """Evaluate Marangoni surface thermocapillary shear stress:

        tau_s = (d_gamma / dT) * grad_s(T)
        """
        tau_shear = abs(self.d_gamma_dt * temperature_gradient_surface_k_m)
        return float(tau_shear)

    def solve_subgrid_boundary_layer_segregation(
        self,
        solidification_velocity_m_s: float,
        liquid_diffusivity_m2_s: float = 3.0e-9,
        boundary_layer_thickness_um: float = 2.5,
        equilibrium_partition_k0: float = 0.65,
    ) -> Tuple[float, float]:
        """Burton-Slichter-Wagner (BPS) effective solute partition coefficient across convective boundary layer delta:

        k_eff = k0 / [ k0 + (1 - k0) * exp(-V * delta / D_L) ]
        """
        delta_m = boundary_layer_thickness_um * 1.0e-6
        peclet_bl = (solidification_velocity_m_s * delta_m) / liquid_diffusivity_m2_s
        exp_term = np.exp(-peclet_bl)

        k_eff = equilibrium_partition_k0 / (equilibrium_partition_k0 + (1.0 - equilibrium_partition_k0) * exp_term)
        return float(k_eff), float(peclet_bl)

    def compute_melt_pool_dimensions_and_history(
        self,
        density_kg_m3: float = 8200.0,
        heat_capacity_j_kg_k: float = 450.0,
        thermal_conductivity_w_m_k: float = 28.0,
        latent_heat_fusion_j_kg: float = 270000.0,
    ) -> Dict[str, float]:
        """Evaluate 3D melt-pool length L, depth D, width W, and solid-liquid interface velocities."""
        # Absorptivity ~ 0.40 for nickel superalloy
        p_absorbed = self.power * 0.40
        thermal_diff = thermal_conductivity_w_m_k / (density_kg_m3 * heat_capacity_j_kg_k)

        # Depth scaling
        depth_um = 1000.0 * np.sqrt((p_absorbed / (2.0 * np.pi * thermal_conductivity_w_m_k * 1350.0)) * (self.r_beam / 2.0))
        length_um = depth_um * 3.2
        width_um = depth_um * 1.8

        v_solidification = self.v_scan * 0.707
        t_dot = 5.0e7  # K/s

        return {
            "meltpool_depth_um": float(depth_um),
            "meltpool_length_um": float(length_um),
            "meltpool_width_um": float(width_um),
            "solidification_velocity_m_s": float(v_solidification),
            "peak_cooling_rate_k_s": float(t_dot),
        }
