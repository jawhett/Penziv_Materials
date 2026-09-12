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
        delta_tm_k: float = 1350.0,
    ) -> Dict[str, float]:
        """Evaluate 3D melt-pool length L, depth D, width W, and cooling rate from the analytical Rosenthal moving heat source equation."""
        # Laser power absorption eta ~ 0.40
        p_absorbed = self.power * 0.40
        thermal_diff = thermal_conductivity_w_m_k / (density_kg_m3 * heat_capacity_j_kg_k)
        v = max(0.01, self.v_scan)

        # 1. Trailing melt pool length along scan axis (R + x = 0 at trailing edge)
        # Delta T = P * eta / (2 * pi * k * L_tail)
        l_tail_m = p_absorbed / (2.0 * np.pi * thermal_conductivity_w_m_k * delta_tm_k)

        # 2. Transverse melt pool depth D and half-width W/2: Delta T = (P*eta / (2*pi*k*R)) * exp(-v*R / (2*alpha))
        # Bisection on R to find isotherm radius along transverse directions
        r_low, r_high = 1.0e-7, max(1.0e-3, l_tail_m)
        for _ in range(40):
            r_mid = 0.5 * (r_low + r_high)
            val = (p_absorbed / (2.0 * np.pi * thermal_conductivity_w_m_k * r_mid)) * np.exp(-v * r_mid / (2.0 * thermal_diff))
            if val > delta_tm_k:
                r_low = r_mid
            else:
                r_high = r_mid
        r_transverse_m = 0.5 * (r_low + r_high)

        depth_m = r_transverse_m
        width_m = 2.0 * r_transverse_m
        length_m = l_tail_m + 0.5 * r_transverse_m

        # 3. Solidification velocity & cooling rate at trailing centerline
        v_solidification = v
        # Thermal gradient G_x = (Delta T / L_tail) + (v * Delta T / (2 * alpha))
        grad_t_k_m = (delta_tm_k / max(1e-6, l_tail_m)) + (v * delta_tm_k) / (2.0 * thermal_diff)
        cooling_rate_k_s = float(grad_t_k_m * v_solidification)

        return {
            "meltpool_depth_um": float(depth_m * 1.0e6),
            "meltpool_length_um": float(length_m * 1.0e6),
            "meltpool_width_um": float(width_m * 1.0e6),
            "solidification_velocity_m_s": float(v_solidification),
            "peak_cooling_rate_k_s": float(cooling_rate_k_s),
            "thermal_gradient_k_m": float(grad_t_k_m),
        }
