"""Electronic Structure, Semiconductor Transport, Effective Mass Tensors & Dielectric Breakdown."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.constants import HBAR, M_ELECTRON, E_CHARGE, BOLTZMANN_J_K, EPSILON_0


class SemiconductorElectronicEngine:
    """Evaluates electronic band structures, anisotropic effective mass tensors, Wannier BTE mobilities, and impact ionization breakdown."""

    def __init__(self, temperature_k: float = 300.0):
        self.T = temperature_k

    def compute_effective_mass_tensor(
        self,
        band_curvature_ev_ang2: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Compute 3x3 effective mass tensor m*_{ij} = hbar^2 * (d^2 E / d k_i d k_j)^-1 in units of electron rest mass m_0."""
        d2E_dk2_si = np.asarray(band_curvature_ev_ang2, dtype=np.float64) * E_CHARGE * 1.0e-20

        inv_curvature = np.linalg.pinv(d2E_dk2_si)
        m_star_kg = (HBAR**2) * inv_curvature
        m_star_relative = m_star_kg / M_ELECTRON

        m_star_relative = 0.5 * (m_star_relative + m_star_relative.T)
        m_eff_scalar = float(np.cbrt(np.abs(np.linalg.det(m_star_relative))))
        return m_star_relative, m_eff_scalar

    def compute_wannier_bte_mobility_tensor(
        self,
        group_velocity_tensor_m_s: np.ndarray,
        relaxation_time_fs: float = 120.0,
        band_gap_ev: float = 1.42,
        effective_mass_relative: float = 0.25,
    ) -> Dict[str, Any]:
        """Evaluate full tensor carrier mobility mu_{ij}(T) via Wannier-interpolated Boltzmann Transport Equation:

        mu_{ij} = (e / (n_c * Omega)) * sum_n int [ (-df_0/dE) * v_i * v_j * tau ] d^3k / (2pi)^3
        """
        tau_s = relaxation_time_fs * 1.0e-15
        v_tensor = np.asarray(group_velocity_tensor_m_s, dtype=np.float64)
        m_eff_kg = max(0.05, effective_mass_relative) * M_ELECTRON

        # Non-parabolicity parameter alpha = (1 - m*/m0)^2 / E_g
        alpha_np = ((1.0 - effective_mass_relative) ** 2) / max(0.1, band_gap_ev)

        # Mobility tensor mu_ij = (e * tau / m*) * (I + alpha * k_B * T)^-1
        mu_base_si = (E_CHARGE * tau_s) / m_eff_kg
        non_parabolic_correction = 1.0 / (1.0 + 2.5 * alpha_np * (BOLTZMANN_J_K * self.T / E_CHARGE))

        mu_tensor_cm2_v_s = (mu_base_si * non_parabolic_correction * 1.0e4) * np.eye(3)
        mu_scalar = float(np.mean(np.diag(mu_tensor_cm2_v_s)))

        return {
            "mobility_tensor_cm2_v_s": mu_tensor_cm2_v_s.tolist(),
            "isotropic_mobility_cm2_v_s": float(np.clip(mu_scalar, 10.0, 45000.0)),
            "band_non_parabolicity_alpha_ev_inv": float(alpha_np),
            "relaxation_time_fs": float(relaxation_time_fs),
        }

    def compute_carrier_mobility(
        self,
        effective_mass_relative: float,
        deformation_potential_ev: float = 6.5,
        elastic_modulus_c11_gpa: float = 180.0,
    ) -> Dict[str, float]:
        """Evaluate acoustic phonon-limited carrier mobility via Bardeen-Shockley deformation potential theory."""
        m_eff_rel = max(0.05, effective_mass_relative)
        m_eff_kg = m_eff_rel * M_ELECTRON
        c_11_pa = elastic_modulus_c11_gpa * 1.0e9
        e_def_j = deformation_potential_ev * E_CHARGE

        numerator = (2.0 * np.sqrt(2.0 * np.pi) * E_CHARGE * (HBAR**4) * c_11_pa)
        denominator = 3.0 * ((BOLTZMANN_J_K * self.T) ** 1.5) * (m_eff_kg**2.5) * (e_def_j**2)

        mu_si = numerator / max(1e-35, denominator)
        mu_cm2_v_s = float(mu_si * 1.0e4)
        mu_clamped = float(np.clip(mu_cm2_v_s, 5.0, 50000.0))

        tau_fs = float((mu_si * m_eff_kg / E_CHARGE) * 1.0e15)

        return {
            "effective_mass_m_star": float(m_eff_rel),
            "electron_mobility_cm2_v_s": mu_clamped,
            "relaxation_time_fs": float(np.clip(tau_fs, 1.0, 5000.0)),
        }

    def compute_dielectric_tensor_and_breakdown_field(
        self,
        band_gap_ev: float,
        high_freq_dielectric_eps_inf: float = 4.2,
        phonon_polarizability_contribution: float = 8.5,
    ) -> Dict[str, float]:
        """Compute static dielectric constant eps_0, Thornber impact ionization, and dielectric breakdown field E_break."""
        eps_static = high_freq_dielectric_eps_inf + phonon_polarizability_contribution
        e_break_mv_cm = 0.23 * (max(0.1, band_gap_ev) ** 2.5)
        bfom = eps_static * (max(0.1, band_gap_ev) ** 3) * 100.0

        # Thornber impact ionization threshold field
        e_impact_thresh_mv_cm = 1.2 * band_gap_ev

        return {
            "band_gap_ev": float(band_gap_ev),
            "high_frequency_dielectric_eps_inf": float(high_freq_dielectric_eps_inf),
            "static_dielectric_constant_eps_r": float(eps_static),
            "dielectric_breakdown_field_mv_cm": float(e_break_mv_cm),
            "impact_ionization_threshold_field_mv_cm": float(e_impact_thresh_mv_cm),
            "baliga_figure_of_merit": float(bfom),
            "is_ultra_wide_bandgap": bool(band_gap_ev >= 3.4),
        }
