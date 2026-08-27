"""Electronic Structure, Semiconductor Transport, Effective Mass Tensors & Dielectric Breakdown."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.constants import HBAR, M_ELECTRON, E_CHARGE, BOLTZMANN_J_K, EPSILON_0


class SemiconductorElectronicEngine:
    """Evaluates electronic band structures, anisotropic effective mass tensors, carrier mobility, and dielectric breakdown."""

    def __init__(self, temperature_k: float = 300.0):
        self.T = temperature_k

    def compute_effective_mass_tensor(
        self,
        band_curvature_ev_ang2: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Compute 3x3 effective mass tensor m*_{ij} = hbar^2 * (d^2 E / d k_i d k_j)^-1 in units of electron rest mass m_0."""
        # Curvature d2E/dk2 in eV*Å^2 -> J*m^2
        d2E_dk2_si = np.asarray(band_curvature_ev_ang2, dtype=np.float64) * E_CHARGE * 1.0e-20

        # m* = hbar^2 / (d^2 E / d k^2)
        inv_curvature = np.linalg.pinv(d2E_dk2_si)
        m_star_kg = (HBAR**2) * inv_curvature
        m_star_relative = m_star_kg / M_ELECTRON

        # Symmetrize
        m_star_relative = 0.5 * (m_star_relative + m_star_relative.T)
        m_eff_scalar = float(np.cbrt(np.abs(np.linalg.det(m_star_relative))))
        return m_star_relative, m_eff_scalar

    def compute_carrier_mobility(
        self,
        effective_mass_relative: float,
        deformation_potential_ev: float = 6.5,
        elastic_modulus_c11_gpa: float = 180.0,
    ) -> Dict[str, float]:
        """Evaluate acoustic phonon-limited carrier mobility via Bardeen-Shockley deformation potential theory:

        mu = (2 * sqrt(2 * pi) * e * hbar^4 * C_11) / (3 * (k_B * T)^(3/2) * (m*)^(5/2) * E_def^2)
        """
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
        """Compute static dielectric constant eps_0 and dielectric breakdown field E_break."""
        eps_static = high_freq_dielectric_eps_inf + phonon_polarizability_contribution
        e_break_mv_cm = 0.23 * (max(0.1, band_gap_ev) ** 2.5)
        bfom = eps_static * (max(0.1, band_gap_ev) ** 3) * 100.0

        return {
            "band_gap_ev": float(band_gap_ev),
            "high_frequency_dielectric_eps_inf": float(high_freq_dielectric_eps_inf),
            "static_dielectric_constant_eps_r": float(eps_static),
            "dielectric_breakdown_field_mv_cm": float(e_break_mv_cm),
            "baliga_figure_of_merit": float(bfom),
            "is_ultra_wide_bandgap": bool(band_gap_ev >= 3.4),
        }
