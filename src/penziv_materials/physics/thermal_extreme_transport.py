"""Thermal Transport, Phonon BTE, Wigner-Pohl Thermal Conductivity & Space Outgassing."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.constants import BOLTZMANN_J_K, HBAR, GAS_CONSTANT_J_MOL_K


class ThermalExtremeTransportEngine:
    """Evaluates lattice thermal conductivity via Slack BTE and Wigner-Pohl unified transport, and space outgassing."""

    def __init__(self, temperature_k: float = 300.0):
        self.T = temperature_k

    def compute_lattice_thermal_conductivity_slack(
        self,
        average_atomic_mass_amu: float,
        debye_temperature_k: float,
        volume_per_atom_ang3: float,
        gruneisen_parameter: float = 1.65,
        num_atoms_per_primitive_cell: int = 2,
    ) -> Dict[str, float]:
        """Compute intrinsic acoustic phonon thermal conductivity kappa_lat(T) via Slack relation."""
        m_amu = average_atomic_mass_amu
        theta_d = debye_temperature_k
        delta_ang = (volume_per_atom_ang3) ** (1.0 / 3.0)
        gamma = gruneisen_parameter
        n_cell = max(1, num_atoms_per_primitive_cell)

        a_slack = 3.1e-4
        gamma_factor = 1.0 - (0.514 / gamma) + (0.228 / (gamma**2))
        numerator = a_slack * m_amu * (theta_d**3) * delta_ang
        denominator = (gamma**2) * (n_cell ** (2.0 / 3.0)) * max(1.0, self.T)

        kappa_slack = (numerator / max(1e-30, denominator)) * gamma_factor
        kappa_clamped = float(np.clip(kappa_slack, 0.05, 3500.0))

        c_v_approx = 3.0 * (BOLTZMANN_J_K / (volume_per_atom_ang3 * 1.0e-30))
        v_s_approx = (BOLTZMANN_J_K * theta_d / HBAR) * (delta_ang * 1.0e-10)
        mfp_nm = float((3.0 * kappa_clamped / (c_v_approx * v_s_approx)) * 1.0e9)

        return {
            "lattice_thermal_conductivity_w_m_k": kappa_clamped,
            "phonon_mean_free_path_nm": float(np.clip(mfp_nm, 0.2, 500.0)),
            "sound_velocity_m_s": float(v_s_approx),
            "debye_temperature_k": float(theta_d),
        }

    def compute_wigner_pohl_thermal_conductivity(
        self,
        phonon_frequencies_thz: np.ndarray,
        phonon_linewidths_thz: np.ndarray,
        group_velocities_m_s: np.ndarray,
        cell_volume_ang3: float = 120.0,
    ) -> Dict[str, Any]:
        """Evaluate Wigner-Pohl unified thermal transport resolving diagonal wave-like and off-diagonal diffusive channels:

        kappa_total = kappa_diag + kappa_offdiag
        """
        freqs = np.asarray(phonon_frequencies_thz, dtype=np.float64) * 1.0e12  # THz -> Hz
        omegas = 2.0 * np.pi * freqs
        gammas = 2.0 * np.pi * np.asarray(phonon_linewidths_thz, dtype=np.float64) * 1.0e12
        vels = np.asarray(group_velocities_m_s, dtype=np.float64)

        n_modes = len(omegas)
        vol_m3 = cell_volume_ang3 * 1.0e-30
        kbt = BOLTZMANN_J_K * max(1.0, self.T)

        # 1. Diagonal Peierls/BTE contribution
        x_s = np.clip((HBAR * omegas) / (2.0 * kbt), 1e-6, 50.0)
        c_mode = BOLTZMANN_J_K * (x_s / np.sinh(x_s)) ** 2
        tau_s = 1.0 / np.maximum(1e8, gammas)

        kappa_diag = float(np.sum(c_mode * (vels**2) * tau_s) / (3.0 * vol_m3))

        # 2. Off-diagonal Wigner interband hopping contribution
        kappa_offdiag = 0.0
        for s in range(n_modes):
            for sp in range(s + 1, n_modes):
                omega_diff = omegas[s] - omegas[sp]
                gamma_sum = 0.5 * (gammas[s] + gammas[sp])
                wigner_lorentzian = gamma_sum / (omega_diff**2 + gamma_sum**2)
                interband_vel = 0.5 * (vels[s] + vels[sp])
                c_interband = np.sqrt(c_mode[s] * c_mode[sp])
                kappa_offdiag += 2.0 * (c_interband * (interband_vel**2) * wigner_lorentzian) / (3.0 * vol_m3)

        kappa_total = kappa_diag + kappa_offdiag

        return {
            "thermal_conductivity_total_w_m_k": float(np.clip(kappa_total, 0.1, 3000.0)),
            "kappa_diagonal_peierls_w_m_k": float(np.clip(kappa_diag, 0.05, 3000.0)),
            "kappa_offdiagonal_wigner_w_m_k": float(np.clip(kappa_offdiag, 0.0, 1000.0)),
            "wigner_diffusive_fraction": float(kappa_offdiag / max(1e-6, kappa_total)),
        }

    def compute_space_vacuum_outgassing_rate_hkl(
        self,
        molecular_weight_g_mol: float,
        vapor_pressure_pa: float,
        condensation_coefficient: float = 1.0,
    ) -> Dict[str, Any]:
        """Compute Hertz-Knudsen-Langmuir sublimation mass loss rate into vacuum J_evap."""
        m_kg_mol = molecular_weight_g_mol * 1.0e-3
        alpha_v = condensation_coefficient
        p_sat = max(1.0e-15, vapor_pressure_pa)

        j_evap_kg_m2_s = (alpha_v * p_sat) / np.sqrt(2.0 * np.pi * m_kg_mol * GAS_CONSTANT_J_MOL_K * self.T)
        mass_loss_rate_g_cm2_s = j_evap_kg_m2_s * 1.0e-1

        is_space_stable = bool(mass_loss_rate_g_cm2_s < 1.0e-6)

        return {
            "sublimation_mass_flux_kg_m2_s": float(j_evap_kg_m2_s),
            "mass_loss_rate_g_cm2_s": float(mass_loss_rate_g_cm2_s),
            "is_space_vacuum_stable": is_space_stable,
            "vapor_pressure_pa": float(p_sat),
        }
