"""Thermal Transport, Phonon BTE, Wigner-Pohl Thermal Conductivity & Anharmonic 3-Phonon Scattering."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.constants import BOLTZMANN_J_K, HBAR, GAS_CONSTANT_J_MOL_K, E_CHARGE


class ThermalExtremeTransportEngine:
    """Evaluates lattice thermal conductivity via Slack BTE, Wigner-Pohl unified transport, 3-phonon IFCs, and Seebeck tensors."""

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
        kappa_lat = float(max(1e-4, kappa_slack))

        c_v_approx = 3.0 * (BOLTZMANN_J_K / (volume_per_atom_ang3 * 1.0e-30))
        v_s_approx = (BOLTZMANN_J_K * theta_d / HBAR) * (delta_ang * 1.0e-10)
        mfp_nm = float((3.0 * kappa_lat / (c_v_approx * v_s_approx)) * 1.0e9)

        return {
            "lattice_thermal_conductivity_w_m_k": kappa_lat,
            "phonon_mean_free_path_nm": float(max(0.1, mfp_nm)),
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
        freqs = np.asarray(phonon_frequencies_thz, dtype=np.float64) * 1.0e12
        omegas = 2.0 * np.pi * freqs
        gammas = 2.0 * np.pi * np.asarray(phonon_linewidths_thz, dtype=np.float64) * 1.0e12
        vels = np.asarray(group_velocities_m_s, dtype=np.float64)

        n_modes = len(omegas)
        vol_m3 = cell_volume_ang3 * 1.0e-30
        kbt = BOLTZMANN_J_K * max(1.0, self.T)

        x_s = np.clip((HBAR * omegas) / (2.0 * kbt), 1e-6, 50.0)
        c_mode = BOLTZMANN_J_K * (x_s / np.sinh(x_s)) ** 2
        tau_s = 1.0 / np.maximum(1e8, gammas)

        kappa_diag = float(np.sum(c_mode * (vels**2) * tau_s) / (3.0 * vol_m3))

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
            "thermal_conductivity_total_w_m_k": float(max(1e-4, kappa_total)),
            "kappa_diagonal_peierls_w_m_k": float(max(1e-4, kappa_diag)),
            "kappa_offdiagonal_wigner_w_m_k": float(max(0.0, kappa_offdiag)),
            "wigner_diffusive_fraction": float(kappa_offdiag / max(1e-6, kappa_total)),
        }

    def compute_anharmonic_3phonon_scattering_tensor(
        self,
        group_velocity_tensor_m_s: np.ndarray,      # (N_modes, 3)
        scattering_rates_thz: np.ndarray,           # (N_modes,)
        phonon_heat_capacities_j_k: np.ndarray,     # (N_modes,)
        cell_volume_ang3: float = 120.0,
    ) -> Dict[str, Any]:
        """Compute full 3x3 anisotropic lattice thermal conductivity tensor kappa_{alpha beta}(T) from 3-phonon IFCs."""
        vol_m3 = cell_volume_ang3 * 1.0e-30
        tau_s = 1.0 / np.maximum(1e9, scattering_rates_thz * 1.0e12)
        v = np.asarray(group_velocity_tensor_m_s, dtype=np.float64)
        c_v = np.asarray(phonon_heat_capacities_j_k, dtype=np.float64)

        kappa_3x3 = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                kappa_3x3[i, j] = np.sum(c_v * v[:, i] * v[:, j] * tau_s) / vol_m3

        return {
            "thermal_conductivity_tensor_w_m_k": kappa_3x3.tolist(),
            "isotropic_kappa_w_m_k": float(np.mean(np.diag(kappa_3x3))),
            "anisotropy_ratio": float(np.max(np.diag(kappa_3x3)) / max(1e-6, np.min(np.diag(kappa_3x3)))),
        }

    def compute_thermoelectric_figure_of_merit_zt(
        self,
        seebeck_coeff_uv_k: float,
        electrical_conductivity_s_m: float,
        thermal_conductivity_w_m_k: float,
    ) -> Dict[str, float]:
        """Evaluate thermoelectric figure of merit ZT = (S^2 * sigma * T) / kappa."""
        s_v_k = seebeck_coeff_uv_k * 1.0e-6
        power_factor = (s_v_k**2) * electrical_conductivity_s_m
        zt = (power_factor * self.T) / max(0.01, thermal_conductivity_w_m_k)

        return {
            "zt": float(zt),
            "power_factor_w_m_k2": float(power_factor),
            "seebeck_v_k": float(s_v_k),
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

    def compute_cahill_pohl_minimum_thermal_conductivity(
        self,
        number_density_atoms_m3: float,
        longitudinal_sound_velocity_m_s: float,
        transverse_sound_velocity_m_s: float,
        n_integration_steps: int = 50,
    ) -> Dict[str, float]:
        """Evaluate Cahill-Pohl minimum thermal conductivity limit for disordered/amorphous media."""
        n_dens = max(1.0e26, number_density_atoms_m3)
        v_l = max(500.0, longitudinal_sound_velocity_m_s)
        v_t = max(300.0, transverse_sound_velocity_m_s)

        # Cutoff Debye temperatures for 1 longitudinal and 2 transverse acoustic polarizations
        theta_i = [
            v_i * (HBAR / BOLTZMANN_J_K) * ((6.0 * (np.pi**2) * n_dens) ** (1.0 / 3.0))
            for v_i in [v_l, v_t, v_t]
        ]
        velocities = [v_l, v_t, v_t]

        prefactor = ((np.pi / 6.0) ** (1.0 / 3.0)) * BOLTZMANN_J_K * (n_dens ** (2.0 / 3.0))
        kappa_min = 0.0

        for v_i, theta in zip(velocities, theta_i):
            upper_limit = theta / max(1.0, self.T)
            if upper_limit > 50.0:
                # Low temperature limit: integral -> pi^4 / 15
                integral = (np.pi ** 4) / 15.0
            else:
                x_vals = np.linspace(1e-4, upper_limit, n_integration_steps)
                dx = x_vals[1] - x_vals[0]
                integrand = (x_vals**3 * np.exp(x_vals)) / ((np.exp(x_vals) - 1.0) ** 2)
                integral = float(np.sum(integrand) * dx)

            term = v_i * ((self.T / max(1.0, theta)) ** 2) * integral
            kappa_min += term

        kappa_total = prefactor * kappa_min
        return {
            "cahill_pohl_kappa_min_w_m_k": float(np.clip(kappa_total, 0.05, 50.0)),
            "debye_cutoff_longitudinal_k": float(theta_i[0]),
            "debye_cutoff_transverse_k": float(theta_i[1]),
        }
