"""First-Principles Linearized Phonon & Electron Boltzmann Transport Equation (BTE) Solvers."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.constants import BOLTZMANN_J_K, BOLTZMANN_EV_K, HBAR, E_CHARGE


class AbInitioBoltzmannTransportEngine:
    """First-principles thermal and electrical transport resolving full tensor response."""

    @staticmethod
    def solve_phonon_bte_tensor(
        frequencies_thz: np.ndarray,      # Shape (N_modes,)
        group_velocities_m_s: np.ndarray, # Shape (N_modes, 3)
        scattering_rates_thz: np.ndarray, # Shape (N_modes,) - from 3-phonon & 4-phonon phase space
        cell_volume_ang3: float,
        temperature_k: float,
    ) -> np.ndarray:
        """Solve exact linearized Phonon BTE under Relaxation Time Approximation:

        kappa_ab = (1 / V) * sum_s C_s(T) * v_{s,a} * v_{s,b} * tau_s
        """
        vol_m3 = cell_volume_ang3 * 1.0e-30
        omegas = 2.0 * np.pi * np.asarray(frequencies_thz, dtype=np.float64) * 1.0e12
        vels = np.asarray(group_velocities_m_s, dtype=np.float64)
        scatt = np.asarray(scattering_rates_thz, dtype=np.float64)

        kbt = BOLTZMANN_J_K * max(1.0, temperature_k)
        x = np.clip((HBAR * omegas) / (2.0 * kbt), 1e-6, 50.0)
        c_v_modes = BOLTZMANN_J_K * (x / np.sinh(x)) ** 2  # J/K per mode
        tau_s = 1.0 / np.maximum(1e8, scatt * 1.0e12)

        kappa_tensor = np.zeros((3, 3), dtype=np.float64)
        for a in range(3):
            for b in range(3):
                kappa_tensor[a, b] = np.sum(c_v_modes * vels[:, a] * vels[:, b] * tau_s) / vol_m3
        return kappa_tensor

    @staticmethod
    def solve_electron_bte_tensor(
        energies_ev: np.ndarray,          # (N_energy_bins,)
        dos_states_ev: np.ndarray,        # (N_energy_bins,)
        group_velocities_m_s: np.ndarray, # (N_energy_bins, 3)
        relaxation_times_fs: np.ndarray,  # (N_energy_bins,)
        cell_volume_ang3: float,
        temperature_k: float,
        fermi_energy_ev: float = 0.0,
    ) -> Dict[str, Any]:
        """Solve electronic transport integrals: electrical conductivity sigma, Seebeck coefficient S, and electronic thermal conductivity kappa_e."""
        vol_m3 = cell_volume_ang3 * 1.0e-30
        e_grid = np.asarray(energies_ev, dtype=np.float64)
        dos = np.asarray(dos_states_ev, dtype=np.float64)
        vels = np.asarray(group_velocities_m_s, dtype=np.float64)
        tau_s = np.asarray(relaxation_times_fs, dtype=np.float64) * 1.0e-15

        kbt_ev = BOLTZMANN_EV_K * max(1.0, temperature_k)
        diff_e = (e_grid - fermi_energy_ev) / max(1e-6, kbt_ev)
        # Derivative of Fermi-Dirac distribution: -df/dE = 1 / (4 k_B T cosh^2( (E - E_F) / 2 k_B T ))
        df_de = 1.0 / (4.0 * kbt_ev * E_CHARGE * (np.cosh(np.clip(diff_e / 2.0, -30.0, 30.0)) ** 2))

        # Transport distribution tensor Sigma_{ab}(E) = e^2 * v_a * v_b * tau(E) * g(E)
        de = np.gradient(e_grid) * E_CHARGE  # in Joules

        l0_tensor = np.zeros((3, 3), dtype=np.float64)
        l1_tensor = np.zeros((3, 3), dtype=np.float64)
        l2_tensor = np.zeros((3, 3), dtype=np.float64)

        for a in range(3):
            for b in range(3):
                sigma_e = (E_CHARGE**2) * vels[:, a] * vels[:, b] * tau_s * (dos / vol_m3)
                l0_tensor[a, b] = np.sum(sigma_e * df_de * de)
                l1_tensor[a, b] = np.sum(sigma_e * (e_grid - fermi_energy_ev) * df_de * de)
                l2_tensor[a, b] = np.sum(sigma_e * ((e_grid - fermi_energy_ev) ** 2) * df_de * de)

        # Electrical conductivity sigma = L0
        sigma_tensor = l0_tensor
        inv_sigma = np.linalg.pinv(sigma_tensor)

        # Seebeck tensor S = - (1 / (e T)) * L1 . sigma^-1
        seebeck_tensor_uv_k = - (1.0 / (max(1.0, temperature_k))) * np.dot(l1_tensor, inv_sigma) * 1.0e6

        # Electronic thermal conductivity kappa_e = (1 / T) * (L2 - L1 . sigma^-1 . L1)
        kappa_e_tensor = (1.0 / (max(1.0, temperature_k))) * (l2_tensor - np.dot(l1_tensor, np.dot(inv_sigma, l1_tensor))) * (1.0 / E_CHARGE)

        return {
            "electrical_conductivity_tensor_s_m": sigma_tensor.tolist(),
            "seebeck_tensor_uv_k": seebeck_tensor_uv_k.tolist(),
            "electronic_thermal_conductivity_tensor_w_m_k": kappa_e_tensor.tolist(),
            "isotropic_conductivity_s_m": float(np.mean(np.diag(sigma_tensor))),
            "isotropic_seebeck_uv_k": float(np.mean(np.diag(seebeck_tensor_uv_k))),
            "isotropic_kappa_e_w_m_k": float(np.mean(np.diag(kappa_e_tensor))),
        }
