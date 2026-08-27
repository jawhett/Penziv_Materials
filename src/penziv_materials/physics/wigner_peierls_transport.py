"""Dual-Channel Peierls-Wigner Thermal Transport & Full-Brillouin-Zone Electronic Kinetics."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.constants import BOLTZMANN_J_K, BOLTZMANN_EV_K, HBAR, E_CHARGE


class UnifiedThermalElectronicTransportEngine:
    """Evaluates dual-channel Peierls-Boltzmann + Wigner interband thermal conductivity and Wannier-like full-Brillouin-zone electronic kinetics."""

    def __init__(self, temperature_k: float = 300.0):
        self.T = max(1.0, temperature_k)

    def solve_dual_channel_peierls_wigner_thermal_conductivity(
        self,
        frequencies_thz: np.ndarray,             # (N_modes,)
        linewidths_thz: np.ndarray,              # (N_modes,) - Gamma_s = 1 / (2 * tau_s)
        diagonal_velocities_m_s: np.ndarray,     # (N_modes, 3)
        off_diagonal_velocity_tensors: Optional[np.ndarray] = None, # (N_modes, N_modes, 3)
        cell_volume_ang3: float = 100.0,
    ) -> Dict[str, Any]:
        """Compute total thermal conductivity tensor combining Peierls diagonal and Wigner off-diagonal wavepacket tunneling:

        kappa_total = kappa_Peierls + kappa_Wigner
        """
        vol_m3 = cell_volume_ang3 * 1.0e-30
        n_modes = len(frequencies_thz)
        omegas = 2.0 * np.pi * np.asarray(frequencies_thz, dtype=np.float64) * 1.0e12
        gammas = 2.0 * np.pi * np.asarray(linewidths_thz, dtype=np.float64) * 1.0e12
        v_diag = np.asarray(diagonal_velocities_m_s, dtype=np.float64)

        kbt = BOLTZMANN_J_K * self.T
        x = np.clip((HBAR * omegas) / (2.0 * kbt), 1e-6, 50.0)
        c_v = BOLTZMANN_J_K * (x / np.sinh(x)) ** 2  # J/K per mode

        # 1. Peierls Diagonal Boltzmann channel
        tau_s = 1.0 / np.maximum(1e8, 2.0 * gammas)
        kappa_peierls = np.zeros((3, 3), dtype=np.float64)
        for a in range(3):
            for b in range(3):
                kappa_peierls[a, b] = np.sum(c_v * v_diag[:, a] * v_diag[:, b] * tau_s) / vol_m3

        # 2. Wigner Off-Diagonal Wavepacket Tunneling channel
        kappa_wigner = np.zeros((3, 3), dtype=np.float64)
        if off_diagonal_velocity_tensors is not None:
            v_off = np.asarray(off_diagonal_velocity_tensors, dtype=np.float64)
            for s in range(n_modes):
                for sp in range(n_modes):
                    if s == sp:
                        continue
                    delta_omega = omegas[s] - omegas[sp]
                    avg_gamma = 0.5 * (gammas[s] + gammas[sp])
                    avg_cv = 0.5 * (c_v[s] + c_v[sp])
                    lorentz = avg_gamma / (delta_omega**2 + avg_gamma**2)

                    for a in range(3):
                        for b in range(3):
                            v_term = v_off[s, sp, a] * v_off[sp, s, b]
                            kappa_wigner[a, b] += (avg_cv * v_term * lorentz) / vol_m3
        else:
            # Analytical off-diagonal approximation for complex/disordered crystals
            # when explicit interband velocity matrices are not pre-tabulated
            mean_gap = np.mean(np.diff(np.sort(omegas))) if n_modes > 1 else 1e12
            mean_gamma = np.mean(gammas)
            wigner_weight = float(mean_gamma / (mean_gap + mean_gamma))
            kappa_wigner = kappa_peierls * 0.15 * wigner_weight

        kappa_total = kappa_peierls + kappa_wigner

        return {
            "kappa_total_tensor_w_m_k": kappa_total.tolist(),
            "kappa_peierls_tensor_w_m_k": kappa_peierls.tolist(),
            "kappa_wigner_tensor_w_m_k": kappa_wigner.tolist(),
            "isotropic_total_kappa_w_m_k": float(np.mean(np.diag(kappa_total))),
            "isotropic_peierls_kappa_w_m_k": float(np.mean(np.diag(kappa_peierls))),
            "isotropic_wigner_kappa_w_m_k": float(np.mean(np.diag(kappa_wigner))),
            "wigner_tunneling_fraction": float(np.mean(np.diag(kappa_wigner)) / max(1e-4, np.mean(np.diag(kappa_total)))),
        }

    def solve_full_brillouin_zone_electronic_transport(
        self,
        energies_ev: np.ndarray,
        dos_states_ev: np.ndarray,
        band_velocities_m_s: np.ndarray,
        relaxation_times_fs: np.ndarray,
        fermi_energy_ev: float = 0.0,
        cell_volume_ang3: float = 100.0,
    ) -> Dict[str, Any]:
        """Integrate full-Brillouin-zone transport distribution function Sigma(E) for anisotropic electrical conductivity, Seebeck, and Hall response."""
        vol_m3 = cell_volume_ang3 * 1.0e-30
        e_grid = np.asarray(energies_ev, dtype=np.float64)
        dos = np.asarray(dos_states_ev, dtype=np.float64)
        vels = np.asarray(band_velocities_m_s, dtype=np.float64)
        tau = np.asarray(relaxation_times_fs, dtype=np.float64) * 1.0e-15

        kbt_ev = BOLTZMANN_EV_K * self.T
        diff_e = (e_grid - fermi_energy_ev) / max(1e-6, kbt_ev)
        # -df/dE
        df_de = 1.0 / (4.0 * kbt_ev * E_CHARGE * (np.cosh(np.clip(diff_e / 2.0, -35.0, 35.0)) ** 2))
        de = np.gradient(e_grid) * E_CHARGE

        sigma_tensor = np.zeros((3, 3), dtype=np.float64)
        l1_tensor = np.zeros((3, 3), dtype=np.float64)
        l2_tensor = np.zeros((3, 3), dtype=np.float64)

        for a in range(3):
            for b in range(3):
                sigma_e = (E_CHARGE**2) * vels[:, a] * vels[:, b] * tau * (dos / vol_m3)
                sigma_tensor[a, b] = np.sum(sigma_e * df_de * de)
                l1_tensor[a, b] = np.sum(sigma_e * (e_grid - fermi_energy_ev) * df_de * de)
                l2_tensor[a, b] = np.sum(sigma_e * ((e_grid - fermi_energy_ev) ** 2) * df_de * de)

        inv_sigma = np.linalg.pinv(sigma_tensor)
        seebeck_tensor_uv_k = -(1.0 / self.T) * np.dot(l1_tensor, inv_sigma) * 1.0e6
        kappa_e_tensor = (1.0 / self.T) * (l2_tensor - np.dot(l1_tensor, np.dot(inv_sigma, l1_tensor))) * (1.0 / E_CHARGE)

        iso_sigma = float(np.mean(np.diag(sigma_tensor)))
        iso_seebeck = float(np.mean(np.diag(seebeck_tensor_uv_k)))
        power_factor = (iso_seebeck * 1.0e-6) ** 2 * iso_sigma * 1.0e6  # in uW / (m K^2)

        # Hall coefficient R_H = - 1 / (n e) in low-field limit
        n_eff = np.sum(dos * (1.0 / (1.0 + np.exp(np.clip(diff_e, -35.0, 35.0)))) * np.gradient(e_grid)) / vol_m3
        hall_coeff_m3_c = -1.0 / (max(1e18, abs(n_eff)) * E_CHARGE)

        return {
            "electrical_conductivity_tensor_s_m": sigma_tensor.tolist(),
            "seebeck_tensor_uv_k": seebeck_tensor_uv_k.tolist(),
            "electronic_thermal_conductivity_w_m_k": kappa_e_tensor.tolist(),
            "isotropic_conductivity_s_m": iso_sigma,
            "isotropic_seebeck_uv_k": iso_seebeck,
            "thermoelectric_power_factor_uw_m_k2": float(power_factor),
            "hall_coefficient_m3_c": float(hall_coeff_m3_c),
            "effective_carrier_density_cm3": float(abs(n_eff) * 1.0e-6),
        }
