"""Scale-Bridging Multi-Channel Matthiessen Electronic and Phonon Transport Engine."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from penziv_materials.core.constants import (
    BOLTZMANN_J_K,
    BOLTZMANN_EV_K,
    HBAR,
    E_CHARGE,
    M_ELECTRON,
    VACUUM_PERMITTIVITY,
)


class MatthiessenTransportEngine:
    """Evaluates unified electronic and lattice thermal transport via coupled multi-channel Matthiessen relaxation."""

    def __init__(self, temperature_k: float = 300.0):
        self.T = max(1.0, float(temperature_k))

    def compute_electronic_relaxation_rates(
        self,
        effective_mass_ratio: float = 1.0,
        deformation_potential_ev: float = 8.5,
        static_dielectric_constant: float = 12.0,
        high_freq_dielectric_constant: float = 10.0,
        longitudinal_sound_velocity_m_s: float = 5000.0,
        density_kg_m3: float = 7800.0,
        ionized_impurity_density_m3: float = 1.0e20,
        dislocation_density_m2: float = 1.0e12,
        grain_size_um: float = 30.0,
        optical_phonon_energy_ev: float = 0.035,
    ) -> Dict[str, float]:
        """Evaluate electronic momentum relaxation rates across all competing microscopic scattering channels:

        1/tau_total = 1/tau_ac + 1/tau_pop + 1/tau_imp + 1/tau_dis + 1/tau_gb
        """
        m_eff = effective_mass_ratio * M_ELECTRON
        k_b_t = BOLTZMANN_J_K * self.T
        k_b_t_ev = BOLTZMANN_EV_K * self.T

        # 1. Acoustic Phonon Deformation Potential Scattering (Bardeen-Shockley)
        e_ac_j = deformation_potential_ev * E_CHARGE
        c_elastic = density_kg_m3 * (longitudinal_sound_velocity_m_s**2)
        # Average thermal velocity v_th = sqrt(3 k_B T / m_eff)
        v_th = np.sqrt(max(1e2, 3.0 * k_b_t / m_eff))
        rate_ac = (np.sqrt(2.0) * (e_ac_j**2) * (m_eff**1.5) * np.sqrt(k_b_t) * k_b_t) / (
            np.pi * (HBAR**4) * c_elastic
        )

        # 2. Polar Optical Phonon Fröhlich Scattering (dominant in polar semiconductors/oxides)
        eps_s = static_dielectric_constant * VACUUM_PERMITTIVITY
        eps_inf = max(1.0, high_freq_dielectric_constant) * VACUUM_PERMITTIVITY
        eps_p_inv = max(0.0, (1.0 / eps_inf) - (1.0 / eps_s))
        hw_opt_j = optical_phonon_energy_ev * E_CHARGE
        # Bose-Einstein occupation of optical phonons
        n_pop = 1.0 / max(1e-6, np.exp(min(50.0, optical_phonon_energy_ev / max(1e-4, k_b_t_ev))) - 1.0)
        # Fröhlich dimensionless coupling constant alpha_F
        alpha_frohlich = float(
            (E_CHARGE**2) / (4.0 * np.pi * VACUUM_PERMITTIVITY * HBAR)
            * np.sqrt(m_eff / max(1e-35, 2.0 * hw_opt_j))
            * (eps_p_inv * VACUUM_PERMITTIVITY)
        ) if eps_p_inv > 1e-18 else 0.0

        if alpha_frohlich > 1e-6:
            w_lo = hw_opt_j / HBAR
            rate_pop = float((2.0 * alpha_frohlich * w_lo * n_pop) / np.sqrt(1.0 + (hw_opt_j / max(1e-25, k_b_t))))
        else:
            rate_pop = 0.0

        # 3. Brooks-Herring Ionized Impurity Scattering
        n_i = max(1e18, ionized_impurity_density_m3)
        # Screening length r_Debye = sqrt(eps_s k_B T / (e^2 n_i))
        r_debye = np.sqrt(max(1e-20, (eps_s * k_b_t) / ((E_CHARGE**2) * n_i)))
        # k_wave = m_eff * v_th / hbar
        k_wave = m_eff * v_th / HBAR
        b_screen = 4.0 * (k_wave**2) * (r_debye**2)
        xi_bh = np.log(max(1.01, 1.0 + b_screen)) - (b_screen / (1.0 + b_screen))
        rate_imp = (n_i * (E_CHARGE**4) * xi_bh) / (
            16.0 * np.pi * np.sqrt(2.0 * m_eff) * (eps_s**2) * ((max(1e-25, k_b_t))**1.5)
        )

        # 4. Dislocation Core & Strain Field Scattering (Dexter-Seeger)
        rho_dis = max(1e10, dislocation_density_m2)
        rate_dis = (rho_dis * (e_ac_j**2) * m_eff) / (
            16.0 * HBAR * k_b_t * density_kg_m3 * longitudinal_sound_velocity_m_s
        )

        # 5. Grain Boundary Barrier & Interface Scattering
        d_grain_m = max(0.01, grain_size_um) * 1.0e-6
        rate_gb = v_th / d_grain_m

        # 6. Mott s-d Interband Scattering for Transition Metals with d-band DOS
        rate_sd = float(1.2e14 * (max(0.0, effective_mass_ratio - 1.0) / max(0.5, deformation_potential_ev)))

        rate_total = float(rate_ac + rate_pop + rate_imp + rate_dis + rate_gb + rate_sd)
        tau_total = 1.0 / max(1e6, rate_total)

        # Carrier mobility: Evaluate Fröhlich polaron coupling constant alpha_F
        alpha_frohlich = float(
            (E_CHARGE**2) / (4.0 * np.pi * VACUUM_PERMITTIVITY * HBAR)
            * np.sqrt(m_eff / max(1e-35, 2.0 * hw_opt_j))
            * (eps_p_inv * VACUUM_PERMITTIVITY)
        ) if eps_p_inv > 1e-18 else 0.0

        if alpha_frohlich >= 1.0:
            # Strong electron-phonon coupling collapses wavepacket to localized small polaron (Holstein hopping)
            e_hop_j = 0.12 * hw_opt_j * max(1.5, alpha_frohlich)
            a_jump = 3.0e-10
            w_lo = hw_opt_j / HBAR
            mu_polaron = ((E_CHARGE * (a_jump**2) * w_lo) / (6.0 * k_b_t)) * np.exp(-min(40.0, e_hop_j / max(1e-25, k_b_t)))
            mobility_cm2_v_s = float(np.clip(mu_polaron * 1.0e4, 0.05, 0.5))
        else:
            # Delocalized Bloch wave momentum relaxation
            mobility_m2_v_s = (E_CHARGE * tau_total) / m_eff
            mobility_cm2_v_s = float(np.clip(mobility_m2_v_s * 1.0e4, 0.05, 50000.0))

        return {
            "rate_acoustic_phonon_s_inv": float(rate_ac),
            "rate_polar_optical_phonon_s_inv": float(rate_pop),
            "rate_ionized_impurity_s_inv": float(rate_imp),
            "rate_dislocation_s_inv": float(rate_dis),
            "rate_grain_boundary_s_inv": float(rate_gb),
            "rate_mott_sd_s_inv": float(rate_sd),
            "rate_total_s_inv": rate_total,
            "relaxation_time_tau_s": float(tau_total),
            "carrier_mobility_cm2_v_s": mobility_cm2_v_s,
            "frohlich_coupling_alpha": alpha_frohlich,
        }

    def compute_coupled_multichannel_thermal_conductivity(
        self,
        average_atomic_mass_amu: float,
        debye_temperature_k: float,
        unit_cell_volume_ang3: float,
        sound_velocity_m_s: float,
        gruneisen_gamma: float = 1.5,
        carrier_concentration_m3: float = 1.0e28,
        carrier_mobility_cm2_v_s: float = 40.0,
        solute_fraction: float = 0.0,
        solute_mass_difference_ratio: float = 0.2,
        dislocation_density_m2: float = 1.0e12,
        grain_size_um: float = 30.0,
    ) -> Dict[str, float]:
        """Calculate total thermal conductivity kappa_total = kappa_lattice + kappa_electronic with full scattering channels."""
        k_b = BOLTZMANN_J_K
        vol_m3 = unit_cell_volume_ang3 * 1.0e-30
        a_lattice = (vol_m3) ** (1.0 / 3.0)
        theta_d = debye_temperature_k
        v_s = sound_velocity_m_s

        # 1. Phonon Umklapp 3-phonon scattering rate
        rate_umklapp = (
            2.0 * (gruneisen_gamma**2) * k_b * self.T * (BOLTZMANN_J_K * theta_d / HBAR)
        ) / (average_atomic_mass_amu * 1.66054e-27 * (v_s**2))

        # 2. Phonon Point Defect / Mass-Fluctuation Solute Scattering (Klemens)
        mass_variance = float(solute_fraction * (1.0 - solute_fraction) * (solute_mass_difference_ratio**2))
        omega_debye = (k_b * theta_d) / HBAR
        rate_solute = (vol_m3 * mass_variance * (omega_debye**4)) / (4.0 * np.pi * (v_s**3))

        # 3. Phonon Dislocation Core & Strain Field Scattering
        rho_dis = max(1e10, dislocation_density_m2)
        rate_dis_phonon = 0.02 * (gruneisen_gamma**2) * (a_lattice**2) * rho_dis * omega_debye

        # 4. Phonon Grain Boundary Boundary Scattering (Casimir)
        d_grain_m = max(0.01, grain_size_um) * 1.0e-6
        rate_boundary = v_s / d_grain_m

        # Total Phonon Relaxation Rate & Lattice Conductivity
        total_phonon_rate = max(1e8, rate_umklapp + rate_solute + rate_dis_phonon + rate_boundary)
        tau_phonon = 1.0 / total_phonon_rate

        # Dulong-Petit / Debye Heat Capacity per volume
        c_v = 3.0 * (k_b / vol_m3)
        kappa_lat_kinetic = (1.0 / 3.0) * c_v * (v_s**2) * tau_phonon

        # Cahill-Pohl Minimum Thermal Conductivity (Ioffe-Regel Limit)
        kappa_min = 0.5 * (k_b / (a_lattice**2)) * v_s * ((self.T / theta_d)**0.5 if self.T < theta_d else 1.0)
        kappa_lattice = float(max(kappa_min, kappa_lat_kinetic))

        # 5. Electronic Thermal Conductivity via Wiedemann-Franz Law with Degeneracy Lorenz Ratio
        mu_m2 = carrier_mobility_cm2_v_s * 1.0e-4
        sigma_el = carrier_concentration_m3 * E_CHARGE * mu_m2  # S/m
        is_metal = carrier_concentration_m3 > 1.0e26
        lorenz_number = 2.44e-8 if is_metal else 1.65e-8

        # Mott-Ioffe-Regel saturation in concentrated multi-component solid solutions
        sigma_mir_sat = float(1.2e6)  # Minimum metallic conductivity limit ~ 1.2e6 S/m
        if is_metal and solute_fraction > 0.05:
            sigma_eff = (sigma_el * sigma_mir_sat) / max(1.0, sigma_el + sigma_mir_sat)
        else:
            sigma_eff = sigma_el

        kappa_electronic = float(lorenz_number * sigma_eff * self.T)
        kappa_total = float(kappa_lattice + kappa_electronic)

        return {
            "lattice_thermal_conductivity_w_m_k": kappa_lattice,
            "electronic_thermal_conductivity_w_m_k": kappa_electronic,
            "total_thermal_conductivity_w_m_k": kappa_total,
            "electrical_conductivity_s_m": float(sigma_el),
            "electrical_resistivity_uohm_cm": float((1.0 / max(1e-12, sigma_el)) * 1.0e8) if sigma_el > 1e-6 else 1e12,
            "phonon_mean_free_path_nm": float(v_s * tau_phonon * 1.0e9),
            "ioffe_regel_min_conductivity_w_m_k": float(kappa_min),
        }
