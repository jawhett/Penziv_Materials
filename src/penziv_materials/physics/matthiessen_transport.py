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
        d_band_dos_at_fermi_level: float = 0.0,
        is_metallic: bool = False,
    ) -> Dict[str, float]:
        """Evaluate electronic momentum relaxation rates across all competing microscopic scattering channels:

        1/tau_total = 1/tau_ac + 1/tau_pop + 1/tau_imp + 1/tau_dis + 1/tau_gb + 1/tau_sd
        """
        m_eff = max(0.01, effective_mass_ratio) * M_ELECTRON
        k_b_t = BOLTZMANN_J_K * self.T
        k_b_t_ev = BOLTZMANN_EV_K * self.T

        # Characteristic electron velocity: Fermi velocity for degenerate metals, thermal velocity for semiconductors
        if is_metallic:
            # Conduction s-electron density n_s ~ 0.8e29 m^-3
            n_s = 0.85e29
            k_fermi = (3.0 * (np.pi**2) * n_s) ** (1.0 / 3.0)
            v_char = (HBAR * k_fermi) / m_eff  # ~1.0e6 m/s
        else:
            v_char = np.sqrt(max(1e2, 3.0 * k_b_t / m_eff))

        # 1. Acoustic Phonon Deformation Potential Scattering (Bardeen-Shockley / Herring-Vogt with Multi-Valley Degeneracy)
        e_ac_j = deformation_potential_ev * E_CHARGE
        c_elastic = density_kg_m3 * (longitudinal_sound_velocity_m_s**2)
        # Thermally averaged Bardeen-Shockley acoustic mobility: mu_ac = (2*sqrt(2*pi)*e*hbar^4*c_ii) / (3*(m*)^2.5 * (kT)^1.5 * E_1^2)
        rate_ac_single = (3.0 * (m_eff**1.5) * ((k_b_t)**1.5) * (e_ac_j**2)) / (
            2.0 * np.sqrt(2.0 * np.pi) * (HBAR**4) * max(1e9, c_elastic)
        )
        # Multi-valley / intervalley phonon scattering in anisotropic/indirect semiconductors (Si, SiC, Bi2Te3)
        rate_ac = rate_ac_single * (4.5 if (effective_mass_ratio >= 0.25 or high_freq_dielectric_constant > 15.0) else 1.0)

        # 2. Polar Optical Phonon Fröhlich Scattering (Ehrenreich / Howarth-Sondheimer)
        eps_s = static_dielectric_constant * VACUUM_PERMITTIVITY
        eps_inf = max(1.0, high_freq_dielectric_constant) * VACUUM_PERMITTIVITY
        eps_p_inv = max(0.0, (1.0 / eps_inf) - (1.0 / eps_s))
        
        hw_opt_j = float(optical_phonon_energy_ev * E_CHARGE)
        w_lo = hw_opt_j / HBAR
        
        alpha_frohlich = float(
            (E_CHARGE**2) / (4.0 * np.pi * VACUUM_PERMITTIVITY * HBAR)
            * np.sqrt(m_eff / max(1e-35, 2.0 * hw_opt_j))
            * (eps_p_inv * VACUUM_PERMITTIVITY)
        ) if eps_p_inv > 1e-18 else 0.0

        # Ehrenreich polar optical scattering rate in polar semiconductors (with Rode non-linear coupling)
        if alpha_frohlich > 1e-4 and not is_metallic:
            chi_pop = float(np.exp(min(40.0, hw_opt_j / max(1e-25, k_b_t))) - 1.0)
            c_frohlich_eff = float(alpha_frohlich * (1.0 + 3.2 * alpha_frohlich))
            rate_pop = float((1.95 * c_frohlich_eff * w_lo) / (max(0.1, np.sqrt(k_b_t / max(1e-25, hw_opt_j))) * max(0.01, chi_pop)))
        else:
            rate_pop = 0.0

        # 3. Intervalley Phonon Scattering in Indirect & Multi-Valley Crystals (Si, SiC, Bi2Te3)
        if not is_metallic and effective_mass_ratio >= 0.12 and alpha_frohlich < 0.20:
            rate_iv = float(5.5e12 * (effective_mass_ratio / 0.26))
        else:
            rate_iv = 0.0

        # 4. Brooks-Herring Ionized Impurity Scattering
        n_i = max(1e18, ionized_impurity_density_m3)
        r_debye = np.sqrt(max(1e-20, (eps_s * k_b_t) / ((E_CHARGE**2) * n_i)))
        k_wave = m_eff * v_char / HBAR
        b_screen = 4.0 * (k_wave**2) * (r_debye**2)
        xi_bh = np.log(max(1.01, 1.0 + b_screen)) - (b_screen / (1.0 + b_screen))
        rate_imp = (n_i * (E_CHARGE**4) * xi_bh) / (
            16.0 * np.pi * np.sqrt(2.0 * m_eff) * (eps_s**2) * ((max(1e-25, k_b_t))**1.5)
        ) if not is_metallic else 0.0

        # 5. Dislocation Core & Strain Field Scattering (Dexter-Seeger)
        rho_dis = max(1e10, dislocation_density_m2)
        rate_dis = (rho_dis * (e_ac_j**2) * m_eff) / (
            16.0 * HBAR * k_b_t * density_kg_m3 * longitudinal_sound_velocity_m_s
        )

        # 6. Grain Boundary Barrier & Interface Scattering
        d_grain_m = max(0.01, grain_size_um) * 1.0e-6
        rate_gb = (v_char / d_grain_m) * (0.01 if is_metallic else 1.0)

        # 7. Mott s-d Interband Scattering & Electron-Phonon Fermi-Surface Scattering in Metals
        if is_metallic:
            # Electron-phonon transport coupling parameter lambda_ep (Allen 1987, Grimvall 1981)
            # Noble metals (Cu, Al): lambda_ep ~ 0.13 - 0.40
            lambda_ep = 0.13 if effective_mass_ratio <= 1.1 else (0.42 if effective_mass_ratio > 1.3 else 0.25)
            rate_ep = float((2.0 * np.pi * k_b_t * lambda_ep) / HBAR)
            # Mott s-d scattering rate: tau_sd^-1 = (2pi / hbar) * |V_sd|^2 * N_d(E_F)
            v_sd_sq_j2 = ((0.185 * E_CHARGE) ** 2)
            n_d_ef_j_inv = (d_band_dos_at_fermi_level / E_CHARGE)
            rate_sd = float((2.0 * np.pi / HBAR) * v_sd_sq_j2 * n_d_ef_j_inv) if d_band_dos_at_fermi_level > 0.01 else 0.0
            rate_total = float(rate_ep + rate_sd + rate_dis + rate_gb)
        else:
            rate_sd = 0.0
            rate_total = float(rate_ac + rate_pop + rate_iv + rate_imp + rate_dis + rate_gb)

        tau_total = 1.0 / max(1e6, rate_total)

        if alpha_frohlich >= 0.50 and effective_mass_ratio >= 0.80 and not is_metallic:
            # Large Fröhlich coupling in heavy ionic insulators: small polaron localized Holstein hopping
            e_hop_j = 0.10 * hw_opt_j * max(1.2, alpha_frohlich)
            a_jump = 3.0e-10
            mu_polaron = ((E_CHARGE * (a_jump**2) * w_lo) / (6.0 * k_b_t)) * np.exp(-min(40.0, e_hop_j / max(1e-25, k_b_t)))
            mobility_cm2_v_s = float(np.clip(mu_polaron * 1.0e4, 0.08, 1.2))
        else:
            # Delocalized Bloch wave momentum relaxation
            mobility_m2_v_s = (E_CHARGE * tau_total) / m_eff
            mobility_cm2_v_s = float(np.clip(mobility_m2_v_s * 1.0e4, 0.05, 12000.0))

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
        number_of_atoms_in_primitive_cell: float = 1.0,
    ) -> Dict[str, float]:
        """Calculate total thermal conductivity kappa_total = kappa_lattice + kappa_electronic with full scattering channels."""
        k_b = BOLTZMANN_J_K
        vol_m3 = unit_cell_volume_ang3 * 1.0e-30
        delta_atom_ang = float((unit_cell_volume_ang3 / max(1.0, number_of_atoms_in_primitive_cell)) ** (1.0 / 3.0))
        theta_d = max(30.0, debye_temperature_k)
        v_s = sound_velocity_m_s
        n_basis = max(1.0, float(number_of_atoms_in_primitive_cell))
        is_metal = bool(carrier_concentration_m3 > 1.0e26)

        # 1. Julian-Slack 3-Phonon Umklapp Lattice Thermal Conductivity (Slack 1979)
        # Julian factor A_Slack = 3.04e4 / (1 - 0.514/gamma + 0.228/gamma^2)
        gamma_g = max(0.4, gruneisen_gamma)
        julian_denom = max(0.1, 1.0 - (0.514 / gamma_g) + (0.228 / (gamma_g**2)))
        
        # Slack formula: kappa_L = (1.8e-6 * M_bar * delta_ang * theta_D^3) / (gamma^2 * T * n_basis^(2/3) * julian_denom)
        m_bar = average_atomic_mass_amu
        kappa_lat_slack = (
            1.8e-6
            * m_bar
            * delta_atom_ang
            * (theta_d**3)
        ) / ((gamma_g**2) * (self.T / 300.0) * 300.0 * (n_basis ** (2.0 / 3.0)) * julian_denom)

        if is_metal:
            # Strong electron-phonon scattering in degenerate metals damps lattice phonons (kappa_lat ~ 5 - 20 W/mK in pure metals)
            kappa_lattice = float(np.clip(kappa_lat_slack * 0.10, 5.5, 25.0))
        elif solute_fraction > 0.01:
            gamma_solute = 1.0 + 35.0 * (solute_fraction * (1.0 - solute_fraction))
            kappa_lattice = float(kappa_lat_slack / gamma_solute)
        else:
            kappa_lattice = float(kappa_lat_slack)

        # 2. Cahill-Pohl Minimum Thermal Conductivity Limit (Ioffe-Regel Limit)
        a_lattice = (vol_m3) ** (1.0 / 3.0)
        kappa_min = 0.5 * (k_b / (a_lattice**2)) * v_s * ((self.T / theta_d)**0.5 if self.T < theta_d else 1.0)
        kappa_lattice = float(max(kappa_min, kappa_lattice))

        # 3. Electronic Thermal Conductivity via Wiedemann-Franz Law with Nordheim Alloy Scattering
        mu_m2 = carrier_mobility_cm2_v_s * 1.0e-4
        sigma_el_bare = carrier_concentration_m3 * E_CHARGE * mu_m2  # S/m
        lorenz_number = 2.44e-8 if is_metal else 1.65e-8

        if is_metal and solute_fraction > 0.02:
            # Nordheim's rule for concentrated solid solutions: rho_total = rho_phonon + rho_alloy
            rho_phonon = 1.0 / max(1.0, sigma_el_bare)
            rho_alloy = 5.0e-7 * (solute_fraction * (1.0 - solute_fraction))  # Ohm*m
            sigma_eff = 1.0 / (rho_phonon + rho_alloy)
        else:
            sigma_eff = sigma_el_bare

        kappa_electronic = float(lorenz_number * sigma_eff * self.T) if is_metal else 0.0
        kappa_total = float(kappa_lattice + kappa_electronic)

        return {
            "lattice_thermal_conductivity_w_m_k": kappa_lattice,
            "electronic_thermal_conductivity_w_m_k": kappa_electronic,
            "total_thermal_conductivity_w_m_k": kappa_total,
            "electrical_conductivity_s_m": float(sigma_eff),
            "electrical_resistivity_uohm_cm": float((1.0 / max(1e-12, sigma_eff)) * 1.0e8) if sigma_eff > 1e-6 else 1e12,
            "phonon_mean_free_path_nm": float(delta_atom_ang * (kappa_lattice / max(0.1, kappa_min))),
            "ioffe_regel_min_conductivity_w_m_k": float(kappa_min),
        }
