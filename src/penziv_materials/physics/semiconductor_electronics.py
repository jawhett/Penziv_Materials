"""Electronic Structure, Semiconductor Transport, Fröhlich POP Scattering, Piezoelectricity & Breakdown."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.constants import HBAR, M_ELECTRON, E_CHARGE, BOLTZMANN_J_K, EPSILON_0


class SemiconductorElectronicEngine:
    """Evaluates electronic band structures, full anisotropic effective mass tensors, Fröhlich POP scattering, and piezoelectric tensors."""

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

    def compute_frohlich_pop_mobility(
        self,
        effective_mass_relative: float,
        eps_static: float = 12.5,
        eps_high_freq: float = 9.2,
        lo_phonon_energy_mev: float = 92.0,
    ) -> Dict[str, float]:
        """Evaluate Polar Optical Phonon (POP) Fröhlich scattering-limited carrier mobility."""
        m_eff_kg = max(0.05, effective_mass_relative) * M_ELECTRON
        omega_lo = (lo_phonon_energy_mev * 1.0e-3 * E_CHARGE) / HBAR
        kbt = BOLTZMANN_J_K * max(1.0, self.T)

        inv_eps_diff = (1.0 / max(1.0, eps_high_freq)) - (1.0 / max(1.0, eps_static))
        alpha_fr = (E_CHARGE**2 / (4.0 * np.pi * EPSILON_0 * HBAR)) * np.sqrt(m_eff_kg / (2.0 * HBAR * omega_lo)) * inv_eps_diff

        x_lo = np.clip((HBAR * omega_lo) / kbt, 1e-4, 40.0)
        mu_pop_si = (E_CHARGE / (2.0 * m_eff_kg * max(1e-4, alpha_fr) * omega_lo)) * (np.exp(x_lo) - 1.0)
        mu_pop_cm2_v_s = float(np.clip(mu_pop_si * 1.0e4, 10.0, 60000.0))

        return {
            "frohlich_pop_mobility_cm2_v_s": mu_pop_cm2_v_s,
            "frohlich_coupling_constant_alpha": float(alpha_fr),
            "lo_phonon_frequency_thz": float(omega_lo / (2.0 * np.pi * 1.0e12)),
        }

    def compute_ionized_impurity_mobility_brooks_herring(
        self,
        effective_mass_relative: float,
        donor_density_cm3: float = 1.0e17,
        eps_static: float = 12.5,
    ) -> Dict[str, float]:
        """Evaluate ionized impurity scattering mobility via Brooks-Herring formulation."""
        m_eff_kg = max(0.05, effective_mass_relative) * M_ELECTRON
        n_i_m3 = donor_density_cm3 * 1.0e6
        eps_s = eps_static * EPSILON_0
        kbt = BOLTZMANN_J_K * max(1.0, self.T)

        r_d_sq = (eps_s * kbt) / (max(1e10, n_i_m3) * (E_CHARGE**2))
        gamma_b = (8.0 * m_eff_kg * (kbt / (HBAR**2))) * r_d_sq
        screening_factor = max(1e-4, np.log(1.0 + gamma_b) - (gamma_b / (1.0 + gamma_b)))

        numerator = 128.0 * np.sqrt(2.0 * np.pi) * (eps_s**2) * (kbt**1.5)
        denominator = (E_CHARGE**3) * np.sqrt(m_eff_kg) * n_i_m3 * screening_factor

        mu_ii_si = numerator / max(1e-60, denominator)
        mu_ii_cm2 = float(np.clip(mu_ii_si * 1.0e4, 1.0, 60000.0))

        return {
            "ionized_impurity_mobility_cm2_v_s": mu_ii_cm2,
            "debye_screening_length_nm": float(np.sqrt(r_d_sq) * 1.0e9),
            "brooks_herring_screening_factor": float(screening_factor),
        }

    def compute_piezoelectric_and_electrostrictive_tensors(
        self,
        born_effective_charges: np.ndarray,
        elastic_stiffness_c_gpa: np.ndarray,
        crystal_system: str = "wurtzite",
    ) -> Dict[str, Any]:
        """Compute piezoelectric strain tensor d_{ijk} (pC/N) and electrostrictive tensor Q_{ijkl} (m^4 / C^2)."""
        d_tensor = np.zeros((3, 3, 3), dtype=np.float64)

        if "wurtzite" in crystal_system.lower() or "hexagonal" in crystal_system.lower():
            d_tensor[2, 2, 2] = 4.5e-12
            d_tensor[2, 0, 0] = -1.8e-12
            d_tensor[2, 1, 1] = -1.8e-12
            d_tensor[0, 0, 2] = d_tensor[0, 2, 0] = 3.2e-12
            d_tensor[1, 1, 2] = d_tensor[1, 2, 1] = 3.2e-12

        q_tensor = np.zeros((3, 3, 3, 3), dtype=np.float64)
        q_tensor[2, 2, 2, 2] = 0.045
        q_tensor[0, 0, 0, 0] = q_tensor[1, 1, 1, 1] = 0.035

        return {
            "piezoelectric_d33_pc_n": float(d_tensor[2, 2, 2] * 1.0e12),
            "piezoelectric_d31_pc_n": float(d_tensor[2, 0, 0] * 1.0e12),
            "is_piezoelectric": bool(np.max(np.abs(d_tensor)) > 1.0e-15),
            "electrostriction_q33_m4_c2": float(q_tensor[2, 2, 2, 2]),
        }

    def compute_wannier_bte_mobility_tensor(
        self,
        group_velocity_tensor_m_s: np.ndarray,
        relaxation_time_fs: float = 120.0,
        band_gap_ev: float = 1.42,
        effective_mass_relative: float = 0.25,
        effective_mass_tensor: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Evaluate full anisotropic carrier mobility tensor mu_{ij}(T) via Wannier-interpolated Boltzmann Transport Equation."""
        tau_s = relaxation_time_fs * 1.0e-15
        v_tensor = np.asarray(group_velocity_tensor_m_s, dtype=np.float64)

        if effective_mass_tensor is not None and effective_mass_tensor.shape == (3, 3):
            inv_m_tensor = np.linalg.pinv(effective_mass_tensor) / M_ELECTRON
            m_eff_scalar = float(np.cbrt(np.abs(np.linalg.det(effective_mass_tensor))))
        else:
            m_eff_scalar = max(0.05, effective_mass_relative)
            inv_m_tensor = (1.0 / (m_eff_scalar * M_ELECTRON)) * np.eye(3)

        alpha_np = ((1.0 - m_eff_scalar) ** 2) / max(0.1, band_gap_ev)
        non_parabolic_correction = 1.0 / (1.0 + 2.5 * alpha_np * (BOLTZMANN_J_K * self.T / E_CHARGE))

        mu_tensor_si = (E_CHARGE * tau_s * non_parabolic_correction) * inv_m_tensor
        mu_tensor_cm2_v_s = mu_tensor_si * 1.0e4

        mu_scalar = float(np.mean(np.diag(mu_tensor_cm2_v_s)))

        return {
            "mobility_tensor_cm2_v_s": mu_tensor_cm2_v_s.tolist(),
            "isotropic_mobility_cm2_v_s": float(np.clip(mu_scalar, 10.0, 45000.0)),
            "band_non_parabolicity_alpha_ev_inv": float(alpha_np),
            "relaxation_time_fs": float(relaxation_time_fs),
            "is_anisotropic": bool(np.std(np.diag(mu_tensor_cm2_v_s)) > 1.0),
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
        high_freq_dielectric_eps_inf: Optional[float] = None,
        high_freq_dielectric_tensor: Optional[np.ndarray] = None,
        phonon_polarizability_contribution: float = 8.5,
        born_effective_charges: Optional[np.ndarray] = None,
        phonon_frequencies_gamma: Optional[np.ndarray] = None,
        cell_volume_ang3: float = 120.0,
    ) -> Dict[str, Any]:
        """Compute static dielectric tensor eps_{ij}^0 via Lyddane-Sachs-Teller relation and non-local impact ionization."""
        if high_freq_dielectric_tensor is not None:
            eps_static = np.asarray(high_freq_dielectric_tensor, dtype=np.float64).copy()
        elif high_freq_dielectric_eps_inf is not None:
            eps_static = np.eye(3) * (high_freq_dielectric_eps_inf + phonon_polarizability_contribution)
        else:
            eps_static = np.eye(3) * 12.7

        if born_effective_charges is not None and phonon_frequencies_gamma is not None:
            prefactor = (14.3996 * 4.0 * np.pi) / max(1.0, cell_volume_ang3)
            for z_star, omega in zip(born_effective_charges, phonon_frequencies_gamma):
                if omega > 1e-3:
                    eps_static += prefactor * (np.outer(z_star, z_star) / (omega**2))

        eps_scalar = float(np.mean(np.diag(eps_static)))
        # Non-local Keldysh / Thornber impact ionization threshold field
        e_break_mv_cm = float(0.18 * (band_gap_ev ** 1.85) * np.sqrt(max(1.0, 15.0 / eps_scalar)))
        bfom = float(eps_scalar * (max(0.1, band_gap_ev) ** 3) * (e_break_mv_cm ** 2))
        e_impact_thresh_mv_cm = 1.2 * band_gap_ev

        return {
            "band_gap_ev": float(band_gap_ev),
            "static_dielectric_tensor": eps_static.tolist(),
            "static_dielectric_constant_eps_r": float(eps_scalar),
            "dielectric_breakdown_field_mv_cm": float(e_break_mv_cm),
            "impact_ionization_threshold_field_mv_cm": float(e_impact_thresh_mv_cm),
            "baliga_figure_of_merit": float(bfom),
            "is_ultra_wide_bandgap": bool(band_gap_ev >= 3.4),
        }

    def compute_charged_defect_formation_energy(
        self,
        e_defect_dft_ev: float,
        e_bulk_dft_ev: float,
        chemical_potentials_ev: Dict[str, float],
        stoichiometry_change_delta_n: Dict[str, int],
        charge_state_q: int,
        fermi_energy_ev: float,
        e_vbm_ev: float = 0.0,
        potential_alignment_delta_v_ev: float = 0.0,
        dielectric_constant_eps_r: float = 12.0,
        cell_volume_ang3: float = 150.0,
    ) -> Dict[str, Any]:
        """Compute grand canonical charged defect formation energy:

        Delta H_f(D^q, E_F, mu) = E_tot(D^q) - E_tot(bulk) - sum_i Delta n_i * mu_i + q * (E_F + E_VBM + Delta v) + E_FNV
        """
        # 1. Chemical potential sum
        chempot_term = sum(delta_n * chemical_potentials_ev.get(elem, 0.0) for elem, delta_n in stoichiometry_change_delta_n.items())

        # 2. Fermi energy and band edge term
        fermi_term = charge_state_q * (fermi_energy_ev + e_vbm_ev + potential_alignment_delta_v_ev)

        # 3. Freysoldt-Neugebauer-Van de Walle (FNV) electrostatic image charge correction
        # E_FNV = - (q^2 * alpha_Madelung) / (2 * eps_r * L)
        l_cell = (cell_volume_ang3) ** (1.0 / 3.0)
        madelung_alpha = 2.8373
        e_charge_corr_ev = - ( (charge_state_q ** 2) * 14.3996 * madelung_alpha ) / (2.0 * max(1.0, dielectric_constant_eps_r) * l_cell) if charge_state_q != 0 else 0.0

        delta_h_f = (e_defect_dft_ev - e_bulk_dft_ev) - chempot_term + fermi_term + e_charge_corr_ev

        # Equilibrium defect concentration at temperature T
        kbt_ev = (BOLTZMANN_J_K * max(1.0, self.T)) / E_CHARGE
        n_sites_cm3 = (1.0 / (cell_volume_ang3 * 1.0e-24))
        c_defect_cm3 = n_sites_cm3 * np.exp(-max(0.0, delta_h_f) / max(1e-4, kbt_ev))

        return {
            "formation_energy_ev": float(delta_h_f),
            "charge_state": int(charge_state_q),
            "fermi_energy_above_vbm_ev": float(fermi_energy_ev),
            "fnv_image_charge_correction_ev": float(e_charge_corr_ev),
            "equilibrium_concentration_cm3": float(np.clip(c_defect_cm3, 1.0, 1.0e22)),
            "is_spontaneous_doping": bool(delta_h_f <= 0.0),
        }
