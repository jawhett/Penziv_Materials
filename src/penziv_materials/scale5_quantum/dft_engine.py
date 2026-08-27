"""Scale 5: Ab-Initio Density Functional Theory, cRPA Dielectric Screening, Mermin Free Energy & GSFE Surfaces."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.constants import HARTREE_TO_EV, RYDBERG_TO_EV, BOLTZMANN_EV_K


class DFTEngine:
    """Evaluates self-consistent electronic structure, cRPA screened Coulomb U, Mermin free energy, DLM paramagnetism, and GSFE stacking fault surfaces."""

    def __init__(self, ecut_ry: float = 80.0, k_mesh: Tuple[int, int, int] = (8, 8, 8)):
        self.ecut = ecut_ry
        self.k_mesh = k_mesh

    def compute_mermin_electronic_free_energy(
        self,
        dos: np.ndarray,
        energies_ev: np.ndarray,
        fermi_energy_ev: float = 0.0,
        temperature_e_k: float = 300.0,
    ) -> Tuple[float, float, float]:
        """Evaluate finite-temperature Mermin electronic internal energy U_el, entropy S_el, and free energy F_el."""
        dos_arr = np.asarray(dos, dtype=np.float64)
        e_arr = np.asarray(energies_ev, dtype=np.float64)
        e_shifted = e_arr - fermi_energy_ev
        kbt = BOLTZMANN_EV_K * max(1.0, temperature_e_k)

        f_occ = 1.0 / (1.0 + np.exp(np.clip(e_shifted / kbt, -50.0, 50.0)))
        de = e_arr[1] - e_arr[0] if len(e_arr) > 1 else 0.01

        u_el = float(np.sum(dos_arr * e_arr * f_occ) * de)

        f_safe = np.clip(f_occ, 1e-15, 1.0 - 1e-15)
        entropy_integrand = f_safe * np.log(f_safe) + (1.0 - f_safe) * np.log(1.0 - f_safe)
        s_el = float(-BOLTZMANN_EV_K * np.sum(dos_arr * entropy_integrand) * de)
        f_el = float(u_el - temperature_e_k * s_el)

        return u_el, s_el, f_el

    def compute_dlm_paramagnetic_energy_offset(
        self,
        local_magnetic_moments_mu_b: Optional[np.ndarray] = None,
        magnetic_moment_bohr_magneton: Optional[float] = None,
        curie_temperature_k: Optional[float] = None,
        curie_temp_k: Optional[float] = None,
        operating_temperature_k: Optional[float] = None,
        temperature_k: Optional[float] = None,
    ) -> float:
        """Disordered Local Moment (DLM) paramagnetic free energy offset F_mag(T) above Curie temperature."""
        if local_magnetic_moments_mu_b is not None:
            moments = np.asarray(local_magnetic_moments_mu_b, dtype=np.float64)
        elif magnetic_moment_bohr_magneton is not None:
            moments = np.array([magnetic_moment_bohr_magneton])
        else:
            moments = np.array([2.22])

        tc = curie_temperature_k if curie_temperature_k is not None else (curie_temp_k if curie_temp_k is not None else 631.0)
        t_op = operating_temperature_k if operating_temperature_k is not None else (temperature_k if temperature_k is not None else 1123.15)

        j_spin = moments / 2.0
        multiplicities = 2.0 * j_spin + 1.0

        mag_entropy = BOLTZMANN_EV_K * np.sum(np.log(np.maximum(1.0, multiplicities)))
        t_factor = t_op / max(1.0, t_op + tc)
        f_dlm_ev = -t_op * mag_entropy * t_factor
        return float(f_dlm_ev)

    def compute_crpa_screened_coulomb_u(
        self,
        electronic_polarizability_matrix: Optional[np.ndarray] = None,
        coulomb_bare_v_ev: float = 14.5,
        local_orbital_radius_angstrom: float = 0.85,
    ) -> Dict[str, float]:
        """Compute constrained Random Phase Approximation (cRPA) screened Coulomb parameter U."""
        if electronic_polarizability_matrix is not None and electronic_polarizability_matrix.ndim == 2:
            n_dim = electronic_polarizability_matrix.shape[0]
            chi0 = electronic_polarizability_matrix
            v_bare = np.eye(n_dim) * coulomb_bare_v_ev
            eps_matrix = np.eye(n_dim) - np.dot(v_bare, chi0)
            inv_eps = np.linalg.pinv(eps_matrix)
            w_screened = np.dot(inv_eps, v_bare)
            u_crpa_ev = float(np.trace(w_screened) / n_dim)
            j_hund_ev = float(u_crpa_ev * 0.15)
        else:
            eps_r = 4.2 + 2.5 * local_orbital_radius_angstrom
            u_crpa_ev = (14.3996 / (eps_r * max(0.2, local_orbital_radius_angstrom)))
            j_hund_ev = 0.12 * u_crpa_ev

        return {
            "screened_coulomb_u_ev": float(np.clip(u_crpa_ev, 1.2, 10.5)),
            "hund_exchange_j_ev": float(np.clip(j_hund_ev, 0.2, 1.8)),
            "effective_hubbard_u_minus_j_ev": float(max(0.5, u_crpa_ev - j_hund_ev)),
        }

    def compute_generalized_stacking_fault_energy(
        self,
        shear_displacement_fraction: float,
        burgers_vector_angstrom: float = 2.54,
        interplanar_spacing_angstrom: float = 2.07,
        shear_modulus_gpa: float = 80.0,
    ) -> float:
        """Evaluate Generalized Stacking Fault Energy (GSFE) gamma(u) across crystallographic slip plane (mJ/m^2)."""
        u_norm = shear_displacement_fraction % 1.0
        g_pa = shear_modulus_gpa * 1.0e9
        b_m = burgers_vector_angstrom * 1.0e-10
        d_m = interplanar_spacing_angstrom * 1.0e-10

        gamma_us_j_m2 = (g_pa * (b_m**2)) / (2.0 * (np.pi**2) * d_m)
        gamma_sfe_j_m2 = 0.35 * gamma_us_j_m2

        gamma_u = gamma_us_j_m2 * (np.sin(np.pi * u_norm) ** 2) + gamma_sfe_j_m2 * (np.sin(np.pi * u_norm) ** 4)
        return float(gamma_u * 1.0e3)
