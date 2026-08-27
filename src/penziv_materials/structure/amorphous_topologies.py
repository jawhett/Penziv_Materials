"""Amorphous Structures, Spatial Point Process Descriptors (RDF/Voronoi), STZ Plasticity & VRH Transport."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.constants import BOLTZMANN_J_K, BOLTZMANN_EV_K


class AmorphousTopologyEngine:
    """Evaluates disordered atomic networks, Radial Distribution Functions g(r), STZ glass plasticity, and VRH transport."""

    def __init__(self, temperature_k: float = 300.0):
        self.T = temperature_k

    def compute_radial_distribution_function(
        self,
        positions_angstrom: np.ndarray,
        box_length_angstrom: float = 25.0,
        dr: float = 0.05,
        r_max: float = 10.0,
    ) -> Dict[str, Any]:
        """Compute Radial Distribution Function g(r) for amorphous/disordered systems."""
        pos = np.asarray(positions_angstrom, dtype=np.float64)
        n_atoms = len(pos)
        if n_atoms < 2:
            return {"r_bins": [], "g_r": [], "coordination_number_first_shell": 0.0}

        # Pairwise distance calculation with minimum image convention
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        diff -= box_length_angstrom * np.round(diff / box_length_angstrom)
        distances = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(distances, np.inf)

        r_bins = np.arange(dr, r_max + dr, dr)
        hist, _ = np.histogram(distances, bins=np.append(0.0, r_bins))

        # Normalization by ideal gas spherical shell volume
        volume = box_length_angstrom**3
        number_density = n_atoms / volume
        shell_volumes = (4.0 / 3.0) * np.pi * (r_bins**3 - (r_bins - dr) ** 3)
        ideal_counts = number_density * shell_volumes * n_atoms

        g_r = hist / np.maximum(1e-6, ideal_counts)

        # First coordination shell peak
        peak_idx = int(np.argmax(g_r[: int(4.0 / dr)]))
        first_peak_r = float(r_bins[peak_idx])

        # Integrate first coordination shell
        first_min_idx = peak_idx + int(np.argmin(g_r[peak_idx : int(5.0 / dr)]))
        cn_first = float(np.sum(hist[:first_min_idx]) / n_atoms)

        return {
            "r_bins_angstrom": r_bins.tolist(),
            "g_r": g_r.tolist(),
            "first_neighbor_distance_angstrom": first_peak_r,
            "coordination_number_first_shell": cn_first,
        }

    def compute_voronoi_polyhedral_indices(
        self,
        average_coordination: float = 12.0,
        fraction_icosahedral_order: float = 0.22,
    ) -> Dict[str, Any]:
        """Compute Voronoi polyhedral signature <n3, n4, n5, n6> characterizing short-to-medium range order in metallic glasses/liquids."""
        f_ico = np.clip(fraction_icosahedral_order, 0.0, 1.0)
        # Full icosahedra <0, 0, 12, 0> vs distorted coordination
        n3 = float(0.1 * (1.0 - f_ico))
        n4 = float(2.0 * (1.0 - f_ico))
        n5 = float(12.0 * f_ico + 8.0 * (1.0 - f_ico))
        n6 = float(2.0 * (1.0 - f_ico))

        return {
            "voronoi_index_signature": [n3, n4, n5, n6],
            "fraction_icosahedral_order": float(f_ico),
            "is_amorphous_glass_forming": bool(f_ico >= 0.15),
        }

    def compute_shear_transformation_zone_plasticity(
        self,
        applied_shear_stress_mpa: float,
        stz_activation_volume_ang3: float = 120.0,
        stz_free_energy_barrier_ev: float = 1.65,
        reference_shear_rate_s_inv: float = 1.0e11,
    ) -> Dict[str, float]:
        """Argon-Bulatov Shear Transformation Zone (STZ) constitutive model for amorphous yield:

        gamma_dot_STZ = gamma_dot_0 * exp(-(Delta F - tau * Omega_STZ) / (k_B * T))
        """
        tau_pa = applied_shear_stress_mpa * 1.0e6
        omega_m3 = stz_activation_volume_ang3 * 1.0e-30
        work_j = tau_pa * omega_m3
        work_ev = work_j / 1.602176634e-19

        effective_barrier_ev = max(0.05, stz_free_energy_barrier_ev - work_ev)
        k_b_t_ev = BOLTZMANN_EV_K * max(1.0, self.T)

        gamma_dot = reference_shear_rate_s_inv * np.exp(-effective_barrier_ev / k_b_t_ev)

        # Critical STZ yield stress tau_y
        tau_yield_mpa = (stz_free_energy_barrier_ev * 1.602176634e-19) / (omega_m3 * 1.0e6)

        return {
            "stz_plastic_shear_rate_s_inv": float(np.clip(gamma_dot, 1e-15, 1e9)),
            "stz_effective_barrier_ev": float(effective_barrier_ev),
            "amorphous_yield_stress_mpa": float(np.clip(tau_yield_mpa * 0.75, 50.0, 3500.0)),
        }

    def compute_variable_range_hopping_transport(
        self,
        localization_length_angstrom: float = 8.5,
        density_of_states_at_ef_ev_cm3: float = 1.2e20,
        regime: str = "Mott",
    ) -> Dict[str, float]:
        """Mott (p=1/4) or Efros-Shklovskii (p=1/2) Variable Range Hopping in disordered insulators/semiconductors:

        sigma(T) = sigma_0 * exp(-(T_0 / T)^p)
        """
        loc_len_cm = localization_length_angstrom * 1.0e-8

        if "mott" in regime.lower():
            p_exp = 0.25
            # T_0 = 18 / (k_B * N(E_F) * xi^3)
            t_0 = 18.0 / (BOLTZMANN_EV_K * density_of_states_at_ef_ev_cm3 * (loc_len_cm**3))
        else:
            p_exp = 0.50
            # Efros-Shklovskii Coulomb gap VRH
            t_0 = 2.8e4

        exponent = (t_0 / max(1.0, self.T)) ** p_exp
        sigma_vrh_s_cm = 1.0e2 * np.exp(-min(100.0, exponent))

        return {
            "vrh_conductivity_s_cm": float(sigma_vrh_s_cm),
            "characteristic_temperature_t0_k": float(t_0),
            "hopping_distance_nm": float((0.375 * localization_length_angstrom * ((t_0 / max(1.0, self.T)) ** 0.25)) * 0.1),
            "vrh_exponent_p": float(p_exp),
        }
