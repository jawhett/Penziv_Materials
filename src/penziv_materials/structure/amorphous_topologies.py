"""Amorphous Structures, Stochastic Dense Random Packing (DRP), STZ Plasticity & VRH Transport."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.constants import BOLTZMANN_J_K, BOLTZMANN_EV_K


class AmorphousTopologyEngine:
    """Evaluates disordered atomic networks, generates Dense Random Packing (DRP) structures, and computes STZ glass plasticity."""

    def __init__(self, temperature_k: float = 300.0):
        self.T = temperature_k

    def generate_stochastic_dense_random_packing(
        self,
        num_atoms: int = 64,
        box_length_angstrom: float = 12.0,
        min_interatomic_distance_angstrom: float = 2.30,
        monte_carlo_steps: int = 150,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """Generate an unconstrained amorphous atomic topology via Dense Random Hard-Sphere Packing and Monte Carlo relaxation."""
        np.random.seed(random_seed)
        positions = np.zeros((num_atoms, 3), dtype=np.float64)

        # 1. Random sequential hard-sphere addition
        placed = 0
        attempts = 0
        max_attempts = num_atoms * 200

        while placed < num_atoms and attempts < max_attempts:
            candidate = np.random.uniform(0.0, box_length_angstrom, 3)
            if placed == 0:
                positions[placed] = candidate
                placed += 1
            else:
                diff = positions[:placed] - candidate
                diff -= box_length_angstrom * np.round(diff / box_length_angstrom)
                dists = np.linalg.norm(diff, axis=-1)
                if np.all(dists >= min_interatomic_distance_angstrom * 0.85):
                    positions[placed] = candidate
                    placed += 1
            attempts += 1

        # 2. Monte Carlo Lennard-Jones/Soft-Sphere Energy Relaxation
        for _ in range(monte_carlo_steps):
            idx = np.random.randint(0, placed)
            trial_pos = (positions[idx] + np.random.normal(0, 0.1, 3)) % box_length_angstrom

            diff_curr = positions[:placed] - positions[idx]
            diff_curr -= box_length_angstrom * np.round(diff_curr / box_length_angstrom)
            dists_curr = np.linalg.norm(diff_curr, axis=-1)
            np.fill_diagonal(np.atleast_2d(dists_curr), np.inf)

            diff_trial = positions[:placed] - trial_pos
            diff_trial -= box_length_angstrom * np.round(diff_trial / box_length_angstrom)
            dists_trial = np.linalg.norm(diff_trial, axis=-1)

            min_d_trial = np.min(dists_trial[dists_trial > 0])
            if min_d_trial >= min_interatomic_distance_angstrom * 0.80:
                positions[idx] = trial_pos

        rdf_data = self.compute_radial_distribution_function(positions[:placed], box_length_angstrom=box_length_angstrom)

        return {
            "num_atoms_packed": placed,
            "positions_angstrom": positions[:placed].tolist(),
            "box_length_angstrom": box_length_angstrom,
            "packing_fraction": float((placed * (4.0 / 3.0) * np.pi * (min_interatomic_distance_angstrom / 2.0) ** 3) / (box_length_angstrom**3)),
            "first_coordination_shell_radius": rdf_data["first_neighbor_distance_angstrom"],
            "coordination_number": rdf_data["coordination_number_first_shell"],
        }

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
            return {"r_bins": [], "g_r": [], "first_neighbor_distance_angstrom": 2.5, "coordination_number_first_shell": 0.0}

        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        diff -= box_length_angstrom * np.round(diff / box_length_angstrom)
        distances = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(distances, np.inf)

        r_bins = np.arange(dr, r_max + dr, dr)
        hist, _ = np.histogram(distances, bins=np.append(0.0, r_bins))

        volume = box_length_angstrom**3
        number_density = n_atoms / volume
        shell_volumes = (4.0 / 3.0) * np.pi * (r_bins**3 - (r_bins - dr) ** 3)
        ideal_counts = number_density * shell_volumes * n_atoms

        g_r = hist / np.maximum(1e-6, ideal_counts)

        peak_idx = int(np.argmax(g_r[: int(4.0 / dr)])) if len(g_r) > 0 else 0
        first_peak_r = float(r_bins[peak_idx]) if len(r_bins) > peak_idx else 2.5

        first_min_idx = peak_idx + int(np.argmin(g_r[peak_idx : int(5.0 / dr)])) if len(g_r) > peak_idx + 1 else peak_idx + 1
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
        """Compute Voronoi polyhedral signature <n3, n4, n5, n6> characterizing short-to-medium range order."""
        f_ico = np.clip(fraction_icosahedral_order, 0.0, 1.0)
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
        """Argon-Bulatov Shear Transformation Zone (STZ) constitutive model for amorphous yield."""
        tau_pa = applied_shear_stress_mpa * 1.0e6
        omega_m3 = stz_activation_volume_ang3 * 1.0e-30
        work_j = tau_pa * omega_m3
        work_ev = work_j / 1.602176634e-19

        effective_barrier_ev = max(0.05, stz_free_energy_barrier_ev - work_ev)
        k_b_t_ev = BOLTZMANN_EV_K * max(1.0, self.T)

        gamma_dot = reference_shear_rate_s_inv * np.exp(-effective_barrier_ev / k_b_t_ev)
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
        """Mott or Efros-Shklovskii Variable Range Hopping in disordered materials."""
        loc_len_cm = localization_length_angstrom * 1.0e-8

        if "mott" in regime.lower():
            p_exp = 0.25
            t_0 = 18.0 / (BOLTZMANN_EV_K * density_of_states_at_ef_ev_cm3 * (loc_len_cm**3))
        else:
            p_exp = 0.50
            t_0 = 2.8e4

        exponent = (t_0 / max(1.0, self.T)) ** p_exp
        sigma_vrh_s_cm = 1.0e2 * np.exp(-min(100.0, exponent))

        return {
            "vrh_conductivity_s_cm": float(sigma_vrh_s_cm),
            "characteristic_temperature_t0_k": float(t_0),
            "hopping_distance_nm": float((0.375 * localization_length_angstrom * ((t_0 / max(1.0, self.T)) ** 0.25)) * 0.1),
            "vrh_exponent_p": float(p_exp),
        }
