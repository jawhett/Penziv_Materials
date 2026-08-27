"""Solid-State Ion Migration Kinetics & Diffusivity Engine (CI-NEB, AIMD, Nernst-Einstein)."""

import math
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.constants import KB_EV, EV_TO_JOULE, BOLTZMANN_J_K, ELEMENTARY_CHARGE_C, AVOGADRO_N_A


class SolidStateIonTransportEngine:
    """Calculates ion hopping barriers, diffusion coefficients, and ionic conductivity for monovalent (Li+, Na+) and multivalent (Mg2+, Zn2+, Ca2+) cations."""

    def __init__(self, mobile_ion_charge_z: int = 2, ionic_radius_angstrom: float = 0.72):
        self.charge_z = mobile_ion_charge_z
        self.r_ion = ionic_radius_angstrom

    def compute_ci_neb_migration_barrier(
        self,
        energy_path_ev: np.ndarray,
        reaction_coordinate: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Extract activation energy barrier Delta E_a and transition state location along CI-NEB minimum energy path (MEP)."""
        e_path = np.asarray(energy_path_ev, dtype=np.float64)
        if len(e_path) < 3:
            return {"activation_energy_ev": 0.0, "reaction_energy_ev": 0.0, "ts_index": 0}

        e_initial = e_path[0]
        e_final = e_path[-1]
        ts_index = int(np.argmax(e_path))
        e_ts = e_path[ts_index]

        delta_e_forward = float(e_ts - e_initial)
        delta_e_backward = float(e_ts - e_final)
        delta_e_reaction = float(e_final - e_initial)

        return {
            "activation_energy_ev": max(0.0, delta_e_forward),
            "reverse_barrier_ev": max(0.0, delta_e_backward),
            "reaction_energy_ev": delta_e_reaction,
            "transition_state_index": ts_index,
        }

    def compute_multivalent_polarization_penalty(
        self,
        anion_polarizability_ang3: float = 3.88,  # S2- is ~3.88 Å³, Se2- ~4.5 Å³, O2- ~2.0 Å³
        dielectric_constant_epsilon_r: float = 12.5,
    ) -> float:
        """Evaluate multivalent electrostatic trapping penalty Delta E_pol for high charge-density cations (Mg2+, Ca2+):

        Delta E_pol ~ (z * e)^2 / (8 * pi * eps_0 * eps_r * r_ion) * (1 - 1/eps_r)
        """
        eps_0 = 8.8541878128e-12
        r_m = self.r_ion * 1.0e-10
        q = self.charge_z * ELEMENTARY_CHARGE_C

        # Born solvation/polarization trapping energy in dielectric cage (eV)
        e_pol_joules = (q**2 / (8.0 * math.pi * eps_0 * dielectric_constant_epsilon_r * r_m)) * (
            1.0 - 1.0 / max(1.0, dielectric_constant_epsilon_r)
        )
        e_pol_ev = e_pol_joules / EV_TO_JOULE

        # Highly polarizable anion sublattices (like S2-, Se2-) screen multivalent charge and lower the barrier
        polarization_screening_factor = 1.0 / (1.0 + 0.15 * anion_polarizability_ang3)
        effective_penalty_ev = e_pol_ev * polarization_screening_factor * 0.05
        return float(effective_penalty_ev)

    def compute_msd_and_diffusivity_aimd(
        self,
        trajectory_positions_angstrom: np.ndarray,  # Shape: (n_timesteps, n_ions, 3)
        timestep_fs: float = 2.0,
    ) -> Tuple[float, float, np.ndarray]:
        """Evaluate 3D Mean-Squared Displacement (MSD) and Einstein diffusion coefficient D_ion (cm²/s):

        MSD(t) = < |r_i(t) - r_i(0)|^2 >
        D_ion = lim_{t -> inf} MSD(t) / (6 * t)
        """
        n_steps, n_ions, _ = trajectory_positions_angstrom.shape
        r0 = trajectory_positions_angstrom[0]  # (n_ions, 3)

        msd = np.zeros(n_steps, dtype=np.float64)
        for t in range(n_steps):
            disp = trajectory_positions_angstrom[t] - r0
            sq_disp = np.sum(disp**2, axis=-1)  # (n_ions,)
            msd[t] = np.mean(sq_disp)  # Å²

        # Linear regression on last 60% of trajectory
        fit_start = int(0.4 * n_steps)
        time_ps = np.arange(n_steps) * (timestep_fs / 1000.0)  # ps

        if n_steps > 5:
            slope, _ = np.polyfit(time_ps[fit_start:], msd[fit_start:], 1)  # Å²/ps
            # Convert Å²/ps to cm²/s: 1 Å²/ps = 1e-16 cm² / 1e-12 s = 1e-4 cm²/s
            d_ion_cm2_s = (slope / 6.0) * 1.0e-4
        else:
            d_ion_cm2_s = 0.0

        return max(0.0, float(d_ion_cm2_s)), float(msd[-1]), msd

    def compute_nernst_einstein_ionic_conductivity(
        self,
        diffusivity_cm2_s: float,
        carrier_concentration_cm3: float,
        temperature_k: float,
        haven_ratio: float = 1.0,
        electronic_diffusivity_cm2_s: float = 1.0e-14,
    ) -> Dict[str, float]:
        """Compute ionic conductivity sigma_ion (mS/cm), electronic conductivity sigma_e, and transference number t_ion:

        sigma_ion = (n_ion * (z * e)^2 * D_ion) / (H_R * k_B * T)
        t_ion = sigma_ion / (sigma_ion + sigma_e)
        """
        q = self.charge_z * ELEMENTARY_CHARGE_C
        kbt_joules = BOLTZMANN_J_K * max(1.0, temperature_k)

        # D in cm²/s -> m²/s
        d_m2_s = diffusivity_cm2_s * 1.0e-4
        n_m3 = carrier_concentration_cm3 * 1.0e6

        sigma_ion_s_m = (n_m3 * (q**2) * d_m2_s) / (haven_ratio * kbt_joules)
        sigma_ion_ms_cm = sigma_ion_s_m * 10.0  # S/m to mS/cm

        d_e_m2_s = electronic_diffusivity_cm2_s * 1.0e-4
        sigma_e_s_m = (n_m3 * (ELEMENTARY_CHARGE_C**2) * d_e_m2_s) / kbt_joules
        sigma_e_ms_cm = sigma_e_s_m * 10.0

        total_sigma = sigma_ion_ms_cm + sigma_e_ms_cm
        transference_number = sigma_ion_ms_cm / max(1e-12, total_sigma)

        return {
            "ionic_conductivity_ms_cm": float(sigma_ion_ms_cm),
            "ionic_conductivity_s_cm": float(sigma_ion_ms_cm * 1e-3),
            "electronic_conductivity_s_cm": float(sigma_e_ms_cm * 1e-3),
            "transference_number_t_ion": float(transference_number),
            "is_superionic_conductor": bool(sigma_ion_ms_cm >= 1.0 and transference_number >= 0.99),
        }
