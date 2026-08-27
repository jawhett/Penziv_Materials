"""Space & Nuclear Radiation Physics: NRT DPA, Directional E_d(theta, phi) & PKA Defect Cascades."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np


class RadiationDamageEngine:
    """Evaluates radiation damage in materials exposed to fast neutrons, protons, and cosmic ions."""

    def __init__(self, ambient_temperature_k: float = 300.0):
        self.T = ambient_temperature_k

    def compute_directional_displacement_energy_surface(
        self,
        theta_rad: float,
        phi_rad: float,
        e_d_min_ev: float = 24.0,
        e_d_max_ev: float = 85.0,
    ) -> float:
        """Compute directional threshold displacement energy E_d(theta, phi) reflecting crystallographic anisotropy."""
        # Spherical harmonic angular variation along close-packed vs open channels
        harmonics = np.sin(theta_rad)**4 * np.cos(4.0 * phi_rad) + np.cos(theta_rad)**4
        e_d = e_d_min_ev + (e_d_max_ev - e_d_min_ev) * (0.5 * (1.0 + harmonics))
        return float(np.clip(e_d, e_d_min_ev, e_d_max_ev))

    def compute_nrt_displacements_per_atom(
        self,
        damage_energy_t_dam_kev: float,
        threshold_displacement_energy_e_d_ev: float = 40.0,
        ion_fluence_ions_cm2: float = 1.0e16,
        atomic_density_atoms_cm3: float = 8.5e22,
    ) -> Dict[str, Any]:
        """Compute Frenkel pair defects via the Norgett-Robinson-Torrens (NRT) standard dpa model:

        N_NRT = 0.8 * T_dam / (2 * E_d)
        """
        e_dam_ev = damage_energy_t_dam_kev * 1.0e3
        e_d_ev = max(10.0, threshold_displacement_energy_e_d_ev)

        if e_dam_ev < e_d_ev:
            n_displacements = 0.0
        elif e_dam_ev < (2.0 * e_d_ev / 0.8):
            n_displacements = 1.0
        else:
            n_displacements = (0.8 * e_dam_ev) / (2.0 * e_d_ev)

        # In-cascade defect survival fraction (A-PBM arc-dpa model)
        # eta_survival = (1 - c_rec) * (T_dam / T_crit)^-m
        eta_survival = float(np.clip(0.35 * (max(1.0, damage_energy_t_dam_kev) ** -0.22), 0.05, 0.95))
        surviving_frenkel_pairs = float(n_displacements * eta_survival)

        dpa_total = float((n_displacements * ion_fluence_ions_cm2) / atomic_density_atoms_cm3)

        return {
            "nrt_frenkel_pairs_per_pka": float(n_displacements),
            "in_cascade_survival_fraction": eta_survival,
            "surviving_defects_per_pka": surviving_frenkel_pairs,
            "total_displacements_per_atom_dpa": float(dpa_total),
            "is_radiation_tolerant": bool(eta_survival < 0.25 and e_d_ev >= 40.0),
        }
