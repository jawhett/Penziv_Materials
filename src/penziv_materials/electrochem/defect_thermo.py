"""Charged Point Defect Thermodynamics, Fermi-Level Pinning & Electronic Leakage Engine."""

import math
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.constants import KB_EV, ELEMENTARY_CHARGE_C, EV_TO_JOULE


class ChargedDefectThermoEngine:
    """Calculates defect formation energies Delta H_f(D,q) with FNV image-charge corrections and electronic leakage risk."""

    def __init__(self, band_gap_ev: float = 4.5, dielectric_constant_epsilon_r: float = 14.0):
        self.eg = band_gap_ev
        self.eps_r = dielectric_constant_epsilon_r

    def compute_fnv_image_charge_correction(
        self,
        defect_charge_q: int,
        supercell_volume_ang3: float = 1200.0,
    ) -> float:
        """Freysoldt-Neugebauer-Van de Walle (FNV) electrostatic image-charge finite-size correction (eV):

        E_corr = (q^2 * alpha_M) / (2 * eps_r * eps_0 * L)
        """
        if defect_charge_q == 0:
            return 0.0

        # Madelung constant for cubic cell alpha_M ~ 2.8373
        madelung_alpha = 2.8373
        l_eff_ang = supercell_volume_ang3 ** (1.0 / 3.0)
        l_eff_m = l_eff_ang * 1.0e-10

        eps_0 = 8.8541878128e-12
        e_c = ELEMENTARY_CHARGE_C

        # Energy in Joules
        e_corr_j = ((defect_charge_q * e_c) ** 2 * madelung_alpha) / (2.0 * self.eps_r * eps_0 * l_eff_m)
        e_corr_ev = e_corr_j / EV_TO_JOULE
        return float(e_corr_ev)

    def compute_defect_formation_energy(
        self,
        e_defect_dft_ev: float,
        e_pristine_dft_ev: float,
        defect_charge_q: int,
        fermi_level_ef_ev: float,  # Relative to Valence Band Maximum (VBM = 0)
        chemical_potential_deltas: Dict[str, float],
        stoichiometry_deltas: Dict[str, int],  # e.g., {"Mg": -1} for vacancy
        supercell_volume_ang3: float = 1200.0,
    ) -> Dict[str, float]:
        """Compute defect formation energy Delta H_f(D, q, E_F, mu):

        Delta H_f = E_tot(D, q) - E_tot(bulk) + sum_i (Delta n_i * mu_i) + q * (E_VBM + E_F) + E_FNV_corr
        """
        e_fnv = self.compute_fnv_image_charge_correction(defect_charge_q, supercell_volume_ang3)

        # Chemical potential sum
        chempot_term = 0.0
        for elem, dn in stoichiometry_deltas.items():
            # If atom removed (vacancy), dn = -1, we add +mu to energy
            chempot_term -= dn * chemical_potential_deltas.get(elem, 0.0)

        # Fermi level contribution relative to VBM
        ef_clamped = np.clip(fermi_level_ef_ev, 0.0, self.eg)
        fermi_term = defect_charge_q * ef_clamped

        delta_h_f = (e_defect_dft_ev - e_pristine_dft_ev) + chempot_term + fermi_term + e_fnv

        return {
            "defect_formation_energy_ev": float(delta_h_f),
            "fnv_correction_ev": float(e_fnv),
            "charge_q": defect_charge_q,
            "fermi_level_ev": float(ef_clamped),
        }

    def evaluate_electronic_leakage_and_dendrite_risk(
        self,
        conduction_band_min_vs_metal_redox_v: float,  # VBM/CBM vs Mg/Mg2+ or Na/Na+
        trap_state_depth_ev: float = 0.8,
        temperature_k: float = 300.0,
    ) -> Dict[str, Any]:
        """Evaluate whether electronic carrier density allows internal dendrite reduction (sigma_e > 1e-10 S/cm)."""
        # Thermal activation of electronic trap states
        kbt = KB_EV * temperature_k
        n_trap = np.exp(-trap_state_depth_ev / kbt)
        sigma_e_s_cm = 1.0e-3 * n_trap  # Estimated electronic conductivity

        # If CBM is below or near redox potential (0 V), reduction occurs inside pore walls
        internal_reduction_risk = bool(conduction_band_min_vs_metal_redox_v < 0.20 or sigma_e_s_cm > 1.0e-10)

        # Critical current density J_crit (mA/cm2) scaling before dendrite short
        j_crit_ma_cm2 = 1.5 if not internal_reduction_risk else 0.05

        return {
            "estimated_electronic_conductivity_s_cm": float(sigma_e_s_cm),
            "internal_pore_reduction_risk": internal_reduction_risk,
            "critical_current_density_j_crit_ma_cm2": float(j_crit_ma_cm2),
            "is_electronically_insulating": bool(sigma_e_s_cm < 1.0e-10),
        }


class ChargedDefectThermodynamicsEngine:
    """Calculates defect equilibrium, transition levels, and carrier pinning without empirical approximations."""

    @staticmethod
    def compute_defect_formation_energy(
        e_defect_dft_ev: float,
        e_bulk_dft_ev: float,
        chemical_potentials: Dict[str, float],
        stoichiometric_change: Dict[str, int],
        charge_q: int,
        fermi_level_ev: float,
        vbm_energy_ev: float = 0.0,
        dielectric_constant: float = 14.0,
        unit_cell_volume_ang3: float = 1200.0,
        potential_alignment_v: float = 0.0,
    ) -> float:
        """Calculate defect formation energy Delta H_f(X^q, E_F, mu_i) with FNV image charge correction."""
        chempot_sum = sum(stoichiometric_change[elem] * chemical_potentials.get(elem, 0.0) for elem in stoichiometric_change)
        L = unit_cell_volume_ang3 ** (1.0 / 3.0)
        madelung_constant = 2.8373
        e_corr_fnv = (charge_q**2 * 14.3996 * madelung_constant) / (2.0 * max(1.0, dielectric_constant) * max(1.0, L))
        delta_h_f = (
            (e_defect_dft_ev - e_bulk_dft_ev)
            - chempot_sum
            + charge_q * (fermi_level_ev + vbm_energy_ev + potential_alignment_v)
            + e_corr_fnv
        )
        return float(delta_h_f)

    @classmethod
    def compute_charge_transition_levels(
        cls,
        e_defect_by_charge: Dict[int, float],
        e_bulk: float,
        chemical_potentials: Dict[str, float],
        stoich_change: Dict[str, int],
        bandgap_ev: float,
        vbm_ev: float = 0.0,
        dielectric_constant: float = 14.0,
        volume_ang3: float = 1200.0,
    ) -> List[Dict[str, Any]]:
        """Identify ionization energy crossing points (epsilon(q1/q2)) across the band gap."""
        charges = sorted(list(e_defect_by_charge.keys()))
        transition_levels = []

        for i in range(len(charges) - 1):
            q1, q2 = charges[i], charges[i + 1]
            dq = q1 - q2
            if dq == 0:
                continue

            h1_0 = cls.compute_defect_formation_energy(
                e_defect_by_charge[q1], e_bulk, chemical_potentials, stoich_change, q1, 0.0, vbm_ev, dielectric_constant, volume_ang3
            )
            h2_0 = cls.compute_defect_formation_energy(
                e_defect_by_charge[q2], e_bulk, chemical_potentials, stoich_change, q2, 0.0, vbm_ev, dielectric_constant, volume_ang3
            )

            ef_trans = (h1_0 - h2_0) / float(q2 - q1)
            if 0.0 <= ef_trans <= bandgap_ev:
                transition_levels.append({
                    "transition": f"epsilon({q1:+d}/{q2:+d})",
                    "fermi_level_above_vbm_ev": float(ef_trans),
                    "is_deep_trap": bool(0.2 * bandgap_ev < ef_trans < 0.8 * bandgap_ev),
                })
        return transition_levels

