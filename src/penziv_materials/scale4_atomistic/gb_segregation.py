"""Langmuir-McLean Grain Boundary Solute Segregation & Rice-Wang Interfacial Embrittlement Engine."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from penziv_materials.core.constants import R_GAS, BOLTZMANN_J_K, AVOGADRO_NUMBER


class GrainBoundarySegregationEngine:
    """Solves multi-component equilibrium grain boundary solute segregation and evaluates Rice-Wang interfacial embrittlement."""

    def __init__(self, temperature_k: float = 300.0):
        self.T = max(1.0, float(temperature_k))

    def compute_elastic_strain_segregation_enthalpy(
        self,
        matrix_shear_modulus_gpa: float,
        solute_bulk_modulus_gpa: float,
        matrix_covalent_radius_ang: float,
        solute_covalent_radius_ang: float,
    ) -> float:
        """Evaluate elastic strain energy driving solute segregation via Friedel-Eshelby misfit inclusion model:

        delta_H_elastic in J/mol
        """
        mu_m = matrix_shear_modulus_gpa * 1.0e9
        k_s = solute_bulk_modulus_gpa * 1.0e9
        r_m = matrix_covalent_radius_ang * 1.0e-10
        r_s = solute_covalent_radius_ang * 1.0e-10

        dr = r_s - r_m
        if abs(dr) < 1e-15:
            return 0.0

        # Elastic strain energy per atom: E_strain = (24 * pi * K_s * mu_m * r_m * r_s * dr^2) / (3 K_s r_s + 4 mu_m r_m)
        denom = 3.0 * k_s * r_s + 4.0 * mu_m * r_m
        e_strain_per_atom = (24.0 * np.pi * k_s * mu_m * r_m * r_s * (dr**2)) / max(1e-10, denom)

        # Enthalpy of segregation is negative of strain release at open GB volume
        delta_h_seg_j_mol = -1.0 * float(e_strain_per_atom * AVOGADRO_NUMBER)
        # Clamped to physical range [-120 kJ/mol, 0 kJ/mol]
        return float(np.clip(delta_h_seg_j_mol, -120000.0, 0.0))

    def solve_multicomponent_mclean_segregation(
        self,
        bulk_concentrations: Dict[str, float],
        segregation_free_energies_j_mol: Optional[Dict[str, float]] = None,
        matrix_element: str = "Fe",
    ) -> Dict[str, float]:
        """Solve Langmuir-McLean multi-component competitive segregation isotherm:

        X_i^GB = [X_i^bulk * exp(-dG_i / RT)] / [1 + sum_j X_j^bulk * (exp(-dG_j / RT) - 1)]
        """
        r_t = R_GAS * self.T
        gb_concentrations: Dict[str, float] = {}

        # Default standard segregation free energies (dG_seg < 0 implies strong GB enrichment)
        # Standard values for common solutes in transition metals (kJ/mol)
        default_dg: Dict[str, float] = {
            "P": -45000.0,
            "S": -65000.0,
            "Sb": -38000.0,
            "Sn": -32000.0,
            "As": -28000.0,
            "B": -55000.0,
            "C": -50000.0,
            "N": -48000.0,
            "O": -70000.0,
            "H": -35000.0,
            "Cr": -12000.0,
            "Mo": -18000.0,
            "Nb": -25000.0,
            "V": -15000.0,
            "Ti": -22000.0,
            "Al": -8000.0,
            "Si": -10000.0,
            "Ni": -5000.0,
        }

        dg_map = segregation_free_energies_j_mol or default_dg

        # Compute numerator terms
        terms = {}
        sum_terms = 0.0
        for elem, c_bulk in bulk_concentrations.items():
            if elem == matrix_element or c_bulk <= 0.0:
                continue
            dg_i = dg_map.get(elem, -15000.0)
            exp_term = np.exp(min(40.0, -dg_i / r_t))
            terms[elem] = c_bulk * exp_term
            sum_terms += c_bulk * (exp_term - 1.0)

        denom = max(1e-6, 1.0 + sum_terms)

        total_gb_solute = 0.0
        for elem, num in terms.items():
            c_gb = float(np.clip(num / denom, 0.0, 0.95))
            gb_concentrations[elem] = c_gb
            total_gb_solute += c_gb

        gb_concentrations[matrix_element] = max(0.05, 1.0 - total_gb_solute)
        return gb_concentrations

    def evaluate_rice_wang_interfacial_embrittlement(
        self,
        gb_solute_concentrations: Dict[str, float],
        clean_gb_surface_energy_j_m2: float = 0.85,
        clean_free_surface_energy_j_m2: float = 2.20,
    ) -> Dict[str, Any]:
        """Evaluate Rice-Wang thermodynamic theory of interfacial cohesive work of separation:

        2 * gamma_int(c_GB) = 2 * gamma_fs(c_fs) - gamma_gb(c_gb)
        W_ad = 2 * gamma_fs - gamma_gb
        """
        # Clean boundary work of adhesion
        w_ad_clean = 2.0 * clean_free_surface_energy_j_m2 - clean_gb_surface_energy_j_m2

        # Solute embrittlement potencies (d(dG_seg) / dc in J/mol per monolayer)
        # Negative potency = Embrittler (e.g. S, P, Sb, Sn, O, H)
        # Positive potency = Cohesion enhancer (e.g. B, C, Mo, W)
        embrittlement_potency_j_m2: Dict[str, float] = {
            "S": -1.80,
            "P": -1.25,
            "Sb": -0.95,
            "Sn": -0.75,
            "As": -0.65,
            "O": -2.10,
            "H": -1.50,
            "B": +0.85,
            "C": +0.65,
            "Mo": +0.40,
            "W": +0.35,
            "Cr": +0.10,
            "Nb": +0.25,
            "Ni": +0.05,
        }

        delta_w_ad = 0.0
        for elem, c_gb in gb_solute_concentrations.items():
            potency = embrittlement_potency_j_m2.get(elem, 0.0)
            delta_w_ad += potency * c_gb

        w_ad_effective = float(max(0.2, w_ad_clean + delta_w_ad))
        cohesion_ratio = float(w_ad_effective / w_ad_clean)

        # Fracture toughness scaling factor K_Ic(GB) / K_Ic(bulk)
        kic_degradation_factor = float(np.sqrt(cohesion_ratio))

        # DBTT Shift (Ductile-to-Brittle Transition Temperature shift in Kelvin)
        # Empirical Segher relation: delta_DBTT = -350 * (delta_W_ad / W_ad_clean)
        dbtt_shift_k = float(-350.0 * (delta_w_ad / w_ad_clean))

        is_embrittled = cohesion_ratio < 0.85

        return {
            "clean_work_of_adhesion_j_m2": float(w_ad_clean),
            "effective_work_of_adhesion_j_m2": w_ad_effective,
            "interfacial_cohesion_ratio": cohesion_ratio,
            "fracture_toughness_retention_factor": kic_degradation_factor,
            "dbtt_temperature_shift_k": dbtt_shift_k,
            "is_intergranular_embrittlement_risk": is_embrittled,
        }
