"""Cohesive Zone Interface Mechanics, DFT Work of Separation & Coupled PNP-Biot Chemomechanics."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.constants import R_GAS, FARADAY_CONSTANT, EPSILON_0


class CohesiveZoneInterfaceEngine:
    """Evaluates multi-material interface fracture toughness, bilinear/exponential traction-separation laws, and coupled PNP-Biot chemomechanics."""

    def __init__(self, temperature_k: float = 300.0):
        self.T = max(1.0, temperature_k)

    def compute_work_of_separation(
        self,
        surface_energy_phase1_j_m2: float,
        surface_energy_phase2_j_m2: float,
        interface_energy_j_m2: float,
    ) -> Dict[str, float]:
        """Compute Dupré ab initio work of separation:

        W_sep = gamma_1 + gamma_2 - gamma_int
        """
        w_sep = surface_energy_phase1_j_m2 + surface_energy_phase2_j_m2 - interface_energy_j_m2
        w_sep = max(0.01, w_sep)
        return {
            "work_of_separation_w_sep_j_m2": float(w_sep),
            "surface_energy_1_j_m2": float(surface_energy_phase1_j_m2),
            "surface_energy_2_j_m2": float(surface_energy_phase2_j_m2),
            "interface_energy_j_m2": float(interface_energy_j_m2),
            "is_thermodynamically_adherent": bool(w_sep > 0.0),
        }

    def evaluate_exponential_traction_separation(
        self,
        normal_opening_delta_n_nm: float,
        shear_opening_delta_t_nm: float,
        work_of_separation_j_m2: float = 1.5,
        characteristic_opening_delta_0_nm: float = 0.5,
    ) -> Dict[str, float]:
        """Evaluate Xu-Needleman exponential cohesive traction-separation response:

        T_n = (W_sep / delta_0) * (delta_n / delta_0) * exp(-delta_n / delta_0)
        """
        w_sep = max(0.01, work_of_separation_j_m2)
        d0 = max(0.05, characteristic_opening_delta_0_nm) * 1.0e-9
        dn = normal_opening_delta_n_nm * 1.0e-9
        dt = shear_opening_delta_t_nm * 1.0e-9

        sig_max = w_sep / (np.e * d0)  # Peak cohesive traction in Pa
        norm_arg = max(0.0, dn / d0)
        t_n_pa = (w_sep / d0) * norm_arg * np.exp(1.0 - norm_arg)
        t_t_pa = 2.0 * (w_sep / d0) * (dt / d0) * np.exp(-norm_arg) * np.exp(-(dt / d0)**2)

        return {
            "normal_traction_t_n_mpa": float(t_n_pa * 1.0e-6),
            "shear_traction_t_t_mpa": float(t_t_pa * 1.0e-6),
            "peak_cohesive_strength_mpa": float(sig_max * 1.0e-6),
            "opening_to_peak_ratio": float(normal_opening_delta_n_nm / characteristic_opening_delta_0_nm),
            "is_debonded": bool(normal_opening_delta_n_nm > 5.0 * characteristic_opening_delta_0_nm),
        }

    def solve_coupled_pnp_biot_fluxes(
        self,
        concentration_field_mol_m3: np.ndarray,
        electric_potential_v: np.ndarray,
        hydrostatic_stress_field_mpa: np.ndarray,
        diffusivity_m2_s: float = 1.0e-12,
        ion_valence_z: int = 1,
        partial_molar_volume_m3_mol: float = 1.0e-5,
        dx_m: float = 1.0e-9,
    ) -> Dict[str, Any]:
        """Solve coupled mass-charge-stress fluxes:

        J = - D * grad(c) - (z F D / R T) c * grad(phi) + (D Omega / R T) c * grad(sigma_h)
        """
        c = np.asarray(concentration_field_mol_m3, dtype=np.float64)
        phi = np.asarray(electric_potential_v, dtype=np.float64)
        sig_h = np.asarray(hydrostatic_stress_field_mpa, dtype=np.float64) * 1.0e6

        grad_c = np.stack(np.gradient(c, dx_m), axis=-1)
        grad_phi = np.stack(np.gradient(phi, dx_m), axis=-1)
        grad_sig = np.stack(np.gradient(sig_h, dx_m), axis=-1)

        rt = R_GAS * self.T
        f_const = FARADAY_CONSTANT

        # Flux components
        j_diff = -diffusivity_m2_s * grad_c
        j_mig = -(ion_valence_z * f_const * diffusivity_m2_s / rt) * c[..., np.newaxis] * grad_phi
        j_stress = (diffusivity_m2_s * partial_molar_volume_m3_mol / rt) * c[..., np.newaxis] * grad_sig

        j_net = j_diff + j_mig + j_stress
        j_mag = np.linalg.norm(j_net, axis=-1)

        return {
            "max_ion_flux_mol_m2_s": float(np.max(j_mag)),
            "mean_ion_flux_mol_m2_s": float(np.mean(j_mag)),
            "migration_to_diffusion_ratio": float(np.mean(np.linalg.norm(j_mig, axis=-1)) / max(1e-12, np.mean(np.linalg.norm(j_diff, axis=-1)))),
            "stress_to_diffusion_ratio": float(np.mean(np.linalg.norm(j_stress, axis=-1)) / max(1e-12, np.mean(np.linalg.norm(j_diff, axis=-1)))),
            "is_space_charge_limited": bool(np.mean(np.linalg.norm(j_mig, axis=-1)) > np.mean(np.linalg.norm(j_diff, axis=-1))),
        }

    def solve_reactive_interdiffusion_stefan_growth(
        self,
        time_seconds: float,
        pre_exponential_k0_m2_s: float = 1.2e-4,
        activation_energy_q_j_mol: float = 145000.0,
        intermetallic_phase_name: str = "Ni3Al",
    ) -> Dict[str, Any]:
        """Solve Stefan multiphase reactive interdiffusion parabolic layer growth:

        x(t) = sqrt(2 * k_p * t),  k_p = k_0 * exp(-Q / (R * T))
        """
        rt = R_GAS * max(1.0, self.T)
        k_p = pre_exponential_k0_m2_s * np.exp(-activation_energy_q_j_mol / rt)
        layer_thickness_m = np.sqrt(2.0 * k_p * max(0.0, time_seconds))
        layer_thickness_um = float(layer_thickness_m * 1.0e6)
        growth_rate_um_h = float((np.sqrt(k_p / (2.0 * max(1.0, time_seconds))) * 1.0e6 * 3600.0) if time_seconds > 0 else 0.0)

        return {
            "intermetallic_phase": intermetallic_phase_name,
            "layer_thickness_microns": layer_thickness_um,
            "growth_rate_microns_per_hour": growth_rate_um_h,
            "parabolic_rate_constant_kp_m2_s": float(k_p),
            "temperature_k": float(self.T),
            "exposure_time_hours": float(time_seconds / 3600.0),
        }
