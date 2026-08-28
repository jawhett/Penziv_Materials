"""Thermomechanical History Engine: Predicts variations in Yield, Fracture Toughness, Plasticity, and Fatigue Parameters."""

from enum import Enum
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from pydantic import BaseModel, Field
from penziv_materials.core.constants import BOLTZMANN_J_K, R_GAS


class ProcessingRoute(str, Enum):
    """Standard industrial and advanced thermomechanical processing pathways."""
    ANNEALED_RECRYSTALLIZED = "annealed_recrystallized"
    COLD_WORKED_50PCT = "cold_worked_50pct"
    SOLUTION_TREATED_PEAK_AGED_T6 = "solution_treated_peak_aged_t6"
    ADDITIVE_LPBF_AS_PRINTED = "additive_lpbf_as_printed"
    ADDITIVE_LPBF_HIP_AGED = "additive_lpbf_hip_aged"


class ThermomechanicalHistoryParameters(BaseModel):
    """Input processing parameters for a thermomechanical history pathway."""
    route: ProcessingRoute = ProcessingRoute.ANNEALED_RECRYSTALLIZED
    temperature_k: float = 298.15
    prior_cold_work_strain: float = 0.0
    cooling_rate_k_s: float = 1.0
    post_anneal_temp_k: Optional[float] = None
    post_anneal_time_hours: Optional[float] = None
    residual_stress_mpa: float = 0.0
    void_volume_fraction: float = 0.0001


class ThermomechanicalPropertyResponse(BaseModel):
    """Output structural, fracture, plastic, and fatigue response conditioned on thermomechanical history."""
    processing_route: str
    effective_grain_size_um: float
    dislocation_density_m2: float
    precipitate_volume_fraction: float
    
    # Plasticity & Tensile Properties
    yield_strength_mpa: float
    ultimate_tensile_strength_mpa: float
    uniform_elongation_percent: float
    total_elongation_to_failure_percent: float
    strain_hardening_exponent_n: float
    work_hardening_coefficient_k_mpa: float
    
    # Fracture Toughness
    fracture_toughness_k_ic_mpa_sqrt_m: float
    critical_crack_tip_opening_displacement_ctod_um: float
    plastic_zone_radius_rp_mm: float
    
    # Cyclic Fatigue Parameters
    fatigue_endurance_limit_sigma_e_mpa: float
    basquin_fatigue_strength_coeff_sigma_f_prime_mpa: float
    basquin_exponent_b: float
    coffin_manson_fatigue_ductility_coeff_eps_f_prime: float
    coffin_manson_exponent_c: float
    transition_fatigue_life_cycles_nt: float
    paris_law_c: float
    paris_law_m: float
    fatigue_threshold_delta_k_th_mpa_sqrt_m: float


class ThermomechanicalHistoryEngine:
    """Predicts physical microstructural evolution, yield strength, work hardening, fracture toughness, and fatigue parameters under thermomechanical processing."""

    def __init__(
        self,
        burgers_vector_m: float = 2.54e-10,
        shear_modulus_gpa: float = 80.0,
        poisson_ratio: float = 0.30,
        taylor_factor: float = 3.067,
    ):
        self.b = burgers_vector_m
        self.G_pa = shear_modulus_gpa * 1.0e9
        self.nu = poisson_ratio
        self.M = taylor_factor

    def predict_properties_from_history(
        self,
        base_yield_strength_mpa: float,
        base_youngs_modulus_gpa: float,
        history: ThermomechanicalHistoryParameters,
        lattice_friction_stress_mpa: Optional[float] = None,
    ) -> ThermomechanicalPropertyResponse:
        """Compute full physical property alterations conditioned on thermomechanical route."""
        route = history.route
        E_gpa = base_youngs_modulus_gpa
        E_pa = E_gpa * 1.0e9
        G_pa = E_pa / (2.0 * (1.0 + self.nu))
        sigma_0 = lattice_friction_stress_mpa or max(20.0, base_yield_strength_mpa * 0.35)

        # 1. Microstructural State Variables by Route
        if route == ProcessingRoute.ANNEALED_RECRYSTALLIZED:
            d_um = 55.0
            rho_disl = 1.0e12  # Well-annealed low dislocation density
            f_v = 0.001
            r_p_nm = 5.0
            sigma_res_mpa = 0.0
            void_frac = max(1e-5, history.void_volume_fraction)
            k_surf = 1.0

        elif route == ProcessingRoute.COLD_WORKED_50PCT:
            d_um = 18.0  # Grain elongation / subdivision
            rho_disl = 2.5e15  # Heavy dislocation forest
            f_v = 0.001
            r_p_nm = 5.0
            sigma_res_mpa = 180.0  # High tensile residual stress
            void_frac = max(2e-4, history.void_volume_fraction * 2.0)
            k_surf = 0.85

        elif route == ProcessingRoute.SOLUTION_TREATED_PEAK_AGED_T6:
            d_um = 35.0
            rho_disl = 5.0e13
            f_v = 0.045  # Dense nanoscale precipitation
            r_p_nm = 8.0  # Optimal Orowan looping radius
            sigma_res_mpa = 40.0
            void_frac = max(1e-5, history.void_volume_fraction)
            k_surf = 0.95

        elif route == ProcessingRoute.ADDITIVE_LPBF_AS_PRINTED:
            d_um = 8.0  # Rapid solidification fine cell structure
            rho_disl = 8.0e14  # High cellular dislocation density
            f_v = 0.010
            r_p_nm = 3.0
            sigma_res_mpa = max(220.0, base_yield_strength_mpa * 0.65)  # Severe thermal tensile stress
            void_frac = max(5e-4, history.void_volume_fraction * 5.0)  # Gas/lack-of-fusion pores
            k_surf = 0.42  # Severe as-printed unmachined surface roughness / notch knockdown

        elif route == ProcessingRoute.ADDITIVE_LPBF_HIP_AGED:
            d_um = 42.0  # Recrystallized / coarsened
            rho_disl = 8.0e12  # Dislocation recovery
            f_v = 0.035  # Post-HIP precipitation
            r_p_nm = 12.0
            sigma_res_mpa = 15.0  # Complete residual stress relief
            void_frac = 1e-5  # Isostatic pore closure
            k_surf = 0.95  # Machined & polished finish

        else:
            d_um = 30.0
            rho_disl = 1.0e13
            f_v = 0.01
            r_p_nm = 6.0
            sigma_res_mpa = 0.0
            void_frac = 1e-4
            k_surf = 1.0

        # 2. Physics-Based Strengthening Mechanisms
        # Hall-Petch grain boundary strengthening
        k_hp_mpa_sqrt_um = 12.5
        delta_sigma_hp = k_hp_mpa_sqrt_um / np.sqrt(max(0.5, d_um))

        # Taylor dislocation forest hardening: Delta sigma_T = M * alpha * G * b * sqrt(rho)
        alpha_taylor = 0.33
        delta_sigma_taylor = (self.M * alpha_taylor * (G_pa * 1e-6) * self.b * np.sqrt(rho_disl))

        # Orowan precipitation strengthening: Delta sigma_Orowan
        if f_v > 0.002:
            l_spacing_m = (r_p_nm * 1e-9) * np.sqrt(2.0 * np.pi / (3.0 * f_v))
            delta_sigma_orowan = (
                (0.81 * self.M * (G_pa * 1e-6) * self.b)
                / (2.0 * np.pi * np.sqrt(1.0 - self.nu) * max(1e-9, l_spacing_m - 2.0 * r_p_nm * 1e-9))
                * np.log(max(1.5, 2.0 * r_p_nm * 1e-9 / self.b))
            )
        else:
            delta_sigma_orowan = 0.0

        # Total Yield Strength
        sigma_y = float(sigma_0 + delta_sigma_hp + delta_sigma_taylor + delta_sigma_orowan)

        # 3. Plasticity, Work Hardening & Ductility
        if route == ProcessingRoute.COLD_WORKED_50PCT:
            n_exp = 0.08  # Saturated work hardening
            eps_u = 3.5  # Uniform elongation %
            eps_f = 12.0  # Total failure elongation %
        elif route == ProcessingRoute.ADDITIVE_LPBF_AS_PRINTED:
            n_exp = 0.12
            eps_u = 8.0
            eps_f = 16.0
        elif route == ProcessingRoute.SOLUTION_TREATED_PEAK_AGED_T6:
            n_exp = 0.16
            eps_u = 14.0
            eps_f = 24.0
        elif route == ProcessingRoute.ADDITIVE_LPBF_HIP_AGED:
            n_exp = 0.18
            eps_u = 18.0
            eps_f = 32.0
        else:  # Annealed
            n_exp = 0.26  # High strain hardening capacity
            eps_u = 32.0
            eps_f = 48.0

        K_work_hard = float(sigma_y * 1.65)
        sigma_uts = float(sigma_y * (1.0 + (eps_u / 100.0) ** n_exp * 0.45))

        # 4. Fracture Toughness K_Ic & Plastic Zone Size
        # Rice-Johnson ductile fracture model: K_Ic = sqrt(2 * E * gamma_eff / (1 - nu^2))
        gamma_surface_j_m2 = 2.2
        # Plastic work of dissipation scales with ductility eps_f
        gamma_plastic_dissipation = 2200.0 * ((eps_f / 30.0) ** 1.5) * (800.0 / max(250.0, sigma_y))
        gamma_eff = gamma_surface_j_m2 * (1.0 + gamma_plastic_dissipation)

        k_ic_pa_sqrt_m = np.sqrt((2.0 * E_pa * gamma_eff) / max(0.1, 1.0 - self.nu**2))
        k_ic = float(np.clip(k_ic_pa_sqrt_m * 1.0e-6, 15.0, 250.0))

        # Critical CTOD: delta_c = K_Ic^2 / (m * sigma_y * E)
        delta_ctod_um = float((k_ic**2 * 1.0e6) / (1.5 * sigma_y * E_gpa * 1000.0) * 1.0e3)
        # Irwin plastic zone radius r_p = (1 / 6*pi) * (K_Ic / sigma_y)^2
        r_p_mm = float((1.0 / (6.0 * np.pi)) * ((k_ic / sigma_y) ** 2) * 1000.0)

        # 5. Cyclic Fatigue Parameters
        # Unnotched smooth endurance limit sigma_e,0 ~ 0.45 * sigma_uts
        sigma_e_intrinsic = 0.45 * sigma_uts
        # Modified by Goodman mean/residual stress & surface condition
        residual_reduction = max(0.15, 1.0 - (sigma_res_mpa / max(1.0, sigma_uts)))
        pore_reduction = max(0.20, 1.0 - (void_frac * 200.0))
        sigma_e = float(max(25.0, sigma_e_intrinsic * k_surf * pore_reduction * residual_reduction))

        # Basquin High-Cycle Fatigue Parameters: sigma_a = sigma_f' * (2*N_f)^b
        sigma_f_prime = float(sigma_uts + 345.0)
        b_basquin = float(-np.log10(max(1.1, (2.0 * sigma_f_prime) / max(10.0, sigma_e))) / 6.0)
        b_basquin = float(np.clip(b_basquin, -0.16, -0.06))

        # Coffin-Manson Low-Cycle Fatigue Parameters: Delta eps_p / 2 = eps_f' * (2*N_f)^c
        eps_f_prime = float(np.log(1.0 / max(0.1, 1.0 - (eps_f / 100.0) * 0.75)))
        c_coffin = float(-0.55 - (sigma_y / 4000.0) * 0.15)

        # Transition Fatigue Life N_t (cycles where elastic and plastic strains intersect)
        nt_cycles = float(0.5 * ((eps_f_prime * (E_gpa * 1000.0)) / max(1.0, sigma_f_prime)) ** (1.0 / (b_basquin - c_coffin)))
        nt_cycles = float(np.clip(nt_cycles, 50.0, 50000.0))

        # Paris Law Fatigue Crack Propagation: da/dN = C * (Delta K)^m
        m_paris = float(2.8 + (150.0 / max(20.0, k_ic)) * 0.6)
        c_paris = float(1.2e-11 * (80.0 / max(10.0, E_gpa)) ** 2)
        delta_k_th = float(np.clip(0.12 * k_ic, 2.0, 12.0))

        return ThermomechanicalPropertyResponse(
            processing_route=route.value,
            effective_grain_size_um=float(round(d_um, 2)),
            dislocation_density_m2=float(rho_disl),
            precipitate_volume_fraction=float(round(f_v, 4)),
            yield_strength_mpa=float(round(sigma_y, 1)),
            ultimate_tensile_strength_mpa=float(round(sigma_uts, 1)),
            uniform_elongation_percent=float(round(eps_u, 1)),
            total_elongation_to_failure_percent=float(round(eps_f, 1)),
            strain_hardening_exponent_n=float(round(n_exp, 3)),
            work_hardening_coefficient_k_mpa=float(round(K_work_hard, 1)),
            fracture_toughness_k_ic_mpa_sqrt_m=float(round(k_ic, 1)),
            critical_crack_tip_opening_displacement_ctod_um=float(round(delta_ctod_um, 2)),
            plastic_zone_radius_rp_mm=float(round(r_p_mm, 3)),
            fatigue_endurance_limit_sigma_e_mpa=float(round(sigma_e, 1)),
            basquin_fatigue_strength_coeff_sigma_f_prime_mpa=float(round(sigma_f_prime, 1)),
            basquin_exponent_b=float(round(b_basquin, 4)),
            coffin_manson_fatigue_ductility_coeff_eps_f_prime=float(round(eps_f_prime, 3)),
            coffin_manson_exponent_c=float(round(c_coffin, 3)),
            transition_fatigue_life_cycles_nt=float(round(nt_cycles, 0)),
            paris_law_c=float(c_paris),
            paris_law_m=float(round(m_paris, 2)),
            fatigue_threshold_delta_k_th_mpa_sqrt_m=float(round(delta_k_th, 2)),
        )
