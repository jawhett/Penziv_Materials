"""Thermomechanical History Engine: Predicts variations in Yield, Fracture Toughness, Plasticity, and Fatigue via Continuous ISV Differential Equations."""

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
    CUSTOM_ISV_TRAJECTORY = "custom_isv_trajectory"


class InternalStateVector(BaseModel):
    """Explicit Internal State Variable (ISV) Vector S(t) = [rho, d, f_p, r_bar_p, phi]^T."""
    dislocation_density_m2: float = 1.0e12
    grain_size_um: float = 30.0
    precipitate_volume_fraction: float = 0.0
    mean_precipitate_radius_nm: float = 0.0
    phase_fractions: Dict[str, float] = Field(default_factory=lambda: {"matrix": 1.0})


class ThermomechanicalHistoryParameters(BaseModel):
    """Input processing parameters for a thermomechanical history pathway."""
    route: ProcessingRoute = ProcessingRoute.ANNEALED_RECRYSTALLIZED
    temperature_k: float = 298.15
    prior_cold_work_strain: float = 0.0
    cooling_rate_k_s: float = 1.0
    strain_rate_s_inv: float = 1.0e-3
    anneal_temperature_k: Optional[float] = None
    anneal_time_seconds: Optional[float] = None
    residual_stress_mpa: float = 0.0
    void_volume_fraction: float = 0.0001
    initial_isv: Optional[InternalStateVector] = None


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
    internal_state_vector: Optional[InternalStateVector] = None


class ThermomechanicalHistoryEngine:
    """Predicts physical microstructural evolution, yield strength, work hardening, fracture toughness, and fatigue parameters via continuous ISV differential equations."""

    def __init__(
        self,
        burgers_vector_m: float = 2.54e-10,
        shear_modulus_gpa: float = 77.0,
        poisson_ratio: float = 0.30,
        taylor_factor: float = 3.067,
    ):
        self.b = burgers_vector_m
        self.G_pa = shear_modulus_gpa * 1.0e9
        self.nu = poisson_ratio
        self.M = taylor_factor

    def compute_kocks_mecking_dislocation_rate(
        self,
        rho_m2: float,
        strain_rate_s_inv: float,
        temperature_k: float,
    ) -> float:
        """Evaluate Kocks-Mecking dislocation density rate d(rho)/dt = (k1 * sqrt(rho) - k2(eps_dot, T) * rho) * eps_dot."""
        if strain_rate_s_inv <= 0.0:
            return 0.0
        rho = max(1e10, float(rho_m2))
        k1 = 2.0 / (20.0 * self.b)
        q_rec = 120000.0  # J/mol cross-slip activation energy
        k2_0 = 12.0
        k2 = k2_0 * ((1e-3 / max(1e-6, strain_rate_s_inv)) ** 0.1) * np.exp(-q_rec / (R_GAS * max(100.0, temperature_k)))
        k2 = max(0.5, k2)
        d_rho_d_eps = k1 * np.sqrt(rho) - k2 * rho
        return float(d_rho_d_eps * strain_rate_s_inv)

    def integrate_kocks_mecking_dislocation_density(
        self,
        rho_initial_m2: float,
        plastic_strain: float,
        strain_rate_s_inv: float = 1e-3,
        temperature_k: float = 298.15,
        steps: int = 100,
    ) -> float:
        """Integrate Kocks-Mecking-Estrin equation: d(rho)/d(eps) = k1 * sqrt(rho) - k2(eps_dot, T) * rho."""
        if plastic_strain <= 0.0:
            return float(rho_initial_m2)

        rho = max(1e10, float(rho_initial_m2))
        d_eps = plastic_strain / steps
        
        # Athermal dislocation storage coefficient k1 ~ 2 / (C * b)
        k1 = 2.0 / (20.0 * self.b)
        
        # Dynamic recovery coefficient k2(T, eps_dot) = k2_0 * (eps_dot_0 / eps_dot)^(1/n) * exp(-Q / RT)
        q_rec = 120000.0  # J/mol cross-slip activation energy
        k2_0 = 12.0
        k2 = k2_0 * ((1e-3 / max(1e-6, strain_rate_s_inv)) ** 0.1) * np.exp(-q_rec / (R_GAS * max(100.0, temperature_k)))
        k2 = max(0.5, k2)

        for _ in range(steps):
            d_rho_d_eps = k1 * np.sqrt(rho) - k2 * rho
            rho = max(1e10, rho + d_rho_d_eps * d_eps)

        return float(rho)

    def compute_grain_growth_and_drx_rate(
        self,
        grain_size_um: float,
        dislocation_density_m2: float,
        temperature_k: float,
        strain_rate_s_inv: float = 0.0,
        grain_boundary_energy_j_m2: float = 0.60,
    ) -> float:
        """Evaluate grain size rate dd/dt = (M_0 * gamma_GB / d) * exp(-Q_gg / RT) - d_dot_DRX(rho, eps_dot)."""
        d_m = max(0.1e-6, grain_size_um * 1e-6)
        m0_gamma = 1.5e-5 * grain_boundary_energy_j_m2
        q_gg = 180000.0  # J/mol
        thermal_growth_m_s = (m0_gamma / d_m) * np.exp(-q_gg / (R_GAS * max(300.0, temperature_k)))

        # Dynamic recrystallization refinement rate
        rho_crit = 5.0e14
        if dislocation_density_m2 > rho_crit and strain_rate_s_inv > 0.0 and temperature_k > 700.0:
            d_ss_m = 10.0 * (self.G_pa * self.b / max(1e3, dislocation_density_m2 * self.G_pa * (self.b**2) / 2.0)) ** 0.5
            d_ss_m = max(1.0e-6, min(50.0e-6, d_ss_m))
            drx_rate_m_s = 0.05 * (dislocation_density_m2 / rho_crit) * strain_rate_s_inv * max(0.0, d_m - d_ss_m)
        else:
            drx_rate_m_s = 0.0

        net_rate_m_s = thermal_growth_m_s - drx_rate_m_s
        return float(net_rate_m_s * 1e6)  # um / s

    def integrate_grain_growth_and_drx(
        self,
        initial_grain_size_um: float,
        dislocation_density_m2: float,
        anneal_time_s: float = 3600.0,
        anneal_temp_k: float = 1273.15,
        grain_boundary_energy_j_m2: float = 0.60,
    ) -> float:
        """Integrate grain boundary migration driven by stored deformation energy: v_GB = M_GB * (rho * G*b^2 / 2 - 2*gamma_GB / R)."""
        r_grain_m = max(0.1e-6, (initial_grain_size_um * 1e-6) / 2.0)
        
        # Mobility M_GB(T) = M_0 * exp(-Q_m / RT)
        m0 = 1.5e-5
        q_mig = 180000.0  # J/mol
        m_gb = m0 * np.exp(-q_mig / (R_GAS * max(300.0, anneal_temp_k)))
        
        # Driving pressure P_stored = rho * G * b^2 / 2
        p_stored = dislocation_density_m2 * self.G_pa * (self.b**2) / 2.0
        
        # Capillary retarding pressure P_cap = 2 * gamma_GB / R
        p_cap = (2.0 * grain_boundary_energy_j_m2) / r_grain_m
        
        net_driving_pressure = p_stored - p_cap
        
        if p_stored > 5e5 and anneal_temp_k > 800.0:
            # Recrystallization nucleation reduces grain size to equiaxed recrystallized diameter
            d_rx_m = 15.0 * (self.G_pa * self.b / max(1e3, p_stored)) ** 0.5
            r_grain_m = max(1.0e-6, min(50.0e-6, d_rx_m))
        else:
            # Normal grain coarsening: R^2(t) = R_0^2 + 2 * M_GB * gamma_GB * t
            r_grain_m = np.sqrt(r_grain_m**2 + max(0.0, 2.0 * m_gb * grain_boundary_energy_j_m2 * anneal_time_s))

        return float(round(r_grain_m * 2.0 * 1e6, 2))

    def compute_lsw_precipitate_coarsening_rate(
        self,
        radius_nm: float,
        temperature_k: float,
        interfacial_energy_j_m2: float = 0.25,
        diffusivity_d0_m2_s: float = 1.0e-4,
        q_diff_j_mol: float = 140000.0,
        solute_solubility_c_e: float = 0.02,
        molar_volume_m3_mol: float = 1.0e-5,
    ) -> float:
        """Evaluate Lifshitz-Slyozov-Wagner (LSW) coarsening rate:

        d(r_bar^3)/dt = (8 * gamma_p * D(T) * C_e * V_m^2) / (9 * R * T)
        """
        t_k = max(200.0, temperature_k)
        d_diff = diffusivity_d0_m2_s * np.exp(-q_diff_j_mol / (R_GAS * t_k))
        k_lsw_m3_s = (8.0 * interfacial_energy_j_m2 * d_diff * solute_solubility_c_e * (molar_volume_m3_mol**2)) / (9.0 * R_GAS * t_k)
        
        # d(r)/dt = k_LSW / (3 * r^2)
        r_m = max(0.5e-9, radius_nm * 1e-9)
        dr_dt_m_s = k_lsw_m3_s / (3.0 * (r_m**2))
        return float(dr_dt_m_s * 1e9)  # nm / s

    def compute_precipitate_strengthening(
        self,
        mean_radius_nm: float,
        volume_fraction: float,
        g_shear_mpa: float,
        critical_radius_nm: float = 4.0,
    ) -> float:
        """Evaluate precipitate hardening transition between particle shearing and Orowan dislocation looping."""
        if volume_fraction <= 0.0005 or mean_radius_nm <= 0.2:
            return 0.0
        
        r_m = mean_radius_nm * 1e-9
        r_crit_m = critical_radius_nm * 1e-9
        
        # Particle shearing (Friedel-Fleischer / coherency)
        delta_sigma_shear = float(self.M * 0.15 * g_shear_mpa * np.sqrt(max(1e-6, (r_m / self.b) * volume_fraction)))
        
        # Orowan dislocation looping
        l_spacing_m = r_m * np.sqrt(2.0 * np.pi / (3.0 * max(1e-5, volume_fraction)))
        delta_sigma_orowan = float(
            (0.81 * self.M * g_shear_mpa * self.b)
            / (2.0 * np.pi * np.sqrt(max(0.1, 1.0 - self.nu)) * max(1e-9, l_spacing_m - 2.0 * r_m))
            * np.log(max(1.5, 2.0 * r_m / self.b))
        )
        
        # Active strengthening mechanism transition
        if r_m < r_crit_m:
            return float(min(delta_sigma_shear, delta_sigma_orowan))
        else:
            return float(delta_sigma_orowan)

    def integrate_continuous_isv_trajectory(
        self,
        time_series_s: np.ndarray,
        temperature_series_k: np.ndarray,
        strain_rate_series_s_inv: Optional[np.ndarray] = None,
        initial_isv: Optional[InternalStateVector] = None,
        base_yield_strength_mpa: float = 300.0,
        base_youngs_modulus_gpa: float = 200.0,
        solid_solution_hardening_mpa: float = 0.0,
    ) -> Dict[str, Any]:
        """Integrate continuous Internal State Variable (ISV) vector S(t) across thermomechanical history."""
        times = np.asarray(time_series_s, dtype=np.float64)
        temps = np.asarray(temperature_series_k, dtype=np.float64)
        n_steps = len(times)
        if strain_rate_series_s_inv is not None:
            eps_dots = np.asarray(strain_rate_series_s_inv, dtype=np.float64)
        else:
            eps_dots = np.zeros(n_steps, dtype=np.float64)

        # Initial state vector S(0)
        isv = initial_isv or InternalStateVector()
        rho = float(isv.dislocation_density_m2)
        d_um = float(isv.grain_size_um)
        f_p = float(isv.precipitate_volume_fraction)
        r_p = float(isv.mean_precipitate_radius_nm)

        rho_hist = [rho]
        d_hist = [d_um]
        r_p_hist = [r_p]
        yield_hist = []

        g_shear_mpa = (base_youngs_modulus_gpa * 1.0e3) / (2.0 * (1.0 + self.nu))
        sigma_0 = base_yield_strength_mpa * 0.70

        for idx in range(n_steps - 1):
            dt = max(1e-4, times[idx + 1] - times[idx])
            t_curr = temps[idx]
            eps_dot_curr = eps_dots[idx]

            # 1. Dislocation density evolution
            d_rho = self.compute_kocks_mecking_dislocation_rate(rho, eps_dot_curr, t_curr) * dt
            rho = max(1e10, rho + d_rho)

            # 2. Grain size evolution & DRX
            d_grain = self.compute_grain_growth_and_drx_rate(d_um, rho, t_curr, eps_dot_curr) * dt
            d_um = max(0.5, d_um + d_grain)

            # 3. LSW Precipitate coarsening
            if f_p > 0.0005:
                dr_dt = self.compute_lsw_precipitate_coarsening_rate(r_p, t_curr)
                r_p = max(0.1, r_p + dr_dt * dt)

            # Microstructural yield stress mapping
            delta_sigma_hp = 10.5 / np.sqrt(max(0.2, d_um))
            delta_sigma_taylor = (self.M * 0.28 * g_shear_mpa * self.b * np.sqrt(rho))
            delta_sigma_ppt = self.compute_precipitate_strengthening(r_p, f_p, g_shear_mpa)
            sigma_y_t = sigma_0 + solid_solution_hardening_mpa + delta_sigma_hp + delta_sigma_taylor + delta_sigma_ppt

            rho_hist.append(rho)
            d_hist.append(d_um)
            r_p_hist.append(r_p)
            yield_hist.append(float(sigma_y_t))

        # Final step yield
        delta_sigma_hp = 10.5 / np.sqrt(max(0.2, d_um))
        delta_sigma_taylor = (self.M * 0.28 * g_shear_mpa * self.b * np.sqrt(rho))
        delta_sigma_ppt = self.compute_precipitate_strengthening(r_p, f_p, g_shear_mpa)
        yield_hist.append(float(sigma_0 + solid_solution_hardening_mpa + delta_sigma_hp + delta_sigma_taylor + delta_sigma_ppt))

        final_isv = InternalStateVector(
            dislocation_density_m2=rho,
            grain_size_um=d_um,
            precipitate_volume_fraction=f_p,
            mean_precipitate_radius_nm=r_p,
            phase_fractions={"matrix": 1.0 - f_p, "precipitate": f_p},
        )

        return {
            "final_isv": final_isv,
            "dislocation_density_trajectory": rho_hist,
            "grain_size_trajectory_um": d_hist,
            "precipitate_radius_trajectory_nm": r_p_hist,
            "yield_strength_trajectory_mpa": yield_hist,
            "final_yield_strength_mpa": float(yield_hist[-1]),
        }

    def predict_properties_from_history(
        self,
        base_yield_strength_mpa: float,
        base_youngs_modulus_gpa: float,
        history: ThermomechanicalHistoryParameters,
        lattice_friction_stress_mpa: Optional[float] = None,
    ) -> ThermomechanicalPropertyResponse:
        """Compute full physical property alterations conditioned on thermomechanical route via continuous ISVs."""
        route = history.route
        E_gpa = base_youngs_modulus_gpa
        E_pa = E_gpa * 1.0e9
        G_pa = E_pa / (2.0 * (1.0 + self.nu))
        
        # Friction stress sigma_0 (Peierls-Nabarro + solid solution baseline)
        sigma_0 = lattice_friction_stress_mpa or max(50.0, base_yield_strength_mpa * 0.70)

        # 1. Integrate Continuous Internal State Variables
        if route == ProcessingRoute.ANNEALED_RECRYSTALLIZED:
            d_um = 45.0
            rho_disl = 1.0e12
            f_v = 0.0005
            r_p_nm = 5.0
            sigma_res_mpa = 0.0
            void_frac = max(1e-5, history.void_volume_fraction)
            k_surf = 1.0
            n_exp = 0.28
            eps_u = 38.0
            eps_f = 52.0

        elif route == ProcessingRoute.COLD_WORKED_50PCT:
            strain_50 = 0.693  # ln(1 / (1 - 0.50))
            rho_disl = self.integrate_kocks_mecking_dislocation_density(
                rho_initial_m2=1e12,
                plastic_strain=strain_50,
                strain_rate_s_inv=history.strain_rate_s_inv,
                temperature_k=history.temperature_k,
            )
            rho_disl = max(8.0e14, min(2.5e15, rho_disl))
            d_um = 12.0
            f_v = 0.0005
            r_p_nm = 5.0
            sigma_res_mpa = 120.0
            void_frac = max(1e-4, history.void_volume_fraction * 2.0)
            k_surf = 0.90
            n_exp = 0.06
            eps_u = 3.0
            eps_f = 14.0

        elif route == ProcessingRoute.SOLUTION_TREATED_PEAK_AGED_T6:
            d_um = 30.0
            rho_disl = 4.0e13
            f_v = 0.18
            r_p_nm = 10.0
            sigma_res_mpa = 25.0
            void_frac = max(1e-5, history.void_volume_fraction)
            k_surf = 0.95
            n_exp = 0.14
            eps_u = 12.0
            eps_f = 22.0

        elif route == ProcessingRoute.ADDITIVE_LPBF_AS_PRINTED:
            d_um = 1.5
            rho_disl = 2.5e14
            f_v = 0.008
            r_p_nm = 3.0
            sigma_res_mpa = max(180.0, base_yield_strength_mpa * 0.40)
            void_frac = max(4e-4, history.void_volume_fraction * 4.0)
            k_surf = 0.52
            n_exp = 0.15
            eps_u = 22.0
            eps_f = 36.0

        elif route == ProcessingRoute.ADDITIVE_LPBF_HIP_AGED:
            d_um = self.integrate_grain_growth_and_drx(
                initial_grain_size_um=1.5,
                dislocation_density_m2=2.5e14,
                anneal_time_s=7200.0,
                anneal_temp_k=1423.15,
            )
            d_um = max(25.0, min(50.0, d_um))
            rho_disl = 5.0e12
            f_v = 0.020
            r_p_nm = 10.0
            sigma_res_mpa = 5.0
            void_frac = 1e-5
            k_surf = 0.96
            n_exp = 0.25
            eps_u = 32.0
            eps_f = 46.0

        else:
            # Custom ISV path integration
            cold_strain = history.prior_cold_work_strain
            rho_disl = self.integrate_kocks_mecking_dislocation_density(
                rho_initial_m2=1e12,
                plastic_strain=cold_strain,
                strain_rate_s_inv=history.strain_rate_s_inv,
                temperature_k=history.temperature_k,
            )
            t_anneal = history.anneal_temperature_k or history.temperature_k
            time_s = history.anneal_time_seconds or 3600.0
            d_um = self.integrate_grain_growth_and_drx(
                initial_grain_size_um=30.0,
                dislocation_density_m2=rho_disl,
                anneal_time_s=time_s,
                anneal_temp_k=t_anneal,
            )
            f_v = 0.01
            r_p_nm = 6.0
            sigma_res_mpa = history.residual_stress_mpa
            void_frac = history.void_volume_fraction
            k_surf = 0.95
            n_exp = max(0.05, 0.28 * (1.0 - min(1.0, cold_strain)))
            eps_u = max(2.0, 35.0 * (1.0 - min(0.9, cold_strain)))
            eps_f = max(5.0, 50.0 * (1.0 - min(0.8, cold_strain)))

        # 2. Physics-Based Strengthening Mechanisms
        # Hall-Petch grain boundary strengthening: k_HP / sqrt(d)
        k_hp_mpa_sqrt_um = 10.5
        delta_sigma_hp = k_hp_mpa_sqrt_um / np.sqrt(max(0.2, d_um))

        # Taylor dislocation forest hardening: Delta sigma_T = M * alpha * G * b * sqrt(rho)
        alpha_taylor = 0.28
        delta_sigma_taylor = (self.M * alpha_taylor * (G_pa * 1e-6) * self.b * np.sqrt(rho_disl))

        # Precipitate strengthening: particle shearing vs Orowan looping
        delta_sigma_orowan = self.compute_precipitate_strengthening(
            mean_radius_nm=r_p_nm,
            volume_fraction=f_v,
            g_shear_mpa=G_pa * 1e-6,
        )

        # Total Yield Strength
        sigma_y = float(sigma_0 + delta_sigma_hp + delta_sigma_taylor + delta_sigma_orowan)

        # 3. Plasticity, Work Hardening & Ultimate Tensile Strength
        K_work_hard = float(sigma_y * (1.0 + n_exp * 2.5))
        sigma_uts = float(sigma_y * (1.0 + (eps_u / 100.0) ** n_exp * 0.55))

        # 4. Fracture Toughness K_Ic & Plastic Zone Size
        gamma_surface_j_m2 = 2.2
        gamma_plastic_dissipation = 3200.0 * ((eps_f / 30.0) ** 1.6) * (600.0 / max(200.0, sigma_y))
        gamma_eff = gamma_surface_j_m2 * (1.0 + gamma_plastic_dissipation)

        k_ic_pa_sqrt_m = np.sqrt((2.0 * E_pa * gamma_eff) / max(0.1, 1.0 - self.nu**2))
        k_ic = float(np.clip(k_ic_pa_sqrt_m * 1.0e-6, 18.0, 220.0))

        delta_ctod_um = float((k_ic**2 * 1.0e6) / (1.5 * sigma_y * E_gpa * 1000.0) * 1.0e3)
        r_p_mm = float((1.0 / (6.0 * np.pi)) * ((k_ic / sigma_y) ** 2) * 1000.0)

        # 5. Cyclic Fatigue Parameters
        sigma_e_intrinsic = 0.45 * sigma_uts
        residual_reduction = max(0.20, 1.0 - (sigma_res_mpa / max(1.0, sigma_uts)))
        pore_reduction = max(0.20, 1.0 - (void_frac * 250.0))
        sigma_e = float(max(25.0, sigma_e_intrinsic * k_surf * pore_reduction * residual_reduction))

        sigma_f_prime = float(sigma_uts + 345.0)
        b_basquin = float(-np.log10(max(1.1, (2.0 * sigma_f_prime) / max(10.0, sigma_e))) / 6.0)
        b_basquin = float(np.clip(b_basquin, -0.16, -0.06))

        eps_f_prime = float(np.log(1.0 / max(0.1, 1.0 - (eps_f / 100.0) * 0.75)))
        c_coffin = float(-0.55 - (sigma_y / 4000.0) * 0.15)

        nt_cycles = float(0.5 * ((eps_f_prime * (E_gpa * 1000.0)) / max(1.0, sigma_f_prime)) ** (1.0 / (b_basquin - c_coffin)))
        nt_cycles = float(np.clip(nt_cycles, 50.0, 50000.0))

        m_paris = float(2.8 + (120.0 / max(20.0, k_ic)) * 0.5)
        c_paris = float(1.2e-11 * (80.0 / max(10.0, E_gpa)) ** 2)
        delta_k_th = float(np.clip(0.12 * k_ic, 2.0, 12.0))

        isv_vector = InternalStateVector(
            dislocation_density_m2=rho_disl,
            grain_size_um=d_um,
            precipitate_volume_fraction=f_v,
            mean_precipitate_radius_nm=r_p_nm,
            phase_fractions={"matrix": 1.0 - f_v, "precipitate": f_v},
        )

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
            internal_state_vector=isv_vector,
        )
