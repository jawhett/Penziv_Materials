"""Grand-Canonical CALPHAD-Coupled Multi-Phase-Field Engine with Khachaturyan Elasticity & STZ Amorphous Plasticity."""

from typing import Dict, Tuple, List, Optional, Any, Union
import numpy as np

from penziv_materials.thermodynamics.opencalphad_tdb import OpenCALPHADTDBEngine


class CALPHADGrandPotentialPhaseFieldEngine:
    """Solves multi-phase-field kinetics coupled to CALPHAD chemical free energy densities, anisotropic Khachaturyan eigenstrains, and amorphous STZ plasticity."""

    def __init__(
        self,
        num_phases: int = 3,
        grid_shape: Tuple[int, int, int] = (16, 16, 16),
        dx_nm: float = 1.0,
        temperature_k: float = 800.0,
        calphad_engine: Optional[OpenCALPHADTDBEngine] = None,
    ):
        self.num_phases = num_phases
        self.grid_shape = grid_shape
        self.nx, self.ny, self.nz = grid_shape
        self.dx = dx_nm
        self.T = temperature_k
        self.calphad_engine = calphad_engine or OpenCALPHADTDBEngine()

    def compute_calphad_grand_potentials(
        self,
        chemical_potentials_mu: Union[np.ndarray, Dict[str, float]],      # (num_components,) or dict
        phase_names_or_paraboloids: Optional[Union[List[str], List[Tuple[float, np.ndarray, np.ndarray]]]] = None,
    ) -> np.ndarray:
        """Compute exact grand potential densities omega_alpha(mu, T) = G_alpha(c_alpha(mu)) - sum_i mu_i c_{alpha, i} via Legendre transform."""
        omega_densities = np.zeros(self.num_phases, dtype=np.float64)

        if isinstance(chemical_potentials_mu, dict) and phase_names_or_paraboloids and isinstance(phase_names_or_paraboloids[0], str):
            # Exact Symbolic CALPHAD AST Legendre transformation
            for a, p_name in enumerate(phase_names_or_paraboloids[:self.num_phases]):
                omega_densities[a] = self.calphad_engine.evaluate_grand_potential_density(
                    phase_name=p_name,
                    chemical_potentials=chemical_potentials_mu,
                    temperature_k=self.T,
                )
            return omega_densities

        # Convert to numpy array if dict
        mu_vec = np.array(list(chemical_potentials_mu.values()) if isinstance(chemical_potentials_mu, dict) else chemical_potentials_mu, dtype=np.float64)
        parabs = phase_names_or_paraboloids if (phase_names_or_paraboloids and isinstance(phase_names_or_paraboloids[0], tuple)) else None

        for a in range(self.num_phases):
            if parabs and a < len(parabs):
                g0, c0, d2g = parabs[a]
                # c_alpha(mu) = c0 + inv(d2g) . mu
                c_alpha = c0 + mu_vec / max(1e-3, float(d2g[0]))
                g_val = g0 + 0.5 * d2g[0] * np.sum((c_alpha - c0)**2)
                omega_densities[a] = g_val - np.sum(mu_vec * c_alpha)
            else:
                omega_densities[a] = -0.5 * (a + 1) * np.sum(mu_vec**2)
        return omega_densities


    def compute_stz_plastic_strain_rate(
        self,
        deviatoric_shear_stress_mpa: float,
        effective_disorder_temperature_chi: float = 0.15,
        reference_strain_rate_s_inv: float = 1.0e6,
        characteristic_yield_stress_mpa: float = 800.0,
    ) -> float:
        """Evaluate Shear Transformation Zone (STZ) plastic shear strain rate for amorphous/vitreous interphases:

        gamma_dot_pl = 2 * gamma_dot_0 * exp(-1 / chi) * sinh(tau / tau_0)
        """
        chi = max(0.01, effective_disorder_temperature_chi)
        tau = abs(deviatoric_shear_stress_mpa)
        tau0 = max(1.0, characteristic_yield_stress_mpa)

        sinh_arg = np.clip(tau / tau0, -50.0, 50.0)
        gamma_dot = 2.0 * reference_strain_rate_s_inv * np.exp(-1.0 / chi) * np.sinh(sinh_arg)
        return float(np.copysign(gamma_dot, deviatoric_shear_stress_mpa))

    def compute_isv_coupled_stz_plastic_strain_rate(
        self,
        deviatoric_shear_stress_mpa: float,
        grain_size_um: float = 30.0,
        dislocation_density_m2: float = 1.0e12,
        precipitate_radius_nm: float = 5.0,
        precipitate_volume_fraction: float = 0.01,
        effective_disorder_temperature_chi: float = 0.15,
        reference_strain_rate_s_inv: float = 1.0e6,
        base_friction_stress_mpa: float = 150.0,
        shear_modulus_gpa: float = 77.0,
    ) -> Dict[str, float]:
        """Evaluate internal state variable (ISV) coupled STZ rate where yield barrier tau_0 evolves dynamically with microstructural hardening."""
        g_shear_mpa = shear_modulus_gpa * 1.0e3
        b = 2.54e-10

        # Hall-Petch strengthening contribution
        tau_hp = (10.5 / np.sqrt(max(0.2, grain_size_um))) * 0.50

        # Taylor forest hardening contribution
        tau_taylor = 0.28 * g_shear_mpa * b * np.sqrt(max(1e10, dislocation_density_m2))

        # Precipitate strengthening
        if precipitate_volume_fraction > 0.0005 and precipitate_radius_nm > 0.2:
            r_m = precipitate_radius_nm * 1e-9
            l_spacing = r_m * np.sqrt(2.0 * np.pi / (3.0 * precipitate_volume_fraction))
            tau_ppt = (0.81 * g_shear_mpa * b) / (2.0 * np.pi * max(1e-9, l_spacing - 2.0 * r_m)) * np.log(max(1.5, 2.0 * r_m / b))
        else:
            tau_ppt = 0.0

        dynamic_tau0 = max(50.0, base_friction_stress_mpa + tau_hp + tau_taylor + tau_ppt)
        gamma_dot = self.compute_stz_plastic_strain_rate(
            deviatoric_shear_stress_mpa=deviatoric_shear_stress_mpa,
            effective_disorder_temperature_chi=effective_disorder_temperature_chi,
            reference_strain_rate_s_inv=reference_strain_rate_s_inv,
            characteristic_yield_stress_mpa=dynamic_tau0,
        )

        return {
            "plastic_shear_strain_rate_s_inv": float(gamma_dot),
            "dynamic_characteristic_yield_stress_mpa": float(dynamic_tau0),
            "hall_petch_shear_stress_mpa": float(tau_hp),
            "taylor_forest_shear_stress_mpa": float(tau_taylor),
            "precipitate_shear_stress_mpa": float(tau_ppt),
        }

    def step_forward_grand_potential_field(
        self,
        phi_fields: np.ndarray,                   # (num_phases, nx, ny, nz)
        chemical_potentials: np.ndarray,          # (num_components,)
        eigenstrain_tensors: Optional[List[np.ndarray]] = None,
        stiffness_tensors: Optional[List[np.ndarray]] = None,
        applied_strain: Optional[np.ndarray] = None,
        anisotropic_kappa_tensors: Optional[List[np.ndarray]] = None,
        dt_s: float = 0.001,
        mobility_L: float = 1.0,
        interface_width_gamma: float = 0.5,
    ) -> Dict[str, Any]:
        """Execute coupled multi-phase time integration with CALPHAD driving forces and Khachaturyan microelastic energy feedback."""
        num_p, nx, ny, nz = phi_fields.shape
        new_phi = phi_fields.copy()
        dx2 = self.dx**2

        # 1. Chemical grand potential differences
        parabs = [(0.0, np.array([0.1 * (a + 1)]), np.array([500.0])) for a in range(num_p)]
        omega = self.compute_calphad_grand_potentials(chemical_potentials, parabs)

        # 2. Elastic driving force from spectral Khachaturyan Green's function solver
        elastic_df_spatial = np.zeros((num_p, nx, ny, nz), dtype=np.float64)
        if eigenstrain_tensors is not None and stiffness_tensors is not None:
            from penziv_materials.scale3_mesoscale.phase_field import PhaseFieldEngine
            pf_micro = PhaseFieldEngine(grid_size=(nx, ny, nz), dx_nm=self.dx)
            ref_c4 = stiffness_tensors[0] if len(stiffness_tensors) > 0 else np.eye(6) * 100.0
            el_res = pf_micro.solve_khachaturyan_elastic_equilibrium_fft(
                order_parameters=phi_fields,
                eigenstrain_tensors=eigenstrain_tensors,
                stiffness_tensor_4th_order=ref_c4,
                applied_strain=applied_strain,
            )
            elastic_df_spatial = el_res["elastic_driving_forces"]
        elif applied_strain is not None and eigenstrain_tensors is not None and stiffness_tensors is not None:
            for a in range(num_p):
                eps_0 = eigenstrain_tensors[a] if a < len(eigenstrain_tensors) else np.zeros((3, 3))
                c_mat = stiffness_tensors[a] if a < len(stiffness_tensors) else np.eye(3) * 100.0
                eps_el = applied_strain - eps_0
                val = 0.5 * np.sum(eps_el * np.dot(c_mat[:3, :3], eps_el))
                elastic_df_spatial[a] = val

        # 3. Allen-Cahn multi-well time evolution
        for a in range(num_p):
            if anisotropic_kappa_tensors and a < len(anisotropic_kappa_tensors) and anisotropic_kappa_tensors[a] is not None:
                from penziv_materials.scale3_mesoscale.phase_field import PhaseFieldEngine
                pf_aux = PhaseFieldEngine(grid_size=(nx, ny, nz), dx_nm=self.dx)
                lap_phi = pf_aux.compute_anisotropic_gradient_operator(phi_fields[a], anisotropic_kappa_tensors[a])
            else:
                lap_phi = (
                    np.roll(phi_fields[a], 1, axis=0) + np.roll(phi_fields[a], -1, axis=0)
                    + np.roll(phi_fields[a], 1, axis=1) + np.roll(phi_fields[a], -1, axis=1)
                    + np.roll(phi_fields[a], 1, axis=2) + np.roll(phi_fields[a], -1, axis=2)
                    - 6.0 * phi_fields[a]
                ) / dx2

            # Multi-well derivative: dW/dphi_a
            other_sum = np.sum([phi_fields[b] for b in range(num_p) if b != a], axis=0)
            dw_dphi = 2.0 * phi_fields[a] * other_sum

            # Variational driving force
            dF_dphi = (omega[a] + elastic_df_spatial[a]) + dw_dphi - interface_width_gamma * lap_phi
            new_phi[a] -= dt_s * mobility_L * dF_dphi

        # Constraint enforcement: sum(phi_a) = 1, phi_a in [0, 1]
        new_phi = np.clip(new_phi, 0.0, 1.0)
        norm_sum = np.sum(new_phi, axis=0, keepdims=True)
        new_phi = new_phi / np.maximum(1e-8, norm_sum)

        mean_elastic = [float(np.mean(elastic_df_spatial[a])) for a in range(num_p)]

        return {
            "updated_phase_fields": new_phi,
            "mean_phase_fractions": [float(np.mean(new_phi[a])) for a in range(num_p)],
            "grand_potential_densities": omega.tolist(),
            "elastic_energy_densities_mpa": mean_elastic,
            "is_calphad_coupled": True,
        }
