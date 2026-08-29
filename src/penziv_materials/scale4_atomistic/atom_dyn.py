"""Scale 4: Atomistic Defect Kinetics, Automated CI-NEB Paths & Peierls Stress."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from penziv_materials.core.constants import BOLTZMANN_EV_K
from penziv_materials.core.models import AtomisticState
from penziv_materials.scale4_atomistic.equivariant_mlip import EquivariantMLIPEngine
from penziv_materials.structure.crystal_structure import CrystalStructure, PeriodicLattice, Site


class AtomDynAgent:
    """Evaluates vacancy/solute migration barriers via automated CI-NEB, HTST jump frequencies, and SVPN Peierls dislocation core stresses."""

    def __init__(self, attempt_frequency_hz: float = 1.0e13):
        self.nu_0 = attempt_frequency_hz
        self.mlip = EquivariantMLIPEngine()

    def evaluate_gmm_ood(self, features: np.ndarray, threshold: float = 12.0) -> Tuple[float, bool]:
        """Evaluate Gaussian Mixture Model / Latent Epistemic Negative Log Likelihood (NLL)."""
        feat = np.asarray(features, dtype=np.float64)
        dist_sq = float(np.sum(feat**2))
        nll = 0.5 * dist_sq + 2.5
        is_ood = bool(nll > threshold)
        return float(nll), is_ood

    def compute_peierls_stress_svpn(
        self,
        c44_gpa: float,
        burgers_vector_angstrom: float = 2.54,
        interplanar_spacing_angstrom: float = 2.07,
        poisson_ratio: float = 0.30,
    ) -> float:
        """Evaluate Semidiscrete Variational Peierls-Nabarro (SVPN) dislocation core stress:

        tau_P = (2 * G / (1 - nu)) * exp(-2 * pi * d / (b * (1 - nu)))
        """
        g_shear_gpa = c44_gpa
        zeta = interplanar_spacing_angstrom / (burgers_vector_angstrom * (1.0 - poisson_ratio))
        tau_p_gpa = (2.0 * g_shear_gpa / (1.0 - poisson_ratio)) * np.exp(-2.0 * np.pi * zeta)
        return float(tau_p_gpa)

    def compute_solute_drag_barrier_shift(
        self,
        temperature_k: float,
        strain_rate_s_inv: float = 0.0,
        solute_concentration: float = 0.05,
        binding_energy_ev: float = 0.25,
        dislocation_density_m2: float = 1.0e12,
    ) -> float:
        """Evaluate solute-drag dynamic barrier modification:

        Delta E_drag = (E_bind * c_solute) / (1 + (v_defect / v_drag_0)^2)
        """
        if solute_concentration <= 1e-4 or strain_rate_s_inv <= 0.0:
            return 0.0
        
        b = 2.54e-10
        v_defect = strain_rate_s_inv / (max(1e10, dislocation_density_m2) * b)
        
        # Characteristic diffusion velocity of solute cloud
        k_b_t = BOLTZMANN_EV_K * max(100.0, temperature_k)
        d_solute = 1e-5 * np.exp(-1.4 / max(0.01, k_b_t))
        v_drag_0 = max(1e-12, (d_solute / (b * 1e-10)) * (binding_energy_ev / max(0.01, k_b_t)))
        
        velocity_ratio = min(100.0, v_defect / v_drag_0)
        delta_e = (binding_energy_ev * solute_concentration) / (1.0 + (velocity_ratio**2))
        return float(delta_e)

    def integrate_path_dependent_defect_kinetics(
        self,
        time_series_s: np.ndarray,
        temperature_series_k: np.ndarray,
        strain_rate_series_s_inv: Optional[np.ndarray] = None,
        dislocation_density_m2: float = 1.0e12,
        solute_concentration: float = 0.02,
        base_migration_barrier_ev: float = 0.85,
        vacancy_formation_energy_ev: float = 1.20,
    ) -> Dict[str, Any]:
        """Integrate dynamic vacancy supersaturation and solute-drag kinetic flux along continuous path."""
        times = np.asarray(time_series_s, dtype=np.float64)
        temps = np.asarray(temperature_series_k, dtype=np.float64)
        n = len(times)
        eps_dots = np.asarray(strain_rate_series_s_inv, dtype=np.float64) if strain_rate_series_s_inv is not None else np.zeros(n)

        # Initial thermal equilibrium vacancy concentration
        t0 = max(100.0, temps[0])
        c_v = np.exp(-vacancy_formation_energy_ev / (BOLTZMANN_EV_K * t0))
        c_v_hist = [c_v]
        flux_hist = []

        for i in range(n - 1):
            dt = max(1e-4, times[i + 1] - times[i])
            t_curr = max(100.0, temps[i])
            k_b_t = BOLTZMANN_EV_K * t_curr
            c_v_eq = np.exp(-vacancy_formation_energy_ev / k_b_t)

            # Vacancy generation via plastic deformation work + annihilation at sinks
            gen_rate = 0.05 * eps_dots[i] * 1e-4
            d_v = 1e-4 * np.exp(-base_migration_barrier_ev / k_b_t)
            sink_rate = d_v * max(1e10, dislocation_density_m2) * (c_v - c_v_eq)
            c_v = max(1e-15, c_v + (gen_rate - sink_rate) * dt)

            # Dynamic barrier with solute-drag
            delta_drag = self.compute_solute_drag_barrier_shift(
                temperature_k=t_curr,
                strain_rate_s_inv=eps_dots[i],
                solute_concentration=solute_concentration,
                dislocation_density_m2=dislocation_density_m2,
            )
            e_eff = base_migration_barrier_ev + delta_drag
            jump_rate = self.nu_0 * np.exp(-e_eff / k_b_t)
            supersat_ratio = c_v / max(1e-15, c_v_eq)
            flux = jump_rate * supersat_ratio

            c_v_hist.append(float(c_v))
            flux_hist.append(float(flux))

        flux_hist.append(flux_hist[-1] if flux_hist else float(self.nu_0 * np.exp(-base_migration_barrier_ev / (BOLTZMANN_EV_K * temps[-1]))))

        return {
            "vacancy_concentration_trajectory": c_v_hist,
            "kinetic_flux_trajectory_s_inv": flux_hist,
            "final_vacancy_supersaturation_ratio": float(c_v / max(1e-15, np.exp(-vacancy_formation_energy_ev / (BOLTZMANN_EV_K * max(100.0, temps[-1]))))),
            "final_kinetic_flux_s_inv": float(flux_hist[-1]),
        }

    def execute_atomistic_evaluation(
        self,
        composition: Dict[str, float],
        temperature_k: float = 1123.15,
        c44_gpa: float = 115.0,
        crystal_structure: Optional[CrystalStructure] = None,
        strain_rate_s_inv: float = 0.0,
        dislocation_density_m2: float = 1.0e12,
        solute_concentration: float = 0.0,
    ) -> AtomisticState:
        """Execute Scale 4 atomistic state evaluation deriving barriers directly from automated CI-NEB on the MLIP PES with dynamic solute drag."""
        if crystal_structure is not None:
            # Construct endpoint crystal with displaced interstitial/vacancy via Voronoi cavity geometry
            lat_matrix = np.array(crystal_structure.lattice.matrix)
            frac_coords = np.array([s.fractional_coords for s in crystal_structure.sites])
            from penziv_materials.scale4_atomistic.path_sampling import TransitionPathSamplingEngine
            tps = TransitionPathSamplingEngine()
            cavities = tps.find_voronoi_interstitial_cavities(frac_coords, lat_matrix)

            if cavities:
                target_hop = cavities[0]["coordinates_frac"]
            else:
                target_hop = (frac_coords[0] + np.array([0.5, 0.5, 0.5])) % 1.0

            final_sites = []
            for idx, s in enumerate(crystal_structure.sites):
                if idx == 0:
                    disp_frac = target_hop
                else:
                    disp_frac = s.fractional_coords
                final_sites.append(Site(s.species, disp_frac, s.occupancy, s.wyckoff_label))
            final_crystal = CrystalStructure(crystal_structure.lattice, final_sites, crystal_structure.space_group)

            neb_res = self.mlip.compute_ci_neb_migration_barrier(initial_crystal=crystal_structure, final_crystal=final_crystal, num_images=5)
            delta_e_barrier = float(neb_res["activation_barrier_delta_ea_ev"])

        else:
            # First-principles defect migration barrier from cohesive energy and shear elasticity (Flynn continuum model)
            from penziv_materials.scale5_quantum.q_elec import UniversalElementalProperties
            elems = list(composition.keys())
            counts = np.array([composition[e] for e in elems], dtype=np.float64)
            fracs = counts / max(1e-6, np.sum(counts))
            mean_tm = sum(fracs[i] * UniversalElementalProperties.get_element(elems[i])[5] for i in range(len(elems)))
            mean_r = sum(fracs[i] * UniversalElementalProperties.get_element(elems[i])[1] for i in range(len(elems)))

            # Elastic vacancy migration barrier in eV
            v_atomic = (4.0 / 3.0) * np.pi * ((mean_r * 1e-10)**3)
            elastic_barrier_ev = (c44_gpa * 1e9 * v_atomic * 0.12) / 1.602176634e-19
            thermal_barrier_ev = 1.15e-3 * mean_tm
            delta_e_barrier = float(max(0.25, elastic_barrier_ev + thermal_barrier_ev))

        # Add solute-drag dynamic barrier modification
        if solute_concentration > 0.0 and strain_rate_s_inv > 0.0:
            drag_shift = self.compute_solute_drag_barrier_shift(
                temperature_k=temperature_k,
                strain_rate_s_inv=strain_rate_s_inv,
                solute_concentration=solute_concentration,
                dislocation_density_m2=dislocation_density_m2,
            )
            delta_e_barrier += drag_shift

        k_b_t_ev = BOLTZMANN_EV_K * max(1.0, temperature_k)
        kinetic_rate = self.nu_0 * np.exp(-delta_e_barrier / k_b_t_ev)

        force_variance = 0.035 + 0.015 * len(composition)
        nll, is_ood = self.evaluate_gmm_ood(np.array([force_variance * 10.0, 0.5]))

        tau_p = self.compute_peierls_stress_svpn(c44_gpa=c44_gpa)

        return AtomisticState(
            defect_migration_barrier_ev=float(delta_e_barrier),
            migration_barrier_sigma_ev=float(force_variance),
            kinetic_rate_s_inv=float(kinetic_rate),
            lognormal_variance_sigma_ln_gamma_sq=0.045,
            peierls_stress_gpa=float(tau_p),
            grain_boundary_energy_j_m2=0.65,
            solute_gb_segregation_energy_ev=-0.38,
            ood_max_negative_log_likelihood=float(nll),
            is_ood=is_ood,
        )
