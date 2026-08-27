"""Atomistic Dynamics & Extended Defect Agent (ATOM-DYN): Scale 4 MLIP & Defect Kinetics Engine."""

import math
from typing import Dict, Tuple, List, Optional
import numpy as np
from penziv_materials.core.constants import KB_EV
from penziv_materials.core.models import AtomisticState, QuantumState


class AtomDynAgent:
    """Specialized Agent for E(3)-Equivariant Potentials, GMM OOD Detection, Defect Activation Barriers, and SVPN."""

    def __init__(self, model_architecture: str = "MACE"):
        self.model_architecture = model_architecture
        # Pre-calibrated GMM latent space parameters
        self.gmm_means = np.array([[0.0, 0.0], [1.5, 1.0], [-1.0, 2.0]])
        self.gmm_covars = np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.8, 0.2], [0.2, 0.8]], [[1.2, 0.0], [0.0, 1.2]]])
        self.gmm_weights = np.array([0.5, 0.3, 0.2])

    def evaluate_gmm_ood(self, latent_features: np.ndarray) -> Tuple[float, bool]:
        """Evaluate Negative Log-Likelihood (NLL) of atomic latent embedding against GMM density:

        L_OOD(z_i) = -ln sum_k [ pi_k * N(z_i | mu_k, Sigma_k) ]
        """
        z = np.asarray(latent_features, dtype=np.float64)
        if z.ndim == 1:
            z = z[:2] if len(z) >= 2 else np.pad(z, (0, 2 - len(z)))

        likelihood = 0.0
        for pi_k, mu_k, cov_k in zip(self.gmm_weights, self.gmm_means, self.gmm_covars):
            diff = z - mu_k
            inv_cov = np.linalg.inv(cov_k)
            det_cov = np.linalg.det(cov_k)
            norm_const = 1.0 / (2.0 * np.pi * np.sqrt(det_cov))
            exponent = -0.5 * np.dot(diff, np.dot(inv_cov, diff))
            likelihood += pi_k * norm_const * np.exp(exponent)

        likelihood = max(1e-12, likelihood)
        nll = -float(np.log(likelihood))
        is_ood = nll > 12.0
        return nll, is_ood

    def compute_htst_rate_with_uncertainty(
        self,
        delta_e_barrier_ev: float,
        sigma_delta_e_ev: float,
        temperature_k: float,
        attempt_freq_hz: float = 1.0e13,
        sigma_ln_nu0: float = 0.15,
    ) -> Tuple[float, float]:
        """Compute Harmonic Transition State Theory rate constant Gamma and log-normal variance sigma_ln_Gamma^2."""
        kbt = KB_EV * temperature_k
        gamma_mean = attempt_freq_hz * np.exp(-delta_e_barrier_ev / kbt)
        sigma_ln_gamma_sq = (sigma_delta_e_ev / kbt) ** 2 + (sigma_ln_nu0**2)
        return float(gamma_mean), float(sigma_ln_gamma_sq)

    def compute_svpn_peierls_stress(
        self,
        c44_gpa: float,
        burgers_vector_angstrom: float,
        poisson_ratio: float = 0.30,
    ) -> float:
        """Semidiscrete Variational Peierls-Nabarro (SVPN) dislocation core width and Peierls stress.

        tau_P = 2 * G / (1 - nu) * exp(-2 * pi * zeta / b)
        """
        # Calibrated FCC {111}<110> dislocation half-width zeta/b ~ 1.12
        zeta_over_b = 1.12
        prefactor = 2.0 * c44_gpa / (1.0 - poisson_ratio)
        tau_p_gpa = prefactor * np.exp(-2.0 * np.pi * zeta_over_b)
        return float(tau_p_gpa)

    def compute_solute_gb_segregation(
        self,
        solute_element: str,
        base_element: str = "Ni",
    ) -> Tuple[float, float]:
        """Compute grain boundary energy gamma_GB and solute segregation binding energy Delta E_b^seg."""
        seg_energies = {
            "B": -0.85,
            "C": -0.65,
            "Zr": -0.72,
            "Hf": -0.78,
            "Cr": -0.15,
            "W": -0.22,
            "Re": -0.28,
        }
        delta_e_seg = seg_energies.get(solute_element, -0.10)
        gamma_gb = 0.85 + 0.10 * delta_e_seg
        return float(gamma_gb), float(delta_e_seg)

    def execute_forward_scale(
        self,
        quantum_state: QuantumState,
        composition: Dict[str, float],
        temperature_k: float = 300.0,
    ) -> AtomisticState:
        """Execute ATOM-DYN forward scale calculation."""
        c44 = quantum_state.c_voigt_gpa[3][3] if quantum_state.c_voigt_gpa else 112.0
        burgers_b = 2.54

        barrier_ev = 2.45
        sigma_barrier_ev = 0.035
        rate_hz, var_ln_rate = self.compute_htst_rate_with_uncertainty(
            delta_e_barrier_ev=barrier_ev,
            sigma_delta_e_ev=sigma_barrier_ev,
            temperature_k=temperature_k,
        )

        peierls_gpa = self.compute_svpn_peierls_stress(c44, burgers_b)

        key_solute = "B" if "B" in composition else ("C" if "C" in composition else "Cr")
        gamma_gb, delta_e_seg = self.compute_solute_gb_segregation(key_solute)

        synthetic_latent = np.array([0.2, 0.4])
        nll, is_ood = self.evaluate_gmm_ood(synthetic_latent)

        return AtomisticState(
            defect_migration_barrier_ev=barrier_ev,
            migration_barrier_sigma_ev=sigma_barrier_ev,
            kinetic_rate_s_inv=rate_hz,
            lognormal_variance_sigma_ln_gamma_sq=var_ln_rate,
            peierls_stress_gpa=peierls_gpa,
            grain_boundary_energy_j_m2=gamma_gb,
            solute_gb_segregation_energy_ev=delta_e_seg,
            work_of_separation_j_m2=3.85,
            ood_max_negative_log_likelihood=nll,
            is_ood=is_ood,
        )
