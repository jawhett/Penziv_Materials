"""Handshake validation gates enforcing physical, numerical, economic, and toxicological consistency."""

import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from penziv_materials.core.constants import (
    TOL_FORCE_RESIDUAL_EV_ANG,
    TOL_OOD_GMM_NLL_DEFAULT,
    TOL_LOGNORMAL_RATE_VAR,
    TOL_RVE_STRESS_CONVERGENCE,
    TOL_COMPOUND_VARIANCE_BOUND,
    TOL_WASSERSTEIN_DISTANCE,
    TOL_KS_PVALUE_MIN,
)
from penziv_materials.core.models import (
    MaterialCandidate,
    ValidationReceipt,
    ValidationStatus,
)


class HandshakeGatekeeper:
    """Enforces zero-compromise physical consistency, EHS, distribution-matching, and error-bounding gates across the multiscale pyramid."""

    @staticmethod
    def now_iso() -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    @classmethod
    def validate_force_residual(cls, max_force_residual: float) -> ValidationReceipt:
        """Scale 5 <-> 4: Force Residual Gate: max_I ||F_I + grad_R E_tot|| < 1e-4 eV/A."""
        passed = max_force_residual < TOL_FORCE_RESIDUAL_EV_ANG
        return ValidationReceipt(
            gate_name="Scale 5-4: Ab Initio Force Residual Gate",
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            metric_value=max_force_residual,
            threshold=TOL_FORCE_RESIDUAL_EV_ANG,
            details=f"Max atomic force residual is {max_force_residual:.2e} eV/Å (threshold {TOL_FORCE_RESIDUAL_EV_ANG:.2e} eV/Å).",
            timestamp=cls.now_iso(),
        )

    @classmethod
    def validate_ood_density(cls, max_nll: float, threshold: float = TOL_OOD_GMM_NLL_DEFAULT) -> ValidationReceipt:
        """Scale 5 <-> 4: GMM / Ensemble OOD Density Gate."""
        is_ood = max_nll > threshold
        status = ValidationStatus.ROUTED_TO_HIGH_FIDELITY if is_ood else ValidationStatus.PASSED
        return ValidationReceipt(
            gate_name="Scale 4-5: Multi-Modal OOD Density Gate",
            status=status,
            metric_value=max_nll,
            threshold=threshold,
            details=(
                f"GMM negative log-likelihood NLL={max_nll:.2f} "
                f"({'OUT-OF-DISTRIBUTION: routed to Q-ELEC for single-point DFT' if is_ood else 'IN-DISTRIBUTION: verified for MLIP inference'})."
            ),
            timestamp=cls.now_iso(),
        )

    @classmethod
    def validate_stacking_fault_positivity(cls, min_gamma: float) -> ValidationReceipt:
        """Scale 4 <-> 3: Planar Fault Energy Gate supporting stable slip, TWIP, and TRIP martensitic transformation."""
        if min_gamma > 45.0:
            status = ValidationStatus.PASSED
            desc = f"Stable positive planar fault energy: {min_gamma:.2f} mJ/m² (dislocation glide regime)."
        elif min_gamma > 0.0:
            status = ValidationStatus.PASSED
            desc = f"TRIP/TWIP low positive planar fault regime: {min_gamma:.2f} mJ/m² (martensitic transformation and mechanical twinning driver)."
        else:
            status = ValidationStatus.FAILED
            desc = f"Thermodynamically unstable planar fault: {min_gamma:.2f} mJ/m² (barrierless spontaneous shear transformation of parent phase)."

        return ValidationReceipt(
            gate_name="Scale 4-3: Planar Fault Energy Gate",
            status=status,
            metric_value=min_gamma,
            threshold=0.0,
            details=desc,
            timestamp=cls.now_iso(),
        )

    @classmethod
    def validate_lognormal_rate_variance(cls, sigma_ln_gamma_sq: float) -> ValidationReceipt:
        """Scale 4 <-> 3: Log-Normal Kinetic Rate Variance Gate: sigma_ln_gamma^2 < 0.25."""
        passed = sigma_ln_gamma_sq < TOL_LOGNORMAL_RATE_VAR
        return ValidationReceipt(
            gate_name="Scale 4-3: Log-Normal Kinetic Rate Variance Gate",
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            metric_value=sigma_ln_gamma_sq,
            threshold=TOL_LOGNORMAL_RATE_VAR,
            details=f"Kinetic rate variance σ_lnΓ² = {sigma_ln_gamma_sq:.4f} (threshold {TOL_LOGNORMAL_RATE_VAR}).",
            timestamp=cls.now_iso(),
        )

    @classmethod
    def validate_rve_convergence(cls, relative_error: float) -> ValidationReceipt:
        """Scale 3 <-> 2: RVE Stress Convergence Gate: ||<sigma_2L> - <sigma_L>|| < 0.015."""
        passed = relative_error < TOL_RVE_STRESS_CONVERGENCE
        return ValidationReceipt(
            gate_name="Scale 3-2: RVE Homogenization Convergence Gate",
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            metric_value=relative_error,
            threshold=TOL_RVE_STRESS_CONVERGENCE,
            details=f"RVE domain doubling error is {relative_error:.4f} (threshold {TOL_RVE_STRESS_CONVERGENCE}).",
            timestamp=cls.now_iso(),
        )

    @classmethod
    def validate_clausius_duhem_dissipation(cls, dissipation_rate: float) -> ValidationReceipt:
        """Scale 2 <-> 1: Clausius-Duhem Dissipation Positivity: D_int >= 0."""
        passed = dissipation_rate >= 0.0
        return ValidationReceipt(
            gate_name="Scale 2-1: Clausius-Duhem Dissipation Positivity Gate",
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            metric_value=dissipation_rate,
            threshold=0.0,
            details=f"Internal plastic dissipation rate is {dissipation_rate:.4e} ({'Thermodynamically admissible D_int >= 0' if passed else 'Violates 2nd Law of Thermodynamics'}).",
            timestamp=cls.now_iso(),
        )

    @classmethod
    def validate_compound_variance_bound(cls, compound_variance_ratio: float) -> ValidationReceipt:
        """Meta-Scale: Total compound scale variance sigma_tot^2 / mu^2 < 0.15."""
        passed = compound_variance_ratio < TOL_COMPOUND_VARIANCE_BOUND
        return ValidationReceipt(
            gate_name="Meta-Scale: Compound Scale Uncertainty Error Bounding Gate",
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            metric_value=compound_variance_ratio,
            threshold=TOL_COMPOUND_VARIANCE_BOUND,
            details=f"Compound multiscale uncertainty ratio σ_tot²/μ² = {compound_variance_ratio:.4f} (threshold {TOL_COMPOUND_VARIANCE_BOUND}).",
            timestamp=cls.now_iso(),
        )

    @classmethod
    def validate_toxicity_and_banned_species(
        cls,
        banned_elements: List[str],
        epa_hazard_score: float,
        is_encapsulated_electronic: bool = False,
    ) -> ValidationReceipt:
        """EHS Gate: flags toxic heavy metals with context-aware exemptions for hermetically encapsulated optoelectronics/semiconductors."""
        if is_encapsulated_electronic:
            passed = epa_hazard_score < 7.0
            details = f"Encapsulated industrial semiconductor/thermoelectric: EPA Score {epa_hazard_score:.2f}."
        else:
            passed = len(banned_elements) == 0 and epa_hazard_score < 4.5
            details = (
                f"EPA CompTox hazard score is {epa_hazard_score:.2f}. "
                + (f"Banned species detected: {', '.join(banned_elements)}." if banned_elements else "No restricted toxic elements.")
            )

        return ValidationReceipt(
            gate_name="Pre-Compute: Toxicity & Banned Species Gate",
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            metric_value=epa_hazard_score,
            threshold=4.5,
            details=details,
            timestamp=cls.now_iso(),
        )

    @classmethod
    def validate_supply_chain_resilience(cls, weighted_hhi_refining: float) -> ValidationReceipt:
        """Economic Gate: Flags extreme geopolitical refining concentration (HHI > 6000)."""
        passed = weighted_hhi_refining < 6000.0
        return ValidationReceipt(
            gate_name="Economic: Supply Chain & Geopolitical Resilience Gate",
            status=ValidationStatus.PASSED if passed else ValidationStatus.WARNING,
            metric_value=weighted_hhi_refining,
            threshold=6000.0,
            details=f"Refining HHI = {weighted_hhi_refining:.0f} ({'Resilient' if passed else 'Extreme supply chain disruption risk'}).",
            timestamp=cls.now_iso(),
        )

    @classmethod
    def validate_distribution_matching(
        cls,
        predicted_samples: np.ndarray,
        experimental_samples: np.ndarray,
        max_wasserstein_distance: float = TOL_WASSERSTEIN_DISTANCE,
        min_ks_pvalue: float = TOL_KS_PVALUE_MIN,
        property_name: str = "Yield Strength",
    ) -> ValidationReceipt:
        """Handshake Distribution Gate: Validates predicted property distribution against experimental distributions via Wasserstein-1 distance and Kolmogorov-Smirnov 2-sample testing."""
        from scipy.stats import wasserstein_distance, ks_2samp
        pred_arr = np.asarray(predicted_samples, dtype=np.float64)
        exp_arr = np.asarray(experimental_samples, dtype=np.float64)

        # Normalize distributions by pooled standard deviation to compute scale-invariant Wasserstein distance
        pool_scale = max(1e-3, float(np.std(exp_arr)))
        w1_dist = float(wasserstein_distance(pred_arr, exp_arr) / pool_scale)
        ks_res = ks_2samp(pred_arr, exp_arr)
        ks_stat = float(ks_res.statistic)
        ks_pvalue = float(ks_res.pvalue)

        passed = bool(w1_dist <= max_wasserstein_distance or ks_pvalue >= min_ks_pvalue)

        return ValidationReceipt(
            gate_name=f"Process-Benchmarking: {property_name} Distribution Matching Gate",
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            metric_value=w1_dist,
            threshold=max_wasserstein_distance,
            details=(
                f"Normalized Wasserstein distance W_1 = {w1_dist:.4f} (threshold {max_wasserstein_distance:.4f}), "
                f"KS-statistic D = {ks_stat:.4f}, p-value = {ks_pvalue:.4f} (min {min_ks_pvalue:.4f})."
            ),
            timestamp=cls.now_iso(),
        )

    @classmethod
    def sample_process_uncertainty_monte_carlo(
        cls,
        mean_temperature_history_k: np.ndarray,
        std_temperature_history_k: np.ndarray,
        mean_strain_rate_s_inv: np.ndarray,
        std_strain_rate_s_inv: np.ndarray,
        time_series_s: np.ndarray,
        num_samples: int = 30,
        eval_fn: Optional[Any] = None,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """Monte Carlo sampling across the processing uncertainty envelope T(t) ~ N(mu_T, sigma_T^2) and eps_dot(t) ~ N(mu_eps, sigma_eps^2)."""
        rng = np.random.default_rng(random_seed)
        times = np.asarray(time_series_s, dtype=np.float64)

        mu_t = np.asarray(mean_temperature_history_k, dtype=np.float64)
        sigma_t = np.asarray(std_temperature_history_k, dtype=np.float64)
        mu_eps = np.asarray(mean_strain_rate_s_inv, dtype=np.float64)
        sigma_eps = np.asarray(std_strain_rate_s_inv, dtype=np.float64)

        sampled_properties = []
        for _ in range(num_samples):
            sampled_t = np.maximum(200.0, rng.normal(mu_t, np.maximum(1e-3, sigma_t)))
            sampled_eps = np.maximum(0.0, rng.normal(mu_eps, np.maximum(1e-6, sigma_eps)))
            if eval_fn is not None:
                val = float(eval_fn(times, sampled_t, sampled_eps))
            else:
                from penziv_materials.scale1_process.thermomechanical_history import ThermomechanicalHistoryEngine
                eng = ThermomechanicalHistoryEngine()
                res = eng.integrate_continuous_isv_trajectory(
                    time_series_s=times,
                    temperature_series_k=sampled_t,
                    strain_rate_series_s_inv=sampled_eps,
                )
                val = float(res["final_yield_strength_mpa"])
            sampled_properties.append(val)

        samples_arr = np.array(sampled_properties, dtype=np.float64)
        return {
            "num_samples": num_samples,
            "sampled_property_values": samples_arr.tolist(),
            "mean": float(np.mean(samples_arr)),
            "std": float(np.std(samples_arr)),
            "p10": float(np.percentile(samples_arr, 10)),
            "p50": float(np.percentile(samples_arr, 50)),
            "p90": float(np.percentile(samples_arr, 90)),
        }

    @classmethod
    def validate_candidate(cls, candidate: MaterialCandidate) -> List[ValidationReceipt]:
        """Execute full suite of physical, thermodynamic, and numerical scale handshake gates for a candidate."""
        receipts = []

        if candidate.quantum:
            receipts.append(cls.validate_force_residual(candidate.quantum.max_force_residual_ev_ang))
            receipts.append(cls.validate_stacking_fault_positivity(candidate.quantum.sro_stacking_fault_energy_mj_m2))

        if candidate.atomistic:
            receipts.append(cls.validate_ood_density(candidate.atomistic.ood_max_negative_log_likelihood))
            receipts.append(cls.validate_lognormal_rate_variance(candidate.atomistic.lognormal_variance_sigma_ln_gamma_sq))

        if candidate.mesoscale:
            receipts.append(cls.validate_rve_convergence(candidate.mesoscale.rve_mesh_convergence_error))

        if candidate.continuum:
            receipts.append(cls.validate_clausius_duhem_dissipation(candidate.continuum.clausius_duhem_dissipation_w_m3))

        if candidate.assimilation:
            receipts.append(cls.validate_compound_variance_bound(candidate.assimilation.compound_variance_ratio))

        return receipts
