"""Handshake validation gates enforcing physical and numerical consistency across all scale boundaries."""

import datetime
from typing import List, Dict, Any
import numpy as np
from penziv_materials.core.constants import (
    TOL_FORCE_RESIDUAL_EV_ANG,
    TOL_OOD_GMM_NLL_DEFAULT,
    TOL_LOGNORMAL_RATE_VAR,
    TOL_RVE_STRESS_CONVERGENCE,
    TOL_COMPOUND_VARIANCE_BOUND,
)
from penziv_materials.core.models import (
    MaterialCandidate,
    ValidationReceipt,
    ValidationStatus,
)


class HandshakeGatekeeper:
    """Enforces zero-compromise physical consistency and error-bounding gates across the multiscale pyramid."""

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
        """Scale 4 <-> 3: Stacking Fault Positivity Gate: min gamma(u, alpha_SRO) > 0 for all u != 0."""
        passed = min_gamma > 0.0
        return ValidationReceipt(
            gate_name="Scale 4-3: Planar Fault Energy Positivity Gate",
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            metric_value=min_gamma,
            threshold=0.0,
            details=f"Minimum non-zero stacking fault energy is {min_gamma:.2f} mJ/m² ({'Positive' if passed else 'Unphysical negative planar fault'}).",
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
            gate_name="Scale 2-1: Thermodynamic Clausius-Duhem Dissipation Gate",
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            metric_value=dissipation_rate,
            threshold=0.0,
            details=f"Internal plastic dissipation D_int = {dissipation_rate:.2e} W/m³ ({'Thermodynamically admissible' if passed else 'Violates 2nd law'}).",
            timestamp=cls.now_iso(),
        )

    @classmethod
    def validate_compound_variance(cls, compound_variance_ratio: float) -> ValidationReceipt:
        """Meta-Scale: Compound Scale Variance Bound: sigma_tot^2 / mu^2 < 0.15."""
        passed = compound_variance_ratio < TOL_COMPOUND_VARIANCE_BOUND
        return ValidationReceipt(
            gate_name="Meta-Scale: Compound Uncertainty Error Bounding Gate",
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            metric_value=compound_variance_ratio,
            threshold=TOL_COMPOUND_VARIANCE_BOUND,
            details=f"Compound variance ratio is {compound_variance_ratio:.4f} (threshold {TOL_COMPOUND_VARIANCE_BOUND}).",
            timestamp=cls.now_iso(),
        )

    @classmethod
    def validate_candidate(cls, candidate: MaterialCandidate) -> List[ValidationReceipt]:
        """Execute full handshake validation suite on a material candidate."""
        receipts: List[ValidationReceipt] = []

        if candidate.atomistic:
            receipts.append(cls.validate_ood_density(candidate.atomistic.ood_max_negative_log_likelihood))
            receipts.append(cls.validate_lognormal_rate_variance(candidate.atomistic.lognormal_variance_sigma_ln_gamma_sq))

        if candidate.quantum:
            receipts.append(cls.validate_stacking_fault_positivity(candidate.quantum.sro_stacking_fault_energy_mj_m2))

        if candidate.mesoscale:
            receipts.append(cls.validate_rve_convergence(candidate.mesoscale.rve_mesh_convergence_error))

        if candidate.continuum:
            receipts.append(cls.validate_clausius_duhem_dissipation(candidate.continuum.clausius_duhem_dissipation_w_m3))

        if candidate.assimilation:
            receipts.append(cls.validate_compound_variance(candidate.assimilation.compound_variance_ratio))

        candidate.validation_receipts = receipts
        return receipts
