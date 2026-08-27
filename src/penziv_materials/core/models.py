"""Data models and Pydantic schemas for multiscale state, scale transfer packets, and validation receipts."""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import numpy as np


class CrystalSystem(str, Enum):
    CUBIC = "cubic"
    HEXAGONAL = "hexagonal"
    TETRAGONAL = "tetragonal"
    ORTHORHOMBIC = "orthorhombic"
    TRIGONAL = "trigonal"
    MONOCLINIC = "monoclinic"
    TRICLINIC = "triclinic"
    ISOTROPIC = "isotropic"


class ValidationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    ROUTED_TO_HIGH_FIDELITY = "routed_to_high_fidelity"


class ValidationReceipt(BaseModel):
    gate_name: str
    status: ValidationStatus
    metric_value: float
    threshold: float
    details: str
    timestamp: str


class QuantumState(BaseModel):
    formula: str
    space_group: str
    temperature_k: float = 300.0
    formation_energy_ev_atom: float
    helmholtz_free_energy_ev_atom: float
    c_voigt_gpa: List[List[float]] = Field(default_factory=list)
    thermal_expansion_coeff: float = 1.2e-5  # 1/K
    band_gap_ev: Optional[float] = None
    sro_stacking_fault_energy_mj_m2: float = 45.0
    delta_learning_offset_ev: float = 0.0


class AtomisticState(BaseModel):
    defect_migration_barrier_ev: float
    migration_barrier_sigma_ev: float
    kinetic_rate_s_inv: float
    lognormal_variance_sigma_ln_gamma_sq: float
    peierls_stress_gpa: float
    grain_boundary_energy_j_m2: float
    solute_gb_segregation_energy_ev: float
    work_of_separation_j_m2: float = 3.5
    ood_max_negative_log_likelihood: float = 4.2
    is_ood: bool = False


class MesoscaleState(BaseModel):
    rve_dimension_um: float = 50.0
    average_grain_size_um: float = 15.0
    crss_basal_gpa: float = 0.120
    asymmetric_hardening_q: float = 1.4
    solute_trapping_partition_k: float = 0.85
    rve_mesh_convergence_error: float = 0.008
    void_volume_fraction: float = 0.0001


class ContinuumState(BaseModel):
    yield_strength_mpa: float = 850.0
    ultimate_tensile_strength_mpa: float = 1250.0
    fracture_toughness_k_ic_mpa_sqrt_m: float = 85.0
    steady_state_creep_rate_s_inv: float = 1.2e-9
    weibull_modulus_m: float = 14.5
    paris_law_c: float = 3.2e-11
    paris_law_m: float = 3.1
    clausius_duhem_dissipation_w_m3: float = 1.5e5


class ProcessState(BaseModel):
    solidification_cooling_rate_k_s: float = 1.0e4
    thermal_gradient_k_m: float = 5.0e6
    solidification_velocity_m_s: float = 0.025
    residual_stress_max_mpa: float = 240.0
    oxide_growth_parabolic_rate_kp: float = 1.5e-14
    min_ore_extraction_exergy_mj_kg: float = 48.2
    synthesizability_score: float = 0.92


class SimToRealAssimilation(BaseModel):
    xrd_phase_fraction_error: float = 0.021
    ebsd_odf_kl_divergence: float = 0.045
    nanoindentation_h0_gpa: float = 6.2
    nanoindentation_h_star_nm: float = 180.0
    compound_variance_ratio: float = 0.082


class MaterialCandidate(BaseModel):
    name: str
    composition: Dict[str, float]
    target_temperature_k: float = 1123.15  # 850 C
    quantum: Optional[QuantumState] = None
    atomistic: Optional[AtomisticState] = None
    mesoscale: Optional[MesoscaleState] = None
    continuum: Optional[ContinuumState] = None
    process: Optional[ProcessState] = None
    assimilation: Optional[SimToRealAssimilation] = None
    validation_receipts: List[ValidationReceipt] = Field(default_factory=list)
    pareto_rank: Optional[int] = None
