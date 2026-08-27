"""Meta-Orchestrator: Autonomous Multiscale Forward & Inverse Prediction Engine."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from penziv_materials.core.models import (
    MaterialCandidate,
    QuantumState,
    AtomisticState,
    MesoscaleState,
    ContinuumState,
    ProcessState,
    SimToRealAssimilation,
    ValidationReceipt,
    ValidationStatus,
)
from penziv_materials.scale5_quantum.q_elec import QElecAgent
from penziv_materials.scale4_atomistic.atom_dyn import AtomDynAgent
from penziv_materials.scale3_mesoscale.meso_kinetic import MesoKineticAgent
from penziv_materials.scale3_mesoscale.phase_field import PhaseFieldEngine
from penziv_materials.scale2_continuum.cont_micro import ContMicroAgent
from penziv_materials.scale2_continuum.cpfft_solver import CPFFTSolver
from penziv_materials.scale1_process.proc_mfg import ProcMfgAgent
from penziv_materials.scale1_process.meltpool_cfd import MeltPoolCFDEngine
from penziv_materials.meta_bridge.uq_bridge import UQBridgeAgent
from penziv_materials.validation.born_stability import BornStabilityValidator
from penziv_materials.validation.handshake_gates import HandshakeGatekeeper


class MetaOrchestrator:
    """Master orchestrator driving the 5-scale physics execution pipeline and UQ handshake validation."""

    def __init__(self):
        self.q_elec = QElecAgent()
        self.atom_dyn = AtomDynAgent()
        self.meso_kinetic = MesoKineticAgent()
        self.phase_field = PhaseFieldEngine(grid_size=(16, 16))
        self.cont_micro = ContMicroAgent()
        self.proc_mfg = ProcMfgAgent()
        self.meltpool_cfd = MeltPoolCFDEngine()
        self.uq_bridge = UQBridgeAgent()

    def run_forward_multiscale_prediction(
        self,
        candidate_name: str,
        composition: Dict[str, float],
        target_temperature_k: float = 1123.15,
        applied_stress_mpa: Optional[float] = None,
        applied_creep_stress_mpa: Optional[float] = None,
        crystal_system: str = "FCC",
    ) -> MaterialCandidate:
        """Execute full forward multiscale pipeline across all 5 physical tiers with dynamic solver integration."""
        stress_val = applied_creep_stress_mpa if applied_creep_stress_mpa is not None else (applied_stress_mpa if applied_stress_mpa is not None else 250.0)

        # 1. Scale 5: Quantum Electronic Structure & Miedema Free Energy
        formula = "".join(f"{k}{int(v*100)}" for k, v in composition.items())
        q_state = self.q_elec.execute_quantum_state_evaluation(
            formula=formula,
            composition=composition,
            temperature_k=target_temperature_k,
        )
        c_voigt_matrix = np.array(q_state.c_voigt_gpa)

        # 2. Scale 4: Atomistic Dynamics & Kinetic Rates
        atom_state = self.atom_dyn.execute_atomistic_evaluation(
            composition=composition,
            temperature_k=target_temperature_k,
            c44_gpa=c_voigt_matrix[3, 3] if c_voigt_matrix.shape == (6, 6) else 115.0,
        )

        # 3. Scale 3: Mesoscale Microstructure & Active Phase-Field Morphology Parameterization
        c_pf = np.ones((16, 16)) * 0.50
        eta_pf = np.zeros((16, 16))
        c_pf_new, eta_pf_new = self.phase_field.step_forward_semi_implicit(c_pf, eta_pf, dt=0.01)
        precipitate_vol_frac = float(np.clip(np.mean(c_pf_new > 0.55), 0.20, 0.75))

        meso_state = self.meso_kinetic.execute_mesoscale_evaluation(
            composition=composition,
            tau_p_gpa=atom_state.peierls_stress_gpa,
            gamma_sfe_mj_m2=q_state.sro_stacking_fault_energy_mj_m2,
            precipitate_vol_frac=precipitate_vol_frac,
            precipitate_radius_nm=30.0,
        )

        # 4. Scale 2: Continuum Homogenization & Full-Field Dynamic CPFFT
        d_grain_um = max(1.0, meso_state.average_grain_size_um)
        hall_petch_bonus_mpa = 150.0 / np.sqrt(d_grain_um)

        cont_state = self.cont_micro.execute_continuum_evaluation(
            tau_crss_gpa=meso_state.crss_basal_gpa,
            c_voigt_gpa=c_voigt_matrix,
            temperature_k=target_temperature_k,
            applied_stress_mpa=stress_val,
            grain_size_um=d_grain_um,
        )
        cont_state.yield_strength_mpa += hall_petch_bonus_mpa
        cont_state.ultimate_tensile_strength_mpa += hall_petch_bonus_mpa

        # Active CPFFT solver parameterized by crystal system symmetry
        cpfft_solver = CPFFTSolver(crystal_system=crystal_system)
        strain_tensor = np.diag([0.001, -0.0005, -0.0005])
        cpfft_res = cpfft_solver.step_plastic_slip_and_gnd(
            applied_strain_rate=strain_tensor,
            dt_s=0.01,
            c_voigt_gpa=c_voigt_matrix,
            crss_gpa=meso_state.crss_basal_gpa,
        )
        cont_state.clausius_duhem_dissipation_w_m3 = max(cont_state.clausius_duhem_dissipation_w_m3, cpfft_res["plastic_dissipation_rate"])

        # 5. Scale 1: Process Solidification, CFD Melt-Pool & Synthesizability
        k_bulk, g_shear, e_young, nu_p = self.cont_micro.compute_voigt_reuss_hill_moduli(c_voigt_matrix)
        proc_state = self.proc_mfg.execute_process_evaluation(
            composition=composition,
            youngs_modulus_gpa=e_young,
            yield_strength_mpa=cont_state.yield_strength_mpa,
            thermal_expansion_coeff=q_state.thermal_expansion_coeff,
        )
        cfd_res = self.meltpool_cfd.compute_melt_pool_dimensions_and_history()
        proc_state.solidification_cooling_rate_k_s = float(cfd_res.get("cooling_rate_k_s", proc_state.solidification_cooling_rate_k_s))
        proc_state.thermal_gradient_k_m = float(cfd_res.get("thermal_gradient_k_m", proc_state.thermal_gradient_k_m))

        candidate = MaterialCandidate(
            name=candidate_name,
            composition=composition,
            quantum=q_state,
            atomistic=atom_state,
            mesoscale=meso_state,
            continuum=cont_state,
            process=proc_state,
        )

        # 6. Meta-Scale: Sim-to-Real Assimilation
        assimilation = self.uq_bridge.execute_sim_to_real_assimilation(candidate)
        candidate.assimilation = assimilation

        # 7. Scale Handshake Validation Suite & Born Mechanical Stability
        receipts = HandshakeGatekeeper.validate_candidate(candidate)
        if q_state.c_voigt_gpa:
            born_receipt = BornStabilityValidator.validate(c_voigt_matrix)
            receipts.insert(0, born_receipt)

        candidate.validation_receipts = receipts
        return candidate

    def compute_pareto_front(
        self,
        candidates: List[MaterialCandidate],
    ) -> List[Tuple[MaterialCandidate, int]]:
        ranks = []
        n = len(candidates)

        for i in range(n):
            c_i = candidates[i]
            domination_count = 0

            for j in range(n):
                if i == j:
                    continue
                c_j = candidates[j]
                if self._dominates(c_j, c_i):
                    domination_count += 1

            ranks.append((c_i, domination_count + 1))

        ranks.sort(key=lambda x: x[1])
        return ranks

    def _dominates(self, c1: MaterialCandidate, c2: MaterialCandidate) -> bool:
        if not c1.continuum or not c2.continuum or not c1.process or not c2.process:
            return False

        ys_better = c1.continuum.yield_strength_mpa >= c2.continuum.yield_strength_mpa
        creep_better = c1.continuum.steady_state_creep_rate_s_inv <= c2.continuum.steady_state_creep_rate_s_inv
        kic_better = c1.continuum.fracture_toughness_k_ic_mpa_sqrt_m >= c2.continuum.fracture_toughness_k_ic_mpa_sqrt_m
        exergy_better = c1.process.min_ore_extraction_exergy_mj_kg <= c2.process.min_ore_extraction_exergy_mj_kg

        strictly_better = (
            c1.continuum.yield_strength_mpa > c2.continuum.yield_strength_mpa
            or c1.continuum.steady_state_creep_rate_s_inv < c2.continuum.steady_state_creep_rate_s_inv
            or c1.continuum.fracture_toughness_k_ic_mpa_sqrt_m > c2.continuum.fracture_toughness_k_ic_mpa_sqrt_m
            or c1.process.min_ore_extraction_exergy_mj_kg < c2.process.min_ore_extraction_exergy_mj_kg
        )

        return (ys_better and creep_better and kic_better and exergy_better) and strictly_better
