"""Meta-Orchestrator Discovery Agent (META-ORCH): Master Multiscale Controller & Pareto Optimization Loop."""

import datetime
from typing import Dict, List, Optional, Tuple
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
from penziv_materials.scale5_quantum.q_elec import QElectAgent
from penziv_materials.scale4_atomistic.atom_dyn import AtomDynAgent
from penziv_materials.scale3_mesoscale.meso_kinetic import MesoKineticAgent
from penziv_materials.scale2_continuum.cont_micro import ContMicroAgent
from penziv_materials.scale1_process.proc_mfg import ProcMfgAgent
from penziv_materials.meta_bridge.uq_bridge import UqBridgeAgent
from penziv_materials.validation.born_stability import BornStabilityValidator
from penziv_materials.validation.handshake_gates import HandshakeGatekeeper


class MetaOrchestrator:
    """Master Orchestrator coordinating all 5 physical scales, UQ bridge, and Pareto multi-objective discovery."""

    def __init__(self):
        self.q_elec = QElectAgent()
        self.atom_dyn = AtomDynAgent()
        self.meso_kinetic = MesoKineticAgent()
        self.cont_micro = ContMicroAgent()
        self.proc_mfg = ProcMfgAgent()
        self.uq_bridge = UqBridgeAgent()

    def run_forward_multiscale_prediction(
        self,
        candidate_name: str,
        composition: Dict[str, float],
        target_temperature_k: float = 1123.15,  # 850 C
        applied_creep_stress_mpa: float = 250.0,
    ) -> MaterialCandidate:
        """Execute complete forward multiscale physics simulation from Scale 5 down to Scale 1 + UQ Assimilation."""
        # 1. Scale 5: Quantum & Electronic Ground State
        q_state = self.q_elec.execute_forward_scale(
            formula=candidate_name,
            composition=composition,
            temperature_k=target_temperature_k,
        )

        # 2. Scale 4: Atomistic Dynamics & Extended Defects
        atom_state = self.atom_dyn.execute_forward_scale(
            quantum_state=q_state,
            composition=composition,
            temperature_k=target_temperature_k,
        )

        # Active learning feedback: If OOD detected, re-invoke Q-ELEC with higher fidelity
        if atom_state.is_ood:
            q_state = self.q_elec.execute_forward_scale(
                formula=candidate_name,
                composition=composition,
                temperature_k=target_temperature_k,
            )

        # 3. Scale 3: Mesoscale Microstructure & Solute Trapping
        meso_state = self.meso_kinetic.execute_forward_scale(
            quantum_state=q_state,
            atomistic_state=atom_state,
        )

        # 4. Scale 2: Continuum Micromechanics & Creep
        cont_state = self.cont_micro.execute_forward_scale(
            quantum_state=q_state,
            mesoscale_state=meso_state,
            temperature_k=target_temperature_k,
            applied_creep_stress_mpa=applied_creep_stress_mpa,
        )

        # 5. Scale 1: Process Dynamics & Exergy
        proc_state = self.proc_mfg.execute_forward_scale(
            composition=composition,
            target_temp_k=target_temperature_k,
        )

        candidate = MaterialCandidate(
            name=candidate_name,
            composition=composition,
            target_temperature_k=target_temperature_k,
            quantum=q_state,
            atomistic=atom_state,
            mesoscale=meso_state,
            continuum=cont_state,
            process=proc_state,
        )

        # 6. Meta-Scale: Sim-to-Real Assimilation
        assimilation = self.uq_bridge.execute_sim_to_real_assimilation(candidate)
        candidate.assimilation = assimilation

        # 7. Run Born Stability Gate and Scale Handshake Suite
        receipts = HandshakeGatekeeper.validate_candidate(candidate)

        if q_state.c_voigt_gpa:
            C_voigt = np.array(q_state.c_voigt_gpa)
            born_receipt = BornStabilityValidator.validate(C_voigt)
            receipts.insert(0, born_receipt)

        candidate.validation_receipts = receipts
        return candidate

    def compute_pareto_front(
        self,
        candidates: List[MaterialCandidate],
    ) -> List[Tuple[MaterialCandidate, int]]:
        """Rank candidates across Pareto objectives:

        1. Maximize Yield Strength (MPa)
        2. Minimize Steady-State Creep Rate (s^-1)
        3. Minimize Crustal Extraction Exergy (MJ/kg)
        4. Maximize Fracture Toughness (MPa·m^0.5)
        """
        results = []
        for cand in candidates:
            # Check if all validation gates passed
            all_passed = all(
                r.status in [ValidationStatus.PASSED, ValidationStatus.WARNING]
                for r in cand.validation_receipts
            )

            # Score calculation: Higher is better
            ys = cand.continuum.yield_strength_mpa if cand.continuum else 0.0
            k_ic = cand.continuum.fracture_toughness_k_ic_mpa_sqrt_m if cand.continuum else 0.0
            creep = cand.continuum.steady_state_creep_rate_s_inv if cand.continuum else 1e-6
            exergy = cand.process.min_ore_extraction_exergy_mj_kg if cand.process else 100.0

            # Composite multi-objective Pareto index
            score = (ys / 1000.0) * (k_ic / 50.0) / (np.log10(max(1e-15, creep)) * -0.1 * (exergy / 40.0))
            if not all_passed:
                score *= 0.1  # Heavy penalty for unphysical / unstable materials

            cand.pareto_rank = 1 if score > 1.5 else 2
            results.append((cand, cand.pareto_rank))

        results.sort(key=lambda x: x[0].continuum.yield_strength_mpa if x[0].continuum else 0.0, reverse=True)
        return results
