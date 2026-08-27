"""Autonomous Pareto Alloy Discovery Engine: Inverse Design & Multi-Objective Screening."""

from typing import Dict, List, Optional, Tuple, Any
import datetime
import numpy as np
from pydantic import BaseModel, Field

from penziv_materials.core.models import (
    MaterialCandidate,
    ValidationStatus,
)
from penziv_materials.orchestration.meta_orchestrator import MetaOrchestrator


class DiscoveryTargetConstraints(BaseModel):
    min_yield_strength_mpa: float = 1000.0
    max_steady_state_creep_rate_s_inv: float = 1.0e-12
    min_fracture_toughness_k_ic: float = 60.0
    max_crustal_exergy_mj_kg: float = 90.0
    target_temperature_k: float = 1123.15  # 850 C
    applied_creep_stress_mpa: float = 250.0


class ParetoDiscoveryResult(BaseModel):
    total_screened: int
    physically_stable_count: int
    pareto_optimal_candidates: List[MaterialCandidate]
    top_candidate: Optional[MaterialCandidate] = None
    target_constraints: DiscoveryTargetConstraints
    timestamp: str


class AlloyDiscoveryEngine:
    """Autonomous inverse design search loop exploring composition space for multi-objective performance."""

    def __init__(self, orchestrator: Optional[MetaOrchestrator] = None):
        self.orchestrator = orchestrator or MetaOrchestrator()

    def generate_random_compositions(
        self,
        base_elements: List[str],
        n_samples: int = 50,
        primary_element: str = "Ni",
        primary_fraction_range: Tuple[float, float] = (0.45, 0.70),
        random_seed: Optional[int] = 42,
    ) -> List[Dict[str, float]]:
        """Sample physically realistic alloy compositions using Dirichlet / constrained uniform distributions."""
        if random_seed is not None:
            np.random.seed(random_seed)

        compositions = []
        secondary_elements = [el for el in base_elements if el != primary_element]

        for _ in range(n_samples):
            # Sample primary fraction
            prim_frac = np.random.uniform(primary_fraction_range[0], primary_fraction_range[1])
            rem_frac = 1.0 - prim_frac

            # Sample secondary distribution via Dirichlet
            alpha = np.ones(len(secondary_elements))
            # Minor microalloying preference for B, C, Zr
            for i, el in enumerate(secondary_elements):
                if el in ["B", "C"]:
                    alpha[i] = 0.15
                elif el in ["Al", "Ti", "Cr"]:
                    alpha[i] = 2.0
                elif el in ["Nb", "Mo", "W", "Ta"]:
                    alpha[i] = 1.2

            sec_weights = np.random.dirichlet(alpha)
            comp = {primary_element: round(float(prim_frac), 4)}
            for el, w in zip(secondary_elements, sec_weights):
                comp[el] = round(float(rem_frac * w), 4)

            compositions.append(comp)

        return compositions

    def evaluate_pareto_front(
        self,
        candidates: List[MaterialCandidate],
    ) -> List[MaterialCandidate]:
        """Extract Non-Dominated Pareto Rank-1 candidates across:

        1. Maximize Yield Strength (sigma_y)
        2. Minimize Creep Rate (eps_dot) -> maximize -log10(eps_dot)
        3. Maximize Fracture Toughness (K_Ic)
        4. Minimize Exergy (Ex_min) -> maximize -Ex_min
        """
        # Filter strictly stable candidates
        stable_candidates = [
            c
            for c in candidates
            if all(
                r.status in [ValidationStatus.PASSED, ValidationStatus.WARNING]
                for r in c.validation_receipts
            )
        ]

        if not stable_candidates:
            return []

        # Multi-objective criteria matrix: [sigma_y, -log10(creep), K_Ic, -Ex_min]
        matrix = []
        for c in stable_candidates:
            ys = c.continuum.yield_strength_mpa if c.continuum else 0.0
            creep = c.continuum.steady_state_creep_rate_s_inv if c.continuum else 1.0
            k_ic = c.continuum.fracture_toughness_k_ic_mpa_sqrt_m if c.continuum else 0.0
            exergy = c.process.min_ore_extraction_exergy_mj_kg if c.process else 1000.0

            log_creep_inv = -np.log10(max(1e-25, creep))
            matrix.append([ys, log_creep_inv, k_ic, -exergy])

        matrix_np = np.array(matrix, dtype=np.float64)
        n_candidates = len(stable_candidates)
        is_pareto = np.ones(n_candidates, dtype=bool)

        for i in range(n_candidates):
            for j in range(n_candidates):
                if i != j:
                    # j dominates i if j is >= in all objectives and strictly > in at least one
                    if np.all(matrix_np[j] >= matrix_np[i]) and np.any(matrix_np[j] > matrix_np[i]):
                        is_pareto[i] = False
                        break

        pareto_list = []
        for idx, (cand, pareto_flag) in enumerate(zip(stable_candidates, is_pareto)):
            if pareto_flag:
                cand.pareto_rank = 1
                pareto_list.append(cand)
            else:
                cand.pareto_rank = 2

        # Sort pareto list by yield strength descending
        pareto_list.sort(
            key=lambda c: c.continuum.yield_strength_mpa if c.continuum else 0.0,
            reverse=True,
        )
        return pareto_list

    def discover_optimal_alloys(
        self,
        base_elements: List[str],
        constraints: DiscoveryTargetConstraints,
        n_samples: int = 40,
        prefix_name: str = "Penziv-Opt",
    ) -> ParetoDiscoveryResult:
        """Run full autonomous discovery screening over sampled composition space."""
        compositions = self.generate_random_compositions(
            base_elements=base_elements,
            n_samples=n_samples,
        )

        all_candidates: List[MaterialCandidate] = []
        for i, comp in enumerate(compositions, 1):
            cand_name = f"{prefix_name}-{i:03d}"
            cand = self.orchestrator.run_forward_multiscale_prediction(
                candidate_name=cand_name,
                composition=comp,
                target_temperature_k=constraints.target_temperature_k,
                applied_creep_stress_mpa=constraints.applied_creep_stress_mpa,
            )
            all_candidates.append(cand)

        # Extract Pareto front
        pareto_candidates = self.evaluate_pareto_front(all_candidates)

        # Filter candidates meeting strict constraints
        constrained_pareto = []
        for c in pareto_candidates:
            if c.continuum and c.process:
                if (
                    c.continuum.yield_strength_mpa >= constraints.min_yield_strength_mpa
                    and c.continuum.steady_state_creep_rate_s_inv <= constraints.max_steady_state_creep_rate_s_inv
                    and c.continuum.fracture_toughness_k_ic_mpa_sqrt_m >= constraints.min_fracture_toughness_k_ic
                    and c.process.min_ore_extraction_exergy_mj_kg <= constraints.max_crustal_exergy_mj_kg
                ):
                    constrained_pareto.append(c)

        top_cand = constrained_pareto[0] if constrained_pareto else (pareto_candidates[0] if pareto_candidates else None)

        stable_count = sum(
            1
            for c in all_candidates
            if all(
                r.status in [ValidationStatus.PASSED, ValidationStatus.WARNING]
                for r in c.validation_receipts
            )
        )

        return ParetoDiscoveryResult(
            total_screened=len(all_candidates),
            physically_stable_count=stable_count,
            pareto_optimal_candidates=constrained_pareto if constrained_pareto else pareto_candidates,
            top_candidate=top_cand,
            target_constraints=constraints,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
