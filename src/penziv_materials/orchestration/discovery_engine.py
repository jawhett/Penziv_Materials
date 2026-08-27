"""Autonomous Discovery Engine: Inverse Design & Multi-Objective Pareto Frontier Optimization."""

import datetime
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from pydantic import BaseModel, Field

from penziv_materials.core.models import (
    MaterialCandidate,
    ValidationReceipt,
    ValidationStatus,
)
from penziv_materials.orchestration.meta_orchestrator import MetaOrchestrator


class DiscoveryTargetConstraints(BaseModel):
    """Target performance bounds and operating constraints for candidate screening."""
    min_yield_strength_mpa: float = 950.0
    max_steady_state_creep_rate_s_inv: float = 1.0e-11
    min_fracture_toughness_k_ic: float = 65.0
    max_crustal_exergy_mj_kg: float = 90.0
    target_temperature_k: float = 1123.15  # 850 °C
    applied_creep_stress_mpa: float = 250.0


class ParetoDiscoveryResult(BaseModel):
    """Dataset encapsulating screened candidates, Pareto front, and top recommendation."""
    total_screened: int
    physically_stable_count: int
    pareto_optimal_candidates: List[MaterialCandidate]
    top_candidate: Optional[MaterialCandidate] = None
    target_constraints: DiscoveryTargetConstraints
    timestamp: str


class AlloyDiscoveryEngine:
    """Autonomous engine for multi-objective composition sampling, multiscale screening, and Pareto ranking."""

    def __init__(self):
        self.orchestrator = MetaOrchestrator()

    def generate_random_compositions(
        self,
        base_elements: List[str],
        n_samples: int = 30,
        primary_element: str = "Ni",
        random_seed: Optional[int] = 42,
    ) -> List[Dict[str, float]]:
        """Generate physically reasonable alloy composition vectors normalized to sum to 1.0."""
        if random_seed is not None:
            np.random.seed(random_seed)

        compositions = []
        for _ in range(n_samples):
            # Dirichlet-distributed random fractions
            raw_weights = np.random.dirichlet(np.ones(len(base_elements)) * 1.2)
            comp_dict = {elem: float(w) for elem, w in zip(base_elements, raw_weights)}

            # Ensure primary matrix element has minimum threshold (e.g. Ni >= 45%)
            if primary_element in comp_dict and comp_dict[primary_element] < 0.45:
                deficit = 0.50 - comp_dict[primary_element]
                comp_dict[primary_element] = 0.50
                # Rescale other elements
                other_sum = sum(w for k, w in comp_dict.items() if k != primary_element)
                if other_sum > 0:
                    for k in comp_dict:
                        if k != primary_element:
                            comp_dict[k] = (comp_dict[k] / other_sum) * (1.0 - 0.50)

            # Round and normalize
            total = sum(comp_dict.values())
            normalized_comp = {k: round(v / total, 4) for k, v in comp_dict.items()}
            compositions.append(normalized_comp)

        return compositions

    def evaluate_pareto_front(
        self,
        candidates: List[MaterialCandidate],
    ) -> List[MaterialCandidate]:
        """Compute Non-Dominated Pareto Rank across candidates."""
        ranked_pairs = self.orchestrator.compute_pareto_front(candidates)
        pareto_optimal = [c for c, rank in ranked_pairs if rank == 1]

        # Annotate candidates with pareto rank
        for c, rank in ranked_pairs:
            c.pareto_rank = rank

        return pareto_optimal

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
                r.status in [ValidationStatus.PASSED, ValidationStatus.WARNING, ValidationStatus.ROUTED_TO_HIGH_FIDELITY]
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
