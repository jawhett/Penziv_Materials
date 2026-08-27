"""Autonomous Multi-Scale Inverse Materials Discovery & Optimization Engine."""

import datetime
from typing import List, Dict, Tuple, Optional, Any, Callable
import numpy as np
from pydantic import BaseModel, Field
from penziv_materials.core.models import MaterialCandidate, ValidationStatus
from penziv_materials.orchestration.meta_orchestrator import MetaOrchestrator


class DiscoveryTargetConstraints(BaseModel):
    """Multi-objective target constraints across continuum, process, and mesoscale properties."""
    min_yield_strength_mpa: float = 800.0
    max_steady_state_creep_rate_s_inv: float = 1.0e-9
    min_fracture_toughness_k_ic: float = 40.0
    max_crustal_exergy_mj_kg: float = 150.0
    target_temperature_k: float = 1123.15
    max_solidification_crack_index: float = 0.35


class ParetoDiscoveryResult(BaseModel):
    """Aggregated result of multi-objective Pareto optimization."""
    total_screened: int
    physically_stable_count: int
    pareto_optimal_candidates: List[MaterialCandidate]
    top_candidate: Optional[MaterialCandidate] = None
    execution_timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())


class GenericDiscoveryTarget(BaseModel):
    """Generalized multi-objective target metrics (minimization or maximization)."""
    objective_names: List[str]
    objective_directions: List[str]  # "maximize" or "minimize"
    hard_constraints: Dict[str, Tuple[float, float]] = Field(default_factory=dict)


class GeneralizedParetoDiscoveryEngine:
    """Multi-objective Bayesian and Pareto Frontier explorer over arbitrary material classes."""

    def __init__(self):
        self.orchestrator = MetaOrchestrator()

    def evaluate_pareto_front(
        self,
        candidates: List[MaterialCandidate],
        target: GenericDiscoveryTarget,
    ) -> List[MaterialCandidate]:
        """Compute Non-Dominated Pareto Rank for arbitrary tensor and scalar metrics."""
        ranked_candidates = []
        n = len(candidates)

        for i in range(n):
            c_i = candidates[i]
            domination_count = 0
            for j in range(n):
                if i == j:
                    continue
                c_j = candidates[j]
                if self._check_dominance(c_j, c_i, target):
                    domination_count += 1
            c_i.pareto_rank = domination_count + 1
            if domination_count == 0:
                ranked_candidates.append(c_i)
        return ranked_candidates

    def _check_dominance(
        self,
        c1: MaterialCandidate,
        c2: MaterialCandidate,
        target: GenericDiscoveryTarget,
    ) -> bool:
        """Evaluate strict Pareto dominance across dynamic objective vectors."""
        v1 = [self._extract_metric(c1, name) for name in target.objective_names]
        v2 = [self._extract_metric(c2, name) for name in target.objective_names]

        if any(x is None for x in v1) or any(x is None for x in v2):
            return False

        better_or_equal = True
        strictly_better = False

        for val1, val2, direction in zip(v1, v2, target.objective_directions):
            if direction == "maximize":
                if val1 < val2:
                    better_or_equal = False
                if val1 > val2:
                    strictly_better = True
            else:
                if val1 > val2:
                    better_or_equal = False
                if val1 < val2:
                    strictly_better = True

        return better_or_equal and strictly_better

    def _extract_metric(self, candidate: MaterialCandidate, metric_name: str) -> Optional[float]:
        """Dynamically extract scalar metric from arbitrary multiscale sub-states."""
        for state in [candidate.quantum, candidate.atomistic, candidate.mesoscale, candidate.continuum, candidate.process]:
            if state and hasattr(state, metric_name):
                val = getattr(state, metric_name)
                if isinstance(val, (int, float)):
                    return float(val)
        return None


class AutonomousDiscoveryEngine:
    """Orchestrates candidate generation, scale simulation, Pareto frontier mapping, and candidate selection."""

    def __init__(self):
        self.orchestrator = MetaOrchestrator()
        self.gen_engine = GeneralizedParetoDiscoveryEngine()

    def generate_candidate_compositions(
        self,
        base_elements: List[str],
        n_samples: int = 10,
        primary_element: Optional[str] = None,
        random_seed: int = 42,
    ) -> List[Dict[str, float]]:
        """Generate compositions via Dirichlet sampling with optional solvent base element weighting."""
        np.random.seed(random_seed)
        k = len(base_elements)
        if k == 0:
            return []

        compositions = []
        for _ in range(n_samples):
            if primary_element and primary_element in base_elements:
                prim_frac = float(np.random.uniform(0.45, 0.65))
                rem_elements = [e for e in base_elements if e != primary_element]
                rem_fracs = np.random.dirichlet(np.ones(len(rem_elements))) * (1.0 - prim_frac)
                comp = {primary_element: round(prim_frac, 4)}
                for elem, frac in zip(rem_elements, rem_fracs):
                    comp[elem] = round(float(frac), 4)
            else:
                alpha = np.ones(k) * 1.2
                s = np.random.dirichlet(alpha)
                comp = {elem: round(float(frac), 4) for elem, frac in zip(base_elements, s)}

            tot = sum(comp.values())
            comp = {k: round(v / tot, 4) for k, v in comp.items()}
            compositions.append(comp)

        return compositions

    def generate_random_compositions(
        self,
        base_elements: List[str],
        n_samples: int = 10,
        primary_element: Optional[str] = None,
        random_seed: int = 42,
    ) -> List[Dict[str, float]]:
        """Generate random compositions with solvent or Dirichlet sampling."""
        return self.generate_candidate_compositions(
            base_elements=base_elements,
            n_samples=n_samples,
            primary_element=primary_element,
            random_seed=random_seed,
        )

    def discover_optimal_alloys(
        self,
        base_elements: List[str],
        constraints: DiscoveryTargetConstraints,
        n_samples: int = 8,
        prefix_name: str = "Penziv-HEA",
    ) -> ParetoDiscoveryResult:
        """Run full autonomous discovery loop across candidate compositions."""
        compositions = self.generate_candidate_compositions(base_elements, n_samples=n_samples)
        all_candidates = []

        for idx, comp in enumerate(compositions):
            cand_name = f"{prefix_name}-{idx+1:03d}"
            cand = self.orchestrator.run_forward_multiscale_prediction(
                candidate_name=cand_name,
                composition=comp,
                target_temperature_k=constraints.target_temperature_k,
            )
            all_candidates.append(cand)

        ranked = self.orchestrator.compute_pareto_front(all_candidates)
        for c, rank in ranked:
            c.pareto_rank = rank

        pareto_candidates = [c for c, rank in ranked if rank == 1]

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
        )


# Backward compatibility alias
AlloyDiscoveryEngine = AutonomousDiscoveryEngine
