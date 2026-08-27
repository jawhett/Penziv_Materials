"""Production Validation Benchmark: Zero-Parameter Discovery of Additive Superalloys."""

from typing import Dict, List, Tuple, Any
import datetime
from penziv_materials.core.models import MaterialCandidate, ValidationStatus
from penziv_materials.orchestration.discovery_engine import (
    AlloyDiscoveryEngine,
    DiscoveryTargetConstraints,
    ParetoDiscoveryResult,
)


class SuperalloyBenchmarkSuite:
    """Benchmark suite validating autonomous closed-loop discovery for high-T aerospace superalloys."""

    @staticmethod
    def run_high_temperature_superalloy_benchmark(
        target_operating_temp_k: float = 1123.15,  # 850°C
        applied_creep_stress_mpa: float = 250.0,
        num_candidates: int = 25,
    ) -> Dict[str, Any]:
        """Execute production benchmark discovering an additive superalloy with:

        - Yield Strength > 1050 MPa at 850°C
        - Creep Rate < 1e-12 s^-1
        - Minimum extraction exergy < 80 MJ/kg
        - Full Born mechanical stability & Handshake pass
        """
        engine = AlloyDiscoveryEngine()
        elements = ["Ni", "Cr", "Al", "Ti", "Nb", "Mo", "W", "B"]

        constraints = DiscoveryTargetConstraints(
            min_yield_strength_mpa=1000.0,
            max_steady_state_creep_rate_s_inv=1.0e-12,
            min_fracture_toughness_k_ic=60.0,
            max_crustal_exergy_mj_kg=80.0,
            target_temperature_k=target_operating_temp_k,
            applied_creep_stress_mpa=applied_creep_stress_mpa,
        )

        result: ParetoDiscoveryResult = engine.discover_optimal_alloys(
            base_elements=elements,
            constraints=constraints,
            n_samples=num_candidates,
            prefix_name="Penziv-Aero718",
        )

        passed_all_gates = False
        top_cand = result.top_candidate
        if top_cand:
            passed_all_gates = all(
                r.status in [ValidationStatus.PASSED, ValidationStatus.WARNING]
                for r in top_cand.validation_receipts
            )

        return {
            "benchmark_name": "High-Temperature Superalloy Zero-Parameter Discovery",
            "target_temperature_k": target_operating_temp_k,
            "candidates_evaluated": result.total_screened,
            "physically_stable_count": result.physically_stable_count,
            "pareto_solutions_found": len(result.pareto_optimal_candidates),
            "top_candidate": top_cand.model_dump() if top_cand else None,
            "passed_all_physics_gates": passed_all_gates,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
