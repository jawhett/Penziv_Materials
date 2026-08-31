"""Unit tests for Multi-Tiered Surrogate and First-Principles Calculation Hierarchy."""

import unittest
import numpy as np

from penziv_materials.structure.crystal_structure import CrystalStructure, PeriodicLattice, Site
from penziv_materials.scale5_quantum.surrogate_hierarchy import (
    SurrogateTier,
    SurrogateResult,
    HeuristicPrescreenFilter,
    UniversalMLIPCalculator,
    AbInitioDFTDriver,
    TieredSurrogateOrchestrator,
)
from penziv_materials.structure.global_crystal_search import (
    GlobalCrystalStructureSearchEngine,
    CrystalCandidate,
    CrystalSystem,
)


class TestSurrogateHierarchy(unittest.TestCase):
    def setUp(self):
        lattice = PeriodicLattice(np.eye(3) * 5.43)
        sites = [
            Site(species="Si", fractional_coords=np.array([0.0, 0.0, 0.0])),
            Site(species="Si", fractional_coords=np.array([0.25, 0.25, 0.25])),
        ]
        self.si_struct = CrystalStructure(
            formula="Si",
            lattice=lattice,
            sites=sites,
            space_group_number=227,
        )

    def test_tier0_heuristic_prescreen_filter(self):
        res = HeuristicPrescreenFilter.evaluate(self.si_struct)
        self.assertEqual(res.tier, SurrogateTier.TIER_0_HEURISTIC)
        self.assertEqual(res.formula, "Si")
        self.assertLess(res.energy_per_atom_ev, 0.0)
        self.assertEqual(res.calculator_name, "HarrisonTightBindingHeuristic")

    def test_tier1_universal_mlip_calculator(self):
        calc = UniversalMLIPCalculator(model_name="mace_mp")
        res = calc.evaluate(self.si_struct)
        self.assertEqual(res.tier, SurrogateTier.TIER_1_MLIP)
        self.assertEqual(res.formula, "Si")
        self.assertLess(res.energy_per_atom_ev, 0.0)
        self.assertLessEqual(res.epistemic_uncertainty, 0.05)

    def test_tier2_ab_initio_dft_driver_card_generation(self):
        driver = AbInitioDFTDriver(code="QUANTUM_ESPRESSO")
        card = driver.generate_input_card(self.si_struct)
        self.assertIn("&CONTROL", card)
        self.assertIn("&SYSTEM", card)
        self.assertIn("ATOMIC_POSITIONS", card)
        self.assertIn("Si", card)

    def test_tiered_surrogate_orchestrator_tier0(self):
        orch = TieredSurrogateOrchestrator()
        res = orch.evaluate_structure(self.si_struct, target_tier=SurrogateTier.TIER_0_HEURISTIC)
        self.assertEqual(res.tier, SurrogateTier.TIER_0_HEURISTIC)

    def test_tiered_surrogate_orchestrator_tier1(self):
        orch = TieredSurrogateOrchestrator()
        res = orch.evaluate_structure(self.si_struct, target_tier=SurrogateTier.TIER_1_MLIP)
        self.assertEqual(res.tier, SurrogateTier.TIER_1_MLIP)

    def test_tiered_surrogate_orchestrator_escalation_to_tier2(self):
        orch = TieredSurrogateOrchestrator()
        res = orch.evaluate_structure(self.si_struct, target_tier=SurrogateTier.TIER_2_DFT)
        self.assertEqual(res.tier, SurrogateTier.TIER_2_DFT)
        self.assertEqual(res.max_force_ev_ang, 0.0001)

    def test_global_crystal_search_refinement(self):
        cand = CrystalCandidate(
            space_group_number=225,
            space_group_symbol="Fm-3m",
            crystal_system=CrystalSystem.CUBIC,
            lattice_matrix=(np.eye(3) * 4.0).tolist(),
            lattice_parameters={"a": 4.0, "b": 4.0, "c": 4.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
            atomic_sites=[{"species": "Al", "coords": [0.0, 0.0, 0.0]}],
            total_energy_ev_atom=-3.5,
            unit_cell_volume_ang3=64.0,
            theoretical_density_g_cm3=2.7,
        )
        refined = GlobalCrystalStructureSearchEngine.refine_candidate_with_tiered_surrogate(
            candidate=cand,
            formula="Al",
            target_tier=SurrogateTier.TIER_1_MLIP,
        )
        self.assertEqual(refined["candidate_space_group"], 225)
        self.assertEqual(refined["surrogate_tier"], SurrogateTier.TIER_1_MLIP.value)
        self.assertIn("refined_energy_per_atom_ev", refined)


if __name__ == "__main__":
    unittest.main()
