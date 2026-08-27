"""Unit tests for Economics, Supply Chain Criticality, Toxicity EHS, and TEA tools."""

import unittest
import numpy as np

from penziv_materials.economics.commodity_pricing import CommodityPricingEngine
from penziv_materials.economics.supply_chain_risk import SupplyChainRiskEngine
from penziv_materials.economics.toxicity_ehs import ToxicityEHSEngine
from penziv_materials.economics.techno_economics import TechnoEconomicsEngine
from penziv_materials.economics.economic_tools import (
    get_composition_cost,
    evaluate_supply_chain_risk,
    evaluate_toxicity_and_regulations,
    compute_techno_economic_lcos,
)
from penziv_materials.validation.handshake_gates import HandshakeGatekeeper
from penziv_materials.core.models import ValidationStatus


class TestEconomicsEHS(unittest.TestCase):
    def test_commodity_pricing_engine(self):
        pricing = CommodityPricingEngine()
        fractions = {"Mg": 0.30, "Zr": 0.40, "S": 0.30}
        res = pricing.compute_weighted_composition_cost(fractions, purity_grade="battery_grade_99_9")
        self.assertGreater(res["raw_material_cost_usd_kg"], 0.0)
        self.assertIn("purity_multiplier_applied", res)

    def test_supply_chain_risk_engine(self):
        risk = SupplyChainRiskEngine()
        fractions = {"Na": 0.50, "P": 0.20, "S": 0.30}
        res = risk.evaluate_composition_supply_chain_risk(fractions)
        self.assertLess(res["weighted_hhi_refining"], 3000.0)
        self.assertTrue(res["is_geopolitically_resilient"])

    def test_toxicity_and_banned_species_filtering(self):
        ehs = ToxicityEHSEngine()
        # Safe composition
        safe_res = ehs.evaluate_composition_ehs_and_carbon({"Mg": 0.5, "S": 0.5})
        self.assertTrue(safe_res["is_regulatory_compliant"])
        self.assertEqual(len(safe_res["banned_elements_detected"]), 0)

        # Toxic / banned composition containing Cadmium or Mercury
        banned_res = ehs.evaluate_composition_ehs_and_carbon({"Cd": 0.3, "Te": 0.7})
        self.assertFalse(banned_res["is_regulatory_compliant"])
        self.assertIn("Cd", banned_res["banned_elements_detected"])

    def test_techno_economic_lcos(self):
        tea = TechnoEconomicsEngine()
        lcos = tea.compute_electrolyte_lcos_floor(
            electrolyte_raw_cost_usd_kg=15.0,
            electrolyte_layer_thickness_um=20.0,
            cell_areal_capacity_mah_cm2=4.0,
            nominal_cell_voltage_v=3.2,
        )
        self.assertGreater(lcos["electrolyte_cost_contribution_usd_kwh"], 0.0)
        self.assertTrue(lcos["is_below_target_floor_50_usd_kwh"])

        thermal = tea.compute_thermal_synthesis_energy_budget(sintering_temp_c=180.0)
        self.assertTrue(thermal["is_low_energy_cold_sintering"])

    def test_programmatic_agent_tools(self):
        # 1. get_composition_cost
        cost_tool = get_composition_cost({"Mg": 0.25, "Zr": 0.35, "S": 0.40}, target_mass_kg=5.0)
        self.assertIn("raw_material_cost_usd_kg", cost_tool)
        self.assertEqual(cost_tool["total_batch_cost_usd"], cost_tool["raw_material_cost_usd_kg"] * 5.0)

        # 2. evaluate_supply_chain_risk
        risk_tool = evaluate_supply_chain_risk(["Na", "Si", "P", "S"])
        self.assertIn("weighted_hhi_mining", risk_tool)

        # 3. evaluate_toxicity_and_regulations
        tox_tool = evaluate_toxicity_and_regulations("Mg1.2Sc0.2Zr1.8S6")
        self.assertTrue(tox_tool["is_regulatory_compliant"])

        # 4. compute_techno_economic_lcos
        lcos_tool = compute_techno_economic_lcos(
            material_params={"raw_material_cost_usd_kg": 18.5, "thickness_um": 25.0, "sintering_temp_c": 200.0}
        )
        self.assertIn("combined_techno_economic_score", lcos_tool)

    def test_handshake_gates_ehs_and_supply(self):
        gatekeeper = HandshakeGatekeeper()
        receipt_ehs = gatekeeper.validate_toxicity_and_banned_species(banned_elements=[], epa_hazard_score=2.1)
        self.assertEqual(receipt_ehs.status, ValidationStatus.PASSED)

        receipt_banned = gatekeeper.validate_toxicity_and_banned_species(banned_elements=["Hg"], epa_hazard_score=9.5)
        self.assertEqual(receipt_banned.status, ValidationStatus.FAILED)

        receipt_supply = gatekeeper.validate_supply_chain_resilience(weighted_hhi_refining=2200.0)
        self.assertEqual(receipt_supply.status, ValidationStatus.PASSED)


if __name__ == "__main__":
    unittest.main()
