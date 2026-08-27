"""Commodity Spot Pricing, Precursor Multipliers & Raw Material Cost Engine."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np


class CommodityPricingEngine:
    """Calculates weighted raw material costs, commercial precursor premiums, and spot price volatility."""

    # Benchmark global spot prices ($/kg) across metals, battery precursors, and ceramics
    COMMODITY_SPOT_PRICES_USD_KG: Dict[str, float] = {
        # Base & Structural Metals
        "Fe": 0.15,
        "Al": 2.45,
        "Cu": 9.20,
        "Ni": 16.50,
        "Cr": 9.80,
        "Ti": 8.50,
        "Mo": 45.00,
        "W": 38.00,
        "Nb": 48.00,
        "B": 4.50,
        # Alkali & Alkaline Earth Battery Metals
        "Li": 22.00,  # Lithium carbonate equivalent normalized
        "Na": 2.10,   # Abundant sodium precursor
        "Mg": 3.80,   # High-abundance magnesium
        "Ca": 3.20,
        "Zn": 2.80,
        "K": 2.50,
        # Transition & Post-Transition Ceramics / Polyanions
        "Zr": 32.00,
        "Sc": 3200.00,  # Expensive rare dopant penalty
        "Y": 35.00,
        "La": 12.00,
        "Si": 2.10,
        "P": 3.40,
        "S": 0.25,
        "O": 0.05,
        "Cl": 0.30,
        "Br": 4.20,
        "I": 45.00,
        "F": 3.80,
        "Se": 28.00,
        "Te": 65.00,
        "Ge": 1850.00,  # High cost penalty
    }

    # Precursor chemical synthesis / purity grade multipliers
    PURITY_GRADE_MULTIPLIERS: Dict[str, float] = {
        "technical_grade": 1.0,
        "battery_grade_99_9": 1.45,
        "semiconductor_grade_99_999": 3.80,
    }

    def compute_weighted_composition_cost(
        self,
        element_mass_fractions: Dict[str, float],
        purity_grade: str = "battery_grade_99_9",
        batch_size_kg: float = 1.0,
    ) -> Dict[str, float]:
        """Compute total weighted raw material cost per kg ($/kg) and total batch cost:

        Cost_raw = sum_i (w_i * Price_i) * Multiplier_purity
        """
        multiplier = self.PURITY_GRADE_MULTIPLIERS.get(purity_grade, 1.45)
        total_raw_cost_per_kg = 0.0
        cost_breakdown = {}

        for elem, w_i in element_mass_fractions.items():
            unit_price = self.COMMODITY_SPOT_PRICES_USD_KG.get(elem, 25.0)  # Default fallback
            elem_cost = w_i * unit_price
            total_raw_cost_per_kg += elem_cost
            cost_breakdown[elem] = float(elem_cost * multiplier)

        total_cost_per_kg = total_raw_cost_per_kg * multiplier
        total_batch_cost = total_cost_per_kg * batch_size_kg

        return {
            "raw_material_cost_usd_kg": float(total_cost_per_kg),
            "total_batch_cost_usd": float(total_batch_cost),
            "purity_multiplier_applied": float(multiplier),
            "is_low_cost_commercial_viable": bool(total_cost_per_kg <= 25.0),
        }
