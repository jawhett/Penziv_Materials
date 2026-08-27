"""Supply Chain Criticality, USGS Critical Minerals & Herfindahl-Hirschman Index (HHI)."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np


class SupplyChainRiskEngine:
    """Evaluates mining & refining geopolitical concentration HHI = sum(s_i^2) and USGS/DOE/EU critical mineral risk."""

    # USGS & World Bank Geopolitical Market Concentration data (s_i in percentages, HHI in [0, 10000])
    # HHI < 1500: Unconcentrated, 1500-2500: Moderate, > 2500: Highly concentrated / extreme risk
    CRITICAL_MINERAL_REGISTRY: Dict[str, Dict[str, Any]] = {
        "Li": {"hhi_mining": 3100, "hhi_refining": 5800, "doe_critical": True, "byproduct": False, "recycling_fraction": 0.05},
        "Na": {"hhi_mining": 850, "hhi_refining": 920, "doe_critical": False, "byproduct": False, "recycling_fraction": 0.40},
        "Mg": {"hhi_mining": 6400, "hhi_refining": 7100, "doe_critical": True, "byproduct": False, "recycling_fraction": 0.35},
        "Ca": {"hhi_mining": 600, "hhi_refining": 750, "doe_critical": False, "byproduct": False, "recycling_fraction": 0.50},
        "Zn": {"hhi_mining": 1400, "hhi_refining": 1800, "doe_critical": False, "byproduct": False, "recycling_fraction": 0.45},
        "Ni": {"hhi_mining": 2200, "hhi_refining": 3400, "doe_critical": True, "byproduct": False, "recycling_fraction": 0.40},
        "Co": {"hhi_mining": 5200, "hhi_refining": 6800, "doe_critical": True, "byproduct": True, "recycling_fraction": 0.30},
        "Sc": {"hhi_mining": 6800, "hhi_refining": 8200, "doe_critical": True, "byproduct": True, "recycling_fraction": 0.01},
        "Y":  {"hhi_mining": 7500, "hhi_refining": 8900, "doe_critical": True, "byproduct": True, "recycling_fraction": 0.02},
        "Zr": {"hhi_mining": 2800, "hhi_refining": 3200, "doe_critical": True, "byproduct": False, "recycling_fraction": 0.10},
        "Ti": {"hhi_mining": 1900, "hhi_refining": 2600, "doe_critical": True, "byproduct": False, "recycling_fraction": 0.40},
        "Si": {"hhi_mining": 4800, "hhi_refining": 5200, "doe_critical": False, "byproduct": False, "recycling_fraction": 0.25},
        "P":  {"hhi_mining": 3100, "hhi_refining": 3600, "doe_critical": False, "byproduct": False, "recycling_fraction": 0.15},
        "S":  {"hhi_mining": 750, "hhi_refining": 800, "doe_critical": False, "byproduct": True, "recycling_fraction": 0.80},
        "Al": {"hhi_mining": 1800, "hhi_refining": 3200, "doe_critical": True, "byproduct": False, "recycling_fraction": 0.50},
        "Fe": {"hhi_mining": 1200, "hhi_refining": 1400, "doe_critical": False, "byproduct": False, "recycling_fraction": 0.60},
        "Ge": {"hhi_mining": 5800, "hhi_refining": 7400, "doe_critical": True, "byproduct": True, "recycling_fraction": 0.05},
        "Nb": {"hhi_mining": 7800, "hhi_refining": 8100, "doe_critical": True, "byproduct": False, "recycling_fraction": 0.15},
    }

    def evaluate_composition_supply_chain_risk(
        self,
        element_mass_fractions: Dict[str, float],
    ) -> Dict[str, Any]:
        """Evaluate weighted HHI mining & refining concentration, critical mineral flags, and byproduct dependency:

        HHI_weighted = sum_i (w_i * HHI_i)
        """
        weighted_hhi_mining = 0.0
        weighted_hhi_refining = 0.0
        weighted_recycling = 0.0
        critical_elements_present = []
        byproduct_elements_present = []

        for elem, w_i in element_mass_fractions.items():
            reg = self.CRITICAL_MINERAL_REGISTRY.get(
                elem,
                {"hhi_mining": 2000, "hhi_refining": 2500, "doe_critical": False, "byproduct": False, "recycling_fraction": 0.20},
            )
            weighted_hhi_mining += w_i * reg["hhi_mining"]
            weighted_hhi_refining += w_i * reg["hhi_refining"]
            weighted_recycling += w_i * reg["recycling_fraction"]

            if reg["doe_critical"] and w_i > 0.02:
                critical_elements_present.append(elem)
            if reg["byproduct"] and w_i > 0.02:
                byproduct_elements_present.append(elem)

        # Risk categories: Low (<1500), Medium (1500-3500), High (3500-6000), Extreme (>6000)
        risk_level = (
            "LOW" if weighted_hhi_refining < 1500 else (
                "MODERATE" if weighted_hhi_refining < 3500 else (
                    "HIGH" if weighted_hhi_refining < 6000 else "EXTREME"
                )
            )
        )

        return {
            "weighted_hhi_mining": float(weighted_hhi_mining),
            "weighted_hhi_refining": float(weighted_hhi_refining),
            "weighted_recycling_fraction": float(weighted_recycling),
            "critical_minerals_detected": critical_elements_present,
            "byproduct_dependent_elements": byproduct_elements_present,
            "supply_disruption_risk_level": risk_level,
            "is_geopolitically_resilient": bool(weighted_hhi_refining <= 3500),
        }
