"""Environmental Health & Safety (EHS), EPA CompTox, REACH SVHC & Embodied Carbon Engine."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np


class ToxicityEHSEngine:
    """Evaluates regulatory compliance (REACH SVHC, GHS, EPA CompTox) and embodied carbon footprints."""

    # Explicitly banned toxic / restricted heavy metals (Zero tolerance pre-compute filter)
    STRICTLY_BANNED_TOXIC_ELEMENTS = {"Tl", "Cd", "As", "Hg", "Pb", "Be", "U", "Th", "Pu"}

    # GHS Hazard Classifications & REACH SVHC flags
    ELEMENT_EHS_REGISTRY: Dict[str, Dict[str, Any]] = {
        "Mg": {"ghs_codes": ["H228 (Flammable Solid)"], "svhc": False, "epa_hazard_score": 1.2, "embodied_co2_kg_kg": 14.5},
        "Na": {"ghs_codes": ["H260 (Water-Reactive Cat 1)"], "svhc": False, "epa_hazard_score": 2.1, "embodied_co2_kg_kg": 4.8},
        "Li": {"ghs_codes": ["H314 (Skin Corr 1B)"], "svhc": False, "epa_hazard_score": 3.4, "embodied_co2_kg_kg": 18.2},
        "Ca": {"ghs_codes": ["H261 (Water-Reactive)"], "svhc": False, "epa_hazard_score": 1.5, "embodied_co2_kg_kg": 3.2},
        "Zn": {"ghs_codes": ["H410 (Aquatic Acute 1)"], "svhc": False, "epa_hazard_score": 2.0, "embodied_co2_kg_kg": 3.6},
        "Ni": {"ghs_codes": ["H351 (Carc 2)", "H372 (STOT RE 1)"], "svhc": True, "epa_hazard_score": 4.8, "embodied_co2_kg_kg": 12.8},
        "Co": {"ghs_codes": ["H350 (Carc 1B)", "H360F (Repr 1B)"], "svhc": True, "epa_hazard_score": 5.0, "embodied_co2_kg_kg": 24.5},
        "Cr": {"ghs_codes": ["H317 (Skin Sens 1)"], "svhc": False, "epa_hazard_score": 2.8, "embodied_co2_kg_kg": 6.5},
        "Ti": {"ghs_codes": ["Non-Hazardous"], "svhc": False, "epa_hazard_score": 0.5, "embodied_co2_kg_kg": 8.4},
        "Sc": {"ghs_codes": ["Non-Hazardous"], "svhc": False, "epa_hazard_score": 1.0, "embodied_co2_kg_kg": 85.0},
        "Zr": {"ghs_codes": ["H250 (Pyr Sol 1)"], "svhc": False, "epa_hazard_score": 1.8, "embodied_co2_kg_kg": 9.2},
        "Si": {"ghs_codes": ["Non-Hazardous"], "svhc": False, "epa_hazard_score": 0.4, "embodied_co2_kg_kg": 5.5},
        "P":  {"ghs_codes": ["H228 (Flammable Solid)"], "svhc": False, "epa_hazard_score": 2.5, "embodied_co2_kg_kg": 4.2},
        "S":  {"ghs_codes": ["H315 (Skin Irrit 2)"], "svhc": False, "epa_hazard_score": 1.1, "embodied_co2_kg_kg": 0.3},
        "Se": {"ghs_codes": ["H301 (Acute Tox 3)", "H373 (STOT RE 2)"], "svhc": False, "epa_hazard_score": 3.8, "embodied_co2_kg_kg": 15.0},
        "Al": {"ghs_codes": ["Non-Hazardous"], "svhc": False, "epa_hazard_score": 0.6, "embodied_co2_kg_kg": 11.2},
        "Fe": {"ghs_codes": ["Non-Hazardous"], "svhc": False, "epa_hazard_score": 0.3, "embodied_co2_kg_kg": 1.8},
    }

    def evaluate_composition_ehs_and_carbon(
        self,
        element_mass_fractions: Dict[str, float],
    ) -> Dict[str, Any]:
        """Screen for strictly banned elements, evaluate EPA CompTox score, REACH SVHCs, and embodied CO2."""
        elements_set = set(element_mass_fractions.keys())
        banned_found = elements_set.intersection(self.STRICTLY_BANNED_TOXIC_ELEMENTS)

        if banned_found:
            return {
                "is_regulatory_compliant": False,
                "banned_elements_detected": list(banned_found),
                "rejection_reason": f"Contains strictly banned toxic species: {', '.join(banned_found)}",
                "epa_comptox_hazard_score": 10.0,
                "embodied_carbon_kg_co2_kg": 999.0,
            }

        weighted_comptox = 0.0
        weighted_carbon_co2 = 0.0
        active_ghs_codes = set()
        svhc_present = []

        for elem, w_i in element_mass_fractions.items():
            ehs = self.ELEMENT_EHS_REGISTRY.get(
                elem,
                {"ghs_codes": ["Non-Hazardous"], "svhc": False, "epa_hazard_score": 1.5, "embodied_co2_kg_kg": 8.0},
            )
            weighted_comptox += w_i * ehs["epa_hazard_score"]
            weighted_carbon_co2 += w_i * ehs["embodied_co2_kg_kg"]

            if w_i > 0.03:
                for code in ehs["ghs_codes"]:
                    if code != "Non-Hazardous":
                        active_ghs_codes.add(code)
                if ehs["svhc"]:
                    svhc_present.append(elem)

        # Environmental compliance flag: Low toxicity (hazard < 3.5) and no SVHC above threshold
        is_eco_friendly = weighted_comptox < 3.5 and len(svhc_present) == 0

        return {
            "is_regulatory_compliant": True,
            "banned_elements_detected": [],
            "epa_comptox_hazard_score": float(weighted_comptox),
            "embodied_carbon_kg_co2_kg": float(weighted_carbon_co2),
            "reach_svhc_substances": svhc_present,
            "ghs_hazard_classifications": list(active_ghs_codes),
            "is_eco_friendly": is_eco_friendly,
        }
