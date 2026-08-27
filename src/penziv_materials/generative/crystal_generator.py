"""Generative Crystal & Off-Stoichiometric Superionic Framework Synthesizer."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np


class GenerativeCrystalSynthesizer:
    """Proposes novel unlisted crystal frameworks, off-stoichiometric site disorder, and polyanion networks."""

    def __init__(self, target_carrier_cation: str = "Mg"):
        self.carrier = target_carrier_cation

    def generate_off_stoichiometric_superionic_candidate(
        self,
        framework_archetype: str = "Thio-NASICON",
        doping_element: str = "Sc",
        doping_fraction: float = 0.20,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """Propose off-stoichiometric interstitial/vacancy tuned framework formula and Wyckoff occupation:

        e.g., Mg_{1 + x} Sc_x Ti_{2 - x} (PO4)3  or  Na_{3 + x} Sc_x Zr_{2 - x} Si2 P O12
        """
        np.random.seed(random_seed)

        if self.carrier == "Mg":
            # Multivalent sulfide / selenide / phosphate framework with aliovalent dopant
            # Baseline: MgZr4(PO4)6 -> aliovalent Sc3+ doping creates extra mobile Mg2+ interstitials
            x = float(np.clip(doping_fraction, 0.05, 0.40))
            mg_content = 1.0 + 0.5 * x
            formula = f"Mg{mg_content:.2f}Sc{x:.2f}Zr{2.0 - x:.2f}(PS4)3"
            space_group = "R-3c"
            anion = "S"
            bottleneck_radius_angstrom = 2.45 + 0.3 * np.random.rand()
        else:
            # Na superionic conductor (NASICON)
            x = float(np.clip(doping_fraction, 0.05, 0.50))
            na_content = 3.0 + x
            formula = f"Na{na_content:.2f}Sc{x:.2f}Zr{2.0 - x:.2f}Si2PO12"
            space_group = "C2/c"
            anion = "O"
            bottleneck_radius_angstrom = 2.60 + 0.2 * np.random.rand()

        # Fractional coordinates of mobile carrier Wyckoff sites
        wyckoff_positions = [
            {"site": "18e", "element": self.carrier, "occupancy": 0.65, "coord": [0.33, 0.0, 0.25]},
            {"site": "6b", "element": self.carrier, "occupancy": 0.35, "coord": [0.0, 0.0, 0.0]},
            {"site": "12c", "element": doping_element, "occupancy": x / 2.0, "coord": [0.0, 0.0, 0.14]},
        ]

        return {
            "candidate_formula": formula,
            "framework_archetype": framework_archetype,
            "space_group": space_group,
            "target_carrier": self.carrier,
            "anion_type": anion,
            "bottleneck_radius_angstrom": float(bottleneck_radius_angstrom),
            "wyckoff_sites": wyckoff_positions,
            "mobile_cation_excess_x": float(x),
        }
