"""Programmatic Agent Tool Interfaces for Economics, Supply Chain, Toxicity, and TEA."""

import re
from typing import Dict, List, Optional, Any
from penziv_materials.economics.commodity_pricing import CommodityPricingEngine
from penziv_materials.economics.supply_chain_risk import SupplyChainRiskEngine
from penziv_materials.economics.toxicity_ehs import ToxicityEHSEngine
from penziv_materials.economics.techno_economics import TechnoEconomicsEngine

_pricing_engine = CommodityPricingEngine()
_risk_engine = SupplyChainRiskEngine()
_toxicity_engine = ToxicityEHSEngine()
_tea_engine = TechnoEconomicsEngine()


def _parse_formula_to_mass_fractions(formula: str) -> Dict[str, float]:
    """Helper to convert chemical formula string into approximate elemental mass fractions."""
    atomic_masses = {
        "H": 1.008, "Li": 6.94, "Be": 9.012, "B": 10.81, "C": 12.011, "N": 14.007,
        "O": 15.999, "F": 18.998, "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085,
        "P": 30.974, "S": 32.06, "Cl": 35.45, "K": 39.098, "Ca": 40.078, "Sc": 44.956,
        "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Co": 58.933,
        "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.630, "As": 74.922,
        "Se": 78.971, "Br": 79.904, "Y": 88.906, "Zr": 91.224, "Nb": 92.906, "Mo": 95.95,
        "W": 183.84, "La": 138.91, "Ta": 180.95, "Tl": 204.38, "Pb": 207.2, "Cd": 112.41,
        "Hg": 200.59, "Te": 127.60, "I": 126.90,
    }
    tokens = re.findall(r"([A-Z][a-z]*)(\d*\.?\d*)", formula)
    if not tokens:
        return {"Mg": 0.5, "S": 0.5}

    elem_counts: Dict[str, float] = {}
    for elem, count_str in tokens:
        count = float(count_str) if count_str else 1.0
        elem_counts[elem] = elem_counts.get(elem, 0.0) + count

    total_mass = sum(count * atomic_masses.get(elem, 50.0) for elem, count in elem_counts.items())
    mass_fractions = {elem: (count * atomic_masses.get(elem, 50.0)) / max(1e-6, total_mass) for elem, count in elem_counts.items()}
    return mass_fractions


def get_composition_cost(elements: Dict[str, float], target_mass_kg: float = 1.0) -> Dict[str, Any]:
    """Calculates weighted raw material cost, spot volatility, and commercial precursor price.

    Args:
        elements: Dictionary of element mass fractions, e.g. {"Mg": 0.20, "Sc": 0.10, "Zr": 0.30, "S": 0.40}.
        target_mass_kg: Target production batch mass in kilograms.

    Returns:
        Dictionary containing $/kg raw cost, total batch cost, and commercial viability flags.
    """
    return _pricing_engine.compute_weighted_composition_cost(
        element_mass_fractions=elements,
        batch_size_kg=target_mass_kg,
    )


def evaluate_supply_chain_risk(elements: List[str]) -> Dict[str, Any]:
    """Returns HHI mining score, HHI refining score, and global recycling fraction.

    Args:
        elements: List of constituent chemical symbols, e.g. ["Mg", "Sc", "Zr", "S"].

    Returns:
        Dictionary containing weighted HHI mining/refining scores and USGS critical mineral flags.
    """
    equal_weight = 1.0 / max(1, len(elements))
    mass_fractions = {elem: equal_weight for elem in elements}
    return _risk_engine.evaluate_composition_supply_chain_risk(mass_fractions)


def evaluate_toxicity_and_regulations(chemical_formula: str, phases: Optional[List[str]] = None) -> Dict[str, Any]:
    """Returns GHS hazard classifications, REACH SVHC flags, and EPA CompTox hazard score.

    Args:
        chemical_formula: Stoichiometric chemical formula string (e.g. "Mg1.2Sc0.2Zr1.8S6").
        phases: Optional list of coexisting phases.

    Returns:
        Dictionary with regulatory compliance boolean, banned element list, and EPA hazard score.
    """
    mass_fractions = _parse_formula_to_mass_fractions(chemical_formula)
    return _toxicity_engine.evaluate_composition_ehs_and_carbon(mass_fractions)


def compute_techno_economic_lcos(material_params: Dict[str, Any], cell_architecture: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Calculates estimated $/kWh raw-material floor and processing thermal budget.

    Args:
        material_params: Dict containing raw_cost_usd_kg, sintering_temp_c, thickness_um.
        cell_architecture: Dict containing nominal_voltage_v, areal_capacity_mah_cm2.

    Returns:
        Dictionary containing $/kWh electrolyte floor and kWh/kg sintering thermal budget.
    """
    raw_cost = material_params.get("raw_material_cost_usd_kg", 25.0)
    thickness = material_params.get("thickness_um", 25.0)
    sinter_temp = material_params.get("sintering_temp_c", 850.0)

    cell = cell_architecture or {}
    voltage = cell.get("nominal_cell_voltage_v", 3.2)
    capacity = cell.get("cell_areal_capacity_mah_cm2", 4.0)

    lcos_res = _tea_engine.compute_electrolyte_lcos_floor(
        electrolyte_raw_cost_usd_kg=raw_cost,
        electrolyte_layer_thickness_um=thickness,
        cell_areal_capacity_mah_cm2=capacity,
        nominal_cell_voltage_v=voltage,
    )

    thermal_res = _tea_engine.compute_thermal_synthesis_energy_budget(
        sintering_temp_c=sinter_temp,
    )

    return {
        **lcos_res,
        **thermal_res,
        "combined_techno_economic_score": float(
            100.0 - lcos_res["electrolyte_cost_contribution_usd_kwh"] - thermal_res["synthesis_energy_cost_usd_kg"]
        ),
    }
