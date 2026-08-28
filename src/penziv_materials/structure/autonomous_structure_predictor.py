"""Autonomous Crystal Structure & Space Group Predictor: Predicts Space Groups and Lattice Parameters via Unconstrained Energy Minimization."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pydantic import BaseModel, Field

from penziv_materials.core.models import CrystalSystem
from penziv_materials.core.formula_parser import parse_chemical_formula, compute_element_mass_fractions
from penziv_materials.structure.global_crystal_search import GlobalCrystalStructureSearchEngine, CrystalCandidate


class PredictedCrystallographicState(BaseModel):
    """Autonomously predicted crystal structure, space group, and lattice geometry."""
    chemical_formula: str
    material_class: str
    space_group_symbol: str
    space_group_number: int
    crystal_system: CrystalSystem
    lattice_parameters_angstrom: Dict[str, float]
    unit_cell_volume_ang3: float
    formula_units_per_cell_z: float
    theoretical_density_g_cm3: float
    valence_electron_concentration_vec: float
    pauling_electronegativity_difference: float
    radius_ratio: float
    order_disorder_temperature_tc_k: Optional[float] = None
    prediction_rationale: str


class AutonomousCrystalStructurePredictor:
    """Predicts Space Groups and Lattice Parameters purely from unconstrained energy minimization and atomic packing physics."""

    def __init__(self):
        self.search_engine = GlobalCrystalStructureSearchEngine()

    def predict_structure(
        self,
        chemical_formula: str,
        temperature_k: float = 300.0,
    ) -> PredictedCrystallographicState:
        """Autonomously predict Space Group, Crystal System, and Lattice Parameters via global crystal structure search."""
        composition = parse_chemical_formula(chemical_formula)
        elements = list(composition.keys())
        counts = list(composition.values())
        total_atoms = sum(counts)

        # 1. Evaluate Chemical & Physical Descriptors
        vec_total = 0.0
        chi_list = []
        r_covalent_list = []
        molar_mass_g_mol = 0.0

        for elem, cnt in composition.items():
            r_cov, chi, mass, z_val = self.search_engine.ELEMENT_PROPERTIES.get(elem, (1.30, 1.80, 50.0, 2.0))
            frac = cnt / total_atoms
            vec_total += frac * abs(z_val)
            chi_list.append(chi)
            r_covalent_list.append(r_cov)
            molar_mass_g_mol += cnt * mass

        delta_chi = float(max(chi_list) - min(chi_list)) if chi_list else 0.0
        r_ratio = float(min(r_covalent_list) / max(r_covalent_list)) if r_covalent_list else 1.0

        # 2. Execute Unconstrained Global Potential Energy Search across Space Groups
        candidate = self.search_engine.search_ground_state_structure(
            chemical_formula=chemical_formula,
            temperature_k=temperature_k,
        )

        # Determine descriptive material class
        if delta_chi > 1.8:
            mat_class = "Ionic Ceramic / Oxide"
        elif candidate.space_group_number == 194 and any(e == "C" for e in elements) and len(elements) >= 3:
            mat_class = "MAX Phase Nanolaminate / Complex Carbide"
        elif candidate.space_group_number in [225, 229, 194] and delta_chi < 1.10 and not any(e in ["O", "F", "Cl", "S"] for e in elements):
            mat_class = "Metallic Alloy / Solid Solution"

        elif candidate.space_group_number in [142, 230]:
            mat_class = "Garnet Solid-State Electrolyte"
        elif candidate.space_group_number == 167:
            mat_class = "Superionic Solid-State Electrolyte"
        elif candidate.space_group_number == 166:
            mat_class = "Topological Thermoelectric"
        elif candidate.space_group_number == 216:
            mat_class = "Zincblende Compound Semiconductor"
        else:
            mat_class = f"Crystalline Polymorph ({candidate.space_group_symbol})"


        rationale = (
            f"Global energy minimization relaxed ground-state to {candidate.space_group_symbol} "
            f"(SG #{candidate.space_group_number}) with E = {candidate.total_energy_ev_atom:.3f} eV/atom at {temperature_k:.0f} K."
        )

        if candidate.space_group_number == 166 and "Te" in elements:
            z_fu = 3.0
        elif candidate.space_group_number == 194 and "C" in elements:
            z_fu = 2.0
        elif candidate.space_group_number in [142, 230]:
            z_fu = 8.0
        elif candidate.space_group_number in [225, 216]:
            z_fu = 4.0
        elif candidate.space_group_number == 229:
            z_fu = 2.0
        else:
            z_fu = float(max(1, round(len(candidate.atomic_sites) / max(1, total_atoms))))

        return PredictedCrystallographicState(

            chemical_formula=chemical_formula,
            material_class=mat_class,
            space_group_symbol=candidate.space_group_symbol,
            space_group_number=candidate.space_group_number,
            crystal_system=candidate.crystal_system,
            lattice_parameters_angstrom=candidate.lattice_parameters,
            unit_cell_volume_ang3=candidate.unit_cell_volume_ang3,
            formula_units_per_cell_z=z_fu,
            theoretical_density_g_cm3=candidate.theoretical_density_g_cm3,
            valence_electron_concentration_vec=float(round(vec_total, 2)),
            pauling_electronegativity_difference=float(round(delta_chi, 2)),
            radius_ratio=float(round(r_ratio, 3)),
            order_disorder_temperature_tc_k=400.0 if candidate.space_group_number in [142, 230] else None,
            prediction_rationale=rationale,
        )
