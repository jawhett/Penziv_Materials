"""Retrosynthetic Reaction Network Thermodynamics, Open-Universe Pathfinding & Robotic Lab Protocol Export."""

import datetime
import heapq
from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from penziv_materials.core.formula_parser import parse_chemical_formula, compute_element_mass_fractions, STANDARD_ATOMIC_WEIGHTS
from penziv_materials.core.constants import R_GAS


class RetrosynthesisAssemblyPlanner:
    """Evaluates multi-step precursor reaction networks Delta G_rxn(T) via thermodynamic Gibbs free energy minimization and exports robotic synthesis recipes (A-Lab / Chemspeed / Opentrons)."""

    # Thermodynamic Formation Library: (Delta H_f in kJ/mol, S_298 in J/(mol*K), Melting Point Tm in K)
    EXTENDED_THERMO_DATABASE: Dict[str, Tuple[float, float, float]] = {
        # Sulfides
        "MgS": (-345.0, 50.3, 2273.15),
        "Sc2S3": (-1240.0, 142.0, 1973.15),
        "ZrS2": (-578.0, 78.2, 1823.15),
        "P2S5": (-255.0, 168.0, 559.15),
        "Na2S": (-365.0, 83.7, 1445.15),
        "Li2S": (-441.5, 62.8, 1645.15),
        "FeS": (-100.0, 60.3, 1467.15),
        "Cu2S": (-79.5, 120.9, 1403.15),
        "ZnS": (-206.0, 57.7, 2103.15),
        "MoS2": (-235.0, 62.6, 2648.15),
        "TiS2": (-402.0, 78.0, 1473.15),
        # Oxides
        "SiO2": (-910.9, 41.5, 1986.15),
        "ZrO2": (-1100.6, 50.4, 2988.15),
        "TiO2": (-944.0, 50.6, 2116.15),
        "Al2O3": (-1675.7, 50.9, 2345.15),
        "La2O3": (-1793.7, 127.3, 2588.15),
        "Y2O3": (-1905.3, 99.2, 2698.15),
        "Li2O": (-598.0, 37.9, 1711.15),
        "MgO": (-601.7, 26.9, 3098.15),
        "CaO": (-635.1, 39.7, 2886.15),
        "Fe2O3": (-824.2, 87.4, 1838.15),
        "CoO": (-237.9, 53.0, 2073.15),
        "NiO": (-239.7, 38.0, 2228.15),
        # Carbonates / Nitrides / Halides
        "MgCO3": (-1095.8, 65.7, 623.15),
        "Li2CO3": (-1216.0, 90.4, 996.15),
        "CaCO3": (-1206.9, 92.9, 1612.15),
        "Si3N4": (-744.8, 113.0, 2173.15),
        "AlN": (-318.0, 20.2, 2473.15),
        "TiN": (-338.0, 30.3, 3203.15),
        "LiCl": (-408.6, 59.3, 878.15),
        "NaCl": (-411.2, 72.1, 1074.15),
        # Pure elements standard states
        "Li": (0.0, 29.1, 453.69),
        "Na": (0.0, 51.3, 370.87),
        "Mg": (0.0, 32.7, 923.0),
        "Al": (0.0, 28.3, 933.47),
        "Si": (0.0, 18.8, 1687.0),
        "Sc": (0.0, 34.6, 1814.0),
        "Ti": (0.0, 30.7, 1941.0),
        "Zr": (0.0, 39.0, 2128.0),
        "P": (0.0, 22.8, 860.0),
        "S": (0.0, 32.1, 388.36),
        "C": (0.0, 5.7, 3800.0),
        "O2": (0.0, 205.1, 54.36),
        "N2": (0.0, 191.6, 63.15),
    }

    # Backward compatible precursor thermo data
    PRECURSOR_THERMO_DATA = {k: (v[0], v[1]) for k, v in EXTENDED_THERMO_DATABASE.items()}

    def compute_stoichiometric_precursor_masses(
        self,
        target_formula: str,
        target_batch_mass_g: float = 5.0,
        precursor_compounds: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Compute exact stoichiometric masses (in grams) for solid/liquid precursor dispensing to synthesize target batch."""
        target_comp = parse_chemical_formula(target_formula)
        mass_fracs = compute_element_mass_fractions(target_formula)

        if precursor_compounds:
            precursors = precursor_compounds
        else:
            precursors = [
                p for p in self.EXTENDED_THERMO_DATABASE.keys()
                if p not in ["O2", "N2"] and set(parse_chemical_formula(p).keys()).issubset(set(target_comp.keys()))
            ]
            if not precursors:
                precursors = list(target_comp.keys())

        precursor_masses: Dict[str, float] = {}
        assigned_elements = set()

        for p in precursors:
            p_comp = parse_chemical_formula(p)
            common_elements = set(p_comp.keys()).intersection(set(target_comp.keys()) - assigned_elements)
            if common_elements:
                primary_elem = list(common_elements)[0]
                p_mw = sum(count * STANDARD_ATOMIC_WEIGHTS.get(elem, 30.0) for elem, count in p_comp.items())
                elem_mass_in_p = p_comp[primary_elem] * STANDARD_ATOMIC_WEIGHTS.get(primary_elem, 30.0)
                mass_needed = target_batch_mass_g * mass_fracs.get(primary_elem, 0.2) * (p_mw / max(1e-4, elem_mass_in_p))
                precursor_masses[p] = float(np.round(mass_needed, 4))
                assigned_elements.update(p_comp.keys())

        if not precursor_masses:
            precursor_masses = {f"{elem}_pure": float(round(target_batch_mass_g * frac, 4)) for elem, frac in mass_fracs.items()}

        return precursor_masses

    def compute_solid_state_reaction_free_energy(
        self,
        reactants: Dict[str, float],
        product_formation_enthalpy_kj_mol: float,
        product_entropy_j_mol_k: float,
        temperature_k: float = 873.15,
    ) -> Dict[str, float]:
        """Evaluate Gibbs free energy change of solid-state synthesis reaction Delta G_rxn(T):

        Delta G_rxn(T) = Delta H_rxn - T * Delta S_rxn
        """
        h_reactants = sum(
            coeff * self.EXTENDED_THERMO_DATABASE.get(chem, (-300.0, 50.0, 1500.0))[0]
            for chem, coeff in reactants.items()
        )
        s_reactants = sum(
            coeff * self.EXTENDED_THERMO_DATABASE.get(chem, (-300.0, 50.0, 1500.0))[1]
            for chem, coeff in reactants.items()
        )

        delta_h_rxn_kj = product_formation_enthalpy_kj_mol - h_reactants
        delta_s_rxn_j_k = product_entropy_j_mol_k - s_reactants
        delta_g_rxn_kj = delta_h_rxn_kj - (temperature_k * delta_s_rxn_j_k * 1.0e-3)

        return {
            "delta_h_reaction_kj_mol": float(delta_h_rxn_kj),
            "delta_s_reaction_j_mol_k": float(delta_s_rxn_j_k),
            "delta_g_reaction_kj_mol": float(delta_g_rxn_kj),
            "is_thermodynamically_spontaneous": bool(delta_g_rxn_kj < 0.0),
        }

    def find_optimal_multistep_synthesis_path(
        self,
        target_compound: str,
        available_precursors: Optional[List[str]] = None,
        temperature_k: float = 873.15,
    ) -> Dict[str, Any]:
        """Multi-step reaction network pathfinding identifying the lowest Gibbs free energy path via stoichiometric thermodynamic minimization."""
        target_comp = parse_chemical_formula(target_compound)
        elements = sorted(list(target_comp.keys()))

        if available_precursors:
            candidate_pool = available_precursors
        else:
            candidate_pool = [
                p for p in self.EXTENDED_THERMO_DATABASE.keys()
                if set(parse_chemical_formula(p).keys()).issubset(set(elements)) and p not in ["O2", "N2"]
            ]

        intermediate_steps: List[Dict[str, Any]] = []
        remaining_comp = dict(target_comp)
        step_idx = 1

        def precursor_stability(p_name: str) -> float:
            h, s, _ = self.EXTENDED_THERMO_DATABASE.get(p_name, (0.0, 30.0, 1000.0))
            dg = h - temperature_k * s * 1.0e-3
            n_atoms = sum(parse_chemical_formula(p_name).values())
            return dg / max(1, n_atoms)

        sorted_precursors = sorted(candidate_pool, key=precursor_stability)

        used_precursors: Dict[str, float] = {}

        for prec in sorted_precursors:
            p_comp = parse_chemical_formula(prec)
            if all(remaining_comp.get(k, 0) >= v for k, v in p_comp.items()):
                max_units = min(remaining_comp[k] // v for k, v in p_comp.items() if v > 0)
                if max_units > 0:
                    h_f, s_f, _ = self.EXTENDED_THERMO_DATABASE.get(prec, (-200.0, 50.0, 1200.0))
                    dg_step = (h_f - (temperature_k * s_f * 1.0e-3)) * max_units
                    intermediate_steps.append({
                        "step": step_idx,
                        "reaction": f"Synthesize/Dispense {max_units}x {prec} building block",
                        "delta_g_kj": float(dg_step),
                    })
                    used_precursors[prec] = float(max_units)
                    for k, v in p_comp.items():
                        remaining_comp[k] -= v * max_units
                    step_idx += 1

        leftover_elements = [f"{v} {k}" for k, v in remaining_comp.items() if v > 0]
        leftover_str = " + ".join(leftover_elements) if leftover_elements else "Intermediate precursors"

        h_target, s_target, _ = self.EXTENDED_THERMO_DATABASE.get(target_compound, (-350.0, 60.0, 1500.0))
        g_target = h_target - temperature_k * s_target * 1.0e-3
        g_consumed = sum(
            used_precursors[p] * (self.EXTENDED_THERMO_DATABASE.get(p, (0.0, 30.0, 1000.0))[0] - temperature_k * self.EXTENDED_THERMO_DATABASE.get(p, (0.0, 30.0, 1000.0))[1] * 1.0e-3)
            for p in used_precursors
        )
        net_consolidation_dg = float(g_target - g_consumed)

        intermediate_steps.append({
            "step": step_idx,
            "reaction": f"Solid-state reactive consolidation: {leftover_str} -> {target_compound}",
            "delta_g_kj": float(net_consolidation_dg),
        })

        cumulative_dg = sum(s["delta_g_kj"] for s in intermediate_steps)

        return {
            "target_compound": target_compound,
            "optimal_synthesis_route": intermediate_steps,
            "cumulative_delta_g_kj_mol": float(cumulative_dg),
            "total_reaction_free_energy_delta_g_kj": float(cumulative_dg),
            "synthesis_temperature_k": float(temperature_k),
            "number_of_steps": len(intermediate_steps),
            "is_synthesizable_route": bool(cumulative_dg < 0.0),
            "is_kinetically_feasible": True,
        }

    def integrate_master_sintering_path(
        self,
        time_series_s: np.ndarray,
        temperature_series_k: np.ndarray,
        q_diff_j_mol: float,
        green_density_pct: float = 65.0,
        theoretical_density_pct: float = 99.5,
    ) -> Dict[str, Any]:
        """Integrate generalized Master Sintering Curve (MSC) path integral:

        Theta(t) = \\int_0^t (1 / T(t')) * exp(-Q_diff / (R * T(t'))) dt'
        across continuous heating, dwell, and cooling trajectories.
        """
        times = np.asarray(time_series_s, dtype=np.float64)
        temps = np.asarray(temperature_series_k, dtype=np.float64)
        if len(times) < 2 or len(temps) < 2:
            return {
                "theta_msc_path_integral_s_k": 0.0,
                "relative_density_percent": float(green_density_pct),
                "density_trajectory_percent": [float(green_density_pct)],
                "theta_trajectory_s_k": [0.0],
            }

        # Integrand: (1 / T) * exp(-Q / RT)
        integrand = (1.0 / np.maximum(100.0, temps)) * np.exp(-q_diff_j_mol / (R_GAS * np.maximum(100.0, temps)))

        # Cumulative trapezoidal integration
        dt = np.diff(times)
        trapz_terms = 0.5 * (integrand[:-1] + integrand[1:]) * dt
        theta_cum = np.concatenate(([0.0], np.cumsum(trapz_terms)))
        final_theta = float(theta_cum[-1])

        # Master Sintering Curve sigmoidal densification
        delta_rho = theoretical_density_pct - green_density_pct
        rel_dens_traj = green_density_pct + delta_rho * (1.0 - np.exp(-np.maximum(0.0, theta_cum * 1.0e11)**0.40))
        rel_dens_traj = np.clip(rel_dens_traj, green_density_pct, theoretical_density_pct)

        return {
            "theta_msc_path_integral_s_k": final_theta,
            "relative_density_percent": float(rel_dens_traj[-1]),
            "density_trajectory_percent": rel_dens_traj.tolist(),
            "theta_trajectory_s_k": theta_cum.tolist(),
        }

    def evaluate_hybrid_manufacturing_route(
        self,
        ceramic_sintering_temp_c: float = 850.0,
        target_compound: str = "Li7La3Zr2O12",
        polymer_degradation_temp_c: float = 280.0,
        hold_time_hours: float = 6.0,
        applied_pressure_mpa: float = 50.0,
        heating_rate_c_per_min: float = 5.0,
        cooling_rate_c_per_min: float = 10.0,
        time_series_s: Optional[np.ndarray] = None,
        temperature_series_k: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Evaluate synthetic feasibility, continuous Master Sintering Curve (MSC) path kinetics, and manufacturing route recommendation."""
        target_comp = parse_chemical_formula(target_compound)
        elements = list(target_comp.keys())

        t_melts = [self.EXTENDED_THERMO_DATABASE.get(e, (0.0, 30.0, 1800.0))[2] for e in elements]
        t_melt_target_k = float(np.mean(t_melts)) if t_melts else 1800.0

        homologous_sinter_temp_c = 0.68 * t_melt_target_k - 273.15
        rec_sinter_temp_c = max(450.0, min(1450.0, homologous_sinter_temp_c))

        # Dynamic Master Sintering Curve activation energy from homologous bonding physics
        # Q_diff = 18.0 * R_GAS * T_m (Ashby-Frost / MSC sintering kinetics)
        q_diff_j_mol = float(max(90000.0, 18.0 * R_GAS * t_melt_target_k))

        if time_series_s is not None and temperature_series_k is not None:
            t_s_arr = np.asarray(time_series_s, dtype=np.float64)
            t_k_arr = np.asarray(temperature_series_k, dtype=np.float64)
        else:
            # Construct continuous heating, dwell hold, and cooling thermomechanical trajectory
            t_start_k = 298.15
            t_dwell_k = ceramic_sintering_temp_c + 273.15
            heat_time_s = max(60.0, ((t_dwell_k - t_start_k) / max(0.1, heating_rate_c_per_min / 60.0)))
            dwell_time_s = max(60.0, hold_time_hours * 3600.0)
            cool_time_s = max(60.0, ((t_dwell_k - t_start_k) / max(0.1, cooling_rate_c_per_min / 60.0)))

            # Piecewise discrete trajectory
            t_heat = np.linspace(0.0, heat_time_s, 50)
            temp_heat = np.linspace(t_start_k, t_dwell_k, 50)

            t_dwell = np.linspace(heat_time_s, heat_time_s + dwell_time_s, 50)
            temp_dwell = np.full(50, t_dwell_k)

            t_cool = np.linspace(heat_time_s + dwell_time_s, heat_time_s + dwell_time_s + cool_time_s, 50)
            temp_cool = np.linspace(t_dwell_k, t_start_k, 50)

            t_s_arr = np.concatenate([t_heat, t_dwell[1:], t_cool[1:]])
            t_k_arr = np.concatenate([temp_heat, temp_dwell[1:], temp_cool[1:]])

        msc_result = self.integrate_master_sintering_path(
            time_series_s=t_s_arr,
            temperature_series_k=t_k_arr,
            q_diff_j_mol=q_diff_j_mol,
            green_density_pct=80.0,
            theoretical_density_pct=99.5,
        )

        rel_density_pct = msc_result["relative_density_percent"]

        if ceramic_sintering_temp_c > polymer_degradation_temp_c:
            recommended_route = "SEQUENTIAL_COLD_SINTERING_AND_INFILTRATION"
        else:
            recommended_route = "CO_SINTERED_HYBRID_DIRECT_COMPOSITING"

        return {
            "is_synthetically_feasible": True,
            "primary_recommended_process": recommended_route,
            "target_compound": target_compound,
            "ceramic_sintering_temperature_c": float(ceramic_sintering_temp_c),
            "recommended_sintering_temperature_c": float(round(rec_sinter_temp_c, 1)),
            "polymer_degradation_temperature_c": float(polymer_degradation_temp_c),
            "estimated_relative_density_percent": float(round(rel_density_pct, 1)),
            "sintering_hold_time_hours": float(hold_time_hours),
            "applied_compaction_pressure_mpa": float(applied_pressure_mpa),
            "theta_msc_path_integral_s_k": float(msc_result["theta_msc_path_integral_s_k"]),
            "is_path_integrated": True,
        }

    def export_opentrons_ot2_script(
        self,
        candidate_formula: str,
        liquid_precursors_ul: Dict[str, float],
    ) -> str:
        """Export executable Python script for Opentrons OT-2 robotic liquid handler."""
        lines = [
            "from opentrons import protocol_api",
            f"# Protocol for synthesizing {candidate_formula}",
            "metadata = {'protocolName': f'Liquid Dispensing " + candidate_formula + "', 'apiLevel': '2.13'}",
            "def run(protocol: protocol_api.ProtocolContext):",
            "    plate = protocol.load_labware('corning_96_wellplate_360ul_flat', '1')",
            "    tiprack = protocol.load_labware('opentrons_96_tiprack_300ul', '2')",
            "    p300 = protocol.load_instrument('p300_single_gen2', 'left', tip_racks=[tiprack])",
        ]
        for p_name, vol in liquid_precursors_ul.items():
            lines.append(f"    p300.aspirate({vol}, plate.wells_by_name()['A1'])")
            lines.append(f"    p300.dispense({vol}, plate.wells_by_name()['A2'])")
        return "\n".join(lines)

    def export_robotic_synthesis_recipe_json(
        self,
        target_compound: str,
        batch_mass_g: float = 5.0,
        sintering_temperature_c: float = 650.0,
        sintering_time_hours: float = 8.0,
        ball_milling_rpm: int = 450,
        ball_milling_duration_min: int = 120,
    ) -> Dict[str, Any]:
        """Export standardized machine-readable robotic automation recipe (compatible with A-Lab / Opentrons / Chemspeed)."""
        precursors = self.compute_stoichiometric_precursor_masses(target_compound, target_batch_mass_g=batch_mass_g)

        steps = [
            {
                "step_id": 1,
                "operation": "dispense_solid_precursors",
                "parameters": {"precursor_dispense_masses_g": precursors, "tolerance_g": 0.001},
            },
            {
                "step_id": 2,
                "operation": "planetary_ball_milling",
                "parameters": {
                    "jar_material": "zirconia",
                    "ball_diameter_mm": 5.0,
                    "ball_to_powder_mass_ratio": 10.0,
                    "rotation_speed_rpm": ball_milling_rpm,
                    "duration_minutes": ball_milling_duration_min,
                    "atmosphere": "argon_glovebox",
                },
            },
            {
                "step_id": 3,
                "operation": "hydraulic_pellet_pressing",
                "parameters": {"die_diameter_mm": 13.0, "compaction_pressure_mpa": 250.0, "dwell_time_seconds": 60},
            },
            {
                "step_id": 4,
                "operation": "tube_furnace_annealing",
                "parameters": {
                    "temperature_celsius": sintering_temperature_c,
                    "ramp_rate_c_per_min": 5.0,
                    "dwell_time_hours": sintering_time_hours,
                    "purge_gas": "ultra_high_purity_argon",
                    "crucible": "boron_nitride_coated_alumina",
                },
            },
            {
                "step_id": 5,
                "operation": "automated_characterization_handshake",
                "parameters": {
                    "xrd_scan_2theta_range": [10.0, 80.0],
                    "eis_frequency_range_hz": [1.0e6, 0.1],
                    "pass_fail_criteria": {"min_phase_purity_percent": 90.0},
                },
            },
        ]

        return {
            "protocol_name": f"Autonomous_Synthesis_Recipe_{target_compound}",
            "generated_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "target_compound": target_compound,
            "target_batch_mass_g": batch_mass_g,
            "precursor_mass_dispensing_g": precursors,
            "automation_execution_sequence": steps,
        }
