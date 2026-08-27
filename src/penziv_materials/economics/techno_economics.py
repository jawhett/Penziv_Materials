"""Techno-Economic Analysis (TEA), Levelized Cost of Storage (LCOS) & Synthesis Energy Budget."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np


class TechnoEconomicsEngine:
    """Calculates battery-cell $/kWh raw material floors, thermal sintering energy, and manufacturing overheads."""

    def __init__(self):
        pass

    def compute_electrolyte_lcos_floor(
        self,
        electrolyte_raw_cost_usd_kg: float,
        electrolyte_layer_thickness_um: float = 25.0,
        electrolyte_density_g_cm3: float = 2.4,
        cell_areal_capacity_mah_cm2: float = 4.0,
        nominal_cell_voltage_v: float = 3.2,
    ) -> Dict[str, float]:
        """Compute electrolyte raw-material cost contribution per kWh of battery capacity ($/kWh):

        Mass_electrolyte = thickness * Area * density
        Energy_cell = Areal_capacity * Voltage
        Cost_per_kWh = (Mass_electrolyte * Cost_raw_kg) / Energy_kWh
        """
        # Mass of electrolyte per cm2 (g/cm2)
        thickness_cm = electrolyte_layer_thickness_um * 1.0e-4
        mass_g_cm2 = thickness_cm * electrolyte_density_g_cm3
        mass_kg_cm2 = mass_g_cm2 * 1.0e-3

        # Energy per cm2 (Wh/cm2)
        energy_wh_cm2 = (cell_areal_capacity_mah_cm2 * 1.0e-3) * nominal_cell_voltage_v
        energy_kwh_cm2 = energy_wh_cm2 * 1.0e-3

        cost_usd_kwh = (mass_kg_cm2 * electrolyte_raw_cost_usd_kg) / max(1e-9, energy_kwh_cm2)

        return {
            "electrolyte_cost_contribution_usd_kwh": float(cost_usd_kwh),
            "electrolyte_areal_mass_g_cm2": float(mass_g_cm2),
            "is_below_target_floor_50_usd_kwh": bool(cost_usd_kwh <= 50.0),
        }

    def compute_thermal_synthesis_energy_budget(
        self,
        sintering_temp_c: float,
        sintering_time_hours: float = 4.0,
        furnace_efficiency: float = 0.45,
    ) -> Dict[str, float]:
        """Evaluate electric energy required to synthesize/sinter 1 kg of material (kWh/kg) and associated cost ($/kg):

        E_therm = (C_p * Delta T + Q_sinter) / eta_furnace
        """
        delta_t = max(0.0, sintering_temp_c - 25.0)
        c_p_j_kg_k = 800.0  # J/(kg·K) average ceramic specific heat
        e_sensible_j = c_p_j_kg_k * delta_t
        # Radiation & holding loss ~ 250 W/kg * time
        e_holding_j = 250.0 * (sintering_time_hours * 3600.0)

        total_energy_j = (e_sensible_j + e_holding_j) / furnace_efficiency
        energy_kwh_kg = total_energy_j / 3.6e6  # Joules to kWh

        # Commercial electricity rate ~$0.08 / kWh
        synthesis_energy_cost_usd_kg = energy_kwh_kg * 0.08

        return {
            "synthesis_energy_kwh_kg": float(energy_kwh_kg),
            "synthesis_energy_cost_usd_kg": float(synthesis_energy_cost_usd_kg),
            "is_low_energy_cold_sintering": bool(sintering_temp_c <= 250.0),
        }
