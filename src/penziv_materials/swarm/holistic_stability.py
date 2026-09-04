"""Variational Continuum Composite Hamiltonian & Multi-Phase Constraint Relaxation."""

from typing import Dict, Tuple, List, Optional, Any, Sequence
import numpy as np


class HolisticStabilityRelaxationEngine:
    """Evaluates variational composite equilibrium across arbitrary N-phase microstructures where local phase fragility is stabilized by multi-phase boundary constraints."""

    def __init__(self, biot_coefficient_alpha: float = 0.75):
        self.alpha_biot = biot_coefficient_alpha

    def evaluate_multiphase_hamiltonian(
        self,
        phase_volume_fractions: Dict[str, float],
        phase_strain_energy_densities_mj_m3: Dict[str, float],
        phase_critical_strain_energies_mj_m3: Optional[Dict[str, float]] = None,
        interfacial_energies_mj_m3: Optional[Dict[Tuple[str, str], float]] = None,
        fluid_pressure_work_mj_m3: float = 0.0,
        fluid_volume_fraction: float = 0.0,
        eigenstrain_energy_mj_m3: float = 0.0,
        phase_gradient_energy_mj_m3: float = 0.0,
        stress_concentration_tensor_gpa: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute generalized N-phase variational composite potential energy:

        Pi = sum_k phi_k * U_k + 0.5 * sum_k kappa_k |grad phi_k|^2 - phi_fluid * (alpha_Biot * W_fluid) + sum_{k < l} Gamma_{kl} + U_eigen
        """
        # Sum bulk elastic energies
        u_bulk_total = sum(
            phase_volume_fractions.get(p, 0.0) * phase_strain_energy_densities_mj_m3.get(p, 0.0)
            for p in phase_volume_fractions
        )

        # Microstructural stress concentration adjustment via linear elastic shear compliance:
        # U_dev = s_ij * s_ij / (4 * G_0) [MJ/m^3] with reference G_0 = 50.0 GPa
        stress_conc_energy = 0.0
        if stress_concentration_tensor_gpa is not None:
            sig = np.asarray(stress_concentration_tensor_gpa, dtype=np.float64)
            dev_stress = sig - (np.trace(sig) / 3.0) * np.eye(3)
            g0_gpa = 50.0
            stress_conc_energy = float((np.sum(dev_stress**2) * 1000.0) / (4.0 * g0_gpa))

        # Fluid/Pore pressure relief
        w_fluid_support = fluid_volume_fraction * (self.alpha_biot * fluid_pressure_work_mj_m3)

        # Interfacial energy density
        gamma_total = sum(interfacial_energies_mj_m3.values()) if interfacial_energies_mj_m3 else 0.0

        # Variational total energy density including non-local phase gradients
        pi_total_system = (u_bulk_total + phase_gradient_energy_mj_m3 + stress_conc_energy + gamma_total + eigenstrain_energy_mj_m3) - w_fluid_support

        # Check local phase overstress vs overall composite stabilization
        crit_energies = phase_critical_strain_energies_mj_m3 or {"matrix": 110.0}
        any_phase_overstressed = any(
            phase_strain_energy_densities_mj_m3.get(p, 0.0) > crit_energies.get(p, 110.0)
            for p in phase_volume_fractions
        )

        # System effective critical energy
        crit_effective = sum(
            phase_volume_fractions.get(p, 0.0) * crit_energies.get(p, 110.0)
            for p in phase_volume_fractions
        ) if phase_volume_fractions else 110.0

        # Local Griffith / LEFM crack initiation evaluation:
        # Extrinsic fracture resistance R_eff is governed strictly by solid-phase bridging:
        # R_eff = U_crit + Delta U_solid_bridging (polymer ligament bridging and interface debonding).
        # External hydrostatic fluid support acts via Terzaghi effective stress (sigma' = sigma - alpha * p * I)
        # as compressive channel-wall confinement reducing net tensile burst driving energy.
        delta_u_solid_bridging = float(gamma_total * 8.0 + eigenstrain_energy_mj_m3 * 0.5)
        local_griffith_fracture_initiated = False
        for p in phase_volume_fractions:
            u_p = phase_strain_energy_densities_mj_m3.get(p, 0.0)
            u_crit_p = crit_energies.get(p, 110.0)
            effective_resistance_p = u_crit_p + delta_u_solid_bridging
            # Net tensile driving energy under external compressive fluid confinement
            u_driving_p = max(0.0, u_p - w_fluid_support)
            if u_driving_p > effective_resistance_p:
                local_griffith_fracture_initiated = True
                break

        is_composite_stabilized = bool(pi_total_system < crit_effective and not local_griffith_fracture_initiated)

        gate_decision = (
            "ACCEPTED_VIA_HOLISTIC_RELAXATION"
            if (any_phase_overstressed and is_composite_stabilized)
            else (
                "ACCEPTED_STANDARD"
                if not any_phase_overstressed and not local_griffith_fracture_initiated
                else ("REJECTED_LOCAL_GRIFFITH_FRACTURE" if local_griffith_fracture_initiated else "REJECTED_GLOBAL_INSTABILITY")
            )
        )

        return {
            "total_system_free_energy_mj_m3": float(pi_total_system),
            "bulk_elastic_energy_density_mj_m3": float(u_bulk_total),
            "fluid_support_energy_mj_m3": float(w_fluid_support),
            "interface_energy_density_mj_m3": float(gamma_total),
            "eigenstrain_energy_mj_m3": float(eigenstrain_energy_mj_m3),
            "isolated_phase_fragility_detected": any_phase_overstressed,
            "composite_co_design_stabilized": is_composite_stabilized,
            "handshake_gate_decision": gate_decision,
        }

    def evaluate_composite_system_hamiltonian(
        self,
        ceramic_elastic_energy_density_mj_m3: float,
        fluid_pressure_work_mj_m3: float,
        polymer_interfacial_traction_energy_mj_m3: float,
        vol_fraction_ceramic: float,
        vol_fraction_fluid: float,
        vol_fraction_polymer: float,
        interface_area_density_m2_m3: float = 1.2e7,
        specific_interface_energy_j_m2: float = 0.045,
    ) -> Dict[str, Any]:
        """Backward-compatible 3-phase composite Hamiltonian evaluation."""
        u_ceramic = vol_fraction_ceramic * ceramic_elastic_energy_density_mj_m3
        w_fluid_support = vol_fraction_fluid * (self.alpha_biot * fluid_pressure_work_mj_m3)
        u_polymer = vol_fraction_polymer * polymer_interfacial_traction_energy_mj_m3

        gamma_interface_mj_m3 = (interface_area_density_m2_m3 * specific_interface_energy_j_m2) * 1.0e-6
        pi_total_system = (u_ceramic + u_polymer + gamma_interface_mj_m3) - w_fluid_support

        critical_strain_energy_mj_m3 = 110.0
        # Extrinsic toughening / fracture resistance enhancement (LEFM R-curve):
        # Solid-phase polymer ligament bridging and interfacial debonding increase effective crack resistance
        delta_u_bridging = float(u_polymer * 2.0 + gamma_interface_mj_m3 * 15.0)
        effective_fracture_resistance = critical_strain_energy_mj_m3 + delta_u_bridging
        # External hydrostatic fluid support acts via Terzaghi effective stress as compressive channel confinement
        u_driving_ceramic = max(0.0, ceramic_elastic_energy_density_mj_m3 - w_fluid_support)
        local_griffith_fracture = u_driving_ceramic > effective_fracture_resistance

        is_isolated_ceramic_overstressed = ceramic_elastic_energy_density_mj_m3 > critical_strain_energy_mj_m3
        is_composite_stabilized = bool(pi_total_system < critical_strain_energy_mj_m3 and not local_griffith_fracture)

        gate_decision = (
            "ACCEPTED_VIA_HOLISTIC_RELAXATION"
            if (is_isolated_ceramic_overstressed and is_composite_stabilized)
            else (
                "ACCEPTED_STANDARD"
                if not is_isolated_ceramic_overstressed and not local_griffith_fracture
                else ("REJECTED_LOCAL_GRIFFITH_FRACTURE" if local_griffith_fracture else "REJECTED_GLOBAL_INSTABILITY")
            )
        )

        return {
            "total_system_free_energy_mj_m3": float(pi_total_system),
            "isolated_ceramic_fragility_detected": is_isolated_ceramic_overstressed,
            "composite_co_design_stabilized": is_composite_stabilized,
            "handshake_gate_decision": gate_decision,
            "biot_fluid_support_energy_mj_m3": float(w_fluid_support),
            "interface_energy_density_mj_m3": float(gamma_interface_mj_m3),
        }
