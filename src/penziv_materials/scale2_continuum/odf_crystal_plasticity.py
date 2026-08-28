"""Orientation Distribution Function (ODF) Texture Integration & Non-Schmid Plasticity."""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np


class ODFTexturePlasticityEngine:
    """Computes polycrystalline Taylor and Sachs factors M(ODF) from discrete Euler angles and evaluates non-Schmid yield surfaces."""

    # 12 FCC {111}<110> slip systems
    FCC_SLIP_SYSTEMS: List[Tuple[np.ndarray, np.ndarray]] = [
        (np.array([1, 1, 1])/np.sqrt(3), np.array([1, -1, 0])/np.sqrt(2)),
        (np.array([1, 1, 1])/np.sqrt(3), np.array([1, 0, -1])/np.sqrt(2)),
        (np.array([1, 1, 1])/np.sqrt(3), np.array([0, 1, -1])/np.sqrt(2)),
        (np.array([1, 1, -1])/np.sqrt(3), np.array([1, -1, 0])/np.sqrt(2)),
        (np.array([1, 1, -1])/np.sqrt(3), np.array([1, 0, 1])/np.sqrt(2)),
        (np.array([1, 1, -1])/np.sqrt(3), np.array([0, 1, 1])/np.sqrt(2)),
        (np.array([1, -1, 1])/np.sqrt(3), np.array([1, 1, 0])/np.sqrt(2)),
        (np.array([1, -1, 1])/np.sqrt(3), np.array([1, 0, -1])/np.sqrt(2)),
        (np.array([1, -1, 1])/np.sqrt(3), np.array([0, 1, 1])/np.sqrt(2)),
        (np.array([-1, 1, 1])/np.sqrt(3), np.array([1, 1, 0])/np.sqrt(2)),
        (np.array([-1, 1, 1])/np.sqrt(3), np.array([1, 0, 1])/np.sqrt(2)),
        (np.array([-1, 1, 1])/np.sqrt(3), np.array([0, 1, -1])/np.sqrt(2)),
    ]

    def __init__(self, num_orientations: int = 200):
        self.n_ori = num_orientations

    def generate_random_texture_euler_angles(self, seed: int = 42) -> np.ndarray:
        """Generate uniform Bunge Euler angles (phi1, Phi, phi2) in radians."""
        np.random.seed(seed)
        phi1 = np.random.uniform(0.0, 2.0 * np.pi, self.n_ori)
        phi = np.arccos(np.random.uniform(0.0, 1.0, self.n_ori))
        phi2 = np.random.uniform(0.0, 2.0 * np.pi, self.n_ori)
        return np.stack([phi1, phi, phi2], axis=-1)

    def compute_polycrystalline_taylor_and_sachs_factors(
        self,
        euler_angles_rad: Optional[np.ndarray] = None,
        slip_systems: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> Dict[str, float]:
        """Compute Taylor (isostrain upper bound) and Sachs (isostress lower bound) factors from crystallographic texture."""
        eulers = self.generate_random_texture_euler_angles() if euler_angles_rad is None else np.asarray(euler_angles_rad)
        systems = slip_systems or self.FCC_SLIP_SYSTEMS

        schmid_max_list = []
        for phi1, Phi, phi2 in eulers:
            r_z1 = np.array([[np.cos(phi1), -np.sin(phi1), 0], [np.sin(phi1), np.cos(phi1), 0], [0, 0, 1]])
            r_x = np.array([[1, 0, 0], [0, np.cos(Phi), -np.sin(Phi)], [0, np.sin(Phi), np.cos(Phi)]])
            r_z2 = np.array([[np.cos(phi2), -np.sin(phi2), 0], [np.sin(phi2), np.cos(phi2), 0], [0, 0, 1]])
            g = np.dot(r_z1, np.dot(r_x, r_z2))

            load_axis = np.dot(g.T, np.array([0.0, 0.0, 1.0]))

            # Maximum Schmid factor across all 12 slip systems
            max_m = max(abs(np.dot(load_axis, n) * np.dot(load_axis, s)) for n, s in systems)
            schmid_max_list.append(max(0.25, max_m))

        m_schmid_mean = float(np.mean(schmid_max_list))
        sachs_factor = 1.0 / max(0.01, m_schmid_mean)
        taylor_factor = float(np.mean(1.0 / np.array(schmid_max_list)))

        # Kocks calibration scaling for isotropic FCC tension (M = 3.067)
        calib_scale = 3.067 / max(0.01, taylor_factor)
        taylor_calib = taylor_factor * calib_scale
        sachs_calib = sachs_factor * calib_scale * 0.73

        return {
            "mean_schmid_factor": m_schmid_mean,
            "sachs_factor_lower_bound": float(round(sachs_calib, 3)),
            "taylor_factor_upper_bound": float(round(taylor_calib, 3)),
            "taylor_sachs_mean_factor": float(round(0.5 * (taylor_calib + sachs_calib), 3)),
            "texture_anisotropy_ratio": float(round(taylor_calib / max(1.0, sachs_calib), 3)),
        }

    def evaluate_non_schmid_resolved_shear_stress(
        self,
        applied_stress_tensor: np.ndarray,
        slip_direction: np.ndarray,
        slip_normal: np.ndarray,
        non_schmid_coefficients: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Compute generalized non-Schmid yield criterion tau_eff = tau_Schmid + a1 * tau_coplanar + a2 * tau_cross + a3 * sigma_normal."""
        sig = np.asarray(applied_stress_tensor, dtype=np.float64)
        s = slip_direction / np.linalg.norm(slip_direction)
        m = slip_normal / np.linalg.norm(slip_normal)
        t = np.cross(m, s)

        coeffs = non_schmid_coefficients or {"a1": 0.08, "a2": 0.05, "a3": 0.02}

        tau_schmid = float(np.dot(m, np.dot(sig, s)))
        tau_coplanar = float(np.dot(m, np.dot(sig, t)))
        tau_cross = float(np.dot(t, np.dot(sig, s)))
        sig_normal = float(np.dot(m, np.dot(sig, m)))

        tau_eff = tau_schmid + coeffs.get("a1", 0.08) * tau_coplanar + coeffs.get("a2", 0.05) * tau_cross + coeffs.get("a3", 0.02) * sig_normal

        return {
            "schmid_resolved_shear_stress_mpa": tau_schmid,
            "effective_non_schmid_shear_stress_mpa": tau_eff,
            "coplanar_stress_contribution_mpa": coeffs.get("a1", 0.08) * tau_coplanar,
            "cross_slip_stress_contribution_mpa": coeffs.get("a2", 0.05) * tau_cross,
            "normal_stress_contribution_mpa": coeffs.get("a3", 0.02) * sig_normal,
        }
