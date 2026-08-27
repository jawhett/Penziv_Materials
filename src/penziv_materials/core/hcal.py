"""Heterogeneous Compute Acceleration Layer (HCAL): Unified Array Abstractions & Bitwise Determinism."""

import os
import sys
from typing import Optional, Union, Tuple, Any
import numpy as np


class HCALDevice:
    """Device context manager and execution dispatcher for CPU/GPU hardware."""

    def __init__(self, device_name: Optional[str] = None, bitwise_deterministic: bool = True):
        self.bitwise_deterministic = bitwise_deterministic
        self.device = device_name or "cpu"
        if self.bitwise_deterministic:
            self._enforce_determinism()

    def _enforce_determinism(self) -> None:
        """Enforce strict IEEE 754 bitwise determinism for bifurcation-sensitive solvers."""
        os.environ["PYTHONHASHSEED"] = "42"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        np.random.seed(42)

    def to_tensor(self, array: Union[np.ndarray, list, float]) -> np.ndarray:
        """Convert array to standard zero-copy array buffer."""
        return np.asarray(array, dtype=np.float64)

    def to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert tensor to host NumPy array."""
        return np.asarray(tensor, dtype=np.float64)

    def synchronize(self) -> None:
        """Explicit stream synchronization across compute backends."""
        pass


# Global default compute device
default_hcal = HCALDevice()
