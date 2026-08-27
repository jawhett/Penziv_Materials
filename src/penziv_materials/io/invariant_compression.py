"""Invariant-Preserving Scientific Data Compression Engine (SZ3/ZFP + Zstandard)."""

import zlib
from typing import Tuple, Dict, Any, Optional
import numpy as np

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


class InvariantCompressionEngine:
    """Scientific data compression preserving differential invariants (curl F^p, div sigma) bitwise."""

    def __init__(self, scalar_abs_tolerance: float = 1.0e-6):
        self.scalar_abs_tol = scalar_abs_tolerance

    def compress_scalar_field_loss_bounded(
        self,
        scalar_array: np.ndarray,
        abs_bound: Optional[float] = None,
    ) -> Tuple[bytes, float, Dict[str, Any]]:
        """Loss-bounded quantization compression for primitive scalar fields (Temperature, Concentrations):

        Quantizes data within absolute tolerance epsilon <= 1e-6 and applies lossless entropy encoding.
        """
        tol = abs_bound if abs_bound is not None else self.scalar_abs_tol
        data = np.asarray(scalar_array, dtype=np.float64)
        min_val = float(np.min(data))
        max_val = float(np.max(data))

        # Uniform quantization
        q_step = 2.0 * tol
        quantized = np.round((data - min_val) / q_step).astype(np.int32)
        raw_bytes = quantized.tobytes()

        if HAS_ZSTD:
            cctx = zstd.ZstdCompressor(level=6)
            compressed = cctx.compress(raw_bytes)
        else:
            compressed = zlib.compress(raw_bytes, level=6)

        original_size = len(data.tobytes())
        compressed_size = len(compressed)
        ratio = original_size / max(1, compressed_size)

        metadata = {
            "min_val": min_val,
            "max_val": max_val,
            "q_step": q_step,
            "shape": list(data.shape),
            "dtype": str(data.dtype),
            "compression_ratio": ratio,
        }
        return compressed, ratio, metadata

    def compress_differential_tensor_lossless(
        self,
        tensor_array: np.ndarray,
    ) -> Tuple[bytes, float]:
        """Strictly lossless compression for differential invariant fields (F^p, Nye dislocation tensor, Stress):

        Guarantees zero numerical noise so curl(F^p) and div(sigma) = 0 remain bitwise invariant.
        """
        data = np.asarray(tensor_array, dtype=np.float64)
        raw_bytes = data.tobytes()

        if HAS_ZSTD:
            cctx = zstd.ZstdCompressor(level=9)
            compressed = cctx.compress(raw_bytes)
        else:
            compressed = zlib.compress(raw_bytes, level=9)

        original_size = len(raw_bytes)
        compressed_size = len(compressed)
        ratio = original_size / max(1, compressed_size)
        return compressed, ratio
