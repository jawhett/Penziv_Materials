"""Unit tests for Dynamic README generation and Predicted vs Actual Parity Scatter Plot rendering."""

import unittest
from pathlib import Path
import tempfile
import os

from penziv_materials.benchmarks.residual_graph_generator import ResidualGraphGenerator
from penziv_materials.benchmarks.dynamic_readme_generator import DynamicReadmeGenerator, PROPERTIES_META, BENCHMARK_GROUND_TRUTH


class TestDynamicReadmeAndParityGraphs(unittest.TestCase):
    def setUp(self):
        self.generator = DynamicReadmeGenerator()

    def test_residual_graph_generator_parity_svg(self):
        sample_data = [
            {"formula": "Cu", "label": "Pure Metal", "pred": 8.97, "act": 8.96, "residual": 0.01, "error_pct": 0.11},
            {"formula": "Al", "label": "Light Metal", "pred": 2.70, "act": 2.70, "residual": 0.0, "error_pct": 0.0},
            {"formula": "CaO", "label": "Ceramic Oxide", "pred": 3.35, "act": 3.34, "residual": 0.01, "error_pct": 0.30},
        ]
        svg = ResidualGraphGenerator.generate_property_parity_svg(
            property_name="Density",
            unit="g/cm³",
            material_data=sample_data,
            mape=0.14,
        )
        self.assertIn("<svg", svg)
        self.assertIn("</svg>", svg)
        self.assertIn("Ideal 1:1 Parity", svg)
        self.assertIn("Cu", svg)
        self.assertIn("Al", svg)
        self.assertIn("CaO", svg)

    def test_benchmark_ground_truth_integrity(self):
        self.assertGreaterEqual(len(BENCHMARK_GROUND_TRUTH), 24)
        self.assertEqual(len(PROPERTIES_META), 12)
        for f, gt in BENCHMARK_GROUND_TRUTH.items():
            self.assertIn("density_g_cm3", gt)
            self.assertIn("youngs_modulus_gpa", gt)
            self.assertIn("thermal_conductivity_w_m_k", gt)
            self.assertIn("band_gap_ev", gt)
            self.assertIn("citation", gt)

    def test_run_benchmark_and_compute_residuals(self):
        res = self.generator.run_benchmark_and_compute_residuals()
        self.assertGreaterEqual(len(res["raw_reports"]), 24)
        self.assertEqual(len(res["property_datasets"]), 12)
        self.assertIn("density", res["mapes"])
        self.assertIn("youngs_modulus", res["mapes"])
        self.assertIn("thermal_conductivity", res["mapes"])
        self.assertIn("bandgap", res["mapes"])

    def test_execute_and_update_readme(self):
        res = self.generator.execute_and_update()
        self.assertTrue(Path(res["readme_path"]).exists())
        self.assertEqual(len(res["graphs_generated"]), 12)
        for pkey, gpath in res["graphs_generated"].items():
            self.assertTrue(Path(gpath).exists(), f"Graph {pkey} at {gpath} does not exist")


if __name__ == "__main__":
    unittest.main()
