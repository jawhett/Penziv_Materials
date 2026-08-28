"""High-Resolution Vector SVG Predicted vs Actual Parity Graph Generator for Penziv Materials Benchmarks.

Strictly adheres to the Serene Zenith design system (BRAND.md) and Tufte data visualization standards.
Generates 1 Predicted vs Actual Parity Graph per property with all benchmarked materials plotted on each single graph.
"""

from typing import List, Dict, Any, Optional
import math
import html


class ResidualGraphGenerator:
    """Generates standalone, responsive vector SVG Predicted vs Actual parity graphs."""

    # Brand Colors from BRAND.md (Serene Zenith Dark Theme)
    COLOR_BG_CANVAS = "#040D18"
    COLOR_BG_SURFACE = "#0F172A"
    COLOR_BG_SURFACE_ELEVATED = "#1E293B"
    COLOR_BORDER_SUBTLE = "#1E293B"
    COLOR_BORDER_STRONG = "#334155"
    COLOR_TEXT_PRIMARY = "#F8FAFC"
    COLOR_TEXT_SECONDARY = "#94A3B8"
    COLOR_TEXT_MUTED = "#64748B"
    COLOR_CYAN_PRIMARY = "#22D3EE"
    COLOR_CYAN_DARK = "#0891B2"
    COLOR_TEAL_SUCCESS = "#2DD4BF"
    COLOR_AMBER_ATTENTION = "#FBBF24"
    COLOR_ROSE_CRITICAL = "#FB7185"

    @classmethod
    def get_color_for_error(cls, error_pct: float) -> str:
        """Return brand color based on absolute percentage error magnitude."""
        abs_err = abs(error_pct)
        if abs_err <= 5.0:
            return cls.COLOR_TEAL_SUCCESS
        elif abs_err <= 15.0:
            return cls.COLOR_CYAN_PRIMARY
        elif abs_err <= 30.0:
            return cls.COLOR_AMBER_ATTENTION
        else:
            return cls.COLOR_ROSE_CRITICAL

    @classmethod
    def generate_property_parity_svg(
        cls,
        property_name: str,
        unit: str,
        material_data: List[Dict[str, Any]],
        mape: float,
        width: int = 960,
        height: int = 540,
    ) -> str:
        """Generate a Predicted vs Actual Parity Scatter SVG for a single property with all materials.
        
        material_data items contain:
          - formula: str (e.g. 'Cu')
          - label: str (short display label)
          - pred: float (predicted value)
          - act: float (ground truth value)
          - residual: float (pred - act)
          - error_pct: float (relative percentage error)
        """
        n = len(material_data)
        if n == 0:
            return "<svg></svg>"

        # Margins
        margin_top = 80
        margin_bottom = 75
        margin_left = 95
        margin_right = 50

        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom

        preds = [d["pred"] for d in material_data]
        acts = [d["act"] for d in material_data]

        # Determine axis range encompassing both predictions and ground truth
        all_vals = preds + acts
        min_v = min(all_vals)
        max_v = max(all_vals)

        # Handle zero or negative bounds gracefully
        if min_v >= 0:
            axis_min = 0.0
            axis_max = max_v * 1.18 if max_v > 0 else 1.0
        else:
            axis_min = min_v * 1.15
            axis_max = max_v * 1.15

        # Avoid zero range
        if axis_max <= axis_min:
            axis_max = axis_min + 1.0

        def scale_x(val: float) -> float:
            norm = (val - axis_min) / (axis_max - axis_min)
            return margin_left + norm * plot_width

        def scale_y(val: float) -> float:
            norm = (val - axis_min) / (axis_max - axis_min)
            return margin_top + (1.0 - norm) * plot_height

        # Calculate R^2 and RMSE
        diffs = [p - a for p, a in zip(preds, acts)]
        rmse = math.sqrt(sum(d**2 for d in diffs) / n)
        mean_act = sum(acts) / n
        ss_tot = sum((a - mean_act)**2 for a in acts)
        ss_res = sum(d**2 for d in diffs)
        r2 = max(0.0, 1.0 - (ss_res / ss_tot)) if ss_tot > 1e-6 else 1.0

        # Build Ticks (5 ticks across axis)
        tick_vals = [
            axis_min,
            axis_min + (axis_max - axis_min) * 0.25,
            axis_min + (axis_max - axis_min) * 0.50,
            axis_min + (axis_max - axis_min) * 0.75,
            axis_max,
        ]

        svg_parts = []
        svg_parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="100%" height="auto" style="background-color: {cls.COLOR_BG_CANVAS}; '
            f'border-radius: 12px; font-family: Inter, -apple-system, sans-serif;">'
        )

        # Embedded Styles & Gradients
        svg_parts.append("""
        <defs>
          <linearGradient id="gradParityCard" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#0F172A" />
            <stop offset="100%" stop-color="#0B1324" />
          </linearGradient>
          <filter id="glowEffect" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>
        """)

        # Background Card
        svg_parts.append(
            f'<rect x="2" y="2" width="{width - 4}" height="{height - 4}" rx="12" '
            f'fill="url(#gradParityCard)" stroke="{cls.COLOR_BORDER_SUBTLE}" stroke-width="1.5" />'
        )

        # Header Title & Property Units
        unit_str = f" [{unit}]" if unit else ""
        svg_parts.append(
            f'<text x="{margin_left}" y="36" fill="{cls.COLOR_TEXT_PRIMARY}" font-size="16" '
            f'font-weight="700" letter-spacing="-0.01em">{html.escape(property_name)} — Predicted vs Actual Parity{html.escape(unit_str)}</text>'
        )
        svg_parts.append(
            f'<text x="{margin_left}" y="56" fill="{cls.COLOR_TEXT_SECONDARY}" font-size="11">'
            f'First-Principles Simulation Parity • 10 Benchmark Materials • Ideal 1:1 Parity Line</text>'
        )

        # Metrics KPI Badge (Top-Right: R^2, MAPE, RMSE)
        badge_w = 260
        badge_x = width - margin_right - badge_w
        svg_parts.append(
            f'<g transform="translate({badge_x}, 20)">'
            f'<rect width="{badge_w}" height="32" rx="6" fill="{cls.COLOR_BG_SURFACE_ELEVATED}" '
            f'stroke="{cls.COLOR_BORDER_STRONG}" stroke-width="1" />'
            f'<text x="12" y="20" fill="{cls.COLOR_CYAN_PRIMARY}" font-family="JetBrains Mono, monospace" font-size="11" font-weight="700">R² = {r2:.3f}</text>'
            f'<text x="95" y="20" fill="{cls.COLOR_TEAL_SUCCESS}" font-family="JetBrains Mono, monospace" font-size="11" font-weight="700">MAPE: {mape:.1f}%</text>'
            f'<text x="185" y="20" fill="{cls.COLOR_TEXT_SECONDARY}" font-family="JetBrains Mono, monospace" font-size="10.5">RMSE: {rmse:.2f}</text>'
            f'</g>'
        )

        # Gridlines (X and Y)
        for t_val in tick_vals:
            px = scale_x(t_val)
            py = scale_y(t_val)

            # Vertical gridline
            svg_parts.append(
                f'<line x1="{px:.1f}" y1="{margin_top}" x2="{px:.1f}" y2="{margin_top + plot_height}" '
                f'stroke="{cls.COLOR_BORDER_SUBTLE}" stroke-width="1" stroke-dasharray="3,3" opacity="0.6" />'
            )
            # Horizontal gridline
            svg_parts.append(
                f'<line x1="{margin_left}" y1="{py:.1f}" x2="{margin_left + plot_width}" y2="{py:.1f}" '
                f'stroke="{cls.COLOR_BORDER_SUBTLE}" stroke-width="1" stroke-dasharray="3,3" opacity="0.6" />'
            )

            # Tick Labels
            val_fmt = f"{t_val:.1f}" if abs(t_val) < 100 else f"{t_val:.0f}"
            if abs(t_val) < 0.01:
                val_fmt = "0.0"

            # Y-Tick Label
            svg_parts.append(
                f'<text x="{margin_left - 10}" y="{py + 4:.1f}" fill="{cls.COLOR_TEXT_MUTED}" '
                f'font-family="JetBrains Mono, monospace" font-size="10" text-anchor="end">{val_fmt}</text>'
            )
            # X-Tick Label
            svg_parts.append(
                f'<text x="{px:.1f}" y="{margin_top + plot_height + 18:.1f}" fill="{cls.COLOR_TEXT_MUTED}" '
                f'font-family="JetBrains Mono, monospace" font-size="10" text-anchor="middle">{val_fmt}</text>'
            )

        # Plot Boundary Axes Box
        svg_parts.append(
            f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" '
            f'fill="none" stroke="{cls.COLOR_BORDER_STRONG}" stroke-width="1.2" />'
        )

        # 1:1 Parity Diagonal Line (y = x)
        x_start, y_start = scale_x(axis_min), scale_y(axis_min)
        x_end, y_end = scale_x(axis_max), scale_y(axis_max)

        # +/- 10% Confidence Envelope Polygon
        upper_start_y = scale_y(axis_min * 1.10)
        upper_end_y = scale_y(axis_max * 1.10)
        lower_start_y = scale_y(axis_min * 0.90)
        lower_end_y = scale_y(axis_max * 0.90)

        poly_points = f"{x_start:.1f},{upper_start_y:.1f} {x_end:.1f},{upper_end_y:.1f} {x_end:.1f},{lower_end_y:.1f} {x_start:.1f},{lower_start_y:.1f}"
        svg_parts.append(
            f'<polygon points="{poly_points}" fill="{cls.COLOR_CYAN_PRIMARY}" opacity="0.06" />'
        )

        # 1:1 Parity Line
        svg_parts.append(
            f'<line x1="{x_start:.1f}" y1="{y_start:.1f}" x2="{x_end:.1f}" y2="{y_end:.1f}" '
            f'stroke="{cls.COLOR_TEXT_MUTED}" stroke-width="1.8" stroke-dasharray="6,4" />'
        )

        # Parity Line Text Annotation
        mid_x = (x_start + x_end) / 2.0
        mid_y = (y_start + y_end) / 2.0
        svg_parts.append(
            f'<text x="{mid_x + 15:.1f}" y="{mid_y - 12:.1f}" fill="{cls.COLOR_TEXT_MUTED}" '
            f'font-size="9.5" font-weight="600" transform="rotate(-32 {mid_x} {mid_y})">Ideal 1:1 Parity (y = x)</text>'
        )

        # Axis Titles
        svg_parts.append(
            f'<text x="{margin_left + plot_width / 2.0:.1f}" y="{margin_top + plot_height + 40:.1f}" '
            f'fill="{cls.COLOR_TEXT_PRIMARY}" font-size="11.5" font-weight="600" text-anchor="middle">'
            f'Ground Truth / Literature Actual Value {html.escape(unit_str)}</text>'
        )
        svg_parts.append(
            f'<text x="24" y="{margin_top + plot_height / 2.0:.1f}" fill="{cls.COLOR_TEXT_PRIMARY}" '
            f'font-size="11.5" font-weight="600" text-anchor="middle" transform="rotate(-90 24 {margin_top + plot_height / 2.0})">'
            f'Penziv Multiscale Predicted Value {html.escape(unit_str)}</text>'
        )

        # Plot Data Points, Residual Drops, and Labels for all Materials
        for i, item in enumerate(material_data):
            p_val = item["pred"]
            a_val = item["act"]
            err_pct = item["error_pct"]
            formula = item["formula"]
            color = cls.get_color_for_error(err_pct)

            px = scale_x(a_val)
            py = scale_y(p_val)
            diag_y = scale_y(a_val)

            # Vertical Residual Drop Stem from Parity Line to Point
            svg_parts.append(
                f'<line x1="{px:.1f}" y1="{diag_y:.1f}" x2="{px:.1f}" y2="{py:.1f}" '
                f'stroke="{color}" stroke-width="1.4" stroke-dasharray="2,2" opacity="0.85" />'
            )

            # Point Circle with Outer Ring
            svg_parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6.5" fill="{color}" '
                f'stroke="{cls.COLOR_BG_CANVAS}" stroke-width="2" filter="url(#glowEffect)" />'
            )

            # Material Label & Metric Tag (smartly offset)
            offset_x = 10 if px < (margin_left + plot_width - 80) else -10
            anchor = "start" if offset_x > 0 else "end"
            offset_y = -8 if py < diag_y else 14

            pct_text = f"{err_pct:+.1f}%" if abs(err_pct) > 0.05 else "0.0%"
            svg_parts.append(
                f'<text x="{px + offset_x:.1f}" y="{py + offset_y:.1f}" fill="{cls.COLOR_TEXT_PRIMARY}" '
                f'font-family="JetBrains Mono, monospace" font-size="9.5" font-weight="700" text-anchor="{anchor}">'
                f'{html.escape(formula)} <tspan fill="{color}" font-size="8.5">({pct_text})</tspan></text>'
            )

        # Footer Legend
        legend_y = height - 12
        svg_parts.append(
            f'<g transform="translate({margin_left}, {legend_y})" font-size="9.5">'
            f'<circle cx="5" cy="-3" r="4.5" fill="{cls.COLOR_TEAL_SUCCESS}" />'
            f'<text x="15" y="0" fill="{cls.COLOR_TEXT_SECONDARY}">|Δ| ≤ 5% (Exact)</text>'
            f'<circle cx="125" cy="-3" r="4.5" fill="{cls.COLOR_CYAN_PRIMARY}" />'
            f'<text x="135" y="0" fill="{cls.COLOR_TEXT_SECONDARY}">|Δ| ≤ 15% (High Precision)</text>'
            f'<circle cx="275" cy="-3" r="4.5" fill="{cls.COLOR_AMBER_ATTENTION}" />'
            f'<text x="285" y="0" fill="{cls.COLOR_TEXT_SECONDARY}">|Δ| ≤ 30% (VRH Bounds)</text>'
            f'<circle cx="425" cy="-3" r="4.5" fill="{cls.COLOR_ROSE_CRITICAL}" />'
            f'<text x="435" y="0" fill="{cls.COLOR_TEXT_SECONDARY}">|Δ| &gt; 30%</text>'
            f'<line x1="535" y1="-3" x2="555" y2="-3" stroke="{cls.COLOR_TEXT_MUTED}" stroke-dasharray="4,3" stroke-width="1.5" />'
            f'<text x="562" y="0" fill="{cls.COLOR_TEXT_MUTED}">1:1 Parity Line</text>'
            f'</g>'
        )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)
