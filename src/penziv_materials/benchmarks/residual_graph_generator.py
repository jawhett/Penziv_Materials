"""Academic Publication-Quality Vector SVG Predicted vs Actual Parity Graph Generator.

Formats parity scatter plots in a rigorous scientific journal aesthetic (Physical Review / Nature standard):
- High-contrast black and white / grayscale palette.
- Academic serif typography (Times New Roman / DejaVu Serif / TeX Gyre Termes).
- Clear, large axis titles with formal bracketed SI units.
- Formal scientific statistics inset box (N, R², MAPE, RMSE).
- 1:1 ideal parity diagonal line (y = x) with shaded ±10% confidence bounds and vertical residual stems.
"""

from typing import List, Dict, Any, Optional
import math
import html


class ResidualGraphGenerator:
    """Generates publication-quality monochrome vector SVG Predicted vs Actual parity graphs."""

    # Academic Monochrome Palette (Publication Standard)
    COLOR_CANVAS = "#FFFFFF"
    COLOR_AXIS_FRAME = "#111827"
    COLOR_GRID_LINE = "#E5E7EB"
    COLOR_PARITY_LINE = "#111827"
    COLOR_CONFIDENCE_FILL = "rgba(0, 0, 0, 0.04)"
    COLOR_CONFIDENCE_BORDER = "#9CA3AF"
    COLOR_POINT_FILL = "#1F2937"
    COLOR_POINT_BORDER = "#000000"
    COLOR_POINT_STEM = "#6B7280"
    COLOR_TEXT_PRIMARY = "#111827"
    COLOR_TEXT_SECONDARY = "#4B5563"
    COLOR_TEXT_MUTED = "#6B7280"
    COLOR_INSET_BG = "#F9FAFB"
    COLOR_INSET_BORDER = "#D1D5DB"

    @classmethod
    def generate_property_parity_svg(
        cls,
        property_name: str,
        unit: str,
        material_data: List[Dict[str, Any]],
        mape: float,
        width: int = 980,
        height: int = 580,
    ) -> str:
        """Generate an academic black-and-white Predicted vs Actual Parity Scatter SVG.
        
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

        # Generous publication margins for large academic labels
        margin_top = 75
        margin_bottom = 85
        margin_left = 105
        margin_right = 55

        plot_width = width - margin_left - margin_right
        plot_height = height - margin_top - margin_bottom

        preds = [float(d["pred"]) for d in material_data]
        acts = [float(d["act"]) for d in material_data]

        # Logarithmic Scale Domain Bounds
        pos_vals = [p for p in preds if p > 0] + [a for a in acts if a > 0]
        if not pos_vals:
            pos_vals = [1.0]
        min_pos = min(pos_vals)
        max_pos = max(pos_vals)

        log_min = float(math.floor(math.log10(min_pos)))
        log_max = float(math.ceil(math.log10(max_pos)))

        if log_max <= log_min:
            log_max = log_min + 1.0

        def to_log(val: float) -> float:
            if val <= 0:
                return log_min
            return max(log_min, min(log_max, math.log10(val)))

        def scale_x(val: float) -> float:
            lv = to_log(val)
            norm = (lv - log_min) / (log_max - log_min)
            return margin_left + norm * plot_width

        def scale_y(val: float) -> float:
            lv = to_log(val)
            norm = (lv - log_min) / (log_max - log_min)
            return margin_top + (1.0 - norm) * plot_height

        def scale_y_log(log_val: float) -> float:
            norm = (log_val - log_min) / (log_max - log_min)
            return margin_top + (1.0 - norm) * plot_height

        # Calculate exact statistical metrics (R^2, RMSE)
        diffs = [p - a for p, a in zip(preds, acts)]
        rmse = math.sqrt(sum(d**2 for d in diffs) / n)
        mean_act = sum(acts) / n
        ss_tot = sum((a - mean_act)**2 for a in acts)
        ss_res = sum(d**2 for d in diffs)
        r2 = max(0.0, 1.0 - (ss_res / ss_tot)) if ss_tot > 1e-6 else 1.0

        # Compute integer decade ticks and minor sub-decade ticks
        decade_ints = list(range(int(math.floor(log_min)), int(math.ceil(log_max)) + 1))

        def format_decade(k: int) -> str:
            superscripts = {"-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"}
            exp_str = "".join(superscripts.get(c, c) for c in str(k))
            if k == 0:
                return "1"
            elif k == 1:
                return "10"
            elif k == 2:
                return "100"
            elif k == 3:
                return "10³"
            elif k == 4:
                return "10⁴"
            elif k == 5:
                return "10⁵"
            elif k == -1:
                return "0.1"
            elif k == -2:
                return "0.01"
            elif k == -3:
                return "0.001"
            else:
                return f"10{exp_str}"

        svg_parts = []
        svg_parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="100%" height="auto" style="background-color: {cls.COLOR_CANVAS}; '
            f'border: 1px solid #E5E7EB; border-radius: 4px; font-family: \'Times New Roman\', Times, \'DejaVu Serif\', \'TeX Gyre Termes\', Georgia, serif;">'
        )

        # SVG Definitions & Clip Path
        svg_parts.append(f"""
        <defs>
          <clipPath id="plotClip">
            <rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" />
          </clipPath>
        </defs>
        """)

        # Background White Canvas
        svg_parts.append(
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="{cls.COLOR_CANVAS}" />'
        )

        # Header Title (Academic Figure Heading)
        unit_str = f" [{unit}]" if unit else ""
        svg_parts.append(
            f'<text x="{margin_left}" y="34" fill="{cls.COLOR_TEXT_PRIMARY}" font-size="18" '
            f'font-weight="bold" letter-spacing="-0.01em">'
            f'Predicted vs. Actual Parity (Log-Log): {html.escape(property_name)}{html.escape(unit_str)}</text>'
        )
        svg_parts.append(
            f'<text x="{margin_left}" y="54" fill="{cls.COLOR_TEXT_SECONDARY}" font-size="12" font-style="italic">'
            f'Zero-Parameter First-Principles Multiscale Predictions vs. Experimental Benchmark (Logarithmic Axes, N = {n})</text>'
        )

        # Academic Statistics Inset Box (Top-Left or Bottom-Right depending on data density)
        inset_w = 175
        inset_h = 92
        inset_x = width - margin_right - inset_w - 15
        inset_y = margin_top + 15

        svg_parts.append(
            f'<g transform="translate({inset_x}, {inset_y})">'
            f'<rect width="{inset_w}" height="{inset_h}" rx="3" fill="{cls.COLOR_INSET_BG}" '
            f'stroke="{cls.COLOR_INSET_BORDER}" stroke-width="1.2" />'
            f'<text x="14" y="22" fill="{cls.COLOR_TEXT_PRIMARY}" font-size="12" font-weight="bold">Statistical Summary</text>'
            f'<line x1="14" y1="28" x2="{inset_w - 14}" y2="28" stroke="{cls.COLOR_INSET_BORDER}" stroke-width="0.8" />'
            f'<text x="14" y="44" fill="{cls.COLOR_TEXT_PRIMARY}" font-size="11.5">Sample Size: <tspan font-weight="bold">N = {n}</tspan></text>'
            f'<text x="14" y="60" fill="{cls.COLOR_TEXT_PRIMARY}" font-size="11.5">Determination: <tspan font-weight="bold">R² = {r2:.4f}</tspan></text>'
            f'<text x="14" y="76" fill="{cls.COLOR_TEXT_PRIMARY}" font-size="11.5">Mean Abs % Err: <tspan font-weight="bold">MAPE = {mape:.2f}%</tspan></text>'
            f'</g>'
        )

        # Minor Sub-Decade Gridlines (m * 10^k for m in 2..9)
        for k in range(int(math.floor(log_min)), int(math.ceil(log_max))):
            for mult in [2, 3, 4, 5, 6, 7, 8, 9]:
                sub_val = mult * (10.0 ** k)
                sub_log = math.log10(sub_val)
                if log_min <= sub_log <= log_max:
                    spx = scale_x(sub_val)
                    spy = scale_y(sub_val)
                    # Vertical minor grid
                    svg_parts.append(
                        f'<line x1="{spx:.1f}" y1="{margin_top}" x2="{spx:.1f}" y2="{margin_top + plot_height}" '
                        f'stroke="#F3F4F6" stroke-width="0.6" />'
                    )
                    # Horizontal minor grid
                    svg_parts.append(
                        f'<line x1="{margin_left}" y1="{spy:.1f}" x2="{margin_left + plot_width}" y2="{spy:.1f}" '
                        f'stroke="#F3F4F6" stroke-width="0.6" />'
                    )
                    # Minor axis tick marks
                    svg_parts.append(
                        f'<line x1="{margin_left - 3}" y1="{spy:.1f}" x2="{margin_left}" y2="{spy:.1f}" '
                        f'stroke="{cls.COLOR_AXIS_FRAME}" stroke-width="0.8" opacity="0.6" />'
                    )
                    svg_parts.append(
                        f'<line x1="{spx:.1f}" y1="{margin_top + plot_height}" x2="{spx:.1f}" y2="{margin_top + plot_height + 3}" '
                        f'stroke="{cls.COLOR_AXIS_FRAME}" stroke-width="0.8" opacity="0.6" />'
                    )

        # Major Decade Gridlines and Ticks
        for k in decade_ints:
            val_decade = 10.0 ** k
            px = scale_x(val_decade)
            py = scale_y(val_decade)

            # Major vertical gridline
            svg_parts.append(
                f'<line x1="{px:.1f}" y1="{margin_top}" x2="{px:.1f}" y2="{margin_top + plot_height}" '
                f'stroke="{cls.COLOR_GRID_LINE}" stroke-width="1.0" stroke-dasharray="3,3" />'
            )
            # Major horizontal gridline
            svg_parts.append(
                f'<line x1="{margin_left}" y1="{py:.1f}" x2="{margin_left + plot_width}" y2="{py:.1f}" '
                f'stroke="{cls.COLOR_GRID_LINE}" stroke-width="1.0" stroke-dasharray="3,3" />'
            )

            lbl = format_decade(k)

            # Y-Tick Label & Major Tick Mark
            svg_parts.append(
                f'<line x1="{margin_left - 6}" y1="{py:.1f}" x2="{margin_left}" y2="{py:.1f}" '
                f'stroke="{cls.COLOR_AXIS_FRAME}" stroke-width="1.3" />'
            )
            svg_parts.append(
                f'<text x="{margin_left - 10}" y="{py + 4:.1f}" fill="{cls.COLOR_TEXT_PRIMARY}" '
                f'font-size="12.5" font-weight="bold" text-anchor="end">{lbl}</text>'
            )

            # X-Tick Label & Major Tick Mark
            svg_parts.append(
                f'<line x1="{px:.1f}" y1="{margin_top + plot_height}" x2="{px:.1f}" y2="{margin_top + plot_height + 6}" '
                f'stroke="{cls.COLOR_AXIS_FRAME}" stroke-width="1.3" />'
            )
            svg_parts.append(
                f'<text x="{px:.1f}" y="{margin_top + plot_height + 23:.1f}" fill="{cls.COLOR_TEXT_PRIMARY}" '
                f'font-size="12.5" font-weight="bold" text-anchor="middle">{lbl}</text>'
            )

        # 1:1 Parity Diagonal Geometry (Log-Log is a straight 45 deg line)
        x_start, y_start = scale_x(10.0 ** log_min), scale_y(10.0 ** log_min)
        x_end, y_end = scale_x(10.0 ** log_max), scale_y(10.0 ** log_max)

        # ±10% Confidence Envelope Polygon in Log-Log (Parallel Offset Ribbon)
        log_offset_up = math.log10(1.10)
        log_offset_down = math.log10(0.90)

        upper_start_y = scale_y_log(log_min + log_offset_up)
        upper_end_y = scale_y_log(log_max + log_offset_up)
        lower_start_y = scale_y_log(log_min + log_offset_down)
        lower_end_y = scale_y_log(log_max + log_offset_down)

        poly_points = f"{x_start:.1f},{upper_start_y:.1f} {x_end:.1f},{upper_end_y:.1f} {x_end:.1f},{lower_end_y:.1f} {x_start:.1f},{lower_start_y:.1f}"
        svg_parts.append(
            f'<polygon points="{poly_points}" fill="{cls.COLOR_CONFIDENCE_FILL}" clip-path="url(#plotClip)" />'
        )
        svg_parts.append(
            f'<line x1="{x_start:.1f}" y1="{upper_start_y:.1f}" x2="{x_end:.1f}" y2="{upper_end_y:.1f}" '
            f'stroke="{cls.COLOR_CONFIDENCE_BORDER}" stroke-width="1.0" stroke-dasharray="3,3" clip-path="url(#plotClip)" />'
        )
        svg_parts.append(
            f'<line x1="{x_start:.1f}" y1="{lower_start_y:.1f}" x2="{x_end:.1f}" y2="{lower_end_y:.1f}" '
            f'stroke="{cls.COLOR_CONFIDENCE_BORDER}" stroke-width="1.0" stroke-dasharray="3,3" clip-path="url(#plotClip)" />'
        )

        # 1:1 Parity Line (Bold dashed black)
        svg_parts.append(
            f'<line x1="{x_start:.1f}" y1="{y_start:.1f}" x2="{x_end:.1f}" y2="{y_end:.1f}" '
            f'stroke="{cls.COLOR_PARITY_LINE}" stroke-width="1.6" stroke-dasharray="6,4" clip-path="url(#plotClip)" />'
        )

        # Parity Line Text Annotation
        mid_x = (x_start + x_end) / 2.0
        mid_y = (y_start + y_end) / 2.0
        svg_parts.append(
            f'<text x="{mid_x + 20:.1f}" y="{mid_y - 12:.1f}" fill="{cls.COLOR_TEXT_MUTED}" '
            f'font-size="11" font-style="italic" transform="rotate(-34 {mid_x} {mid_y})">Ideal 1:1 Parity (y = x)</text>'
        )

        # Plot Boundary Axes Box (Standard 4-Sided Academic Box Frame)
        svg_parts.append(
            f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" '
            f'fill="none" stroke="{cls.COLOR_AXIS_FRAME}" stroke-width="1.4" />'
        )

        # Major Axis Labels (Large Academic Typography)
        svg_parts.append(
            f'<text x="{margin_left + plot_width / 2.0:.1f}" y="{margin_top + plot_height + 52:.1f}" '
            f'fill="{cls.COLOR_TEXT_PRIMARY}" font-size="14.5" font-weight="bold" text-anchor="middle">'
            f'Experimental Ground Truth Value{html.escape(unit_str)} [Log₁₀ Scale]</text>'
        )
        svg_parts.append(
            f'<text x="32" y="{margin_top + plot_height / 2.0:.1f}" fill="{cls.COLOR_TEXT_PRIMARY}" '
            f'font-size="14.5" font-weight="bold" text-anchor="middle" transform="rotate(-90 32 {margin_top + plot_height / 2.0})">'
            f'Penziv Predicted Value{html.escape(unit_str)} [Log₁₀ Scale]</text>'
        )

        # Plot Data Points, Residual Drops, and High-Legibility Labels
        for i, item in enumerate(material_data):
            p_val = float(item["pred"])
            a_val = float(item["act"])
            err_pct = float(item["error_pct"])
            formula = item["formula"]

            px = scale_x(a_val)
            py = scale_y(p_val)
            diag_y = scale_y(a_val)

            # Vertical Residual Drop Stem
            svg_parts.append(
                f'<line x1="{px:.1f}" y1="{diag_y:.1f}" x2="{px:.1f}" y2="{py:.1f}" '
                f'stroke="{cls.COLOR_POINT_STEM}" stroke-width="1.2" stroke-dasharray="2,2" opacity="0.75" clip-path="url(#plotClip)" />'
            )

            # Circular Point Marker (Crisp Black Circle with White Halo)
            svg_parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.5" fill="{cls.COLOR_POINT_FILL}" '
                f'stroke="{cls.COLOR_CANVAS}" stroke-width="1.5" />'
            )
            svg_parts.append(
                f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5.5" fill="none" '
                f'stroke="{cls.COLOR_POINT_BORDER}" stroke-width="1.2" />'
            )

            # Material Label Placement
            offset_x = 8 if px < (margin_left + plot_width - 80) else -8
            anchor = "start" if offset_x > 0 else "end"
            offset_y = -7 if py <= diag_y else 13

            # Format formula string with subscript numbers if applicable
            clean_formula = html.escape(formula)
            pct_str = f"{err_pct:+.1f}%" if abs(err_pct) > 0.05 else "0.0%"
            
            svg_parts.append(
                f'<text x="{px + offset_x:.1f}" y="{py + offset_y:.1f}" fill="{cls.COLOR_TEXT_PRIMARY}" '
                f'font-size="10.5" font-weight="600" text-anchor="{anchor}">'
                f'{clean_formula} <tspan fill="{cls.COLOR_TEXT_MUTED}" font-size="9.5">({pct_str})</tspan></text>'
            )

        # Academic Footer Legend
        legend_y = height - 16
        svg_parts.append(
            f'<g transform="translate({margin_left}, {legend_y})" font-size="11" fill="{cls.COLOR_TEXT_SECONDARY}">'
            f'<circle cx="6" cy="-4" r="5" fill="{cls.COLOR_POINT_FILL}" stroke="{cls.COLOR_POINT_BORDER}" stroke-width="1.0" />'
            f'<text x="18" y="0">First-Principles Evaluation Point</text>'
            f'<line x1="210" y1="-4" x2="235" y2="-4" stroke="{cls.COLOR_PARITY_LINE}" stroke-dasharray="5,3" stroke-width="1.5" />'
            f'<text x="242" y="0">Ideal 1:1 Parity Line</text>'
            f'<rect x="380" y="-8" width="18" height="10" fill="{cls.COLOR_CONFIDENCE_FILL}" stroke="{cls.COLOR_CONFIDENCE_BORDER}" stroke-dasharray="2,2" stroke-width="0.8" />'
            f'<text x="405" y="0">±10% Confidence Interval</text>'
            f'<line x1="560" y1="-4" x2="585" y2="-4" stroke="{cls.COLOR_POINT_STEM}" stroke-dasharray="2,2" stroke-width="1.2" />'
            f'<text x="592" y="0">Residual Drop Stem</text>'
            f'</g>'
        )

        svg_parts.append("</svg>")
        return "\n".join(svg_parts)
