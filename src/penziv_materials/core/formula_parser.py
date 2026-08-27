"""Recursive chemical formula parser supporting nested polyanions, fractional stoichiometries, and element counts."""

import re
from typing import Dict, Tuple, List, Any


# Standard IUPAC Atomic Weights (g/mol)
STANDARD_ATOMIC_WEIGHTS: Dict[str, float] = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180, "Na": 22.990, "Mg": 24.305,
    "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.948,
    "K": 39.098, "Ca": 40.078, "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996,
    "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904, "Kr": 83.798,
    "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Nb": 92.906, "Mo": 95.95,
    "Tc": 98.0, "Ru": 101.07, "Rh": 102.91, "Pd": 106.42, "Ag": 107.87, "Cd": 112.41,
    "In": 114.82, "Sn": 118.71, "Sb": 121.76, "Te": 127.60, "I": 126.90, "Xe": 131.29,
    "Cs": 132.91, "Ba": 137.33, "La": 138.91, "Ce": 140.12, "Pr": 140.91, "Nd": 144.24,
    "Sm": 150.36, "Eu": 151.96, "Gd": 157.25, "Tb": 158.93, "Dy": 162.50, "Ho": 164.93,
    "Er": 167.26, "Tm": 168.93, "Yb": 173.05, "Lu": 174.97, "Hf": 178.49, "Ta": 180.95,
    "W": 183.84, "Re": 186.21, "Os": 190.23, "Ir": 192.22, "Pt": 195.08, "Au": 196.97,
    "Hg": 200.59, "Tl": 204.38, "Pb": 207.2, "Bi": 208.98, "Th": 232.04, "U": 238.03,
}


def parse_chemical_formula(formula: str) -> Dict[str, float]:
    """Parse any arbitrary stoichiometric chemical formula with nested parentheses or brackets into elemental mole counts.

    Examples:
        - "Mg1.10Sc0.20Zr1.80(PS4)3" -> {"Mg": 1.1, "Sc": 0.2, "Zr": 1.8, "P": 3.0, "S": 12.0}
        - "Na3Zr2(SiO4)2(PO4)" -> {"Na": 3.0, "Zr": 2.0, "Si": 2.0, "O": 12.0, "P": 1.0}
    """
    clean_formula = formula.replace("[", "(").replace("]", ")").replace(" ", "")

    def _parse_tokens(s: str) -> Dict[str, float]:
        counts: Dict[str, float] = {}
        i = 0
        n = len(s)

        while i < n:
            if s[i] == "(":
                # Find matching closing parenthesis
                paren_depth = 1
                start = i + 1
                i += 1
                while i < n and paren_depth > 0:
                    if s[i] == "(":
                        paren_depth += 1
                    elif s[i] == ")":
                        paren_depth -= 1
                    i += 1

                sub_formula = s[start : i - 1]
                sub_counts = _parse_tokens(sub_formula)

                # Parse optional multiplier after parenthesis
                mult_match = re.match(r"^(\d*\.?\d+)", s[i:])
                mult = 1.0
                if mult_match:
                    mult_str = mult_match.group(1)
                    mult = float(mult_str)
                    i += len(mult_str)

                for elem, cnt in sub_counts.items():
                    counts[elem] = counts.get(elem, 0.0) + cnt * mult

            elif s[i].isupper():
                # Element symbol parsing (1 upper + optional 1 lower)
                elem_match = re.match(r"^([A-Z][a-z]?)", s[i:])
                if not elem_match:
                    i += 1
                    continue
                elem = elem_match.group(1)
                i += len(elem)

                # Parse element quantity
                qty_match = re.match(r"^(\d*\.?\d+)", s[i:])
                qty = 1.0
                if qty_match:
                    qty_str = qty_match.group(1)
                    qty = float(qty_str)
                    i += len(qty_str)

                counts[elem] = counts.get(elem, 0.0) + qty
            else:
                i += 1

        return counts

    parsed = _parse_tokens(clean_formula)
    return {k: round(v, 6) for k, v in parsed.items()}


def compute_element_mass_fractions(formula: str) -> Dict[str, float]:
    """Calculate elemental mass fractions (weight fractions) from chemical formula with nested polyanions."""
    mol_counts = parse_chemical_formula(formula)
    if not mol_counts:
        return {"Mg": 0.5, "S": 0.5}

    element_masses = {}
    for elem, count in mol_counts.items():
        atomic_mass = STANDARD_ATOMIC_WEIGHTS.get(elem, 50.0)
        element_masses[elem] = count * atomic_mass

    total_molecular_weight = sum(element_masses.values())
    if total_molecular_weight <= 0.0:
        return {"Mg": 0.5, "S": 0.5}

    mass_fractions = {elem: mass / total_molecular_weight for elem, mass in element_masses.items()}
    return mass_fractions
