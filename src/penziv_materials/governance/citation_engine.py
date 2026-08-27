"""Automated Provenance Citation Engine & Academic Governance Module."""

from typing import Dict, List, Optional
import datetime


class CitationEngine:
    """Dynamically mints provenance metadata, BibTeX dependency trees, and CITATION.cff records."""

    @staticmethod
    def generate_bibtex(
        title: str = "Penziv Materials Discovery",
        author: str = "Jawhett et al.",
        year: int = 2026,
        doi: str = "10.5281/zenodo.13488178",
    ) -> str:
        """Generate BibTeX citation entry."""
        bibtex = f"""@software{{penziv_materials_{year},
  author       = {{{author}}},
  title        = {{{{{title}: Autonomous Multiscale First-Principles Materials Property Prediction}}}},
  month        = aug,
  year         = {year},
  publisher    = {{Zenodo}},
  version      = {{3.2.0}},
  doi          = {{{doi}}},
  url          = {{https://github.com/jawhett/Penziv_Materials}}
}}"""
        return bibtex

    @staticmethod
    def assemble_execution_dependency_tree(invoked_solvers: List[str]) -> str:
        """Assemble full academic dependency citation tree for all solvers and theories invoked in an execution graph."""
        citations = {
            "SCAN_metaGGA": "Sun, J., Ruzsinszky, A., & Perdew, J. P. (2015). Strongly constrained and appropriately normed semilocal density functional. Phys. Rev. Lett., 115(3), 036402.",
            "TDEP_phonons": "Hellman, O., Abrikosov, I. A., & Simak, S. I. (2011). Lattice dynamics of anharmonic solids from first principles. Phys. Rev. B, 84(18), 180301.",
            "MACE_MLIP": "Batatia, I., Kovacs, D. P., Simm, G. N., Ortner, C., & Csanyi, G. (2022). MACE: Higher order equivariant message passing neural networks. NeurIPS 2022.",
            "DAMASK_CPFFT": "Roters, F., Diehl, M., Shanthraj, P., et al. (2019). DAMASK: The Dusseldorf Advanced Material Simulation Kit. Comput. Mater. Sci., 158, 420-478.",
            "Nix_Gao_Indentation": "Nix, W. D., & Gao, H. (1998). Indentation size effects in crystalline materials. J. Mech. Phys. Solids, 46(3), 411-425.",
            "CGM_Solute_Trapping": "Aziz, M. J. (1982). Model for solute trapping during rapid solidification. J. Appl. Phys., 53(2), 1158-1168.",
        }

        lines = ["# Multiscale Physics Provenance & Citation Dependency Tree\n"]
        for solver in invoked_solvers:
            if solver in citations:
                lines.append(f"- **{solver}**: {citations[solver]}")
            else:
                lines.append(f"- **{solver}**: Standard first-principles multiscale formulation.")

        return "\n".join(lines)
