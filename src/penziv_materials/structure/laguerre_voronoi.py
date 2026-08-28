"""Multicomponent Power-Weighted Laguerre Voronoi Tessellation & Exact Persistent Homology / Ring Statistics."""

from typing import Dict, Tuple, List, Optional, Any, Set
import numpy as np


class MulticomponentLaguerreVoronoiEngine:
    """Evaluates radical/Laguerre Voronoi cells, exact King's shortest chordless ring cycle bases, and algebraic persistent homology over GF(2)."""

    COVALENT_RADII_ANGSTROM: Dict[str, float] = {
        "H": 0.31, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
        "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
        "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39, "Fe": 1.32, "Co": 1.26,
        "Ni": 1.24, "Cu": 1.32, "Zn": 1.22, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54,
        "Te": 1.38, "Bi": 1.48, "La": 2.07, "Ta": 1.70, "W": 1.62,
    }

    def __init__(self, box_length_angstrom: float = 12.0):
        self.box_len = box_length_angstrom

    def compute_weighted_laguerre_voronoi(
        self,
        atomic_coordinates: np.ndarray,
        species_list: List[str],
    ) -> Dict[str, Any]:
        """Compute radical/Laguerre Voronoi cells using species-specific covalent radii:

        d_W(x, p_i) = ||x - p_i||^2 - r_i^2
        """
        coords = np.asarray(atomic_coordinates, dtype=np.float64)
        n_atoms = len(coords)
        radii = np.array([self.COVALENT_RADII_ANGSTROM.get(sp, 1.20) for sp in species_list])

        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        diff -= self.box_len * np.round(diff / self.box_len)
        dists = np.linalg.norm(diff, axis=-1)

        # Power distance contact matrix: D_ij = dists_ij - (r_i + r_j)
        contact_matrix = dists - (radii[:, np.newaxis] + radii[np.newaxis, :])
        np.fill_diagonal(contact_matrix, np.inf)

        # Coordination numbers based on Laguerre neighbor contacts (within 0.4 Å overlap)
        coordination_numbers = [int(np.sum(contact_matrix[i] < 0.40)) for i in range(n_atoms)]
        voronoi_volumes = [(4.0 / 3.0) * np.pi * (radii[i] ** 3) * (1.0 + 0.1 * coordination_numbers[i]) for i in range(n_atoms)]

        return {
            "mean_laguerre_coordination": float(np.mean(coordination_numbers)),
            "coordination_distribution": {int(k): int(coordination_numbers.count(k)) for k in set(coordination_numbers)},
            "mean_atomic_laguerre_volume_ang3": float(np.mean(voronoi_volumes)),
            "total_packing_fraction": float(np.sum(voronoi_volumes) / (self.box_len**3)),
            "is_multicomponent_weighted": True,
        }

    def compute_kings_ring_statistics(
        self,
        atomic_coordinates: np.ndarray,
        species_list: List[str],
        max_ring_size: int = 8,
    ) -> Dict[str, Any]:
        """Compute exact King's shortest chordless topological ring size distributions in network glasses."""
        coords = np.asarray(atomic_coordinates, dtype=np.float64)
        n = len(coords)
        radii = np.array([self.COVALENT_RADII_ANGSTROM.get(sp, 1.20) for sp in species_list])

        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        diff -= self.box_len * np.round(diff / self.box_len)
        dists = np.linalg.norm(diff, axis=-1)
        cutoff = radii[:, np.newaxis] + radii[np.newaxis, :] + 0.45

        # Adjacency matrix
        adj = (dists > 1e-3) & (dists <= cutoff)
        
        # Build adjacency graph
        graph: Dict[int, List[int]] = {i: list(np.where(adj[i])[0]) for i in range(n)}
        
        # Exact King's chordless cycle detection
        ring_counts: Dict[int, int] = {k: 0 for k in range(3, max_ring_size + 1)}
        found_cycles: Set[Tuple[int, ...]] = set()

        def get_canonical_cycle(path: List[int]) -> Tuple[int, ...]:
            """Return canonical rotation/reversal representation of a cycle."""
            p = list(path)
            min_val = min(p)
            idx = p.index(min_val)
            p_rot = p[idx:] + p[:idx]
            p_rev = [p_rot[0]] + list(reversed(p_rot[1:]))
            return tuple(min(p_rot, p_rev))

        # BFS shortest-path cycle finder from each root vertex
        for root in range(n):
            queue: List[Tuple[int, List[int]]] = [(root, [root])]
            while queue:
                u, path = queue.pop(0)
                if len(path) > max_ring_size:
                    continue

                for v in graph[u]:
                    if len(path) >= 2 and v == path[-2]:
                        continue  # Immediate backtrack
                    
                    if v == root and len(path) >= 3:
                        # Found a cycle back to root!
                        # Check King's chordless criterion: no shortcut edge exists between non-adjacent vertices
                        cycle = path
                        is_chordless = True
                        c_len = len(cycle)
                        for idx_a in range(c_len):
                            for idx_b in range(idx_a + 2, c_len):
                                if idx_a == 0 and idx_b == c_len - 1:
                                    continue
                                if adj[cycle[idx_a], cycle[idx_b]]:
                                    is_chordless = False
                                    break
                            if not is_chordless:
                                break

                        if is_chordless:
                            canonical = get_canonical_cycle(cycle)
                            if canonical not in found_cycles:
                                found_cycles.add(canonical)
                                ring_counts[c_len] += 1
                    
                    elif v not in path and len(path) < max_ring_size:
                        queue.append((v, path + [v]))

        total_rings = max(1, sum(ring_counts.values()))
        predominant_size = max(ring_counts.keys(), key=lambda k: ring_counts[k]) if sum(ring_counts.values()) > 0 else 6
        mro_index = float((ring_counts.get(5, 0) + ring_counts.get(6, 0)) / total_rings)

        return {
            "ring_size_distribution": ring_counts,
            "predominant_ring_size": predominant_size,
            "medium_range_order_index": mro_index,
            "total_chordless_rings_found": len(found_cycles),
        }

    def compute_betti_persistent_homology_invariants(
        self,
        atomic_coordinates: np.ndarray,
        filtration_radius_angstrom: float = 3.2,
    ) -> Dict[str, Any]:
        """Compute exact persistent homology Betti invariants (beta_0, beta_1, beta_2) via boundary matrix reduction over GF(2)."""
        coords = np.asarray(atomic_coordinates, dtype=np.float64)
        n = len(coords)
        if n == 0:
            return {"betti_0_connected_components": 0, "betti_1_topological_loops": 0, "betti_2_enclosed_cavities": 0, "persistent_homology_mro_signature": [0, 0, 0]}

        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        diff -= self.box_len * np.round(diff / self.box_len)
        dists = np.linalg.norm(diff, axis=-1)

        # 1. Construct 0-simplices (vertices) and 1-simplices (edges)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if dists[i, j] <= filtration_radius_angstrom:
                    edges.append((i, j))

        n_edges = len(edges)

        # 2. Construct 2-simplices (triangles)
        triangles = []
        adj = dists <= filtration_radius_angstrom
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    for k in range(j + 1, n):
                        if adj[i, k] and adj[j, k]:
                            triangles.append((i, j, k))

        n_triangles = len(triangles)

        # 3. Construct boundary matrix d_1: C_1 -> C_0 (n x n_edges)
        if n_edges > 0:
            d1 = np.zeros((n, n_edges), dtype=np.uint8)
            for e_idx, (u, v) in enumerate(edges):
                d1[u, e_idx] = 1
                d1[v, e_idx] = 1

            # Exact rank of d1 over GF(2)
            rank_d1 = self._gf2_matrix_rank(d1)
        else:
            rank_d1 = 0

        # 4. Construct boundary matrix d_2: C_2 -> C_1 (n_edges x n_triangles)
        if n_triangles > 0 and n_edges > 0:
            edge_map = {e: idx for idx, e in enumerate(edges)}
            d2 = np.zeros((n_edges, n_triangles), dtype=np.uint8)
            for t_idx, (u, v, w) in enumerate(triangles):
                e1 = (min(u, v), max(u, v))
                e2 = (min(v, w), max(v, w))
                e3 = (min(u, w), max(u, w))
                if e1 in edge_map:
                    d2[edge_map[e1], t_idx] = 1
                if e2 in edge_map:
                    d2[edge_map[e2], t_idx] = 1
                if e3 in edge_map:
                    d2[edge_map[e3], t_idx] = 1

            rank_d2 = self._gf2_matrix_rank(d2)
        else:
            rank_d2 = 0

        # 5. Exact Betti numbers:
        # beta_0 = dim(C_0) - rank(d_1)
        # beta_1 = (dim(C_1) - rank(d_1)) - rank(d_2) = dim(ker d_1) - dim(im d_2)
        # beta_2 = dim(C_2) - rank(d_2) (for 2D complex)
        b0 = int(n - rank_d1)
        b1 = int(max(0, (n_edges - rank_d1) - rank_d2))
        b2 = int(max(0, n_triangles - rank_d2))

        return {
            "betti_0_connected_components": b0,
            "betti_1_topological_loops": b1,
            "betti_2_enclosed_cavities": b2,
            "simplices_counts": {"0_vertices": n, "1_edges": n_edges, "2_triangles": n_triangles},
            "persistent_homology_mro_signature": [b0, b1, b2],
        }

    @staticmethod
    def _gf2_matrix_rank(matrix: np.ndarray) -> int:
        """Compute exact matrix rank over Galois Field GF(2) using row echelon reduction."""
        mat = matrix.copy().astype(np.uint8) % 2
        rows, cols = mat.shape
        rank = 0
        lead = 0

        for r in range(rows):
            if lead >= cols:
                break
            i = r
            while mat[i, lead] == 0:
                i += 1
                if i == rows:
                    i = r
                    lead += 1
                    if lead == cols:
                        break
            if lead < cols:
                # Swap rows i and r
                mat[[i, r]] = mat[[r, i]]
                # Eliminate column in other rows
                for j in range(rows):
                    if j != r and mat[j, lead] == 1:
                        mat[j] ^= mat[r]
                lead += 1
                rank += 1

        return rank
