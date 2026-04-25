#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ricci Flow on Trust Networks
Ollivier-Ricci curvature for economic network analysis

Author: Marc Daghar
Licence: CC BY-SA 4.0
Mention: Free Dr Aafia Siddiqui !
"""

import numpy as np
import networkx as nx
from scipy.optimize import linear_sum_assignment
from typing import Tuple, Dict


def ollivier_ricci_curvature(G: nx.Graph, edge: Tuple, alpha: float = 0.5) -> float:
    """
    Compute Ollivier-Ricci curvature for an edge.
    
    Lower curvature = more hyperbolic/tense
    Higher curvature = more flat/abundant
    
    Based on Wasserstein-1 distance between neighborhood distributions.
    """
    u, v = edge
    
    # Closed neighborhoods
    N_u = set(G.neighbors(u)) | {u}
    N_v = set(G.neighbors(v)) | {v}
    
    def dist(a, b):
        """Shortest path distance (capped)"""
        try:
            return min(1.0, nx.shortest_path_length(G, a, b) / 2.0)
        except:
            return 1.0
    
    nodes_union = list(N_u.union(N_v))
    n = len(nodes_union)
    
    # Build cost matrix for optimal transport
    cost_matrix = np.zeros((n, n))
    for i, a in enumerate(nodes_union):
        for j, b in enumerate(nodes_union):
            if a in N_u and b in N_v:
                cost_matrix[i, j] = dist(a, b)
            else:
                cost_matrix[i, j] = 1e6
    
    # Solve optimal transport (Earth Mover's Distance)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    wasserstein = cost_matrix[row_ind, col_ind].sum() / n
    
    curvature = 1.0 - 2.0 * wasserstein
    return np.clip(curvature, -1.0, 1.0)


def compute_all_curvatures(G: nx.Graph, alpha: float = 0.5) -> Dict[Tuple, float]:
    """Compute Ricci curvature for all edges in the graph"""
    curvatures = {}
    for u, v in G.edges():
        curvatures[(u, v)] = ollivier_ricci_curvature(G, (u, v), alpha)
        curvatures[(v, u)] = curvatures[(u, v)]  # Symmetric
    return curvatures


def apply_ricci_flow(G: nx.Graph, curvatures: Dict[Tuple, float], dt: float = 0.01) -> nx.Graph:
    """
    Apply Ricci flow to the graph.
    
    Updates edge weights based on curvature:
    - Positive curvature -> weight decreases (more connected)
    - Negative curvature -> weight increases (more tension)
    """
    for (u, v), curvature in curvatures.items():
        if (u, v) in G.edges():
            current_weight = G[u][v].get('weight', 1.0)
            new_weight = current_weight + curvature * dt
            G[u][v]['weight'] = max(0.1, min(10.0, new_weight))
    return G


def curvature_to_phase(curvature: float) -> str:
    """Convert curvature to phase description"""
    if curvature > 0.3:
        return "abundance"
    elif curvature < -0.3:
        return "scarcity"
    else:
        return "equilibrium"


def trust_weighted_curvature(G: nx.Graph, curvatures: Dict[Tuple, float]) -> float:
    """
    Compute trust-weighted median curvature for YCCP signal.
    """
    weighted_curvatures = []
    for (u, v), curv in curvatures.items():
        trust_u = G.nodes[u].get('trust', 0.5)
        trust_v = G.nodes[v].get('trust', 0.5)
        w = (trust_u + trust_v) / 2.0
        weighted_curvatures.append(curv * w)
    
    if not weighted_curvatures:
        return 0.0
    return np.median(weighted_curvatures)


def detect_critical_point(curvatures: Dict[Tuple, float], threshold: float = 0.1) -> bool:
    """
    Detect if system is near critical point.
    Critical point is when curvature distribution has high variance.
    """
    curv_vals = list(curvatures.values())
    if len(curv_vals) < 2:
        return False
    variance = np.var(curv_vals)
    return variance < threshold  # Low variance = near critical


def create_trust_network(n_nodes: int = 50, edge_prob: float = 0.1) -> nx.Graph:
    """Create a random trust network"""
    G = nx.erdos_renyi_graph(n_nodes, edge_prob)
    
    # Add trust attributes
    for node in G.nodes():
        G.nodes[node]['trust'] = np.random.uniform(0.3, 0.9)
        G.nodes[node]['recommendation'] = np.random.uniform(0, 1)
    
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(0.5, 1.5)
        G[u][v]['trust'] = (G.nodes[u]['trust'] + G.nodes[v]['trust']) / 2
    
    return G


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create test network
    G = create_trust_network(30, 0.15)
    
    # Compute curvatures
    curvatures = compute_all_curvatures(G)
    
    print("=" * 60)
    print("RICCI FLOW ON TRUST NETWORK")
    print("=" * 60)
    print(f"Number of edges: {len(G.edges())}")
    print(f"Mean curvature: {np.mean(list(curvatures.values())):.3f}")
    print(f"Curvature std: {np.std(list(curvatures.values())):.3f}")
    
    # Phase detection
    median_curv = trust_weighted_curvature(G, curvatures)
    print(f"Trust-weighted median curvature: {median_curv:.3f}")
    print(f"Phase: {curvature_to_phase(median_curv)}")
    print(f"Near critical point: {detect_critical_point(curvatures)}")
    
    # Apply Ricci flow
    G = apply_ricci_flow(G, curvatures, dt=0.05)
    print("\nAfter Ricci flow:")
    print(f"Mean weight: {np.mean([G[u][v]['weight'] for u,v in G.edges()]):.3f}")
def critical_density_search(trust_graph, density_range=np.linspace(0.05, 0.5, 20)):
    critical_values = []
    for density in density_range:
        # Rewire graph to target density while preserving degree sequence
        graph = rewire_to_density(trust_graph, density)
        
        # Run simulation until either:
        # - System crashes (bankruptcy cascade)
        # - System stabilizes (survives)
        
        survival_rate = monte_carlo_survival(graph)
        critical_values.append((density, survival_rate))
    
    # Find density where survival rate drops sharply
    threshold = find_phase_transition(critical_values)
    print(f"Critical density threshold: {threshold}")
    return threshold
