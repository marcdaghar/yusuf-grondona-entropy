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

"""
Adaptive network rewiring based on Ollivier-Ricci curvature.
Implements Schweitzer et al. (2009) insights:
- Networks have critical density thresholds
- Beyond threshold, metastability collapses
- Disassortative structures provide robustness within limits
"""

import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class PhaseTransitionResult:
    critical_density: float
    survival_rates: List[float]
    densities_tested: List[float]
    collapse_point: int

class AdaptiveTrustNetwork:
    """
    A trust graph that rewires based on Ricci curvature and detects phase transitions.
    """
    
    def __init__(self, 
                 n_agents: int,
                 initial_density: float = 0.1,
                 curvature_threshold: float = 0.01):
        """
        Parameters:
        - n_agents: number of nodes
        - initial_density: initial edge probability
        - curvature_threshold: edges with curvature below this are candidates for rewiring
        """
        self.n_agents = n_agents
        self.graph = nx.fast_gnp_random_graph(n_agents, initial_density, seed=42)
        self.curvature_threshold = curvature_threshold
        self.curvature_history = []
        self.density_history = [initial_density]
        
        # Ensure undirected for Ricci flow
        self.graph = self.graph.to_undirected()
        
    def compute_ollivier_ricci_curvature(self, epsilon: float = 0.1) -> Dict[Tuple[int, int], float]:
        """
        Compute Ollivier-Ricci curvature for all edges.
        
        Approximate using the formula:
        κ(x,y) = 1 - W(m_x, m_y) / d(x,y)
        
        where W is the Wasserstein-1 distance between probability distributions
        centered at x and y.
        
        For computational efficiency, we use the approximation:
        κ(x,y) ≈ (deg(x) + deg(y) - 2 * common_neighbors) / (deg(x) + deg(y))
        """
        curvatures = {}
        
        for (u, v) in self.graph.edges():
            deg_u = self.graph.degree(u)
            deg_v = self.graph.degree(v)
            
            # Count common neighbors (triangles)
            neighbors_u = set(self.graph.neighbors(u))
            neighbors_v = set(self.graph.neighbors(v))
            common = len(neighbors_u.intersection(neighbors_v))
            
            # Approximate curvature (simplified but computationally tractable)
            # High curvature = well-connected, low curvature = bridge/cut edge
            if deg_u + deg_v > 0:
                curvature = (common * 2) / (deg_u + deg_v)
            else:
                curvature = 0.0
            
            curvatures[(u, v)] = curvature
            curvatures[(v, u)] = curvature  # symmetric
        
        self.curvature_history.append(curvatures)
        return curvatures
    
    def adaptive_rewire(self, 
                        rewiring_rate: float = 0.05,
                        preserve_degree_sequence: bool = True):
        """
        Rewire low-curvature edges to potentially increase stability.
        
        Schweitzer et al.: Networks that are bipartite or disassortative lend robustness.
        We rewire to increase disassortativity (connect high-degree to low-degree nodes)
        """
        curvatures = self.compute_ollivier_ricci_curvature()
        
        # Find edges below threshold
        weak_edges = [(u, v) for (u, v), k in curvatures.items() 
                      if k < self.curvature_threshold and u < v]
        
        # Rewire a fraction of weak edges
        n_rewire = int(len(weak_edges) * rewiring_rate)
        
        if n_rewire == 0:
            return
        
        # Sort by curvature (lowest first)
        weak_edges_sorted = sorted(weak_edges, key=lambda e: curvatures[e])
        
        for (u, v) in weak_edges_sorted[:n_rewire]:
            # Remove weak edge
            self.graph.remove_edge(u, v)
            
            if preserve_degree_sequence:
                # Find nodes to connect that increase disassortativity
                # High-degree nodes should connect to low-degree nodes
                available_nodes = list(set(range(self.n_agents)) - {u} - set(self.graph.neighbors(u)))
                
                if available_nodes:
                    # Target: node with degree most different from u's degree
                    target = min(available_nodes, 
                                key=lambda w: abs(self.graph.degree(w) - self.graph.degree(u)))
                    self.graph.add_edge(u, target)
            else:
                # Random rewiring
                target = np.random.choice([w for w in range(self.n_agents) 
                                          if w != u and not self.graph.has_edge(u, w)])
                self.graph.add_edge(u, target)
        
        # Update density
        current_density = 2 * self.graph.number_of_edges() / (self.n_agents * (self.n_agents - 1))
        self.density_history.append(current_density)
    
    def compute_disassortativity(self) -> float:
        """Compute degree assortativity (negative = disassortative)."""
        return nx.degree_assortativity_coefficient(self.graph)
    
    def rich_club_coefficient(self, percentile: float = 0.9) -> float:
        """
        Compute rich club coefficient: connectivity among top-degree nodes.
        
        Schweitzer et al.: Rich clubs form the core of the network.
        """
        degrees = dict(self.graph.degree())
        threshold = np.percentile(list(degrees.values()), percentile * 100)
        
        # Nodes with degree >= threshold
        rich_nodes = [n for n, d in degrees.items() if d >= threshold]
        
        if len(rich_nodes) < 2:
            return 0.0
        
        # Edges among rich nodes
        rich_edges = self.graph.subgraph(rich_nodes).number_of_edges()
        max_possible = len(rich_nodes) * (len(rich_nodes) - 1) / 2
        
        return rich_edges / max_possible if max_possible > 0 else 0.0


def find_critical_density(n_agents: int = 50,
                          density_range: np.ndarray = None,
                          monte_carlo_runs: int = 100,
                          time_steps: int = 200) -> PhaseTransitionResult:
    """
    Find the critical density threshold where the network becomes unstable.
    
    Schweitzer et al.: "Metastable dynamical oscillations become unstable 
    when overall density passes a critical threshold."
    """
    if density_range is None:
        density_range = np.linspace(0.02, 0.3, 15)
    
    survival_rates = []
    
    print("=" * 60)
    print("PHASE TRANSITION DETECTION (Schweitzer et al. 2009)")
    print("=" * 60)
    print(f"Testing densities: {density_range}")
    
    for density in density_range:
        survivors = 0
        
        for run in range(monte_carlo_runs):
            # Create network at this density
            network = AdaptiveTrustNetwork(n_agents, initial_density=density)
            
            # Simulate for time_steps with adaptive rewiring
            crashed = False
            for step in range(time_steps):
                network.adaptive_rewire(rewiring_rate=0.03)
                
                # Crash condition: network fragments OR average curvature collapses
                if network.graph.number_of_edges() < n_agents / 2:
                    crashed = True
                    break
                
                # Check curvature collapse
                curvatures = network.compute_ollivier_ricci_curvature()
                mean_curvature = np.mean(list(curvatures.values()))
                if mean_curvature < 0.01:  # Near zero curvature = critical
                    crashed = True
                    break
            
            if not crashed:
                survivors += 1
        
        survival_rate = survivors / monte_carlo_runs
        survival_rates.append(survival_rate)
        print(f"Density = {density:.3f}: Survival rate = {survival_rate:.2f}")
    
    # Find critical density (where survival rate drops below 0.5)
    critical_idx = np.argmax(np.array(survival_rates) < 0.5)
    if critical_idx > 0:
        critical_density = density_range[critical_idx]
        collapse_point = critical_idx
    else:
        critical_density = density_range[-1]
        collapse_point = len(density_range)
    
    print(f"\nCRITICAL DENSITY THRESHOLD: {critical_density:.3f}")
    print(f"Below this: system is metastable. Above: systemic collapse becomes likely.")
    
    return PhaseTransitionResult(
        critical_density=critical_density,
        survival_rates=survival_rates,
        densities_tested=density_range.tolist(),
        collapse_point=collapse_point
    )


def plot_phase_diagram(result: PhaseTransitionResult):
    """Visualize the phase transition."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(result.densities_tested, result.survival_rates, 'o-', color='red', linewidth=2, markersize=8)
    plt.axvline(x=result.critical_density, color='black', linestyle='--', 
                label=f'Critical Density = {result.critical_density:.3f}')
    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
    plt.fill_between(result.densities_tested, 0, result.survival_rates, 
                     where=[d < result.critical_density for d in result.densities_tested],
                     color='green', alpha=0.3, label='Metastable Region')
    plt.fill_between(result.densities_tested, 0, result.survival_rates,
                     where=[d >= result.critical_density for d in result.densities_tested],
                     color='red', alpha=0.3, label='Collapse Region')
    plt.xlabel('Network Density')
    plt.ylabel('Survival Rate')
    plt.title('Phase Transition in Trust Networks (Schweitzer et al. 2009)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('phase_transition.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    # Test the adaptive network
    net = AdaptiveTrustNetwork(30, initial_density=0.1)
    print(f"Initial disassortativity: {net.compute_disassortativity():.4f}")
    print(f"Initial rich club coefficient: {net.rich_club_coefficient():.4f}")
    
    # Find critical density
    result = find_critical_density(n_agents=40, density_range=np.linspace(0.05, 0.4, 10), monte_carlo_runs=50)
    plot_phase_diagram(result)
