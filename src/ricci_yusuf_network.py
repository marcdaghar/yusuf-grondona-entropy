"""
Geometric Ricci Network (GRN) with Yusuf's Counter-Cyclical Principle (YCCP)
Minimal simulation on a bimetal world (Gold/Silver) before the Great Divergence.

Author: Marc Daghar
Based on: Ollivier-Ricci curvature, Surah Yusuf (12:47-48)

This model simulates a network of trade zones in early modern Europe (1550-1650)
where:
- Gold/Silver ratio evolves through market forces + Yusuf counter-cyclical principle
- Ricci curvature measures geometric tension in the trust/trade network
- YCCP acts as a governor: during tense phases (negative curvature), 
  the system favors silver (liquidity); during abundant phases, it favors gold (store of value)
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')


# -------------------------------
# 1. Ollivier-Ricci curvature on a graph (simplified)
# -------------------------------

def ollivier_ricci_curvature(G, edge, alpha=0.5):
    """
    Approximate Ollivier-Ricci curvature for an edge (u,v).
    
    Based on Wasserstein-1 distance between neighborhood distributions.
    Lower curvature = more hyperbolic/tense, higher = more flat/abundant.
    
    Parameters:
    -----------
    G : networkx.Graph
        The graph representing the trade network
    edge : tuple (u, v)
        The edge to compute curvature for
    alpha : float
        Parameter for lazy random walk (not used in simplified version)
    
    Returns:
    --------
    curvature : float in [-1, 1]
        Positive = abundant/flat, Negative = tense/hyperbolic
    """
    u, v = edge
    
    # Closed neighborhoods (including self)
    N_u = set(G.neighbors(u)) | {u}
    N_v = set(G.neighbors(v)) | {v}
    
    # Degree normalization
    deg_u = G.degree(u)
    deg_v = G.degree(v)
    
    # Mass distributions (uniform)
    mass_u = {node: 1.0 / len(N_u) for node in N_u}
    mass_v = {node: 1.0 / len(N_v) for node in N_v}
    
    # Distance function (shortest path, capped)
    def dist(a, b):
        return min(1.0, nx.shortest_path_length(G, a, b) / 2.0)
    
    # Build cost matrix for optimal transport
    nodes_union = list(N_u.union(N_v))
    n = len(nodes_union)
    cost_matrix = np.zeros((n, n))
    
    for i, a in enumerate(nodes_union):
        for j, b in enumerate(nodes_union):
            if a in N_u and b in N_v:
                cost_matrix[i, j] = dist(a, b)
            else:
                cost_matrix[i, j] = 1e6  # impossible assignment
    
    # Solve optimal transport (Earth Mover's Distance)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    wasserstein = cost_matrix[row_ind, col_ind].sum() / n
    
    # Curvature normalized to [-1, 1]
    curvature = 1.0 - 2.0 * wasserstein
    return np.clip(curvature, -1.0, 1.0)


# -------------------------------
# 2. Geometric Ricci Network (GRN) with YCCP
# -------------------------------

class GeometricRicciNetwork:
    """
    Geometric Ricci Network with Yusuf Counter-Cyclical Principle.
    
    This class simulates the co-evolution of:
    - Bimetal ratio (gold/silver prices)
    - Trade network geometry (Ricci curvature)
    - Regional preferences for gold vs silver
    - Trade volumes
    
    The Yusuf principle acts as a governor: when the network is tense (negative
    median curvature), the system shifts toward silver (liquidity). When abundant
    (positive curvature), it shifts toward gold (store of value).
    """
    
    def __init__(self, nodes, edges, bimetal_ratio_init=15.0):
        """
        Parameters:
        -----------
        nodes : list of str
            Names of trade zones (cities/regions)
        edges : list of (zone1, zone2, trade_volume)
            Trade connections with initial volumes
        bimetal_ratio_init : float
            Initial gold/silver price ratio (silver = 1)
        """
        self.G = nx.Graph()
        for node in nodes:
            self.G.add_node(node)
        for u, v, vol in edges:
            self.G.add_edge(u, v, volume=vol)
        
        # Bimetal state: gold_price / silver_price
        self.ratio = bimetal_ratio_init  # gold in silver units
        
        # Node-level preferences: 0 = prefer gold, 1 = prefer silver
        self.pref = {node: 0.5 for node in nodes}
        
        # History storage for analysis
        self.history = {
            'ratio': [],
            'curvature': [],
            'yusuf_signal': []
        }
    
    def compute_curvatures(self):
        """Compute Ricci curvature for all edges in the network."""
        curvatures = {}
        for u, v in self.G.edges():
            curvatures[(u, v)] = ollivier_ricci_curvature(self.G, (u, v))
        return curvatures
    
    def yusuf_signal(self, curvatures):
        """
        YCCP: return global signal based on median curvature.
        
        Signal = median curvature clamped to [-0.5, 0.5] for stability.
        Negative = tense phase (scarcity / silver preference)
        Positive = abundant phase (abundance / gold preference)
        """
        curv_vals = list(curvatures.values())
        median_curv = np.median(curv_vals)
        return np.clip(median_curv, -0.5, 0.5)
    
    def yusuf_force(self, curvature, yusuf_global_signal, alpha=0.1, beta=0.05):
        """
        Per-edge Yusuf force modifies the effective curvature.
        
        If global signal < 0 (tense), push toward silver (negative force).
        If global signal > 0 (abundant), push toward gold (positive force).
        """
        if yusuf_global_signal < 0:
            # Tense phase: enhance silver preference
            return -alpha * (1.0 + abs(curvature))
        else:
            # Abundant phase: enhance gold preference
            return +beta * (1.0 + curvature)
    
    def update_preferences(self, curvatures, yusuf_global):
        """Update node-level metal preferences based on local curvature + YCCP."""
        new_pref = {}
        for node in self.G.nodes():
            # Average curvature of incident edges
            incident = list(self.G.edges(node))
            if incident:
                local_curv = np.mean([curvatures[edge] for edge in incident if edge in curvatures])
            else:
                local_curv = 0.0
            
            # YCCP modulation
            if yusuf_global < 0:
                # Tense: push toward silver (pref > 0.5)
                delta = -0.1 * local_curv
            else:
                # Abundant: push toward gold (pref < 0.5)
                delta = -0.05 * local_curv
            
            new_pref[node] = np.clip(self.pref[node] + delta, 0.0, 1.0)
        
        return new_pref
    
    def update_ratio(self, preferences, curvatures, yusuf_global):
        """
        Update bimetal ratio based on aggregate preferences and YCCP.
        
        If global tense (yusuf < 0): silver appreciation (ratio decreases)
        If global abundant (yusuf > 0): gold appreciation (ratio increases)
        """
        # Average preference for silver (higher = more silver demand)
        avg_silver_pref = np.mean(list(preferences.values()))
        
        # Baseline adjustment from preferences
        pref_delta = (avg_silver_pref - 0.5) * 0.05
        
        # YCCP override (counter-cyclical governor)
        if yusuf_global < 0:
            yusuf_delta = -0.02 * abs(yusuf_global)  # decrease ratio (silver up)
        else:
            yusuf_delta = +0.01 * yusuf_global       # increase ratio (gold up)
        
        # Curvature effect: negative curvature -> silver demand
        avg_curv = np.mean(list(curvatures.values())) if curvatures else 0.0
        curv_delta = -0.01 * avg_curv
        
        total_delta = pref_delta + yusuf_delta + curv_delta
        new_ratio = self.ratio * (1.0 + total_delta)
        new_ratio = np.clip(new_ratio, 8.0, 25.0)  # historical bounds
        
        return new_ratio
    
    def step(self, dt=1.0):
        """One time step of the GRN evolution."""
        # Compute current geometry
        curvatures = self.compute_curvatures()
        yusuf_signal = self.yusuf_signal(curvatures)
        
        # Update preferences (neuroeconomic + YCCP)
        self.pref = self.update_preferences(curvatures, yusuf_signal)
        
        # Update bimetal ratio (market + YCCP)
        self.ratio = self.update_ratio(self.pref, curvatures, yusuf_signal)
        
        # Store history
        self.history['ratio'].append(self.ratio)
        self.history['curvature'].append(np.mean(list(curvatures.values())) if curvatures else 0.0)
        self.history['yusuf_signal'].append(yusuf_signal)
        
        # Optional: update graph volumes (simplified)
        for u, v in self.G.edges():
            curv = curvatures[(u, v)]
            vol = self.G[u][v]['volume']
            vol_change = 0.01 * (0.5 - abs(curv)) * (1.0 if yusuf_signal > 0 else -1.0)
            self.G[u][v]['volume'] = max(0.1, vol * (1.0 + vol_change))
        
        return curvatures, yusuf_signal
    
    def simulate(self, steps=100):
        """Run simulation for given number of steps."""
        for t in range(steps):
            self.step()
        return self.history


# -------------------------------
# 3. Create pre-Great Divergence bimetal world (Europe, 1550-1650)
# -------------------------------

# Nodes: major trading zones in early modern Europe
nodes = ['Spain', 'France', 'England', 'Netherlands', 'HRE']

# Edges with initial trade volumes (simplified)
edges = [
    ('Spain', 'France', 1.2),       # silver from Americas -> France
    ('Spain', 'Netherlands', 1.5),  # silver to Dutch Republic
    ('France', 'England', 0.8),     # wine for wool
    ('France', 'HRE', 1.0),         # grain for metals
    ('Netherlands', 'England', 1.3), # cloth trade
    ('Netherlands', 'HRE', 0.9),    # Baltic grain
    ('England', 'HRE', 0.7),        # tin, wool
]

# Initialize GRN with YCCP enabled
grn = GeometricRicciNetwork(nodes, edges, bimetal_ratio_init=15.0)


# -------------------------------
# 4. Run simulation with YCCP enabled vs disabled (comparison)
# -------------------------------

print("=" * 60)
print("Geometric Ricci Network (GRN) with Yusuf's Counter-Cyclical Principle")
print("Bimetal World Simulation (Europe 1550-1650)")
print("=" * 60)

print("\nRunning simulation with YCCP (Yusuf principle enabled)...")
history_yccp = grn.simulate(steps=200)

# Create a second GRN without YCCP for comparison
grn_no_yccp = GeometricRicciNetwork(nodes, edges, bimetal_ratio_init=15.0)

# Override update_ratio to remove YCCP effect
def update_ratio_no_yccp(self, preferences, curvatures, yusuf_global):
    avg_silver_pref = np.mean(list(preferences.values()))
    pref_delta = (avg_silver_pref - 0.5) * 0.05
    avg_curv = np.mean(list(curvatures.values())) if curvatures else 0.0
    curv_delta = -0.01 * avg_curv
    new_ratio = self.ratio * (1.0 + pref_delta + curv_delta)
    return np.clip(new_ratio, 8.0, 25.0)

grn_no_yccp.update_ratio = update_ratio_no_yccp.__get__(grn_no_yccp)

print("Running simulation without YCCP (pure market forces)...")
history_no_yccp = grn_no_yccp.simulate(steps=200)


# -------------------------------
# 5. Visualization
# -------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Ratio over time
axes[0, 0].plot(history_yccp['ratio'], label='With YCCP', color='darkgoldenrod', linewidth=1.5)
axes[0, 0].plot(history_no_yccp['ratio'], label='Without YCCP', color='silver', linewidth=1.5)
axes[0, 0].axhline(y=15.0, color='gray', linestyle='--', label='Initial ratio (1:15)', alpha=0.7)
axes[0, 0].set_ylabel('Gold/Silver Price Ratio', fontsize=11)
axes[0, 0].set_xlabel('Time Step (years)', fontsize=11)
axes[0, 0].set_title('Bimetal Ratio Evolution', fontsize=12)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Mean curvature over time
axes[0, 1].plot(history_yccp['curvature'], label='With YCCP', color='darkred', linewidth=1.5)
axes[0, 1].plot(history_no_yccp['curvature'], label='Without YCCP', color='navy', linewidth=1.5)
axes[0, 1].axhline(y=0.0, color='gray', linestyle='--', alpha=0.7)
axes[0, 1].set_ylabel('Mean Ricci Curvature', fontsize=11)
axes[0, 1].set_xlabel('Time Step (years)', fontsize=11)
axes[0, 1].set_title('Geometric Tension (negative = tense / scarcity)', fontsize=12)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Yusuf signal (only with YCCP)
axes[1, 0].plot(history_yccp['yusuf_signal'], color='green', linewidth=1.5, label='Yusuf signal')
axes[1, 0].axhline(y=0.0, color='gray', linestyle='--', alpha=0.7)
axes[1, 0].set_ylabel('YCCP Signal', fontsize=11)
axes[1, 0].set_xlabel('Time Step (years)', fontsize=11)
axes[1, 0].set_title('Yusuf Counter-Cyclical Signal (negative = silver phase)', fontsize=12)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].fill_between(range(len(history_yccp['yusuf_signal'])), 
                         history_yccp['yusuf_signal'], 0,
                         where=np.array(history_yccp['yusuf_signal']) < 0,
                         color='silver', alpha=0.3, label='Silver phase')
axes[1, 0].fill_between(range(len(history_yccp['yusuf_signal'])), 
                         history_yccp['yusuf_signal'], 0,
                         where=np.array(history_yccp['yusuf_signal']) > 0,
                         color='gold', alpha=0.3, label='Gold phase')

# Final preference distribution
prefs_yccp = list(grn.pref.values())
prefs_no = list(grn_no_yccp.pref.values())

x = np.arange(len(nodes))
width = 0.35

bars1 = axes[1, 1].bar(x - width/2, prefs_yccp, width, label='With YCCP', color='darkgoldenrod', alpha=0.8)
bars2 = axes[1, 1].bar(x + width/2, prefs_no, width, label='Without YCCP', color='silver', alpha=0.8)

axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(nodes, rotation=45, ha='right')
axes[1, 1].set_ylabel('Silver Preference (0=gold, 1=silver)', fontsize=11)
axes[1, 1].set_title('Final Regional Preferences', fontsize=12)
axes[1, 1].legend()
axes[1, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
axes[1, 1].set_ylim(0, 1)
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    height = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.suptitle('Geometric Ricci Network (GRN) with Yusuf\'s Counter-Cyclical Principle\nPre-Great Divergence Bimetal World (1550-1650)', 
             y=1.02, fontsize=14)
plt.savefig('ricci_yusuf_simulation.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()


# -------------------------------
# 6. Print summary
# -------------------------------

print("\n" + "=" * 60)
print("SIMULATION SUMMARY")
print("=" * 60)

print(f"\nFinal gold/silver ratio with YCCP:     {history_yccp['ratio'][-1]:.2f}")
print(f"Final gold/silver ratio without YCCP:  {history_no_yccp['ratio'][-1]:.2f}")

if history_yccp['ratio'][-1] < history_no_yccp['ratio'][-1]:
    print(f"\n📊 YCCP effect: ratio DECREASED (silver appreciated) by {abs(history_yccp['ratio'][-1] - history_no_yccp['ratio'][-1]):.2f}")
else:
    print(f"\n📊 YCCP effect: ratio INCREASED (gold appreciated) by {abs(history_yccp['ratio'][-1] - history_no_yccp['ratio'][-1]):.2f}")

print(f"\n📈 Regional silver preference (with YCCP):")
for node, pref in zip(nodes, prefs_yccp):
    if pref > 0.6:
        metal = "🥈 SILVER"
    elif pref < 0.4:
        metal = "🥇 GOLD"
    else:
        metal = "⚖️ BALANCED"
    print(f"   {node:12s} : {pref:.2f}  {metal}")

print(f"\n📉 Regional silver preference (without YCCP):")
for node, pref in zip(nodes, prefs_no):
    if pref > 0.6:
        metal = "🥈 SILVER"
    elif pref < 0.4:
        metal = "🥇 GOLD"
    else:
        metal = "⚖️ BALANCED"
    print(f"   {node:12s} : {pref:.2f}  {metal}")

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)
print("""
The Geometric Ricci Network (GRN) with Yusuf's Counter-Cyclical Principle (YCCP)
demonstrates how a counter-cyclical governor stabilizes a bimetal economy.

- When the trade network becomes tense (negative Ricci curvature → scarcity),
  YCCP pushes toward silver (liquidity for transactions)
- When the network is abundant (positive curvature → prosperity),
  YCCP pushes toward gold (store of value for savings)

Without YCCP, the bimetal ratio drifts arbitrarily based on market preferences.
With YCCP, the system self-regulates according to the geometric state of the network.

This provides a formal, computational model of Surah Yusuf (12:47-48):
"Save in abundance, consume from stocks in scarcity."
""")

print("\n✅ Simulation complete. Image saved as 'ricci_yusuf_simulation.png'")
