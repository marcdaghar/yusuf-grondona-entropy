"""
Geometric Ricci Network (GRN) with Mesa - Multi-Agent Simulation
Integrates: YCCP, Ricci curvature, Clinical Psychology (Vohs), Trust/Recommendation

Author: Marc Daghar
Based on:
- Mesa (Agent-Based Modeling Framework)
- Ollivier-Ricci curvature for network geometry
- Vohs (2015): money primes reduce social stress
- Yusuf Counter-Cyclical Principle (Surah Yusuf 12:47-48)
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from scipy.optimize import linear_sum_assignment
import warnings
warnings.filterwarnings('ignore')


# -------------------------------
# 1. Ricci Curvature Function
# -------------------------------

def ollivier_ricci_curvature(G, edge):
    """
    Compute Ollivier-Ricci curvature for an edge.
    
    Lower curvature = more hyperbolic/tense
    Higher curvature = more flat/abundant
    """
    u, v = edge
    
    # Closed neighborhoods (including self)
    N_u = set(G.neighbors(u)) | {u}
    N_v = set(G.neighbors(v)) | {v}
    
    # Distance function (shortest path, capped)
    def dist(a, b):
        try:
            return min(1.0, nx.shortest_path_length(G, a, b) / 2.0)
        except:
            return 1.0
    
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
# 2. Agent definition (Economic Zone)
# -------------------------------

class EconomicAgent(Agent):
    """Agent representing a zone/nation with psychological and economic states."""
    
    def __init__(self, unique_id, model, node_name):
        super().__init__(unique_id, model)
        self.name = node_name
        
        # Economic state
        self.pref_silver = 0.5  # 0 = prefer gold, 1 = prefer silver
        self.local_demand = 0.5
        
        # Psychological state (Vohs)
        self.awareness = 0.5
        self.stress = 0.0
        self.neuro_bias = np.random.normal(0, 0.1)
        
        # Social capital
        self.trust = 0.5
        self.recommendation = 0.0
        self.reputation_history = []
        
        # Decision log
        self.last_decision_quality = 0.5
        self.deviation_justification = 0.0
        self.local_curvature = 0.0
    
    def perceive_environment(self):
        """Agent perceives local curvature and global YCCP signal."""
        # Local curvature from incident edges
        incident_edges = [(u, v) for (u, v) in self.model.graph.edges() 
                         if u == self.unique_id or v == self.unique_id]
        curvatures = []
        for u, v in incident_edges:
            if (u, v) in self.model.edge_curvatures:
                curvatures.append(self.model.edge_curvatures[(u, v)])
            elif (v, u) in self.model.edge_curvatures:
                curvatures.append(self.model.edge_curvatures[(v, u)])
        self.local_curvature = np.mean(curvatures) if curvatures else 0.0
        
        # Update stress (Vohs: negative curvature increases stress)
        self.stress = 1.0 / (1.0 + np.exp(5.0 * self.local_curvature))
        
        # Update awareness (higher trust -> higher awareness)
        self.awareness = 1.0 / (1.0 + np.exp(-3.0 * (self.trust - 0.5)))
        
        # Neuroeconomic bias drifts slowly
        self.neuro_bias += np.random.normal(0, 0.01)
        self.neuro_bias = np.clip(self.neuro_bias, -0.2, 0.2)
    
    def decide(self):
        """Make decision about metal preference based on psychology and YCCP."""
        target = 1.0 if self.model.yusuf_signal < 0 else 0.0  # 1=silver, 0=gold
        
        if self.awareness > 0.6:
            # Rational following of YCCP
            delta = 0.1 * (target - self.pref_silver)
            self.deviation_justification = 1.0
        else:
            # Emotion-driven
            if self.stress > 0.7:
                # Flight to silver (safe haven)
                delta = 0.05 * (1.0 - self.pref_silver)
                self.deviation_justification = 0.3
            else:
                # Random exploration
                delta = 0.02 * np.random.randn()
                self.deviation_justification = 0.5
        
        # Add neuroeconomic bias
        delta += self.neuro_bias * 0.05
        
        # Add social influence (neighbors with high trust)
        neighbors = self.model.graph.neighbors(self.unique_id)
        neighbor_prefs = []
        for n in neighbors:
            agent = self.model.agents_by_id.get(n)
            if agent and agent.trust > 0.6:
                neighbor_prefs.append(agent.pref_silver)
        if neighbor_prefs:
            social_influence = 0.03 * (np.mean(neighbor_prefs) - self.pref_silver)
            delta += social_influence
        
        self.pref_silver = np.clip(self.pref_silver + delta, 0.0, 1.0)
        
        # Local demand follows preference
        self.local_demand = self.pref_silver * (1 + self.stress * 0.2)
    
    def receive_feedback(self, decision_quality):
        """Gamification: update trust and recommendation based on decision quality."""
        self.last_decision_quality = decision_quality
        
        if decision_quality > 0.7:
            self.trust = min(1.0, self.trust + 0.05)
            self.recommendation = min(1.0, self.recommendation + 0.1)
        elif decision_quality < 0.3:
            self.trust = max(0.0, self.trust - 0.03)
            self.recommendation = max(-1.0, self.recommendation - 0.05)
        
        # Mean reversion
        self.trust = 0.95 * self.trust + 0.05 * 0.5
        self.reputation_history.append(self.trust)
        
        # Neuroplasticity: good decisions reduce future bias
        if decision_quality > 0.8:
            self.neuro_bias *= 0.99
    
    def step(self):
        """Agent step: perceive, decide, then later receive feedback."""
        self.perceive_environment()
        self.decide()
        return self.pref_silver, self.local_demand


# -------------------------------
# 3. Mesa Model
# -------------------------------

class GRNModel(Model):
    """Geometric Ricci Network model with autonomous agents."""
    
    def __init__(self, nodes, edges, bimetal_ratio_init=15.0, seed=None):
        super().__init__(seed=seed)
        self.nodes = nodes
        self.edges = edges
        self.ratio = bimetal_ratio_init
        self.step_count = 0
        
        # Build graph
        self.graph = nx.Graph()
        for node in nodes:
            self.graph.add_node(node)
        for u, v, vol in edges:
            self.graph.add_edge(u, v, volume=vol)
        
        # Create agents
        self.agents_by_id = {}
        self.schedule = RandomActivation(self)
        for i, node in enumerate(nodes):
            agent = EconomicAgent(i, self, node)
            self.agents_by_id[i] = agent
            self.schedule.add(agent)
        
        # Edge curvature cache
        self.edge_curvatures = {}
        self.yusuf_signal = 0.0
        
        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Gold/Silver Ratio": lambda m: m.ratio,
                "Mean Trust": lambda m: np.mean([a.trust for a in m.agents_by_id.values()]),
                "Mean Stress": lambda m: np.mean([a.stress for a in m.agents_by_id.values()]),
                "Mean Preference": lambda m: np.mean([a.pref_silver for a in m.agents_by_id.values()]),
                "Yusuf Signal": lambda m: m.yusuf_signal,
                "Mean Awareness": lambda m: np.mean([a.awareness for a in m.agents_by_id.values()]),
            },
            agent_reporters={
                "Preference": "pref_silver",
                "Trust": "trust",
                "Stress": "stress",
                "Awareness": "awareness",
                "Decision Quality": "last_decision_quality",
            }
        )
    
    def compute_curvatures(self):
        """Update all edge curvatures."""
        for u, v in self.graph.edges():
            self.edge_curvatures[(u, v)] = ollivier_ricci_curvature(self.graph, (u, v))
    
    def compute_weighted_curvature(self):
        """Compute trust-weighted curvature for YCCP."""
        weighted = []
        for (u, v), curv in self.edge_curvatures.items():
            agent_u = self.agents_by_id.get(u)
            agent_v = self.agents_by_id.get(v)
            if agent_u and agent_v:
                w = (agent_u.trust + agent_v.trust) / 2.0
                weighted.append(curv * w)
        return np.median(weighted) if weighted else 0.0
    
    def update_yusuf_signal(self):
        """Update global YCCP signal based on weighted curvature."""
        median_curv = self.compute_weighted_curvature()
        self.yusuf_signal = np.clip(median_curv, -0.5, 0.5)
    
    def evaluate_decisions(self):
        """Peer evaluation: compute decision quality for each agent."""
        qualities = {}
        for agent in self.agents_by_id.values():
            target = 1.0 if self.yusuf_signal < 0 else 0.0
            deviation = abs(agent.pref_silver - target)
            
            # Justification from psychology
            justification = agent.awareness * (1.0 - agent.stress)
            quality = 1.0 - deviation * (0.5 + 0.5 * justification)
            quality = np.clip(quality, 0.0, 1.0)
            qualities[agent.unique_id] = quality
        return qualities
    
    def update_ratio(self):
        """Update bimetal ratio based on trust-weighted preferences."""
        total_weight = 0.0
        weighted_pref = 0.0
        for agent in self.agents_by_id.values():
            total_weight += agent.trust
            weighted_pref += agent.pref_silver * agent.trust
        
        avg_pref = weighted_pref / total_weight if total_weight > 0 else 0.5
        
        if self.yusuf_signal < 0:
            delta = -0.02 * abs(self.yusuf_signal) * avg_pref
        else:
            delta = 0.01 * self.yusuf_signal * (1.0 - avg_pref)
        
        self.ratio *= (1.0 + delta)
        self.ratio = np.clip(self.ratio, 8.0, 25.0)
    
    def step(self):
        """Advance model by one step."""
        self.step_count += 1
        
        # 1. Update geometry
        self.compute_curvatures()
        self.update_yusuf_signal()
        
        # 2. Agents perceive, decide, and act
        for agent in self.schedule.agents:
            agent.step()
        
        # 3. Evaluate decision quality (gamification)
        qualities = self.evaluate_decisions()
        for agent in self.schedule.agents:
            agent.receive_feedback(qualities[agent.unique_id])
        
        # 4. Update global ratio
        self.update_ratio()
        
        # 5. Update edge volumes based on activity
        for u, v in self.graph.edges():
            agent_u = self.agents_by_id.get(u)
            agent_v = self.agents_by_id.get(v)
            if agent_u and agent_v:
                activity = (agent_u.local_demand + agent_v.local_demand) / 2.0
                current_vol = self.graph[u][v]['volume']
                new_vol = current_vol * (0.99 + 0.02 * activity)
                self.graph[u][v]['volume'] = max(0.1, min(3.0, new_vol))
        
        # Collect data
        self.datacollector.collect(self)


# -------------------------------
# 4. Run Simulation
# -------------------------------

def run_simulation(steps=200, visualize=True):
    """
    Run the GRN multi-agent simulation.
    
    Parameters:
    -----------
    steps : int
        Number of simulation steps
    visualize : bool
        Whether to show visualization
    
    Returns:
    --------
    model, model_data, agent_data
    """
    
    # Define network (Europe 1550-1650)
    nodes = ['Spain', 'France', 'England', 'Netherlands', 'HRE']
    edges = [
        ('Spain', 'France', 1.2),       # silver from Americas -> France
        ('Spain', 'Netherlands', 1.5),  # silver to Dutch Republic
        ('France', 'England', 0.8),     # wine for wool
        ('France', 'HRE', 1.0),         # grain for metals
        ('Netherlands', 'England', 1.3), # cloth trade
        ('Netherlands', 'HRE', 0.9),    # Baltic grain
        ('England', 'HRE', 0.7),        # tin, wool
    ]
    
    # Create and run model
    model = GRNModel(nodes, edges, bimetal_ratio_init=15.0)
    for _ in range(steps):
        model.step()
    
    # Get data
    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()
    
    if visualize:
        # Visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Model-level time series
        axes[0, 0].plot(model_data['Gold/Silver Ratio'], color='darkgoldenrod', linewidth=2)
        axes[0, 0].axhline(y=15.0, color='gray', linestyle='--', label='Initial')
        axes[0, 0].set_ylabel('Ratio')
        axes[0, 0].set_title('Gold/Silver Ratio')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(model_data['Mean Trust'], color='green', linewidth=2)
        axes[0, 1].set_ylabel('Trust')
        axes[0, 1].set_title('Mean Social Trust')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].plot(model_data['Mean Stress'], color='red', linewidth=2)
        axes[0, 2].set_ylabel('Stress')
        axes[0, 2].set_title('Mean Neuroeconomic Stress')
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 0].plot(model_data['Yusuf Signal'], color='purple', linewidth=2)
        axes[1, 0].axhline(y=0.0, color='gray', linestyle='--')
        axes[1, 0].set_ylabel('Signal')
        axes[1, 0].set_title('YCCP Signal (negative = silver phase)')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(model_data['Mean Preference'], color='blue', linewidth=2)
        axes[1, 1].axhline(y=0.5, color='gray', linestyle='--')
        axes[1, 1].set_ylabel('Silver Preference')
        axes[1, 1].set_title('Mean Agent Preference')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].plot(model_data['Mean Awareness'], color='orange', linewidth=2)
        axes[1, 2].set_ylabel('Awareness')
        axes[1, 2].set_title('Mean Self-Awareness (Vohs)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('GRN Multi-Agent Simulation with YCCP & Clinical Psychology', y=1.02, fontsize=12)
        plt.show()
        
        # Final agent states
        print("\n" + "=" * 70)
        print("FINAL AGENT STATES")
        print("=" * 70)
        print(f"{'Zone':12s} | {'Pref(Silver)':12s} | {'Trust':8s} | {'Stress':8s} | {'Aware':8s} | {'Quality':10s}")
        print("-" * 70)
        for agent in model.agents_by_id.values():
            print(f"{agent.name:12s} | {agent.pref_silver:12.3f} | {agent.trust:8.3f} | {agent.stress:8.3f} | {agent.awareness:8.3f} | {agent.last_decision_quality:10.3f}")
        
        print("\n" + "=" * 70)
        print(f"Final Gold/Silver Ratio: {model.ratio:.2f}")
        print(f"Final Yusuf Signal: {model.yusuf_signal:.3f}")
        print("=" * 70)
    
    return model, model_data, agent_data


# -------------------------------
# 5. Interactive Parameter Exploration (Jupyter)
# -------------------------------

def interactive_exploration():
    """
    Run multiple simulations with different parameters.
    Requires ipywidgets (for Jupyter notebooks).
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
        
        def run_with_params(trust_init, awareness_init):
            nodes = ['Spain', 'France', 'England', 'Netherlands', 'HRE']
            edges = [
                ('Spain', 'France', 1.2), ('Spain', 'Netherlands', 1.5),
                ('France', 'England', 0.8), ('France', 'HRE', 1.0),
                ('Netherlands', 'England', 1.3), ('Netherlands', 'HRE', 0.9),
                ('England', 'HRE', 0.7)
            ]
            model = GRNModel(nodes, edges)
            
            # Override initial agent parameters
            for agent in model.agents_by_id.values():
                agent.trust = trust_init
                agent.awareness = awareness_init
                agent.neuro_bias = np.random.normal(0, 0.1)
            
            for _ in range(100):
                model.step()
            
            return model.ratio, np.mean([a.trust for a in model.agents_by_id.values()])
        
        # Create interactive controls
        trust_slider = widgets.FloatSlider(min=0.0, max=1.0, step=0.1, value=0.5, description='Initial Trust')
        awareness_slider = widgets.FloatSlider(min=0.0, max=1.0, step=0.1, value=0.5, description='Initial Awareness')
        output = widgets.Output()
        
        def on_change(change):
            with output:
                output.clear_output()
                ratio, trust = run_with_params(trust_slider.value, awareness_slider.value)
                print(f"Final Ratio: {ratio:.2f}")
                print(f"Final Mean Trust: {trust:.3f}")
        
        trust_slider.observe(on_change, names='value')
        awareness_slider.observe(on_change, names='value')
        display(trust_slider, awareness_slider, output)
        on_change(None)
        
    except ImportError:
        print("ipywidgets not available. Install with: pip install ipywidgets")
        print("This function is for Jupyter notebooks only.")


# -------------------------------
# 6. Main Execution
# -------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("GEOMETRIC RICCI NETWORK (GRN) - MULTI-AGENT SIMULATION")
    print("With: YCCP, Ricci Curvature, Vohs Psychology, Trust/Recommendation")
    print("=" * 70)
    
    # Run standard simulation
    model, model_data, agent_data = run_simulation(steps=200, visualize=True)
    
    # Uncomment for interactive exploration (Jupyter only):
    # interactive_exploration()
