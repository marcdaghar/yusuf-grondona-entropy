"""
Geometric Ricci Network (GRN) with Mesa - Advanced Multi-Agent Simulation
Integrates: YCCP, Ricci curvature, Clinical Psychology (Vohs), Trust/Recommendation

ENHANCEMENTS:
1. Spatial environment (NetworkGrid) for agent positioning on graph
2. Batch runs for hyperparameter exploration (mesa.batchrunner)
3. Web interface (Mesa's visualization server with Solara)
4. Q-learning for agents to learn optimal response to YCCP

Author: Marc Daghar
Based on:
- Mesa (Agent-Based Modeling Framework)
- Ollivier-Ricci curvature for network geometry
- Vohs (2015): money primes reduce social stress
- Yusuf Counter-Cyclical Principle (Surah Yusuf 12:47-48)
- Q-learning (Watkins, 1989)
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import warnings
import json
import os

warnings.filterwarnings('ignore')

# Try to import visualization libraries (optional)
try:
    from mesa.visualization import SolaraViz
    from mesa.visualization.components import make_plot_measure, make_space_matplotlib
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: SolaraViz not available. Install with: pip install mesa[vis]")


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
                cost_matrix[i, j] = 1e6
    
    # Solve optimal transport
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    wasserstein = cost_matrix[row_ind, col_ind].sum() / n
    
    curvature = 1.0 - 2.0 * wasserstein
    return np.clip(curvature, -1.0, 1.0)


# -------------------------------
# 2. Q-Learning Agent
# -------------------------------

class QLearningAgent(Agent):
    """
    Economic Agent with Q-learning capability.
    
    Learns optimal metal preference (gold/silver) based on:
    - YCCP signal (state)
    - Reward from decision quality
    """
    
    def __init__(self, unique_id, model, node_name, 
                 learning_rate=0.1, discount_factor=0.95, 
                 exploration_rate=0.2):
        super().__init__(unique_id, model)
        self.name = node_name
        
        # Q-table: state (YCCP signal quantized) -> action (0=gold, 1=silver)
        # 3 states: negative (-1), neutral (0), positive (1)
        self.q_table = np.zeros((3, 2))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
        # Economic state
        self.pref_silver = 0.5
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
        self.last_action = 0
        self.last_state = 0
        self.local_curvature = 0.0
        
        # Learning history
        self.q_learning_history = []
    
    def _quantize_state(self, yusuf_signal):
        """Convert continuous YCCP signal to discrete state."""
        if yusuf_signal < -0.15:
            return 0  # Negative (tense, favor silver)
        elif yusuf_signal > 0.15:
            return 2  # Positive (abundant, favor gold)
        else:
            return 1  # Neutral
    
    def _get_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.exploration_rate:
            # Explore: random action
            return np.random.randint(0, 2)
        else:
            # Exploit: best action from Q-table
            return np.argmax(self.q_table[state])
    
    def _update_q_table(self, state, action, reward, next_state):
        """Q-learning update rule."""
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.discount_factor * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error
        self.q_learning_history.append({
            'step': self.model.step_count,
            'state': state,
            'action': action,
            'reward': reward,
            'q_value': self.q_table[state, action]
        })
    
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
        """Make decision using Q-learning."""
        # Get current state from YCCP signal
        state = self._quantize_state(self.model.yusuf_signal)
        self.last_state = state
        
        # Choose action
        action = self._get_action(state)
        self.last_action = action
        
        # Convert action to preference
        # action=0: prefer gold (pref_silver decreases)
        # action=1: prefer silver (pref_silver increases)
        if action == 0:
            target = 0.0  # gold
        else:
            target = 1.0  # silver
        
        # Smooth update (not instantaneous)
        self.pref_silver = 0.9 * self.pref_silver + 0.1 * target
        
        # Add psychological modulation
        if self.awareness > 0.6:
            # Rational following
            pass
        elif self.stress > 0.7:
            # Stress adds noise
            self.pref_silver += 0.05 * (1.0 - self.pref_silver) * np.random.randn()
        
        self.pref_silver = np.clip(self.pref_silver, 0.0, 1.0)
        self.local_demand = self.pref_silver * (1 + self.stress * 0.2)
    
    def receive_reward(self, decision_quality):
        """Receive reward from decision quality and update Q-learning."""
        self.last_decision_quality = decision_quality
        
        # Reward is decision quality (0 to 1)
        reward = decision_quality
        
        # Get next state
        next_state = self._quantize_state(self.model.yusuf_signal)
        
        # Update Q-table
        self._update_q_table(self.last_state, self.last_action, reward, next_state)
        
        # Gamification: trust update
        if decision_quality > 0.7:
            self.trust = min(1.0, self.trust + 0.05)
            self.recommendation = min(1.0, self.recommendation + 0.1)
        elif decision_quality < 0.3:
            self.trust = max(0.0, self.trust - 0.03)
            self.recommendation = max(-1.0, self.recommendation - 0.05)
        
        # Mean reversion
        self.trust = 0.95 * self.trust + 0.05 * 0.5
        self.reputation_history.append(self.trust)
        
        # Neuroplasticity
        if decision_quality > 0.8:
            self.neuro_bias *= 0.99
    
    def step(self):
        """Agent step: perceive, decide, then later receive reward."""
        self.perceive_environment()
        self.decide()
        return self.pref_silver, self.local_demand
    
    def get_q_table_summary(self):
        """Return Q-table summary."""
        return {
            'gold_preference': float(self.q_table[0, 0]),
            'silver_preference': float(self.q_table[0, 1]),
            'learning_rate': self.learning_rate,
            'exploration_rate': self.exploration_rate
        }


# -------------------------------
# 3. Mesa Model with Spatial Grid
# -------------------------------

class GRNAdvancedModel(Model):
    """
    Geometric Ricci Network model with:
    - NetworkGrid for spatial agent positioning
    - Q-learning agents
    - YCCP signal
    """
    
    def __init__(self, 
                 nodes: List[str],
                 edges: List[Tuple[str, str, float]],
                 bimetal_ratio_init: float = 15.0,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 exploration_rate: float = 0.2,
                 trust_init: float = 0.5,
                 awareness_init: float = 0.5,
                 seed: int = None):
        
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
        
        # Spatial grid (NetworkGrid positions agents on graph nodes)
        self.grid = NetworkGrid(self.graph)
        
        # Create agents with Q-learning
        self.agents_by_id = {}
        self.schedule = RandomActivation(self)
        
        for i, node in enumerate(nodes):
            agent = QLearningAgent(
                i, self, node,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                exploration_rate=exploration_rate
            )
            agent.trust = trust_init
            agent.awareness = awareness_init
            agent.neuro_bias = np.random.normal(0, 0.1)
            
            self.agents_by_id[i] = agent
            self.schedule.add(agent)
            
            # Place agent on grid
            self.grid.place_agent(agent, node)
        
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
                "Mean Q-Value": lambda m: np.mean([np.max(a.q_table) for a in m.agents_by_id.values()]),
                "Exploration Rate": lambda m: np.mean([a.exploration_rate for a in m.agents_by_id.values()]),
            },
            agent_reporters={
                "Preference": "pref_silver",
                "Trust": "trust",
                "Stress": "stress",
                "Awareness": "awareness",
                "Decision Quality": "last_decision_quality",
                "Q-Gold": lambda a: a.q_table[a._quantize_state(a.model.yusuf_signal), 0] if hasattr(a, 'q_table') else 0,
                "Q-Silver": lambda a: a.q_table[a._quantize_state(a.model.yusuf_signal), 1] if hasattr(a, 'q_table') else 0,
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
        
        # 3. Evaluate decision quality (reward)
        qualities = self.evaluate_decisions()
        for agent in self.schedule.agents:
            agent.receive_reward(qualities[agent.unique_id])
        
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
    
    def get_agent_q_tables(self):
        """Export all agent Q-tables for analysis."""
        return {agent.name: agent.q_table.tolist() for agent in self.agents_by_id.values()}


# -------------------------------
# 4. Batch Runner for Hyperparameter Exploration
# -------------------------------

@dataclass
class HyperparameterConfig:
    """Configuration for batch runs."""
    learning_rate: List[float]
    discount_factor: List[float]
    exploration_rate: List[float]
    trust_init: List[float]
    awareness_init: List[float]
    steps: int = 200
    iterations: int = 10


def run_batch_experiments(config: HyperparameterConfig, 
                         nodes: List[str], 
                         edges: List[Tuple[str, str, float]]) -> Dict:
    """
    Run batch experiments to explore hyperparameters.
    
    Returns:
    - Dictionary with results for each parameter combination
    """
    results = []
    
    # Create parameter grid
    param_grid = {
        "learning_rate": config.learning_rate,
        "discount_factor": config.discount_factor,
        "exploration_rate": config.exploration_rate,
        "trust_init": config.trust_init,
        "awareness_init": config.awareness_init,
    }
    
    total_combinations = (len(config.learning_rate) * len(config.discount_factor) * 
                          len(config.exploration_rate) * len(config.trust_init) * 
                          len(config.awareness_init))
    
    print("=" * 70)
    print("BATCH RUN: HYPERPARAMETER EXPLORATION")
    print(f"Total combinations: {total_combinations}")
    print(f"Iterations per combination: {config.iterations}")
    print("=" * 70)
    
    for lr in config.learning_rate:
        for df in config.discount_factor:
            for er in config.exploration_rate:
                for ti in config.trust_init:
                    for ai in config.awareness_init:
                        
                        iteration_results = []
                        for iteration in range(config.iterations):
                            model = GRNAdvancedModel(
                                nodes, edges,
                                learning_rate=lr,
                                discount_factor=df,
                                exploration_rate=er,
                                trust_init=ti,
                                awareness_init=ai,
                                seed=iteration
                            )
                            
                            for _ in range(config.steps):
                                model.step()
                            
                            # Collect final metrics
                            final_ratio = model.ratio
                            final_trust = np.mean([a.trust for a in model.agents_by_id.values()])
                            final_stress = np.mean([a.stress for a in model.agents_by_id.values()])
                            final_pref = np.mean([a.pref_silver for a in model.agents_by_id.values()])
                            final_q = np.mean([np.max(a.q_table) for a in model.agents_by_id.values()])
                            
                            iteration_results.append({
                                'final_ratio': final_ratio,
                                'final_trust': final_trust,
                                'final_stress': final_stress,
                                'final_pref': final_pref,
                                'final_q': final_q
                            })
                        
                        # Average over iterations
                        avg_results = {
                            'learning_rate': lr,
                            'discount_factor': df,
                            'exploration_rate': er,
                            'trust_init': ti,
                            'awareness_init': ai,
                            'avg_final_ratio': np.mean([r['final_ratio'] for r in iteration_results]),
                            'std_final_ratio': np.std([r['final_ratio'] for r in iteration_results]),
                            'avg_final_trust': np.mean([r['final_trust'] for r in iteration_results]),
                            'avg_final_stress': np.mean([r['final_stress'] for r in iteration_results]),
                            'avg_final_q': np.mean([r['final_q'] for r in iteration_results]),
                        }
                        results.append(avg_results)
                        
                        print(f"LR={lr:.2f} DF={df:.2f} ER={er:.2f} Trust={ti:.2f} Aware={ai:.2f} "
                              f"→ Ratio={avg_results['avg_final_ratio']:.2f} "
                              f"Trust={avg_results['avg_final_trust']:.3f} "
                              f"Q={avg_results['avg_final_q']:.3f}")
    
    return results


def analyze_batch_results(results: List[Dict]) -> Dict:
    """
    Analyze batch results to find best parameters.
    
    Returns:
    - Best parameters by different criteria
    """
    # Best by final trust (highest)
    best_by_trust = max(results, key=lambda x: x['avg_final_trust'])
    
    # Best by ratio stability (closest to initial 15.0)
    best_by_stability = min(results, key=lambda x: abs(x['avg_final_ratio'] - 15.0))
    
    # Best by Q-value (highest learning)
    best_by_q = max(results, key=lambda x: x['avg_final_q'])
    
    # Best by low stress
    best_by_stress = min(results, key=lambda x: x['avg_final_stress'])
    
    return {
        'best_by_trust': best_by_trust,
        'best_by_stability': best_by_stability,
        'best_by_q_learning': best_by_q,
        'best_by_low_stress': best_by_stress,
        'summary': {
            'trust_range': (min(r['avg_final_trust'] for r in results), 
                           max(r['avg_final_trust'] for r in results)),
            'ratio_range': (min(r['avg_final_ratio'] for r in results),
                           max(r['avg_final_ratio'] for r in results))
        }
    }


# -------------------------------
# 5. Visualization Functions (Mesa Solara)
# -------------------------------

def agent_portrayal(agent):
    """Define how agents are displayed in the visualization."""
    # Color based on metal preference
    if agent.pref_silver > 0.6:
        color = "silver"
    elif agent.pref_silver < 0.4:
        color = "gold"
    else:
        color = "lightblue"
    
    # Size based on trust
    size = 20 + agent.trust * 30
    
    return {
        "color": color,
        "size": size,
        "edgecolor": "black",
        "alpha": 0.8 + agent.stress * 0.2
    }


def create_visualization(model):
    """Create Solara visualization for the model."""
    if not VISUALIZATION_AVAILABLE:
        print("SolaraViz not available. Install with: pip install mesa[vis]")
        return None
    
    # Define measures to plot
    measures = [
        make_plot_measure("Gold/Silver Ratio"),
        make_plot_measure("Mean Trust"),
        make_plot_measure("Mean Stress"),
        make_plot_measure("Yusuf Signal"),
        make_plot_measure("Mean Q-Value"),
    ]
    
    # Create space visualization
    space_drawer = make_space_matplotlib(agent_portrayal)
    
    # Create visualization page
    viz = SolaraViz(
        model,
        components=[space_drawer] + measures,
        model_params={
            "learning_rate": {
                "type": "slider",
                "value": 0.1,
                "min": 0.01,
                "max": 0.5,
                "step": 0.01
            },
            "exploration_rate": {
                "type": "slider",
                "value": 0.2,
                "min": 0.0,
                "max": 0.5,
                "step": 0.01
            },
            "trust_init": {
                "type": "slider",
                "value": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.05
            }
        },
        name="GRN with Q-Learning"
    )
    
    return viz


# -------------------------------
# 6. Main Execution
# -------------------------------

def run_simulation_with_learning(steps=200, visualize=True):
    """
    Run the advanced GRN simulation with Q-learning.
    
    Parameters:
    -----------
    steps : int
        Number of simulation steps
    visualize : bool
        Whether to show static visualization
    
    Returns:
    --------
    model, model_data, agent_data
    """
    
    # Define network (Europe 1550-1650)
    nodes = ['Spain', 'France', 'England', 'Netherlands', 'HRE', 'Italy', 'Germany']
    edges = [
        ('Spain', 'France', 1.2),
        ('Spain', 'Netherlands', 1.5),
        ('France', 'England', 0.8),
        ('France', 'HRE', 1.0),
        ('Netherlands', 'England', 1.3),
        ('Netherlands', 'HRE', 0.9),
        ('England', 'HRE', 0.7),
        ('France', 'Italy', 1.1),
        ('Italy', 'HRE', 0.8),
        ('Germany', 'HRE', 1.0),
        ('Germany', 'Netherlands', 0.9),
    ]
    
    # Create and run model with Q-learning
    model = GRNAdvancedModel(
        nodes, edges,
        learning_rate=0.15,
        discount_factor=0.95,
        exploration_rate=0.2,
        trust_init=0.5,
        awareness_init=0.5
    )
    
    for _ in range(steps):
        model.step()
    
    # Get data
    model_data = model.datacollector.get_model_vars_dataframe()
    agent_data = model.datacollector.get_agent_vars_dataframe()
    
    if visualize:
        # Visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Model-level time series
        axes[0, 0].plot(model_data['Gold/Silver Ratio'], color='darkgoldenrod', linewidth=2)
        axes[0, 0].axhline(y=15.0, color='gray', linestyle='--', label='Initial')
        axes[0, 0].set_ylabel('Ratio')
        axes[0, 0].set_title('Gold/Silver Ratio')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(model_data['Mean Trust'], color='green', linewidth=2)
        axes[0, 1].set_ylabel('Trust')
        axes[0, 1].set_title('Mean Social Trust (Gamification)')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].plot(model_data['Mean Q-Value'], color='purple', linewidth=2)
        axes[0, 2].set_ylabel('Max Q-Value')
        axes[0, 2].set_title('Mean Q-Value (Learning Progress)')
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 0].plot(model_data['Yusuf Signal'], color='red', linewidth=2)
        axes[1, 0].axhline(y=0.0, color='gray', linestyle='--')
        axes[1, 0].set_ylabel('Signal')
        axes[1, 0].set_title('YCCP Signal (negative = silver phase)')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(model_data['Mean Preference'], color='blue', linewidth=2)
        axes[1, 1].axhline(y=0.5, color='gray', linestyle='--')
        axes[1, 1].set_ylabel('Silver Preference')
        axes[1, 1].set_title('Mean Agent Preference (Q-Learning)')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].plot(model_data['Mean Stress'], color='orange', linewidth=2)
        axes[1, 2].set_ylabel('Stress')
        axes[1, 2].set_title('Mean Neuroeconomic Stress')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('GRN Multi-Agent Simulation with Q-Learning & YCCP', y=1.02, fontsize=12)
        plt.show()
        
        # Final agent states with Q-learning summary
        print("\n" + "=" * 80)
        print("FINAL AGENT STATES WITH Q-LEARNING")
        print("=" * 80)
        print(f"{'Zone':12s} | {'Pref(Silver)':12s} | {'Trust':8s} | {'Stress':8s} | {'Q-Gold':8s} | {'Q-Silver':8s} | {'Exploit':8s}")
        print("-" * 80)
        
        for agent in model.agents_by_id.values():
            state = agent._quantize_state(model.yusuf_signal)
            q_gold = agent.q_table[state, 0] if hasattr(agent, 'q_table') else 0
            q_silver = agent.q_table[state, 1] if hasattr(agent, 'q_table') else 0
            exploit_rate = 1 - agent.exploration_rate
            
            print(f"{agent.name:12s} | {agent.pref_silver:12.3f} | {agent.trust:8.3f} | {agent.stress:8.3f} | {q_gold:8.3f} | {q_silver:8.3f} | {exploit_rate:8.2f}")
        
        print("\n" + "=" * 80)
        print(f"Final Gold/Silver Ratio: {model.ratio:.2f}")
        print(f"Final Yusuf Signal: {model.yusuf_signal:.3f}")
        print(f"Mean Max Q-Value: {np.mean([np.max(a.q_table) for a in model.agents_by_id.values()]):.3f}")
        print("=" * 80)
    
    return model, model_data, agent_data


# -------------------------------
# 7. Batch Run Example
# -------------------------------

def run_batch_example():
    """Example of batch run for hyperparameter exploration."""
    
    # Define network
    nodes = ['Spain', 'France', 'England', 'Netherlands', 'HRE']
    edges = [
        ('Spain', 'France', 1.2), ('Spain', 'Netherlands', 1.5),
        ('France', 'England', 0.8), ('France', 'HRE', 1.0),
        ('Netherlands', 'England', 1.3), ('Netherlands', 'HRE', 0.9),
        ('England', 'HRE', 0.7)
    ]
    
    # Configuration
    config = HyperparameterConfig(
        learning_rate=[0.05, 0.1, 0.2],
        discount_factor=[0.9, 0.95],
        exploration_rate=[0.1, 0.2, 0.3],
        trust_init=[0.3, 0.5, 0.7],
        awareness_init=[0.3, 0.5, 0.7],
        steps=100,
        iterations=5
    )
    
    # Run batch
    results = run_batch_experiments(config, nodes, edges)
    
    # Analyze results
    analysis = analyze_batch_results(results)
    
    print("\n" + "=" * 70)
    print("BATCH RUN ANALYSIS")
    print("=" * 70)
    
    print("\n📊 Best by Trust:")
    best = analysis['best_by_trust']
    print(f"   LR={best['learning_rate']:.2f}, DF={best['discount_factor']:.2f}, "
          f"ER={best['exploration_rate']:.2f}, Trust={best['trust_init']:.2f}, Aware={best['awareness_init']:.2f}")
    print(f"   → Final Trust: {best['avg_final_trust']:.3f}, Ratio: {best['avg_final_ratio']:.2f}")
    
    print("\n📊 Best by Q-Learning:")
    best_q = analysis['best_by_q_learning']
    print(f"   LR={best_q['learning_rate']:.2f}, DF={best_q['discount_factor']:.2f}, "
          f"ER={best_q['exploration_rate']:.2f}")
    print(f"   → Max Q-Value: {best_q['avg_final_q']:.3f}")
    
    print("\n📊 Best by Stability (ratio close to 15):")
    best_s = analysis['best_by_stability']
    print(f"   → Final Ratio: {best_s['avg_final_ratio']:.2f} (±{best_s['std_final_ratio']:.3f})")
    
    return results, analysis


# -------------------------------
# 8. Save/Load Model
# -------------------------------

def save_model(model: GRNAdvancedModel, filepath: str):
    """Save model parameters and Q-tables to JSON."""
    data = {
        'step_count': model.step_count,
        'ratio': model.ratio,
        'yusuf_signal': model.yusuf_signal,
        'nodes': model.nodes,
        'edges': [(u, v, model.graph[u][v]['volume']) for u, v in model.graph.edges()],
        'agent_q_tables': model.get_agent_q_tables(),
        'agent_states': {
            agent.name: {
                'pref_silver': agent.pref_silver,
                'trust': agent.trust,
                'stress': agent.stress,
                'awareness': agent.awareness,
                'learning_rate': agent.learning_rate,
                'exploration_rate': agent.exploration_rate
            } for agent in model.agents_by_id.values()
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Model saved to {filepath}")


def load_model(filepath: str, nodes: List[str], edges: List[Tuple[str, str, float]]) -> GRNAdvancedModel:
    """Load model from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Create new model
    model = GRNAdvancedModel(nodes, edges)
    model.step_count = data['step_count']
    model.ratio = data['ratio']
    model.yusuf_signal = data['yusuf_signal']
    
    # Restore Q-tables
    for agent in model.agents_by_id.values():
        if agent.name in data['agent_q_tables']:
            agent.q_table = np.array(data['agent_q_tables'][agent.name])
        
        if agent.name in data['agent_states']:
            state = data['agent_states'][agent.name]
            agent.pref_silver = state['pref_silver']
            agent.trust = state['trust']
            agent.stress = state['stress']
            agent.awareness = state['awareness']
    
    print(f"Model loaded from {filepath}")
    return model


# -------------------------------
# 9. Main
# -------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("GEOMETRIC RICCI NETWORK (GRN) - ADVANCED MULTI-AGENT SIMULATION")
    print("With: YCCP, Ricci Curvature, Vohs Psychology, Q-Learning")
    print("Enhancements: Spatial Grid, Batch Runs, Web Viz, Reinforcement Learning")
    print("=" * 70)
    
    # Run standard simulation with learning
    model, model_data, agent_data = run_simulation_with_learning(steps=200, visualize=True)
    
    # Uncomment to run batch experiments:
    # results, analysis = run_batch_example()
    
    # Uncomment to save model:
    # save_model(model, "grn_model_saved.json")
    
    # Uncomment to create web visualization (requires mesa[vis]):
    # if VISUALIZATION_AVAILABLE:
    #     viz = create_visualization(model)
    #     viz.show()
