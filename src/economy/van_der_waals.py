"""
Van der Waals Equation of State for Economics
Based on Roddier (2017)

(P + a/V²)(V - b) = R·T

Where:
P = Demand (use value / social pressure)
V = Production volume
T = Supply (exchange value / economic temperature)
a = Public goods coefficient (network effects)
b = Minimum survival volume
R = Economic "gas constant"
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class VanDerWaalsEconomy:
    """
    Implements van der Waals equation of state for economics
    Detects phase transitions: gas (rich) → unstable (crash) → liquid (poor)
    """
    
    a: float = 0.5      # Public goods coefficient (transport, communications)
    b: float = 0.2      # Minimum survival volume (food, water, basic housing)
    R: float = 1.0      # Economic "gas constant"
    
    def demand_from_supply(self, V: float, T: float) -> float:
        """
        Calculate demand P from production V and supply T
        
        van der Waals: P = (R·T)/(V - b) - a/V²
        """
        if V <= self.b:
            return float('inf')  # Cannot go below survival volume
        return (self.R * T) / (V - self.b) - self.a / (V * V)
    
    def supply_from_demand(self, P: float, V: float) -> float:
        """Calculate supply T from demand P and production V"""
        if V <= self.b:
            return float('inf')
        # T = (P + a/V²)(V - b)/R
        return (P + self.a / (V * V)) * (V - self.b) / self.R
    
    def critical_point(self) -> Tuple[float, float, float]:
        """Calculate critical point (Pc, Vc, Tc)"""
        Vc = 3 * self.b
        Pc = self.a / (27 * self.b * self.b)
        Tc = (8 * self.a) / (27 * self.R * self.b)
        return Pc, Vc, Tc
    
    def detect_phase_transition(self, P: float, V: float, T: float) -> str:
        """
        Detect if economy is in unstable region (the "fold")
        
        Returns:
        - "gas": Vapor phase (rich, independent agents)
        - "liquid": Liquid phase (poor, trapped agents)
        - "critical": Near critical point - opalescence
        - "supercritical": Continuous phase
        - "unstable": In the fold - collapse imminent
        """
        Pc, Vc, Tc = self.critical_point()
        
        # Check critical proximity
        if abs(T - Tc) / Tc < 0.05:
            return "critical"
        
        if T > Tc:
            return "supercritical"
        
        # Below critical: check if in unstable fold
        P_ideal = self.R * T / V  # Ideal gas law
        
        if P < P_ideal * 0.7:
            return "gas"      # Vapor phase (rich, independent)
        elif P > P_ideal * 1.3:
            return "liquid"   # Liquid phase (poor, trapped)
        else:
            return "unstable"  # In the fold - collapse imminent
    
    def compute_entropy_production(self, V: float, T: float, V_prev: float, T_prev: float) -> float:
        """
        Compute entropy production rate (dissipation)
        From Roddier's thermodynamic framework
        """
        # Ideal gas entropy: S = R * ln(V * T^(1/(γ-1)))
        # For economic system, γ ≈ 1.2 (between ideal gas and incompressible)
        gamma = 1.2
        R = self.R
        
        S_current = R * np.log(V * T ** (1/(gamma - 1)))
        S_prev = R * np.log(V_prev * T_prev ** (1/(gamma - 1)))
        
        return max(0, S_current - S_prev)  # Non-negative entropy production
    
    def compute_seneca_risk(self, V: float, T: float, V_prev: float, T_prev: float) -> float:
        """
        Compute Seneca effect risk (collapse from complexity)
        
        The Seneca effect: collapse is often faster than growth
        Returns 0-1, higher = more likely to collapse
        """
        # Rate of change of production
        dV = V - V_prev
        
        if dV >= 0:
            return 0.0  # Not collapsing
        
        # If thermal shock exceeds threshold
        dT = T - T_prev
        
        # Seneca cliff: collapse probability increases with:
        # - Rapid production drop
        # - High temperature change (energy shock)
        risk = min(1.0, (-dV / V_prev) * 5.0 + abs(dT / T_prev) * 2.0)
        
        return risk


class YusufSystemWithPhaseTransition:
    """
    Yusuf system with van der Waals phase transition detection
    """
    
    def __init__(self, config, vdw: VanDerWaalsEconomy = None):
        self.config = config
        self.vdw = vdw or VanDerWaalsEconomy()
        self.S = np.zeros(config.n_steps)
        self.C = np.zeros(config.n_steps)
        self.phase = np.zeros(config.n_steps, dtype='<U20')
        self.crisis_detected = False
        self.entropy_production = np.zeros(config.n_steps)
        self.seneca_risk = np.zeros(config.n_steps)
    
    def run(self) -> dict:
        """Run simulation with phase transition detection"""
        config = self.config
        t = np.linspace(0, config.T, config.n_steps)
        
        # Production function (cyclical)
        period = getattr(config, 'period', 20)
        P = config.P_mean + config.P_amplitude * np.sin(2 * np.pi * t / period)
        
        # Initial stock
        self.S[0] = config.need * 2
        
        V_prev = self.S[0] + config.P_mean
        T_prev = 1.0 / (config.interest_rate + 0.01) if hasattr(config, 'interest_rate') else 1.0
        
        for i in range(1, config.n_steps):
            production = P[i]
            stock_prev = self.S[i-1]
            
            # Yusuf rule
            if production > config.P_bar:
                # Abundance: save
                consumption = min(production, config.need * 0.7)
            elif production < config.P_underline:
                # Scarcity: draw from stock
                needed = config.need - production
                withdrawal = min(stock_prev, needed)
                consumption = production + withdrawal
            else:
                consumption = min(production, config.need)
            
            # Update stock
            dS = (production - consumption) * config.dt
            self.S[i] = max(0, stock_prev + dS)
            self.C[i] = consumption
            
            # Van der Waals phase detection
            V = self.S[i] + config.P_mean  # Production volume proxy
            T_econ = 1.0 / (config.interest_rate + 0.01) if hasattr(config, 'interest_rate') else 1.0
            P_demand = config.need / max(self.S[i], 0.1)  # Demand proxy
            
            self.phase[i] = self.vdw.detect_phase_transition(P_demand, V, T_econ)
            
            # Compute entropy production
            self.entropy_production[i] = self.vdw.compute_entropy_production(V, T_econ, V_prev, T_prev)
            
            # Compute Seneca risk
            self.seneca_risk[i] = self.vdw.compute_seneca_risk(V, T_econ, V_prev, T_prev)
            
            # Crisis detection
            if self.phase[i] == "unstable" and self.S[i] < config.need * 0.5:
                self.crisis_detected = True
            
            V_prev, T_prev = V, T_econ
        
        return {
            "t": t,
            "P": P,
            "S": self.S,
            "C": self.C,
            "phase": self.phase,
            "crisis_detected": self.crisis_detected,
            "entropy_production": self.entropy_production,
            "seneca_risk": self.seneca_risk
        }
