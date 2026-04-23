#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yusuf Counter-Cycle Model – Main Simulation
Yusuf (counter-cycle) vs Capitalist (debt, interest) system

Author: Marc Daghar
Licence: CC BY-SA 4.0
Mention: Free Dr Aafia Siddiqui !
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any


@dataclass
class YusufConfig:
    """Configuration for the Yusuf counter-cycle model"""
    
    # Time parameters
    T: float = 100.0              # Simulation duration (years)
    dt: float = 0.1               # Time step (years)
    
    # Economic parameters
    need: float = 0.7             # Minimum consumption need per year
    P_mean: float = 1.0           # Average production
    P_amplitude: float = 0.5      # Cycle amplitude (abundance/scarcity)
    period: float = 14.0          # Cycle period (years, 7+7 from Surah Yusuf)
    
    # Capitalist parameters (reference system)
    interest_rate: float = 0.05    # Annual interest rate
    
    # Yusuf parameters
    stock_initial: float = 0.5     # Initial stock (years of consumption)
    threshold_factor: float = 0.3  # Factor for abundance/scarcity thresholds
    
    # Gamification (social credit)
    gamification_enabled: bool = True
    compliance_threshold: float = 0.8   # Threshold for benefits
    penalty_rate: float = 0.3           # Penalty for non-compliance
    reward_rate: float = 0.1            # Reward for compliance
    
    # Noise
    noise_amplitude: float = 0.03        # Production noise
    
    @property
    def P_bar(self) -> float:
        """Abundance threshold (high)"""
        return self.P_mean + self.P_amplitude * self.threshold_factor
    
    @property
    def P_underline(self) -> float:
        """Scarcity threshold (low)"""
        return self.P_mean - self.P_amplitude * self.threshold_factor
    
    @property
    def n_steps(self) -> int:
        """Number of time steps"""
        return int(self.T / self.dt)


@dataclass
class SimulationResult:
    """Results of a simulation"""
    t: np.ndarray                    # Time
    P: np.ndarray                    # Production
    S: np.ndarray                    # Stock
    C: np.ndarray                    # Consumption
    compliance: Optional[np.ndarray] = None   # Compliance score
    config: Optional[YusufConfig] = None
    system_name: str = ""
    crisis_detected: bool = False
    
    @property
    def coverage_ratio(self) -> np.ndarray:
        """Stock / need ratio"""
        return self.S / (self.config.need if self.config else 0.7)
    
    @property
    def final_stock(self) -> float:
        return self.S[-1] if len(self.S) > 0 else 0.0
    
    @property
    def mean_consumption(self) -> float:
        return float(np.mean(self.C))
    
    @property
    def consumption_volatility(self) -> float:
        return float(np.std(self.C))
    
    @property
    def solvency_rate(self) -> float:
        """Percentage of time where stock > 0"""
        return float(np.sum(self.S > 0) / len(self.S) * 100)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_name": self.system_name,
            "final_stock": self.final_stock,
            "mean_consumption": self.mean_consumption,
            "consumption_volatility": self.consumption_volatility,
            "solvency_rate": self.solvency_rate,
            "crisis_detected": self.crisis_detected
        }


class YusufSystem:
    """
    Implementation of the Yusuf counter-cycle system.
    
    Rule:
    - Abundance (P > P_bar) : Minimise consumption, surplus goes to stock
    - Scarcity (P < P_underline) : Draw from stock to maintain consumption
    - Equilibrium : Consumption = production
    """
    
    def __init__(self, config: YusufConfig):
        self.config = config
        self.reset()
    
    def reset(self) -> None:
        """Reset system state"""
        self.S = np.zeros(self.config.n_steps)
        self.C = np.zeros(self.config.n_steps)
        self.S[0] = self.config.stock_initial
        self.compliance = np.zeros(self.config.n_steps) if self.config.gamification_enabled else None
    
    def _compute_production(self) -> np.ndarray:
        """Generate cyclic production with optional noise"""
        t = np.linspace(0, self.config.T, self.config.n_steps)
        
        # Base cycle (period of 14 years)
        P = self.config.P_mean + self.config.P_amplitude * np.sin(2 * np.pi * t / self.config.period)
        
        # Add noise
        if self.config.noise_amplitude > 0:
            noise = np.random.normal(0, self.config.noise_amplitude, len(t))
            P = P + noise
        
        return np.maximum(P, 0.1)
    
    def _update_compliance(self, idx: int, behavior_correct: bool) -> float:
        """Update compliance score (social credit)"""
        if not self.config.gamification_enabled:
            return 0.0
        
        if idx == 0:
            score = 1.0
        else:
            score = self.compliance[idx - 1]
        
        if behavior_correct:
            score = min(1.0, score + self.config.reward_rate * self.config.dt)
        else:
            score = max(0.0, score - self.config.penalty_rate * self.config.dt)
        
        return score
    
    def _get_effective_need(self, score: float) -> float:
        """Modulate need by compliance score"""
        if not self.config.gamification_enabled:
            return self.config.need
        
        if score >= self.config.compliance_threshold:
            # Reward: reduced need
            return self.config.need * (1 - self.config.reward_rate)
        else:
            # Penalty: increased need
            return self.config.need * (1 + self.config.penalty_rate)
    
    def run(self) -> SimulationResult:
        """Run the simulation"""
        self.reset()
        
        P = self._compute_production()
        t = np.linspace(0, self.config.T, self.config.n_steps)
        
        P_bar = self.config.P_bar
        P_underline = self.config.P_underline
        crisis_detected = False
        
        for i in range(1, self.config.n_steps):
            production = P[i]
            stock_prev = self.S[i-1]
            score_prev = self.compliance[i-1] if self.compliance is not None else 1.0
            
            effective_need = self._get_effective_need(score_prev)
            
            # Yusuf rule
            if production > P_bar:
                # ABUNDANCE: save
                self.C[i] = min(production, effective_need)
                dS = (production - self.C[i]) * self.config.dt
                self.S[i] = stock_prev + dS
                behavior_correct = (self.C[i] <= effective_need + 0.1)
                
            elif production < P_underline:
                # SCARCITY: draw from stock
                needed_from_stock = max(0, effective_need - production)
                max_withdraw = stock_prev / self.config.dt if self.config.dt > 0 else 0
                withdraw = min(needed_from_stock, max_withdraw)
                self.C[i] = production + withdraw
                dS = (production - self.C[i]) * self.config.dt
                self.S[i] = stock_prev + dS
                behavior_correct = (withdraw <= stock_prev + 1e-6)
                
            else:
                # EQUILIBRIUM
                self.C[i] = production
                self.S[i] = stock_prev
                behavior_correct = True
            
            # Prevent negative stock
            if self.S[i] < 0:
                self.S[i] = 0
                crisis_detected = True
            
            # Update compliance
            if self.compliance is not None:
                self.compliance[i] = self._update_compliance(i, behavior_correct)
        
        return SimulationResult(
            t=t, P=P, S=self.S, C=self.C,
            compliance=self.compliance, config=self.config,
            system_name="Yusuf (counter-cycle)",
            crisis_detected=crisis_detected
        )


class CapitalistSystem:
    """
    Reference capitalist system with compound interest.
    """
    
    def __init__(self, config: YusufConfig):
        self.config = config
        self.reset()
    
    def reset(self) -> None:
        self.S = np.zeros(self.config.n_steps)
        self.C = np.zeros(self.config.n_steps)
        self.D = np.zeros(self.config.n_steps)
        self.S[0] = self.config.stock_initial
        self.D[0] = 0.5
    
    def _compute_production(self) -> np.ndarray:
        t = np.linspace(0, self.config.T, self.config.n_steps)
        P = self.config.P_mean + self.config.P_amplitude * np.sin(2 * np.pi * t / self.config.period)
        if self.config.noise_amplitude > 0:
            noise = np.random.normal(0, self.config.noise_amplitude, len(t))
            P = P + noise
        return np.maximum(P, 0.1)
    
    def run(self) -> SimulationResult:
        self.reset()
        P = self._compute_production()
        t = np.linspace(0, self.config.T, self.config.n_steps)
        crisis_detected = False
        
        for i in range(1, self.config.n_steps):
            dt = self.config.dt
            
            # Compound interest on debt
            self.D[i] = self.D[i-1] * (1 + self.config.interest_rate * dt)
            debt_service = self.D[i-1] * self.config.interest_rate * dt
            
            # Consumption
            available = P[i] - debt_service
            self.C[i] = min(available, self.config.need)
            
            # Stock dynamics
            dS = (P[i] - self.C[i]) * dt - debt_service
            self.S[i] = max(0, self.S[i-1] + dS)
            
            if self.S[i] == 0 and self.S[i-1] > 0:
                crisis_detected = True
        
        return SimulationResult(
            t=t, P=P, S=self.S, C=self.C,
            config=self.config, system_name="Capitalist (debt, interest)",
            crisis_detected=crisis_detected
        )


class ScenarioComparator:
    """Compares Yusuf and Capitalist systems"""
    
    def __init__(self, config: YusufConfig = None):
        self.config = config or YusufConfig()
    
    def run_single(self) -> Tuple[SimulationResult, SimulationResult]:
        """Run a single comparison"""
        np.random.seed(42)
        yusuf = YusufSystem(self.config)
        capitalist = CapitalistSystem(self.config)
        return yusuf.run(), capitalist.run()
    
    def run_monte_carlo(self, n_simulations: int = 100) -> Dict[str, Any]:
        """Run Monte Carlo simulations"""
        yusuf_metrics = []
        capitalist_metrics = []
        
        for seed in range(n_simulations):
            np.random.seed(seed)
            yusuf = YusufSystem(self.config)
            capitalist = CapitalistSystem(self.config)
            y_res = yusuf.run()
            c_res = capitalist.run()
            yusuf_metrics.append(y_res.to_dict())
            capitalist_metrics.append(c_res.to_dict())
        
        def aggregate(metrics_list):
            return {
                "final_stock_mean": np.mean([m["final_stock"] for m in metrics_list]),
                "final_stock_std": np.std([m["final_stock"] for m in metrics_list]),
                "solvency_rate_mean": np.mean([m["solvency_rate"] for m in metrics_list]),
                "consumption_volatility_mean": np.mean([m["consumption_volatility"] for m in metrics_list])
            }
        
        return {
            "yusuf": aggregate(yusuf_metrics),
            "capitalist": aggregate(capitalist_metrics),
            "n_simulations": n_simulations
        }


if __name__ == "__main__":
    config = YusufConfig(T=50, dt=0.5)
    comparator = ScenarioComparator(config)
    y_res, c_res = comparator.run_single()
    
    print("=" * 60)
    print("YUSUF COUNTER-CYCLE MODEL")
    print("=" * 60)
    print(f"Yusuf final stock    : {y_res.final_stock:.2f}")
    print(f"Capitalist final stock: {c_res.final_stock:.2f}")
    print(f"Yusuf solvency       : {y_res.solvency_rate:.1f}%")
    print(f"Capitalist solvency   : {c_res.solvency_rate:.1f}%")
