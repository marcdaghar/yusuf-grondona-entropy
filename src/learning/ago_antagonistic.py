"""
Ago-Antagonistic Controller (Bernard-Weil, Roddier 2023)

Two opposing forces that cooperate by opposing:
• Yang: Expansion, innovation, production
• Yin: Contraction, restoration, maintenance

Yusuf's principle couples these phases through stock
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class AgoAntagonisticController:
    """
    Implements Bernard-Weil's ago-antagonistic systems
    
    Yang (expansion) and Yin (contraction) are opposing but
    cooperate to maintain system homeostasis. The Yusuf stock
    is the coupling variable.
    """
    
    config: any
    yang_phase: bool = True  # Start in expansion
    phase_duration: float = 0.0
    yang_history: list = None
    phase_switch_count: int = 0
    
    def __post_init__(self):
        self.yang_history = []
    
    def update(self, stock: float, production: float, need: float) -> Tuple[float, bool]:
        """
        Update phase based on stock and production
        
        Returns:
        - consumption_multiplier: how much to consume (yang) or save (yin)
        - phase_switch: whether phase changed
        """
        config = self.config
        
        # Coverage ratios
        coverage = stock / max(need, 0.01)
        P_ratio = production / max(getattr(config, 'P_mean', 100.0), 0.01)
        
        # Record phase
        self.yang_history.append(self.yang_phase)
        
        # Yang phase (expansion): stock is abundant
        if coverage > 2.0 and P_ratio > 1.0:
            if not self.yang_phase:
                self.yang_phase = True
                self.phase_duration = 0
                self.phase_switch_count += 1
                return 1.2, True  # Consume more (confidence)
        
        # Yin phase (contraction): stock is low
        elif coverage < 0.5 or P_ratio < 0.7:
            if self.yang_phase:
                self.yang_phase = False
                self.phase_duration = 0
                self.phase_switch_count += 1
                return 0.7, True  # Consume less (austerity)
        
        self.phase_duration += config.dt
        
        # Normal behavior
        if self.yang_phase:
            # Yang: consume normally, invest surplus
            return 1.0, False
        else:
            # Yin: conserve, draw from stock carefully
            conservation_factor = max(0.5, 1.0 - self.phase_duration / 10.0)
            return conservation_factor, False
    
    def get_yang_ratio(self) -> float:
        """Return proportion of time spent in Yang phase"""
        if not self.yang_history:
            return 0.5
        return sum(self.yang_history) / len(self.yang_history)


class YusufSystemWithAgoAntagonistic:
    """
    Yusuf system with ago-antagonistic phase coupling
    """
    
    def __init__(self, config):
        self.config = config
        self.controller = AgoAntagonisticController(config)
        self.S = np.zeros(config.n_steps)
        self.C = np.zeros(config.n_steps)
        self.compliance = np.ones(config.n_steps)
        self.yang_phase_history = np.zeros(config.n_steps, dtype=bool)
        self.phase_switches = 0
    
    def run(self) -> dict:
        """Run simulation with ago-antagonistic control"""
        config = self.config
        t = np.linspace(0, config.T, config.n_steps)
        
        # Production function
        period = getattr(config, 'period', 20)
        P = config.P_mean + config.P_amplitude * np.sin(2 * np.pi * t / period)
        
        # Initial stock
        self.S[0] = config.need * 2
        
        for i in range(1, config.n_steps):
            production = P[i]
            stock_prev = self.S[i-1]
            compliance_prev = self.compliance[i-1]
            
            # Get ago-antagonistic adjustment
            consumption_factor, phase_switched = self.controller.update(
                stock_prev, production, config.need
            )
            self.yang_phase_history[i] = self.controller.yang_phase
            
            if phase_switched:
                self.phase_switches += 1
            
            # Adjust need by compliance (social credit system)
            effective_need = config.need * (1 + (1 - compliance_prev) * 0.3)
            effective_need *= consumption_factor
            
            # Yusuf rule with phase-aware consumption
            if production > config.P_bar:
                # Abundance: save aggressively in yang phase
                if self.controller.yang_phase:
                    consumption = min(production, effective_need * 0.8)
                else:
                    consumption = min(production, effective_need)
            
            elif production < config.P_underline:
                # Scarcity: draw from stock more carefully in yin phase
                needed_from_stock = max(0, effective_need - production)
                max_withdraw = stock_prev / config.dt
                
                if not self.controller.yang_phase:
                    # Yin: conserve stock, accept lower consumption
                    needed_from_stock = min(needed_from_stock, effective_need * 0.5)
                
                withdraw = min(needed_from_stock, max_withdraw)
                consumption = production + withdraw
            
            else:
                consumption = min(production, effective_need)
            
            # Update stock
            dS = (production - consumption) * config.dt
            self.S[i] = max(0, stock_prev + dS)
            self.C[i] = consumption
            
            # Update compliance (social credit)
            if hasattr(config, 'with_compliance') and config.with_compliance:
                behavior_correct = (self.S[i] >= 0)
                self.compliance[i] = self._update_compliance(i, behavior_correct)
        
        return {
            "t": t,
            "P": P,
            "S": self.S,
            "C": self.C,
            "compliance": self.compliance,
            "yang_phase": self.yang_phase_history,
            "phase_switches": self.phase_switches,
            "yang_ratio": self.controller.get_yang_ratio()
        }
    
    def _update_compliance(self, i: int, behavior_correct: bool) -> float:
        """Update social credit score"""
        config = self.config
        dt = config.dt
        
        if behavior_correct:
            # Good behavior: compliance increases
            increase = 0.05 * dt
            return min(1.0, self.compliance[i-1] + increase)
        else:
            # Bad behavior: compliance decreases
            decrease = 0.1 * dt
            return max(0.0, self.compliance[i-1] - decrease)
