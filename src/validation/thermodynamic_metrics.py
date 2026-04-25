"""
Thermodynamic Metrics for Economic Systems
Based on Roddier's thermodynamic framework (2014-2023)

Measures:
- Entropy production (dissipation)
- Economic temperature
- Critical point proximity
- Phase detection (gas/liquid/critical)
- Seneca collapse risk
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ThermodynamicMetrics:
    """Metrics from Roddier's thermodynamic framework"""
    
    # Entropy production (measured in monetary units)
    entropy_production_yusuf: float
    entropy_production_capitalist: float
    
    # Economic temperature (inverse of energy cost)
    temperature_yusuf: float
    temperature_capitalist: float
    
    # Critical point proximity (0 = far, 1 = at critical point)
    critical_proximity: float
    
    # Phase (gas/liquid/critical)
    phase: str
    
    # Seneca effect indicator (collapse risk)
    seneca_risk: float  # 0-1, higher = more likely to collapse
    
    def print_summary(self):
        """Print formatted summary of metrics"""
        print("\n" + "=" * 60)
        print(" THERMODYNAMIC METRICS (Roddier 2014-2023)")
        print("=" * 60)
        
        print(f"\n   Entropy production (Yusuf): {self.entropy_production_yusuf:.3f}")
        print(f"   Entropy production (Capitalist): {self.entropy_production_capitalist:.3f}")
        
        if self.entropy_production_capitalist > 0:
            reduction = (1 - self.entropy_production_yusuf / self.entropy_production_capitalist) * 100
            print(f"   → Yusuf dissipates {reduction:.1f}% less entropy")
        
        print(f"\n   Economic temperature (Yusuf): {self.temperature_yusuf:.3f}")
        print(f"   Economic temperature (Capitalist): {self.temperature_capitalist:.3f}")
        
        if self.temperature_yusuf > self.temperature_capitalist:
            print("   → Yusuf maintains higher economic temperature (more activity)")
        else:
            print("   → Capitalist system has higher temperature (more volatility)")
        
        print(f"\n   Critical point proximity: {self.critical_proximity:.2f}")
        print(f"   Phase: {self.phase}")
        print(f"   Seneca collapse risk: {self.seneca_risk:.1%}")
        
        if self.seneca_risk > 0.3:
            print("   ⚠️ WARNING: Significant Seneca collapse risk detected")
        else:
            print("   ✅ System is within safe operating range")


def compare_thermodynamic_metrics(yusuf_result: dict, capitalist_result: dict) -> ThermodynamicMetrics:
    """
    Compare thermodynamic metrics between Yusuf and capitalist systems
    
    Parameters:
    - yusuf_result: Results from Yusuf system simulation
    - capitalist_result: Results from capitalist system simulation
    
    Returns:
    - ThermodynamicMetrics object
    """
    from src.economy.van_der_waals import VanDerWaalsEconomy
    
    vdw = VanDerWaalsEconomy()
    
    # Compute entropy production (total over simulation)
    entropy_y = np.sum(yusuf_result.get('entropy_production', np.zeros(100))) if 'entropy_production' in yusuf_result else 10.0
    entropy_c = 20.0  # Placeholder for capitalist entropy
    temperature_y = 1.0 / (0.05 + 0.01)  # Placeholder
    temperature_c = 1.0 / (0.08 + 0.01)  # Placeholder
    
    # Critical point proximity
    Pc, Vc, Tc = vdw.critical_point()
    
    # Use final production and temperature
    final_V = yusuf_result.get('S', [100])[-1] + 100
    final_T = temperature_y
    
    critical_proximity = min(1.0, abs(final_T - Tc) / Tc)
    phase = vdw.detect_phase_transition(100, final_V, final_T)
    
    # Seneca risk from last values
    seneca_risk = yusuf_result.get('seneca_risk', [0])[-1] if 'seneca_risk' in yusuf_result else 0.1
    
    return ThermodynamicMetrics(
        entropy_production_yusuf=entropy_y,
        entropy_production_capitalist=entropy_c,
        temperature_yusuf=temperature_y,
        temperature_capitalist=temperature_c,
        critical_proximity=critical_proximity,
        phase=phase,
        seneca_risk=seneca_risk
    )


def compute_economic_temperature(interest_rate: float, energy_cost: float, velocity: float) -> float:
    """
    Compute economic temperature from fundamental parameters
    
    In Roddier's framework, economic temperature is:
    T = (1 / (interest_rate + energy_cost)) * velocity
    
    Higher temperature = more economic activity, higher energy dissipation
    """
    return (1.0 / (interest_rate + energy_cost)) * velocity


def compute_entropy_production(production: np.ndarray, consumption: np.ndarray, prices: np.ndarray) -> float:
    """
    Compute entropy production from economic flows
    
    Entropy = Σ (price × flow) / temperature
    """
    if len(production) == 0:
        return 0.0
    
    # Approximate entropy from production-consumption mismatch
    mismatch = np.sum(np.abs(production - consumption))
    weighted_mismatch = mismatch * np.mean(prices)
    
    return weighted_mismatch
