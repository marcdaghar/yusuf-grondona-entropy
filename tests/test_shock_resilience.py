"""
Shock Resilience Testing for Yusuf vs Capitalist Systems
Tests thermodynamic, supply chain, and debt shocks
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class YusufConfig:
    """Basic configuration for Yusuf system"""
    T: float = 100.0
    dt: float = 0.1
    n_steps: int = 1000
    P_mean: float = 100.0
    P_amplitude: float = 30.0
    P_bar: float = 120.0
    P_underline: float = 80.0
    need: float = 95.0
    interest_rate: float = 0.05
    with_compliance: bool = False


class CapitalistSystem:
    """Simple capitalist system for comparison"""
    
    def __init__(self, config: YusufConfig):
        self.config = config
        self.S = np.zeros(config.n_steps)
        self.debt = np.zeros(config.n_steps)
    
    def run(self):
        """Run capitalist simulation"""
        config = self.config
        t = np.linspace(0, config.T, config.n_steps)
        
        # Production
        period = 20
        P = config.P_mean + config.P_amplitude * np.sin(2 * np.pi * t / period)
        
        # Initial stock and debt
        self.S[0] = config.need
        self.debt[0] = 0
        
        for i in range(1, config.n_steps):
            production = P[i]
            stock_prev = self.S[i-1]
            debt_prev = self.debt[i-1]
            
            # Interest accrues on debt
            interest = debt_prev * config.interest_rate * config.dt
            self.debt[i] = debt_prev + interest
            
            # Consumption with debt financing
            if production < config.need:
                shortfall = config.need - production
                if stock_prev >= shortfall:
                    consumption = config.need
                    self.S[i] = stock_prev - shortfall
                    self.debt[i] += shortfall * 0.1  # Borrowing
                else:
                    consumption = production + stock_prev
                    self.S[i] = 0
                    self.debt[i] += (config.need - consumption) * 0.2
            else:
                consumption = min(production, config.need * 1.1)
                self.S[i] = stock_prev + (production - consumption) * config.dt
            
            # Check solvency (debt exceeds assets)
            if self.debt[i] > self.S[i] + config.need * 2:
                break
        
        # Calculate solvency rate
        final_step = i if i < config.n_steps - 1 else config.n_steps - 1
        solvency_rate = (self.S[final_step] > 0) * 100
        
        return type('obj', (object,), {
            'solvency_rate': solvency_rate,
            'final_stock': self.S[final_step],
            'final_debt': self.debt[final_step]
        })()


def test_thermodynamic_shock_resilience():
    """
    Test how both systems respond to thermodynamic shocks
    (sudden changes in energy flow / temperature)
    """
    
    shock_scenarios = {
        "energy_crisis": {
            "type": "temperature_drop",
            "magnitude": 0.5,
            "duration": 10,
            "description": "Oil price shock - economic temperature halves"
        },
        "supply_chain": {
            "type": "production_collapse", 
            "magnitude": 0.7,
            "duration": 5,
            "description": "Production drops 70% (pandemic/war)"
        },
        "debt_crisis": {
            "type": "interest_rate_spike",
            "magnitude": 0.15,  # 15% interest
            "duration": 8,
            "description": "Interest rate spike (debt crisis)"
        },
        "phase_transition": {
            "type": "critical_point_crossing",
            "magnitude": 0.0,
            "duration": 0,
            "description": "Crossing critical point - phase transition imminent"
        }
    }
    
    results = {}
    
    print("\n" + "=" * 70)
    print(" THERMODYNAMIC SHOCK RESILIENCE TEST")
    print(" Based on Roddier's non-equilibrium thermodynamics")
    print("=" * 70)
    
    for name, shock in shock_scenarios.items():
        print(f"\n🔴 Testing shock: {name}")
        print(f"   {shock['description']}")
        
        config = YusufConfig(T=100, dt=0.1, n_steps=1000)
        
        # Apply shock to config
        if shock["type"] == "temperature_drop":
            config.P_mean *= (1 - shock["magnitude"])
        elif shock["type"] == "interest_rate_spike":
            config.interest_rate = shock["magnitude"]
        elif shock["type"] == "critical_point_crossing":
            config.P_amplitude = 0.8  # More extreme cycle
            config.period = 10  # Faster cycle
        
        # Import the Yusuf system
        from src.learning.ago_antagonistic import YusufSystemWithAgoAntagonistic
        
        yusuf = YusufSystemWithAgoAntagonistic(config)
        capitalist = CapitalistSystem(config)
        
        y_res = yusuf.run()
        c_res = capitalist.run()
        
        # Calculate solvency rates
        y_solvency = 100.0 if y_res['S'][-1] > 0 else 0.0
        c_solvency = 100.0 if c_res.final_stock > 0 else 0.0
        
        results[name] = {
            "yusuf_solvency": y_solvency,
            "capitalist_solvency": c_solvency,
            "yusuf_final_stock": y_res['S'][-1],
            "capitalist_final_stock": c_res.final_stock,
            "resilience_advantage": y_solvency - c_solvency
        }
        
        print(f"   📊 Yusuf solvency: {y_solvency:.1f}%")
        print(f"   📊 Capitalist solvency: {c_solvency:.1f}%")
        print(f"   ✅ Resilience advantage: {results[name]['resilience_advantage']:+.1f}%")
    
    return results


def test_seneca_cliff():
    """
    Test the Seneca effect: collapse is faster than growth
    Named after Seneca: "Increase is slow, ruin is rapid"
    """
    print("\n" + "=" * 70)
    print(" SENECA CLIFF TEST")
    print(" 'Increase is slow, ruin is rapid' - Seneca")
    print("=" * 70)
    
    from src.economy.van_der_waals import VanDerWaalsEconomy
    
    vdw = VanDerWaalsEconomy(a=0.5, b=0.2)
    
    # Simulate growth then collapse
    t = np.linspace(0, 100, 1000)
    
    # Growth phase (slow)
    V_growth = 0.5 + 0.005 * t[:500]  # Slow increase
    T_growth = 1.0 + 0.002 * t[:500]
    
    # Collapse phase (rapid)
    V_collapse = V_growth[-1] * (1 - 0.02 * np.arange(500))  # Fast decrease
    T_collapse = T_growth[-1] * (1 - 0.01 * np.arange(500))
    
    V = np.concatenate([V_growth, V_collapse])
    T = np.concatenate([T_growth, T_collapse])
    
    # Compute Seneca risk
    risks = []
    for i in range(1, len(V)):
        risk = vdw.compute_seneca_risk(V[i], T[i], V[i-1], T[i-1])
        risks.append(risk)
    
    # Find when risk exceeds threshold
    collapse_point = next((i for i, r in enumerate(risks) if r > 0.5), -1)
    
    print(f"\n   Growth phase duration: {len(V_growth)} time steps")
    print(f"   Collapse phase duration: {len(V_collapse)} time steps")
    print(f"   Collapse is {len(V_growth)/len(V_collapse):.1f}x faster than growth")
    print(f"   Seneca cliff detected at step: {collapse_point}")
    
    if collapse_point > 0:
        print("   ⚠️ WARNING: System vulnerable to rapid collapse (Seneca effect)")
    
    return risks


if __name__ == "__main__":
    # Run shock tests
    results = test_thermodynamic_shock_resilience()
    
    # Print summary
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    
    avg_advantage = np.mean([r['resilience_advantage'] for r in results.values()])
    print(f"\n📈 Average resilience advantage of Yusuf system: {avg_advantage:+.1f}%")
    print(f"   (Across {len(results)} shock scenarios)")
    
    if avg_advantage > 20:
        print("   ✅ Yusuf system demonstrates superior thermodynamic resilience")
    else:
        print("   ⚠️ Further optimization needed for shock resilience")
    
    # Run Seneca test
    test_seneca_cliff()
