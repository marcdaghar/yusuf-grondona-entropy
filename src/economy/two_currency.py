"""
Two-Currency Bimetallic System
Based on Roddier (2015, 2017) - Watt's governor for currency exchange

- Dominant currency (euro): Store of value, fossil fuel economy
- Cry-currency: Medium of exchange, renewable economy with demurrage
- Adjustable exchange rate encourages circulation
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class TwoCurrencyConfig:
    """Configuration for bi-metallic two-currency system"""
    
    # Time parameters
    T: float = 100.0
    dt: float = 0.1
    n_steps: int = 1000
    
    # Production parameters
    P_mean: float = 100.0
    P_amplitude: float = 30.0
    P_bar: float = 120.0      # Abundance threshold
    P_underline: float = 80.0  # Scarcity threshold
    
    # Need (consumption requirement)
    need: float = 95.0
    
    # Second currency parameters (cry-currency)
    cry_currency_enabled: bool = True
    cry_currency_demurrage: float = 0.03  # 3% annual decay (encourages circulation)
    cry_currency_initial: float = 1.0     # Initial per capita
    
    # Exchange rate mechanism (Watt's governor)
    exchange_rate_fixed: float = 1.0       # Base exchange (1 cry = X euro)
    exchange_rate_elasticity: float = 0.5  # How much rate adjusts to demand
    
    # Tax differentiation (encourages cry-currency use)
    tax_rate_dominant: float = 0.20        # 20% on euro transactions
    tax_rate_cry: float = 0.05             # 5% on cry-currency transactions
    
    # Initial shares
    initial_fossil_share: float = 0.6
    initial_renewable_share: float = 0.4


@dataclass
class TwoCurrencyResult:
    """Results from two-currency simulation"""
    t: np.ndarray
    S_euro: np.ndarray          # Stock in dominant currency
    S_cry: np.ndarray           # Stock in cry-currency
    exchange_rate: np.ndarray   # Exchange rate over time
    P: np.ndarray               # Production
    C: np.ndarray               # Consumption
    phase: np.ndarray           # Economic phase
    renewable_share: np.ndarray
    fossil_share: np.ndarray


class TwoCurrencySystem:
    """
    Implements the two-currency economy from Roddier (2015, 2017)
    
    • Dominant currency (euro): Store of value, fossil fuel economy
    • Cry-currency: Medium of exchange, renewable economy
    • Adjustable exchange rate = Watt's governor
    """
    
    def __init__(self, config: TwoCurrencyConfig):
        self.config = config
        
        # Initialize arrays
        self.S_euro = np.zeros(config.n_steps)
        self.S_cry = np.zeros(config.n_steps)
        self.exchange_rate = np.zeros(config.n_steps)
        self.consumption = np.zeros(config.n_steps)
        self.production = np.zeros(config.n_steps)
        self.phase = np.zeros(config.n_steps, dtype='<U20')
        self.renewable_share = np.zeros(config.n_steps)
        self.fossil_share = np.zeros(config.n_steps)
        
        # Initial conditions
        self.S_euro[0] = config.need * 2  # Initial stock
        self.S_cry[0] = config.need * config.cry_currency_initial
        self.exchange_rate[0] = config.exchange_rate_fixed
        self.renewable_share[0] = config.initial_renewable_share
        self.fossil_share[0] = config.initial_fossil_share
    
    def _compute_production(self, t: float) -> float:
        """Compute production with cyclical variations"""
        config = self.config
        return config.P_mean + config.P_amplitude * np.sin(2 * np.pi * t / config.period) if hasattr(config, 'period') else config.P_mean
    
    def _compute_exchange_rate(self, demand_ratio: float) -> float:
        """
        Exchange rate adjusts like Watt's governor
        
        When cry-currency demand is high, its value appreciates
        (encouraging its use for the renewable economy)
        """
        base = self.config.exchange_rate_fixed
        elasticity = self.config.exchange_rate_elasticity
        
        # Demand ratio: cry transactions / total transactions
        # Higher demand → cry appreciates (lower exchange rate)
        rate = base * (1 - elasticity * (demand_ratio - 0.5))
        return max(0.5, min(1.5, rate))
    
    def _determine_phase(self, production: float, stock: float) -> str:
        """Determine economic phase (abundance/scarcity/equilibrium)"""
        config = self.config
        if production > config.P_bar and stock > config.need:
            return "abundance"
        elif production < config.P_underline or stock < config.need * 0.5:
            return "scarcity"
        else:
            return "equilibrium"
    
    def run(self) -> TwoCurrencyResult:
        """Run two-currency simulation"""
        config = self.config
        t = np.linspace(0, config.T, config.n_steps)
        
        # Pre-compute production
        P = np.array([self._compute_production(ti) for ti in t])
        
        for i in range(1, config.n_steps):
            production = P[i]
            stock_euro_prev = self.S_euro[i-1]
            stock_cry_prev = self.S_cry[i-1]
            
            # Determine phase
            self.phase[i] = self._determine_phase(production, stock_euro_prev + stock_cry_prev * self.exchange_rate[i-1])
            
            # Track shares (gradual shift from fossil to renewable)
            if t[i] < 50:  # First 50 years
                self.renewable_share[i] = min(0.6, self.renewable_share[i-1] + 0.002 * config.dt)
                self.fossil_share[i] = 1 - self.renewable_share[i]
            else:
                self.renewable_share[i] = self.renewable_share[i-1]
                self.fossil_share[i] = self.fossil_share[i-1]
            
            # --- Dominant currency (euro) behavior ---
            # Used for fossil economy, commodities, luxury
            
            # --- Cry-currency behavior ---
            if config.cry_currency_enabled:
                # Demurrage: stock decays (encourages circulation)
                self.S_cry[i] = stock_cry_prev * (1 - config.cry_currency_demurrage * config.dt)
                
                # In abundance: cry-currency circulates more (investment in renewables)
                if self.phase[i] == "abundance":
                    self.S_cry[i] += production * self.renewable_share[i] * 0.3 * config.dt
                
                # In scarcity: cry-currency is used for subsistence
                if self.phase[i] == "scarcity":
                    withdrawal = min(stock_cry_prev, production * self.renewable_share[i])
                    self.S_cry[i] -= withdrawal
            else:
                self.S_cry[i] = stock_cry_prev
            
            # --- Euro currency behavior ---
            # In abundance: save more
            if self.phase[i] == "abundance":
                consumption = min(production, config.need * 0.85)
            elif self.phase[i] == "scarcity":
                # Draw from stock
                needed = config.need - production
                withdrawal = min(stock_euro_prev, needed)
                consumption = production + withdrawal
            else:
                consumption = min(production, config.need)
            
            self.consumption[i] = consumption
            
            # Update euro stock
            dS_euro = (production - consumption) * config.dt
            self.S_euro[i] = max(0, stock_euro_prev + dS_euro)
            
            # Update exchange rate based on relative demand
            demand_ratio = (self.renewable_share[i] / max(self.fossil_share[i], 0.01))
            self.exchange_rate[i] = self._compute_exchange_rate(demand_ratio)
        
        return TwoCurrencyResult(
            t=t,
            S_euro=self.S_euro,
            S_cry=self.S_cry,
            exchange_rate=self.exchange_rate,
            P=P,
            C=self.consumption,
            phase=self.phase,
            renewable_share=self.renewable_share,
            fossil_share=self.fossil_share
        )
