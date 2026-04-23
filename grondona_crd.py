#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grondona System – Commodity Reserve Department (CRD)
Counter-cyclical commodity buffer stock mechanism

Author: Marc Daghar
Licence: CC BY-SA 4.0
Mention: Free Dr Aafia Siddiqui !
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class Commodity:
    """A commodity in the Grondona basket"""
    name: str
    floor_price: float          # Minimum price (trigger to buy)
    ceiling_price: float        # Maximum price (trigger to sell)
    current_price: float        # Current market price
    stockpile: float            # Physical quantity stored
    elasticity: float = 100.0   # Response elasticity to price signals


class CommodityReserveDepartment:
    """
    Grondona System CRD.
    
    Automatically:
    - Buys commodities when prices fall below floor (issues new currency)
    - Sells commodities when prices rise above ceiling (destroys currency)
    
    This creates a counter-cyclical money supply and stabilises both
    the currency and commodity prices.
    """
    
    def __init__(self, commodities: List[Commodity], initial_money_supply: float):
        self.commodities = {c.name: c for c in commodities}
        self.money_supply = initial_money_supply
        self.transaction_cost = 0.001      # 0.1% transaction cost
        self.storage_cost = 0.005          # 0.5% annual storage cost
        
        self.history = {
            'time': [],
            'money_supply': [],
            'total_stockpile_value': [],
            'commodity_prices': {c.name: [] for c in commodities}
        }
    
    def check_market_prices(self, current_prices: Dict[str, float], time_step: float = 1.0) -> Dict[str, float]:
        """
        Core CRD logic.
        
        Returns:
            Dict of transactions (commodity name -> quantity bought/sold)
        """
        transactions = {}
        
        for name, price in current_prices.items():
            commodity = self.commodities[name]
            commodity.current_price = price
            
            if price < commodity.floor_price:
                # BUY: Price below floor - expand money supply
                purchase_qty = (commodity.floor_price - price) * commodity.elasticity
                purchase_qty *= (1 - self.transaction_cost)
                commodity.stockpile += purchase_qty
                
                # Money creation = purchase value
                money_created = purchase_qty * commodity.floor_price
                self.money_supply += money_created
                
                # Apply storage cost
                self.money_supply -= commodity.stockpile * self.storage_cost * time_step
                
                transactions[name] = purchase_qty
                
            elif price > commodity.ceiling_price:
                # SELL: Price above ceiling - contract money supply
                sale_qty = min(commodity.stockpile,
                               (price - commodity.ceiling_price) * commodity.elasticity)
                commodity.stockpile -= sale_qty
                
                # Money destruction = sale value
                money_destroyed = sale_qty * commodity.ceiling_price
                self.money_supply -= money_destroyed
                
                transactions[name] = -sale_qty
        
        return transactions
    
    def get_total_stockpile_value(self) -> float:
        """Calculate total value of all commodity stockpiles"""
        return sum(c.stockpile * c.current_price for c in self.commodities.values())
    
    def get_stockpile_volume(self) -> Dict[str, float]:
        """Get current stockpile volumes"""
        return {name: c.stockpile for name, c in self.commodities.items()}
    
    def record_state(self, t: float):
        """Record current state for analysis"""
        self.history['time'].append(t)
        self.history['money_supply'].append(self.money_supply)
        self.history['total_stockpile_value'].append(self.get_total_stockpile_value())
        for name, c in self.commodities.items():
            self.history['commodity_prices'][name].append(c.current_price)
    
    def get_history_df(self) -> pd.DataFrame:
        """Return history as DataFrame"""
        df = pd.DataFrame({
            'time': self.history['time'],
            'money_supply': self.history['money_supply'],
            'stockpile_value': self.history['total_stockpile_value']
        })
        for name in self.commodities.keys():
            df[f'price_{name}'] = self.history['commodity_prices'][name]
        return df
    
    def velocity_of_money(self, transactions_volume: float) -> float:
        """
        Calculate monetary velocity.
        Velocity = (total transactions) / (money supply per capita)
        """
        if self.money_supply == 0:
            return 0.0
        return transactions_volume / (self.money_supply)


class GrondonaSimulator:
    """Simulates the Grondona system over time"""
    
    def __init__(self, crd: CommodityReserveDepartment, years: int = 50, dt: float = 0.25):
        self.crd = crd
        self.years = years
        self.dt = dt
        self.steps = int(years / dt)
    
    def generate_price_series(self, volatility: float = 0.15) -> List[Dict[str, float]]:
        """Generate stochastic price series for commodities"""
        prices = []
        
        for step in range(self.steps):
            t = step * self.dt
            price_dict = {}
            
            for name, commodity in self.crd.commodities.items():
                mean_price = (commodity.floor_price + commodity.ceiling_price) / 2
                shock = np.random.normal(0, volatility * np.sqrt(self.dt))
                seasonal = 0.1 * np.sin(2 * np.pi * t)
                current = mean_price * (1 + shock + seasonal)
                current = max(commodity.floor_price * 0.8,
                              min(commodity.ceiling_price * 1.2, current))
                price_dict[name] = current
            
            prices.append(price_dict)
        
        return prices
    
    def run(self, volatility: float = 0.15) -> pd.DataFrame:
        """Run the simulation"""
        price_series = self.generate_price_series(volatility)
        
        for step, prices in enumerate(price_series):
            t = step * self.dt
            self.crd.check_market_prices(prices, self.dt)
            self.crd.record_state(t)
        
        return self.crd.get_history_df()


if __name__ == "__main__":
    # Test with 4 commodities (Ahmed 2015 specification)
    commodities = [
        Commodity("Wheat", floor_price=180, ceiling_price=220, current_price=200, stockpile=0),
        Commodity("Copper", floor_price=8000, ceiling_price=12000, current_price=10000, stockpile=0),
        Commodity("Cotton", floor_price=70, ceiling_price=90, current_price=80, stockpile=0),
        Commodity("Rubber", floor_price=140, ceiling_price=180, current_price=160, stockpile=0),
    ]
    
    crd = CommodityReserveDepartment(commodities, initial_money_supply=10000)
    simulator = GrondonaSimulator(crd, years=20, dt=0.25)
    df = simulator.run()
    
    print("=" * 60)
    print("GRONDONA SYSTEM SIMULATION")
    print("=" * 60)
    print(f"Final money supply: {df['money_supply'].iloc[-1]:.2f}")
    print(f"Final stockpile value: {df['stockpile_value'].iloc[-1]:.2f}")
    print(f"Money supply volatility: {df['money_supply'].std():.2f}")
