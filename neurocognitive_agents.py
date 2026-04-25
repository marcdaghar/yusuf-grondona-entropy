#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neurocognitive Agents – Mesa implementation
Pain of paying (insula), anticipated reward (striatum), cognitive control (dlPFC)

Author: Marc Daghar
Licence: CC BY-SA 4.0
Mention: Free Dr Aafia Siddiqui !
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class PaymentMode(Enum):
    CASH = "cash"
    CARD = "card"
    CRYPTO = "crypto"
    MOBILE = "mobile"


class ExpenseCategory(Enum):
    NECESSITY = "necessities"
    LEISURE = "leisure"
    INVESTMENT = "investment"
    SOCIAL = "social"


@dataclass
class NeuralSignals:
    """Simulated brain activation signals"""
    striatum: float = 0.0      # Anticipated reward
    insula: float = 0.0        # Pain of paying
    dlpfc: float = 0.0         # Cognitive control
    amygdala: float = 0.0      # Emotional response


@dataclass
class MentalAccounts:
    """Mental accounting (Thaler, 1985)"""
    necessities: float = 0.35
    leisure: float = 0.25
    savings: float = 0.30
    social: float = 0.10
    
    def get_allocation(self, category: ExpenseCategory) -> float:
        mapping = {
            ExpenseCategory.NECESSITY: self.necessities,
            ExpenseCategory.LEISURE: self.leisure,
            ExpenseCategory.INVESTMENT: self.savings,
            ExpenseCategory.SOCIAL: self.social
        }
        return mapping.get(category, 0.20)


class HumanCognition:
    """
    Neurocognitive model of monetary decision-making.
    
    Integrates:
    - Pain of paying (insula) – Mazar et al. (2017)
    - Anticipated reward (striatum) – Knutson et al. (2007)
    - Cognitive control (dlPFC)
    - Mental accounting (Thaler, 1985)
    - Social influence (contagion of preferences)
    """
    
    def __init__(self, agent, personality_profile: str = "balanced"):
        self.agent = agent
        self._init_neural_parameters(personality_profile)
        
        # Mental accounting
        self.mental_accounts = MentalAccounts()
        
        # Dynamic state
        self.current_mood: float = 0.5
        self.cognitive_load: float = 0.0
        self.social_pressure: Dict[str, float] = defaultdict(float)
        
        # Decision history
        self.decision_history: List[Dict] = []
        self.neural_history: List[NeuralSignals] = []
        
        # Preferred payment mode (socially influenced)
        self.preferred_payment = PaymentMode.CARD
        
        # Social network
        self.social_connections = []
    
    def _init_neural_parameters(self, profile: str):
        """Initialise neural parameters based on personality profile"""
        profiles = {
            "balanced": {'pain_sensitivity': 0.7, 'reward_sensitivity': 0.7, 'self_control': 0.7, 'emotional_reactivity': 0.5},
            "impulsive": {'pain_sensitivity': 0.4, 'reward_sensitivity': 1.2, 'self_control': 0.3, 'emotional_reactivity': 0.8},
            "frugal": {'pain_sensitivity': 1.1, 'reward_sensitivity': 0.5, 'self_control': 0.9, 'emotional_reactivity': 0.4},
            "social": {'pain_sensitivity': 0.6, 'reward_sensitivity': 0.9, 'self_control': 0.5, 'emotional_reactivity': 0.9}
        }
        params = profiles.get(profile, profiles["balanced"])
        self.pain_sensitivity = params['pain_sensitivity']
        self.reward_sensitivity = params['reward_sensitivity']
        self.self_control = params['self_control']
        self.emotional_reactivity = params['emotional_reactivity']
    
    def compute_pain_of_paying(self, amount: float, payment_mode: PaymentMode) -> float:
        """
        Calculate pain of paying (insula activation).
        More abstract payment modes cause less pain.
        """
        base_pain = amount / (self.agent.wealth + 1000)
        
        abstraction_factors = {
            PaymentMode.CASH: 1.0,
            PaymentMode.CARD: 0.55,
            PaymentMode.MOBILE: 0.45,
            PaymentMode.CRYPTO: 0.35
        }
        
        pain = (base_pain * abstraction_factors[payment_mode] * 
                self.pain_sensitivity * (1 + 0.5 * self.cognitive_load))
        
        # Desensitisation from repeated spending
        recent_spending = sum(d['amount'] for d in self.decision_history[-10:] 
                              if d.get('action') == 'spend')
        if recent_spending > self.agent.wealth * 0.3:
            pain *= 0.8
        
        return np.clip(pain, 0.0, 1.0)
    
    def compute_anticipated_reward(self, amount: float, category: ExpenseCategory) -> float:
        """
        Calculate anticipated reward (striatum activation).
        Different categories activate differently.
        """
        category_multipliers = {
            ExpenseCategory.LEISURE: 1.3,
            ExpenseCategory.SOCIAL: 1.2,
            ExpenseCategory.NECESSITY: 0.7,
            ExpenseCategory.INVESTMENT: 0.5
        }
        
        base_reward = amount * category_multipliers.get(category, 1.0)
        mood_mod = 0.8 + 0.4 * self.current_mood
        reward = base_reward * self.reward_sensitivity * mood_mod
        
        return np.clip(reward, 0.0, amount * 1.5)
    
    def compute_cognitive_control(self, pain: float, reward: float) -> float:
        """Simulate cognitive control (dlPFC) in arbitrage"""
        conflict = abs(reward - pain)
        effective_control = self.self_control * (1 - 0.5 * self.cognitive_load)
        return conflict * effective_control
    
    def _get_mental_account_balance(self, category: ExpenseCategory) -> float:
        """Calculate remaining balance in mental account"""
        target_allocation = self.mental_accounts.get_allocation(category)
        current_spending = sum(d['amount'] for d in self.decision_history[-30:]
                                if d.get('category') == category)
        estimated_income = self.agent.wealth * 0.1
        allocated = target_allocation * estimated_income
        return max(0, allocated - current_spending)
    
    def _compute_mental_account_penalty(self, amount: float, category: ExpenseCategory) -> float:
        """Penalty if spending exceeds mental account allocation"""
        balance = self._get_mental_account_balance(category)
        if amount <= balance:
            return 1.0
        overshoot = (amount - balance) / (balance + 1)
        return max(0.3, 1.0 - overshoot)
    
    def update_social_influence(self, neighbors: List):
        """Update cognitive parameters through social contagion"""
        if not neighbors:
            return
        
        neighbor_payments = [n.cognition.preferred_payment for n in neighbors 
                            if hasattr(n, 'cognition')]
        if neighbor_payments:
            dominant = max(set(neighbor_payments), key=neighbor_payments.count)
            influence_strength = 0.05 * len(neighbors) / 10
            if np.random.random() < influence_strength:
                self.preferred_payment = dominant
        
        pain_sensitivities = [n.cognition.pain_sensitivity for n in neighbors 
                             if hasattr(n, 'cognition')]
        if pain_sensitivities:
            self.pain_sensitivity = (0.95 * self.pain_sensitivity + 
                                     0.05 * np.mean(pain_sensitivities))
        
        reward_sensitivities = [n.cognition.reward_sensitivity for n in neighbors 
                                if hasattr(n, 'cognition')]
        if reward_sensitivities:
            self.reward_sensitivity = (0.95 * self.reward_sensitivity + 
                                       0.05 * np.mean(reward_sensitivities))
    
    def compute_social_pressure(self, category: ExpenseCategory) -> float:
        """Calculate social pressure to spend in a category"""
        base_pressure = self.social_pressure.get(category.value, 0.0)
        if hasattr(self.agent, 'social_connections'):
            n_connections = len(self.agent.social_connections)
            pressure = base_pressure * (1 + 0.1 * n_connections)
        else:
            pressure = base_pressure
        return np.clip(pressure, 0.0, 1.0)
    
    def _update_emotional_state(self):
        """Update mood and cognitive load"""
        recent_decisions = self.decision_history[-5:] if self.decision_history else []
        if recent_decisions:
            success_rate = sum(1 for d in recent_decisions 
                               if d.get('satisfaction', 0.5) > 0.6) / len(recent_decisions)
            self.current_mood = 0.5 + 0.3 * (success_rate - 0.5)
        else:
            self.current_mood = 0.5 + 0.1 * np.random.randn()
        
        self.cognitive_load = min(1.0, self.cognitive_load * 0.9 + 0.05 * np.random.rand())
    
    def decide_spend_or_save(self, opportunities: List[Dict]) -> Tuple[str, Optional[Dict]]:
        """
        Main decision function.
        Returns (action, details) where action is 'spend', 'save', or 'invest_web3'
        """
        self._update_emotional_state()
        
        best_utility = -np.inf
        best_decision = None
        
        for opp in opportunities:
            category = opp.get('category', ExpenseCategory.LEISURE)
            amount = opp['amount']
            payment_mode = opp.get('mode', self.preferred_payment)
            
            # Neurocognitive components
            pain = self.compute_pain_of_paying(amount, payment_mode)
            reward = self.compute_anticipated_reward(amount, category)
            control = self.compute_cognitive_control(pain, reward)
            
            # Psychosocial components
            social_pressure = self.compute_social_pressure(category)
            mental_penalty = self._compute_mental_account_penalty(amount, category)
            
            # Emotional valence (amygdala)
            emotional_valence = self._compute_emotional_valence(opp)
            
            # Net utility
            net_utility = (reward * (1 + emotional_valence) - 
                          pain * (1 - control) * (1 + 0.5 * self.cognitive_load))
            net_utility *= mental_penalty
            net_utility *= (1 + social_pressure * 0.5)
            net_utility *= (0.5 + getattr(self.agent, 'passion', 0.5))
            if opp.get('requires_trust', False):
                net_utility *= (0.5 + getattr(self.agent, 'trust', 0.5))
            
            # Record neural signals
            self._record_neural_signals(pain, reward, control, emotional_valence)
            
            if net_utility > best_utility:
                best_utility = net_utility
                best_decision = opp
                best_decision['utility'] = net_utility
        
        SAVE_THRESHOLD = 0.15
        if best_utility < SAVE_THRESHOLD:
            save_amount = self.agent.wealth * 0.05 * (1 + self.self_control)
            self._record_decision('save', {'amount': save_amount})
            return ('save', {'amount': save_amount})
        else:
            self._record_decision('spend', best_decision)
            return ('spend', best_decision)
    
    def _compute_emotional_valence(self, opportunity: Dict) -> float:
        """Emotional response (amygdala)"""
        valence = 0.0
        if 'trust' in opportunity:
            valence += opportunity['trust'] * 0.5
        if 'familiarity' in opportunity:
            valence += opportunity['familiarity'] * 0.3
        if opportunity.get('urgent', False):
            valence += 0.4
        return np.clip(valence * self.emotional_reactivity, -0.5, 0.5)
    
    def _record_neural_signals(self, pain: float, reward: float, control: float, emotion: float):
        signals = NeuralSignals(striatum=reward, insula=pain, dlpfc=control, amygdala=emotion)
        self.neural_history.append(signals)
        if len(self.neural_history) > 100:
            self.neural_history.pop(0)
    
    def _record_decision(self, action: str, details: Dict):
        self.decision_history.append({'action': action, 'timestamp': len(self.decision_history), **details})
        if len(self.decision_history) > 100:
            self.decision_history.pop(0)
    
    def get_neural_profile(self) -> Dict:
        if not self.neural_history:
            return {'striatum': 0, 'insula': 0, 'dlpfc': 0, 'amygdala': 0}
        recent = self.neural_history[-10:]
        return {
            'striatum': np.mean([n.striatum for n in recent]),
            'insula': np.mean([n.insula for n in recent]),
            'dlpfc': np.mean([n.dlpfc for n in recent]),
            'amygdala': np.mean([n.amygdala for n in recent])
        }


class CognitiveAgent:
    """Base class for an agent with neurocognitive faculties"""
    
    def __init__(self, unique_id, model, personality: str = "balanced"):
        self.unique_id = unique_id
        self.model = model
        self.wealth = 1000.0
        self.trust = np.random.uniform(0.3, 0.9)
        self.passion = np.random.uniform(0.2, 0.8)
        self.cognition = HumanCognition(self, personality)
        self.social_connections = []
    
    def step(self):
        """Agent step – generate opportunities and decide"""
        opportunities = self._generate_spending_opportunities()
        action, details = self.cognition.decide_spend_or_save(opportunities)
        
        if action == 'spend':
            self._execute_spending(details)
        elif action == 'save':
            self._execute_saving(details)
        elif action == 'invest_web3':
            self._execute_web3_investment(details)
        
        if self.social_connections:
            self.cognition.update_social_influence(self.social_connections)
    
    def _generate_spending_opportunities(self) -> List[Dict]:
        opportunities = []
        if np.random.random() < 0.7:
            opportunities.append({
                'amount': np.random.uniform(10, 100),
                'category': ExpenseCategory.LEISURE,
                'mode': np.random.choice(list(PaymentMode)),
                'requires_trust': False,
                'familiarity': np.random.uniform(0.3, 0.9)
            })
        if np.random.random() < 0.4:
            opportunities.append({
                'amount': np.random.uniform(20, 150),
                'category': ExpenseCategory.NECESSITY,
                'mode': PaymentMode.CARD,
                'requires_trust': False,
                'urgent': np.random.random() < 0.3
            })
        return opportunities
    
    def _execute_spending(self, details: Dict):
        amount = details['amount']
        if amount <= self.wealth:
            self.wealth -= amount
    
    def _execute_saving(self, details: Dict):
        amount = details['amount']
        self.wealth -= amount
    
    def _execute_web3_investment(self, details: Dict):
        amount = details['amount']
        if amount <= self.wealth:
            self.wealth -= amount


if __name__ == "__main__":
    from mesa import Model
    from mesa.time import RandomActivation
    
    class TestModel(Model):
        def __init__(self, n_agents=10):
            super().__init__()
            self.schedule = RandomActivation(self)
            for i in range(n_agents):
                agent = CognitiveAgent(i, self, np.random.choice(['balanced', 'impulsive', 'frugal', 'social']))
                self.schedule.add(agent)
    
    model = TestModel(10)
    for _ in range(50):
        model.schedule.step()
    
    print("Neurocognitive agents test completed")

class VerbAgent(Agent):
    def step(self):
        # Not just choosing a noun (amount to invest)
        # But executing a verb-sequence:
        
        actions = [
            ("search_for_opportunity", self.scan_market()),
            ("evaluate_trust", self.check_graph_curvature()),
            ("negotiate_term", self.propose_rate()),
            ("execute_transaction", self.exchange()),
            ("update_belief", self.learn_from_outcome())
        ]
        
        for verb, method in actions:
            method()

"""
Ergodicity Economics: Kelly Criterion for Time-Average Wealth Maximization
Based on Peters (2011, 2019) - ergodicity economics
"""

import numpy as np
from typing import Tuple, Optional

class KellyOptimalAgent:
    """
    An agent that maximizes time-average growth rate (Kelly criterion)
    rather than expected value.
    
    For a gamble with p win, b gain (e.g., +50% → b=0.5), 
    and q loss, a loss fraction (e.g., -40% → a=0.4):
    
    Kelly fraction f* = (p*b - q*a) / (b*a)  for multiplicative bets
    
    This is the fraction of wealth to risk that maximizes long-term growth.
    """
    
    def __init__(self, initial_wealth: float = 1000.0):
        self.wealth = initial_wealth
        self.log_wealth_history = [np.log(initial_wealth)]
        
    def kelly_fraction(self, 
                       win_prob: float, 
                       gain_pct: float, 
                       loss_pct: float) -> float:
        """
        Compute optimal fraction to bet.
        
        Parameters:
        - win_prob: probability of winning (0-1)
        - gain_pct: fractional gain if win (e.g., 0.5 for +50%)
        - loss_pct: fractional loss if loss (e.g., 0.4 for -40%)
        
        Returns:
        - f*: fraction of wealth to risk (0 to 1)
        """
        b = gain_pct      # net gain fraction
        a = loss_pct      # net loss fraction (positive number)
        p = win_prob
        q = 1 - p
        
        # Kelly formula for multiplicative bets
        # f* = (p*b - q*a) / (b*a)
        numerator = (p * b) - (q * a)
        denominator = b * a
        
        if denominator == 0:
            return 0.0
        
        f_star = numerator / denominator
        
        # Clamp to [0, 1] - no shorting or leverage beyond wealth
        return np.clip(f_star, 0.0, 1.0)
    
    def apply_bet(self, 
                  fraction: float, 
                  win_prob: float, 
                  gain_pct: float, 
                  loss_pct: float,
                  random_state: Optional[np.random.Generator] = None) -> float:
        """
        Apply a bet using Kelly-optimal fraction and return new wealth.
        """
        if random_state is None:
            random_state = np.random.default_rng()
        
        # Determine outcome
        if random_state.random() < win_prob:
            # Win: wealth increases by gain_pct
            multiplier = 1 + gain_pct * fraction
        else:
            # Loss: wealth decreases by loss_pct
            multiplier = 1 - loss_pct * fraction
        
        self.wealth *= multiplier
        self.log_wealth_history.append(np.log(self.wealth))
        return self.wealth
    
    def time_average_growth_rate(self) -> float:
        """Compute the actual time-average growth rate of wealth."""
        if len(self.log_wealth_history) < 2:
            return 0.0
        # g = (ln(W_T) - ln(W_0)) / T
        return (self.log_wealth_history[-1] - self.log_wealth_history[0]) / len(self.log_wealth_history)


class ExpectedValueAgent:
    """
    Standard expected-value maximizing agent.
    This is the baseline that fails under non-ergodicity.
    """
    
    def __init__(self, initial_wealth: float = 1000.0):
        self.wealth = initial_wealth
        self.history = [initial_wealth]
    
    def apply_bet(self,
                  win_prob: float,
                  gain_pct: float,
                  loss_pct: float,
                  random_state: Optional[np.random.Generator] = None) -> float:
        """
        Expected-value maximizing agent bets full wealth (or fixed fraction)
        because EV is positive.
        """
        if random_state is None:
            random_state = np.random.default_rng()
        
        # Expected value maximizing: if EV > 0, bet everything
        ev = win_prob * gain_pct - (1 - win_prob) * loss_pct
        
        if ev > 0:
            fraction = 1.0  # Full bet (risky!)
        else:
            fraction = 0.0
        
        if random_state.random() < win_prob:
            multiplier = 1 + gain_pct * fraction
        else:
            multiplier = 1 - loss_pct * fraction
        
        self.wealth *= multiplier
        self.history.append(self.wealth)
        return self.wealth


def compare_ergodicity_demonstration():
    """
    Demonstrate the difference between Kelly-optimal and expected-value agents.
    
    The classic multiplicative coin toss:
    - Heads: +50% (gain_pct = 0.5)
    - Tails: -40% (loss_pct = 0.4)
    - Probability: p = 0.5
    
    Expected value per round: +5% (looks good!)
    Time-average growth: negative (~ -5% per round)
    """
    import matplotlib.pyplot as plt
    
    n_rounds = 100
    n_agents = 20  # For ensemble comparison
    
    gain_pct = 0.5   # +50% on heads
    loss_pct = 0.4   # -40% on tails
    win_prob = 0.5
    
    # Ensemble of Kelly agents
    kelly_agents = [KellyOptimalAgent(initial_wealth=1000.0) for _ in range(n_agents)]
    # Ensemble of expected-value agents
    ev_agents = [ExpectedValueAgent(initial_wealth=1000.0) for _ in range(n_agents)]
    
    rng = np.random.default_rng(42)
    
    kelly_wealth_over_time = []
    ev_wealth_over_time = []
    
    for round_i in range(n_rounds):
        # Compute Kelly optimal fraction
        kelly_f = kelly_agents[0].kelly_fraction(win_prob, gain_pct, loss_pct)
        
        round_kelly_wealth = []
        round_ev_wealth = []
        
        for agent in kelly_agents:
            agent.apply_bet(kelly_f, win_prob, gain_pct, loss_pct, rng)
            round_kelly_wealth.append(agent.wealth)
        
        for agent in ev_agents:
            agent.apply_bet(win_prob, gain_pct, loss_pct, rng)
            round_ev_wealth.append(agent.wealth)
        
        kelly_wealth_over_time.append(np.mean(round_kelly_wealth))
        ev_wealth_over_time.append(np.mean(round_ev_wealth))
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(kelly_wealth_over_time, label='Kelly (Time-average optimal)', color='green')
    axes[0].plot(ev_wealth_over_time, label='Expected Value Maximizer', color='red', linestyle='--')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Mean Wealth (log scale)')
    axes[0].set_title('Ensemble Average: Kelly vs Expected Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Also show single trajectories (the real world only gets one path)
    axes[1].plot(kelly_agents[0].log_wealth_history, label='Kelly (single path)', color='green')
    axes[1].plot(ev_agents[0].history, label='Expected Value (single path)', color='red', linestyle='--')
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Log Wealth')
    axes[1].set_title('Single Trajectory: The Only One That Happens')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ergodicity_comparison.png', dpi=150)
    plt.show()
    
    # Statistical validation
    print("=" * 60)
    print("ERGODICITY ECONOMICS DEMONSTRATION")
    print("=" * 60)
    print(f"Kelly optimal fraction: {kelly_f:.3f}")
    print(f"Kelly agent final mean wealth: {np.mean([a.wealth for a in kelly_agents]):.2f}")
    print(f"EV agent final mean wealth: {np.mean([a.wealth for a in ev_agents]):.2f}")
    print(f"Kelly time-average growth rate: {kelly_agents[0].time_average_growth_rate():.4f}")
    print(f"EV time-average growth rate: {ev_agents[0].time_average_growth_rate():.4f}")
    print("\nCONCLUSION: Expected value says +5% per round. Reality is negative growth.")
    print("The Kelly agent survives. The expected-value agent goes to ruin.")
    
    return kelly_agents, ev_agents


if __name__ == "__main__":
    compare_ergodicity_demonstration()
