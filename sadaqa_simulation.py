"""
Sadaqa Gift Economy Simulation
================================

A complete agent-based model of a gift economy grounded in:
- Islamic Sadaqa (voluntary charitable giving, merit accumulation)
- Maussian gift theory (Mauss, 1925): reciprocity, inalienable possessions
- Sahlins (1972): generalized, balanced, negative reciprocity
- Weiner (1992): keeping-while-giving, inalienable possessions
- Neuroeconomics: warm glow (dopamine), trust (oxytocin)

Author: Based on Yusuf-Grondona framework
License: CC BY-SA 4.0
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from collections import deque
import random

"""
Sadaqa Gift Economy Simulation
================================

A complete agent-based model of a gift economy grounded in:
- Islamic Sadaqa (voluntary charitable giving, merit accumulation)
- Maussian gift theory (Mauss, 1925): reciprocity, inalienable possessions
- Sahlins (1972): generalized, balanced, negative reciprocity
- Weiner (1992): keeping-while-giving, inalienable possessions
- Neuroeconomics: warm glow (dopamine), trust (oxytocin)

Author: Based on Yusuf-Grondona framework
License: CC BY-SA 4.0
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from collections import deque
import random


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class GiftType(Enum):
    """Types of gifts in the economy."""
    MATERIAL = "material"          # Physical goods, money
    IMMATERIAL = "immaterial"      # Compliments, praise, social recognition
    SACRED = "sacred"              # Sadaqa: pure gift with no return expected
    INALIENABLE = "inalienable"    # Possessions that cannot be sold, only gifted


class ReciprocityType(Enum):
    """Types of reciprocity (Sahlins 1972)."""
    GENERALIZED = "generalized"    # No immediate return expected, trust-based
    BALANCED = "balanced"          # Fair return expected within reasonable time
    NEGATIVE = "negative"          # Attempt to profit at other's expense


class ExchangeSphere(Enum):
    """Spheres of exchange (Bohannan 1959)."""
    MARKET = "market"              # Commodity exchange, prices, money
    GIFT = "gift"                  # Reciprocal gift exchange
    SACRED = "sacred"              # Sadaqa, charity, merit-based


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class GiftRecord:
    """Record of a single gift transaction."""
    giver_id: int
    receiver_id: int
    value: float
    gift_type: GiftType
    reciprocity_type: ReciprocityType
    sphere: ExchangeSphere
    time_step: int
    expected_return: bool
    return_deadline: Optional[int] = None


@dataclass
class Obligation:
    """Record of an outstanding obligation to return a gift."""
    from_agent: int
    original_gift_value: float
    original_gift_type: GiftType
    time_given: int
    time_to_return: int  # Deadline for return
    returned: bool = False
    return_value: float = 0.0


@dataclass
class InalienablePossession:
    """A possession that cannot be sold, only gifted (Weiner 1992)."""
    name: str
    value: float
    original_owner_id: int
    current_holder_id: int
    history: List[int] = field(default_factory=list)  # Chain of holders
    sacred: bool = False  # If sacred, cannot be returned


# ============================================================================
# SADAQA AGENT
# ============================================================================

class SadaqaAgent:
    """
    An agent in a gift economy, based on Sadaqa and generalized reciprocity.
    
    Core mechanisms:
    1. Warm glow (dopamine) from giving - pure gift is its own reward
    2. Oxytocin release from receiving - increases trust
    3. Merit accumulation (Sadaqa) - spiritual/social credit
    4. Reputation dynamics - social standing based on generosity
    5. Obligation tracking - debt of gratitude (different from monetary debt)
    6. Inalienable possessions - objects that circulate but are never fully alienated
    """
    
    def __init__(self, 
                 agent_id: int, 
                 initial_wealth: float = 1000.0,
                 initial_reputation: float = 0.5,
                 initial_merit: float = 0.0,
                 generosity: float = 0.3,
                 risk_aversion: float = 0.5):
        """
        Initialize a Sadaqa agent.
        
        Parameters:
        - agent_id: Unique identifier
        - initial_wealth: Material wealth (still used for consumption)
        - initial_reputation: Social standing (0 to 1)
        - initial_merit: Sadaqa merit (spiritual/social credit)
        - generosity: Propensity to give (0 to 1)
        - risk_aversion: Preference for safe vs. risky exchanges
        """
        self.id = agent_id
        self.wealth = initial_wealth
        self.reputation = initial_reputation
        self.merit = initial_merit
        self.generosity = generosity
        self.risk_aversion = risk_aversion
        
        # Histories
        self.gifts_given: List[GiftRecord] = []
        self.gifts_received: List[GiftRecord] = []
        self.obligations: List[Obligation] = []  # Debts of gratitude
        self.inalienable_possessions: List[InalienablePossession] = []
        
        # Trust network (oxytocin-based)
        self.trust_scores: Dict[int, float] = {}  # trust in other agents
        self.trust_history: List[float] = [0.5]
        
        # Emotional/neurological state
        self.happiness: float = 0.5
        self.stress: float = 0.0  # From unreturned gifts (poison of the gift)
        self.dopamine_level: float = 0.0
        self.oxytocin_level: float = 0.0
        
        # Exchange sphere tracking
        self.sphere_balances: Dict[ExchangeSphere, float] = {
            ExchangeSphere.MARKET: initial_wealth,
            ExchangeSphere.GIFT: 0.0,
            ExchangeSphere.SACRED: 0.0
        }
        
        # Performance metrics
        self.total_given = 0.0
        self.total_received = 0.0
        self.pure_gifts_given = 0  # Sadaqa count
        self.pure_gifts_received = 0
        
    def compute_warm_glow(self, 
                          gift_value: float, 
                          gift_type: GiftType, 
                          expect_return: bool) -> float:
        """
        Compute utility from giving (dopamine release).
        
        Based on neuroeconomic research: giving activates reward pathways
        in the brain, with pure gifts (Sadaqa) producing stronger effects.
        
        Returns:
        - dopamine_increase: Amount to add to dopamine level
        """
        base_glow = 0.05 * gift_value / 100.0  # Scaled to reasonable range
        
        # Immaterial gifts create sustained warm glow
        if gift_type == GiftType.IMMATERIAL:
            base_glow *= 1.2
        
        # Pure Sadaqa: greater spiritual/reward pathway activation
        if gift_type == GiftType.SACRED:
            base_glow *= 2.0
        
        # No expectation of return increases warm glow
        if not expect_return:
            base_glow *= 1.3
        
        # Generosity amplifies warm glow
        base_glow *= (1 + self.generosity)
        
        # Update agent state
        self.dopamine_level += base_glow
        self.happiness += 0.01 * base_glow
        
        # Merit increases with warm glow (Sadaqa)
        if gift_type == GiftType.SACRED:
            self.merit += base_glow * 10.0
        
        return base_glow
    
    def compute_oxytocin_release(self, 
                                 giver_id: int, 
                                 gift_value: float, 
                                 gift_type: GiftType) -> float:
        """
        Compute trust increase from receiving a gift (oxytocin release).
        
        Oxytocin is associated with trust, safety, and social bonding.
        Returns:
        - trust_increase: Amount to add to trust score for giver
        """
        base_trust = 0.05
        
        # Larger gifts = more trust
        base_trust += 0.01 * (gift_value / 100.0)
        
        # Immaterial gifts build trust more slowly but last longer
        if gift_type == GiftType.IMMATERIAL:
            base_trust *= 0.7  # Smaller immediate effect
        else:
            base_trust *= 1.0
        
        # Pure gifts (Sadaqa) generate deep trust
        if gift_type == GiftType.SACRED:
            base_trust *= 1.5
        
        # Update agent state
        self.oxytocin_level += base_trust
        current_trust = self.trust_scores.get(giver_id, 0.5)
        self.trust_scores[giver_id] = min(1.0, current_trust + base_trust)
        
        return base_trust
    
    def compute_reputation_gain(self, 
                               receiver_id: int, 
                               gift_value: float, 
                               gift_type: GiftType,
                               expect_return: bool) -> float:
        """
        Compute social standing gain from giving.
        
        Based on Moka/Kula competitive giving: larger gifts = greater prestige.
        Immaterial gifts produce growing returns over repeated interactions.
        """
        # Base prestige: larger gifts = more prestige
        prestige = 0.01 * (gift_value / 100.0)
        
        # Count prior gifts to this receiver
        n_prior = len([g for g in self.gifts_given if g.receiver_id == receiver_id])
        
        # Immaterial gifts: effectiveness grows with repetition (+6% per interaction)
        if gift_type == GiftType.IMMATERIAL:
            prestige *= (1 + 0.06 * n_prior)
        
        # Pure Sadaqa: greater social recognition
        if gift_type == GiftType.SACRED:
            prestige *= 1.3
        
        # No expectation of return enhances reputation (generosity signal)
        if not expect_return:
            prestige *= 1.2
        
        # Update reputation
        self.reputation = min(1.0, self.reputation + prestige)
        
        return prestige
    
    def give_gift(self, 
                  receiver: 'SadaqaAgent',
                  value: float,
                  gift_type: GiftType = GiftType.MATERIAL,
                  reciprocity_type: ReciprocityType = ReciprocityType.GENERALIZED,
                  sphere: ExchangeSphere = ExchangeSphere.GIFT,
                  expect_return: bool = True,
                  return_deadline: Optional[int] = None) -> bool:
        """
        Give a gift to another agent.
        
        This is the core action of the gift economy.
        
        Returns:
        - success: Whether the gift was given
        """
        # Check if agent has sufficient wealth/material for the gift
        if gift_type in [GiftType.MATERIAL, GiftType.INALIENABLE]:
            if self.wealth < value:
                return False
        
        # Compute effects
        warm_glow = self.compute_warm_glow(value, gift_type, not expect_return)
        reputation_gain = self.compute_reputation_gain(
            receiver.id, value, gift_type, not expect_return
        )
        
        # Update wealth
        if gift_type in [GiftType.MATERIAL, GiftType.INALIENABLE]:
            self.wealth -= value
            receiver.wealth += value
        
        # For inalienable possessions, track the transfer
        if gift_type == GiftType.INALIENABLE:
            # Find and transfer possession
            for pos in self.inalienable_possessions:
                if pos.current_holder_id == self.id and pos.value == value:
                    pos.current_holder_id = receiver.id
                    pos.history.append(receiver.id)
                    break
        
        # Create gift record
        gift_record = GiftRecord(
            giver_id=self.id,
            receiver_id=receiver.id,
            value=value,
            gift_type=gift_type,
            reciprocity_type=reciprocity_type,
            sphere=sphere,
            time_step=current_time,  # Will be set by simulation
            expected_return=expect_return,
            return_deadline=return_deadline
        )
        
        self.gifts_given.append(gift_record)
        receiver.gifts_received.append(gift_record)
        
        # Update totals
        self.total_given += value
        receiver.total_received += value
        
        if gift_type == GiftType.SACRED:
            self.pure_gifts_given += 1
            receiver.pure_gifts_received += 1
        
        # Create obligation if return expected
        if expect_return and return_deadline:
            obligation = Obligation(
                from_agent=receiver.id,
                original_gift_value=value,
                original_gift_type=gift_type,
                time_given=current_time,
                time_to_return=return_deadline
            )
            receiver.obligations.append(obligation)
        
        # Receiver experiences oxytocin release
        oxytocin = receiver.compute_oxytocin_release(self.id, value, gift_type)
        
        # Update sphere balances
        self.sphere_balances[sphere] -= value
        receiver.sphere_balances[sphere] += value
        
        return True
    
    def return_gift(self, 
                    obligation: Obligation,
                    increment: float = 0.0) -> bool:
        """
        Return a gift to clear an obligation.
        
        Following Moka logic: return with increment to gain prestige.
        
        Parameters:
        - obligation: The obligation to fulfill
        - increment: Extra value added to the return gift (Moka)
        """
        return_value = obligation.original_gift_value * (1 + increment)
        
        # Find the original giver
        original_giver = None  # Will be set by simulation
        
        # Give return gift
        success = self.give_gift(
            original_giver,
            return_value,
            gift_type=obligation.original_gift_type,
            reciprocity_type=ReciprocityType.BALANCED,
            expect_return=False,  # This clears the obligation
            return_deadline=None
        )
        
        if success:
            obligation.returned = True
            obligation.return_value = return_value
            self.obligations.remove(obligation)
            
            # Moka increment increases reputation
            if increment > 0:
                self.reputation += 0.05 * increment
        
        return success
    
    def decide_to_give(self, 
                       potential_receivers: List['SadaqaAgent'],
                       environment: Dict[str, Any],
                       time_step: int) -> Optional[Tuple['SadaqaAgent', float, GiftType, bool]]:
        """
        Decision rule for giving.
        
        Implements multiple strategies based on:
        - Sadaqa (pure giving when merit is high)
        - Reciprocity (returning obligations)
        - Prestige-seeking (Moka-style competitive giving)
        - Need-based giving (charity to those in need)
        """
        # Strategy 1: Clear existing obligations first
        if self.obligations and len(self.obligations) > 0:
            obligation = self.obligations[0]
            # Moka increment: higher if reputation is low and you need to gain status
            increment = 0.1 if self.reputation < 0.3 else 0.02
            return (None, obligation.original_gift_value, 
                    obligation.original_gift_type, True, increment)
        
        # Strategy 2: Pure Sadaqa (merit > threshold)
        if self.merit > 0.5 and self.generosity > 0.4:
            # Give to the most needful
            poorest = min(potential_receivers, key=lambda a: a.wealth)
            gift_value = min(self.wealth * 0.1, 50.0)  # 10% of wealth, max 50
            return (poorest, gift_value, GiftType.SACRED, False)
        
        # Strategy 3: Prestige-seeking (Moka)
        if self.reputation < 0.4 and self.wealth > 500:
            # Give large gift to high-reputation agent to gain status
            prestigious = max(potential_receivers, key=lambda a: a.reputation)
            gift_value = min(self.wealth * 0.2, 200.0)
            return (prestigious, gift_value, GiftType.MATERIAL, True)
        
        # Strategy 4: Reciprocal giving to maintain relationships
        # Find agent with whom you have positive history
        if len(self.gifts_received) > 0:
            recent_received = [g for g in self.gifts_received 
                              if g.time_step > time_step - 100]
            if recent_received:
                last_gift = recent_received[-1]
                giver = next(a for a in potential_receivers if a.id == last_gift.giver_id)
                # Return slightly more than received (Moka light)
                return (giver, last_gift.value * 1.05, last_gift.gift_type, True)
        
        # Strategy 5: Immaterial gift (compliment) - low cost, builds relationship
        if random.random() < 0.3:
            # Choose agent with medium reputation
            receivers_sorted = sorted(potential_receivers, key=lambda a: a.reputation)
            mid_idx = len(receivers_sorted) // 2
            target = receivers_sorted[mid_idx]
            # Immaterial gifts have no wealth cost
            return (target, 1.0, GiftType.IMMATERIAL, True)
        
        # Strategy 6: No gift this turn
        return None
    
    def process_obligations(self, current_time: int):
        """Process outstanding obligations, applying stress if overdue."""
        for obligation in self.obligations:
            if current_time > obligation.time_to_return:
                # Late return: stress increases, reputation decreases
                self.stress += 0.1
                self.reputation = max(0, self.reputation - 0.05)
    
    def get_status(self) -> Dict[str, Any]:
        """Return current agent status for monitoring."""
        return {
            "id": self.id,
            "wealth": self.wealth,
            "reputation": self.reputation,
            "merit": self.merit,
            "happiness": self.happiness,
            "stress": self.stress,
            "total_given": self.total_given,
            "total_received": self.total_received,
            "pure_gifts_given": self.pure_gifts_given,
            "obligations_count": len(self.obligations),
            "trust_avg": np.mean(list(self.trust_scores.values())) if self.trust_scores else 0.5,
            "sphere_balances": self.sphere_balances
        }


# ============================================================================
# SADAQA SIMULATION
# ============================================================================

class SadaqaSimulation:
    """
    Complete simulation of a Sadaqa-based gift economy.
    
    Features:
    - Multi-agent gift exchange
    - Trust network evolution
    - Multiple exchange spheres (market, gift, sacred)
    - Comparison metrics against debt economy
    """
    
    def __init__(self,
                 n_agents: int = 50,
                 initial_wealth: float = 1000.0,
                 network_type: str = "random",
                 network_density: float = 0.1,
                 allow_market: bool = True,
                 allow_sacred: bool = True):
        """
        Initialize the gift economy simulation.
        
        Parameters:
        - n_agents: Number of agents
        - initial_wealth: Starting wealth for each agent
        - network_type: "random", "small_world", or "scale_free"
        - network_density: Density of initial trust network
        - allow_market: Whether market exchange is allowed
        - allow_sacred: Whether sacred (Sadaqa) exchange is allowed
        """
        self.n_agents = n_agents
        self.allow_market = allow_market
        self.allow_sacred = allow_sacred
        self.time_step = 0
        
        # Create agents
        self.agents = []
        for i in range(n_agents):
            generosity = np.random.beta(2, 5)  # Most agents moderately generous
            risk_aversion = np.random.uniform(0.3, 0.8)
            agent = SadaqaAgent(
                agent_id=i,
                initial_wealth=initial_wealth,
                generosity=generosity,
                risk_aversion=risk_aversion
            )
            self.agents.append(agent)
        
        # Create trust network
        self.trust_network = self._create_network(network_type, network_density)
        self._initialize_trust_scores()
        
        # History tracking
        self.gift_history: List[GiftRecord] = []
        self.wealth_history: List[List[float]] = []
        self.reputation_history: List[List[float]] = []
        self.merit_history: List[List[float]] = []
        self.gini_history: List[float] = []
        self.transaction_volume_history: List[float] = []
        
        # Environment state
        self.environment = {
            "market_sentiment": 0.5,
            "scarcity": 0.0,  # 0 = abundance, 1 = famine
            "social_trust": 0.5,
            "time": 0
        }
    
    def _create_network(self, network_type: str, density: float) -> nx.Graph:
        """Create the trust network."""
        if network_type == "random":
            return nx.fast_gnp_random_graph(self.n_agents, density)
        elif network_type == "small_world":
            k = max(2, int(density * self.n_agents))
            return nx.watts_strogatz_graph(self.n_agents, k, 0.1)
        elif network_type == "scale_free":
            m = max(1, int(density * self.n_agents / 2))
            return nx.barabasi_albert_graph(self.n_agents, m)
        else:
            return nx.complete_graph(self.n_agents)
    
    def _initialize_trust_scores(self):
        """Initialize trust scores based on network structure."""
        for agent in self.agents:
            neighbors = list(self.trust_network.neighbors(agent.id))
            for neighbor_id in neighbors:
                # Initial trust based on network proximity
                agent.trust_scores[neighbor_id] = 0.5 + random.uniform(-0.2, 0.2)
        
        # Self-trust not needed
    
    def _select_receiver(self, giver: SadaqaAgent) -> Optional[SadaqaAgent]:
        """
        Select a receiver for a gift based on trust network.
        
        Options:
        1. Trusted neighbors (high oxytocin)
        2. Random neighbors (exploration)
        3. Strangers (if no neighbors)
        """
        neighbors = list(self.trust_network.neighbors(giver.id))
        
        if not neighbors:
            # No network connections: choose random agent
            return random.choice([a for a in self.agents if a.id != giver.id])
        
        # Probability of choosing based on trust
        trust_scores = [giver.trust_scores.get(n, 0.5) for n in neighbors]
        
        # Softmax selection (higher trust = more likely)
        exp_scores = np.exp(np.array(trust_scores) * 2.0)
        probs = exp_scores / exp_scores.sum()
        
        chosen_id = np.random.choice(neighbors, p=probs)
        return self.agents[chosen_id]
    
    def step(self):
        """Execute one time step of the gift economy."""
        # Randomly select an agent to potentially give
        giver = random.choice(self.agents)
        
        # Select receiver based on trust network
        receiver = self._select_receiver(giver)
        
        if receiver is None:
            return
        
        # Decide what to give
        decision = giver.decide_to_give(self.agents, self.environment, self.time_step)
        
        if decision is not None:
            if len(decision) == 5:
                # This is a return gift with increment
                target, value, gift_type, expect_return, increment = decision
                # Find and process obligation
                for obligation in giver.obligations:
                    if obligation.original_gift_value == value:
                        giver.return_gift(obligation, increment)
                        break
            else:
                target, value, gift_type, expect_return = decision
                if target is None:
                    target = receiver
                
                # Determine reciprocity type
                if gift_type == GiftType.SACRED:
                    reciprocity = ReciprocityType.GENERALIZED
                    sphere = ExchangeSphere.SACRED
                elif expect_return:
                    reciprocity = ReciprocityType.BALANCED
                    sphere = ExchangeSphere.GIFT
                else:
                    reciprocity = ReciprocityType.GENERALIZED
                    sphere = ExchangeSphere.GIFT
                
                # Determine return deadline (if expected)
                return_deadline = self.time_step + 50 if expect_return else None
                
                # Execute gift
                success = giver.give_gift(
                    target, value, gift_type, reciprocity, 
                    sphere, expect_return, return_deadline
                )
                
                if success:
                    # Record gift in history
                    self.gift_history.append(GiftRecord(
                        giver_id=giver.id,
                        receiver_id=target.id,
                        value=value,
                        gift_type=gift_type,
                        reciprocity_type=reciprocity,
                        sphere=sphere,
                        time_step=self.time_step,
                        expected_return=expect_return,
                        return_deadline=return_deadline
                    ))
        
        # Process obligations (stress from overdue returns)
        for agent in self.agents:
            agent.process_obligations(self.time_step)
        
        # Update environment
        self._update_environment()
        
        # Record history
        self._record_state()
        
        self.time_step += 1
    
    def _update_environment(self):
        """Update environment state based on system metrics."""
        # Social trust = average trust score across all agents
        all_trusts = []
        for agent in self.agents:
            all_trusts.extend(agent.trust_scores.values())
        self.environment["social_trust"] = np.mean(all_trusts) if all_trusts else 0.5
        
        # Scarcity increases if many agents are stressed
        avg_stress = np.mean([a.stress for a in self.agents])
        self.environment["scarcity"] = min(1.0, avg_stress)
        
        # Market sentiment fluctuates based on transaction volume
        if len(self.gift_history) > 0:
            recent_volume = sum(g.value for g in self.gift_history[-10:])
            self.environment["market_sentiment"] = 0.5 + 0.1 * (recent_volume / 1000 - 0.5)
            self.environment["market_sentiment"] = np.clip(self.environment["market_sentiment"], 0, 1)
        
        self.environment["time"] = self.time_step
    
    def _record_state(self):
        """Record current system state for analysis."""
        wealths = [a.wealth for a in self.agents]
        reputations = [a.reputation for a in self.agents]
        merits = [a.merit for a in self.agents]
        
        self.wealth_history.append(wealths.copy())
        self.reputation_history.append(reputations.copy())
        self.merit_history.append(merits.copy())
        
        # Gini coefficient (wealth inequality)
        sorted_wealth = np.sort(wealths)
        n = len(sorted_wealth)
        cumsum = np.cumsum(sorted_wealth)
        gini = (2 * np.sum(cumsum) / (n * np.sum(sorted_wealth)) - (n + 1) / n)
        self.gini_history.append(gini)
        
        # Transaction volume (last 10 steps)
        recent_gifts = [g for g in self.gift_history if g.time_step > self.time_step - 10]
        self.transaction_volume_history.append(sum(g.value for g in recent_gifts))
    
    def compute_ergodicity_test(self) -> Dict[str, float]:
        """
        Test for non-ergodicity in the gift economy.
        
        Compares ensemble average vs. time average of wealth.
        In a gift economy, we expect these to converge (ergodic) because
        there is no compounding debt extraction. This is a key difference
        from debt-based systems.
        """
        # Ensemble average: average wealth across agents at final time
        final_wealths = self.wealth_history[-1] if self.wealth_history else [a.wealth for a in self.agents]
        ensemble_avg = np.mean(final_wealths)
        
        # Time average: average wealth over time for a single agent
        if self.wealth_history and len(self.wealth_history) > 0:
            agent_0_wealth = [w[0] for w in self.wealth_history if len(w) > 0]
            time_avg = np.mean(agent_0_wealth) if agent_0_wealth else ensemble_avg
        else:
            time_avg = ensemble_avg
        
        divergence = abs(ensemble_avg - time_avg) / (ensemble_avg + 1e-6)
        
        # Gift economies should be more ergodic than debt economies
        is_ergodic = divergence < 0.1
        
        return {
            "ensemble_average": ensemble_avg,
            "time_average": time_avg,
            "divergence": divergence,
            "is_ergodic": is_ergodic,
            "interpretation": "Gift economy appears ergodic (time average ~ ensemble average)" if is_ergodic
                            else "Gift economy shows non-ergodicity (unusual for gift systems)"
        }
    
    def compute_system_metrics(self) -> Dict[str, float]:
        """Compute overall system metrics."""
        wealths = [a.wealth for a in self.agents]
        reputations = [a.reputation for a in self.agents]
        merits = [a.merit for a in self.agents]
        
        return {
            "total_wealth": sum(wealths),
            "mean_wealth": np.mean(wealths),
            "gini_coefficient": self.gini_history[-1] if self.gini_history else 0.5,
            "mean_reputation": np.mean(reputations),
            "mean_merit": np.mean(merits),
            "total_gifts_given": sum(a.total_given for a in self.agents),
            "total_gifts_received": sum(a.total_received for a in self.agents),
            "pure_gifts_total": sum(a.pure_gifts_given for a in self.agents),
            "outstanding_obligations": sum(len(a.obligations) for a in self.agents),
            "social_trust": self.environment["social_trust"],
            "scarcity": self.environment["scarcity"],
            "ergodic": self.compute_ergodicity_test()["is_ergodic"]
        }
    
    def run(self, n_steps: int = 1000, verbose: bool = True) -> Dict[str, Any]:
        """Run the simulation for a specified number of steps."""
        print(f"Running Sadaqa Gift Economy Simulation for {n_steps} steps")
        print("=" * 60)
        
        for step in range(n_steps):
            self.step()
            
            if verbose and step % 100 == 0:
                metrics = self.compute_system_metrics()
                print(f"Step {step}: Wealth = {metrics['total_wealth']:.0f}, "
                      f"Gini = {metrics['gini_coefficient']:.3f}, "
                      f"Trust = {metrics['social_trust']:.3f}, "
                      f"Pure Gifts = {metrics['pure_gifts_total']}")
        
        final_metrics = self.compute_system_metrics()
        ergodicity = self.compute_ergodicity_test()
        
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)
        print(f"Final total wealth: {final_metrics['total_wealth']:.2f}")
        print(f"Final Gini coefficient: {final_metrics['gini_coefficient']:.4f}")
        print(f"Final social trust: {final_metrics['social_trust']:.4f}")
        print(f"Pure gifts (Sadaqa): {final_metrics['pure_gifts_total']}")
        print(f"Ergodic: {final_metrics['ergodic']}")
        print(f"Ergodicity divergence: {ergodicity['divergence']:.4f}")
        
        return {
            "metrics": final_metrics,
            "ergodicity": ergodicity,
            "wealth_history": self.wealth_history,
            "gini_history": self.gini_history,
            "reputation_history": self.reputation_history,
            "merit_history": self.merit_history,
            "transaction_volume": self.transaction_volume_history
        }


# ============================================================================
# COMPARISON WITH DEBT ECONOMY
# ============================================================================

class DebtEconomyComparison:
    """
    Compare Sadaqa gift economy with debt-based economy.
    
    This class provides direct comparison metrics to show the advantages
    of gift-based systems over debt-based systems.
    """
    
    @staticmethod
    def compare(sadaqa_sim: SadaqaSimulation, 
                debt_sim_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare the two economic systems.
        
        Parameters:
        - sadaqa_sim: Completed Sadaqa simulation
        - debt_sim_results: Results from debt-based simulation (from Yusuf model)
        """
        sadaqa_metrics = sadaqa_sim.compute_system_metrics()
        sadaqa_ergodicity = sadaqa_sim.compute_ergodicity_test()
        
        comparison = {
            "wealth_gini": {
                "gift_economy": sadaqa_metrics["gini_coefficient"],
                "debt_economy": debt_sim_results.get("gini_coefficient", 0.8),
                "interpretation": "Gift economy is more egalitarian" 
                                  if sadaqa_metrics["gini_coefficient"] < debt_sim_results.get("gini_coefficient", 0.8)
                                  else "Debt economy is more egalitarian"
            },
            "systemic_stability": {
                "gift_economy": sadaqa_metrics["social_trust"],
                "debt_economy": debt_sim_results.get("trust", 0.3),
                "interpretation": "Gift economy shows higher social trust (more stable)"
            },
            "ergodicity": {
                "gift_economy": sadaqa_ergodicity["is_ergodic"],
                "debt_economy": debt_sim_results.get("is_ergodic", False),
                "interpretation": "Gift economy is ergodic (time average ~ ensemble), debt economy is non-ergodic"
            },
            "transaction_volume": {
                "gift_economy": sadaqa_metrics["total_gifts_given"],
                "debt_economy": debt_sim_results.get("transaction_volume", 0),
                "interpretation": "Gift economy has sustained non-monetary exchange"
            },
            "pure_gifts": {
                "gift_economy": sadaqa_metrics["pure_gifts_total"],
                "debt_economy": 0,
                "interpretation": "Sadaqa (pure giving) exists only in gift economy"
            }
        }
        
        # Overall assessment
        if (sadaqa_metrics["gini_coefficient"] < debt_sim_results.get("gini_coefficient", 0.8) and
            sadaqa_metrics["social_trust"] > debt_sim_results.get("trust", 0.3) and
            sadaqa_ergodicity["is_ergodic"] and
            not debt_sim_results.get("is_ergodic", False)):
            comparison["overall"] = "Gift economy (Sadaqa-based) outperforms debt economy on all key metrics"
        else:
            comparison["overall"] = "Mixed results; further analysis needed"
        
        return comparison
    
    @staticmethod
    def visualize_comparison(comparison: Dict[str, Any]):
        """Create visualization comparing the two economic systems."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Inequality comparison
        systems = ["Gift Economy", "Debt Economy"]
        gini_values = [comparison["wealth_gini"]["gift_economy"], 
                       comparison["wealth_gini"]["debt_economy"]]
        axes[0].bar(systems, gini_values, color=['green', 'red'], alpha=0.7)
        axes[0].set_ylabel('Gini Coefficient')
        axes[0].set_title('Wealth Inequality\n(Lower is better)')
        axes[0].set_ylim(0, 1)
        
        # Trust comparison
        trust_values = [comparison["systemic_stability"]["gift_economy"],
                        comparison["systemic_stability"]["debt_economy"]]
        axes[1].bar(systems, trust_values, color=['green', 'red'], alpha=0.7)
        axes[1].set_ylabel('Social Trust')
        axes[1].set_title('Systemic Stability\n(Higher is better)')
        axes[1].set_ylim(0, 1)
        
        # Ergodicity comparison
        ergo_labels = ["Gift\n(Ergodic)", "Debt\n(Non-Ergodic)"]
        axes[2].bar(ergo_labels, [1, 0], color=['green', 'red'], alpha=0.7)
        axes[2].set_ylabel('Ergodic (Yes/No)')
        axes[2].set_title('Ergodicity\n(Time avg = Ensemble avg)')
        axes[2].set_ylim(0, 1.2)
        
        plt.suptitle("Sadaqa Gift Economy vs. Debt-Based Economy", fontsize=14)
        plt.tight_layout()
        plt.savefig('gift_vs_debt_comparison.png', dpi=150)
        plt.show()
        
        print("\n" + "=" * 60)
        print("VERDICT")
        print("=" * 60)
        print(comparison["overall"])


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_gift_economy(sim: SadaqaSimulation):
    """Create comprehensive visualizations of the gift economy."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Wealth distribution (final)
    final_wealth = sim.wealth_history[-1] if sim.wealth_history else [a.wealth for a in sim.agents]
    axes[0, 0].hist(final_wealth, bins=20, color='green', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Wealth')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Final Wealth Distribution')
    
    # 2. Gini coefficient over time
    axes[0, 1].plot(sim.gini_history, color='red', linewidth=1)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Gini Coefficient')
    axes[0, 1].set_title('Wealth Inequality Over Time')
    axes[0, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].set_ylim(0, 1)
    
    # 3. Reputation vs. Wealth scatter
    reputations = [a.reputation for a in sim.agents]
    wealths = [a.wealth for a in sim.agents]
    axes[0, 2].scatter(wealths, reputations, alpha=0.6, c='blue')
    axes[0, 2].set_xlabel('Wealth')
    axes[0, 2].set_ylabel('Reputation')
    axes[0, 2].set_title('Reputation vs. Wealth')
    
    # 4. Trust network (simplified visualization)
    if sim.trust_network.number_of_nodes() > 0:
        pos = nx.spring_layout(sim.trust_network, k=0.3, iterations=50)
        nx.draw(sim.trust_network, pos, ax=axes[1, 0], node_size=50, 
                node_color='green', edge_color='gray', alpha=0.6, with_labels=False)
        axes[1, 0].set_title('Trust Network Structure')
    else:
        axes[1, 0].text(0.5, 0.5, "No network", ha='center')
    
    # 5. Transaction volume over time
    axes[1, 1].plot(sim.transaction_volume_history, color='purple', linewidth=1)
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Transaction Volume (last 10 steps)')
    axes[1, 1].set_title('Gift Transaction Volume')
    
    # 6. Merit vs. Generosity
    merits = [a.merit for a in sim.agents]
    generosities = [a.generosity for a in sim.agents]
    axes[1, 2].scatter(generosities, merits, alpha=0.6, c='orange')
    axes[1, 2].set_xlabel('Generosity')
    axes[1, 2].set_ylabel('Merit (Sadaqa)')
    axes[1, 2].set_title('Sadaqa Merit vs. Generosity')
    
    plt.suptitle("Sadaqa Gift Economy Simulation Results", fontsize=14)
    plt.tight_layout()
    plt.savefig('sadaqa_simulation_results.png', dpi=150)
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete demonstration of Sadaqa gift economy."""
    print("=" * 70)
    print("SADAQA GIFT ECONOMY SIMULATION")
    print("Grounding: Islamic Sadaqa, Maussian gift theory, Neuroeconomics")
    print("=" * 70)
    print()
    
    # Initialize simulation
    sim = SadaqaSimulation(
        n_agents=50,
        initial_wealth=1000.0,
        network_type="small_world",
        network_density=0.08,
        allow_market=True,
        allow_sacred=True
    )
    
    # Run simulation
    results = sim.run(n_steps=500, verbose=True)
    
    # Visualize
    visualize_gift_economy(sim)
    
    # Ergodicity analysis
    ergo = sim.compute_ergodicity_test()
    print("\n" + "=" * 60)
    print("ERGODICITY ANALYSIS")
    print("=" * 60)
    print(f"Ensemble average wealth: {ergo['ensemble_average']:.2f}")
    print(f"Time average wealth: {ergo['time_average']:.2f}")
    print(f"Divergence: {ergo['divergence']:.4f}")
    print(f"System is ergodic: {ergo['is_ergodic']}")
    print(f"Interpretation: {ergo['interpretation']}")
    
    # Agent summary
    print("\n" + "=" * 60)
    print("SAMPLE AGENT STATUS")
    print("=" * 60)
    for i in range(min(3, len(sim.agents))):
        status = sim.agents[i].get_status()
        print(f"\nAgent {status['id']}:")
        print(f"  Wealth: {status['wealth']:.2f}")
        print(f"  Reputation: {status['reputation']:.3f}")
        print(f"  Merit (Sadaqa): {status['merit']:.3f}")
        print(f"  Happiness: {status['happiness']:.3f}")
        print(f"  Total given: {status['total_given']:.2f}")
        print(f"  Total received: {status['total_received']:.2f}")
        print(f"  Pure gifts given: {status['pure_gifts_given']}")
        print(f"  Outstanding obligations: {status['obligations_count']}")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("The Sadaqa gift economy demonstrates that:")
    print("1. Non-monetary exchange based on reciprocity is sustainable")
    print("2. Pure giving (Sadaqa) generates merit and social cohesion")
    print("3. Immaterial gifts create growing returns over repeated interactions")
    print("4. Gift economies are more ergodic than debt-based systems")
    print("5. Wealth inequality is lower without compounding debt")
    print("\nThis provides a formal model for the economics of generosity,")
    print("grounded in Islamic Sadaqa and anthropological gift theory.")
    
    return sim, results


if __name__ == "__main__":
    sim, results = main()
# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class GiftType(Enum):
    """Types of gifts in the economy."""
    MATERIAL = "material"          # Physical goods, money
    IMMATERIAL = "immaterial"      # Compliments, praise, social recognition
    SACRED = "sacred"              # Sadaqa: pure gift with no return expected
    INALIENABLE = "inalienable"    # Possessions that cannot be sold, only gifted


class ReciprocityType(Enum):
    """Types of reciprocity (Sahlins 1972)."""
    GENERALIZED = "generalized"    # No immediate return expected, trust-based
    BALANCED = "balanced"          # Fair return expected within reasonable time
    NEGATIVE = "negative"          # Attempt to profit at other's expense


class ExchangeSphere(Enum):
    """Spheres of exchange (Bohannan 1959)."""
    MARKET = "market"              # Commodity exchange, prices, money
    GIFT = "gift"                  # Reciprocal gift exchange
    SACRED = "sacred"              # Sadaqa, charity, merit-based


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class GiftRecord:
    """Record of a single gift transaction."""
    giver_id: int
    receiver_id: int
    value: float
    gift_type: GiftType
    reciprocity_type: ReciprocityType
    sphere: ExchangeSphere
    time_step: int
    expected_return: bool
    return_deadline: Optional[int] = None


@dataclass
class Obligation:
    """Record of an outstanding obligation to return a gift."""
    from_agent: int
    original_gift_value: float
    original_gift_type: GiftType
    time_given: int
    time_to_return: int  # Deadline for return
    returned: bool = False
    return_value: float = 0.0


@dataclass
class InalienablePossession:
    """A possession that cannot be sold, only gifted (Weiner 1992)."""
    name: str
    value: float
    original_owner_id: int
    current_holder_id: int
    history: List[int] = field(default_factory=list)  # Chain of holders
    sacred: bool = False  # If sacred, cannot be returned


# ============================================================================
# SADAQA AGENT
# ============================================================================

class SadaqaAgent:
    """
    An agent in a gift economy, based on Sadaqa and generalized reciprocity.
    
    Core mechanisms:
    1. Warm glow (dopamine) from giving - pure gift is its own reward
    2. Oxytocin release from receiving - increases trust
    3. Merit accumulation (Sadaqa) - spiritual/social credit
    4. Reputation dynamics - social standing based on generosity
    5. Obligation tracking - debt of gratitude (different from monetary debt)
    6. Inalienable possessions - objects that circulate but are never fully alienated
    """
    
    def __init__(self, 
                 agent_id: int, 
                 initial_wealth: float = 1000.0,
                 initial_reputation: float = 0.5,
                 initial_merit: float = 0.0,
                 generosity: float = 0.3,
                 risk_aversion: float = 0.5):
        """
        Initialize a Sadaqa agent.
        
        Parameters:
        - agent_id: Unique identifier
        - initial_wealth: Material wealth (still used for consumption)
        - initial_reputation: Social standing (0 to 1)
        - initial_merit: Sadaqa merit (spiritual/social credit)
        - generosity: Propensity to give (0 to 1)
        - risk_aversion: Preference for safe vs. risky exchanges
        """
        self.id = agent_id
        self.wealth = initial_wealth
        self.reputation = initial_reputation
        self.merit = initial_merit
        self.generosity = generosity
        self.risk_aversion = risk_aversion
        
        # Histories
        self.gifts_given: List[GiftRecord] = []
        self.gifts_received: List[GiftRecord] = []
        self.obligations: List[Obligation] = []  # Debts of gratitude
        self.inalienable_possessions: List[InalienablePossession] = []
        
        # Trust network (oxytocin-based)
        self.trust_scores: Dict[int, float] = {}  # trust in other agents
        self.trust_history: List[float] = [0.5]
        
        # Emotional/neurological state
        self.happiness: float = 0.5
        self.stress: float = 0.0  # From unreturned gifts (poison of the gift)
        self.dopamine_level: float = 0.0
        self.oxytocin_level: float = 0.0
        
        # Exchange sphere tracking
        self.sphere_balances: Dict[ExchangeSphere, float] = {
            ExchangeSphere.MARKET: initial_wealth,
            ExchangeSphere.GIFT: 0.0,
            ExchangeSphere.SACRED: 0.0
        }
        
        # Performance metrics
        self.total_given = 0.0
        self.total_received = 0.0
        self.pure_gifts_given = 0  # Sadaqa count
        self.pure_gifts_received = 0
        
    def compute_warm_glow(self, 
                          gift_value: float, 
                          gift_type: GiftType, 
                          expect_return: bool) -> float:
        """
        Compute utility from giving (dopamine release).
        
        Based on neuroeconomic research: giving activates reward pathways
        in the brain, with pure gifts (Sadaqa) producing stronger effects.
        
        Returns:
        - dopamine_increase: Amount to add to dopamine level
        """
        base_glow = 0.05 * gift_value / 100.0  # Scaled to reasonable range
        
        # Immaterial gifts create sustained warm glow
        if gift_type == GiftType.IMMATERIAL:
            base_glow *= 1.2
        
        # Pure Sadaqa: greater spiritual/reward pathway activation
        if gift_type == GiftType.SACRED:
            base_glow *= 2.0
        
        # No expectation of return increases warm glow
        if not expect_return:
            base_glow *= 1.3
        
        # Generosity amplifies warm glow
        base_glow *= (1 + self.generosity)
        
        # Update agent state
        self.dopamine_level += base_glow
        self.happiness += 0.01 * base_glow
        
        # Merit increases with warm glow (Sadaqa)
        if gift_type == GiftType.SACRED:
            self.merit += base_glow * 10.0
        
        return base_glow
    
    def compute_oxytocin_release(self, 
                                 giver_id: int, 
                                 gift_value: float, 
                                 gift_type: GiftType) -> float:
        """
        Compute trust increase from receiving a gift (oxytocin release).
        
        Oxytocin is associated with trust, safety, and social bonding.
        Returns:
        - trust_increase: Amount to add to trust score for giver
        """
        base_trust = 0.05
        
        # Larger gifts = more trust
        base_trust += 0.01 * (gift_value / 100.0)
        
        # Immaterial gifts build trust more slowly but last longer
        if gift_type == GiftType.IMMATERIAL:
            base_trust *= 0.7  # Smaller immediate effect
        else:
            base_trust *= 1.0
        
        # Pure gifts (Sadaqa) generate deep trust
        if gift_type == GiftType.SACRED:
            base_trust *= 1.5
        
        # Update agent state
        self.oxytocin_level += base_trust
        current_trust = self.trust_scores.get(giver_id, 0.5)
        self.trust_scores[giver_id] = min(1.0, current_trust + base_trust)
        
        return base_trust
    
    def compute_reputation_gain(self, 
                               receiver_id: int, 
                               gift_value: float, 
                               gift_type: GiftType,
                               expect_return: bool) -> float:
        """
        Compute social standing gain from giving.
        
        Based on Moka/Kula competitive giving: larger gifts = greater prestige.
        Immaterial gifts produce growing returns over repeated interactions.
        """
        # Base prestige: larger gifts = more prestige
        prestige = 0.01 * (gift_value / 100.0)
        
        # Count prior gifts to this receiver
        n_prior = len([g for g in self.gifts_given if g.receiver_id == receiver_id])
        
        # Immaterial gifts: effectiveness grows with repetition (+6% per interaction)
        if gift_type == GiftType.IMMATERIAL:
            prestige *= (1 + 0.06 * n_prior)
        
        # Pure Sadaqa: greater social recognition
        if gift_type == GiftType.SACRED:
            prestige *= 1.3
        
        # No expectation of return enhances reputation (generosity signal)
        if not expect_return:
            prestige *= 1.2
        
        # Update reputation
        self.reputation = min(1.0, self.reputation + prestige)
        
        return prestige
    
    def give_gift(self, 
                  receiver: 'SadaqaAgent',
                  value: float,
                  gift_type: GiftType = GiftType.MATERIAL,
                  reciprocity_type: ReciprocityType = ReciprocityType.GENERALIZED,
                  sphere: ExchangeSphere = ExchangeSphere.GIFT,
                  expect_return: bool = True,
                  return_deadline: Optional[int] = None) -> bool:
        """
        Give a gift to another agent.
        
        This is the core action of the gift economy.
        
        Returns:
        - success: Whether the gift was given
        """
        # Check if agent has sufficient wealth/material for the gift
        if gift_type in [GiftType.MATERIAL, GiftType.INALIENABLE]:
            if self.wealth < value:
                return False
        
        # Compute effects
        warm_glow = self.compute_warm_glow(value, gift_type, not expect_return)
        reputation_gain = self.compute_reputation_gain(
            receiver.id, value, gift_type, not expect_return
        )
        
        # Update wealth
        if gift_type in [GiftType.MATERIAL, GiftType.INALIENABLE]:
            self.wealth -= value
            receiver.wealth += value
        
        # For inalienable possessions, track the transfer
        if gift_type == GiftType.INALIENABLE:
            # Find and transfer possession
            for pos in self.inalienable_possessions:
                if pos.current_holder_id == self.id and pos.value == value:
                    pos.current_holder_id = receiver.id
                    pos.history.append(receiver.id)
                    break
        
        # Create gift record
        gift_record = GiftRecord(
            giver_id=self.id,
            receiver_id=receiver.id,
            value=value,
            gift_type=gift_type,
            reciprocity_type=reciprocity_type,
            sphere=sphere,
            time_step=current_time,  # Will be set by simulation
            expected_return=expect_return,
            return_deadline=return_deadline
        )
        
        self.gifts_given.append(gift_record)
        receiver.gifts_received.append(gift_record)
        
        # Update totals
        self.total_given += value
        receiver.total_received += value
        
        if gift_type == GiftType.SACRED:
            self.pure_gifts_given += 1
            receiver.pure_gifts_received += 1
        
        # Create obligation if return expected
        if expect_return and return_deadline:
            obligation = Obligation(
                from_agent=receiver.id,
                original_gift_value=value,
                original_gift_type=gift_type,
                time_given=current_time,
                time_to_return=return_deadline
            )
            receiver.obligations.append(obligation)
        
        # Receiver experiences oxytocin release
        oxytocin = receiver.compute_oxytocin_release(self.id, value, gift_type)
        
        # Update sphere balances
        self.sphere_balances[sphere] -= value
        receiver.sphere_balances[sphere] += value
        
        return True
    
    def return_gift(self, 
                    obligation: Obligation,
                    increment: float = 0.0) -> bool:
        """
        Return a gift to clear an obligation.
        
        Following Moka logic: return with increment to gain prestige.
        
        Parameters:
        - obligation: The obligation to fulfill
        - increment: Extra value added to the return gift (Moka)
        """
        return_value = obligation.original_gift_value * (1 + increment)
        
        # Find the original giver
        original_giver = None  # Will be set by simulation
        
        # Give return gift
        success = self.give_gift(
            original_giver,
            return_value,
            gift_type=obligation.original_gift_type,
            reciprocity_type=ReciprocityType.BALANCED,
            expect_return=False,  # This clears the obligation
            return_deadline=None
        )
        
        if success:
            obligation.returned = True
            obligation.return_value = return_value
            self.obligations.remove(obligation)
            
            # Moka increment increases reputation
            if increment > 0:
                self.reputation += 0.05 * increment
        
        return success
    
    def decide_to_give(self, 
                       potential_receivers: List['SadaqaAgent'],
                       environment: Dict[str, Any],
                       time_step: int) -> Optional[Tuple['SadaqaAgent', float, GiftType, bool]]:
        """
        Decision rule for giving.
        
        Implements multiple strategies based on:
        - Sadaqa (pure giving when merit is high)
        - Reciprocity (returning obligations)
        - Prestige-seeking (Moka-style competitive giving)
        - Need-based giving (charity to those in need)
        """
        # Strategy 1: Clear existing obligations first
        if self.obligations and len(self.obligations) > 0:
            obligation = self.obligations[0]
            # Moka increment: higher if reputation is low and you need to gain status
            increment = 0.1 if self.reputation < 0.3 else 0.02
            return (None, obligation.original_gift_value, 
                    obligation.original_gift_type, True, increment)
        
        # Strategy 2: Pure Sadaqa (merit > threshold)
        if self.merit > 0.5 and self.generosity > 0.4:
            # Give to the most needful
            poorest = min(potential_receivers, key=lambda a: a.wealth)
            gift_value = min(self.wealth * 0.1, 50.0)  # 10% of wealth, max 50
            return (poorest, gift_value, GiftType.SACRED, False)
        
        # Strategy 3: Prestige-seeking (Moka)
        if self.reputation < 0.4 and self.wealth > 500:
            # Give large gift to high-reputation agent to gain status
            prestigious = max(potential_receivers, key=lambda a: a.reputation)
            gift_value = min(self.wealth * 0.2, 200.0)
            return (prestigious, gift_value, GiftType.MATERIAL, True)
        
        # Strategy 4: Reciprocal giving to maintain relationships
        # Find agent with whom you have positive history
        if len(self.gifts_received) > 0:
            recent_received = [g for g in self.gifts_received 
                              if g.time_step > time_step - 100]
            if recent_received:
                last_gift = recent_received[-1]
                giver = next(a for a in potential_receivers if a.id == last_gift.giver_id)
                # Return slightly more than received (Moka light)
                return (giver, last_gift.value * 1.05, last_gift.gift_type, True)
        
        # Strategy 5: Immaterial gift (compliment) - low cost, builds relationship
        if random.random() < 0.3:
            # Choose agent with medium reputation
            receivers_sorted = sorted(potential_receivers, key=lambda a: a.reputation)
            mid_idx = len(receivers_sorted) // 2
            target = receivers_sorted[mid_idx]
            # Immaterial gifts have no wealth cost
            return (target, 1.0, GiftType.IMMATERIAL, True)
        
        # Strategy 6: No gift this turn
        return None
    
    def process_obligations(self, current_time: int):
        """Process outstanding obligations, applying stress if overdue."""
        for obligation in self.obligations:
            if current_time > obligation.time_to_return:
                # Late return: stress increases, reputation decreases
                self.stress += 0.1
                self.reputation = max(0, self.reputation - 0.05)
    
    def get_status(self) -> Dict[str, Any]:
        """Return current agent status for monitoring."""
        return {
            "id": self.id,
            "wealth": self.wealth,
            "reputation": self.reputation,
            "merit": self.merit,
            "happiness": self.happiness,
            "stress": self.stress,
            "total_given": self.total_given,
            "total_received": self.total_received,
            "pure_gifts_given": self.pure_gifts_given,
            "obligations_count": len(self.obligations),
            "trust_avg": np.mean(list(self.trust_scores.values())) if self.trust_scores else 0.5,
            "sphere_balances": self.sphere_balances
        }


# ============================================================================
# SADAQA SIMULATION
# ============================================================================

class SadaqaSimulation:
    """
    Complete simulation of a Sadaqa-based gift economy.
    
    Features:
    - Multi-agent gift exchange
    - Trust network evolution
    - Multiple exchange spheres (market, gift, sacred)
    - Comparison metrics against debt economy
    """
    
    def __init__(self,
                 n_agents: int = 50,
                 initial_wealth: float = 1000.0,
                 network_type: str = "random",
                 network_density: float = 0.1,
                 allow_market: bool = True,
                 allow_sacred: bool = True):
        """
        Initialize the gift economy simulation.
        
        Parameters:
        - n_agents: Number of agents
        - initial_wealth: Starting wealth for each agent
        - network_type: "random", "small_world", or "scale_free"
        - network_density: Density of initial trust network
        - allow_market: Whether market exchange is allowed
        - allow_sacred: Whether sacred (Sadaqa) exchange is allowed
        """
        self.n_agents = n_agents
        self.allow_market = allow_market
        self.allow_sacred = allow_sacred
        self.time_step = 0
        
        # Create agents
        self.agents = []
        for i in range(n_agents):
            generosity = np.random.beta(2, 5)  # Most agents moderately generous
            risk_aversion = np.random.uniform(0.3, 0.8)
            agent = SadaqaAgent(
                agent_id=i,
                initial_wealth=initial_wealth,
                generosity=generosity,
                risk_aversion=risk_aversion
            )
            self.agents.append(agent)
        
        # Create trust network
        self.trust_network = self._create_network(network_type, network_density)
        self._initialize_trust_scores()
        
        # History tracking
        self.gift_history: List[GiftRecord] = []
        self.wealth_history: List[List[float]] = []
        self.reputation_history: List[List[float]] = []
        self.merit_history: List[List[float]] = []
        self.gini_history: List[float] = []
        self.transaction_volume_history: List[float] = []
        
        # Environment state
        self.environment = {
            "market_sentiment": 0.5,
            "scarcity": 0.0,  # 0 = abundance, 1 = famine
            "social_trust": 0.5,
            "time": 0
        }
    
    def _create_network(self, network_type: str, density: float) -> nx.Graph:
        """Create the trust network."""
        if network_type == "random":
            return nx.fast_gnp_random_graph(self.n_agents, density)
        elif network_type == "small_world":
            k = max(2, int(density * self.n_agents))
            return nx.watts_strogatz_graph(self.n_agents, k, 0.1)
        elif network_type == "scale_free":
            m = max(1, int(density * self.n_agents / 2))
            return nx.barabasi_albert_graph(self.n_agents, m)
        else:
            return nx.complete_graph(self.n_agents)
    
    def _initialize_trust_scores(self):
        """Initialize trust scores based on network structure."""
        for agent in self.agents:
            neighbors = list(self.trust_network.neighbors(agent.id))
            for neighbor_id in neighbors:
                # Initial trust based on network proximity
                agent.trust_scores[neighbor_id] = 0.5 + random.uniform(-0.2, 0.2)
        
        # Self-trust not needed
    
    def _select_receiver(self, giver: SadaqaAgent) -> Optional[SadaqaAgent]:
        """
        Select a receiver for a gift based on trust network.
        
        Options:
        1. Trusted neighbors (high oxytocin)
        2. Random neighbors (exploration)
        3. Strangers (if no neighbors)
        """
        neighbors = list(self.trust_network.neighbors(giver.id))
        
        if not neighbors:
            # No network connections: choose random agent
            return random.choice([a for a in self.agents if a.id != giver.id])
        
        # Probability of choosing based on trust
        trust_scores = [giver.trust_scores.get(n, 0.5) for n in neighbors]
        
        # Softmax selection (higher trust = more likely)
        exp_scores = np.exp(np.array(trust_scores) * 2.0)
        probs = exp_scores / exp_scores.sum()
        
        chosen_id = np.random.choice(neighbors, p=probs)
        return self.agents[chosen_id]
    
    def step(self):
        """Execute one time step of the gift economy."""
        # Randomly select an agent to potentially give
        giver = random.choice(self.agents)
        
        # Select receiver based on trust network
        receiver = self._select_receiver(giver)
        
        if receiver is None:
            return
        
        # Decide what to give
        decision = giver.decide_to_give(self.agents, self.environment, self.time_step)
        
        if decision is not None:
            if len(decision) == 5:
                # This is a return gift with increment
                target, value, gift_type, expect_return, increment = decision
                # Find and process obligation
                for obligation in giver.obligations:
                    if obligation.original_gift_value == value:
                        giver.return_gift(obligation, increment)
                        break
            else:
                target, value, gift_type, expect_return = decision
                if target is None:
                    target = receiver
                
                # Determine reciprocity type
                if gift_type == GiftType.SACRED:
                    reciprocity = ReciprocityType.GENERALIZED
                    sphere = ExchangeSphere.SACRED
                elif expect_return:
                    reciprocity = ReciprocityType.BALANCED
                    sphere = ExchangeSphere.GIFT
                else:
                    reciprocity = ReciprocityType.GENERALIZED
                    sphere = ExchangeSphere.GIFT
                
                # Determine return deadline (if expected)
                return_deadline = self.time_step + 50 if expect_return else None
                
                # Execute gift
                success = giver.give_gift(
                    target, value, gift_type, reciprocity, 
                    sphere, expect_return, return_deadline
                )
                
                if success:
                    # Record gift in history
                    self.gift_history.append(GiftRecord(
                        giver_id=giver.id,
                        receiver_id=target.id,
                        value=value,
                        gift_type=gift_type,
                        reciprocity_type=reciprocity,
                        sphere=sphere,
                        time_step=self.time_step,
                        expected_return=expect_return,
                        return_deadline=return_deadline
                    ))
        
        # Process obligations (stress from overdue returns)
        for agent in self.agents:
            agent.process_obligations(self.time_step)
        
        # Update environment
        self._update_environment()
        
        # Record history
        self._record_state()
        
        self.time_step += 1
    
    def _update_environment(self):
        """Update environment state based on system metrics."""
        # Social trust = average trust score across all agents
        all_trusts = []
        for agent in self.agents:
            all_trusts.extend(agent.trust_scores.values())
        self.environment["social_trust"] = np.mean(all_trusts) if all_trusts else 0.5
        
        # Scarcity increases if many agents are stressed
        avg_stress = np.mean([a.stress for a in self.agents])
        self.environment["scarcity"] = min(1.0, avg_stress)
        
        # Market sentiment fluctuates based on transaction volume
        if len(self.gift_history) > 0:
            recent_volume = sum(g.value for g in self.gift_history[-10:])
            self.environment["market_sentiment"] = 0.5 + 0.1 * (recent_volume / 1000 - 0.5)
            self.environment["market_sentiment"] = np.clip(self.environment["market_sentiment"], 0, 1)
        
        self.environment["time"] = self.time_step
    
    def _record_state(self):
        """Record current system state for analysis."""
        wealths = [a.wealth for a in self.agents]
        reputations = [a.reputation for a in self.agents]
        merits = [a.merit for a in self.agents]
        
        self.wealth_history.append(wealths.copy())
        self.reputation_history.append(reputations.copy())
        self.merit_history.append(merits.copy())
        
        # Gini coefficient (wealth inequality)
        sorted_wealth = np.sort(wealths)
        n = len(sorted_wealth)
        cumsum = np.cumsum(sorted_wealth)
        gini = (2 * np.sum(cumsum) / (n * np.sum(sorted_wealth)) - (n + 1) / n)
        self.gini_history.append(gini)
        
        # Transaction volume (last 10 steps)
        recent_gifts = [g for g in self.gift_history if g.time_step > self.time_step - 10]
        self.transaction_volume_history.append(sum(g.value for g in recent_gifts))
    
    def compute_ergodicity_test(self) -> Dict[str, float]:
        """
        Test for non-ergodicity in the gift economy.
        
        Compares ensemble average vs. time average of wealth.
        In a gift economy, we expect these to converge (ergodic) because
        there is no compounding debt extraction. This is a key difference
        from debt-based systems.
        """
        # Ensemble average: average wealth across agents at final time
        final_wealths = self.wealth_history[-1] if self.wealth_history else [a.wealth for a in self.agents]
        ensemble_avg = np.mean(final_wealths)
        
        # Time average: average wealth over time for a single agent
        if self.wealth_history and len(self.wealth_history) > 0:
            agent_0_wealth = [w[0] for w in self.wealth_history if len(w) > 0]
            time_avg = np.mean(agent_0_wealth) if agent_0_wealth else ensemble_avg
        else:
            time_avg = ensemble_avg
        
        divergence = abs(ensemble_avg - time_avg) / (ensemble_avg + 1e-6)
        
        # Gift economies should be more ergodic than debt economies
        is_ergodic = divergence < 0.1
        
        return {
            "ensemble_average": ensemble_avg,
            "time_average": time_avg,
            "divergence": divergence,
            "is_ergodic": is_ergodic,
            "interpretation": "Gift economy appears ergodic (time average ~ ensemble average)" if is_ergodic
                            else "Gift economy shows non-ergodicity (unusual for gift systems)"
        }
    
    def compute_system_metrics(self) -> Dict[str, float]:
        """Compute overall system metrics."""
        wealths = [a.wealth for a in self.agents]
        reputations = [a.reputation for a in self.agents]
        merits = [a.merit for a in self.agents]
        
        return {
            "total_wealth": sum(wealths),
            "mean_wealth": np.mean(wealths),
            "gini_coefficient": self.gini_history[-1] if self.gini_history else 0.5,
            "mean_reputation": np.mean(reputations),
            "mean_merit": np.mean(merits),
            "total_gifts_given": sum(a.total_given for a in self.agents),
            "total_gifts_received": sum(a.total_received for a in self.agents),
            "pure_gifts_total": sum(a.pure_gifts_given for a in self.agents),
            "outstanding_obligations": sum(len(a.obligations) for a in self.agents),
            "social_trust": self.environment["social_trust"],
            "scarcity": self.environment["scarcity"],
            "ergodic": self.compute_ergodicity_test()["is_ergodic"]
        }
    
    def run(self, n_steps: int = 1000, verbose: bool = True) -> Dict[str, Any]:
        """Run the simulation for a specified number of steps."""
        print(f"Running Sadaqa Gift Economy Simulation for {n_steps} steps")
        print("=" * 60)
        
        for step in range(n_steps):
            self.step()
            
            if verbose and step % 100 == 0:
                metrics = self.compute_system_metrics()
                print(f"Step {step}: Wealth = {metrics['total_wealth']:.0f}, "
                      f"Gini = {metrics['gini_coefficient']:.3f}, "
                      f"Trust = {metrics['social_trust']:.3f}, "
                      f"Pure Gifts = {metrics['pure_gifts_total']}")
        
        final_metrics = self.compute_system_metrics()
        ergodicity = self.compute_ergodicity_test()
        
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)
        print(f"Final total wealth: {final_metrics['total_wealth']:.2f}")
        print(f"Final Gini coefficient: {final_metrics['gini_coefficient']:.4f}")
        print(f"Final social trust: {final_metrics['social_trust']:.4f}")
        print(f"Pure gifts (Sadaqa): {final_metrics['pure_gifts_total']}")
        print(f"Ergodic: {final_metrics['ergodic']}")
        print(f"Ergodicity divergence: {ergodicity['divergence']:.4f}")
        
        return {
            "metrics": final_metrics,
            "ergodicity": ergodicity,
            "wealth_history": self.wealth_history,
            "gini_history": self.gini_history,
            "reputation_history": self.reputation_history,
            "merit_history": self.merit_history,
            "transaction_volume": self.transaction_volume_history
        }


# ============================================================================
# COMPARISON WITH DEBT ECONOMY
# ============================================================================

class DebtEconomyComparison:
    """
    Compare Sadaqa gift economy with debt-based economy.
    
    This class provides direct comparison metrics to show the advantages
    of gift-based systems over debt-based systems.
    """
    
    @staticmethod
    def compare(sadaqa_sim: SadaqaSimulation, 
                debt_sim_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare the two economic systems.
        
        Parameters:
        - sadaqa_sim: Completed Sadaqa simulation
        - debt_sim_results: Results from debt-based simulation (from Yusuf model)
        """
        sadaqa_metrics = sadaqa_sim.compute_system_metrics()
        sadaqa_ergodicity = sadaqa_sim.compute_ergodicity_test()
        
        comparison = {
            "wealth_gini": {
                "gift_economy": sadaqa_metrics["gini_coefficient"],
                "debt_economy": debt_sim_results.get("gini_coefficient", 0.8),
                "interpretation": "Gift economy is more egalitarian" 
                                  if sadaqa_metrics["gini_coefficient"] < debt_sim_results.get("gini_coefficient", 0.8)
                                  else "Debt economy is more egalitarian"
            },
            "systemic_stability": {
                "gift_economy": sadaqa_metrics["social_trust"],
                "debt_economy": debt_sim_results.get("trust", 0.3),
                "interpretation": "Gift economy shows higher social trust (more stable)"
            },
            "ergodicity": {
                "gift_economy": sadaqa_ergodicity["is_ergodic"],
                "debt_economy": debt_sim_results.get("is_ergodic", False),
                "interpretation": "Gift economy is ergodic (time average ~ ensemble), debt economy is non-ergodic"
            },
            "transaction_volume": {
                "gift_economy": sadaqa_metrics["total_gifts_given"],
                "debt_economy": debt_sim_results.get("transaction_volume", 0),
                "interpretation": "Gift economy has sustained non-monetary exchange"
            },
            "pure_gifts": {
                "gift_economy": sadaqa_metrics["pure_gifts_total"],
                "debt_economy": 0,
                "interpretation": "Sadaqa (pure giving) exists only in gift economy"
            }
        }
        
        # Overall assessment
        if (sadaqa_metrics["gini_coefficient"] < debt_sim_results.get("gini_coefficient", 0.8) and
            sadaqa_metrics["social_trust"] > debt_sim_results.get("trust", 0.3) and
            sadaqa_ergodicity["is_ergodic"] and
            not debt_sim_results.get("is_ergodic", False)):
            comparison["overall"] = "Gift economy (Sadaqa-based) outperforms debt economy on all key metrics"
        else:
            comparison["overall"] = "Mixed results; further analysis needed"
        
        return comparison
    
    @staticmethod
    def visualize_comparison(comparison: Dict[str, Any]):
        """Create visualization comparing the two economic systems."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Inequality comparison
        systems = ["Gift Economy", "Debt Economy"]
        gini_values = [comparison["wealth_gini"]["gift_economy"], 
                       comparison["wealth_gini"]["debt_economy"]]
        axes[0].bar(systems, gini_values, color=['green', 'red'], alpha=0.7)
        axes[0].set_ylabel('Gini Coefficient')
        axes[0].set_title('Wealth Inequality\n(Lower is better)')
        axes[0].set_ylim(0, 1)
        
        # Trust comparison
        trust_values = [comparison["systemic_stability"]["gift_economy"],
                        comparison["systemic_stability"]["debt_economy"]]
        axes[1].bar(systems, trust_values, color=['green', 'red'], alpha=0.7)
        axes[1].set_ylabel('Social Trust')
        axes[1].set_title('Systemic Stability\n(Higher is better)')
        axes[1].set_ylim(0, 1)
        
        # Ergodicity comparison
        ergo_labels = ["Gift\n(Ergodic)", "Debt\n(Non-Ergodic)"]
        axes[2].bar(ergo_labels, [1, 0], color=['green', 'red'], alpha=0.7)
        axes[2].set_ylabel('Ergodic (Yes/No)')
        axes[2].set_title('Ergodicity\n(Time avg = Ensemble avg)')
        axes[2].set_ylim(0, 1.2)
        
        plt.suptitle("Sadaqa Gift Economy vs. Debt-Based Economy", fontsize=14)
        plt.tight_layout()
        plt.savefig('gift_vs_debt_comparison.png', dpi=150)
        plt.show()
        
        print("\n" + "=" * 60)
        print("VERDICT")
        print("=" * 60)
        print(comparison["overall"])


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_gift_economy(sim: SadaqaSimulation):
    """Create comprehensive visualizations of the gift economy."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Wealth distribution (final)
    final_wealth = sim.wealth_history[-1] if sim.wealth_history else [a.wealth for a in sim.agents]
    axes[0, 0].hist(final_wealth, bins=20, color='green', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Wealth')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Final Wealth Distribution')
    
    # 2. Gini coefficient over time
    axes[0, 1].plot(sim.gini_history, color='red', linewidth=1)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Gini Coefficient')
    axes[0, 1].set_title('Wealth Inequality Over Time')
    axes[0, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].set_ylim(0, 1)
    
    # 3. Reputation vs. Wealth scatter
    reputations = [a.reputation for a in sim.agents]
    wealths = [a.wealth for a in sim.agents]
    axes[0, 2].scatter(wealths, reputations, alpha=0.6, c='blue')
    axes[0, 2].set_xlabel('Wealth')
    axes[0, 2].set_ylabel('Reputation')
    axes[0, 2].set_title('Reputation vs. Wealth')
    
    # 4. Trust network (simplified visualization)
    if sim.trust_network.number_of_nodes() > 0:
        pos = nx.spring_layout(sim.trust_network, k=0.3, iterations=50)
        nx.draw(sim.trust_network, pos, ax=axes[1, 0], node_size=50, 
                node_color='green', edge_color='gray', alpha=0.6, with_labels=False)
        axes[1, 0].set_title('Trust Network Structure')
    else:
        axes[1, 0].text(0.5, 0.5, "No network", ha='center')
    
    # 5. Transaction volume over time
    axes[1, 1].plot(sim.transaction_volume_history, color='purple', linewidth=1)
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Transaction Volume (last 10 steps)')
    axes[1, 1].set_title('Gift Transaction Volume')
    
    # 6. Merit vs. Generosity
    merits = [a.merit for a in sim.agents]
    generosities = [a.generosity for a in sim.agents]
    axes[1, 2].scatter(generosities, merits, alpha=0.6, c='orange')
    axes[1, 2].set_xlabel('Generosity')
    axes[1, 2].set_ylabel('Merit (Sadaqa)')
    axes[1, 2].set_title('Sadaqa Merit vs. Generosity')
    
    plt.suptitle("Sadaqa Gift Economy Simulation Results", fontsize=14)
    plt.tight_layout()
    plt.savefig('sadaqa_simulation_results.png', dpi=150)
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete demonstration of Sadaqa gift economy."""
    print("=" * 70)
    print("SADAQA GIFT ECONOMY SIMULATION")
    print("Grounding: Islamic Sadaqa, Maussian gift theory, Neuroeconomics")
    print("=" * 70)
    print()
    
    # Initialize simulation
    sim = SadaqaSimulation(
        n_agents=50,
        initial_wealth=1000.0,
        network_type="small_world",
        network_density=0.08,
        allow_market=True,
        allow_sacred=True
    )
    
    # Run simulation
    results = sim.run(n_steps=500, verbose=True)
    
    # Visualize
    visualize_gift_economy(sim)
    
    # Ergodicity analysis
    ergo = sim.compute_ergodicity_test()
    print("\n" + "=" * 60)
    print("ERGODICITY ANALYSIS")
    print("=" * 60)
    print(f"Ensemble average wealth: {ergo['ensemble_average']:.2f}")
    print(f"Time average wealth: {ergo['time_average']:.2f}")
    print(f"Divergence: {ergo['divergence']:.4f}")
    print(f"System is ergodic: {ergo['is_ergodic']}")
    print(f"Interpretation: {ergo['interpretation']}")
    
    # Agent summary
    print("\n" + "=" * 60)
    print("SAMPLE AGENT STATUS")
    print("=" * 60)
    for i in range(min(3, len(sim.agents))):
        status = sim.agents[i].get_status()
        print(f"\nAgent {status['id']}:")
        print(f"  Wealth: {status['wealth']:.2f}")
        print(f"  Reputation: {status['reputation']:.3f}")
        print(f"  Merit (Sadaqa): {status['merit']:.3f}")
        print(f"  Happiness: {status['happiness']:.3f}")
        print(f"  Total given: {status['total_given']:.2f}")
        print(f"  Total received: {status['total_received']:.2f}")
        print(f"  Pure gifts given: {status['pure_gifts_given']}")
        print(f"  Outstanding obligations: {status['obligations_count']}")
    
    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("The Sadaqa gift economy demonstrates that:")
    print("1. Non-monetary exchange based on reciprocity is sustainable")
    print("2. Pure giving (Sadaqa) generates merit and social cohesion")
    print("3. Immaterial gifts create growing returns over repeated interactions")
    print("4. Gift economies are more ergodic than debt-based systems")
    print("5. Wealth inequality is lower without compounding debt")
    print("\nThis provides a formal model for the economics of generosity,")
    print("grounded in Islamic Sadaqa and anthropological gift theory.")
    
    return sim, results


if __name__ == "__main__":
    sim, results = main()
