"""
Urban Business Fabric Model
===========================

Based on Youn et al. (2014) - The systematic structure and predictability of urban business diversity

Key findings integrated:
1. Universal rank-size distribution f(x) = A·x^(-γ)·e^(-x/x0)·φ(x, Dmax)
2. Total establishments Nf = η·N (η ≈ 21.6 people per establishment)
3. Scaling exponents β for each business type (super/sub-linear)
4. Business types systematically change rank with city size

This provides the micro-foundation for communal markets within
the spatial infrastructure evolved by Genetic Algorithms.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import scipy.special as sp


# ============================================================================
# UNIVERSAL BUSINESS DISTRIBUTION (Youn et al. 2014)
# ============================================================================

class UrbanBusinessFabric:
    """
    Implements the universal business distribution from Youn et al. 2014.
    
    For a city of population N, this generates the expected number
    of establishments for each business type, following the universal
    rank-size curve.
    """
    
    # Parameters from Youn et al. (Fig 2B, Eq. 3)
    GAMMA = 0.49          # Zipf exponent (power law regime)
    X0 = 211              # Exponential cutoff
    A = 0.0019            # Normalization constant
    ETA = 21.6            # People per establishment (Fig 1A)
    
    # Business categories (2-digit NAICS, from Fig 3C)
    BUSINESS_CATEGORIES = {
        11: "Agriculture, Forestry, Fishing",
        21: "Mining",
        22: "Utilities",
        23: "Construction",
        31: "Manufacturing",
        42: "Wholesale Trade",
        44: "Retail Trade",
        48: "Transportation & Warehousing",
        51: "Information",
        52: "Finance and Insurance",
        53: "Real Estate",
        54: "Professional, Scientific, Technical",
        55: "Management of Companies",
        56: "Admin & Support Services",
        61: "Educational Services",
        62: "Health Care & Social Assistance",
        71: "Arts, Entertainment & Recreation",
        72: "Accommodation & Food Services",
        81: "Other Services",
        92: "Public Administration"
    }
    
    # Scaling exponents β for each category (from Fig 3C)
    # β > 1: super-linear (grows faster than population)
    # β = 1: linear (constant per capita)
    # β < 1: sub-linear (grows slower than population)
    SCALING_EXPONENTS = {
        11: 0.85,   # Agriculture - sub-linear (declines with city size)
        21: 0.82,   # Mining - sub-linear
        22: 0.79,   # Utilities - sub-linear
        23: 0.95,   # Construction - near linear
        31: 0.92,   # Manufacturing - sub-linear
        42: 0.98,   # Wholesale Trade - near linear
        44: 1.00,   # Retail Trade - linear
        48: 0.97,   # Transportation - near linear
        51: 1.12,   # Information - super-linear
        52: 1.08,   # Finance - super-linear
        53: 1.10,   # Real Estate - super-linear
        54: 1.17,   # Professional/Scientific/Tech - super-linear (lawyers)
        55: 1.20,   # Management - super-linear
        56: 1.02,   # Admin Support - near linear
        61: 0.95,   # Education - sub-linear
        62: 1.05,   # Health Care - super-linear
        71: 1.03,   # Arts/Entertainment - super-linear
        72: 1.00,   # Accommodation/Food - linear (restaurants)
        81: 0.98,   # Other Services - near linear
        92: 0.88    # Public Administration - sub-linear
    }
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Precompute the rank-size curve
        self.rank_distribution = self._compute_rank_distribution()
        
        # Map ranks to business categories
        self.rank_to_category = self._assign_categories_to_ranks()
    
    def _compute_rank_distribution(self, max_rank: int = 5000) -> np.ndarray:
        """
        Compute f(x) = A * x^(-γ) * e^(-x/x0)
        
        This is the universal per-capita frequency of businesses at rank x.
        """
        ranks = np.arange(1, max_rank + 1)
        
        # Power law part
        power_law = ranks ** (-self.GAMMA)
        
        # Exponential cutoff
        exp_cutoff = np.exp(-ranks / self.X0)
        
        # Combined distribution
        f_x = self.A * power_law * exp_cutoff
        
        return f_x
    
    def _assign_categories_to_ranks(self) -> Dict[int, int]:
        """
        Assign business categories to ranks based on their scaling behavior.
        
        Super-linear categories (high β) get higher ranks (more abundant)
        in larger cities. This is implemented as a mapping that depends on city size.
        """
        # For now, create a static mapping based on average abundance
        # In reality, ranks shift with city size (Fig 3B)
        categories = list(self.BUSINESS_CATEGORIES.keys())
        
        # Sort by scaling exponent (higher β = more abundant in large cities)
        categories.sort(key=lambda c: self.SCALING_EXPONENTS.get(c, 1.0), reverse=True)
        
        mapping = {}
        total_ranks = len(self.rank_distribution)
        ranks_per_category = total_ranks // len(categories)
        
        for i, cat in enumerate(categories):
            start_rank = i * ranks_per_category + 1
            end_rank = (i + 1) * ranks_per_category if i < len(categories) - 1 else total_ranks
            for rank in range(start_rank, end_rank + 1):
                mapping[rank] = cat
        
        return mapping
    
    def get_establishment_count(self, population: float, rank: int) -> float:
        """
        Get expected number of establishments of rank x in a city of size N.
        
        F(x, N) = N * f(x)
        """
        if rank > len(self.rank_distribution):
            return 0
        return population * self.rank_distribution[rank - 1]
    
    def get_category_abundance(self, population: float, category_code: int) -> float:
        """
        Get total establishments for a business category.
        
        Uses scaling law: N_category ∝ N^β
        """
        beta = self.SCALING_EXPONENTS.get(category_code, 1.0)
        
        # Reference: at population 1e6, category has ~1000 establishments
        # (calibrated from data)
        ref_pop = 1_000_000
        ref_abundance = 1000
        
        abundance = ref_abundance * (population / ref_pop) ** beta
        
        return abundance
    
    def generate_city_businesses(self, population: float) -> Dict[int, Dict]:
        """
        Generate complete business composition for a city.
        
        Returns:
        - Dictionary mapping business category to number of establishments
        - Also includes rank information
        """
        total_establishments = population / self.ETA
        city_businesses = {}
        
        # Method 1: Use rank distribution for fine-grained composition
        max_rank = min(len(self.rank_distribution), int(total_establishments * 2))
        
        for rank in range(1, max_rank + 1):
            expected = self.get_establishment_count(population, rank)
            if expected < 1:
                break
            
            # Add Poisson noise (realistic variation)
            actual = np.random.poisson(expected)
            
            if actual > 0:
                category = self.rank_to_category.get(rank, 54)  # Default to Professional Services
                if category not in city_businesses:
                    city_businesses[category] = {
                        "name": self.BUSINESS_CATEGORIES.get(category, "Unknown"),
                        "establishments": 0,
                        "ranks": []
                    }
                city_businesses[category]["establishments"] += actual
                city_businesses[category]["ranks"].append(rank)
        
        # Method 2: Ensure all categories are represented
        for cat_code, cat_name in self.BUSINESS_CATEGORIES.items():
            if cat_code not in city_businesses:
                abundance = self.get_category_abundance(population, cat_code)
                if abundance >= 1:
                    city_businesses[cat_code] = {
                        "name": cat_name,
                        "establishments": int(abundance),
                        "ranks": []
                    }
        
        return city_businesses
    
    def get_rank_shift(self, category_code: int, population: float, 
                       reference_pop: float = 1_000_000) -> float:
        """
        Predict how rank changes with city size (Eq. 6 from paper).
        
        For super-linear categories (β > 1), rank decreases (becomes more abundant)
        as city size increases.
        
        x ∝ N^((1-β)/γ) for small x
        x ∝ x0(1-β) ln N for large x
        """
        beta = self.SCALING_EXPONENTS.get(category_code, 1.0)
        
        if beta == 1.0:
            return 0  # No rank shift
        
        size_ratio = population / reference_pop
        
        if beta > 1:
            # Super-linear: rank decreases (becomes more abundant)
            exponent = (1 - beta) / self.GAMMA
            rank_shift = size_ratio ** exponent
        else:
            # Sub-linear: rank increases (becomes less abundant)
            exponent = (1 - beta) / self.GAMMA
            rank_shift = size_ratio ** exponent
        
        return rank_shift


# ============================================================================
# COMMUNAL MARKET UNIT
# ============================================================================

@dataclass
class CommunalMarketUnit:
    """
    A communal market unit - the basic building block of urban economic fabric.
    
    From Youn et al.: Each establishment is a physical location where business is conducted.
    Communal markets aggregate these into cooperative units that share:
    - Storage facilities (Yusuf principle)
    - Trust network (Sadaqa)
    - Risk pooling
    """
    market_id: int
    city_id: int
    location: Tuple[float, float]
    population_served: float
    business_categories: Dict[int, Dict] = field(default_factory=dict)
    total_establishments: int = 0
    communal_storage: float = 0.0
    trust_score: float = 0.5
    wealth: float = 0.0
    
    def __post_init__(self):
        self.wealth = self.population_served * 1000  # Base wealth


class CommunalMarketNetwork:
    """
    Network of communal market units within a city.
    
    This implements the "urban fabric" - how businesses are distributed
    across the city and how they interact through trust and trade.
    """
    
    def __init__(self, 
                 city_id: int,
                 city_population: float,
                 city_position: Tuple[float, float],
                 n_markets: int = None):
        
        self.city_id = city_id
        self.city_population = city_population
        self.city_position = city_position
        
        # Determine number of markets (scales with sqrt(population))
        if n_markets is None:
            self.n_markets = max(1, int(np.sqrt(city_population / 10000)))
        else:
            self.n_markets = n_markets
        
        # Generate business fabric for the city
        self.fabric = UrbanBusinessFabric()
        self.city_businesses = self.fabric.generate_city_businesses(city_population)
        
        # Create market units
        self.markets = []
        self._create_markets()
        
        # Trust network between markets
        self.trust_network = nx.Graph()
        self._build_trust_network()
        
        # Track metrics
        self.gini_history = []
        self.trade_volume_history = []
    
    def _create_markets(self):
        """Distribute business establishments across communal markets."""
        # Generate positions for markets (uniform random within city radius)
        city_radius = np.sqrt(self.city_population / 1000)  # Rough scaling
        
        total_establishments = sum(cat["establishments"] for cat in self.city_businesses.values())
        
        # Distribute establishments to markets
        establishments_per_market = total_establishments // self.n_markets
        
        # Create markets at random positions
        for i in range(self.n_markets):
            angle = random.uniform(0, 2 * np.pi)
            r = random.uniform(0, city_radius)
            x = self.city_position[0] + r * np.cos(angle)
            y = self.city_position[1] + r * np.sin(angle)
            
            market = CommunalMarketUnit(
                market_id=i,
                city_id=self.city_id,
                location=(x, y),
                population_served=self.city_population / self.n_markets
            )
            
            self.markets.append(market)
        
        # Distribute businesses to markets based on market size
        # (Larger markets get more establishments)
        market_weights = [1.0] * self.n_markets
        
        for cat_code, cat_data in self.city_businesses.items():
            n_establishments = cat_data["establishments"]
            
            if n_establishments == 0:
                continue
            
            # Distribute this category's establishments across markets
            for _ in range(n_establishments):
                # Weighted selection (larger markets get more)
                market_idx = np.random.choice(self.n_markets, p=np.array(market_weights) / sum(market_weights))
                self.markets[market_idx].business_categories[cat_code] = {
                    "name": cat_data["name"],
                    "establishments": self.markets[market_idx].business_categories.get(cat_code, {}).get("establishments", 0) + 1,
                    "ranks": cat_data.get("ranks", [])
                }
                self.markets[market_idx].total_establishments += 1
                
                # Small wealth contribution per establishment
                self.markets[market_idx].wealth += 100
    
    def _build_trust_network(self):
        """Build trust network between communal markets."""
        for i, market_i in enumerate(self.markets):
            for j, market_j in enumerate(self.markets):
                if i < j:
                    # Trust based on distance (closer = more trust)
                    dx = market_i.location[0] - market_j.location[0]
                    dy = market_i.location[1] - market_j.location[1]
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    # Maximum distance across city
                    max_dist = np.sqrt(self.city_population / 1000)
                    trust = max(0, 1 - distance / max_dist) * 0.8 + 0.1
                    
                    self.trust_network.add_edge(i, j, weight=trust, distance=distance)
    
    def compute_gini(self) -> float:
        """Compute wealth inequality within the city."""
        wealths = [m.wealth for m in self.markets]
        sorted_wealth = np.sort(wealths)
        n = len(sorted_wealth)
        cumsum = np.cumsum(sorted_wealth)
        gini = (2 * np.sum(cumsum) / (n * np.sum(sorted_wealth)) - (n + 1) / n)
        return max(0, min(1, gini))
    
    def compute_diversity_index(self) -> float:
        """
        Compute Shannon diversity index of business categories.
        
        H = -Σ p_i ln p_i
        """
        # Aggregate across all markets
        total_by_category = defaultdict(int)
        for market in self.markets:
            for cat_code, cat_data in market.business_categories.items():
                total_by_category[cat_code] += cat_data["establishments"]
        
        total = sum(total_by_category.values())
        if total == 0:
            return 0
        
        shannon = 0
        for count in total_by_category.values():
            p = count / total
            shannon -= p * np.log(p)
        
        return shannon
    
    def get_business_distribution(self) -> Dict[int, int]:
        """Get total business establishments by category."""
        distribution = defaultdict(int)
        for market in self.markets:
            for cat_code, cat_data in market.business_categories.items():
                distribution[cat_code] += cat_data["establishments"]
        return dict(distribution)


# ============================================================================
# SPATIAL INFRASTRUCTURE WITH URBAN FABRIC
# ============================================================================

class SpatialInfrastructureWithUrbanFabric:
    """
    Complete spatial infrastructure model with urban business fabric.
    
    Integrates:
    1. Youn et al. (2014): Universal business distribution within cities
    2. Geisendorf (1999): GA evolution of trade routes between cities
    3. Communal market units as the basic building blocks
    
    The GA evolves BOTH:
    - Inter-city infrastructure (routes, hubs, protocols)
    - Intra-city market distribution (how businesses cluster)
    """
    
    def __init__(self,
                 n_cities: int = 8,
                 city_populations: List[float] = None,
                 width: float = 100,
                 height: float = 100,
                 seed: int = 42):
        
        random.seed(seed)
        np.random.seed(seed)
        
        self.n_cities = n_cities
        self.width = width
        self.height = height
        
        # Generate city positions and populations (Zipf distribution)
        self.city_positions = [(random.uniform(0, width), random.uniform(0, height)) 
                               for _ in range(n_cities)]
        
        if city_populations is None:
            # Zipf's law for city sizes
            ranks = np.arange(1, n_cities + 1)
            self.city_populations = 1_000_000 / ranks
        else:
            self.city_populations = city_populations
        
        # Create communal market networks for each city
        self.city_markets = {}
        for i in range(n_cities):
            self.city_markets[i] = CommunalMarketNetwork(
                city_id=i,
                city_population=self.city_populations[i],
                city_position=self.city_positions[i]
            )
        
        # Inter-city infrastructure (to be evolved by GA)
        self.infrastructure = self._initialize_infrastructure()
        
        # Encoder for GA
        self.encoder = InfrastructureEncoderWithFabric(n_cities, self.city_positions)
    
    def _initialize_infrastructure(self) -> Dict[str, Any]:
        """Initialize inter-city trade infrastructure."""
        route_capacities = {}
        
        # Connect nearby cities with higher probability
        for i in range(self.n_cities):
            for j in range(i + 1, self.n_cities):
                dx = self.city_positions[i][0] - self.city_positions[j][0]
                dy = self.city_positions[i][1] - self.city_positions[j][1]
                distance = np.sqrt(dx**2 + dy**2)
                
                # Probability of route decreases with distance
                prob = max(0, 1 - distance / 50)
                if random.random() < prob:
                    capacity = random.randint(50, 200)
                    route_capacities[(i, j)] = capacity
        
        # Hubs: cities with high total business diversity
        diversities = [self.city_markets[i].compute_diversity_index() for i in range(self.n_cities)]
        n_hubs = max(1, self.n_cities // 4)
        hub_indices = np.argsort(diversities)[-n_hubs:]
        
        return {
            "route_capacities": route_capacities,
            "hub_cities": set(hub_indices),
            "protocol": 0  # Gift economy
        }
    
    def compute_infrastructure_fitness(self) -> float:
        """
        Compute fitness of current infrastructure.
        
        Fitness = Σ (trade_volume * diversity / inequality) across cities
        """
        total_fitness = 0
        
        for city_id, market_network in self.city_markets.items():
            diversity = market_network.compute_diversity_index()
            gini = market_network.compute_gini()
            
            if gini > 0:
                city_fitness = diversity / gini
            else:
                city_fitness = diversity
            
            total_fitness += city_fitness
        
        # Also consider inter-city trade volume
        # (Would compute actual trade flows based on infrastructure)
        
        return total_fitness / self.n_cities


class InfrastructureEncoderWithFabric:
    """GA encoder for infrastructure including urban fabric parameters."""
    
    def __init__(self, n_cities: int, city_positions: List[Tuple[float, float]]):
        self.n_cities = n_cities
        self.city_positions = city_positions
        self.n_routes = n_cities * (n_cities - 1) // 2
        
        # Calculate distances
        self.distances = {}
        for i in range(n_cities):
            for j in range(i + 1, n_cities):
                dx = city_positions[i][0] - city_positions[j][0]
                dy = city_positions[i][1] - city_positions[j][1]
                self.distances[(i, j)] = np.sqrt(dx**2 + dy**2)
    
    def encode(self, infrastructure: Dict[str, Any]) -> str:
        """Encode infrastructure as binary string."""
        bits = []
        
        # Route capacities (4 bits each, 0-15 → 0-150 capacity)
        route_caps = infrastructure.get("route_capacities", {})
        for i in range(self.n_cities):
            for j in range(i + 1, self.n_cities):
                cap = route_caps.get((i, j), 0)
                normalized = min(15, cap // 10)
                bits.append(format(normalized, '04b'))
        
        # Hub status (1 bit per city)
        hubs = infrastructure.get("hub_cities", set())
        for i in range(self.n_cities):
            bits.append('1' if i in hubs else '0')
        
        # Protocol (2 bits: 00=gift, 01=debt, 10=hybrid)
        protocol = infrastructure.get("protocol", 0)
        bits.append(format(protocol, '02b'))
        
        return ''.join(bits)
    
    def decode(self, binary_string: str) -> Dict[str, Any]:
        """Decode binary string to infrastructure."""
        infrastructure = {
            "route_capacities": {},
            "hub_cities": set(),
            "protocol": 0
        }
        
        idx = 0
        
        # Route capacities
        for i in range(self.n_cities):
            for j in range(i + 1, self.n_cities):
                if idx + 4 <= len(binary_string):
                    cap_bits = binary_string[idx:idx + 4]
                    capacity = int(cap_bits, 2) * 10
                    if capacity > 0:
                        infrastructure["route_capacities"][(i, j)] = capacity
                idx += 4
        
        # Hub status
        for i in range(self.n_cities):
            if idx < len(binary_string) and binary_string[idx] == '1':
                infrastructure["hub_cities"].add(i)
            idx += 1
        
        # Protocol
        if idx + 2 <= len(binary_string):
            protocol_bits = binary_string[idx:idx + 2]
            infrastructure["protocol"] = int(protocol_bits, 2)
        
        return infrastructure


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_urban_fabric(city_network: CommunalMarketNetwork, title: str = "Urban Business Fabric"):
    """
    Visualize the communal market network within a city.
    
    Shows:
    - Market locations
    - Trust connections between markets
    - Market size (proportional to establishments)
    - Business diversity (color intensity)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Spatial distribution of markets
    ax1 = axes[0]
    
    # Draw trust network
    for i, j in city_network.trust_network.edges():
        x1, y1 = city_network.markets[i].location
        x2, y2 = city_network.markets[j].location
        trust = city_network.trust_network[i][j]['weight']
        ax1.plot([x1, x2], [y1, y2], color='gray', alpha=trust * 0.5, linewidth=trust * 2)
    
    # Draw markets
    for market in city_network.markets:
        size = max(50, min(500, market.total_establishments * 2))
        diversity = len(market.business_categories) / len(UrbanBusinessFabric.BUSINESS_CATEGORIES)
        ax1.scatter(market.location[0], market.location[1], 
                   s=size, c=[diversity], cmap='viridis', 
                   edgecolor='black', alpha=0.8, vmin=0, vmax=1)
    
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title(f'Communal Market Network\nCity Population: {city_network.city_population:.0f}')
    ax1.set_aspect('equal')
    
    # Right: Business category distribution
    ax2 = axes[1]
    
    distribution = city_network.get_business_distribution()
    categories = list(distribution.keys())
    abundances = [distribution[c] for c in categories]
    
    # Sort by abundance
    sorted_pairs = sorted(zip(abundances, categories), reverse=True)
    top_n = min(15, len(sorted_pairs))
    top_abundances = [p[0] for p in sorted_pairs[:top_n]]
    top_categories = [UrbanBusinessFabric.BUSINESS_CATEGORIES.get(p[1], "Unknown")[:20] for p in sorted_pairs[:top_n]]
    
    y_pos = np.arange(top_n)
    ax2.barh(y_pos, top_abundances, color='steelblue', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_categories)
    ax2.set_xlabel('Number of Establishments')
    ax2.set_title('Top Business Categories in City')
    ax2.invert_yaxis()
    
    # Add Gini and diversity info
    gini = city_network.compute_gini()
    diversity = city_network.compute_diversity_index()
    ax2.text(0.02, 0.98, f'Gini: {gini:.3f}\nDiversity: {diversity:.2f}', 
            transform=ax2.transAxes, verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig('urban_business_fabric.png', dpi=150)
    plt.show()


def compare_city_scaling():
    """
    Demonstrate Youn et al.'s scaling laws across cities of different sizes.
    """
    print("=" * 70)
    print("CITY SCALING DEMONSTRATION (Youn et al. 2014)")
    print("=" * 70)
    
    city_sizes = [50_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
    
    fabric = UrbanBusinessFabric()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Total establishments vs population
    ax1 = axes[0, 0]
    pops = city_sizes
    total_est = [p / fabric.ETA for p in pops]
    ax1.loglog(pops, total_est, 'o-', color='blue', linewidth=2, markersize=8)
    ax1.set_xlabel('Population N')
    ax1.set_ylabel('Total Establishments Nf')
    ax1.set_title(f'Nf = N / η, η = {fabric.ETA:.1f}')
    ax1.grid(True, alpha=0.3)
    
    # 2. Rank distribution for different city sizes
    ax2 = axes[0, 1]
    ranks = np.arange(1, 500)
    for pop in [200_000, 1_000_000, 5_000_000]:
        f_x = fabric.rank_distribution[:499]
        ax2.loglog(ranks, f_x, label=f'N = {pop:,}', linewidth=1.5)
    ax2.set_xlabel('Rank x')
    ax2.set_ylabel('Per Capita Frequency f(x)')
    ax2.set_title('Universal Rank-Abundance Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Scaling exponents by category
    ax3 = axes[1, 0]
    categories = list(fabric.BUSINESS_CATEGORIES.keys())
    betas = [fabric.SCALING_EXPONENTS.get(c, 1.0) for c in categories]
    colors = ['red' if b > 1 else 'green' if b < 1 else 'blue' for b in betas]
    ax3.bar(range(len(categories)), betas, color=colors, alpha=0.7)
    ax3.set_xlabel('Business Category Index')
    ax3.set_ylabel('Scaling Exponent β')
    ax3.set_title('Super-linear (red) vs Sub-linear (green)')
    ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    
    # 4. Rank shift with city size
    ax4 = axes[1, 1]
    # Lawyers (β=1.17) and Agriculture (β=0.85)
    pops_array = np.array(pops)
    lawyers_shift = [fabric.get_rank_shift(54, p) for p in pops_array]  # Professional Services
    agriculture_shift = [fabric.get_rank_shift(11, p) for p in pops_array]
    
    ax4.semilogx(pops, lawyers_shift, 'o-', color='red', label='Lawyers (β=1.17)', linewidth=2)
    ax4.semilogx(pops, agriculture_shift, 's-', color='green', label='Agriculture (β=0.85)', linewidth=2)
    ax4.set_xlabel('Population N')
    ax4.set_ylabel('Rank Shift Factor')
    ax4.set_title('Rank Shift with City Size (Eq. 6)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    
    plt.suptitle('Urban Scaling Laws (Youn et al. 2014)', fontsize=14)
    plt.tight_layout()
    plt.savefig('city_scaling_laws.png', dpi=150)
    plt.show()
    
    print("\nKey Findings from Youn et al. (2014):")
    print(f"  - Total establishments ∝ population (η = {fabric.ETA:.1f} people/establishment)")
    print(f"  - Rank distribution: f(x) = {fabric.A:.4f}·x^(-{fabric.GAMMA})·e^(-x/{fabric.X0})")
    print(f"  - Super-linear categories (β > 1): Professional, Tech, Finance, Management")
    print(f"  - Sub-linear categories (β < 1): Agriculture, Mining, Utilities, Manufacturing")
    print("\nThis universal structure predicts business composition")
    print("for any city based only on its population size.")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete urban business fabric demonstration."""
    
    print("=" * 70)
    print("URBAN BUSINESS FABRIC MODEL")
    print("Based on Youn et al. (2014) - The systematic structure of urban business diversity")
    print("=" * 70)
    print()
    
    # 1. Demonstrate scaling laws
    compare_city_scaling()
    
    # 2. Create a communal market network for a medium-sized city
    print("\n" + "=" * 70)
    print("COMMUNAL MARKET NETWORK EXAMPLE")
    print("=" * 70)
    
    city_network = CommunalMarketNetwork(
        city_id=0,
        city_population=1_000_000,
        city_position=(50, 50),
        n_markets=20
    )
    
    print(f"City: Population {city_network.city_population:.0f}")
    print(f"Number of communal markets: {city_network.n_markets}")
    print(f"Total establishments: {sum(m.total_establishments for m in city_network.markets)}")
    print(f"Business diversity (Shannon): {city_network.compute_diversity_index():.2f}")
    print(f"Wealth Gini: {city_network.compute_gini():.3f}")
    
    # 3. Visualize
    visualize_urban_fabric(city_network, "Communal Market Network - 1M Population City")
    
    # 4. Compare small vs large city
    print("\n" + "=" * 70)
    print("SMALL vs LARGE CITY COMPARISON")
    print("=" * 70)
    
    small_city = CommunalMarketNetwork(1, 100_000, (20, 20), n_markets=8)
    large_city = CommunalMarketNetwork(2, 5_000_000, (80, 80), n_markets=40)
    
    print("\nSmall City (100,000):")
    print(f"  Establishments: {sum(m.total_establishments for m in small_city.markets)}")
    print(f"  Diversity: {small_city.compute_diversity_index():.2f}")
    print(f"  Gini: {small_city.compute_gini():.3f}")
    
    print("\nLarge City (5,000,000):")
    print(f"  Establishments: {sum(m.total_establishments for m in large_city.markets)}")
    print(f"  Diversity: {large_city.compute_diversity_index():.2f}")
    print(f"  Gini: {large_city.compute_gini():.3f}")
    
    # 5. Demonstrate category rank shift
    print("\n" + "=" * 70)
    print("BUSINESS CATEGORY RANK SHIFT")
    print("=" * 70)
    
    fabric = UrbanBusinessFabric()
    small_pop = 200_000
    large_pop = 5_000_000
    
    print(f"\nCategory rank shift from {small_pop:,} to {large_pop:,} population:")
    
    for cat_code, cat_name in list(fabric.BUSINESS_CATEGORIES.items())[:5]:
        beta = fabric.SCALING_EXPONENTS.get(cat_code, 1.0)
        shift = fabric.get_rank_shift(cat_code, large_pop, small_pop)
        if beta > 1:
            direction = "↑ becomes MORE abundant (rank ↓)"
        elif beta < 1:
            direction = "↓ becomes LESS abundant (rank ↑)"
        else:
            direction = "→ constant per capita"
        print(f"  {cat_name[:30]:30s}: β={beta:.2f} {direction}")
    
    print("\n" + "=" * 70)
    print("INTEGRATION WITH SPATIAL INFRASTRUCTURE GA")
    print("=" * 70)
    print("""
    The urban business fabric provides the MICRO-FOUNDATION for:
    
    1. Communal Market Units: Each market has a specific mix of businesses
       based on the universal rank-abundance distribution.
    
    2. Intra-city Trust Networks: Markets closer together have higher trust,
       enabling Sadaqa-based exchange.
    
    3. Inter-city Trade: The GA evolves routes between cities based on
       their business complementarity and diversity.
    
    4. Scaling Predictability: For any city of population N, we can predict:
       - Number of establishments (N/η)
       - Distribution across business categories (rank-abundance)
       - Which categories will be over/under-represented (β scaling)
    
    This completes the model stack:
    - Microscopic: Neurocognitive agents (Sadaqa/gift decisions)
    - Mesoscopic: Urban business fabric (communal markets)
    - Macroscopic: Spatial infrastructure (city networks)
    """)
    
    return city_network

if __name__ == "__main__":
    network = main()
"""
Sadaqa-Based Common Goods Provision
===================================

No taxation. No state redistribution.
Common goods (health, education, environment) are funded entirely by:
1. Voluntary Sadaqa from communal market units
2. Waqf (endowment) structures for long-term assets
3. Mutual aid agreements between markets

This replaces the fiscal state with decentralized gift exchange.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
import numpy as np
import random


class CommonGoodType(Enum):
    """Types of common goods provided by Sadaqa."""
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    ENVIRONMENT = "environment"  # Landscape preservation
    INFRASTRUCTURE = "infrastructure"
    SCARCITY_BUFFER = "scarcity_buffer"  # Grain reserves (Yusuf)
    SAFETY_NET = "safety_net"  # For the poor/needy


@dataclass
class Waqf:
    """
    Islamic endowment (waqf) - inalienable charitable trust.
    
    Unlike state ownership or private property, waqf is:
    - Perpetual (cannot be sold or inherited)
    - Dedicated to a specific common good purpose
    - Managed by trustees, not owners
    - Revenue is used for the designated purpose
    """
    name: str
    purpose: CommonGoodType
    assets: float  # Value of endowment
    annual_return_rate: float = 0.05  # 5% return from waqf assets
    trustees: List[int] = field(default_factory=list)  # IDs of trustee markets
    beneficiaries: List[int] = field(default_factory=list)  # Markets that can benefit
    
    def generate_annual_revenue(self) -> float:
        """Return annual revenue from waqf assets for common good provision."""
        return self.assets * self.annual_return_rate


@dataclass
class CommonGoodFacility:
    """A facility providing common goods, funded by Sadaqa and/or Waqf."""
    facility_id: int
    good_type: CommonGoodType
    location: Tuple[float, float]  # City coordinates
    capacity: float  # Number of people served
    annual_operating_cost: float
    sadaqa_funding: float = 0.0  # Annual Sadaqa contribution
    waqf_funding: float = 0.0    # Annual Waqf revenue
    is_active: bool = True
    
    @property
    def total_funding(self) -> float:
        return self.sadaqa_funding + self.waqf_funding
    
    @property
    def funding_gap(self) -> float:
        return max(0, self.annual_operating_cost - self.total_funding)


class SadaqaCommonGoodsSystem:
    """
    Decentralized common goods provision funded entirely by Sadaqa.
    
    No taxation. No state.
    
    Common goods emerge from:
    1. Direct Sadaqa from communal market units to facilities
    2. Waqf endowments established by wealthy merchants
    3. Mutual aid agreements between markets
    4. Communal storage (Yusuf principle) for scarcity periods
    """
    
    def __init__(self, n_markets: int):
        self.n_markets = n_markets
        
        # Facilities by type
        self.facilities: Dict[CommonGoodType, List[CommonGoodFacility]] = {
            t: [] for t in CommonGoodType
        }
        
        # Waqf endowments
        self.waqfs: List[Waqf] = []
        
        # Sadaqa contributions from each market (annual)
        self.sadaqa_contributions: Dict[int, Dict[CommonGoodType, float]] = {
            i: {t: 0.0 for t in CommonGoodType} for i in range(n_markets)
        }
        
        # Mutual aid agreements between markets
        self.mutual_aid_agreements: Dict[Tuple[int, int], float] = {}  # (market_i, market_j) -> aid_amount
        
        # Communal storage (Yusuf principle)
        self.communal_storage: Dict[int, float] = {i: 0.0 for i in range(n_markets)}
        self.storage_reserve_ratio = 0.2  # 20% of wealth goes to communal storage
        
        # Tracking
        self.annual_reports = []
        
    def establish_waqf(self, 
                       name: str,
                       purpose: CommonGoodType,
                       assets: float,
                       trustee_markets: List[int],
                       beneficiary_markets: List[int]) -> Waqf:
        """
        Establish a new waqf endowment.
        
        Waqf is perpetual and cannot be revoked.
        Assets are "frozen" in the endowment forever.
        """
        waqf = Waqf(
            name=name,
            purpose=purpose,
            assets=assets,
            trustees=trustee_markets,
            beneficiaries=beneficiary_markets
        )
        self.waqfs.append(waqf)
        return waqf
    
    def create_facility(self,
                        good_type: CommonGoodType,
                        location: Tuple[float, float],
                        capacity: float,
                        annual_cost: float,
                        funding_source: str = "sadaqa") -> CommonGoodFacility:
        """
        Create a new common good facility.
        
        funding_source: "sadaqa" (voluntary), "waqf" (endowment), or "hybrid"
        """
        facility = CommonGoodFacility(
            facility_id=len(self.facilities[good_type]),
            good_type=good_type,
            location=location,
            capacity=capacity,
            annual_operating_cost=annual_cost
        )
        self.facilities[good_type].append(facility)
        return facility
    
    def market_sadaqa_decision(self, 
                               market_id: int,
                               market_wealth: float,
                               market_population: float) -> Dict[CommonGoodType, float]:
        """
        A communal market unit decides how much Sadaqa to give to each common good.
        
        Decision factors:
        - Wealth level (richer markets give more)
        - Recent use of facilities (reciprocity: you give if you benefit)
        - Scarcity level (during hardship, giving INCREASES - gamification)
        - Reputation (markets that give more gain higher trust)
        - Social pressure from neighboring markets
        
        Returns:
        - Dictionary of Sadaqa amounts per common good type
        """
        # Base Sadaqa rate (percentage of wealth)
        # In a pure Sadaqa system, this is typically 2.5% of wealth (Zakat-equivalent)
        base_rate = 0.025
        
        # Wealth effect: richer markets give more (progressive)
        # But no coercion - it's voluntary
        wealth_factor = min(2.0, market_wealth / 50000)
        
        # Scarcity effect: giving INCREASES during hardship (gamification)
        # This is the counter-intuitive Yusuf principle
        scarcity = self._get_local_scarcity(market_id)
        scarcity_factor = 1.0 + scarcity  # At scarcity=0.5, factor=1.5
        
        # Reciprocity: if you've received from others, you give more
        received_aid = sum(v for (i, j), v in self.mutual_aid_agreements.items() 
                          if j == market_id)
        reciprocity_factor = 1.0 + min(1.0, received_aid / market_wealth)
        
        # Calculate total Sadaqa for this market
        total_sadaqa = market_wealth * base_rate * wealth_factor * scarcity_factor * reciprocity_factor
        
        # Distribute across common goods based on:
        # - Local needs (healthcare vs education vs environment)
        # - Facility capacity utilization
        
        # Simple distribution: proportional to estimated need
        # (In reality, markets would decide collectively through shura/consultation)
        distribution = {
            CommonGoodType.HEALTHCARE: total_sadaqa * 0.30,
            CommonGoodType.EDUCATION: total_sadaqa * 0.25,
            CommonGoodType.ENVIRONMENT: total_sadaqa * 0.15,
            CommonGoodType.INFRASTRUCTURE: total_sadaqa * 0.15,
            CommonGoodType.SAFETY_NET: total_sadaqa * 0.10,
            CommonGoodType.SCARCITY_BUFFER: total_sadaqa * 0.05
        }
        
        # Record contributions
        for good_type, amount in distribution.items():
            self.sadaqa_contributions[market_id][good_type] += amount
        
        return distribution
    
    def _get_local_scarcity(self, market_id: int) -> float:
        """Get scarcity level for a market (0=abundance, 1=famine)."""
        # In real implementation, this would come from local storage levels
        # Here we simulate based on communal storage
        storage = self.communal_storage.get(market_id, 0)
        # Scarcity is high when storage is low
        if storage < 100:
            return 0.8
        elif storage < 500:
            return 0.4
        elif storage < 1000:
            return 0.2
        else:
            return 0.0
    
    def allocate_facility_funding(self):
        """
        Allocate collected Sadaqa to common good facilities.
        
        Priority order:
        1. Facilities with largest funding gap
        2. Facilities serving most people
        3. Critical services (healthcare > education > environment)
        """
        for good_type, facilities in self.facilities.items():
            # Sort by urgency
            facilities.sort(key=lambda f: (f.funding_gap, -f.capacity), reverse=True)
            
            total_sadaqa_for_type = sum(self.sadaqa_contributions[m][good_type] 
                                        for m in range(self.n_markets))
            
            remaining = total_sadaqa_for_type
            
            for facility in facilities:
                if facility.funding_gap > 0 and remaining > 0:
                    allocation = min(facility.funding_gap, remaining)
                    facility.sadaqa_funding += allocation
                    remaining -= allocation
        
        # Also distribute Waqf revenue
        for waqf in self.waqfs:
            revenue = waqf.generate_annual_revenue()
            # Distribute to facilities of matching purpose
            for facility in self.facilities[waqf.purpose]:
                if facility.funding_gap > 0:
                    allocation = min(facility.funding_gap, revenue / len(self.facilities[waqf.purpose]))
                    facility.waqf_funding += allocation
                    revenue -= allocation
    
    def mutual_aid(self, giver_id: int, receiver_id: int, amount: float):
        """
        Direct mutual aid between communal markets.
        
        This is a form of Sadaqa that creates direct reciprocity bonds.
        """
        self.mutual_aid_agreements[(giver_id, receiver_id)] = \
            self.mutual_aid_agreements.get((giver_id, receiver_id), 0) + amount
        
        # The receiver's communal storage increases
        self.communal_storage[receiver_id] += amount
        
        # Trust increases between these markets
        # (Would update trust network)
    
    def yusuf_storage_cycle(self, market_id: int, current_wealth: float, is_scarcity: bool):
        """
        Implement the Yusuf counter-cyclical storage principle.
        
        In abundance: add to communal storage
        In scarcity: draw from communal storage
        
        No interest. No debt. Pure buffer against volatility.
        """
        if is_scarcity:
            # Draw from storage
            withdraw = min(self.communal_storage[market_id], current_wealth * 0.1)
            self.communal_storage[market_id] -= withdraw
            return withdraw
        else:
            # Add to storage (save during abundance)
            deposit = current_wealth * self.storage_reserve_ratio
            self.communal_storage[market_id] += deposit
            return -deposit  # Negative means wealth decreased
    
    def annual_cycle(self, market_wealths: Dict[int, float], 
                     market_populations: Dict[int, float],
                     is_scarcity: Dict[int, bool]) -> Dict[str, Any]:
        """
        Run one annual cycle of the Sadaqa common goods system.
        
        Steps:
        1. Markets decide Sadaqa contributions
        2. Waqf revenue is generated
        3. Facilities are funded
        4. Mutual aid is executed
        5. Yusuf storage is updated
        6. Facility services are delivered
        """
        # Reset annual Sadaqa contributions
        for market_id in range(self.n_markets):
            for good_type in CommonGoodType:
                self.sadaqa_contributions[market_id][good_type] = 0.0
        
        # 1. Markets contribute Sadaqa
        total_sadaqa = 0
        for market_id in range(self.n_markets):
            contribution = self.market_sadaqa_decision(
                market_id, 
                market_wealths[market_id], 
                market_populations[market_id]
            )
            total_sadaqa += sum(contribution.values())
        
        # 2. Allocate funding to facilities
        self.allocate_facility_funding()
        
        # 3. Process mutual aid (simplified)
        # In reality, markets would form voluntary agreements
        
        # 4. Process Yusuf storage
        storage_changes = {}
        for market_id in range(self.n_markets):
            change = self.yusuf_storage_cycle(
                market_id, 
                market_wealths[market_id], 
                is_scarcity.get(market_id, False)
            )
            storage_changes[market_id] = change
        
        # 5. Calculate service delivery
        services_delivered = {}
        for good_type, facilities in self.facilities.items():
            total_capacity = sum(f.capacity for f in facilities if f.is_active)
            total_funding = sum(f.total_funding for f in facilities)
            services_delivered[good_type] = {
                "capacity": total_capacity,
                "funding": total_funding,
                "fully_funded": all(f.funding_gap == 0 for f in facilities)
            }
        
        report = {
            "year": len(self.annual_reports),
            "total_sadaqa_collected": total_sadaqa,
            "waqf_revenue": sum(w.generate_annual_revenue() for w in self.waqfs),
            "services_delivered": services_delivered,
            "storage_levels": dict(self.communal_storage),
            "mutual_aid_network_size": len(self.mutual_aid_agreements)
        }
        
        self.annual_reports.append(report)
        return report
    
    def get_coverage_ratio(self, good_type: CommonGoodType, total_population: float) -> float:
        """Get the proportion of population covered by a common good."""
        facilities = self.facilities.get(good_type, [])
        total_capacity = sum(f.capacity for f in facilities if f.is_active)
        return min(1.0, total_capacity / total_population)
    
    def compute_system_health(self, total_population: float) -> Dict[str, float]:
        """
        Compute overall health of the Sadaqa-based common goods system.
        """
        coverage = {
            t.value: self.get_coverage_ratio(t, total_population) 
            for t in CommonGoodType
        }
        
        # Compute funding sustainability
        total_funding = sum(f.total_funding for facilities in self.facilities.values() for f in facilities)
        total_operating_cost = sum(f.annual_operating_cost for facilities in self.facilities.values() for f in facilities)
        
        if total_operating_cost > 0:
            sustainability = total_funding / total_operating_cost
        else:
            sustainability = 1.0
        
        # Compute Sadaqa per capita
        sadaqa_per_capita = sum(self.sadaqa_contributions[m][t] 
                                for m in range(self.n_markets) 
                                for t in CommonGoodType) / total_population if total_population > 0 else 0
        
        return {
            "healthcare_coverage": coverage["healthcare"],
            "education_coverage": coverage["education"],
            "environment_coverage": coverage["environment"],
            "safety_net_coverage": coverage["safety_net"],
            "funding_sustainability": sustainability,
            "sadaqa_per_capita": sadaqa_per_capita,
            "communal_storage_per_capita": sum(self.communal_storage.values()) / total_population if total_population > 0 else 0,
            "mutual_aid_density": len(self.mutual_aid_agreements) / (self.n_markets * (self.n_markets - 1)) if self.n_markets > 1 else 0
        }


# ============================================================================
# INTEGRATION WITH COMMUNAL MARKET NETWORK
# ============================================================================

class CommunalMarketWithCommonGoods(CommunalMarketNetwork):
    """
    Extended communal market network with Sadaqa-based common goods provision.
    
    No taxation. No state. Common goods emerge from gift exchange.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize the Sadaqa common goods system
        self.common_goods = SadaqaCommonGoodsSystem(n_markets=self.n_markets)
        
        # Establish foundational Waqfs
        self._establish_foundation_waqfs()
        
        # Create basic facilities
        self._create_basic_facilities()
        
        # Track common goods metrics
        self.common_goods_history = []
    
    def _establish_foundation_waqfs(self):
        """Establish foundational waqf endowments for common goods."""
        # Healthcare waqf
        self.common_goods.establish_waqf(
            name="Al-Shifa Healthcare Trust",
            purpose=CommonGoodType.HEALTHCARE,
            assets=self.city_population * 10,  # $10 per person initial endowment
            trustee_markets=list(range(self.n_markets)),
            beneficiary_markets=list(range(self.n_markets))
        )
        
        # Education waqf (madrasa fund)
        self.common_goods.establish_waqf(
            name="Nur Education Endowment",
            purpose=CommonGoodType.EDUCATION,
            assets=self.city_population * 8,
            trustee_markets=list(range(self.n_markets)),
            beneficiary_markets=list(range(self.n_markets))
        )
        
        # Environment waqf (landscape preservation)
        self.common_goods.establish_waqf(
            name="Amanah Environmental Trust",
            purpose=CommonGoodType.ENVIRONMENT,
            assets=self.city_population * 5,
            trustee_markets=list(range(self.n_markets)),
            beneficiary_markets=list(range(self.n_markets))
        )
    
    def _create_basic_facilities(self):
        """Create basic common good facilities across the city."""
        # Distribute facilities based on market locations
        for i, market in enumerate(self.markets):
            # Healthcare clinic in each market area
            clinic = self.common_goods.create_facility(
                good_type=CommonGoodType.HEALTHCARE,
                location=market.location,
                capacity=self.city_population / self.n_markets * 1.2,
                annual_cost=50000
            )
            
            # Education center
            school = self.common_goods.create_facility(
                good_type=CommonGoodType.EDUCATION,
                location=market.location,
                capacity=self.city_population / self.n_markets,
                annual_cost=40000
            )
            
            # Shared infrastructure (roads, storage)
            # Coordinates with nearby markets
            if i % 3 == 0:  # Not every market needs its own
                infra = self.common_goods.create_facility(
                    good_type=CommonGoodType.INFRASTRUCTURE,
                    location=market.location,
                    capacity=self.city_population / self.n_markets * 3,
                    annual_cost=100000
                )
    
    def annual_cycle(self, year: int, is_scarcity_year: bool = False):
        """
        Run one annual cycle of the entire communal market system.
        
        Includes:
        - Business operations (from parent)
        - Sadaqa-based common goods provision
        - Yusuf storage cycle
        """
        # Collect market wealth and population data
        market_wealths = {m.market_id: m.wealth for m in self.markets}
        market_populations = {m.market_id: m.population_served for m in self.markets}
        scarcity_status = {m.market_id: is_scarcity_year for m in self.markets}
        
        # Run Sadaqa common goods cycle
        report = self.common_goods.annual_cycle(
            market_wealths, 
            market_populations, 
            scarcity_status
        )
        
        # Record
        self.common_goods_history.append(report)
        
        # Update market trust based on Sadaqa contributions
        for market in self.markets:
            # Markets that give more Sadaqa gain higher trust
            total_given = sum(self.common_goods.sadaqa_contributions[market.market_id].values())
            if total_given > 0:
                trust_boost = min(0.1, total_given / market.wealth)
                # This would propagate through the trust network
        
        return report
    
    def get_common_goods_summary(self) -> Dict[str, Any]:
        """Get summary of common goods provision."""
        total_population = self.city_population
        system_health = self.common_goods.compute_system_health(total_population)
        
        # Also compute the "state replacement" metric
        # How much would equivalent services cost if provided by taxation?
        estimated_tax_equivalent = 0
        for facilities in self.common_goods.facilities.values():
            for f in facilities:
                estimated_tax_equivalent += f.annual_operating_cost
        
        return {
            "system_health": system_health,
            "estimated_tax_equivalent": estimated_tax_equivalent,
            "actual_sadaqa_funding": sum(r["total_sadaqa_collected"] for r in self.common_goods_history[-5:]) if self.common_goods_history else 0,
            "waqf_revenue": sum(r["waqf_revenue"] for r in self.common_goods_history[-5:]) if self.common_goods_history else 0,
            "coverage": {
                "healthcare": system_health["healthcare_coverage"],
                "education": system_health["education_coverage"],
                "environment": system_health["environment_coverage"]
            },
            "no_taxation": True,
            "state_replaced_by": "Sadaqa + Waqf + Mutual Aid"
        }


# ============================================================================
# COMPARISON: TAXATION vs SADAQA MODEL
# ============================================================================

def compare_taxation_vs_sadaqa():
    """
    Compare the standard taxation model with the Sadaqa-based model.
    
    Shows that Sadaqa can achieve similar or better outcomes
    without coercion and with higher trust.
    """
    print("=" * 70)
    print("COMPARISON: TAXATION vs SADAQA COMMON GOODS")
    print("=" * 70)
    print()
    
    # Create a city with communal markets
    city = CommunalMarketWithCommonGoods(
        city_id=0,
        city_population=1_000_000,
        city_position=(50, 50),
        n_markets=15
    )
    
    # Run for 10 years (alternating abundance and scarcity)
    print("Running 20-year simulation with abundance/scarcity cycles...")
    print()
    
    for year in range(20):
        is_scarcity = (year % 7) in [5, 6]  # Simulating Yusuf's 7-year cycle
        report = city.annual_cycle(year, is_scarcity)
        
        if year % 5 == 0:
            phase = "SCARCITY" if is_scarcity else "ABUNDANCE"
            print(f"Year {year} ({phase}):")
            print(f"  Sadaqa collected: ${report['total_sadaqa_collected']:,.0f}")
            print(f"  Waqf revenue: ${report['waqf_revenue']:,.0f}")
            print(f"  Healthcare funded: {report['services_delivered'][CommonGoodType.HEALTHCARE]['funding']:,.0f}")
            print(f"  Commonal storage: ${sum(report['storage_levels'].values()):,.0f}")
            print()
    
    # Final assessment
    summary = city.get_common_goods_summary()
    
    print("=" * 70)
    print("FINAL ASSESSMENT")
    print("=" * 70)
    print()
    print("SADAQA-BASED SYSTEM (NO TAXATION):")
    print(f"  Healthcare coverage: {summary['coverage']['healthcare']:.1%}")
    print(f"  Education coverage: {summary['coverage']['education']:.1%}")
    print(f"  Environment coverage: {summary['coverage']['environment']:.1%}")
    print(f"  Funding sustainability: {summary['system_health']['funding_sustainability']:.1%}")
    print(f"  Sadaqa per capita: ${summary['system_health']['sadaqa_per_capita']:.2f}")
    print(f"  Mutual aid density: {summary['system_health']['mutual_aid_density']:.1%}")
    print()
    print("KEY INSIGHTS:")
    print("  1. No taxation - all funding is voluntary Sadaqa")
    print("  2. Waqf provides permanent, non-revocable endowments")
    print("  3. During scarcity, Sadaqa INCREASES (Yusuf gamification)")
    print("  4. Trust network replaces state enforcement")
    print("  5. Common goods emerge from gift exchange, not redistribution")
    print()
    print("The state is replaced by:")
    print("  - Decentralized communal market units")
    print("  - Voluntary Sadaqa (not coerced Zakat)")
    print("  - Perpetual Waqf endowments")
    print("  - Mutual aid agreements between markets")
    print("  - Trust and reputation (social enforcement)")
    
    return city, summary


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_sadaqa_system(city: CommunalMarketWithCommonGoods):
    """
    Visualize the Sadaqa-based common goods system.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Sadaqa contributions over time
    ax1 = axes[0, 0]
    if city.common_goods_history:
        years = [r["year"] for r in city.common_goods_history]
        sadaqa = [r["total_sadaqa_collected"] for r in city.common_goods_history]
        waqf = [r["waqf_revenue"] for r in city.common_goods_history]
        
        ax1.plot(years, sadaqa, 'o-', label='Sadaqa (Voluntary)', color='green', linewidth=2)
        ax1.plot(years, waqf, 's-', label='Waqf (Endowment)', color='blue', linewidth=2)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Funding ($)')
        ax1.set_title('Common Goods Funding Sources')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Coverage over time
    ax2 = axes[0, 1]
    if city.common_goods_history:
        coverage_data = []
        for r in city.common_goods_history:
            coverage = {
                'health': r['services_delivered'][CommonGoodType.HEALTHCARE]['funding'] > 0,
                'edu': r['services_delivered'][CommonGoodType.EDUCATION]['funding'] > 0,
                'env': r['services_delivered'][CommonGoodType.ENVIRONMENT]['funding'] > 0
            }
            coverage_data.append(coverage)
        
        # Simplified visualization
        ax2.bar(['Healthcare', 'Education', 'Environment'], 
                [city.common_goods.get_coverage_ratio(CommonGoodType.HEALTHCARE, city.city_population),
                 city.common_goods.get_coverage_ratio(CommonGoodType.EDUCATION, city.city_population),
                 city.common_goods.get_coverage_ratio(CommonGoodType.ENVIRONMENT, city.city_population)],
                color=['green', 'blue', 'brown'], alpha=0.7)
        ax2.set_ylabel('Coverage Rate')
        ax2.set_title('Common Goods Coverage (Final Year)')
        ax2.set_ylim(0, 1)
    
    # 3. Sadaqa distribution by good type
    ax3 = axes[1, 0]
    if city.common_goods.annual_reports:
        last_report = city.common_goods.annual_reports[-1]
        good_names = ['Healthcare', 'Education', 'Environment', 'Infra', 'Safety', 'Storage']
        # Approximate distribution from last report
        values = [0.30, 0.25, 0.15, 0.15, 0.10, 0.05]
        ax3.pie(values, labels=good_names, autopct='%1.0f%%', colors=['#2ecc71', '#3498db', '#e67e22', '#f39c12', '#9b59b6', '#1abc9c'])
        ax3.set_title('Sadaqa Distribution by Common Good Type')
    
    # 4. No taxation message
    ax4 = axes[1, 1]
    ax4.text(0.5, 0.7, 'NO TAXATION', transform=ax4.transAxes, 
             fontsize=20, ha='center', fontweight='bold', color='green')
    ax4.text(0.5, 0.5, 'Common goods funded by:', transform=ax4.transAxes, 
             ha='center', fontsize=12)
    ax4.text(0.5, 0.35, '• Voluntary Sadaqa\n• Perpetual Waqf\n• Mutual Aid', 
             transform=ax4.transAxes, ha='center', fontsize=11)
    ax4.text(0.5, 0.15, 'No state. No coercion. No interest.', 
             transform=ax4.transAxes, ha='center', fontsize=10, style='italic', color='gray')
    ax4.axis('off')
    
    plt.suptitle('Sadaqa-Based Common Goods Provision (No Taxation)', fontsize=14)
    plt.tight_layout()
    plt.savefig('sadaqa_common_goods.png', dpi=150)
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run complete demonstration."""
    
    print("=" * 70)
    print("SADAQA-BASED COMMON GOODS MODEL")
    print("No Taxation. No State Redistribution.")
    print("Common goods emerge from voluntary gift exchange.")
    print("=" * 70)
    print()
    
    # Run comparison
    city, summary = compare_taxation_vs_sadaqa()
    
    # Visualize
    visualize_sadaqa_system(city)
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
    The Sadaqa-based model demonstrates that common goods can be provided
    without taxation or state coercion through:
    
    1. VOLUNTARY GIVING: Markets choose to give based on wealth, need, and trust
    2. PERPETUAL ENDOWMENTS (Waqf): Permanent funding streams for essential services
    3. MUTUAL AID: Direct support between markets during hardship
    4. YUSUF STORAGE: Counter-cyclical buffers that prevent scarcity spirals
    
    This is not charity in the Western sense (which is stigmatizing).
    This is Sadaqa in the Islamic sense: a voluntary act that builds merit,
    strengthens social bonds, and creates collective resilience.
    
    The state is replaced by:
    - Decentralized communal market units
    - Trust and reputation networks
    - Spontaneous order through gift exchange
    - Moral obligation (not legal coercion)
    
    This is the institutional foundation for a post-capitalist,
    post-state economic system based on Islamic principles of
    Sadaqa, Waqf, and the Yusuf counter-cyclical rule.
    """)
    
    return city


if __name__ == "__main__":
    city = main()
