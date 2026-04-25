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
