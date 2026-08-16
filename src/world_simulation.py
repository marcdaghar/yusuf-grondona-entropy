#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module de simulation mondiale pour la transition du système de réserve fractionnaire
vers un système de type Franc-Jy, basé sur le modèle Yusuf-Grondona.

Auteur: Marc Daghar
Licence: CC BY-SA 4.0
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import json
from pathlib import Path

# Import des modules existants
from yusuf_model import YusufConfig, YusufSystem, CapitalistSystem, SimulationResult
from grondona_crd import GrondonaCRD
from neurocognitive_agents import NeurocognitiveAgent


@dataclass
class NationConfig:
    """Configuration d'une nation dans la simulation mondiale"""
    name: str
    system_type: str  # 'yusuf' ou 'capitalist'
    initial_stock: float = 0.5
    initial_debt: float = 0.5
    need: float = 0.06
    interest_rate: float = 0.22  # Taux d'intérêt pour les nations capitalistes
    gamma_high: float = 0.5  # Taux d'épargne en abondance (Yusuf)
    gamma_low: float = 0.85  # Taux de puisage en rareté (Yusuf)
    threshold_ratio: float = 0.25  # Seuil de basculement
    cycle_period: int = 14  # Période du cycle (7+7 ans)
    production_mean: float = 1.0
    production_amplitude: float = 0.5
    noise_amplitude: float = 0.03
    
    # Paramètres du corridor Grondona
    grondona_enabled: bool = True
    gold_reserve: float = 100.0
    silver_reserve: float = 100.0
    target_ratio: float = 16.0  # Ratio or/argent cible
    
    # Paramètres génétiques (pour l'évolution)
    fitness_weight_stability: float = 0.3
    fitness_weight_solvency: float = 0.4
    fitness_weight_equality: float = 0.2
    fitness_weight_trade: float = 0.1


@dataclass
class WorldConfig:
    """Configuration de la simulation mondiale"""
    years: int = 100
    dt: float = 0.1
    n_generations: int = 50  # Nombre de générations pour l'algorithme génétique
    population_size: int = 20  # Nombre de nations dans la simulation
    mutation_rate: float = 0.05
    crossover_rate: float = 0.7
    
    # Nations initiales
    nations: List[NationConfig] = field(default_factory=list)
    
    # Chocs mondiaux
    global_shock_year: Optional[int] = None
    global_shock_magnitude: float = 0.4


class Nation:
    """
    Représente une nation dans la simulation mondiale.
    Chaque nation a son propre système monétaire (Yusuf ou capitaliste)
    et son propre corridor Grondona.
    """
    
    def __init__(self, config: NationConfig, world_config: WorldConfig):
        self.config = config
        self.world_config = world_config
        self.name = config.name
        
        # Initialisation du système
        if config.system_type == 'yusuf':
            yusuf_config = YusufConfig(
                T=world_config.years,
                dt=world_config.dt,
                need=config.need,
                P_mean=config.production_mean,
                P_amplitude=config.production_amplitude,
                period=config.cycle_period,
                stock_initial=config.initial_stock,
                threshold_factor=config.threshold_ratio,
                noise_amplitude=config.noise_amplitude
            )
            self.system = YusufSystem(yusuf_config)
        else:  # capitalist
            self.system = CapitalistSystem(
                YusufConfig(
                    T=world_config.years,
                    dt=world_config.dt,
                    need=config.need,
                    P_mean=config.production_mean,
                    P_amplitude=config.production_amplitude,
                    period=config.cycle_period,
                    stock_initial=config.initial_stock,
                    noise_amplitude=config.noise_amplitude
                )
            )
            self.system.interest_rate = config.interest_rate
        
        # Corridor Grondona
        if config.grondona_enabled:
            self.grondona = GrondonaCRD(
                gold_reserve=config.gold_reserve,
                silver_reserve=config.silver_reserve,
                target_ratio=config.target_ratio
            )
        else:
            self.grondona = None
        
        # Résultats
        self.results = None
        self.fitness = 0.0
        self.generation = 0
        
        # Stock des agents
        self.agents = []
    
    def run_simulation(self) -> SimulationResult:
        """Exécute la simulation de la nation"""
        if self.config.system_type == 'yusuf':
            self.results = self.system.run()
        else:
            self.results = self.system.run()
        return self.results
    
    def compute_fitness(self) -> float:
        """Calcule la fitness de la nation selon les critères définis"""
        if self.results is None:
            return 0.0
        
        c = self.config
        
        # 1. Stabilité (volatilité de la consommation)
        stability_score = 1.0 / (1.0 + self.results.consumption_volatility)
        
        # 2. Solvabilité
        solvency_score = self.results.solvency_rate / 100.0
        
        # 3. Égalité (Gini inversé)
        gini = self._compute_gini()
        equality_score = 1.0 - gini
        
        # 4. Équilibre commercial (simulé)
        trade_score = self._compute_trade_balance()
        
        # Fitness pondérée
        self.fitness = (
            c.fitness_weight_stability * stability_score +
            c.fitness_weight_solvency * solvency_score +
            c.fitness_weight_equality * equality_score +
            c.fitness_weight_trade * trade_score
        )
        
        return self.fitness
    
    def _compute_gini(self) -> float:
        """Calcule le coefficient de Gini (approximé)"""
        if self.results is None:
            return 0.5
        # Simulation simplifiée : Gini basé sur la volatilité
        return 0.5 * self.results.consumption_volatility / 0.3
    
    def _compute_trade_balance(self) -> float:
        """Calcule un score d'équilibre commercial (approximé)"""
        if self.results is None:
            return 0.5
        # Simuler un déséquilibre basé sur la production
        trade_imbalance = abs(self.results.P[-1] - self.results.C[-1])
        return 1.0 / (1.0 + trade_imbalance)
    
    def mutate(self, mutation_rate: float = 0.05) -> 'Nation':
        """Crée une copie mutée de la nation"""
        new_config = NationConfig(
            name=self.name + "_mutated",
            system_type=self.config.system_type,
            initial_stock=max(0.1, self.config.initial_stock + np.random.normal(0, 0.1)),
            initial_debt=max(0.0, self.config.initial_debt + np.random.normal(0, 0.1)),
            need=max(0.01, self.config.need + np.random.normal(0, 0.01)),
            interest_rate=max(0.0, self.config.interest_rate + np.random.normal(0, 0.02)),
            gamma_high=np.clip(self.config.gamma_high + np.random.normal(0, 0.05), 0.1, 1.0),
            gamma_low=np.clip(self.config.gamma_low + np.random.normal(0, 0.05), 0.1, 1.0),
            threshold_ratio=np.clip(self.config.threshold_ratio + np.random.normal(0, 0.05), 0.01, 0.9),
            grondona_enabled=self.config.grondona_enabled,
            gold_reserve=self.config.gold_reserve + np.random.normal(0, 10),
            silver_reserve=self.config.silver_reserve + np.random.normal(0, 10),
            fitness_weight_stability=self.config.fitness_weight_stability,
            fitness_weight_solvency=self.config.fitness_weight_solvency,
            fitness_weight_equality=self.config.fitness_weight_equality,
            fitness_weight_trade=self.config.fitness_weight_trade
        )
        return Nation(new_config, self.world_config)


class World:
    """
    Simulation mondiale avec algorithme génétique.
    Gère l'évolution des nations vers le système Franc-Jy.
    """
    
    def __init__(self, config: WorldConfig):
        self.config = config
        self.nations: List[Nation] = []
        self.generation = 0
        self.history = []
        self.best_nation = None
        
        # Initialisation des nations
        self._initialize_nations()
    
    def _initialize_nations(self):
        """Initialise la population initiale de nations"""
        self.nations = []
        
        # Nation de référence (Franc-Jy)
        france_config = NationConfig(
            name="France (Franc-Jy)",
            system_type="yusuf",
            initial_stock=0.8,
            need=0.06,
            gamma_high=0.5,
            gamma_low=0.85,
            threshold_ratio=0.25,
            grondona_enabled=True,
            gold_reserve=200.0,
            silver_reserve=200.0
        )
        self.nations.append(Nation(france_config, self.config))
        
        # Nations étrangères (capitalistes)
        for i in range(self.config.population_size - 1):
            config = NationConfig(
                name=f"Nation_{i+1}",
                system_type="capitalist",
                initial_stock=np.random.uniform(0.2, 0.8),
                initial_debt=np.random.uniform(0.3, 1.0),
                need=np.random.uniform(0.04, 0.08),
                interest_rate=np.random.uniform(0.10, 0.30),
                gamma_high=np.random.uniform(0.3, 0.7),
                gamma_low=np.random.uniform(0.7, 0.95),
                threshold_ratio=np.random.uniform(0.1, 0.4),
                grondona_enabled=np.random.choice([True, False], p=[0.3, 0.7])
            )
            self.nations.append(Nation(config, self.config))
    
    def run_generation(self) -> Dict:
        """Exécute une génération de simulation"""
        results = []
        
        for nation in self.nations:
            nation.run_simulation()
            fitness = nation.compute_fitness()
            results.append({
                'name': nation.name,
                'system_type': nation.config.system_type,
                'fitness': fitness,
                'stock_final': nation.results.final_stock,
                'solvency': nation.results.solvency_rate,
                'volatility': nation.results.consumption_volatility,
                'gini': nation._compute_gini(),
                'grondona_enabled': nation.config.grondona_enabled,
                'interest_rate': nation.config.interest_rate,
                'gamma_high': nation.config.gamma_high,
                'gamma_low': nation.config.gamma_low,
                'threshold_ratio': nation.config.threshold_ratio,
                'generation': self.generation
            })
        
        self.history.append(results)
        
        # Identifier la meilleure nation
        best = max(results, key=lambda x: x['fitness'])
        self.best_nation = best
        
        return {
            'generation': self.generation,
            'best_fitness': best['fitness'],
            'best_name': best['name'],
            'best_system': best['system_type'],
            'fitness_mean': np.mean([r['fitness'] for r in results]),
            'fitness_std': np.std([r['fitness'] for r in results]),
            'yusuf_count': sum(1 for r in results if r['system_type'] == 'yusuf'),
            'capitalist_count': sum(1 for r in results if r['system_type'] == 'capitalist'),
            'grondona_count': sum(1 for r in results if r['grondona_enabled'])
        }
    
    def evolve(self) -> List[Dict]:
        """Exécute l'algorithme génétique pour plusieurs générations"""
        generation_results = []
        
        for gen in range(self.config.n_generations):
            self.generation = gen
            result = self.run_generation()
            generation_results.append(result)
            
            # Sélection, croisement et mutation
            self._evolve_population()
        
        return generation_results
    
    def _evolve_population(self):
        """Sélectionne les meilleures nations et crée la nouvelle génération"""
        # Trier par fitness (décroissant)
        sorted_nations = sorted(self.nations, key=lambda n: n.fitness, reverse=True)
        
        # Sélection (top 30%)
        n_select = max(2, int(self.config.population_size * 0.3))
        selected = sorted_nations[:n_select]
        
        # Garder la meilleure nation (élitisme)
        new_nations = [sorted_nations[0]]
        
        # Croisement et mutation
        while len(new_nations) < self.config.population_size:
            # Sélectionner deux parents
            parent1 = np.random.choice(selected)
            parent2 = np.random.choice(selected)
            
            # Croisement (crossover)
            if np.random.random() < self.config.crossover_rate:
                child_config = self._crossover(parent1.config, parent2.config)
            else:
                child_config = parent1.config
            
            # Mutation
            if np.random.random() < self.config.mutation_rate:
                child = self._mutate_nation(child_config)
            else:
                child = Nation(child_config, self.config)
            
            new_nations.append(child)
        
        self.nations = new_nations
    
    def _crossover(self, config1: NationConfig, config2: NationConfig) -> NationConfig:
        """Croise deux configurations de nations"""
        # Choix aléatoire des attributs
        return NationConfig(
            name="crossed",
            system_type=np.random.choice([config1.system_type, config2.system_type]),
            initial_stock=np.random.choice([config1.initial_stock, config2.initial_stock]),
            initial_debt=np.random.choice([config1.initial_debt, config2.initial_debt]),
            need=np.mean([config1.need, config2.need]),
            interest_rate=np.mean([config1.interest_rate, config2.interest_rate]),
            gamma_high=np.mean([config1.gamma_high, config2.gamma_high]),
            gamma_low=np.mean([config1.gamma_low, config2.gamma_low]),
            threshold_ratio=np.mean([config1.threshold_ratio, config2.threshold_ratio]),
            grondona_enabled=config1.grondona_enabled or config2.grondona_enabled,
            gold_reserve=np.mean([config1.gold_reserve, config2.gold_reserve]),
            silver_reserve=np.mean([config1.silver_reserve, config2.silver_reserve]),
            fitness_weight_stability=config1.fitness_weight_stability,
            fitness_weight_solvency=config1.fitness_weight_solvency,
            fitness_weight_equality=config1.fitness_weight_equality,
            fitness_weight_trade=config1.fitness_weight_trade
        )
    
    def _mutate_nation(self, config: NationConfig) -> Nation:
        """Mutate une nation"""
        nation = Nation(config, self.config)
        return nation.mutate(self.config.mutation_rate)
    
    def get_summary(self) -> pd.DataFrame:
        """Retourne un résumé des résultats sous forme de DataFrame"""
        rows = []
        for gen_result in self.history:
            for nation in gen_result:
                rows.append(nation)
        return pd.DataFrame(rows)
    
    def get_evolution_summary(self) -> pd.DataFrame:
        """Retourne l'évolution des métriques clés"""
        rows = []
        for i, gen_result in enumerate(self.history):
            yusuf_nations = [r for r in gen_result if r['system_type'] == 'yusuf']
            capitalist_nations = [r for r in gen_result if r['system_type'] == 'capitalist']
            
            rows.append({
                'generation': i,
                'best_fitness': max(r['fitness'] for r in gen_result),
                'best_system': max(gen_result, key=lambda x: x['fitness'])['system_type'],
                'yusuf_count': len(yusuf_nations),
                'capitalist_count': len(capitalist_nations),
                'grondona_count': sum(1 for r in gen_result if r['grondona_enabled']),
                'mean_fitness_yusuf': np.mean([r['fitness'] for r in yusuf_nations]) if yusuf_nations else 0,
                'mean_fitness_capitalist': np.mean([r['fitness'] for r in capitalist_nations]) if capitalist_nations else 0,
                'mean_interest_rate': np.mean([r['interest_rate'] for r in capitalist_nations]) if capitalist_nations else 0,
                'mean_gamma_high': np.mean([r['gamma_high'] for r in gen_result]),
                'mean_threshold_ratio': np.mean([r['threshold_ratio'] for r in gen_result])
            })
        return pd.DataFrame(rows)


def run_transition_simulation() -> Dict:
    """Exécute la simulation de transition complète"""
    print("=== SIMULATION DE TRANSITION VERS LE SYSTÈME FRANC-JY ===")
    print("1. Initialisation des nations...")
    
    # Configuration de la simulation
    world_config = WorldConfig(
        years=100,
        dt=0.1,
        n_generations=30,
        population_size=30,
        mutation_rate=0.05,
        crossover_rate=0.7,
        global_shock_year=40,
        global_shock_magnitude=0.3
    )
    
    # Création du monde
    world = World(world_config)
    
    print("2. Évolution des nations...")
    results = world.evolve()
    
    print("3. Génération du résumé...")
    summary = world.get_evolution_summary()
    
    # Résultats finaux
    final_generation = results[-1]
    
    print("\n=== RÉSULTATS FINAUX ===")
    print(f"Génération finale: {world.generation}")
    print(f"Meilleure fitness: {final_generation['best_fitness']:.4f}")
    print(f"Meilleur système: {final_generation['best_system']}")
    print(f"Nations Yusuf: {final_generation['yusuf_count']}")
    print(f"Nations capitalistes: {final_generation['capitalist_count']}")
    print(f"Nations avec corridor Grondona: {final_generation['grondona_count']}")
    
    # Sauvegarde des résultats
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    summary.to_csv(output_dir / "evolution_summary.csv", index=False)
    summary.to_json(output_dir / "evolution_summary.json", orient="records")
    
    print(f"\n4. Résultats sauvegardés dans {output_dir}")
    
    return {
        'world': world,
        'results': results,
        'summary': summary,
        'final_generation': final_generation
    }


if __name__ == "__main__":
    np.random.seed(42)
    results = run_transition_simulation()
    
    # Affichage de la dernière ligne du résumé
    print("\n=== DERNIÈRE GÉNÉRATION ===")
    print(results['summary'].tail(1).to_string())