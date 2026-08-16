#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation mondiale intégrée du système Franc-Jy
Combine:
- Système bimétallique à deux monnaies (two_currency.py)
- Équation d'état de van der Waals (van_der_waals.py)
- Contrôle ago-antagoniste (ago_antagonistic.py)
- Métriques thermodynamiques (thermodynamic_metrics.py)
- Réseau de confiance géométrique (ricci_yusuf_network.py)
- Transition génétique des nations (world_simulation.py)

Auteur: Marc Daghar
Licence: CC BY-SA 4.0
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Import des modules existants
from src.economy.two_currency import TwoCurrencySystem, TwoCurrencyConfig
from src.economy.van_der_waals import VanDerWaalsEconomy, YusufSystemWithPhaseTransition
from src.learning.ago_antagonistic import AgoAntagonisticController, YusufSystemWithAgoAntagonistic
from src.validation.thermodynamic_metrics import ThermodynamicMetrics, compare_thermodynamic_metrics
from src.ricci_yusuf_network import GeometricRicciNetwork
from src.world_simulation import World, WorldConfig, NationConfig, Nation


@dataclass
class IntegratedWorldConfig:
    """Configuration de la simulation mondiale intégrée"""
    # Paramètres temporels
    years: int = 100
    dt: float = 0.1
    n_steps: int = 1000
    
    # Paramètres économiques
    need: float = 0.06
    P_mean: float = 1.0
    P_amplitude: float = 0.5
    period: int = 14  # Cycle de 7+7 ans
    
    # Bimétallisme
    bimetal_ratio_init: float = 16.0  # Or/Argent
    gold_reserve: float = 100.0
    silver_reserve: float = 100.0
    
    # Monnaie électronique (Franc-Jy)
    cry_currency_enabled: bool = True
    cry_currency_demurrage: float = 0.03  # Décroissance pour encourager la circulation
    
    # Contrôle ago-antagoniste
    ago_antagonistic_enabled: bool = True
    
    # Gamification (crédit social)
    gamification_enabled: bool = True
    compliance_impact: float = 0.5
    
    # Évolution génétique
    n_generations: int = 30
    population_size: int = 20
    mutation_rate: float = 0.05
    crossover_rate: float = 0.7
    
    # Chocs
    shock_year: Optional[int] = 40
    shock_magnitude: float = 0.3


class IntegratedNation(Nation):
    """
    Nation intégrée avec :
    - Système bimétallique (TwoCurrencySystem)
    - Détection de phase (van der Waals)
    - Contrôle ago-antagoniste
    - Réseau de confiance (Ricci)
    """
    
    def __init__(self, config: NationConfig, world_config: IntegratedWorldConfig):
        super().__init__(config, world_config)
        self.world_config = world_config
        
        # Configuration du système à deux monnaies
        self.two_currency_config = TwoCurrencyConfig(
            T=world_config.years,
            dt=world_config.dt,
            n_steps=world_config.n_steps,
            P_mean=world_config.P_mean,
            P_amplitude=world_config.P_amplitude,
            need=world_config.need,
            cry_currency_enabled=config.system_type == 'yusuf',
            cry_currency_demurrage=world_config.cry_currency_demurrage,
            exchange_rate_fixed=world_config.bimetal_ratio_init,
            tax_rate_dominant=0.20,
            tax_rate_cry=0.05
        )
        
        # Initialisation du système à deux monnaies
        self.two_currency = TwoCurrencySystem(self.two_currency_config)
        
        # Initialisation de van der Waals
        self.vdw = VanDerWaalsEconomy(a=0.5, b=0.2, R=1.0)
        
        # Initialisation du contrôleur ago-antagoniste
        self.ago_controller = AgoAntagonisticController(self.world_config)
        
        # Réseau de confiance géométrique (Ricci)
        self.ricci_network = None
        if config.grondona_enabled:
            nodes = [config.name] + [f"Trade_{i}" for i in range(5)]
            edges = [(nodes[0], nodes[i], 1.0) for i in range(1, len(nodes))]
            self.ricci_network = GeometricRicciNetwork(nodes, edges, world_config.bimetal_ratio_init)
        
        # Résultats enrichis
        self.vdw_results = None
        self.ago_results = None
        self.thermodynamic_metrics = None
    
    def run_simulation(self) -> Dict[str, Any]:
        """Exécute la simulation intégrée"""
        # 1. Simulation du système à deux monnaies
        tw_result = self.two_currency.run()
        
        # 2. Simulation avec détection de phase (van der Waals)
        vdw_system = YusufSystemWithPhaseTransition(self.world_config, self.vdw)
        self.vdw_results = vdw_system.run()
        
        # 3. Simulation avec contrôle ago-antagoniste
        ago_system = YusufSystemWithAgoAntagonistic(self.world_config)
        self.ago_results = ago_system.run()
        
        # 4. Mise à jour du réseau Ricci
        if self.ricci_network:
            for _ in range(10):  # Quelques pas pour stabiliser
                self.ricci_network.step()
        
        # 5. Calcul des métriques thermodynamiques
        self.thermodynamic_metrics = compare_thermodynamic_metrics(
            self.vdw_results, 
            self.ago_results
        )
        
        # 6. Assemblage des résultats
        self.results = {
            'two_currency': tw_result,
            'vdw': self.vdw_results,
            'ago': self.ago_results,
            'thermodynamic': self.thermodynamic_metrics,
            'ricci_ratio': self.ricci_network.ratio if self.ricci_network else self.world_config.bimetal_ratio_init,
            'crisis_detected': self.vdw_results.get('crisis_detected', False)
        }
        
        return self.results
    
    def compute_fitness(self) -> float:
        """Calcule la fitness avec les métriques thermodynamiques"""
        if self.results is None:
            return 0.0
        
        metrics = self.results['thermodynamic']
        
        # Critères de fitness enrichis
        stability_score = 1.0 / (1.0 + self.vdw_results.get('seneca_risk', 0.5))
        solvency_score = self.config.initial_stock / (self.config.initial_debt + 0.1)
        entropy_score = 1.0 / (1.0 + metrics.entropy_production_yusuf)
        resilience_score = 1.0 - metrics.seneca_risk
        
        # Si le système est en phase critique, pénalité
        if metrics.phase == "critical":
            resilience_score *= 0.5
        
        self.fitness = (
            0.25 * stability_score +
            0.25 * solvency_score +
            0.25 * entropy_score +
            0.25 * resilience_score
        )
        
        return self.fitness


class IntegratedWorld(World):
    """Monde intégré avec toutes les composantes"""
    
    def __init__(self, config: IntegratedWorldConfig):
        self.config = config
        self.nations: List[IntegratedNation] = []
        self.generation = 0
        self.history = []
        self.best_nation = None
        
        # Initialisation des nations
        self._initialize_nations()
        
        # Historique des métriques thermodynamiques
        self.thermodynamic_history = []
    
    def _initialize_nations(self):
        """Initialise la population de nations intégrées"""
        self.nations = []
        
        # Nation de référence (Franc-Jy)
        france_config = NationConfig(
            name="France (Franc-Jy)",
            system_type="yusuf",
            initial_stock=0.8,
            need=self.config.need,
            grondona_enabled=True,
            interest_rate=0.0  # Pas d'intérêt dans le système Yusuf
        )
        self.nations.append(IntegratedNation(france_config, self.config))
        
        # Nations étrangères (capitalistes)
        for i in range(self.config.population_size - 1):
            config = NationConfig(
                name=f"Nation_{i+1}",
                system_type="capitalist",
                initial_stock=np.random.uniform(0.2, 0.8),
                initial_debt=np.random.uniform(0.3, 1.0),
                need=np.random.uniform(0.04, 0.08),
                interest_rate=np.random.uniform(0.10, 0.30),
                grondona_enabled=np.random.choice([True, False], p=[0.3, 0.7])
            )
            self.nations.append(IntegratedNation(config, self.config))
    
    def run_generation(self) -> Dict:
        """Exécute une génération avec toutes les composantes"""
        results = []
        
        for nation in self.nations:
            nation.run_simulation()
            fitness = nation.compute_fitness()
            
            # Collecte des métriques thermodynamiques
            if nation.thermodynamic_metrics:
                metrics = nation.thermodynamic_metrics
                self.thermodynamic_history.append({
                    'generation': self.generation,
                    'nation': nation.name,
                    'entropy_yusuf': metrics.entropy_production_yusuf,
                    'entropy_capitalist': metrics.entropy_production_capitalist,
                    'temperature_yusuf': metrics.temperature_yusuf,
                    'temperature_capitalist': metrics.temperature_capitalist,
                    'seneca_risk': metrics.seneca_risk,
                    'phase': metrics.phase,
                    'critical_proximity': metrics.critical_proximity
                })
            
            results.append({
                'name': nation.name,
                'system_type': nation.config.system_type,
                'fitness': fitness,
                'stock_final': nation.results['two_currency'].S_euro[-1],
                'cry_stock': nation.results['two_currency'].S_cry[-1],
                'exchange_rate': nation.results['two_currency'].exchange_rate[-1],
                'ricci_ratio': nation.results['ricci_ratio'],
                'seneca_risk': nation.results['thermodynamic'].seneca_risk,
                'phase': nation.results['thermodynamic'].phase,
                'crisis_detected': nation.results['crisis_detected'],
                'grondona_enabled': nation.config.grondona_enabled,
                'interest_rate': nation.config.interest_rate,
                'generation': self.generation
            })
        
        self.history.append(results)
        
        # Meilleure nation
        best = max(results, key=lambda x: x['fitness'])
        self.best_nation = best
        
        return {
            'generation': self.generation,
            'best_fitness': best['fitness'],
            'best_name': best['name'],
            'best_system': best['system_type'],
            'yusuf_count': sum(1 for r in results if r['system_type'] == 'yusuf'),
            'capitalist_count': sum(1 for r in results if r['system_type'] == 'capitalist'),
            'grondona_count': sum(1 for r in results if r['grondona_enabled']),
            'mean_seneca_risk': np.mean([r['seneca_risk'] for r in results]),
            'mean_fitness': np.mean([r['fitness'] for r in results])
        }
    
    def get_thermodynamic_summary(self) -> pd.DataFrame:
        """Retourne un résumé des métriques thermodynamiques"""
        return pd.DataFrame(self.thermodynamic_history)


def run_integrated_simulation() -> Dict:
    """Exécute la simulation intégrée complète"""
    print("=" * 70)
    print("SIMULATION MONDIALE INTÉGRÉE DU SYSTÈME FRANC-JY")
    print("Avec bimétallisme, van der Waals, contrôle ago-antagoniste et réseau Ricci")
    print("=" * 70)
    
    # Configuration
    config = IntegratedWorldConfig(
        years=100,
        n_generations=25,
        population_size=25,
        shock_year=40,
        shock_magnitude=0.3
    )
    
    print("\n1. Initialisation du monde...")
    world = IntegratedWorld(config)
    
    print("2. Évolution des nations (algorithme génétique)...")
    results = []
    for gen in range(config.n_generations):
        result = world.run_generation()
        results.append(result)
        
        # Affichage de la progression
        if gen % 5 == 0:
            print(f"   Génération {gen+1}: meilleure fitness = {result['best_fitness']:.4f} ({result['best_system']})")
            print(f"                  Nations Yusuf: {result['yusuf_count']}/{config.population_size}")
        
        # Évolution de la population
        world._evolve_population()
    
    print("\n3. Génération des résumés...")
    summary = world.get_summary()
    thermo_summary = world.get_thermodynamic_summary()
    
    # Résultats finaux
    final = results[-1]
    
    print("\n" + "=" * 70)
    print("RÉSULTATS FINAUX")
    print("=" * 70)
    print(f"Générations: {config.n_generations}")
    print(f"Meilleure fitness: {final['best_fitness']:.4f} ({final['best_system']})")
    print(f"Meilleure nation: {final['best_name']}")
    print(f"Nations Yusuf: {final['yusuf_count']}/{config.population_size}")
    print(f"Nations avec Grondona: {final['grondona_count']}/{config.population_size}")
    print(f"Risque Sénèque moyen: {final['mean_seneca_risk']:.4f}")
    
    # Sauvegarde
    output_dir = Path("results/integrated")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary.to_csv(output_dir / "nations_summary.csv", index=False)
    thermo_summary.to_csv(output_dir / "thermodynamic_summary.csv", index=False)
    
    # Sauvegarde de l'évolution
    pd.DataFrame(results).to_csv(output_dir / "generation_evolution.csv", index=False)
    
    print(f"\n4. Résultats sauvegardés dans {output_dir}")
    
    return {
        'world': world,
        'results': results,
        'summary': summary,
        'thermo_summary': thermo_summary,
        'final_generation': final
    }


if __name__ == "__main__":
    np.random.seed(42)
    results = run_integrated_simulation()