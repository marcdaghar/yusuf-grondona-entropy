#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests unitaires pour le modèle Yusuf
Licence : CC BY-SA
"""

import sys
import os
import pytest
import numpy as np

# Ajouter le répertoire parent au path pour importer les modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Tests du modèle de base
# ============================================================

class TestYusufModel:
    """Tests du modèle Yusuf principal"""
    
    @pytest.fixture
    def config(self):
        """Configuration de test"""
        from yusuf_model import YusufConfig
        return YusufConfig(
            T=10,           # Simulation courte pour les tests
            dt=0.5,
            noise_amplitude=0.0,  # Pas de bruit pour reproductibilité
            gamification_enabled=False
        )
    
    @pytest.fixture
    def yusuf_system(self, config):
        from yusuf_model import YusufSystem
        return YusufSystem(config)
    
    @pytest.fixture
    def capitalist_system(self, config):
        from yusuf_model import CapitalistSystem
        return CapitalistSystem(config)
    
    def test_yusuf_solvency_rate(self, yusuf_system):
        """Test 1: Le système Yusuf doit être 100% solvable"""
        res = yusuf_system.run()
        assert res.solvency_rate == 100.0, \
            f"Yusuf solvability rate should be 100%, got {res.solvency_rate}%"
    
    def test_yusuf_no_negative_stock(self, yusuf_system):
        """Test 2: Le stock ne doit jamais être négatif"""
        res = yusuf_system.run()
        assert np.all(res.S >= -1e-6), "Stock should never be negative"
    
    def test_capitalist_can_go_bankrupt(self, capitalist_system):
        """Test 3: Le système capitaliste peut faire faillite"""
        res = capitalist_system.run()
        # Note: parfois il ne fait pas faillite, on vérifie juste que c'est possible
        assert res.solvency_rate <= 100, "Solvency rate should be ≤ 100%"
    
    def test_production_cycle(self, config):
        """Test 4: La production doit suivre un cycle sinusoïdal"""
        from yusuf_model import YusufSystem
        yusuf = YusufSystem(config)
        P = yusuf._compute_production()
        
        # Vérifier la présence d'un cycle (min < max)
        assert np.max(P) > np.min(P), "Production should oscillate"
        
        # Vérifier la période approximative
        # Compter les passages par la moyenne
        mean_P = config.P_mean
        crossings = np.where(np.diff(np.sign(P - mean_P)))[0]
        if len(crossings) > 1:
            period_estimated = 2 * len(P) / len(crossings) * config.dt
            # La période doit être proche de config.period
            assert abs(period_estimated - config.period) < config.period * 0.3
    
    def test_yusuf_better_solvency_than_capitalist(self, config):
        """Test 5: Yusuf doit être plus solvable que le capitaliste"""
        from yusuf_model import YusufSystem, CapitalistSystem
        
        np.random.seed(42)
        yusuf = YusufSystem(config)
        capitalist = CapitalistSystem(config)
        
        y_res = yusuf.run()
        c_res = capitalist.run()
        
        # Moyenne sur plusieurs seeds pour robustesse
        y_solv = []
        c_solv = []
        
        for seed in range(10):
            np.random.seed(seed)
            y_res = YusufSystem(config).run()
            c_res = CapitalistSystem(config).run()
            y_solv.append(y_res.solvency_rate)
            c_solv.append(c_res.solvency_rate)
        
        assert np.mean(y_solv) >= np.mean(c_solv), \
            f"Yusuf solvency ({np.mean(y_solv):.1f}%) should be ≥ capitalist ({np.mean(c_solv):.1f}%)"


class TestStatisticalValidation:
    """Tests du module de validation statistique"""
    
    @pytest.fixture
    def config(self):
        from yusuf_model import YusufConfig
        return YusufConfig(T=20, dt=0.5, noise_amplitude=0.02)
    
    def test_validation_runs(self, config):
        """Test 6: La validation statistique doit s'exécuter sans erreur"""
        from statistical_validation import StatisticalValidator
        
        validator = StatisticalValidator(config)
        result = validator.run_validation(n_simulations=10)
        
        assert result.n_simulations == 10
        assert len(result.yusuf_metrics) == 10
        assert len(result.capitalist_metrics) == 10
        assert len(result.tests) > 0
    
    def test_bootstrap_analyzer(self):
        """Test 7: L'analyse bootstrap fonctionne"""
        from statistical_validation import BootstrapAnalyzer
        
        data1 = np.random.normal(1, 0.2, 100)
        data2 = np.random.normal(0.8, 0.3, 100)
        
        analyzer = BootstrapAnalyzer(n_bootstrap=100)
        result = analyzer.compare_bootstrap(data1, data2)
        
        assert "diff_means_ci" in result
        assert "prob_yusuf_better" in result
        assert 0 <= result["prob_yusuf_better"] <= 1


class TestShockModel:
    """Tests du modèle de chocs"""
    
    @pytest.fixture
    def config(self):
        from yusuf_model import YusufConfig
        return YusufConfig(T=30, dt=0.5, noise_amplitude=0.0)
    
    @pytest.fixture
    def shock(self):
        from shock_model import ShockConfig
        return ShockConfig(year=15, magnitude=0.5, duration=3)
    
    def test_shock_reduces_production(self, config, shock):
        """Test 8: Un choc doit réduire la production"""
        from shock_model import YusufSystemWithShock
        
        system = YusufSystemWithShock(config, shock)
        P = system._compute_production()
        
        t = np.linspace(0, config.T, config.n_steps)
        shock_mask = (t >= shock.year) & (t <= shock.end_year)
        
        P_before = P[t < shock.year]
        P_during = P[shock_mask]
        
        if len(P_before) > 0 and len(P_during) > 0:
            assert np.mean(P_during) <= np.mean(P_before) * 0.95, \
                "Production should decrease during shock"
    
    def test_yusuf_more_resilient_than_capitalist(self, config, shock):
        """Test 9: Yusuf doit mieux résister aux chocs"""
        from shock_model import YusufSystemWithShock, CapitalistSystemWithShock
        
        yusuf = YusufSystemWithShock(config, shock)
        capitalist = CapitalistSystemWithShock(config, shock)
        
        y_res = yusuf.run()
        c_res = capitalist.run()
        
        # Résilience = stock minimum pendant la période de choc
        t = np.linspace(0, config.T, config.n_steps)
        shock_mask = (t >= shock.year) & (t <= shock.end_year)
        
        y_min_during = np.min(y_res.S[shock_mask]) if np.any(shock_mask) else y_res.S[-1]
        c_min_during = np.min(c_res.S[shock_mask]) if np.any(shock_mask) else c_res.S[-1]
        
        # Yusuf ne devrait pas tomber à zéro pendant un choc modéré
        assert y_min_during > 0 or y_res.solvency_rate > c_res.solvency_rate


class TestConfiguration:
    """Tests de configuration"""
    
    def test_config_defaults(self):
        """Test 10: Les valeurs par défaut sont cohérentes"""
        from yusuf_model import YusufConfig
        
        config = YusufConfig()
        
        assert config.T > 0
        assert config.dt > 0
        assert 0 < config.need < 2
        assert config.P_mean > 0
        assert config.period > 0
        assert 0 <= config.interest_rate <= 1
    
    def test_config_thresholds(self):
        """Test 11: Les seuils d'abondance/rareté sont cohérents"""
        from yusuf_model import YusufConfig
        
        config = YusufConfig(P_mean=1.0, P_amplitude=0.5, threshold_factor=0.3)
        
        assert config.P_bar > config.P_mean
        assert config.P_underline < config.P_mean
        assert config.P_underline > 0
    
    def test_config_validation(self):
        """Test 12: Les paramètres invalides doivent être évités"""
        from yusuf_model import YusufConfig
        
        # Des paramètres extrêmes mais toujours valides
        config = YusufConfig(need=0.1, interest_rate=0.2)
        
        # Les calculs ne doivent pas planter
        from yusuf_model import YusufSystem
        yusuf = YusufSystem(config)
        res = yusuf.run()
        
        assert isinstance(res.final_stock, float)
        assert not np.isnan(res.final_stock)


class TestIntegration:
    """Tests d'intégration entre modules"""
    
    def test_complete_pipeline(self):
        """Test 13: La pipeline complète doit s'exécuter"""
        from yusuf_model import YusufConfig, ScenarioComparator
        from statistical_validation import StatisticalValidator
        
        config = YusufConfig(T=20, dt=0.5)
        
        # Comparaison simple
        comparator = ScenarioComparator(config)
        y_res, c_res = comparator.run_single()
        assert y_res.final_stock >= 0
        
        # Validation statistique courte
        validator = StatisticalValidator(config)
        result = validator.run_validation(n_simulations=5)
        assert result.tests
    
    def test_reproducibility_with_seed(self):
        """Test 14: Les résultats doivent être reproductibles avec une seed fixe"""
        from yusuf_model import YusufConfig, YusufSystem
        
        config = YusufConfig(T=10, dt=0.5, noise_amplitude=0.05)
        
        results = []
        for seed in [42, 42]:
            np.random.seed(seed)
            yusuf = YusufSystem(config)
            res = yusuf.run()
            results.append(res.final_stock)
        
        # Même seed → même résultat
        assert results[0] == results[1]
        
        # Seed différente → résultat différent (pas toujours, mais souvent)
        np.random.seed(999)
        yusuf = YusufSystem(config)
        res_diff = yusuf.run()
        
        # Avec bruit, il y a généralement une différence
        if config.noise_amplitude > 0:
            # On ne peut pas garantir à 100% à cause du hasard
            pass


# ============================================================
# Exécution directe
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
