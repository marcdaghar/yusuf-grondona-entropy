#!/usr/bin/env python3
"""Tests unitaires pour le modèle Yusuf - Licence CC BY-SA"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestYusufModel:
    """Tests du modèle Yusuf"""
    
    @pytest.fixture
    def config(self):
        from yusuf_model import YusufConfig
        return YusufConfig(T=10, dt=0.5, noise_amplitude=0.0)
    
    def test_yusuf_solvency_rate(self, config):
        """Le système Yusuf doit être 100% solvable"""
        from yusuf_model import YusufSystem
        yusuf = YusufSystem(config)
        res = yusuf.run()
        assert res.solvency_rate == 100.0
    
    def test_yusuf_no_negative_stock(self, config):
        """Le stock ne doit jamais être négatif"""
        from yusuf_model import YusufSystem
        yusuf = YusufSystem(config)
        res = yusuf.run()
        assert np.all(res.S >= -1e-6)
    
    def test_yusuf_better_than_capitalist(self, config):
        """Yusuf doit être plus résilient que le capitaliste"""
        from yusuf_model import YusufSystem, CapitalistSystem
        
        y_solv = []
        c_solv = []
        
        for seed in range(5):
            np.random.seed(seed)
            y_res = YusufSystem(config).run()
            c_res = CapitalistSystem(config).run()
            y_solv.append(y_res.solvency_rate)
            c_solv.append(c_res.solvency_rate)
        
        assert np.mean(y_solv) >= np.mean(c_solv)
    
    def test_config_thresholds(self):
        """Les seuils d'abondance/rareté sont cohérents"""
        from yusuf_model import YusufConfig
        config = YusufConfig(P_mean=1.0, P_amplitude=0.5, threshold_factor=0.3)
        assert config.P_bar > config.P_mean
        assert config.P_underline < config.P_mean
    
    def test_import_all_modules(self):
        """Tous les modules s'importent correctement"""
        from yusuf_model import YusufConfig, YusufSystem, CapitalistSystem
        from grondona_crd import GrondonaCRD
        from neurocognitive_agents import NeurocognitiveAgent
        from ricci_flow import RicciFlow
        from statistical_validation import StatisticalValidator
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
