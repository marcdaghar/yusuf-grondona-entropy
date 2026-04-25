#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical Validation Module
Monte Carlo, t-tests, Mann-Whitney, Bootstrap, Confidence Intervals

Author: Marc Daghar
Licence: CC BY-SA 4.0
Mention: Free Dr Aafia Siddiqui !
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import json

from yusuf_model import YusufConfig, YusufSystem, CapitalistSystem, SimulationResult


@dataclass
class StatisticalTestResult:
    """Result of a statistical test"""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    interpretation: str
    effect_size: float = None


class StatisticalValidator:
    """Statistical validator comparing Yusuf and Capitalist systems"""
    
    def __init__(self, config: YusufConfig = None, seed: int = 42):
        self.config = config or YusufConfig()
        self.seed = seed
        np.random.seed(seed)
    
    def run_simulations(self, n_simulations: int = 100) -> Tuple[List[SimulationResult], List[SimulationResult]]:
        """Run N simulations of both systems"""
        yusuf_results = []
        capitalist_results = []
        
        for i in range(n_simulations):
            np.random.seed(self.seed + i)
            yusuf = YusufSystem(self.config)
            capitalist = CapitalistSystem(self.config)
            yusuf_results.append(yusuf.run())
            capitalist_results.append(capitalist.run())
        
        return yusuf_results, capitalist_results
    
    def extract_metrics(self, results: List[SimulationResult]) -> List[Dict[str, float]]:
        """Extract key metrics from results"""
        metrics = []
        for res in results:
            metrics.append({
                "final_stock": res.final_stock,
                "mean_consumption": res.mean_consumption,
                "consumption_volatility": res.consumption_volatility,
                "solvency_rate": res.solvency_rate,
                "coverage_ratio_mean": float(np.mean(res.coverage_ratio))
            })
        return metrics
    
    def test_normality(self, data: np.ndarray) -> Tuple[float, bool]:
        """Shapiro-Wilk normality test"""
        if len(data) < 3:
            return 0.0, False
        statistic, p_value = stats.shapiro(data)
        return p_value, p_value > 0.05
    
    def test_difference(self, yusuf_data: np.ndarray, cap_data: np.ndarray, 
                        metric_name: str) -> StatisticalTestResult:
        """Test difference between systems"""
        _, yusuf_normal = self.test_normality(yusuf_data)
        _, cap_normal = self.test_normality(cap_data)
        
        if yusuf_normal and cap_normal:
            statistic, p_value = stats.ttest_ind(yusuf_data, cap_data)
            test_name = f"t-test ({metric_name})"
            pooled_std = np.sqrt((np.var(yusuf_data) + np.var(cap_data)) / 2)
            effect_size = (np.mean(yusuf_data) - np.mean(cap_data)) / pooled_std if pooled_std > 0 else 0
        else:
            statistic, p_value = stats.mannwhitneyu(yusuf_data, cap_data, alternative='two-sided')
            test_name = f"Mann-Whitney U ({metric_name})"
            effect_size = statistic / (len(yusuf_data) * len(cap_data)) - 0.5
        
        mean_diff = np.mean(yusuf_data) - np.mean(cap_data)
        if p_value < 0.05:
            if mean_diff > 0:
                interpretation = f"Yusuf > Capitalist (diff={mean_diff:.3f})"
            else:
                interpretation = f"Capitalist > Yusuf (diff={-mean_diff:.3f})"
        else:
            interpretation = "No significant difference"
        
        return StatisticalTestResult(
            test_name=test_name, statistic=statistic, p_value=p_value,
            significant=p_value < 0.05, interpretation=interpretation, effect_size=abs(effect_size)
        )
    
    def compute_confidence_intervals(self, yusuf_data: np.ndarray, cap_data: np.ndarray,
                                      confidence: float = 0.95) -> Dict[str, Any]:
        """Compute confidence intervals"""
        z = stats.norm.ppf((1 + confidence) / 2)
        
        def ci(data):
            mean = np.mean(data)
            std = np.std(data)
            margin = z * std / np.sqrt(len(data))
            return (mean - margin, mean + margin), mean, std
        
        y_ci, y_mean, y_std = ci(yusuf_data)
        c_ci, c_mean, c_std = ci(cap_data)
        
        return {
            "yusuf": {"mean": y_mean, "ci_lower": y_ci[0], "ci_upper": y_ci[1], "std": y_std},
            "capitalist": {"mean": c_mean, "ci_lower": c_ci[0], "ci_upper": c_ci[1], "std": c_std}
        }
    
    def run_validation(self, n_simulations: int = 100) -> Dict[str, Any]:
        """Run complete statistical validation"""
        yusuf_results, capitalist_results = self.run_simulations(n_simulations)
        
        yusuf_metrics = self.extract_metrics(yusuf_results)
        capitalist_metrics = self.extract_metrics(capitalist_results)
        
        metrics_names = ["final_stock", "mean_consumption", "consumption_volatility", "solvency_rate"]
        yusuf_arrays = {name: np.array([m[name] for m in yusuf_metrics]) for name in metrics_names}
        cap_arrays = {name: np.array([m[name] for m in capitalist_metrics]) for name in metrics_names}
        
        tests = []
        for name in metrics_names:
            tests.append(self.test_difference(yusuf_arrays[name], cap_arrays[name], name))
        
        ci = self.compute_confidence_intervals(yusuf_arrays["final_stock"], cap_arrays["final_stock"])
        
        return {
            "n_simulations": n_simulations,
            "tests": [t.__dict__ for t in tests],
            "confidence_intervals": ci,
            "yusuf_mean_stock": float(np.mean(yusuf_arrays["final_stock"])),
            "capitalist_mean_stock": float(np.mean(cap_arrays["final_stock"])),
            "yusuf_solvency": float(np.mean(yusuf_arrays["solvency_rate"])),
            "capitalist_solvency": float(np.mean(cap_arrays["solvency_rate"]))
        }


class BootstrapAnalyzer:
    """Bootstrap analysis for robustness"""
    
    def __init__(self, n_bootstrap: int = 1000, confidence: float = 0.95):
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
    
    def compare(self, yusuf_data: np.ndarray, cap_data: np.ndarray) -> Dict[str, Any]:
        """Compare systems using bootstrap"""
        n = min(len(yusuf_data), len(cap_data))
        diff_means = []
        
        for _ in range(self.n_bootstrap):
            y_sample = np.random.choice(yusuf_data, size=n, replace=True)
            c_sample = np.random.choice(cap_data, size=n, replace=True)
            diff_means.append(np.mean(y_sample) - np.mean(c_sample))
        
        ci_lower = np.percentile(diff_means, 2.5)
        ci_upper = np.percentile(diff_means, 97.5)
        prob_yusuf_better = np.mean([d > 0 for d in diff_means])
        
        return {
            "diff_means_ci": (ci_lower, ci_upper),
            "prob_yusuf_better": prob_yusuf_better,
            "significant_95": ci_lower > 0 or ci_upper < 0
        }


if __name__ == "__main__":
    validator = StatisticalValidator()
    results = validator.run_validation(n_simulations=50)
    
    print("=" * 60)
    print("STATISTICAL VALIDATION")
    print("=" * 60)
    print(f"Based on {results['n_simulations']} simulations")
    print(f"Yusuf mean stock: {results['yusuf_mean_stock']:.3f}")
    print(f"Capitalist mean stock: {results['capitalist_mean_stock']:.3f}")
    print(f"Yusuf solvency: {results['yusuf_solvency']:.1f}%")
    print(f"Capitalist solvency: {results['capitalist_solvency']:.1f}%")
    print("\nStatistical tests:")

    def ergodicity_test(simulation_runs, time_steps):
    # Ensemble average: across runs at final time
    ensemble_mean = np.mean([run.wealth[-1] for run in simulation_runs])
    
    # Time average: within a single long run
    single_run = simulation_runs[0]
    time_mean = np.mean(single_run.wealth)
    
    divergence = abs(ensemble_mean - time_mean) / ensemble_mean
    
    if divergence > threshold:
        print(f"System is non-ergodic (divergence = {divergence})")
        print("Expected value reasoning fails. Time-average growth matters.")
    
    return divergence
    for test in results['tests']:
        sig = "✓" if test['significant'] else "✗"
        print(f"  {sig} {test['test_name']}: p={test['p_value']:.4f} ({test['interpretation']})")
