"""
Optimisateur de Stratégies pour Backtest
======================================

Ce module optimise automatiquement les paramètres de la stratégie
de trading en utilisant des techniques d'optimisation pour maximiser
les performances historiques.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from dataclasses import dataclass
from .backtest_engine import BacktestEngine
from .config import Config
import optuna
from scipy.optimize import differential_evolution
import warnings

warnings.filterwarnings('ignore')

@dataclass
class OptimizationResult:
    """Résultat d'une optimisation"""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict]
    optimization_metric: str
    total_trials: int
    elapsed_time: float

class ParameterSpace:
    """Définition de l'espace des paramètres à optimiser"""
    
    def __init__(self):
        self.parameters = {}
    
    def add_parameter(self, name: str, param_type: str, **kwargs):
        """
        Ajouter un paramètre à optimiser
        
        Args:
            name: Nom du paramètre
            param_type: Type ('int', 'float', 'categorical')
            **kwargs: Arguments selon le type (min, max, values, etc.)
        """
        self.parameters[name] = {
            'type': param_type,
            **kwargs
        }
    
    def get_default_strategy_space(self):
        """Définir l'espace par défaut pour les paramètres de stratégie"""
        # Paramètres RSI
        self.add_parameter('RSI_PERIOD', 'int', low=10, high=30)
        self.add_parameter('RSI_OVERSOLD', 'int', low=20, high=35)
        self.add_parameter('RSI_OVERBOUGHT', 'int', low=65, high=80)
        
        # Paramètres MACD
        self.add_parameter('MACD_FAST', 'int', low=8, high=15)
        self.add_parameter('MACD_SLOW', 'int', low=20, high=30)
        self.add_parameter('MACD_SIGNAL', 'int', low=6, high=12)
        
        # Paramètres Bollinger Bands
        self.add_parameter('BOLLINGER_PERIOD', 'int', low=15, high=25)
        self.add_parameter('BOLLINGER_STD', 'float', low=1.5, high=2.5)
        
        # Paramètres Moving Averages
        self.add_parameter('MA_FAST', 'int', low=5, high=15)
        self.add_parameter('MA_SLOW', 'int', low=20, high=50)
        
        # Paramètres de gestion des risques
        self.add_parameter('STOP_LOSS_PERCENTAGE', 'float', low=2.0, high=10.0)
        self.add_parameter('TAKE_PROFIT_PERCENTAGE', 'float', low=3.0, high=15.0)
        
        # Paramètres de trading
        self.add_parameter('MIN_SIGNAL_CONFIDENCE', 'float', low=0.3, high=0.8)
        
        return self

class BacktestOptimizer:
    """Optimisateur de paramètres de stratégie"""
    
    def __init__(self, pairs: List[str], start_date: datetime, end_date: datetime,
                 initial_capital: float = 10000, optimization_metric: str = 'total_return'):
        """
        Initialiser l'optimisateur
        
        Args:
            pairs: Liste des paires à trader
            start_date: Date de début du backtest
            end_date: Date de fin du backtest
            initial_capital: Capital initial
            optimization_metric: Métrique à optimiser
        """
        self.pairs = pairs
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.optimization_metric = optimization_metric
        
        # Métriques disponibles
        self.available_metrics = [
            'total_return',
            'sharpe_ratio', 
            'profit_factor',
            'win_rate',
            'max_drawdown_adjusted',  # 1 / max_drawdown
            'calmar_ratio',  # total_return / max_drawdown
            'sortino_ratio'
        ]
        
        if optimization_metric not in self.available_metrics:
            raise ValueError(f"Métrique {optimization_metric} non supportée. Disponibles: {self.available_metrics}")
        
        self.results_cache = {}
        logging.info(f"Optimisateur initialisé: métrique={optimization_metric}")
    
    def optimize_with_optuna(self, parameter_space: ParameterSpace, n_trials: int = 100,
                           n_jobs: int = 1, study_name: str = "strategy_optimization") -> OptimizationResult:
        """
        Optimisation avec Optuna (Bayesian optimization)
        
        Args:
            parameter_space: Espace des paramètres
            n_trials: Nombre d'essais
            n_jobs: Nombre de processus parallèles
            study_name: Nom de l'étude
            
        Returns:
            Résultats de l'optimisation
        """
        logging.info(f"Démarrage de l'optimisation Optuna: {n_trials} essais")
        start_time = datetime.now()
        
        # Créer l'étude Optuna
        study = optuna.create_study(
            direction='maximize' if self.optimization_metric != 'max_drawdown' else 'minimize',
            study_name=study_name
        )
        
        # Fonction objective
        def objective(trial):
            params = {}
            for param_name, param_config in parameter_space.parameters.items():
                if param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config['choices']
                    )
            
            return self._evaluate_parameters(params)
        
        # Lancer l'optimisation
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        # Compiler les résultats
        all_results = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                result = {
                    'params': trial.params,
                    'score': trial.value,
                    'trial_number': trial.number
                }
                all_results.append(result)
        
        optimization_result = OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            all_results=all_results,
            optimization_metric=self.optimization_metric,
            total_trials=len(all_results),
            elapsed_time=elapsed_time
        )
        
        logging.info(f"Optimisation terminée: meilleur score = {study.best_value:.4f}")
        return optimization_result
    
    def optimize_with_grid_search(self, parameter_space: ParameterSpace, 
                                max_combinations: int = 1000) -> OptimizationResult:
        """
        Optimisation par recherche grille
        
        Args:
            parameter_space: Espace des paramètres
            max_combinations: Nombre maximum de combinaisons
            
        Returns:
            Résultats de l'optimisation
        """
        logging.info("Démarrage de l'optimisation par recherche grille")
        start_time = datetime.now()
        
        # Générer toutes les combinaisons
        param_names = list(parameter_space.parameters.keys())
        param_values = []
        
        for param_name in param_names:
            param_config = parameter_space.parameters[param_name]
            
            if param_config['type'] == 'int':
                values = list(range(param_config['low'], param_config['high'] + 1, 
                                  max(1, (param_config['high'] - param_config['low']) // 10)))
            elif param_config['type'] == 'float':
                values = np.linspace(param_config['low'], param_config['high'], 10)
            elif param_config['type'] == 'categorical':
                values = param_config['choices']
            
            param_values.append(values)
        
        # Générer les combinaisons
        all_combinations = list(product(*param_values))
        
        # Limiter le nombre de combinaisons
        if len(all_combinations) > max_combinations:
            np.random.seed(42)
            indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
            all_combinations = [all_combinations[i] for i in indices]
        
        logging.info(f"Évaluation de {len(all_combinations)} combinaisons")
        
        # Évaluer toutes les combinaisons
        all_results = []
        best_score = float('-inf') if self.optimization_metric != 'max_drawdown' else float('inf')
        best_params = None
        
        for i, combination in enumerate(all_combinations):
            if i % 50 == 0:
                progress = (i / len(all_combinations)) * 100
                logging.info(f"Progression: {progress:.1f}%")
            
            params = dict(zip(param_names, combination))
            score = self._evaluate_parameters(params)
            
            result = {
                'params': params,
                'score': score,
                'trial_number': i
            }
            all_results.append(result)
            
            # Mettre à jour le meilleur résultat
            if self.optimization_metric != 'max_drawdown':
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            else:
                if score < best_score:
                    best_score = score
                    best_params = params.copy()
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        optimization_result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_metric=self.optimization_metric,
            total_trials=len(all_results),
            elapsed_time=elapsed_time
        )
        
        logging.info(f"Optimisation terminée: meilleur score = {best_score:.4f}")
        return optimization_result
    
    def optimize_with_differential_evolution(self, parameter_space: ParameterSpace,
                                           maxiter: int = 100) -> OptimizationResult:
        """
        Optimisation par évolution différentielle
        
        Args:
            parameter_space: Espace des paramètres
            maxiter: Nombre maximum d'itérations
            
        Returns:
            Résultats de l'optimisation
        """
        logging.info("Démarrage de l'optimisation par évolution différentielle")
        start_time = datetime.now()
        
        # Préparer les bornes et les paramètres
        bounds = []
        param_names = []
        param_types = []
        
        for param_name, param_config in parameter_space.parameters.items():
            if param_config['type'] in ['int', 'float']:
                bounds.append((param_config['low'], param_config['high']))
                param_names.append(param_name)
                param_types.append(param_config['type'])
        
        if not bounds:
            raise ValueError("Aucun paramètre numérique trouvé pour l'optimisation")
        
        all_results = []
        
        def objective(x):
            params = {}
            for i, (name, param_type) in enumerate(zip(param_names, param_types)):
                if param_type == 'int':
                    params[name] = int(round(x[i]))
                else:
                    params[name] = x[i]
            
            score = self._evaluate_parameters(params)
            
            # Stocker le résultat
            result = {
                'params': params.copy(),
                'score': score,
                'trial_number': len(all_results)
            }
            all_results.append(result)
            
            # Retourner l'opposé si on maximise
            return -score if self.optimization_metric != 'max_drawdown' else score
        
        # Lancer l'optimisation
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=maxiter,
            seed=42,
            atol=1e-6,
            tol=1e-6
        )
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        # Construire les meilleurs paramètres
        best_params = {}
        for i, (name, param_type) in enumerate(zip(param_names, param_types)):
            if param_type == 'int':
                best_params[name] = int(round(result.x[i]))
            else:
                best_params[name] = result.x[i]
        
        best_score = -result.fun if self.optimization_metric != 'max_drawdown' else result.fun
        
        optimization_result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            optimization_metric=self.optimization_metric,
            total_trials=len(all_results),
            elapsed_time=elapsed_time
        )
        
        logging.info(f"Optimisation terminée: meilleur score = {best_score:.4f}")
        return optimization_result
    
    def _evaluate_parameters(self, params: Dict[str, Any]) -> float:
        """
        Évaluer un ensemble de paramètres
        
        Args:
            params: Paramètres à évaluer
            
        Returns:
            Score de performance
        """
        # Cache des résultats pour éviter les recalculs
        params_key = json.dumps(params, sort_keys=True)
        if params_key in self.results_cache:
            return self.results_cache[params_key]
        
        try:
            # Créer une configuration temporaire
            original_config = {}
            for param_name, param_value in params.items():
                if hasattr(Config, param_name):
                    original_config[param_name] = getattr(Config, param_name)
                    setattr(Config, param_name, param_value)
            
            # Exécuter le backtest
            engine = BacktestEngine(self.initial_capital)
            results = engine.run_backtest(
                pairs=self.pairs,
                start_date=self.start_date,
                end_date=self.end_date,
                interval=60  # 1 heure par défaut
            )
            
            # Restaurer la configuration originale
            for param_name, original_value in original_config.items():
                setattr(Config, param_name, original_value)
            
            if results is None or 'summary' not in results:
                score = float('-inf')
            else:
                score = self._calculate_score(results)
            
            # Mettre en cache
            self.results_cache[params_key] = score
            return score
            
        except Exception as e:
            logging.warning(f"Erreur lors de l'évaluation des paramètres {params}: {e}")
            return float('-inf')
    
    def _calculate_score(self, results: Dict) -> float:
        """Calculer le score basé sur la métrique d'optimisation"""
        summary = results['summary']
        trades_stats = results['trades_stats']
        
        if self.optimization_metric == 'total_return':
            return summary['total_return']
        
        elif self.optimization_metric == 'sharpe_ratio':
            return summary['sharpe_ratio']
        
        elif self.optimization_metric == 'profit_factor':
            return trades_stats['profit_factor'] if trades_stats['profit_factor'] != float('inf') else 10.0
        
        elif self.optimization_metric == 'win_rate':
            return trades_stats['win_rate']
        
        elif self.optimization_metric == 'max_drawdown_adjusted':
            # Inverser le drawdown (plus c'est faible, mieux c'est)
            return 1.0 / max(summary['max_drawdown'], 0.1)
        
        elif self.optimization_metric == 'calmar_ratio':
            # Ratio de Calmar: rendement annuel / drawdown maximum
            return summary['total_return'] / max(summary['max_drawdown'], 0.1)
        
        elif self.optimization_metric == 'sortino_ratio':
            # Approximation du ratio de Sortino
            if len(results['equity_curve']) > 1:
                equity_series = pd.Series([eq['equity'] for eq in results['equity_curve']])
                returns = equity_series.pct_change().dropna()
                downside_returns = returns[returns < 0]
                
                if len(downside_returns) > 0:
                    downside_deviation = downside_returns.std()
                    return returns.mean() / downside_deviation * np.sqrt(252)
            
            return 0.0
        
        else:
            return summary['total_return']  # Par défaut
    
    def run_walk_forward_analysis(self, parameter_space: ParameterSpace, 
                                window_months: int = 6, step_months: int = 1) -> Dict:
        """
        Analyse walk-forward pour tester la robustesse
        
        Args:
            parameter_space: Espace des paramètres
            window_months: Taille de la fenêtre d'optimisation (mois)
            step_months: Pas de progression (mois)
            
        Returns:
            Résultats de l'analyse walk-forward
        """
        logging.info("Démarrage de l'analyse walk-forward")
        
        results = []
        current_date = self.start_date
        
        while current_date < self.end_date - timedelta(days=window_months * 30):
            # Définir les périodes d'optimisation et de test
            opt_start = current_date
            opt_end = current_date + timedelta(days=window_months * 30)
            test_start = opt_end
            test_end = min(test_start + timedelta(days=step_months * 30), self.end_date)
            
            logging.info(f"Optimisation: {opt_start.date()} à {opt_end.date()}")
            logging.info(f"Test: {test_start.date()} à {test_end.date()}")
            
            # Optimiser sur la période d'optimisation
            temp_optimizer = BacktestOptimizer(
                self.pairs, opt_start, opt_end, 
                self.initial_capital, self.optimization_metric
            )
            
            optimization_result = temp_optimizer.optimize_with_optuna(
                parameter_space, n_trials=50
            )
            
            # Tester sur la période de test
            test_optimizer = BacktestOptimizer(
                self.pairs, test_start, test_end, 
                self.initial_capital, self.optimization_metric
            )
            
            test_score = test_optimizer._evaluate_parameters(optimization_result.best_params)
            
            results.append({
                'optimization_period': (opt_start, opt_end),
                'test_period': (test_start, test_end),
                'best_params': optimization_result.best_params,
                'optimization_score': optimization_result.best_score,
                'test_score': test_score
            })
            
            current_date += timedelta(days=step_months * 30)
        
        # Calculer les statistiques de l'analyse
        test_scores = [r['test_score'] for r in results]
        opt_scores = [r['optimization_score'] for r in results]
        
        analysis_summary = {
            'periods_analyzed': len(results),
            'avg_test_score': np.mean(test_scores),
            'std_test_score': np.std(test_scores),
            'avg_opt_score': np.mean(opt_scores),
            'consistency': np.corrcoef(opt_scores, test_scores)[0, 1] if len(test_scores) > 1 else 0,
            'results': results
        }
        
        logging.info(f"Analyse walk-forward terminée: {len(results)} périodes")
        logging.info(f"Score test moyen: {analysis_summary['avg_test_score']:.4f}")
        logging.info(f"Cohérence: {analysis_summary['consistency']:.4f}")
        
        return analysis_summary
    
    def save_optimization_results(self, results: OptimizationResult, 
                                filename: str = "optimization_results.json"):
        """Sauvegarder les résultats d'optimisation"""
        output_data = {
            'best_params': results.best_params,
            'best_score': results.best_score,
            'optimization_metric': results.optimization_metric,
            'total_trials': results.total_trials,
            'elapsed_time': results.elapsed_time,
            'timestamp': datetime.now().isoformat(),
            'top_10_results': sorted(results.all_results, 
                                   key=lambda x: x['score'], reverse=True)[:10]
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logging.info(f"Résultats sauvegardés dans {filename}")
    
    def analyze_parameter_sensitivity(self, results: OptimizationResult) -> Dict:
        """
        Analyser la sensibilité des paramètres
        
        Args:
            results: Résultats de l'optimisation
            
        Returns:
            Analyse de sensibilité
        """
        if len(results.all_results) < 10:
            return {}
        
        # Convertir en DataFrame
        df = pd.DataFrame(results.all_results)
        
        # Extraire les paramètres
        param_data = pd.json_normalize(df['params'])
        param_data['score'] = df['score']
        
        sensitivity_analysis = {}
        
        for param in param_data.columns:
            if param == 'score':
                continue
            
            # Calculer la corrélation avec le score
            correlation = param_data[param].corr(param_data['score'])
            
            # Analyser la distribution des valeurs
            param_values = param_data[param]
            
            sensitivity_analysis[param] = {
                'correlation_with_score': correlation,
                'mean': param_values.mean(),
                'std': param_values.std(),
                'min': param_values.min(),
                'max': param_values.max(),
                'best_value': results.best_params.get(param)
            }
        
        return sensitivity_analysis