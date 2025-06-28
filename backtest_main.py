#!/usr/bin/env python3
"""
Script Principal de Backtest pour Bot de Trading Crypto
=====================================================

Ce script lance le système de backtest complet avec :
- Téléchargement des données de Kraken
- Simulation de trading avec la stratégie existante
- Visualisation des performances
- Optimisation automatique des paramètres

Usage:
    python backtest_main.py --pairs BTCEUR ETHEUR --days 180 --capital 10000
    python backtest_main.py --optimize --method optuna --trials 100
    python backtest_main.py --walk-forward --window 6 --step 1
"""

import argparse
import sys
import os
import logging
from datetime import datetime, timedelta
from typing import List
import pandas as pd

# Ajouter le répertoire src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.backtest_engine import BacktestEngine
from src.backtest_visualization import BacktestVisualizer
from src.backtest_optimizer import BacktestOptimizer, ParameterSpace
from src.config import Config

def setup_logging(debug: bool = False):
    """Configurer le logging"""
    level = logging.DEBUG if debug else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('backtest.log'),
            logging.StreamHandler()
        ]
    )

def print_banner():
    """Afficher la bannière du backtest"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    SYSTÈME DE BACKTEST                       ║
║                  Bot de Trading Crypto                       ║
║                                                              ║
║  📊 Téléchargement de données Kraken                         ║
║  🔄 Simulation de stratégie de trading                       ║
║  📈 Visualisation des performances                           ║
║  🎯 Optimisation automatique des paramètres                  ║
╚══════════════════════════════════════════════════════════════╝
    """)

def run_basic_backtest(pairs: List[str], days: int, capital: float, 
                      interval: int = 60, output_dir: str = "backtest_reports") -> dict:
    """
    Exécuter un backtest de base
    
    Args:
        pairs: Liste des paires à trader
        days: Nombre de jours d'historique
        capital: Capital initial
        interval: Intervalle en minutes
        output_dir: Répertoire de sortie
        
    Returns:
        Résultats du backtest
    """
    # Calculer les dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    logging.info(f"=== BACKTEST DE BASE ===")
    logging.info(f"Paires: {pairs}")
    logging.info(f"Période: {start_date.date()} à {end_date.date()}")
    logging.info(f"Capital initial: {capital}€")
    logging.info(f"Intervalle: {interval} minutes")
    
    # Créer le moteur de backtest
    engine = BacktestEngine(initial_capital=capital)
    
    # Exécuter le backtest
    results = engine.run_backtest(
        pairs=pairs,
        start_date=start_date,
        end_date=end_date,
        interval=interval
    )
    
    if results is None:
        logging.error("Échec du backtest - résultats None")
        # Créer un résultat d'erreur par défaut
        results = {
            'error': 'Échec du backtest - résultats None',
            'summary': {
                'initial_capital': capital,
                'final_capital': capital,
                'total_return': 0.0,
                'total_profit_loss': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
            },
            'trades_stats': {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'total_fees': 0.0,
            },
            'time_stats': {
                'avg_hold_time_hours': 0.0,
                'max_hold_time_hours': 0.0,
                'min_hold_time_hours': 0.0,
            },
            'equity_curve': [],
            'trades': [],
            'trades_df': pd.DataFrame(columns=[
                'timestamp', 'pair', 'action', 'price', 'volume', 
                'profit_loss', 'profit_loss_percent', 'hold_duration', 'fees'
            ])
        }
    
    # Générer les visualisations
    try:
        visualizer = BacktestVisualizer(results, output_dir)
        
        # Créer le rapport complet
        html_report = visualizer.generate_complete_report(
            title=f"Backtest {', '.join(pairs)} - {days} jours"
        )
        
        # Créer le rapport de synthèse
        summary_report = visualizer.generate_summary_report()
        
        logging.info(f"📊 Rapport HTML: {html_report}")
        logging.info(f"📄 Rapport synthèse: {summary_report}")
        
    except Exception as e:
        logging.error(f"Erreur lors de la génération des visualisations: {e}")
        if 'error' not in results:
            results['error'] = f"Erreur de visualisation: {e}"
    
    return results

def run_optimization(pairs: List[str], days: int, capital: float,
                    method: str = 'optuna', trials: int = 100, 
                    metric: str = 'total_return') -> dict:
    """
    Exécuter l'optimisation des paramètres
    
    Args:
        pairs: Liste des paires à trader
        days: Nombre de jours d'historique
        capital: Capital initial
        method: Méthode d'optimisation ('optuna', 'grid', 'differential')
        trials: Nombre d'essais
        metric: Métrique à optimiser
        
    Returns:
        Résultats de l'optimisation
    """
    # Calculer les dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    logging.info(f"=== OPTIMISATION DES PARAMÈTRES ===")
    logging.info(f"Méthode: {method}")
    logging.info(f"Essais: {trials}")
    logging.info(f"Métrique: {metric}")
    
    # Créer l'optimisateur
    optimizer = BacktestOptimizer(
        pairs=pairs,
        start_date=start_date,
        end_date=end_date,
        initial_capital=capital,
        optimization_metric=metric
    )
    
    # Définir l'espace des paramètres
    param_space = ParameterSpace().get_default_strategy_space()
    
    # Lancer l'optimisation selon la méthode
    if method == 'optuna':
        results = optimizer.optimize_with_optuna(param_space, n_trials=trials)
    elif method == 'grid':
        results = optimizer.optimize_with_grid_search(param_space, max_combinations=trials)
    elif method == 'differential':
        results = optimizer.optimize_with_differential_evolution(param_space, maxiter=trials)
    else:
        raise ValueError(f"Méthode d'optimisation non supportée: {method}")
    
    # Sauvegarder les résultats
    optimizer.save_optimization_results(results, f"optimization_{method}_{metric}.json")
    
    # Analyser la sensibilité des paramètres
    sensitivity = optimizer.analyze_parameter_sensitivity(results)
    
    # Afficher les résultats
    print(f"\n=== RÉSULTATS DE L'OPTIMISATION ===")
    print(f"Meilleur score ({metric}): {results.best_score:.4f}")
    print(f"Temps d'exécution: {results.elapsed_time:.1f}s")
    print(f"Essais réalisés: {results.total_trials}")
    
    print(f"\n=== MEILLEURS PARAMÈTRES ===")
    for param, value in results.best_params.items():
        print(f"{param}: {value}")
    
    print(f"\n=== ANALYSE DE SENSIBILITÉ ===")
    for param, analysis in sensitivity.items():
        corr = analysis['correlation_with_score']
        print(f"{param}: corrélation = {corr:.3f}, valeur optimale = {analysis['best_value']}")
    
    # Tester les paramètres optimisés
    logging.info("Test des paramètres optimisés...")
    
    # Appliquer temporairement les meilleurs paramètres
    original_params = {}
    for param_name, param_value in results.best_params.items():
        if hasattr(Config, param_name):
            original_params[param_name] = getattr(Config, param_name)
            setattr(Config, param_name, param_value)
    
    # Exécuter un backtest avec les paramètres optimisés
    optimized_results = run_basic_backtest(
        pairs, days, capital, output_dir="backtest_optimized"
    )
    
    # Restaurer les paramètres originaux
    for param_name, original_value in original_params.items():
        setattr(Config, param_name, original_value)
    
    return {
        'optimization_results': results,
        'sensitivity_analysis': sensitivity,
        'backtest_results': optimized_results
    }

def run_walk_forward_analysis(pairs: List[str], days: int, capital: float,
                             window_months: int = 6, step_months: int = 1) -> dict:
    """
    Exécuter une analyse walk-forward
    
    Args:
        pairs: Liste des paires à trader
        days: Nombre total de jours d'historique
        capital: Capital initial
        window_months: Taille de la fenêtre d'optimisation
        step_months: Pas de progression
        
    Returns:
        Résultats de l'analyse walk-forward
    """
    # Calculer les dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    logging.info(f"=== ANALYSE WALK-FORWARD ===")
    logging.info(f"Fenêtre d'optimisation: {window_months} mois")
    logging.info(f"Pas de progression: {step_months} mois")
    
    # Créer l'optimisateur
    optimizer = BacktestOptimizer(
        pairs=pairs,
        start_date=start_date,
        end_date=end_date,
        initial_capital=capital,
        optimization_metric='total_return'
    )
    
    # Définir l'espace des paramètres
    param_space = ParameterSpace().get_default_strategy_space()
    
    # Lancer l'analyse walk-forward
    wf_results = optimizer.run_walk_forward_analysis(
        param_space, window_months, step_months
    )
    
    # Afficher les résultats
    print(f"\n=== RÉSULTATS WALK-FORWARD ===")
    print(f"Périodes analysées: {wf_results['periods_analyzed']}")
    print(f"Score test moyen: {wf_results['avg_test_score']:.4f}")
    print(f"Écart-type test: {wf_results['std_test_score']:.4f}")
    print(f"Score optimisation moyen: {wf_results['avg_opt_score']:.4f}")
    print(f"Cohérence (corrélation): {wf_results['consistency']:.4f}")
    
    # Sauvegarder les résultats
    import json
    with open('walk_forward_results.json', 'w') as f:
        json.dump(wf_results, f, indent=2, default=str)
    
    return wf_results

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Système de Backtest pour Bot de Trading Crypto')
    
    # Arguments généraux
    parser.add_argument('--pairs', nargs='+', default=['BTCEUR'], 
                       help='Paires de trading (ex: BTCEUR ETHEUR)')
    parser.add_argument('--days', type=int, default=180, 
                       help='Nombre de jours d\'historique')
    parser.add_argument('--capital', type=float, default=10000, 
                       help='Capital initial en euros')
    parser.add_argument('--interval', type=int, default=60, 
                       help='Intervalle en minutes (1, 5, 15, 30, 60, 240, 1440)')
    parser.add_argument('--output-dir', default='backtest_reports', 
                       help='Répertoire de sortie des rapports')
    parser.add_argument('--debug', action='store_true', 
                       help='Mode debug (plus de logs)')
    
    # Arguments d'optimisation
    parser.add_argument('--optimize', action='store_true', 
                       help='Lancer l\'optimisation des paramètres')
    parser.add_argument('--method', choices=['optuna', 'grid', 'differential'], 
                       default='optuna', help='Méthode d\'optimisation')
    parser.add_argument('--trials', type=int, default=100, 
                       help='Nombre d\'essais pour l\'optimisation')
    parser.add_argument('--metric', default='total_return',
                       choices=['total_return', 'sharpe_ratio', 'profit_factor', 
                               'win_rate', 'max_drawdown_adjusted', 'calmar_ratio'],
                       help='Métrique à optimiser')
    
    # Arguments walk-forward
    parser.add_argument('--walk-forward', action='store_true', 
                       help='Lancer l\'analyse walk-forward')
    parser.add_argument('--window', type=int, default=6, 
                       help='Taille de la fenêtre d\'optimisation (mois)')
    parser.add_argument('--step', type=int, default=1, 
                       help='Pas de progression (mois)')
    
    # Arguments spéciaux
    parser.add_argument('--config-test', action='store_true', 
                       help='Tester la configuration')
    
    args = parser.parse_args()
    
    # Configuration du logging
    setup_logging(args.debug)
    print_banner()
    
    # Vérifier la configuration
    if args.config_test:
        print("Test de la configuration...")
        try:
            # Tester l'importation des modules
            from src.backtest_data import BacktestDataManager
            print("✅ Modules importés avec succès")
            
            # Tester la connexion Kraken
            data_manager = BacktestDataManager()
            test_data = data_manager.download_historical_data('BTCEUR', 1, 7)
            if test_data is not None and not test_data.empty:
                print("✅ Connexion Kraken fonctionnelle")
                print(f"✅ Données de test: {len(test_data)} points")
            else:
                print("❌ Problème de connexion Kraken")
            
            return
            
        except Exception as e:
            print(f"❌ Erreur de configuration: {e}")
            return
    
    try:
        # Valider les arguments
        if args.interval not in [1, 5, 15, 30, 60, 240, 1440]:
            raise ValueError(f"Intervalle non supporté: {args.interval}")
        
        if args.days < 30:
            raise ValueError("Minimum 30 jours d'historique requis")
        
        if args.capital < 100:
            raise ValueError("Capital minimum: 100€")
        
        # Exécuter selon le mode choisi
        if args.walk_forward:
            logging.info("🔄 Mode: Analyse Walk-Forward")
            results = run_walk_forward_analysis(
                pairs=args.pairs,
                days=args.days,
                capital=args.capital,
                window_months=args.window,
                step_months=args.step
            )
            print(f"\n✅ Analyse walk-forward terminée!")
            
        elif args.optimize:
            logging.info("🎯 Mode: Optimisation des paramètres")
            results = run_optimization(
                pairs=args.pairs,
                days=args.days,
                capital=args.capital,
                method=args.method,
                trials=args.trials,
                metric=args.metric
            )
            print(f"\n✅ Optimisation terminée!")
            
        else:
            logging.info("📊 Mode: Backtest de base")
            results = run_basic_backtest(
                pairs=args.pairs,
                days=args.days,
                capital=args.capital,
                interval=args.interval,
                output_dir=args.output_dir
            )
            print(f"\n✅ Backtest terminé!")
        
        if results:
            print(f"📁 Résultats sauvegardés dans: {args.output_dir}")
            
            # Ouvrir le rapport HTML si possible
            if not args.optimize and not args.walk_forward:
                html_file = os.path.join(args.output_dir, 'backtest_dashboard.html')
                if os.path.exists(html_file):
                    print(f"🌐 Ouvrez le rapport: file://{os.path.abspath(html_file)}")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Arrêt demandé par l'utilisateur")
    except Exception as e:
        logging.error(f"❌ Erreur fatale: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()