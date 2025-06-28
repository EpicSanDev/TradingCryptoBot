"""
Bot de Trading Avancé Multi-Paire
=================================

Ce module implémente un bot de trading avancé avec support multi-paire,
modes spot et futures, et money management sophistiqué.
"""

import time
import schedule
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import asyncio
from collections import defaultdict
import numpy as np
import pandas as pd

from .config import Config
from .advanced_kraken_client import AdvancedKrakenClient
from .money_management import MoneyManager
from .strategy import TradingStrategy
from .institutional_strategy import InstitutionalStrategy, MarketRegime
try:
    from .indicators import TechnicalIndicators
except ImportError:
    from .indicators_pandas import TechnicalIndicatorsPandas as TechnicalIndicators


class TradingMode:
    """Modes de trading disponibles"""
    INSTITUTIONAL = "institutional"
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    CUSTOM = "custom"


class AdvancedTradingBot:
    """Bot de trading avancé multi-paire avec capacités institutionnelles"""
    
    def __init__(self, trading_mode: str = 'spot', strategy_mode: str = TradingMode.INSTITUTIONAL):
        """
        Initialiser le bot de trading avancé
        
        Args:
            trading_mode: 'spot' ou 'futures'
            strategy_mode: Mode de stratégie ('institutional', 'aggressive', 'conservative', 'custom')
        """
        self.trading_mode = trading_mode
        self.strategy_mode = strategy_mode
        self.is_running = False
        self.last_check = None
        
        # Initialiser les composants
        self._init_components()
        
        # État des positions par paire
        self.positions = {}
        self.open_orders = {}
        self.trade_history = []
        self.performance_metrics = defaultdict(float)
        
        # Risk management institutionnel
        self.portfolio_heat = 0.0  # Chaleur du portefeuille (0-1)
        self.correlation_monitor = {}
        self.execution_queue = []
        
        # Threading pour le multi-paire
        self.executor = ThreadPoolExecutor(max_workers=len(Config.TRADING_PAIRS))
        self.lock = threading.Lock()
        
        # Event loop pour les tâches asynchrones
        self.event_loop = None
        
        logging.info(f"Bot de trading avancé initialisé en mode {trading_mode} avec stratégie {strategy_mode}")
        
    def _init_components(self):
        """Initialiser tous les composants du bot"""
        try:
            # Valider la configuration
            Config.validate()
            
            # Initialiser les clients selon le mode
            if self.trading_mode == 'futures':
                self.spot_client = AdvancedKrakenClient('spot')
                self.futures_client = AdvancedKrakenClient('futures')
                self.active_client = self.futures_client
            else:
                self.spot_client = AdvancedKrakenClient('spot')
                self.futures_client = None
                self.active_client = self.spot_client
            
            # Initialiser les autres composants
            self.money_manager = MoneyManager(self.active_client)
            
            # Initialiser la stratégie selon le mode
            if self.strategy_mode == TradingMode.INSTITUTIONAL:
                self.strategy = InstitutionalStrategy(self.active_client)
                self.use_institutional = True
            else:
                self.strategy = TradingStrategy(self.active_client)
                self.use_institutional = False
            
            # Initialiser les composants avancés
            self._init_advanced_components()
            
            logging.info("Tous les composants initialisés avec succès")
            
        except Exception as e:
            logging.error(f"Erreur lors de l'initialisation des composants: {e}")
            raise
    
    def _init_advanced_components(self):
        """Initialiser les composants avancés pour le trading institutionnel"""
        try:
            # Moniteur de corrélation
            self.correlation_monitor = {
                'update_interval': 3600,  # 1 heure
                'last_update': None,
                'matrix': {}
            }
            
            # Gestionnaire d'exécution avancé
            self.execution_manager = {
                'slicing_enabled': True,
                'iceberg_orders': {},
                'vwap_targets': {},
                'execution_algos': ['TWAP', 'VWAP', 'POV', 'IS']  # Implementation Shortfall
            }
            
            # Moniteur de performance en temps réel
            self.performance_monitor = {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0
            }
            
            # Système d'alertes avancé
            self.alert_system = {
                'risk_alerts': [],
                'opportunity_alerts': [],
                'system_alerts': [],
                'thresholds': {
                    'max_portfolio_heat': 0.7,
                    'min_liquidity': 0.2,
                    'max_correlation': 0.8,
                    'max_slippage': 0.003
                }
            }
            
        except Exception as e:
            logging.error(f"Erreur lors de l'initialisation des composants avancés: {e}")
    
    def start(self):
        """Démarrer le bot de trading"""
        try:
            logging.info("Démarrage du bot de trading avancé...")
            
            # Vérifier la connexion
            balance = self.active_client.get_account_balance()
            if balance is None:
                raise Exception("Impossible de se connecter à Kraken")
            
            logging.info("Connexion à Kraken établie")
            logging.info(f"Solde du compte: {balance}")
            
            # Initialiser l'event loop pour les tâches asynchrones
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)
            
            # Planifier les vérifications pour chaque paire
            for pair in Config.TRADING_PAIRS:
                schedule.every(Config.CHECK_INTERVAL).minutes.do(
                    self.run_trading_cycle, pair
                )
            
            # Planifier la vérification globale
            schedule.every(Config.CHECK_INTERVAL).minutes.do(self.run_global_cycle)
            
            # Planifier les tâches institutionnelles
            if self.use_institutional:
                schedule.every(15).minutes.do(self._update_correlation_matrix)
                schedule.every(5).minutes.do(self._monitor_portfolio_risk)
                schedule.every(1).minutes.do(self._process_execution_queue)
                schedule.every(30).minutes.do(self._update_performance_metrics)
            
            self.is_running = True
            logging.info(f"Bot démarré - Vérification toutes les {Config.CHECK_INTERVAL} minutes")
            logging.info(f"Paires surveillées: {', '.join(Config.TRADING_PAIRS)}")
            logging.info(f"Mode de stratégie: {self.strategy_mode}")
            
            # Exécuter la première vérification immédiatement
            self.run_global_cycle()
            
            # Boucle principale
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)
                
        except KeyboardInterrupt:
            logging.info("Arrêt du bot demandé par l'utilisateur")
            self.stop()
        except Exception as e:
            logging.error(f"Erreur dans la boucle principale: {e}")
            self.stop()
    
    def stop(self):
        """Arrêter le bot de trading"""
        logging.info("Arrêt du bot de trading...")
        self.is_running = False
        schedule.clear()
        
        # Fermer toutes les positions si configuré
        if Config.CLOSE_POSITIONS_ON_STOP:
            self._close_all_positions("Bot arrêté - fermeture de sécurité")
        
        self.executor.shutdown(wait=True)
        
        # Sauvegarder l'état final
        self._save_state()
        
        logging.info("Bot arrêté proprement")
    
    def run_global_cycle(self):
        """Exécuter un cycle de trading global pour toutes les paires"""
        logging.info("=== Début du cycle de trading global ===")
        
        # Mettre à jour les métriques globales
        self._update_global_metrics()
        
        # Vérifier la santé du système
        system_health = self._check_system_health()
        if not system_health['healthy']:
            logging.error(f"Système non sain: {system_health['reason']}")
            self._handle_system_issue(system_health)
            return
        
        # Exécuter les cycles en parallèle
        futures = []
        for pair in Config.TRADING_PAIRS:
            future = self.executor.submit(self.run_trading_cycle, pair)
            futures.append(future)
        
        # Attendre la completion de tous les cycles
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    logging.info(f"Cycle terminé pour {result}")
            except Exception as e:
                logging.error(f"Erreur dans un cycle de trading: {e}")
        
        # Post-traitement global
        self._post_cycle_analysis()
        
        # Afficher le résumé global
        self._log_global_summary()
        
        self.last_check = datetime.now()
        logging.info("=== Fin du cycle de trading global ===\n")
    
    def run_trading_cycle(self, pair: str):
        """Exécuter un cycle de trading pour une paire spécifique"""
        try:
            # Vérifier si on a suffisamment de fonds
            if not self._check_sufficient_funds(pair):
                logging.info(f"Cycle de trading ignoré pour {pair} - fonds insuffisants")
                return pair
            
            # Analyser la paire selon le mode
            if self.use_institutional:
                analysis = self.strategy.analyze_market_advanced(pair)
            else:
                analysis = self._analyze_pair(pair)
            
            if not analysis:
                return pair
            
            # Enrichir l'analyse avec des données supplémentaires
            analysis = self._enrich_analysis(pair, analysis)
            
            # Vérifier la gestion des risques
            risk_check = self._check_risk_management(pair, analysis)
            if risk_check:
                logging.info(f"Action de gestion des risques pour {pair}: {risk_check}")
                return pair
            
            # Prendre les décisions de trading
            self._make_trading_decisions(pair, analysis)
            
            return pair
            
        except Exception as e:
            logging.error(f"Erreur dans le cycle de trading pour {pair}: {e}")
            return pair
    
    def _update_global_metrics(self):
        """Mettre à jour les métriques globales du portefeuille"""
        try:
            # Calculer la chaleur du portefeuille
            total_exposure = sum(p.get('value', 0) for p in self.positions.values())
            effective_capital = self.money_manager.get_effective_capital()
            
            if effective_capital > 0:
                self.portfolio_heat = total_exposure / effective_capital
            else:
                self.portfolio_heat = 0
            
            # Mettre à jour les métriques de performance
            if self.use_institutional:
                self._update_performance_metrics()
            
        except Exception as e:
            logging.error(f"Erreur lors de la mise à jour des métriques globales: {e}")
    
    def _check_system_health(self) -> Dict:
        """Vérifier la santé globale du système"""
        try:
            # Vérifier la connexion API
            if not self.active_client.test_connection():
                return {'healthy': False, 'reason': 'Connexion API perdue'}
            
            # Vérifier la chaleur du portefeuille
            if self.portfolio_heat > self.alert_system['thresholds']['max_portfolio_heat']:
                return {'healthy': False, 'reason': f'Chaleur du portefeuille trop élevée: {self.portfolio_heat:.2%}'}
            
            # Vérifier le drawdown
            if self.money_manager.current_drawdown > Config.MAX_DRAWDOWN:
                return {'healthy': False, 'reason': f'Drawdown maximum atteint: {self.money_manager.current_drawdown:.2%}'}
            
            # Vérifier les erreurs système
            if len(self.alert_system['system_alerts']) > 10:
                return {'healthy': False, 'reason': 'Trop d\'erreurs système détectées'}
            
            return {'healthy': True, 'reason': 'OK'}
            
        except Exception as e:
            logging.error(f"Erreur lors de la vérification de santé: {e}")
            return {'healthy': False, 'reason': f'Erreur de vérification: {e}'}
    
    def _handle_system_issue(self, health_status: Dict):
        """Gérer les problèmes système détectés"""
        try:
            reason = health_status['reason']
            
            if 'Connexion API' in reason:
                # Tenter de reconnecter
                logging.info("Tentative de reconnexion...")
                self._init_components()
                
            elif 'Chaleur du portefeuille' in reason:
                # Réduire l'exposition
                logging.warning("Réduction de l'exposition du portefeuille")
                self._reduce_portfolio_exposure()
                
            elif 'Drawdown' in reason:
                # Mode défensif
                logging.warning("Activation du mode défensif")
                self._activate_defensive_mode()
                
            else:
                # Enregistrer l'alerte
                self.alert_system['system_alerts'].append({
                    'timestamp': datetime.now(),
                    'issue': reason,
                    'action': 'monitoring'
                })
                
        except Exception as e:
            logging.error(f"Erreur lors de la gestion du problème système: {e}")
    
    def _enrich_analysis(self, pair: str, analysis: Dict) -> Dict:
        """Enrichir l'analyse avec des données supplémentaires"""
        try:
            # Ajouter les corrélations
            if pair in self.correlation_monitor['matrix']:
                analysis['correlations'] = self.correlation_monitor['matrix'][pair]
            
            # Ajouter l'historique de performance
            pair_performance = self._get_pair_performance(pair)
            analysis['historical_performance'] = pair_performance
            
            # Ajouter le contexte de marché global
            market_context = self._get_market_context()
            analysis['market_context'] = market_context
            
            # Ajouter les niveaux de liquidité
            if self.use_institutional and 'microstructure' in analysis:
                analysis['liquidity_levels'] = self._analyze_liquidity_levels(pair)
            
            return analysis
            
        except Exception as e:
            logging.error(f"Erreur lors de l'enrichissement de l'analyse: {e}")
            return analysis
    
    def _check_risk_management(self, pair: str, analysis: Dict):
        """Vérifier la gestion des risques pour une paire"""
        current_price = analysis['current_price']
        
        # Vérifier les positions ouvertes
        if pair in self.positions:
            position = self.positions[pair]
            
            # Vérifier les stops dynamiques
            action = self._check_dynamic_stops(pair, position, current_price, analysis)
            if action:
                return self._execute_risk_action(pair, position, action, analysis)
        
        # Vérifier les limites de corrélation
        if self.use_institutional:
            correlation_risk = self._check_correlation_risk(pair)
            if correlation_risk:
                return f"Risque de corrélation élevé: {correlation_risk}"
        
        # Vérifier les conditions de marché anormales
        if self.use_institutional and 'ml_signals' in analysis:
            if analysis['ml_signals'].get('anomaly', False):
                return "Conditions de marché anormales détectées"
        
        return None
    
    def _check_dynamic_stops(self, pair: str, position: Dict, current_price: float, analysis: Dict) -> Optional[str]:
        """Vérifier les stops dynamiques avec trailing et breakeven"""
        entry_price = position['entry_price']
        position_type = position['type']
        
        # Calculer le profit/perte actuel
        if position_type == 'long':
            pnl_percent = ((current_price - entry_price) / entry_price) * 100
        else:
            pnl_percent = ((entry_price - current_price) / entry_price) * 100
        
        # Breakeven stop
        if pnl_percent > 2 and not position.get('breakeven_set', False):
            logging.info(f"Activation du breakeven stop pour {pair}")
            position['stop_loss'] = entry_price * 1.001  # 0.1% au-dessus du breakeven
            position['breakeven_set'] = True
        
        # Trailing stop dynamique
        if Config.USE_TRAILING_STOP and pnl_percent > 5:
            new_stop = self.money_manager.calculate_dynamic_stop_loss(
                pair, entry_price, current_price, position_type
            )
            
            if position_type == 'long' and new_stop > position.get('stop_loss', 0):
                position['stop_loss'] = new_stop
                logging.info(f"Trailing stop mis à jour pour {pair}: {new_stop:.6f}")
            elif position_type == 'short' and new_stop < position.get('stop_loss', float('inf')):
                position['stop_loss'] = new_stop
                logging.info(f"Trailing stop mis à jour pour {pair}: {new_stop:.6f}")
        
        # Vérifier si le stop est atteint
        if position_type == 'long' and current_price <= position.get('stop_loss', 0):
            return 'STOP_LOSS'
        elif position_type == 'short' and current_price >= position.get('stop_loss', float('inf')):
            return 'STOP_LOSS'
        
        # Vérifier le take profit
        take_profit = position.get('take_profit')
        if take_profit:
            if position_type == 'long' and current_price >= take_profit:
                return 'TAKE_PROFIT'
            elif position_type == 'short' and current_price <= take_profit:
                return 'TAKE_PROFIT'
        
        # Vérifier le temps en position (pour éviter les positions stagnantes)
        time_in_position = datetime.now() - position['entry_time']
        if time_in_position > timedelta(days=Config.MAX_POSITION_DAYS):
            return 'TIME_EXIT'
        
        return None
    
    def _execute_risk_action(self, pair: str, position: Dict, action: str, analysis: Dict):
        """Exécuter une action de gestion des risques"""
        try:
            logging.warning(f"Exécution de l'action de risque {action} pour {pair}")
            
            if self.use_institutional:
                # Utiliser l'exécution algorithmique pour minimiser l'impact
                self._queue_algorithmic_order(pair, 'sell', position['volume'], 
                                            urgency='high', reason=action)
            else:
                # Exécution directe
                self._execute_sell_order(pair, analysis)
            
            # Enregistrer l'événement
            self.alert_system['risk_alerts'].append({
                'timestamp': datetime.now(),
                'pair': pair,
                'action': action,
                'price': analysis['current_price'],
                'pnl': self._calculate_position_pnl(position, analysis['current_price'])
            })
            
            return f"Position fermée: {action}"
            
        except Exception as e:
            logging.error(f"Erreur lors de l'exécution de l'action de risque: {e}")
            return f"Erreur: {e}"
    
    def _make_trading_decisions(self, pair: str, analysis: Dict):
        """Prendre des décisions de trading pour une paire"""
        recommendation = analysis['recommendation']
        current_price = analysis['current_price']
        
        # Log détaillé pour le mode institutionnel
        if self.use_institutional:
            self._log_institutional_analysis(pair, analysis)
        
        # Vérifier si on a déjà une position sur cette paire
        if pair in self.positions:
            # Gérer la position existante
            self._manage_existing_position(pair, analysis)
            return
        
        # Vérifier si l'exposition doit être réduite
        if self.money_manager.should_reduce_exposure():
            logging.warning("Exposition réduite en raison du drawdown")
            return
        
        # Décisions selon le mode
        if self.use_institutional:
            self._make_institutional_decision(pair, analysis)
        else:
            self._make_standard_decision(pair, analysis)
    
    def _make_institutional_decision(self, pair: str, analysis: Dict):
        """Prendre une décision de trading institutionnelle"""
        recommendation = analysis['recommendation']
        confidence = analysis.get('confidence', 0)
        
        # Vérifier les seuils institutionnels
        if confidence < self.strategy.institutional_confidence_threshold:
            logging.info(f"Confiance insuffisante pour {pair}: {confidence:.2%}")
            return
        
        # Vérifier la stratégie d'exécution recommandée
        execution_strategy = recommendation.get('execution_strategy', {})
        
        if recommendation['action'] == 'BUY':
            # Calculer la taille de position institutionnelle
            position_size = self.strategy.calculate_position_size_institutional(pair, analysis)
            
            if position_size > 0:
                # Ajouter à la queue d'exécution algorithmique
                self._queue_algorithmic_order(
                    pair, 'buy', position_size,
                    strategy=execution_strategy.get('method', 'LIMIT_ORDER'),
                    urgency='normal',
                    analysis=analysis
                )
        
        elif recommendation['action'] == 'SELL' and pair in self.positions:
            # Vente avec exécution optimisée
            position = self.positions[pair]
            self._queue_algorithmic_order(
                pair, 'sell', position['volume'],
                strategy=execution_strategy.get('method', 'LIMIT_ORDER'),
                urgency='normal',
                analysis=analysis
            )
    
    def _make_standard_decision(self, pair: str, analysis: Dict):
        """Prendre une décision de trading standard"""
        recommendation = analysis['recommendation']
        
        if recommendation['action'] == 'BUY' and self._should_buy(pair, analysis):
            self._execute_buy_order(pair, analysis)
        elif recommendation['action'] == 'SELL' and self._should_sell(pair, analysis):
            self._execute_sell_order(pair, analysis)
        else:
            logging.info(f"Aucune action de trading requise pour {pair}")
    
    def _queue_algorithmic_order(self, pair: str, side: str, volume: float, 
                                strategy: str = 'LIMIT_ORDER', urgency: str = 'normal',
                                analysis: Optional[Dict] = None, reason: str = ''):
        """Ajouter un ordre à la queue d'exécution algorithmique"""
        order = {
            'id': f"{pair}_{side}_{datetime.now().timestamp()}",
            'pair': pair,
            'side': side,
            'volume': volume,
            'strategy': strategy,
            'urgency': urgency,
            'analysis': analysis,
            'reason': reason,
            'created_at': datetime.now(),
            'status': 'pending',
            'filled': 0,
            'remaining': volume,
            'slices': []
        }
        
        with self.lock:
            self.execution_queue.append(order)
        
        logging.info(f"Ordre algorithmique ajouté: {order['id']}")
    
    def _process_execution_queue(self):
        """Traiter la queue d'exécution algorithmique"""
        if not self.execution_queue:
            return
        
        with self.lock:
            pending_orders = [o for o in self.execution_queue if o['status'] == 'pending']
        
        for order in pending_orders:
            try:
                self._execute_algorithmic_order(order)
            except Exception as e:
                logging.error(f"Erreur lors de l'exécution algorithmique: {e}")
                order['status'] = 'error'
                order['error'] = str(e)
    
    def _execute_algorithmic_order(self, order: Dict):
        """Exécuter un ordre selon l'algorithme choisi"""
        strategy = order['strategy']
        
        if strategy == 'MARKET_ORDER':
            self._execute_market_slice(order, order['remaining'])
            
        elif strategy == 'LIMIT_ORDER':
            self._execute_limit_slice(order)
            
        elif strategy == 'ICEBERG':
            self._execute_iceberg_slice(order)
            
        elif strategy in ['TWAP', 'VWAP']:
            self._execute_time_weighted_slice(order)
        
        # Vérifier si l'ordre est complété
        if order['remaining'] <= 0:
            order['status'] = 'completed'
            logging.info(f"Ordre algorithmique complété: {order['id']}")
    
    def _execute_market_slice(self, order: Dict, size: float):
        """Exécuter une tranche au marché"""
        try:
            pair = order['pair']
            side = order['side']
            
            # Placer l'ordre
            if side == 'buy':
                result = self.active_client.place_market_buy_order(pair, size)
            else:
                result = self.active_client.place_market_sell_order(pair, size)
            
            if result:
                order['filled'] += size
                order['remaining'] -= size
                order['slices'].append({
                    'size': size,
                    'timestamp': datetime.now(),
                    'type': 'market',
                    'result': result
                })
                
        except Exception as e:
            logging.error(f"Erreur lors de l'exécution market slice: {e}")
    
    def _execute_limit_slice(self, order: Dict):
        """Exécuter une tranche avec ordre limite"""
        try:
            pair = order['pair']
            side = order['side']
            
            # Obtenir le meilleur prix
            ticker = self.active_client.get_ticker(pair)
            if not ticker:
                return
            
            if side == 'buy':
                # Placer légèrement au-dessus du bid pour une exécution passive
                price = float(ticker['b'][0]) * 1.0001
                size = min(order['remaining'], order['volume'] * 0.2)  # 20% max par slice
                
                result = self.active_client.place_limit_order(pair, 'buy', price, size)
            else:
                # Placer légèrement en dessous de l'ask
                price = float(ticker['a'][0]) * 0.9999
                size = min(order['remaining'], order['volume'] * 0.2)
                
                result = self.active_client.place_limit_order(pair, 'sell', price, size)
            
            if result:
                order['slices'].append({
                    'size': size,
                    'price': price,
                    'timestamp': datetime.now(),
                    'type': 'limit',
                    'result': result,
                    'status': 'pending'
                })
                
        except Exception as e:
            logging.error(f"Erreur lors de l'exécution limit slice: {e}")
    
    def _execute_iceberg_slice(self, order: Dict):
        """Exécuter une tranche d'ordre iceberg"""
        try:
            # Taille visible réduite (10% du total)
            visible_size = order['volume'] * 0.1
            slice_size = min(order['remaining'], visible_size)
            
            # Utiliser limit order pour la discrétion
            self._execute_limit_slice({**order, 'volume': slice_size})
            
        except Exception as e:
            logging.error(f"Erreur lors de l'exécution iceberg slice: {e}")
    
    def _execute_time_weighted_slice(self, order: Dict):
        """Exécuter une tranche TWAP/VWAP"""
        try:
            # Calculer le temps écoulé et restant
            elapsed = datetime.now() - order['created_at']
            target_duration = timedelta(minutes=10)  # Durée cible de 10 minutes
            
            # Calculer la progression
            progress = min(elapsed.total_seconds() / target_duration.total_seconds(), 1.0)
            
            # Calculer la taille de la slice selon la progression
            target_filled = order['volume'] * progress
            slice_size = min(target_filled - order['filled'], order['remaining'])
            
            if slice_size > 0:
                self._execute_market_slice(order, slice_size)
                
        except Exception as e:
            logging.error(f"Erreur lors de l'exécution TWAP/VWAP slice: {e}")
    
    def _update_correlation_matrix(self):
        """Mettre à jour la matrice de corrélation entre les paires"""
        try:
            logging.info("Mise à jour de la matrice de corrélation...")
            
            # Collecter les données de prix pour toutes les paires
            price_data = {}
            for pair in Config.TRADING_PAIRS:
                ohlc = self.active_client.get_ohlc_data(pair, interval=60)  # 1h
                if ohlc is not None and not ohlc.empty:
                    price_data[pair] = ohlc['close'].pct_change().dropna()
            
            # Calculer les corrélations
            correlation_matrix = {}
            for pair1 in Config.TRADING_PAIRS:
                correlation_matrix[pair1] = {}
                for pair2 in Config.TRADING_PAIRS:
                    if pair1 in price_data and pair2 in price_data:
                        # Aligner les séries temporelles
                        aligned_data = pd.concat([price_data[pair1], price_data[pair2]], axis=1, join='inner')
                        if len(aligned_data) > 30:  # Au moins 30 points
                            corr = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
                            correlation_matrix[pair1][pair2] = corr
                        else:
                            correlation_matrix[pair1][pair2] = 0
                    else:
                        correlation_matrix[pair1][pair2] = 0
            
            self.correlation_monitor['matrix'] = correlation_matrix
            self.correlation_monitor['last_update'] = datetime.now()
            
            # Alerter sur les corrélations élevées
            self._check_high_correlations(correlation_matrix)
            
        except Exception as e:
            logging.error(f"Erreur lors de la mise à jour des corrélations: {e}")
    
    def _check_high_correlations(self, correlation_matrix: Dict):
        """Vérifier et alerter sur les corrélations élevées"""
        high_corr_threshold = self.alert_system['thresholds']['max_correlation']
        
        for pair1, correlations in correlation_matrix.items():
            for pair2, corr in correlations.items():
                if pair1 != pair2 and abs(corr) > high_corr_threshold:
                    # Vérifier si on a des positions sur les deux paires
                    if pair1 in self.positions and pair2 in self.positions:
                        alert = {
                            'timestamp': datetime.now(),
                            'type': 'high_correlation',
                            'pairs': [pair1, pair2],
                            'correlation': corr,
                            'message': f"Corrélation élevée détectée: {pair1}-{pair2} = {corr:.3f}"
                        }
                        self.alert_system['risk_alerts'].append(alert)
                        logging.warning(alert['message'])
    
    def _monitor_portfolio_risk(self):
        """Monitorer le risque global du portefeuille"""
        try:
            # Calculer les métriques de risque
            if self.use_institutional:
                risk_metrics = self.strategy.get_risk_metrics(self.positions)
            else:
                risk_metrics = self._calculate_basic_risk_metrics()
            
            # Vérifier les seuils
            if risk_metrics.get('var_95', 0) < -0.05:  # VaR 95% > 5%
                logging.warning(f"VaR élevée détectée: {risk_metrics['var_95']:.2%}")
            
            if risk_metrics.get('exposure_ratio', 0) > 0.8:
                logging.warning(f"Exposition élevée: {risk_metrics['exposure_ratio']:.2%}")
            
            # Mettre à jour les métriques
            self.performance_metrics['current_risk'] = risk_metrics
            
        except Exception as e:
            logging.error(f"Erreur lors du monitoring du risque: {e}")
    
    def _calculate_basic_risk_metrics(self) -> Dict:
        """Calculer les métriques de risque basiques"""
        total_exposure = sum(p.get('value', 0) for p in self.positions.values())
        position_count = len(self.positions)
        
        return {
            'total_exposure': total_exposure,
            'exposure_ratio': total_exposure / self.money_manager.get_effective_capital() if self.money_manager.get_effective_capital() > 0 else 0,
            'position_count': position_count,
            'avg_position_size': total_exposure / position_count if position_count > 0 else 0
        }
    
    def _update_performance_metrics(self):
        """Mettre à jour les métriques de performance avancées"""
        try:
            # Récupérer l'historique des trades
            if len(self.trade_history) < 2:
                return
            
            # Calculer les rendements
            returns = []
            for i in range(1, len(self.trade_history)):
                trade = self.trade_history[i]
                if 'profit_loss_percent' in trade:
                    returns.append(trade['profit_loss_percent'] / 100)
            
            if not returns:
                return
            
            returns = np.array(returns)
            
            # Sharpe Ratio (annualisé)
            if len(returns) > 1 and np.std(returns) > 0:
                periods_per_year = 365 * 24 / (Config.CHECK_INTERVAL / 60)  # Ajuster selon la fréquence
                self.performance_monitor['sharpe_ratio'] = (np.mean(returns) * np.sqrt(periods_per_year)) / np.std(returns)
            
            # Sortino Ratio (ne considère que la volatilité négative)
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 1:
                downside_std = np.std(negative_returns)
                if downside_std > 0:
                    self.performance_monitor['sortino_ratio'] = np.mean(returns) / downside_std
            
            # Win Rate
            winning_trades = sum(1 for r in returns if r > 0)
            self.performance_monitor['win_rate'] = winning_trades / len(returns) if len(returns) > 0 else 0
            
            # Profit Factor
            gross_profit = sum(r for r in returns if r > 0)
            gross_loss = abs(sum(r for r in returns if r < 0))
            if gross_loss > 0:
                self.performance_monitor['profit_factor'] = gross_profit / gross_loss
            
            # Expectancy
            self.performance_monitor['expectancy'] = np.mean(returns)
            
            # Max Drawdown
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            self.performance_monitor['max_drawdown'] = np.min(drawdown)
            
        except Exception as e:
            logging.error(f"Erreur lors de la mise à jour des métriques de performance: {e}")
    
    def _reduce_portfolio_exposure(self):
        """Réduire l'exposition du portefeuille"""
        try:
            # Identifier les positions les moins performantes
            positions_by_pnl = []
            for pair, position in self.positions.items():
                current_price = self.active_client.get_current_price(pair)
                if current_price:
                    pnl = self._calculate_position_pnl(position, current_price)
                    positions_by_pnl.append((pair, pnl))
            
            # Trier par PnL (pire en premier)
            positions_by_pnl.sort(key=lambda x: x[1])
            
            # Fermer les positions les moins performantes
            target_reduction = 0.3  # Réduire de 30%
            current_exposure = sum(p.get('value', 0) for p in self.positions.values())
            target_exposure = current_exposure * (1 - target_reduction)
            
            for pair, pnl in positions_by_pnl:
                if current_exposure <= target_exposure:
                    break
                
                position = self.positions[pair]
                logging.warning(f"Fermeture de position pour réduire l'exposition: {pair} (PnL: {pnl:.2%})")
                
                # Queue la fermeture
                self._queue_algorithmic_order(pair, 'sell', position['volume'],
                                            urgency='high', reason='exposure_reduction')
                
                current_exposure -= position.get('value', 0)
                
        except Exception as e:
            logging.error(f"Erreur lors de la réduction de l'exposition: {e}")
    
    def _activate_defensive_mode(self):
        """Activer le mode défensif suite à un drawdown élevé"""
        try:
            logging.warning("Activation du mode défensif")
            
            # Augmenter les seuils de confiance
            if self.use_institutional:
                self.strategy.institutional_confidence_threshold = min(0.9, self.strategy.institutional_confidence_threshold + 0.1)
            
            # Réduire la taille des positions futures
            Config.FIXED_POSITION_SIZE *= 0.5
            Config.MAX_RISK_PER_TRADE *= 0.5
            
            # Fermer les positions à perte
            for pair, position in list(self.positions.items()):
                current_price = self.active_client.get_current_price(pair)
                if current_price:
                    pnl = self._calculate_position_pnl(position, current_price)
                    if pnl < -0.02:  # Perte > 2%
                        logging.warning(f"Fermeture défensive de {pair} (PnL: {pnl:.2%})")
                        self._queue_algorithmic_order(pair, 'sell', position['volume'],
                                                    urgency='high', reason='defensive_mode')
                        
        except Exception as e:
            logging.error(f"Erreur lors de l'activation du mode défensif: {e}")
    
    def _calculate_position_pnl(self, position: Dict, current_price: float) -> float:
        """Calculer le PnL d'une position"""
        entry_price = position['entry_price']
        position_type = position.get('type', 'long')
        
        if position_type == 'long':
            return (current_price - entry_price) / entry_price
        else:
            return (entry_price - current_price) / entry_price
    
    def _log_institutional_analysis(self, pair: str, analysis: Dict):
        """Logger les détails de l'analyse institutionnelle"""
        if 'institutional_score' not in analysis:
            return
        
        score = analysis['institutional_score']
        logging.info(f"\n=== Analyse Institutionnelle {pair} ===")
        logging.info(f"Score composite: {score['composite_score']:.3f}")
        logging.info(f"Confiance: {score['confidence']:.2%}")
        
        if 'components' in score:
            logging.info("Composants du score:")
            for component, value in score['components'].items():
                logging.info(f"  - {component}: {value:.3f}")
        
        if 'microstructure' in analysis:
            micro = analysis['microstructure']
            logging.info(f"Liquidité: {micro.get('liquidity_score', 0):.2f}")
            logging.info(f"Impact estimé: {micro.get('market_impact', 0):.2%}")
            logging.info(f"Imbalance: {micro.get('order_book_imbalance', 0):.3f}")
        
        if 'regime' in analysis:
            regime = analysis['regime']
            logging.info(f"Régime: {regime.get('main_regime', 'UNKNOWN')}")
            logging.info(f"Volatilité: {regime.get('volatility_regime', 'UNKNOWN')}")
    
    def _post_cycle_analysis(self):
        """Analyse post-cycle pour optimisation"""
        try:
            # Analyser les opportunités manquées
            self._analyze_missed_opportunities()
            
            # Optimiser les paramètres dynamiquement
            if self.use_institutional:
                self._optimize_parameters()
            
            # Nettoyer les alertes anciennes
            self._cleanup_old_alerts()
            
        except Exception as e:
            logging.error(f"Erreur dans l'analyse post-cycle: {e}")
    
    def _analyze_missed_opportunities(self):
        """Analyser les opportunités potentiellement manquées"""
        # Simplifiée pour l'exemple
        pass
    
    def _optimize_parameters(self):
        """Optimiser les paramètres de trading dynamiquement"""
        # Ajuster les seuils selon la performance récente
        if self.performance_monitor['win_rate'] < 0.4:
            # Augmenter la sélectivité
            if self.use_institutional:
                self.strategy.institutional_confidence_threshold = min(0.85, 
                    self.strategy.institutional_confidence_threshold + 0.05)
    
    def _cleanup_old_alerts(self):
        """Nettoyer les alertes anciennes"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        for alert_type in ['risk_alerts', 'opportunity_alerts', 'system_alerts']:
            alerts = self.alert_system[alert_type]
            self.alert_system[alert_type] = [
                a for a in alerts if a['timestamp'] > cutoff_time
            ]
    
    def _save_state(self):
        """Sauvegarder l'état du bot"""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'positions': self.positions,
                'trade_history': self.trade_history[-100:],  # Derniers 100 trades
                'performance_metrics': dict(self.performance_monitor),
                'correlation_matrix': self.correlation_monitor['matrix'],
                'alerts': {
                    'risk': self.alert_system['risk_alerts'][-50:],
                    'opportunity': self.alert_system['opportunity_alerts'][-50:],
                    'system': self.alert_system['system_alerts'][-50:]
                }
            }
            
            with open('bot_state.json', 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logging.info("État du bot sauvegardé")
            
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde de l'état: {e}")
    
    def _close_all_positions(self, reason: str):
        """Fermer toutes les positions ouvertes"""
        for pair in list(self.positions.keys()):
            position = self.positions[pair]
            logging.warning(f"Fermeture de {pair}: {reason}")
            self._queue_algorithmic_order(pair, 'sell', position['volume'],
                                        urgency='immediate', reason=reason)

    def _check_sufficient_funds(self, pair: str) -> bool:
        """Vérifier si le compte a suffisamment de fonds pour trader"""
        try:
            # Mettre à jour le solde du compte
            self.money_manager.update_account_balance()
            
            available_balance = self.money_manager.available_balance
            min_order_value = 10.0  # Kraken minimum order value
            
            if available_balance < min_order_value:
                logging.warning(f"Fonds insuffisants pour trader {pair}: {available_balance:.2f} EUR < {min_order_value} EUR minimum")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Erreur lors de la vérification des fonds: {e}")
            return False

    def _get_pair_performance(self, pair: str) -> Dict:
        """Obtenir l'historique de performance pour une paire"""
        pair_trades = [t for t in self.trade_history if t.get('pair') == pair]
        
        if not pair_trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'best_trade': 0.0,
                'worst_trade': 0.0
            }
        
        wins = sum(1 for t in pair_trades if t.get('profit_loss', 0) > 0)
        profits = [t.get('profit_loss_percent', 0) for t in pair_trades]
        
        return {
            'total_trades': len(pair_trades),
            'win_rate': wins / len(pair_trades) if pair_trades else 0,
            'avg_profit': np.mean(profits) if profits else 0,
            'best_trade': max(profits) if profits else 0,
            'worst_trade': min(profits) if profits else 0
        }
    
    def _get_market_context(self) -> Dict:
        """Obtenir le contexte global du marché"""
        # Analyser l'état général du marché crypto
        btc_performance = self._get_pair_performance('XXBTZEUR')
        
        return {
            'btc_trend': 'UP' if btc_performance['avg_profit'] > 0 else 'DOWN',
            'market_volatility': self._estimate_market_volatility(),
            'correlation_strength': self._get_average_correlation()
        }
    
    def _estimate_market_volatility(self) -> str:
        """Estimer la volatilité globale du marché"""
        # Simplifiée - normalement basée sur l'analyse de toutes les paires
        return 'NORMAL'
    
    def _get_average_correlation(self) -> float:
        """Obtenir la corrélation moyenne entre toutes les paires"""
        if not self.correlation_monitor['matrix']:
            return 0.0
        
        correlations = []
        matrix = self.correlation_monitor['matrix']
        
        for pair1 in matrix:
            for pair2 in matrix[pair1]:
                if pair1 != pair2:
                    correlations.append(abs(matrix[pair1][pair2]))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _analyze_liquidity_levels(self, pair: str) -> Dict:
        """Analyser les niveaux de liquidité pour une paire"""
        # Récupérer le carnet d'ordres
        order_book = self.active_client.get_order_book(pair)
        
        if not order_book:
            return {'status': 'unknown', 'depth': 0}
        
        # Calculer la profondeur à différents niveaux
        levels = {}
        for pct in [0.1, 0.5, 1.0, 2.0]:  # 0.1%, 0.5%, 1%, 2%
            bid_depth = 0
            ask_depth = 0
            
            mid_price = (float(order_book['bids'][0][0]) + float(order_book['asks'][0][0])) / 2
            target_bid = mid_price * (1 - pct / 100)
            target_ask = mid_price * (1 + pct / 100)
            
            # Calculer le volume jusqu'au niveau cible
            for bid in order_book['bids']:
                if float(bid[0]) >= target_bid:
                    bid_depth += float(bid[1])
            
            for ask in order_book['asks']:
                if float(ask[0]) <= target_ask:
                    ask_depth += float(ask[1])
            
            levels[f'{pct}%'] = {
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'total': bid_depth + ask_depth
            }
        
        return levels
    
    def _check_correlation_risk(self, pair: str) -> Optional[str]:
        """Vérifier le risque de corrélation pour une paire"""
        if not self.correlation_monitor['matrix'] or pair not in self.correlation_monitor['matrix']:
            return None
        
        correlations = self.correlation_monitor['matrix'][pair]
        high_corr_threshold = self.alert_system['thresholds']['max_correlation']
        
        # Vérifier les positions existantes
        for other_pair, corr in correlations.items():
            if other_pair != pair and other_pair in self.positions:
                if abs(corr) > high_corr_threshold:
                    return f"Corrélation élevée avec {other_pair}: {corr:.3f}"
        
        return None
    
    def _manage_existing_position(self, pair: str, analysis: Dict):
        """Gérer une position existante"""
        position = self.positions[pair]
        current_price = analysis['current_price']
        
        # Calculer le PnL actuel
        pnl = self._calculate_position_pnl(position, current_price)
        
        # Vérifier si on doit prendre des profits partiels
        if pnl > 0.05:  # 5% de profit
            # Considérer une prise de profit partielle
            if self.use_institutional:
                # Vérifier si le momentum continue
                if 'recommendation' in analysis and analysis['recommendation']['action'] != 'BUY':
                    # Prendre 50% des profits
                    partial_volume = position['volume'] * 0.5
                    self._queue_algorithmic_order(
                        pair, 'sell', partial_volume,
                        urgency='normal', reason='partial_profit_taking'
                    )
                    position['volume'] -= partial_volume
                    logging.info(f"Prise de profit partielle sur {pair}: {pnl:.2%}")
    
    def _log_global_summary(self):
        """Afficher le résumé global des performances"""
        metrics = self.money_manager.get_performance_metrics()
        balance_summary = self.money_manager.get_balance_summary()
        
        logging.info("=== RÉSUMÉ GLOBAL DES PERFORMANCES ===")
        logging.info(f"Mode de trading: {self.trading_mode}")
        logging.info(f"Trades totaux: {metrics['total_trades']}")
        logging.info(f"Taux de réussite: {metrics['win_rate']:.2f}%")
        logging.info(f"Profit/Perte total: {metrics['total_profit_loss']:.2f}")
        logging.info(f"Drawdown actuel: {metrics['current_drawdown']:.2f}%")
        logging.info(f"Ratio de Sharpe: {metrics['sharpe_ratio']:.2f}")
        
        logging.info("\n=== SOLDE DU COMPTE ===")
        logging.info(f"Solde réel du compte: {balance_summary['account_balance']:.2f} EUR")
        logging.info(f"Capital effectif utilisé: {balance_summary['effective_capital']:.2f} EUR")
        logging.info(f"Capital configuré: {balance_summary['configured_capital']:.2f} EUR")
        logging.info(f"Utilisation du capital: {balance_summary['capital_utilization']:.1f}%")
        if balance_summary['last_update']:
            logging.info(f"Dernière mise à jour: {balance_summary['last_update'].strftime('%H:%M:%S')}")
        
        # Afficher les positions ouvertes
        if self.positions:
            logging.info("\n=== POSITIONS OUVERTES ===")
            for pair, position in self.positions.items():
                pnl = position.get('unrealized_pnl', 0)
                pnl_percent = position.get('unrealized_pnl_percent', 0)
                leverage_info = f" (levier {position['leverage']}x)" if position.get('leverage') else ""
                logging.info(f"{pair}: {position['volume']} @ {position['entry_price']}{leverage_info}")
                logging.info(f"  PnL: {pnl:.2f} ({pnl_percent:.2f}%)")
    
    def get_status(self) -> Dict:
        """Obtenir le statut actuel du bot"""
        return {
            'is_running': self.is_running,
            'trading_mode': self.trading_mode,
            'last_check': self.last_check,
            'active_pairs': list(self.positions.keys()),
            'performance': self.money_manager.get_performance_metrics()
        }
    
    def get_trade_history(self) -> List[Dict]:
        """Obtenir l'historique des trades"""
        with self.lock:
            return self.trade_history.copy()
    
    def get_current_positions(self) -> List[Dict]:
        """Obtenir les positions actuelles"""
        with self.lock:
            return [
                {
                    'pair': pair,
                    'type': pos['type'],
                    'volume': pos['volume'],
                    'entry_price': pos['entry_price'],
                    'current_price': pos.get('current_price', pos['entry_price']),
                    'leverage': pos.get('leverage'),
                    'unrealized_pnl': pos.get('unrealized_pnl', 0),
                    'unrealized_pnl_percent': pos.get('unrealized_pnl_percent', 0)
                }
                for pair, pos in self.positions.items()
            ]
    
    def manual_buy(self, pair: str, volume: Optional[float] = None) -> bool:
        """Achat manuel"""
        try:
            if pair not in Config.TRADING_PAIRS:
                logging.error(f"Paire {pair} non configurée")
                return False
            
            current_price = self.active_client.get_current_price(pair)
            if current_price is None:
                return False
            
            if volume is None:
                volume = self._calculate_position_size(pair, {
                    'current_price': current_price,
                    'recommendation': {'confidence': 0.5}
                })
            
            leverage = Config.get_leverage_for_pair(pair) if self.trading_mode == 'futures' else None
            
            order = self.active_client.place_market_order(pair, 'buy', volume, leverage)
            
            if order:
                with self.lock:
                    self.positions[pair] = {
                        'type': 'long',
                        'entry_price': current_price,
                        'volume': volume,
                        'leverage': leverage,
                        'entry_time': datetime.now()
                    }
                
                self._record_trade(pair, 'buy', volume, current_price, current_price, leverage)
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Erreur lors de l'achat manuel: {e}")
            return False
    
    def manual_sell(self, pair: str) -> bool:
        """Vente manuelle"""
        try:
            if pair not in self.positions:
                logging.error(f"Pas de position à vendre pour {pair}")
                return False
            
            position = self.positions[pair]
            current_price = self.active_client.get_current_price(pair)
            
            if current_price is None:
                return False
            
            order = self.active_client.place_market_order(
                pair, 'sell', position['volume'], position.get('leverage')
            )
            
            if order:
                # Calculer le profit/perte
                entry_price = position['entry_price']
                profit_loss = (current_price - entry_price) * position['volume']
                if position.get('leverage'):
                    profit_loss *= position['leverage']
                
                self.money_manager.update_balance(profit_loss)
                
                with self.lock:
                    del self.positions[pair]
                
                self._record_trade(pair, 'sell', position['volume'], current_price, 
                                 entry_price, position.get('leverage'), profit_loss)
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Erreur lors de la vente manuelle: {e}")
            return False
    
    def _analyze_pair(self, pair: str) -> Optional[Dict]:
        """Analyser une paire de trading"""
        try:
            # Obtenir les données OHLC
            ohlc_data = self.active_client.get_ohlc_data(pair, interval=1)
            if ohlc_data is None:
                return None
            if hasattr(ohlc_data, 'empty') and ohlc_data.empty:
                return None
            
            # Calculer les indicateurs techniques
            indicators_obj = TechnicalIndicators(ohlc_data)
            indicators = indicators_obj.get_latest_indicators()
            
            # Obtenir le prix actuel
            current_price = self.active_client.get_current_price(pair)
            if current_price is None:
                return None
            
            # Analyser avec la stratégie
            analysis = self.strategy.analyze_market(pair)
            if analysis is None:
                return None
            
            # Ajouter les indicateurs calculés
            analysis['indicators'] = indicators
            analysis['current_price'] = current_price
            
            return analysis
            
        except Exception as e:
            logging.error(f"Erreur lors de l'analyse de {pair}: {e}")
            return None
    
    def _should_buy(self, pair: str, analysis: Dict) -> bool:
        """Déterminer si on doit acheter"""
        # Vérifier la force du signal
        confidence = analysis['recommendation'].get('confidence', 0)
        if confidence < Config.MIN_SIGNAL_CONFIDENCE:
            return False
        
        # Vérifier les limites de risque
        position_size = self._calculate_position_size(pair, analysis)
        if not self.money_manager.check_risk_limits(pair, position_size):
            return False
        
        return True
    
    def _should_sell(self, pair: str, analysis: Dict) -> bool:
        """Déterminer si on doit vendre (pour les positions ouvertes)"""
        if pair not in self.positions:
            return False
        
        confidence = analysis['recommendation'].get('confidence', 0)
        return confidence >= Config.MIN_SIGNAL_CONFIDENCE
    
    def _calculate_position_size(self, pair: str, analysis: Dict) -> float:
        """Calculer la taille de position optimale"""
        current_price = analysis['current_price']
        stop_loss_price = self._calculate_stop_loss_price(pair, current_price, 'long')
        
        signal_strength = analysis['recommendation'].get('confidence', 0.5)
        
        # Appliquer le facteur de réduction si nécessaire
        reduction_factor = self.money_manager.get_position_reduction_factor()
        
        base_size = self.money_manager.calculate_position_size(
            pair, signal_strength, current_price, stop_loss_price
        )
        
        return base_size * reduction_factor
    
    def _calculate_stop_loss_price(self, pair: str, current_price: float, position_type: str) -> float:
        """Calculer le prix du stop-loss"""
        stop_percentage = Config.get_stop_loss_for_pair(pair) / 100
        
        if position_type == 'long':
            return current_price * (1 - stop_percentage)
        else:
            return current_price * (1 + stop_percentage)
    
    def _execute_buy_order(self, pair: str, analysis: Dict):
        """Exécuter un ordre d'achat"""
        try:
            position_size = self._calculate_position_size(pair, analysis)
            current_price = analysis['current_price']
            
            # Vérifier la taille minimale de l'ordre
            position_value = position_size * current_price
            min_order_value = 10.0  # Kraken minimum order value in EUR
            
            if position_value < min_order_value:
                logging.warning(f"Position trop petite pour {pair}: {position_value:.2f} EUR < {min_order_value} EUR minimum")
                return
            
            # Déterminer le levier pour les futures
            leverage = None
            if self.trading_mode == 'futures':
                leverage = Config.get_leverage_for_pair(pair)
            
            # Placer l'ordre
            order = self.active_client.place_market_order(
                pair, 'buy', position_size, leverage
            )
            
            if order:
                # Enregistrer la position
                with self.lock:
                    self.positions[pair] = {
                        'type': 'long',
                        'entry_price': current_price,
                        'volume': position_size,
                        'value': position_value,
                        'leverage': leverage,
                        'entry_time': datetime.now(),
                        'stop_loss': self._calculate_stop_loss_price(pair, current_price, 'long')
                    }
                
                # Mettre à jour le money manager
                self.money_manager.update_position_size(pair, position_size)
                
                # Enregistrer le trade
                self._record_trade(pair, 'buy', position_size, current_price, 
                                 current_price, leverage)
                
                logging.info(f"Ordre d'achat exécuté pour {pair}: {position_size:.8f} @ {current_price}")
            else:
                logging.error(f"Échec de l'ordre d'achat pour {pair}")
                
        except Exception as e:
            logging.error(f"Erreur lors de l'exécution de l'achat pour {pair}: {e}")
    
    def _execute_sell_order(self, pair: str, analysis: Dict):
        """Exécuter un ordre de vente"""
        try:
            if pair not in self.positions:
                logging.warning(f"Pas de position à vendre pour {pair}")
                return
            
            position = self.positions[pair]
            current_price = analysis['current_price']
            
            # Placer l'ordre de vente
            order = self.active_client.place_market_order(
                pair, 'sell', position['volume'], position.get('leverage')
            )
            
            if order:
                # Calculer le profit/perte
                entry_price = position['entry_price']
                profit_loss = (current_price - entry_price) * position['volume']
                if position.get('leverage'):
                    profit_loss *= position['leverage']
                
                # Mettre à jour le money manager
                self.money_manager.update_balance(profit_loss)
                
                # Supprimer la position
                with self.lock:
                    del self.positions[pair]
                
                # Enregistrer le trade
                self._record_trade(pair, 'sell', position['volume'], current_price, 
                                 entry_price, position.get('leverage'), profit_loss)
                
                logging.info(f"Ordre de vente exécuté pour {pair}: Profit/Perte = {profit_loss:.2f}")
            else:
                logging.error(f"Échec de l'ordre de vente pour {pair}")
                
        except Exception as e:
            logging.error(f"Erreur lors de l'exécution de la vente pour {pair}: {e}")
    
    def _record_trade(self, pair: str, action: str, volume: float, price: float, 
                     entry_price: float, leverage: Optional[float] = None, 
                     profit_loss: Optional[float] = None):
        """Enregistrer un trade dans l'historique"""
        trade_data = {
            'pair': pair,
            'action': action,
            'volume': volume,
            'price': price,
            'entry_price': entry_price,
            'leverage': leverage,
            'timestamp': datetime.now(),
            'trading_mode': self.trading_mode
        }
        
        if profit_loss is not None:
            trade_data['profit_loss'] = profit_loss
            trade_data['profit_loss_percent'] = (profit_loss / (entry_price * volume)) * 100
        
        with self.lock:
            self.trade_history.append(trade_data)
            self.money_manager.add_trade(trade_data) 