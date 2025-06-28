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

from .config import Config
from .advanced_kraken_client import AdvancedKrakenClient
from .money_management import MoneyManager
from .strategy import TradingStrategy
try:
    from .indicators import TechnicalIndicators
except ImportError:
    from .indicators_pandas import TechnicalIndicatorsPandas as TechnicalIndicators

class AdvancedTradingBot:
    """Bot de trading avancé multi-paire"""
    
    def __init__(self, trading_mode: str = 'spot'):
        """
        Initialiser le bot de trading avancé
        
        Args:
            trading_mode: 'spot' ou 'futures'
        """
        self.trading_mode = trading_mode
        self.is_running = False
        self.last_check = None
        
        # Initialiser les composants
        self._init_components()
        
        # État des positions par paire
        self.positions = {}
        self.open_orders = {}
        self.trade_history = []
        
        # Threading pour le multi-paire
        self.executor = ThreadPoolExecutor(max_workers=len(Config.TRADING_PAIRS))
        self.lock = threading.Lock()
        
        logging.info(f"Bot de trading avancé initialisé en mode {trading_mode}")
        
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
            self.strategy = TradingStrategy(self.active_client)
            
            logging.info("Tous les composants initialisés avec succès")
            
        except Exception as e:
            logging.error(f"Erreur lors de l'initialisation des composants: {e}")
            raise
    
    def start(self):
        """Démarrer le bot de trading"""
        try:
            logging.info("Démarrage du bot de trading avancé...")
            
            # Vérifier la connexion
            balance = self.active_client.get_account_balance()
            if not balance:
                raise Exception("Impossible de se connecter à Kraken")
            
            logging.info("Connexion à Kraken établie")
            logging.info(f"Solde du compte: {balance}")
            
            # Planifier les vérifications pour chaque paire
            for pair in Config.TRADING_PAIRS:
                schedule.every(Config.CHECK_INTERVAL).minutes.do(
                    self.run_trading_cycle, pair
                )
            
            # Planifier la vérification globale
            schedule.every(Config.CHECK_INTERVAL).minutes.do(self.run_global_cycle)
            
            self.is_running = True
            logging.info(f"Bot démarré - Vérification toutes les {Config.CHECK_INTERVAL} minutes")
            logging.info(f"Paires surveillées: {', '.join(Config.TRADING_PAIRS)}")
            
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
        self.executor.shutdown(wait=True)
    
    def run_global_cycle(self):
        """Exécuter un cycle de trading global pour toutes les paires"""
        logging.info("=== Début du cycle de trading global ===")
        
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
        
        # Afficher le résumé global
        self._log_global_summary()
        
        self.last_check = datetime.now()
        logging.info("=== Fin du cycle de trading global ===\n")
    
    def run_trading_cycle(self, pair: str):
        """Exécuter un cycle de trading pour une paire spécifique"""
        try:
            logging.info(f"--- Cycle de trading pour {pair} ---")
            
            # Analyser le marché
            analysis = self._analyze_pair(pair)
            if not analysis:
                logging.warning(f"Impossible d'analyser {pair}")
                return pair
            
            # Vérifier la gestion des risques
            self._check_risk_management(pair, analysis)
            
            # Prendre des décisions de trading
            self._make_trading_decisions(pair, analysis)
            
            # Mettre à jour les positions
            self._update_positions(pair)
            
            logging.info(f"--- Cycle terminé pour {pair} ---")
            return pair
            
        except Exception as e:
            logging.error(f"Erreur lors du cycle de trading pour {pair}: {e}")
            return pair
    
    def _analyze_pair(self, pair: str) -> Optional[Dict]:
        """Analyser une paire de trading"""
        try:
            # Obtenir les données OHLC
            ohlc_data = self.active_client.get_ohlc_data(pair, interval=1)
            if ohlc_data is None or ohlc_data.empty:
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
            if not analysis:
                return None
            
            # Ajouter les indicateurs calculés
            analysis['indicators'] = indicators
            analysis['current_price'] = current_price
            
            return analysis
            
        except Exception as e:
            logging.error(f"Erreur lors de l'analyse de {pair}: {e}")
            return None
    
    def _check_risk_management(self, pair: str, analysis: Dict):
        """Vérifier la gestion des risques pour une paire"""
        current_price = analysis['current_price']
        
        # Vérifier les stop-loss et take-profit pour les positions ouvertes
        if pair in self.positions:
            position = self.positions[pair]
            action = self._check_position_risk(pair, position, current_price)
            
            if action == 'SELL':
                logging.warning(f"Stop-loss/Take-profit atteint pour {pair} - Vente forcée")
                self._execute_forced_sell(pair, position, analysis)
    
    def _check_position_risk(self, pair: str, position: Dict, current_price: float) -> Optional[str]:
        """Vérifier le risque d'une position"""
        entry_price = position['entry_price']
        position_type = position['type']
        
        # Calculer le stop-loss dynamique
        stop_loss_price = self.money_manager.calculate_dynamic_stop_loss(
            pair, entry_price, current_price, position_type
        )
        
        # Vérifier si le stop-loss est atteint
        if position_type == 'long' and current_price <= stop_loss_price:
            return 'SELL'
        elif position_type == 'short' and current_price >= stop_loss_price:
            return 'SELL'
        
        # Vérifier le take-profit
        take_profit_percentage = Config.get_take_profit_for_pair(pair) / 100
        if position_type == 'long':
            take_profit_price = entry_price * (1 + take_profit_percentage)
            if current_price >= take_profit_price:
                return 'SELL'
        else:
            take_profit_price = entry_price * (1 - take_profit_percentage)
            if current_price <= take_profit_price:
                return 'SELL'
        
        return None
    
    def _execute_forced_sell(self, pair: str, position: Dict, analysis: Dict):
        """Exécuter une vente forcée"""
        try:
            volume = position['volume']
            leverage = position.get('leverage')
            
            # Placer l'ordre de vente
            order = self.active_client.place_market_order(
                pair, 'sell', volume, leverage
            )
            
            if order:
                # Mettre à jour la position
                with self.lock:
                    if pair in self.positions:
                        del self.positions[pair]
                
                # Enregistrer le trade
                self._record_trade(pair, 'sell', volume, analysis['current_price'], 
                                 position['entry_price'], leverage)
                
                logging.info(f"Vente forcée exécutée pour {pair}")
            else:
                logging.error(f"Échec de la vente forcée pour {pair}")
                
        except Exception as e:
            logging.error(f"Erreur lors de la vente forcée pour {pair}: {e}")
    
    def _make_trading_decisions(self, pair: str, analysis: Dict):
        """Prendre des décisions de trading pour une paire"""
        recommendation = analysis['recommendation']
        current_price = analysis['current_price']
        
        # Vérifier si on a déjà une position sur cette paire
        if pair in self.positions:
            logging.info(f"Position déjà ouverte sur {pair}, pas de nouveau trade")
            return
        
        # Vérifier si l'exposition doit être réduite
        if self.money_manager.should_reduce_exposure():
            logging.warning("Exposition réduite en raison du drawdown")
            return
        
        if recommendation['action'] == 'BUY' and self._should_buy(pair, analysis):
            self._execute_buy_order(pair, analysis)
        elif recommendation['action'] == 'SELL' and self._should_sell(pair, analysis):
            self._execute_sell_order(pair, analysis)
        else:
            logging.info(f"Aucune action de trading requise pour {pair}")
    
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
                        'leverage': leverage,
                        'entry_time': datetime.now(),
                        'stop_loss': self._calculate_stop_loss_price(pair, current_price, 'long')
                    }
                
                # Mettre à jour le money manager
                self.money_manager.update_position_size(pair, position_size)
                
                # Enregistrer le trade
                self._record_trade(pair, 'buy', position_size, current_price, 
                                 current_price, leverage)
                
                logging.info(f"Ordre d'achat exécuté pour {pair}: {position_size} @ {current_price}")
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
    
    def _update_positions(self, pair: str):
        """Mettre à jour les informations de position"""
        if pair in self.positions:
            current_price = self.active_client.get_current_price(pair)
            if current_price:
                position = self.positions[pair]
                position['current_price'] = current_price
                
                # Calculer le profit/perte non réalisé
                entry_price = position['entry_price']
                unrealized_pnl = (current_price - entry_price) * position['volume']
                if position.get('leverage'):
                    unrealized_pnl *= position['leverage']
                
                position['unrealized_pnl'] = unrealized_pnl
                position['unrealized_pnl_percent'] = (unrealized_pnl / (entry_price * position['volume'])) * 100
    
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