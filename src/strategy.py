import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .config import Config
from .money_management import MoneyManager
try:
    from .indicators import TechnicalIndicators
except ImportError:
    from .indicators_pandas import TechnicalIndicatorsPandas as TechnicalIndicators

class TradingStrategy:
    """Stratégie de trading avec gestion des risques"""
    
    def __init__(self, kraken_client):
        """
        Initialiser la stratégie
        
        Args:
            kraken_client: Instance du client Kraken
        """
        self.kraken_client = kraken_client
        self.money_manager = MoneyManager(kraken_client)
        self.positions = []
        self.trade_history = []
        self.last_analysis = None
        
    def analyze_market(self, pair):
        """
        Analyser le marché et générer des signaux
        
        Args:
            pair (str): Paire de trading
            
        Returns:
            dict: Résultats de l'analyse
        """
        try:
            # Récupérer les données OHLC
            ohlc_data = self.kraken_client.get_ohlc_data(pair)
            if ohlc_data is None:
                return None
            if hasattr(ohlc_data, 'empty') and ohlc_data.empty:
                return None
            
            # Calculer les indicateurs techniques
            indicators = TechnicalIndicators(ohlc_data)
            
            # Obtenir les signaux
            signals = {
                'rsi': indicators.get_rsi_signal(),
                'macd': indicators.get_macd_signal(),
                'bollinger': indicators.get_bollinger_signal(),
                'ma': indicators.get_ma_signal(),
                'stochastic': indicators.get_stochastic_signal(),
                'combined': indicators.get_combined_signal()
            }
            
            # Obtenir les niveaux de support/résistance
            levels = indicators.get_support_resistance_levels()
            
            # Obtenir les dernières valeurs des indicateurs
            latest_indicators = indicators.get_latest_indicators()
            
            # Obtenir le prix actuel
            current_price = self.kraken_client.get_current_price(pair)
            
            analysis = {
                'timestamp': datetime.now(),
                'pair': pair,
                'current_price': current_price,
                'signals': signals,
                'levels': levels,
                'indicators': latest_indicators,
                'recommendation': self._get_recommendation(signals, current_price, levels)
            }
            
            self.last_analysis = analysis
            return analysis
            
        except Exception as e:
            print(f"Erreur lors de l'analyse du marché: {e}")
            return None
    
    def _get_recommendation(self, signals, current_price, levels):
        """
        Générer une recommandation basée sur les signaux
        
        Args:
            signals (dict): Signaux des indicateurs
            current_price (float): Prix actuel
            levels (dict): Niveaux de support/résistance
            
        Returns:
            dict: Recommandation de trading
        """
        if not current_price or not levels:
            return {'action': 'HOLD', 'reason': 'Données insuffisantes'}
        
        # Compter les signaux
        buy_count = sum(1 for signal in signals.values() if signal == 'BUY')
        sell_count = sum(1 for signal in signals.values() if signal == 'SELL')
        
        # Analyser la position par rapport aux niveaux
        support = levels.get('support')
        resistance = levels.get('resistance')
        
        # Calculer la force du signal
        signal_strength = abs(buy_count - sell_count) / len(signals)
        
        # Recommandation basée sur la majorité des signaux
        if buy_count > sell_count and signal_strength >= 0.4:
            # Vérifier si le prix est proche du support
            if support and current_price <= support * 1.02:  # 2% au-dessus du support
                return {
                    'action': 'BUY',
                    'reason': f'Signal d\'achat fort ({buy_count}/{len(signals)} indicateurs) - Prix proche du support',
                    'confidence': signal_strength,
                    'target_price': resistance,
                    'stop_loss': support * 0.98
                }
            else:
                return {
                    'action': 'BUY',
                    'reason': f'Signal d\'achat ({buy_count}/{len(signals)} indicateurs)',
                    'confidence': signal_strength,
                    'target_price': resistance,
                    'stop_loss': support * 0.98 if support else current_price * 0.95
                }
        
        elif sell_count > buy_count and signal_strength >= 0.4:
            # Vérifier si le prix est proche de la résistance
            if resistance and current_price >= resistance * 0.98:  # 2% en dessous de la résistance
                return {
                    'action': 'SELL',
                    'reason': f'Signal de vente fort ({sell_count}/{len(signals)} indicateurs) - Prix proche de la résistance',
                    'confidence': signal_strength,
                    'target_price': support,
                    'stop_loss': resistance * 1.02
                }
            else:
                return {
                    'action': 'SELL',
                    'reason': f'Signal de vente ({sell_count}/{len(signals)} indicateurs)',
                    'confidence': signal_strength,
                    'target_price': support,
                    'stop_loss': resistance * 1.02 if resistance else current_price * 1.05
                }
        
        else:
            return {
                'action': 'HOLD',
                'reason': f'Signaux neutres ({buy_count} achat, {sell_count} vente)',
                'confidence': 1 - signal_strength
            }
    
    def calculate_position_size(self, pair, current_price, signal_strength=0.5):
        """
        Calculer la taille de position optimale en utilisant le MoneyManager
        
        Args:
            pair (str): Paire de trading
            current_price (float): Prix actuel
            signal_strength (float): Force du signal (0-1)
            
        Returns:
            float: Taille de position en volume
        """
        # Calculer le stop-loss pour utiliser avec le MoneyManager
        stop_loss_price = current_price * (1 - Config.get_stop_loss_for_pair(pair) / 100)
        
        # Utiliser le MoneyManager pour calculer la taille de position
        position_value = self.money_manager.calculate_position_size(
            pair, signal_strength, current_price, stop_loss_price
        )
        
        # Convertir la valeur en volume
        volume = position_value / current_price
        
        return volume
    
    def should_buy(self, analysis):
        """
        Déterminer si on doit acheter
        
        Args:
            analysis (dict): Résultats de l'analyse
            
        Returns:
            bool: True si on doit acheter
        """
        if not analysis or analysis['recommendation']['action'] != 'BUY':
            return False
        
        # Vérifier la confiance du signal
        confidence = analysis['recommendation'].get('confidence', 0)
        if confidence < Config.MIN_SIGNAL_CONFIDENCE:
            return False
        
        # Vérifier si on a déjà une position ouverte
        if self.has_open_position(analysis['pair']):
            return False
        
        # Calculer la taille de position proposée pour vérifier les limites de risque
        pair = analysis['pair']
        current_price = analysis['current_price']
        stop_loss_price = current_price * (1 - Config.get_stop_loss_for_pair(pair) / 100)
        
        position_value = self.money_manager.calculate_position_size(
            pair, confidence, current_price, stop_loss_price
        )
        
        # Vérifier les limites de risque avec le MoneyManager
        if not self.money_manager.check_risk_limits(pair, position_value):
            print(f"Limites de risque dépassées pour {pair}")
            return False
        
        # Vérifier si l'exposition doit être réduite
        if self.money_manager.should_reduce_exposure():
            print("Exposition réduite en raison du drawdown")
            return False
        
        return True
    
    def should_sell(self, analysis):
        """
        Déterminer si on doit vendre
        
        Args:
            analysis (dict): Résultats de l'analyse
            
        Returns:
            bool: True si on doit vendre
        """
        if not analysis or analysis['recommendation']['action'] != 'SELL':
            return False
        
        # Vérifier la confiance du signal
        confidence = analysis['recommendation'].get('confidence', 0)
        if confidence < Config.MIN_SIGNAL_CONFIDENCE:
            return False
        
        # Vérifier si on a une position ouverte à vendre
        if not self.has_open_position(analysis['pair']):
            return False
        
        return True
    
    def has_open_position(self, pair):
        """
        Vérifier si on a une position ouverte
        
        Args:
            pair (str): Paire de trading
            
        Returns:
            bool: True si on a une position ouverte
        """
        # Vérifier dans l'historique des trades
        for trade in self.trade_history:
            if (trade['pair'] == pair and 
                trade['action'] == 'BUY' and 
                not trade.get('sold', False)):
                return True
        return False
    
    def get_open_position(self, pair):
        """
        Obtenir la position ouverte pour une paire
        
        Args:
            pair (str): Paire de trading
            
        Returns:
            dict: Position ouverte ou None
        """
        for trade in self.trade_history:
            if (trade['pair'] == pair and 
                trade['action'] == 'BUY' and 
                not trade.get('sold', False)):
                return trade
        return None
    
    def execute_buy_order(self, pair, analysis):
        """
        Exécuter un ordre d'achat avec TP/SL automatiques
        
        Args:
            pair (str): Paire de trading
            analysis (dict): Résultats de l'analyse
            
        Returns:
            bool: True si l'ordre a été exécuté avec succès
        """
        try:
            current_price = analysis['current_price']
            
            # Vérifier la disponibilité des fonds via le MoneyManager
            effective_capital = self.money_manager.get_effective_capital()
            if effective_capital <= 0:
                print("Solde insuffisant ou impossible d'obtenir le solde du compte")
                return False
            
            # Calculer la taille de position en utilisant la force du signal
            signal_strength = analysis['recommendation'].get('confidence', 0.5)
            volume = self.calculate_position_size(pair, current_price, signal_strength)
            
            # Calculer les niveaux TP/SL automatiques si activé
            tp_sl_levels = None
            if Config.should_use_auto_tp_sl():
                # Obtenir la valeur ATR si disponible
                atr_value = None
                if 'indicators' in analysis and 'atr' in analysis['indicators']:
                    atr_value = analysis['indicators']['atr']
                
                tp_sl_levels = Config.calculate_auto_tp_sl_levels(
                    entry_price=current_price,
                    pair=pair,
                    atr_value=atr_value
                )
                
                print(f"TP/SL automatiques calculés:")
                print(f"  Stop Loss: {tp_sl_levels['stop_loss']:.6f} ({tp_sl_levels['stop_loss_percent']:.2f}%)")
                print(f"  Take Profit: {tp_sl_levels['take_profit']:.6f} ({tp_sl_levels['take_profit_percent']:.2f}%)")
                print(f"  Ratio R/R: 1:{tp_sl_levels['risk_reward_ratio']:.1f}")
            
            # Placer l'ordre d'achat
            order = self.kraken_client.place_market_buy_order(pair, volume)
            
            if order:
                # Enregistrer le trade avec les niveaux TP/SL
                trade = {
                    'timestamp': datetime.now(),
                    'pair': pair,
                    'action': 'BUY',
                    'price': current_price,
                    'volume': volume,
                    'amount': volume * current_price,
                    'order_id': order.get('txid', [None])[0] if order else None,
                    'analysis': analysis,
                    'sold': False,
                    'tp_sl_levels': tp_sl_levels,
                    'auto_tp_sl_enabled': Config.should_use_auto_tp_sl()
                }
                
                self.trade_history.append(trade)
                print(f"Ordre d'achat exécuté: {volume} {pair[:4]} à {current_price}")
                
                # Placer les ordres TP/SL automatiques si activé
                if tp_sl_levels and Config.should_use_auto_tp_sl():
                    self._place_auto_tp_sl_orders(pair, volume, tp_sl_levels, trade)
                
                return True
            
            return False
            
        except Exception as e:
            print(f"Erreur lors de l'exécution de l'ordre d'achat: {e}")
            return False
    
    def _place_auto_tp_sl_orders(self, pair, volume, tp_sl_levels, trade):
        """
        Placer automatiquement les ordres de TP et SL
        
        Args:
            pair (str): Paire de trading
            volume (float): Volume de la position
            tp_sl_levels (dict): Niveaux de TP et SL
            trade (dict): Trade associé
        """
        try:
            # Placer l'ordre de stop-loss
            stop_loss_order = self.kraken_client.place_stop_loss_order(
                pair, volume, tp_sl_levels['stop_loss']
            )
            
            # Placer l'ordre de take-profit
            take_profit_order = self.kraken_client.place_take_profit_order(
                pair, volume, tp_sl_levels['take_profit']
            )
            
            # Enregistrer les ordres dans le trade
            trade['stop_loss_order'] = stop_loss_order
            trade['take_profit_order'] = take_profit_order
            
            if stop_loss_order:
                print(f"Ordre Stop-Loss placé à {tp_sl_levels['stop_loss']:.6f}")
            else:
                print("Échec du placement de l'ordre Stop-Loss")
            
            if take_profit_order:
                print(f"Ordre Take-Profit placé à {tp_sl_levels['take_profit']:.6f}")
            else:
                print("Échec du placement de l'ordre Take-Profit")
                
        except Exception as e:
            print(f"Erreur lors du placement des ordres TP/SL automatiques: {e}")
    
    def execute_sell_order(self, pair, analysis):
        """
        Exécuter un ordre de vente
        
        Args:
            pair (str): Paire de trading
            analysis (dict): Résultats de l'analyse
            
        Returns:
            bool: True si l'ordre a été exécuté avec succès
        """
        try:
            current_price = analysis['current_price']
            
            # Obtenir la position ouverte
            position = self.get_open_position(pair)
            if not position:
                print(f"Aucune position ouverte pour {pair}")
                return False
            
            # Placer l'ordre de vente
            order = self.kraken_client.place_market_sell_order(pair, position['volume'])
            
            if order:
                # Marquer la position comme vendue
                position['sold'] = True
                position['sell_price'] = current_price
                position['sell_timestamp'] = datetime.now()
                position['sell_order_id'] = order.get('txid', [None])[0] if order else None
                
                # Calculer le profit/perte
                profit_loss = (current_price - position['price']) * position['volume']
                profit_loss_percent = ((current_price - position['price']) / position['price']) * 100
                
                position['profit_loss'] = profit_loss
                position['profit_loss_percent'] = profit_loss_percent
                
                # Mettre à jour le MoneyManager avec le trade
                trade_data = {
                    'pair': pair,
                    'action': 'sell',
                    'volume': position['volume'],
                    'price': current_price,
                    'entry_price': position['price'],
                    'profit_loss': profit_loss,
                    'timestamp': datetime.now()
                }
                self.money_manager.add_trade(trade_data)
                
                print(f"Ordre de vente exécuté: {position['volume']} {pair[:4]} à {current_price}")
                print(f"Profit/Perte: {profit_loss:.2f} ({profit_loss_percent:.2f}%)")
                return True
            
            return False
            
        except Exception as e:
            print(f"Erreur lors de l'exécution de l'ordre de vente: {e}")
            return False
    
    def check_stop_loss_take_profit(self, pair, current_price):
        """
        Vérifier les stop-loss et take-profit avec support des niveaux automatiques
        
        Args:
            pair (str): Paire de trading
            current_price (float): Prix actuel
            
        Returns:
            str: Action à prendre ('SELL', 'HOLD')
        """
        position = self.get_open_position(pair)
        if not position:
            return 'HOLD'
        
        buy_price = position['price']
        
        # Vérifier si les TP/SL automatiques sont activés
        if position.get('auto_tp_sl_enabled') and position.get('tp_sl_levels'):
            tp_sl_levels = position['tp_sl_levels']
            
            # Vérifier le stop-loss automatique
            if current_price <= tp_sl_levels['stop_loss']:
                print(f"Stop-Loss automatique atteint: {current_price} <= {tp_sl_levels['stop_loss']}")
                return 'SELL'
            
            # Vérifier le take-profit automatique
            if current_price >= tp_sl_levels['take_profit']:
                print(f"Take-Profit automatique atteint: {current_price} >= {tp_sl_levels['take_profit']}")
                return 'SELL'
        
        else:
            # Méthode traditionnelle basée sur les pourcentages
            loss_percent = ((current_price - buy_price) / buy_price) * 100
            profit_percent = ((current_price - buy_price) / buy_price) * 100
            
            # Vérifier le stop-loss
            if loss_percent <= -Config.STOP_LOSS_PERCENTAGE:
                print(f"Stop-Loss traditionnel atteint: {loss_percent:.2f}% <= -{Config.STOP_LOSS_PERCENTAGE}%")
                return 'SELL'
            
            # Vérifier le take-profit
            if profit_percent >= Config.TAKE_PROFIT_PERCENTAGE:
                print(f"Take-Profit traditionnel atteint: {profit_percent:.2f}% >= {Config.TAKE_PROFIT_PERCENTAGE}%")
                return 'SELL'
        
        return 'HOLD'
    
    def get_performance_summary(self):
        """
        Obtenir un résumé des performances
        
        Returns:
            dict: Résumé des performances
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'open_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_profit_loss': 0,
                'win_rate': 0,
                'average_profit_loss': 0
            }
        
        completed_trades = [trade for trade in self.trade_history if trade.get('sold', False)]
        open_trades = [trade for trade in self.trade_history if not trade.get('sold', False)]
        
        if not completed_trades:
            return {
                'total_trades': len(self.trade_history),
                'open_trades': len(open_trades),
                'winning_trades': 0,
                'losing_trades': 0,
                'total_profit_loss': 0,
                'win_rate': 0,
                'average_profit_loss': 0
            }
        
        total_profit_loss = sum(trade.get('profit_loss', 0) for trade in completed_trades)
        winning_trades = len([t for t in completed_trades if t.get('profit_loss', 0) > 0])
        losing_trades = len([t for t in completed_trades if t.get('profit_loss', 0) < 0])
        
        win_rate = (winning_trades / len(completed_trades)) * 100 if completed_trades else 0
        
        return {
            'total_trades': len(completed_trades),
            'open_trades': len(open_trades),
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'total_profit_loss': total_profit_loss,
            'win_rate': win_rate,
            'average_profit_loss': total_profit_loss / len(completed_trades) if completed_trades else 0
        } 