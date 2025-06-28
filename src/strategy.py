import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .config import Config
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
            if ohlc_data is None or ohlc_data.empty:
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
    
    def calculate_position_size(self, current_price, available_balance):
        """
        Calculer la taille de position optimale
        
        Args:
            current_price (float): Prix actuel
            available_balance (float): Solde disponible
            
        Returns:
            float: Taille de position en volume
        """
        # Utiliser le pourcentage maximum défini dans la config
        max_amount = available_balance * Config.MAX_POSITION_SIZE
        
        # Calculer le volume basé sur le prix actuel
        volume = max_amount / current_price
        
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
        if confidence < 0.4:
            return False
        
        # Vérifier si on a déjà une position ouverte
        if self.has_open_position(analysis['pair']):
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
        if confidence < 0.4:
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
        Exécuter un ordre d'achat
        
        Args:
            pair (str): Paire de trading
            analysis (dict): Résultats de l'analyse
            
        Returns:
            bool: True si l'ordre a été exécuté avec succès
        """
        try:
            current_price = analysis['current_price']
            
            # Obtenir le solde disponible
            balance = self.kraken_client.get_account_balance()
            if balance is None or balance.empty:
                print("Impossible d'obtenir le solde du compte ou compte vide")
                return False
            
            # Trouver le solde de la devise de base (ex: EUR pour XXBTZEUR)
            base_currency = pair[4:] if len(pair) > 4 else 'EUR'
            available_balance = 0
            
            # Vérifier si le DataFrame a des données
            if not balance.empty and base_currency in balance.index:
                available_balance = float(balance.loc[base_currency, 'vol'])
            else:
                print(f"Devise {base_currency} non trouvée dans le solde")
                return False
            
            if available_balance < Config.INVESTMENT_AMOUNT:
                print(f"Solde insuffisant: {available_balance} {base_currency}")
                return False
            
            # Calculer la taille de position
            volume = self.calculate_position_size(current_price, Config.INVESTMENT_AMOUNT)
            
            # Placer l'ordre d'achat
            order = self.kraken_client.place_market_buy_order(pair, volume)
            
            if order:
                # Enregistrer le trade
                trade = {
                    'timestamp': datetime.now(),
                    'pair': pair,
                    'action': 'BUY',
                    'price': current_price,
                    'volume': volume,
                    'amount': volume * current_price,
                    'order_id': order.get('txid', [None])[0] if order else None,
                    'analysis': analysis,
                    'sold': False
                }
                
                self.trade_history.append(trade)
                print(f"Ordre d'achat exécuté: {volume} {pair[:4]} à {current_price}")
                return True
            
            return False
            
        except Exception as e:
            print(f"Erreur lors de l'exécution de l'ordre d'achat: {e}")
            return False
    
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
                
                print(f"Ordre de vente exécuté: {position['volume']} {pair[:4]} à {current_price}")
                print(f"Profit/Perte: {profit_loss:.2f} ({profit_loss_percent:.2f}%)")
                return True
            
            return False
            
        except Exception as e:
            print(f"Erreur lors de l'exécution de l'ordre de vente: {e}")
            return False
    
    def check_stop_loss_take_profit(self, pair, current_price):
        """
        Vérifier les stop-loss et take-profit
        
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
        
        # Calculer les pourcentages
        loss_percent = ((current_price - buy_price) / buy_price) * 100
        profit_percent = ((current_price - buy_price) / buy_price) * 100
        
        # Vérifier le stop-loss
        if loss_percent <= -Config.STOP_LOSS_PERCENTAGE:
            return 'SELL'  # Stop-loss atteint
        
        # Vérifier le take-profit
        if profit_percent >= Config.TAKE_PROFIT_PERCENTAGE:
            return 'SELL'  # Take-profit atteint
        
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