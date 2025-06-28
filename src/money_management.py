"""
Module de gestion avancée du money management
============================================

Ce module implémente des stratégies avancées de gestion du capital et des risques
pour le trading multi-paire avec support spot et futures.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from .config import Config

class MoneyManager:
    """Gestionnaire avancé du money management"""
    
    def __init__(self, kraken_client=None):
        """
        Initialiser le gestionnaire de money management
        
        Args:
            kraken_client: Instance du client Kraken pour récupérer le solde
        """
        self.kraken_client = kraken_client
        self.trade_history = []
        self.current_drawdown = 0.0
        self.peak_balance = Config.TOTAL_CAPITAL
        self.current_balance = Config.TOTAL_CAPITAL
        self.account_balance = Config.TOTAL_CAPITAL  # Solde réel du compte
        self.available_balance = Config.TOTAL_CAPITAL  # Solde disponible pour trading
        self.position_sizes = {}
        self.correlation_matrix = {}
        self.last_balance_update = None
        
    def update_account_balance(self) -> bool:
        """
        Mettre à jour le solde du compte depuis Kraken
        
        Returns:
            True si la mise à jour a réussi
        """
        if not self.kraken_client:
            return False
        
        try:
            # Récupérer le solde du compte
            balance_df = self.kraken_client.get_account_balance()
            
            if balance_df is None or balance_df.empty:
                return False
            
            # Calculer le solde total en EUR/USD
            total_balance = 0.0
            total_available = 0.0
            
            for asset, row in balance_df.iterrows():
                amount = row.iloc[0] if hasattr(row, 'iloc') else row
                if isinstance(amount, (int, float)) and amount > 0:
                    # Convertir en EUR si possible (simplifié)
                    if asset in ['ZEUR', 'EUR']:
                        total_balance += amount
                        total_available += amount
                    elif asset in ['ZUSD', 'USD']:
                        # Conversion approximative USD -> EUR (devrait utiliser le taux de change réel)
                        total_balance += amount * 0.85  
                        total_available += amount * 0.85
                    elif asset == 'XXBT':  # Bitcoin
                        # Obtenir le prix actuel du BTC
                        btc_price = self.kraken_client.get_current_price('XXBTZEUR')
                        if btc_price:
                            total_balance += amount * btc_price
                            total_available += amount * btc_price
                    elif asset == 'XETH':  # Ethereum
                        # Obtenir le prix actuel de l'ETH
                        eth_price = self.kraken_client.get_current_price('XETHZEUR')
                        if eth_price:
                            total_balance += amount * eth_price
                            total_available += amount * eth_price
                    # Ajouter d'autres cryptos si nécessaire
            
            # Mettre à jour les soldes
            old_balance = self.account_balance
            self.account_balance = total_balance
            self.available_balance = total_available
            self.last_balance_update = datetime.now()
            
            # Mettre à jour le peak balance si nécessaire
            if self.account_balance > self.peak_balance:
                self.peak_balance = self.account_balance
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = ((self.peak_balance - self.account_balance) / 
                                       self.peak_balance) * 100
            
            # Log de la mise à jour
            if abs(old_balance - self.account_balance) > 0.01:
                from .trading_bot import logging  # Import local pour éviter les cycles
                logging.info(f"Solde du compte mis à jour: {self.account_balance:.2f} EUR "
                           f"(disponible: {self.available_balance:.2f} EUR)")
            
            return True
            
        except Exception as e:
            from .trading_bot import logging  # Import local
            logging.error(f"Erreur lors de la mise à jour du solde: {e}")
            return False
    
    def get_effective_capital(self) -> float:
        """
        Obtenir le capital effectif à utiliser pour les calculs
        
        Returns:
            Capital effectif en tenant compte des préférences et du solde réel
        """
        # Si on n'a pas de client Kraken, utiliser la config
        if not self.kraken_client:
            return Config.TOTAL_CAPITAL
        
        # Mettre à jour le solde si c'est la première fois ou si c'est ancien
        if (self.last_balance_update is None or 
            datetime.now() - self.last_balance_update > timedelta(minutes=30)):
            self.update_account_balance()
        
        # Utiliser le minimum entre le solde configuré et le solde réel
        # Cela permet de limiter l'exposition même si le compte a plus de fonds
        effective_capital = min(Config.TOTAL_CAPITAL, self.available_balance)
        
        return max(effective_capital, 0.0)  # Assurer que c'est positif
        
    def calculate_position_size(self, pair: str, signal_strength: float, 
                              current_price: float, stop_loss_price: float) -> float:
        """
        Calculer la taille optimale de position selon la méthode configurée
        
        Args:
            pair: Paire de trading
            signal_strength: Force du signal (0-1)
            current_price: Prix actuel
            stop_loss_price: Prix du stop-loss
            
        Returns:
            Taille de position en unités de base
        """
        method = Config.POSITION_SIZING_METHOD.lower()
        
        if method == 'fixed':
            return self._fixed_position_sizing(pair)
        elif method == 'kelly':
            return self._kelly_position_sizing(pair, signal_strength, current_price, stop_loss_price)
        elif method == 'martingale':
            return self._martingale_position_sizing(pair)
        else:
            return self._fixed_position_sizing(pair)
    
    def _fixed_position_sizing(self, pair: str) -> float:
        """Taille de position fixe basée sur l'allocation et le capital effectif"""
        effective_capital = self.get_effective_capital()
        allocation = Config.get_allocation_for_pair(pair)
        position_value = (effective_capital * allocation / 100) * Config.FIXED_POSITION_SIZE
        return position_value
    
    def _kelly_position_sizing(self, pair: str, signal_strength: float, 
                             current_price: float, stop_loss_price: float) -> float:
        """
        Calcul Kelly pour optimiser la taille de position
        
        Kelly = (bp - q) / b
        où b = odds, p = probabilité de gain, q = probabilité de perte
        """
        # Calculer les odds (ratio risque/récompense)
        risk = abs(current_price - stop_loss_price)
        reward = abs(current_price * (Config.get_take_profit_for_pair(pair) / 100))
        odds = reward / risk if risk > 0 else 1
        
        # Estimer la probabilité de gain basée sur l'historique et la force du signal
        win_rate = self._get_historical_win_rate(pair)
        adjusted_win_rate = (win_rate + signal_strength) / 2
        
        # Calcul Kelly
        kelly_fraction = (odds * adjusted_win_rate - (1 - adjusted_win_rate)) / odds
        
        # Appliquer la fraction Kelly configurée
        kelly_fraction *= Config.KELLY_FRACTION
        
        # Limiter à la taille maximale autorisée
        max_size = self._get_max_position_size(pair)
        kelly_size = max_size * max(0, min(1, kelly_fraction))
        
        return kelly_size
    
    def _martingale_position_sizing(self, pair: str) -> float:
        """Sizing Martingale - augmente après les pertes"""
        recent_trades = self._get_recent_trades(pair, 5)
        consecutive_losses = 0
        
        for trade in reversed(recent_trades):
            if trade['profit_loss'] < 0:
                consecutive_losses += 1
            else:
                break
        
        base_size = self._get_max_position_size(pair)
        multiplier = 1.5 ** consecutive_losses  # Augmente de 50% par perte consécutive
        
        return min(base_size * multiplier, base_size * 3)  # Limite à 3x
    
    def _get_max_position_size(self, pair: str) -> float:
        """Obtenir la taille maximale de position pour une paire basée sur le capital effectif"""
        effective_capital = self.get_effective_capital()
        allocation = Config.get_allocation_for_pair(pair)
        max_risk_amount = effective_capital * (Config.MAX_RISK_PER_TRADE / 100)
        
        # Ajuster selon le levier pour les futures
        leverage = Config.get_leverage_for_pair(pair)
        max_size = max_risk_amount * leverage
        
        return max_size
    
    def _get_historical_win_rate(self, pair: str) -> float:
        """Obtenir le taux de réussite historique pour une paire"""
        pair_trades = [t for t in self.trade_history if t['pair'] == pair]
        
        if not pair_trades:
            return 0.5  # 50% par défaut
        
        winning_trades = sum(1 for t in pair_trades if t['profit_loss'] > 0)
        return winning_trades / len(pair_trades)
    
    def _get_recent_trades(self, pair: str, count: int) -> List[Dict]:
        """Obtenir les trades récents pour une paire"""
        pair_trades = [t for t in self.trade_history if t['pair'] == pair]
        return pair_trades[-count:] if len(pair_trades) >= count else pair_trades
    
    def check_risk_limits(self, pair: str, position_size: float) -> bool:
        """
        Vérifier si une position respecte les limites de risque
        
        Returns:
            True si la position est acceptable
        """
        # Vérifier le risque par trade
        max_risk = self._get_max_position_size(pair)
        if position_size > max_risk:
            return False
        
        # Vérifier le drawdown
        if self.current_drawdown > Config.MAX_DRAWDOWN:
            return False
        
        # Vérifier le risque corrélé
        if self._check_correlated_risk(pair, position_size):
            return False
        
        return True
    
    def _check_correlated_risk(self, pair: str, position_size: float) -> bool:
        """Vérifier le risque corrélé avec d'autres positions"""
        # Calculer le risque total des paires corrélées
        correlated_pairs = self._get_correlated_pairs(pair)
        total_correlated_risk = sum(
            self.position_sizes.get(cp, 0) for cp in correlated_pairs
        )
        
        total_correlated_risk += position_size
        effective_capital = self.get_effective_capital()
        max_correlated_risk = effective_capital * (Config.MAX_CORRELATED_RISK / 100)
        
        return total_correlated_risk > max_correlated_risk
    
    def _get_correlated_pairs(self, pair: str) -> List[str]:
        """Obtenir les paires corrélées (simplifié)"""
        # Dans une implémentation réelle, calculer la corrélation historique
        # Pour l'instant, utiliser des groupes prédéfinis
        crypto_groups = {
            'major': ['XXBTZEUR', 'XETHZEUR', 'ADAUSD'],
            'altcoins': ['ADAUSD', 'DOTUSD', 'LINKUSD'],
            'stablecoins': ['USDTUSD', 'USDCUSD']
        }
        
        for group in crypto_groups.values():
            if pair in group:
                return [p for p in group if p != pair]
        
        return []
    
    def update_balance(self, profit_loss: float):
        """Mettre à jour le solde et calculer le drawdown"""
        self.current_balance += profit_loss
        
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = ((self.peak_balance - self.current_balance) / 
                                   self.peak_balance) * 100
    
    def add_trade(self, trade_data: Dict):
        """Ajouter un trade à l'historique"""
        self.trade_history.append(trade_data)
        self.update_balance(trade_data.get('profit_loss', 0))
    
    def update_position_size(self, pair: str, size: float):
        """Mettre à jour la taille de position actuelle"""
        self.position_sizes[pair] = size
    
    def should_reduce_exposure(self) -> bool:
        """Déterminer si l'exposition doit être réduite"""
        return self.current_drawdown > Config.MAX_DRAWDOWN
    
    def get_position_reduction_factor(self) -> float:
        """Obtenir le facteur de réduction des positions"""
        if self.current_drawdown > Config.MAX_DRAWDOWN:
            return Config.DRAWDOWN_REDUCTION
        return 1.0
    
    def calculate_dynamic_stop_loss(self, pair: str, entry_price: float, 
                                  current_price: float, position_type: str) -> float:
        """
        Calculer un stop-loss dynamique basé sur l'ATR et les niveaux de support/résistance
        
        Args:
            pair: Paire de trading
            entry_price: Prix d'entrée
            current_price: Prix actuel
            position_type: 'long' ou 'short'
            
        Returns:
            Prix du stop-loss dynamique
        """
        if not Config.USE_TRAILING_STOP:
            return self._calculate_fixed_stop_loss(pair, entry_price, position_type)
        
        # Calculer le stop-loss basé sur l'ATR
        atr = self._get_atr(pair)
        if atr is None:
            return self._calculate_fixed_stop_loss(pair, entry_price, position_type)
        
        # Multiplier l'ATR par un facteur pour le stop-loss
        atr_multiplier = 2.0
        stop_distance = atr * atr_multiplier
        
        if position_type == 'long':
            # Stop-loss en dessous du prix actuel
            dynamic_stop = current_price - stop_distance
            # Ne jamais remonter le stop-loss
            fixed_stop = entry_price * (1 - Config.get_stop_loss_for_pair(pair) / 100)
            return max(dynamic_stop, fixed_stop)
        else:
            # Stop-loss au-dessus du prix actuel
            dynamic_stop = current_price + stop_distance
            # Ne jamais descendre le stop-loss
            fixed_stop = entry_price * (1 + Config.get_stop_loss_for_pair(pair) / 100)
            return min(dynamic_stop, fixed_stop)
    
    def _calculate_fixed_stop_loss(self, pair: str, entry_price: float, 
                                 position_type: str) -> float:
        """Calculer un stop-loss fixe"""
        stop_percentage = Config.get_stop_loss_for_pair(pair) / 100
        
        if position_type == 'long':
            return entry_price * (1 - stop_percentage)
        else:
            return entry_price * (1 + stop_percentage)
    
    def _get_atr(self, pair: str) -> Optional[float]:
        """Obtenir l'ATR (Average True Range) pour une paire"""
        # Dans une implémentation réelle, calculer l'ATR à partir des données OHLC
        # Pour l'instant, retourner une valeur par défaut
        return None
    
    def get_performance_metrics(self) -> Dict:
        """Obtenir les métriques de performance"""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_profit_loss': 0.0,
                'average_profit_loss': 0.0,
                'current_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
        
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for t in self.trade_history if t.get('profit_loss', 0) > 0)
        losing_trades = sum(1 for t in self.trade_history if t.get('profit_loss', 0) < 0)
        win_rate = (winning_trades / total_trades) * 100
        
        total_profit_loss = sum(t.get('profit_loss', 0) for t in self.trade_history)
        average_profit_loss = total_profit_loss / total_trades
        
        # Calculer le ratio de Sharpe simplifié
        returns = [t.get('profit_loss', 0) for t in self.trade_history]
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit_loss': total_profit_loss,
            'average_profit_loss': average_profit_loss,
            'current_drawdown': self.current_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'account_balance': self.account_balance,
            'available_balance': self.available_balance,
            'effective_capital': self.get_effective_capital()
        }
    
    def get_balance_summary(self) -> Dict:
        """
        Obtenir un résumé du solde du compte
        
        Returns:
            Dictionnaire avec les informations de solde
        """
        # Mettre à jour le solde si nécessaire
        if (self.last_balance_update is None or 
            datetime.now() - self.last_balance_update > timedelta(minutes=5)):
            self.update_account_balance()
        
        return {
            'account_balance': self.account_balance,
            'available_balance': self.available_balance,
            'effective_capital': self.get_effective_capital(),
            'configured_capital': Config.TOTAL_CAPITAL,
            'last_update': self.last_balance_update,
            'capital_utilization': (self.get_effective_capital() / Config.TOTAL_CAPITAL * 100) if Config.TOTAL_CAPITAL > 0 else 0
        } 