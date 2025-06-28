"""
Module de gestion avancée du money management
============================================

Ce module implémente des stratégies avancées de gestion du capital et des risques
pour le trading multi-paire avec support spot et futures.
NOUVEAU: Adaptation automatique selon les fonds disponibles sur Kraken.
"""

import math
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from .config import Config

class MoneyManager:
    """Gestionnaire avancé du money management avec adaptation automatique du capital"""
    
    def __init__(self, kraken_client=None):
        """
        Initialiser le gestionnaire de money management
        
        Args:
            kraken_client: Client Kraken pour récupérer le solde en temps réel
        """
        self.kraken_client = kraken_client
        self.trade_history = []
        self.current_drawdown = 0.0
        self.peak_balance = Config.TOTAL_CAPITAL
        self.current_balance = Config.TOTAL_CAPITAL
        self.available_capital = Config.TOTAL_CAPITAL
        self.position_sizes = {}
        self.correlation_matrix = {}
        self.last_balance_update = None
        self.capital_currency = 'EUR'  # Devise principale pour les calculs
        
        # Seuils pour notifications de changement de capital
        self.capital_change_threshold = 0.1  # 10% de changement
        
        logging.info("MoneyManager initialisé avec adaptation automatique du capital")
    
    def update_available_capital(self) -> bool:
        """
        Mettre à jour le capital disponible depuis le compte Kraken
        
        Returns:
            True si la mise à jour a réussi, False sinon
        """
        if not self.kraken_client:
            logging.warning("Pas de client Kraken configuré, utilisation du capital statique")
            return False
        
        try:
            # Récupérer le solde du compte
            balance_df = self.kraken_client.get_account_balance()
            
            if balance_df is None or balance_df.empty:
                logging.warning("Impossible de récupérer le solde du compte")
                return False
            
            # Calculer le capital total disponible
            total_capital = self._calculate_total_capital(balance_df)
            
            if total_capital is None:
                logging.warning("Impossible de calculer le capital total")
                return False
            
            # Vérifier si le capital a significativement changé
            old_capital = self.available_capital
            capital_change_pct = abs(total_capital - old_capital) / old_capital if old_capital > 0 else 0
            
            if capital_change_pct > self.capital_change_threshold:
                logging.info(f"Changement significatif du capital: {old_capital:.2f} → {total_capital:.2f} "
                           f"({capital_change_pct*100:.1f}%)")
            
            # Mettre à jour le capital disponible
            self.available_capital = total_capital
            self.last_balance_update = datetime.now()
            
            # Ajuster le current_balance si c'est la première fois
            if self.current_balance == Config.TOTAL_CAPITAL:
                self.current_balance = total_capital
                self.peak_balance = total_capital
            
            logging.debug(f"Capital disponible mis à jour: {self.available_capital:.2f} {self.capital_currency}")
            return True
            
        except Exception as e:
            logging.error(f"Erreur lors de la mise à jour du capital: {e}")
            return False
    
    def _calculate_total_capital(self, balance_df) -> Optional[float]:
        """
        Calculer le capital total disponible depuis le DataFrame de solde
        
        Args:
            balance_df: DataFrame du solde du compte Kraken
            
        Returns:
            Capital total en devise principale (EUR par défaut)
        """
        try:
            total_value = 0.0
            
            # Parcourir toutes les devises du compte
            for currency, amount in balance_df.iterrows():
                if amount[0] <= 0:  # Ignorer les soldes négatifs ou nuls
                    continue
                
                currency_amount = float(amount[0])
                
                # Convertir en devise principale si nécessaire
                if currency == self.capital_currency:
                    total_value += currency_amount
                else:
                    # Convertir la devise en devise principale
                    converted_amount = self._convert_currency_to_base(currency, currency_amount)
                    if converted_amount is not None:
                        total_value += converted_amount
                        logging.debug(f"{currency}: {currency_amount} → {converted_amount:.2f} {self.capital_currency}")
            
            # Garder une marge de sécurité pour les frais
            safety_margin = 0.95  # 5% de marge
            usable_capital = total_value * safety_margin
            
            logging.info(f"Capital total: {total_value:.2f} {self.capital_currency} "
                        f"(utilisable: {usable_capital:.2f})")
            
            return usable_capital
            
        except Exception as e:
            logging.error(f"Erreur lors du calcul du capital total: {e}")
            return None
    
    def _convert_currency_to_base(self, currency: str, amount: float) -> Optional[float]:
        """
        Convertir une devise vers la devise principale
        
        Args:
            currency: Devise à convertir
            amount: Montant à convertir
            
        Returns:
            Montant converti en devise principale
        """
        try:
            if currency == self.capital_currency:
                return amount
            
            # Construire la paire de conversion vers EUR
            conversion_pairs = [
                f"X{currency}Z{self.capital_currency}",  # Format Kraken standard
                f"{currency}{self.capital_currency}",    # Format simple
                f"{currency}EUR",                        # Format direct
                f"X{currency}ZEUR"                       # Format Kraken avec EUR
            ]
            
            # Essayer de récupérer le prix de conversion
            if self.kraken_client:
                for pair in conversion_pairs:
                    try:
                        price = self.kraken_client.get_current_price(pair)
                        if price is not None:
                            converted = amount * price
                            logging.debug(f"Conversion {currency} → {self.capital_currency}: "
                                        f"{amount} * {price} = {converted:.2f}")
                            return converted
                    except:
                        continue
            
            # Si pas de conversion directe, essayer via BTC ou USD
            if currency not in ['XBT', 'USD', 'ZUSD']:
                btc_amount = self._convert_to_btc(currency, amount)
                if btc_amount is not None:
                    return self._convert_currency_to_base('XBT', btc_amount)
            
            # Valeurs par défaut pour les cryptos principales (approximatif)
            default_rates = {
                'XBT': 45000.0,   # Bitcoin ≈ 45k EUR
                'ETH': 2500.0,    # Ethereum ≈ 2.5k EUR
                'USD': 0.92,      # USD ≈ 0.92 EUR
                'ZUSD': 0.92,     # USD Kraken ≈ 0.92 EUR
                'ADA': 0.35,      # Cardano ≈ 0.35 EUR
                'DOT': 5.0,       # Polkadot ≈ 5 EUR
                'LINK': 12.0      # Chainlink ≈ 12 EUR
            }
            
            if currency in default_rates:
                rate = default_rates[currency]
                converted = amount * rate
                logging.warning(f"Utilisation du taux par défaut pour {currency}: "
                              f"{amount} * {rate} = {converted:.2f} {self.capital_currency}")
                return converted
            
            logging.warning(f"Impossible de convertir {currency} vers {self.capital_currency}")
            return None
            
        except Exception as e:
            logging.error(f"Erreur lors de la conversion {currency} → {self.capital_currency}: {e}")
            return None
    
    def _convert_to_btc(self, currency: str, amount: float) -> Optional[float]:
        """Convertir une devise vers BTC comme intermédiaire"""
        try:
            if not self.kraken_client:
                return None
                
            btc_pairs = [
                f"X{currency}XXBT",
                f"{currency}XBT",
                f"{currency}BTC"
            ]
            
            for pair in btc_pairs:
                try:
                    price = self.kraken_client.get_current_price(pair)
                    if price is not None:
                        return amount * price
                except:
                    continue
            
            return None
            
        except Exception as e:
            logging.error(f"Erreur lors de la conversion {currency} → BTC: {e}")
            return None
    
    def get_dynamic_capital(self) -> float:
        """
        Obtenir le capital disponible (dynamique ou statique)
        
        Returns:
            Capital disponible pour le trading
        """
        # Mettre à jour le capital si plus de 5 minutes depuis la dernière mise à jour
        if (self.last_balance_update is None or 
            datetime.now() - self.last_balance_update > timedelta(minutes=5)):
            self.update_available_capital()
        
        return self.available_capital

    def calculate_position_size(self, pair: str, signal_strength: float, 
                              current_price: float, stop_loss_price: float) -> float:
        """
        Calculer la taille optimale de position selon la méthode configurée
        NOUVEAU: Utilise le capital disponible dynamique
        
        Args:
            pair: Paire de trading
            signal_strength: Force du signal (0-1)
            current_price: Prix actuel
            stop_loss_price: Prix du stop-loss
            
        Returns:
            Taille de position en unités de base
        """
        # Utiliser le capital dynamique au lieu du capital statique
        dynamic_capital = self.get_dynamic_capital()
        
        method = Config.POSITION_SIZING_METHOD.lower()
        
        if method == 'fixed':
            return self._fixed_position_sizing_dynamic(pair, dynamic_capital)
        elif method == 'kelly':
            return self._kelly_position_sizing_dynamic(pair, signal_strength, current_price, 
                                                     stop_loss_price, dynamic_capital)
        elif method == 'martingale':
            return self._martingale_position_sizing_dynamic(pair, dynamic_capital)
        else:
            return self._fixed_position_sizing_dynamic(pair, dynamic_capital)
    
    def _fixed_position_sizing_dynamic(self, pair: str, dynamic_capital: float) -> float:
        """Taille de position fixe basée sur l'allocation et le capital dynamique"""
        allocation = Config.get_allocation_for_pair(pair)
        position_value = (dynamic_capital * allocation / 100) * Config.FIXED_POSITION_SIZE
        
        logging.debug(f"Position fixe pour {pair}: "
                     f"{dynamic_capital:.2f} * {allocation}% * {Config.FIXED_POSITION_SIZE} = {position_value:.2f}")
        
        return position_value
    
    def _kelly_position_sizing_dynamic(self, pair: str, signal_strength: float, 
                                     current_price: float, stop_loss_price: float,
                                     dynamic_capital: float) -> float:
        """
        Calcul Kelly pour optimiser la taille de position avec capital dynamique
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
        
        # Limiter à la taille maximale autorisée (basée sur le capital dynamique)
        max_size = self._get_max_position_size_dynamic(pair, dynamic_capital)
        kelly_size = max_size * max(0, min(1, kelly_fraction))
        
        logging.debug(f"Position Kelly pour {pair}: "
                     f"Kelly fraction={kelly_fraction:.3f}, max_size={max_size:.2f}, "
                     f"final_size={kelly_size:.2f}")
        
        return kelly_size
    
    def _martingale_position_sizing_dynamic(self, pair: str, dynamic_capital: float) -> float:
        """Sizing Martingale avec capital dynamique"""
        recent_trades = self._get_recent_trades(pair, 5)
        consecutive_losses = 0
        
        for trade in reversed(recent_trades):
            if trade['profit_loss'] < 0:
                consecutive_losses += 1
            else:
                break
        
        base_size = self._get_max_position_size_dynamic(pair, dynamic_capital)
        multiplier = 1.5 ** consecutive_losses  # Augmente de 50% par perte consécutive
        
        martingale_size = min(base_size * multiplier, base_size * 3)  # Limite à 3x
        
        logging.debug(f"Position Martingale pour {pair}: "
                     f"consecutive_losses={consecutive_losses}, multiplier={multiplier:.2f}, "
                     f"size={martingale_size:.2f}")
        
        return martingale_size
    
    def _get_max_position_size_dynamic(self, pair: str, dynamic_capital: float) -> float:
        """Obtenir la taille maximale de position pour une paire avec capital dynamique"""
        allocation = Config.get_allocation_for_pair(pair)
        max_risk_amount = dynamic_capital * (Config.MAX_RISK_PER_TRADE / 100)
        
        # Ajuster selon le levier pour les futures
        leverage = Config.get_leverage_for_pair(pair)
        max_size = max_risk_amount * leverage
        
        logging.debug(f"Max position pour {pair}: "
                     f"{dynamic_capital:.2f} * {Config.MAX_RISK_PER_TRADE}% * {leverage}x = {max_size:.2f}")
        
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
        max_risk = self._get_max_position_size_dynamic(pair, self.get_dynamic_capital())
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
        dynamic_capital = self.get_dynamic_capital()
        max_correlated_risk = dynamic_capital * (Config.MAX_CORRELATED_RISK / 100)
        
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
                'win_rate': 0.0,
                'total_profit_loss': 0.0,
                'average_profit_loss': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
        
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for t in self.trade_history if t.get('profit_loss', 0) > 0)
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
            'win_rate': win_rate,
            'total_profit_loss': total_profit_loss,
            'average_profit_loss': average_profit_loss,
            'current_drawdown': self.current_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
    
    def get_capital_info(self) -> Dict:
        """
        Obtenir les informations sur le capital
        
        Returns:
            Dictionnaire avec les informations de capital
        """
        return {
            'static_capital': Config.TOTAL_CAPITAL,
            'available_capital': self.available_capital,
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'last_update': self.last_balance_update,
            'capital_currency': self.capital_currency,
            'auto_update_enabled': self.kraken_client is not None
        }
    
    def force_capital_update(self) -> bool:
        """
        Forcer la mise à jour du capital disponible
        
        Returns:
            True si la mise à jour a réussi
        """
        logging.info("Mise à jour forcée du capital demandée")
        return self.update_available_capital()
    
    def set_capital_currency(self, currency: str):
        """
        Changer la devise principale pour les calculs
        
        Args:
            currency: Nouvelle devise principale (ex: 'EUR', 'USD', 'XBT')
        """
        old_currency = self.capital_currency
        self.capital_currency = currency
        logging.info(f"Devise principale changée: {old_currency} → {currency}")
        
        # Forcer une mise à jour du capital avec la nouvelle devise
        self.force_capital_update()
    
    def get_position_recommendations(self) -> Dict[str, Dict]:
        """
        Obtenir les recommandations de taille de position pour toutes les paires configurées
        
        Returns:
            Dictionnaire avec les recommandations par paire
        """
        recommendations = {}
        dynamic_capital = self.get_dynamic_capital()
        
        for pair in Config.TRADING_PAIRS:
            try:
                # Simuler un signal moyen pour le calcul
                mock_signal_strength = 0.7
                current_price = 50000.0  # Prix fictif pour le calcul
                stop_loss_price = current_price * 0.95  # Stop-loss à -5%
                
                recommended_size = self.calculate_position_size(
                    pair, mock_signal_strength, current_price, stop_loss_price
                )
                
                max_size = self._get_max_position_size_dynamic(pair, dynamic_capital)
                allocation = Config.get_allocation_for_pair(pair)
                
                recommendations[pair] = {
                    'recommended_size': recommended_size,
                    'max_size': max_size,
                    'allocation_percent': allocation,
                    'allocated_capital': dynamic_capital * allocation / 100
                }
                
            except Exception as e:
                logging.error(f"Erreur lors du calcul des recommandations pour {pair}: {e}")
                recommendations[pair] = {
                    'error': str(e)
                }
        
        return recommendations
    
    def log_capital_status(self):
        """Afficher le statut détaillé du capital"""
        capital_info = self.get_capital_info()
        
        logging.info("=== STATUT DU CAPITAL ===")
        logging.info(f"Capital statique configuré: {capital_info['static_capital']:.2f} {capital_info['capital_currency']}")
        logging.info(f"Capital disponible (dynamique): {capital_info['available_capital']:.2f} {capital_info['capital_currency']}")
        logging.info(f"Solde actuel: {capital_info['current_balance']:.2f} {capital_info['capital_currency']}")
        logging.info(f"Pic historique: {capital_info['peak_balance']:.2f} {capital_info['capital_currency']}")
        logging.info(f"Adaptation automatique: {'Activée' if capital_info['auto_update_enabled'] else 'Désactivée'}")
        
        if capital_info['last_update']:
            time_since_update = datetime.now() - capital_info['last_update']
            logging.info(f"Dernière mise à jour: il y a {time_since_update.total_seconds()/60:.1f} minutes")
        else:
            logging.info("Dernière mise à jour: Jamais")
        
        # Afficher les différences significatives
        capital_difference = capital_info['available_capital'] - capital_info['static_capital']
        difference_percent = (capital_difference / capital_info['static_capital']) * 100 if capital_info['static_capital'] > 0 else 0
        
        if abs(difference_percent) > 5:  # Si plus de 5% de différence
            status = "supérieur" if capital_difference > 0 else "inférieur"
            logging.info(f"⚠️  Capital dynamique {status} au capital statique: {difference_percent:+.1f}%")
    
    def get_sizing_method_info(self) -> str:
        """
        Obtenir des informations sur la méthode de sizing utilisée
        
        Returns:
            Description de la méthode
        """
        method = Config.POSITION_SIZING_METHOD.lower()
        
        descriptions = {
            'fixed': f"Méthode FIXE: {Config.FIXED_POSITION_SIZE * 100:.1f}% du capital alloué par paire",
            'kelly': f"Méthode KELLY: Optimisation mathématique (fraction: {Config.KELLY_FRACTION})",
            'martingale': f"Méthode MARTINGALE: Augmentation après pertes (max 3x la taille de base)"
        }
        
        return descriptions.get(method, f"Méthode inconnue: {method}") 