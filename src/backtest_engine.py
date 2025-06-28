"""
Moteur de Backtest pour Stratégies de Trading
===========================================

Ce module simule le trading avec les données historiques en utilisant
la stratégie existante du bot pour évaluer ses performances.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from .strategy import TradingStrategy
from .backtest_data import BacktestDataManager
from .config import Config

@dataclass
class BacktestTrade:
    """Représentation d'un trade de backtest"""
    timestamp: datetime
    pair: str
    action: str  # 'BUY' or 'SELL'
    price: float
    volume: float
    amount: float
    fees: float = 0.0
    profit_loss: Optional[float] = None
    profit_loss_percent: Optional[float] = None
    entry_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    hold_duration: Optional[timedelta] = None
    stop_loss_hit: bool = False
    take_profit_hit: bool = False
    analysis: Optional[Dict] = None

@dataclass
class BacktestPosition:
    """Position ouverte pendant le backtest"""
    pair: str
    entry_timestamp: datetime
    entry_price: float
    volume: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trade_id: str = None

class BacktestEngine:
    """Moteur de simulation de trading"""
    
    def __init__(self, initial_capital: float = 10000, commission_rate: float = 0.001):
        """
        Initialiser le moteur de backtest
        
        Args:
            initial_capital: Capital initial en euros
            commission_rate: Taux de commission (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_rate = commission_rate
        
        # Historique des trades et positions
        self.trades: List[BacktestTrade] = []
        self.positions: Dict[str, BacktestPosition] = {}
        self.equity_curve: List[Dict] = []
        
        # Métriques de performance
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit_loss = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.peak_equity = initial_capital
        
        # Gestionnaire de données
        self.data_manager = BacktestDataManager()
        
        logging.info(f"Moteur de backtest initialisé: Capital={initial_capital}€, Commission={commission_rate*100}%")
    
    def run_backtest(self, pairs: List[str], start_date: datetime, end_date: datetime,
                    interval: int = 60, days_history: int = 365) -> Dict:
        """
        Exécuter un backtest complet
        
        Args:
            pairs: Liste des paires à trader
            start_date: Date de début du backtest
            end_date: Date de fin du backtest
            interval: Intervalle de données en minutes
            days_history: Jours d'historique pour les indicateurs
            
        Returns:
            Résultats du backtest
        """
        logging.info(f"Démarrage du backtest: {start_date} à {end_date}")
        logging.info(f"Paires: {pairs}, Intervalle: {interval}m")
        
        # Télécharger les données historiques
        all_data = {}
        for pair in pairs:
            logging.info(f"Téléchargement des données pour {pair}...")
            data = self.data_manager.download_historical_data(
                pair, interval, days_history + (end_date - start_date).days
            )
            
            if data is not None and self.data_manager.validate_data(data, pair):
                all_data[pair] = data
                logging.info(f"{pair}: {len(data)} points de données")
            else:
                logging.warning(f"Impossible d'obtenir les données pour {pair}")
        
        if not all_data:
            logging.error("Aucune donnée disponible pour le backtest")
            return None
        
        # Initialiser la stratégie avec un client mock
        mock_client = MockKrakenClient(all_data)
        strategy = TradingStrategy(mock_client)
        
        # Exécuter la simulation
        return self._simulate_trading(all_data, strategy, start_date, end_date, interval)
    
    def _simulate_trading(self, all_data: Dict[str, pd.DataFrame], strategy: TradingStrategy,
                         start_date: datetime, end_date: datetime, interval: int) -> Dict:
        """
        Simuler le trading sur la période donnée
        
        Args:
            all_data: Données historiques pour toutes les paires
            strategy: Instance de la stratégie de trading
            start_date: Date de début
            end_date: Date de fin
            interval: Intervalle en minutes
            
        Returns:
            Résultats de la simulation
        """
        logging.info("Début de la simulation de trading...")
        
        # Créer une timeline unifiée
        timestamps = self._create_unified_timeline(all_data, start_date, end_date, interval)
        
        # Simuler chaque timestamp
        for i, current_time in enumerate(timestamps):
            if i % 1000 == 0:
                progress = (i / len(timestamps)) * 100
                logging.info(f"Progression: {progress:.1f}% ({current_time})")
            
            # Mettre à jour l'equity curve
            current_equity = self._calculate_current_equity(current_time, all_data)
            self._update_equity_curve(current_time, current_equity)
            
            # Vérifier les stops et take profits
            self._check_stop_take_profit(current_time, all_data)
            
            # Analyser chaque paire et générer des signaux
            for pair in all_data.keys():
                # Obtenir les données jusqu'à ce timestamp
                pair_data = self._get_data_until_timestamp(all_data[pair], current_time)
                
                if len(pair_data) < 50:  # Pas assez de données pour les indicateurs
                    continue
                
                # Simuler l'analyse de marché
                analysis = self._simulate_market_analysis(pair, pair_data, current_time, strategy)
                
                if analysis is None:
                    continue
                
                # Exécuter les décisions de trading
                if strategy.should_buy(analysis) and pair not in self.positions:
                    self._execute_buy(pair, analysis, current_time)
                elif strategy.should_sell(analysis) and pair in self.positions:
                    self._execute_sell(pair, analysis, current_time)
        
        # Clôturer toutes les positions ouvertes à la fin
        self._close_all_positions(end_date, all_data)
        
        # Calculer les métriques finales
        return self._calculate_final_results()
    
    def _create_unified_timeline(self, all_data: Dict[str, pd.DataFrame], 
                               start_date: datetime, end_date: datetime, interval: int) -> List[datetime]:
        """Créer une timeline unifiée pour toutes les paires"""
        all_timestamps = set()
        
        for pair, data in all_data.items():
            pair_timestamps = data[
                (data['timestamp'] >= start_date) & 
                (data['timestamp'] <= end_date)
            ]['timestamp'].tolist()
            all_timestamps.update(pair_timestamps)
        
        return sorted(list(all_timestamps))
    
    def _get_data_until_timestamp(self, data: pd.DataFrame, timestamp: datetime) -> pd.DataFrame:
        """Obtenir les données jusqu'à un timestamp donné"""
        return data[data['timestamp'] <= timestamp].copy()
    
    def _simulate_market_analysis(self, pair: str, data: pd.DataFrame, 
                                 timestamp: datetime, strategy: TradingStrategy) -> Optional[Dict]:
        """Simuler l'analyse de marché à un timestamp donné"""
        try:
            # Créer un client mock avec les données actuelles
            mock_client = MockKrakenClient({pair: data})
            
            # Analyser le marché avec la stratégie
            analysis = strategy.analyze_market(pair)
            
            if analysis:
                analysis['timestamp'] = timestamp
                analysis['current_price'] = data['close'].iloc[-1] if not data.empty else None
            
            return analysis
            
        except Exception as e:
            logging.debug(f"Erreur lors de l'analyse de {pair} à {timestamp}: {e}")
            return None
    
    def _execute_buy(self, pair: str, analysis: Dict, timestamp: datetime):
        """Exécuter un achat simulé"""
        try:
            current_price = analysis['current_price']
            if current_price is None or current_price <= 0:
                return
            
            # Calculer la taille de position (limité par le capital disponible)
            max_position_value = self.current_capital * 0.2  # Maximum 20% par position
            volume = max_position_value / current_price
            
            if volume * current_price < 10:  # Montant minimum
                return
            
            # Calculer les frais
            fees = volume * current_price * self.commission_rate
            total_cost = volume * current_price + fees
            
            if total_cost > self.current_capital:
                return  # Pas assez de capital
            
            # Créer le trade
            trade = BacktestTrade(
                timestamp=timestamp,
                pair=pair,
                action='BUY',
                price=current_price,
                volume=volume,
                amount=volume * current_price,
                fees=fees,
                analysis=analysis
            )
            
            # Créer la position
            position = BacktestPosition(
                pair=pair,
                entry_timestamp=timestamp,
                entry_price=current_price,
                volume=volume,
                trade_id=f"{pair}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Calculer stop loss et take profit si configurés
            if hasattr(Config, 'STOP_LOSS_PERCENTAGE'):
                position.stop_loss = current_price * (1 - Config.STOP_LOSS_PERCENTAGE / 100)
            if hasattr(Config, 'TAKE_PROFIT_PERCENTAGE'):
                position.take_profit = current_price * (1 + Config.TAKE_PROFIT_PERCENTAGE / 100)
            
            # Mettre à jour le capital et enregistrer
            self.current_capital -= total_cost
            self.positions[pair] = position
            self.trades.append(trade)
            self.total_trades += 1
            
            logging.debug(f"Achat exécuté: {volume:.4f} {pair} à {current_price:.2f}€")
            
        except Exception as e:
            logging.error(f"Erreur lors de l'exécution d'achat pour {pair}: {e}")
    
    def _execute_sell(self, pair: str, analysis: Dict, timestamp: datetime):
        """Exécuter une vente simulée"""
        try:
            if pair not in self.positions:
                return
            
            position = self.positions[pair]
            current_price = analysis['current_price']
            
            if current_price is None or current_price <= 0:
                return
            
            # Calculer les frais de vente
            gross_amount = position.volume * current_price
            fees = gross_amount * self.commission_rate
            net_amount = gross_amount - fees
            
            # Calculer le profit/perte
            total_cost = position.volume * position.entry_price
            profit_loss = net_amount - total_cost
            profit_loss_percent = (profit_loss / total_cost) * 100
            
            # Créer le trade de vente
            trade = BacktestTrade(
                timestamp=timestamp,
                pair=pair,
                action='SELL',
                price=current_price,
                volume=position.volume,
                amount=gross_amount,
                fees=fees,
                profit_loss=profit_loss,
                profit_loss_percent=profit_loss_percent,
                entry_price=position.entry_price,
                exit_timestamp=timestamp,
                hold_duration=timestamp - position.entry_timestamp,
                analysis=analysis
            )
            
            # Mettre à jour les statistiques
            self.current_capital += net_amount
            self.total_profit_loss += profit_loss
            
            if profit_loss > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1
            
            # Supprimer la position et enregistrer le trade
            del self.positions[pair]
            self.trades.append(trade)
            
            logging.debug(f"Vente exécutée: {position.volume:.4f} {pair} à {current_price:.2f}€, P&L: {profit_loss:.2f}€")
            
        except Exception as e:
            logging.error(f"Erreur lors de l'exécution de vente pour {pair}: {e}")
    
    def _check_stop_take_profit(self, timestamp: datetime, all_data: Dict[str, pd.DataFrame]):
        """Vérifier les stop loss et take profit"""
        positions_to_close = []
        
        for pair, position in self.positions.items():
            if pair not in all_data:
                continue
            
            current_price = self.data_manager.get_price_at_timestamp(
                all_data[pair], timestamp, 'close'
            )
            
            if current_price is None:
                continue
            
            # Vérifier stop loss
            if position.stop_loss and current_price <= position.stop_loss:
                positions_to_close.append((pair, current_price, 'STOP_LOSS'))
            
            # Vérifier take profit
            elif position.take_profit and current_price >= position.take_profit:
                positions_to_close.append((pair, current_price, 'TAKE_PROFIT'))
        
        # Fermer les positions
        for pair, price, reason in positions_to_close:
            self._close_position(pair, timestamp, price, reason)
    
    def _close_position(self, pair: str, timestamp: datetime, price: float, reason: str):
        """Fermer une position spécifique"""
        if pair not in self.positions:
            return
        
        position = self.positions[pair]
        
        # Calculer les frais de vente
        gross_amount = position.volume * price
        fees = gross_amount * self.commission_rate
        net_amount = gross_amount - fees
        
        # Calculer le profit/perte
        total_cost = position.volume * position.entry_price
        profit_loss = net_amount - total_cost
        profit_loss_percent = (profit_loss / total_cost) * 100
        
        # Créer le trade de vente
        trade = BacktestTrade(
            timestamp=timestamp,
            pair=pair,
            action='SELL',
            price=price,
            volume=position.volume,
            amount=gross_amount,
            fees=fees,
            profit_loss=profit_loss,
            profit_loss_percent=profit_loss_percent,
            entry_price=position.entry_price,
            exit_timestamp=timestamp,
            hold_duration=timestamp - position.entry_timestamp,
            stop_loss_hit=(reason == 'STOP_LOSS'),
            take_profit_hit=(reason == 'TAKE_PROFIT')
        )
        
        # Mettre à jour les statistiques
        self.current_capital += net_amount
        self.total_profit_loss += profit_loss
        
        if profit_loss > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Supprimer la position et enregistrer le trade
        del self.positions[pair]
        self.trades.append(trade)
        
        logging.debug(f"Position fermée ({reason}): {pair} à {price:.2f}€, P&L: {profit_loss:.2f}€")
    
    def _close_all_positions(self, end_date: datetime, all_data: Dict[str, pd.DataFrame]):
        """Fermer toutes les positions ouvertes à la fin du backtest"""
        for pair in list(self.positions.keys()):
            final_price = self.data_manager.get_price_at_timestamp(
                all_data[pair], end_date, 'close'
            )
            
            if final_price:
                self._close_position(pair, end_date, final_price, 'END_OF_BACKTEST')
    
    def _calculate_current_equity(self, timestamp: datetime, all_data: Dict[str, pd.DataFrame]) -> float:
        """Calculer l'equity actuelle incluant les positions ouvertes"""
        equity = self.current_capital
        
        for pair, position in self.positions.items():
            if pair in all_data:
                current_price = self.data_manager.get_price_at_timestamp(
                    all_data[pair], timestamp, 'close'
                )
                if current_price:
                    position_value = position.volume * current_price
                    equity += position_value - (position.volume * position.entry_price)
        
        return equity
    
    def _update_equity_curve(self, timestamp: datetime, equity: float):
        """Mettre à jour la courbe d'equity et calculer le drawdown"""
        # Mettre à jour le peak
        if equity > self.peak_equity:
            self.peak_equity = equity
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = ((self.peak_equity - equity) / self.peak_equity) * 100
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
        
        # Enregistrer dans la courbe d'equity
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'drawdown': self.current_drawdown,
            'peak_equity': self.peak_equity
        })
    
    def _calculate_final_results(self) -> Dict:
        """Calculer les résultats finaux du backtest"""
        if not self.trades:
            return {'error': 'Aucun trade exécuté'}
        
        # Métriques de base
        win_rate = (self.winning_trades / self.total_trades) * 100 if self.total_trades > 0 else 0
        total_return = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        
        # Calculs avancés
        trades_df = pd.DataFrame([
            {
                'timestamp': trade.timestamp,
                'pair': trade.pair,
                'action': trade.action,
                'price': trade.price,
                'volume': trade.volume,
                'profit_loss': trade.profit_loss or 0,
                'profit_loss_percent': trade.profit_loss_percent or 0,
                'hold_duration': trade.hold_duration.total_seconds() / 3600 if trade.hold_duration else 0,
                'fees': trade.fees
            }
            for trade in self.trades
        ])
        
        # Statistiques des trades
        profitable_trades = trades_df[trades_df['profit_loss'] > 0]
        losing_trades = trades_df[trades_df['profit_loss'] < 0]
        
        avg_win = profitable_trades['profit_loss'].mean() if len(profitable_trades) > 0 else 0
        avg_loss = abs(losing_trades['profit_loss'].mean()) if len(losing_trades) > 0 else 0
        profit_factor = (profitable_trades['profit_loss'].sum() / abs(losing_trades['profit_loss'].sum())) if len(losing_trades) > 0 else float('inf')
        
        # Ratio de Sharpe (simplifié)
        if len(self.equity_curve) > 1:
            equity_series = pd.Series([eq['equity'] for eq in self.equity_curve])
            returns = equity_series.pct_change().dropna()
            sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        results = {
            'summary': {
                'initial_capital': self.initial_capital,
                'final_capital': self.current_capital,
                'total_return': total_return,
                'total_profit_loss': self.total_profit_loss,
                'max_drawdown': self.max_drawdown,
                'sharpe_ratio': sharpe_ratio,
            },
            'trades_stats': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_fees': trades_df['fees'].sum(),
            },
            'time_stats': {
                'avg_hold_time_hours': trades_df['hold_duration'].mean(),
                'max_hold_time_hours': trades_df['hold_duration'].max(),
                'min_hold_time_hours': trades_df['hold_duration'].min(),
            },
            'equity_curve': self.equity_curve,
            'trades': self.trades,
            'trades_df': trades_df
        }
        
        logging.info("=== RÉSULTATS DU BACKTEST ===")
        logging.info(f"Capital initial: {self.initial_capital:.2f}€")
        logging.info(f"Capital final: {self.current_capital:.2f}€")
        logging.info(f"Rendement total: {total_return:.2f}%")
        logging.info(f"Trades totaux: {self.total_trades}")
        logging.info(f"Taux de réussite: {win_rate:.2f}%")
        logging.info(f"Drawdown maximum: {self.max_drawdown:.2f}%")
        logging.info(f"Ratio de Sharpe: {sharpe_ratio:.2f}")
        
        return results


class MockKrakenClient:
    """Client Kraken simulé pour le backtest"""
    
    def __init__(self, historical_data: Dict[str, pd.DataFrame]):
        self.historical_data = historical_data
        self.current_timestamp = None
    
    def get_ohlc_data(self, pair: str):
        """Retourner les données historiques pour une paire"""
        if pair in self.historical_data:
            return self.historical_data[pair]
        return None
    
    def get_current_price(self, pair: str):
        """Retourner le prix actuel (dernier prix disponible)"""
        if pair in self.historical_data:
            data = self.historical_data[pair]
            if not data.empty:
                return data['close'].iloc[-1]
        return None