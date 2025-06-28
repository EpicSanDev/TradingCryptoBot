"""
Module de gestion des données historiques pour backtest
====================================================

Ce module télécharge et prépare les données historiques de Kraken
pour les tests de stratégies de trading.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pickle
import logging
from typing import Optional, Dict, List, Tuple
from .advanced_kraken_client import AdvancedKrakenClient
from .config import Config

class BacktestDataManager:
    """Gestionnaire des données historiques pour backtest"""
    
    def __init__(self, cache_dir: str = "backtest_data"):
        """
        Initialiser le gestionnaire de données
        
        Args:
            cache_dir: Répertoire pour le cache des données
        """
        self.cache_dir = cache_dir
        self.kraken_client = AdvancedKrakenClient('spot')  # Utiliser le client spot
        
        # Créer le répertoire de cache
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        logging.info(f"Gestionnaire de données initialisé avec cache: {cache_dir}")
    
    def download_historical_data(self, pair: str, interval: int = 1, 
                                days: int = 365, force_update: bool = False) -> Optional[pd.DataFrame]:
        """
        Télécharger les données historiques pour une paire
        
        Args:
            pair: Paire de trading (ex: 'BTCEUR')
            interval: Intervalle en minutes (1, 5, 15, 30, 60, 240, 1440)
            days: Nombre de jours d'historique
            force_update: Forcer le téléchargement même si le cache existe
            
        Returns:
            DataFrame avec les données OHLCV
        """
        # Nom du fichier de cache
        cache_file = os.path.join(self.cache_dir, f"{pair}_{interval}m_{days}d.pkl")
        
        # Vérifier le cache si pas de mise à jour forcée
        if not force_update and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                
                # Vérifier la fraîcheur des données (moins de 1 heure)
                if datetime.now() - cached_data['timestamp'] < timedelta(hours=1):
                    logging.info(f"Utilisation du cache pour {pair} ({interval}m, {days}j)")
                    return cached_data['data']
            except Exception as e:
                logging.warning(f"Erreur lors de la lecture du cache: {e}")
        
        # Télécharger les nouvelles données
        logging.info(f"Téléchargement des données pour {pair} ({interval}m, {days}j)...")
        
        try:
            # Calculer la date de début
            since = datetime.now() - timedelta(days=days)
            
            # Télécharger les données OHLC
            ohlc_data = self.kraken_client.get_ohlc_data(pair, interval, since)
            
            if ohlc_data is None or ohlc_data.empty:
                logging.error(f"Aucune donnée obtenue pour {pair}")
                return None
            
            # Nettoyer et formater les données
            data = self._clean_and_format_data(ohlc_data)
            
            # Sauvegarder en cache
            cache_data = {
                'timestamp': datetime.now(),
                'data': data,
                'pair': pair,
                'interval': interval,
                'days': days
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logging.info(f"Données téléchargées et sauvées: {len(data)} points pour {pair}")
            return data
            
        except Exception as e:
            logging.error(f"Erreur lors du téléchargement des données pour {pair}: {e}")
            return None
    
    def _clean_and_format_data(self, ohlc_data: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoyer et formater les données OHLC
        
        Args:
            ohlc_data: Données brutes de Kraken
            
        Returns:
            DataFrame nettoyé et formaté
        """
        data = ohlc_data.copy()
        
        # Renommer les colonnes pour correspondre à nos besoins
        column_mapping = {
            'time': 'timestamp',
            'open': 'open',
            'high': 'high', 
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'vwap': 'vwap',
            'count': 'count'
        }
        
        # Renommer les colonnes existantes
        for old_col, new_col in column_mapping.items():
            if old_col in data.columns:
                data.rename(columns={old_col: new_col}, inplace=True)
        
        # Convertir les timestamps si nécessaire
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        else:
            # Utiliser l'index comme timestamp si pas de colonne timestamp
            data['timestamp'] = pd.to_datetime(data.index)
        
        # S'assurer que les colonnes OHLCV sont des floats
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Supprimer les lignes avec des NaN
        data.dropna(inplace=True)
        
        # Trier par timestamp
        data.sort_values('timestamp', inplace=True)
        data.reset_index(drop=True, inplace=True)
        
        return data
    
    def download_multiple_pairs(self, pairs: List[str], interval: int = 1, 
                               days: int = 365, force_update: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Télécharger les données pour plusieurs paires
        
        Args:
            pairs: Liste des paires de trading
            interval: Intervalle en minutes
            days: Nombre de jours d'historique
            force_update: Forcer le téléchargement
            
        Returns:
            Dictionnaire {pair: DataFrame}
        """
        data_dict = {}
        
        for pair in pairs:
            logging.info(f"Téléchargement de {pair}...")
            data = self.download_historical_data(pair, interval, days, force_update)
            
            if data is not None:
                data_dict[pair] = data
                logging.info(f"{pair}: {len(data)} points de données")
            else:
                logging.warning(f"Échec du téléchargement pour {pair}")
        
        logging.info(f"Téléchargement terminé: {len(data_dict)}/{len(pairs)} paires réussies")
        return data_dict
    
    def get_price_at_timestamp(self, data: pd.DataFrame, timestamp: datetime, 
                              price_type: str = 'close') -> Optional[float]:
        """
        Obtenir le prix à un timestamp donné
        
        Args:
            data: DataFrame avec les données
            timestamp: Timestamp recherché
            price_type: Type de prix ('open', 'high', 'low', 'close')
            
        Returns:
            Prix à ce timestamp
        """
        if data.empty or price_type not in data.columns:
            return None
        
        # Trouver l'index le plus proche
        idx = data['timestamp'].searchsorted(timestamp)
        
        if idx >= len(data):
            idx = len(data) - 1
        elif idx > 0:
            # Choisir l'index le plus proche
            if (timestamp - data['timestamp'].iloc[idx-1]) < (data['timestamp'].iloc[idx] - timestamp):
                idx = idx - 1
        
        return data[price_type].iloc[idx]
    
    def get_data_slice(self, data: pd.DataFrame, start_date: datetime, 
                      end_date: datetime) -> pd.DataFrame:
        """
        Obtenir une tranche de données entre deux dates
        
        Args:
            data: DataFrame avec les données
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            DataFrame avec les données de la période
        """
        if data.empty:
            return data
        
        mask = (data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)
        return data[mask].copy()
    
    def resample_data(self, data: pd.DataFrame, new_interval: str) -> pd.DataFrame:
        """
        Rééchantillonner les données à un nouvel intervalle
        
        Args:
            data: DataFrame avec les données
            new_interval: Nouvel intervalle ('5T', '15T', '1H', '1D', etc.)
            
        Returns:
            DataFrame rééchantillonné
        """
        if data.empty:
            return data
        
        # S'assurer que timestamp est l'index
        data_resampled = data.set_index('timestamp')
        
        # Rééchantillonner
        resampled = data_resampled.resample(new_interval).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # Remettre timestamp comme colonne
        resampled.reset_index(inplace=True)
        
        return resampled
    
    def add_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ajouter des features calculées aux données
        
        Args:
            data: DataFrame avec les données OHLCV
            
        Returns:
            DataFrame avec features supplémentaires
        """
        if data.empty:
            return data
        
        data = data.copy()
        
        # Returns
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Volatilité
        data['volatility_10'] = data['returns'].rolling(10).std()
        data['volatility_30'] = data['returns'].rolling(30).std()
        
        # Prix normalisés
        data['price_normalized'] = (data['close'] - data['close'].rolling(100).mean()) / data['close'].rolling(100).std()
        
        # Volume normalisé
        data['volume_normalized'] = (data['volume'] - data['volume'].rolling(20).mean()) / data['volume'].rolling(20).std()
        
        # Gaps
        data['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
        
        # Range
        data['true_range'] = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                np.abs(data['high'] - data['close'].shift(1)),
                np.abs(data['low'] - data['close'].shift(1))
            )
        )
        
        return data
    
    def validate_data(self, data: pd.DataFrame, pair: str) -> bool:
        """
        Valider la qualité des données
        
        Args:
            data: DataFrame à valider
            pair: Nom de la paire
            
        Returns:
            True si les données sont valides
        """
        if data is None or data.empty:
            logging.error(f"Données vides pour {pair}")
            return False
        
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logging.error(f"Colonnes manquantes pour {pair}: {missing_columns}")
            return False
        
        # Vérifier les valeurs NaN
        nan_counts = data[required_columns].isna().sum()
        if nan_counts.sum() > 0:
            logging.warning(f"Valeurs NaN trouvées pour {pair}: {nan_counts.to_dict()}")
        
        # Vérifier la cohérence des prix OHLC
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        ).sum()
        
        if invalid_ohlc > 0:
            logging.warning(f"Prix OHLC incohérents pour {pair}: {invalid_ohlc} lignes")
        
        # Vérifier les volumes négatifs
        negative_volume = (data['volume'] < 0).sum()
        if negative_volume > 0:
            logging.warning(f"Volumes négatifs pour {pair}: {negative_volume} lignes")
        
        logging.info(f"Validation des données pour {pair}: {len(data)} points, qualité OK")
        return True
    
    def get_data_info(self, data: pd.DataFrame, pair: str) -> Dict:
        """
        Obtenir des informations sur les données
        
        Args:
            data: DataFrame avec les données
            pair: Nom de la paire
            
        Returns:
            Dictionnaire avec les informations
        """
        if data.empty:
            return {}
        
        return {
            'pair': pair,
            'points': len(data),
            'start_date': data['timestamp'].min(),
            'end_date': data['timestamp'].max(),
            'duration_days': (data['timestamp'].max() - data['timestamp'].min()).days,
            'price_range': {
                'min': data['low'].min(),
                'max': data['high'].max(),
                'start': data['open'].iloc[0],
                'end': data['close'].iloc[-1]
            },
            'volume_stats': {
                'total': data['volume'].sum(),
                'mean': data['volume'].mean(),
                'max': data['volume'].max()
            }
        }