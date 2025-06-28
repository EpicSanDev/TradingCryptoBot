"""
Client Kraken Avancé
===================

Client pour interagir avec l'API Kraken avec support des modes spot et futures.
"""

import krakenex
import pykrakenapi as kraken
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from typing import Optional, Dict, Any
from .config import Config

class AdvancedKrakenClient:
    """Client avancé pour interagir avec l'API Kraken"""
    
    def __init__(self, mode: str = 'spot'):
        """
        Initialiser le client Kraken
        
        Args:
            mode: 'spot' ou 'futures'
        """
        self.mode = mode
        self.last_api_call = 0
        self.min_call_interval = Config.MIN_API_INTERVAL  # Utiliser la config
        self.cache = {}
        self.cache_duration = Config.CACHE_DURATION  # Utiliser la config
        
        # Initialiser les clients selon le mode
        if mode == 'spot':
            credentials = Config.get_spot_credentials()
        else:
            credentials = Config.get_futures_credentials()
        
        self.api = krakenex.API(
            key=credentials['api_key'],
            secret=credentials['secret_key']
        )
        self.kraken = kraken.KrakenAPI(self.api)
        
        if Config.DEBUG_MODE:
            logging.info(f"Client Kraken {mode} initialisé avec cache de {self.cache_duration}s et intervalle API de {self.min_call_interval}s")
        else:
            logging.info(f"Client Kraken {mode} initialisé")
        
    def _rate_limit(self):
        """Respecter les limites de fréquence d'appel API avec gestion intelligente"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.min_call_interval:
            sleep_time = self.min_call_interval - time_since_last_call
            logging.debug(f"Rate limit: attente de {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_api_call = time.time()
        
    def _get_cached_data(self, key: str) -> Optional[Any]:
        """Récupérer des données en cache si elles sont encore valides"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_duration:
                return data
            else:
                del self.cache[key]
        return None
        
    def _cache_data(self, key: str, data: Any):
        """Mettre en cache des données avec timestamp"""
        self.cache[key] = (data, time.time())
        
    def get_account_balance(self) -> Optional[pd.DataFrame]:
        """Obtenir le solde du compte avec cache"""
        cache_key = f"balance_{self.mode}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
            
        try:
            self._rate_limit()
            balance = self.kraken.get_account_balance()
            if balance is None:
                logging.warning("Aucun solde trouvé")
                return pd.DataFrame()
            if hasattr(balance, 'empty') and balance.empty:
                logging.warning("Compte vide")
                return pd.DataFrame()
            
            self._cache_data(cache_key, balance)
            return balance
        except Exception as e:
            logging.error(f"Erreur lors de la récupération du solde: {e}")
            return pd.DataFrame()
    
    def get_ticker_info(self, pair: str) -> Optional[pd.DataFrame]:
        """Obtenir les informations du ticker avec cache"""
        cache_key = f"ticker_{pair}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
            
        try:
            self._rate_limit()
            ticker = self.kraken.get_ticker_information(pair)
            if ticker is not None:
                self._cache_data(cache_key, ticker)
            return ticker
        except Exception as e:
            logging.error(f"Erreur lors de la récupération du ticker pour {pair}: {e}")
            return None
    
    def get_ohlc_data(self, pair: str, interval: int = 1, since: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Obtenir les données OHLC (Open, High, Low, Close) avec cache optimisé"""
        cache_key = f"ohlc_{pair}_{interval}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
            
        try:
            self._rate_limit()
            if since is None:
                since = datetime.now() - timedelta(days=30)
            
            ohlc, last = self.kraken.get_ohlc_data(
                pair, 
                interval=interval, 
                since=since
            )
            
            if ohlc is not None:
                # Cache the data without modifying the frequency to avoid warnings
                self._cache_data(cache_key, ohlc)
            
            return ohlc
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des données OHLC pour {pair}: {e}")
            return None
    
    def get_current_price(self, pair: str) -> Optional[float]:
        """Obtenir le prix actuel d'une paire avec cache"""
        cache_key = f"price_{pair}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data
            
        try:
            ticker = self.get_ticker_info(pair)
            if ticker is None:
                return None
            if hasattr(ticker, 'empty') and ticker.empty:
                return None
            
            # Handle different possible data structures
            close_data = ticker['c']
            if isinstance(close_data, list):
                close_price = close_data[0]
            elif hasattr(close_data, 'iloc'):
                close_price = close_data.iloc[0]
            else:
                close_price = close_data
            
            # Ensure we have a valid number
            if isinstance(close_price, (list, tuple)):
                close_price = close_price[0]
            
            # Convert to float safely
            try:
                if isinstance(close_price, (int, float)):
                    price = float(close_price)
                elif isinstance(close_price, str):
                    price = float(close_price)
                else:
                    logging.error(f"Type de prix non supporté: {type(close_price)}")
                    return None
                    
                self._cache_data(cache_key, price)
                return price
            except (ValueError, TypeError):
                logging.error(f"Impossible de convertir le prix en float: {close_price}")
                return None
        except Exception as e:
            logging.error(f"Erreur lors de la récupération du prix pour {pair}: {e}")
            return None
    
    def place_market_order(self, pair: str, side: str, volume: float, 
                          leverage: Optional[float] = None) -> Optional[Dict]:
        """
        Placer un ordre au marché
        
        Args:
            pair: Paire de trading
            side: 'buy' ou 'sell'
            volume: Volume à trader
            leverage: Levier (pour les futures)
            
        Returns:
            Dictionnaire avec les détails de l'ordre
        """
        try:
            self._rate_limit()
            
            # Paramètres de l'ordre
            order_params = {
                'pair': pair,
                'type': side,
                'ordertype': 'market',
                'volume': str(volume)
            }
            
            # Ajouter le levier pour les futures
            if self.mode == 'futures' and leverage:
                order_params['leverage'] = str(leverage)
            
            # Placer l'ordre en utilisant l'API directe
            if side == 'buy':
                result = self.api.query_private('AddOrder', order_params)
            else:
                result = self.api.query_private('AddOrder', order_params)
            
            if result and 'error' not in result:
                order_info = {
                    'txid': result.get('result', {}).get('txid', []),
                    'pair': pair,
                    'side': side,
                    'volume': volume,
                    'leverage': leverage,
                    'mode': self.mode,
                    'timestamp': datetime.now()
                }
                
                logging.info(f"Ordre {side} placé: {order_info}")
                return order_info
            
            return None
            
        except Exception as e:
            logging.error(f"Erreur lors du placement de l'ordre {side} pour {pair}: {e}")
            return None
    
    def place_limit_order(self, pair: str, side: str, volume: float, price: float,
                         leverage: Optional[float] = None) -> Optional[Dict]:
        """
        Placer un ordre à cours limité
        
        Args:
            pair: Paire de trading
            side: 'buy' ou 'sell'
            volume: Volume à trader
            price: Prix limite
            leverage: Levier (pour les futures)
            
        Returns:
            Dictionnaire avec les détails de l'ordre
        """
        try:
            self._rate_limit()
            
            # Paramètres de l'ordre
            order_params = {
                'pair': pair,
                'type': side,
                'ordertype': 'limit',
                'volume': str(volume),
                'price': str(price)
            }
            
            # Ajouter le levier pour les futures
            if self.mode == 'futures' and leverage:
                order_params['leverage'] = str(leverage)
            
            # Placer l'ordre en utilisant l'API directe
            result = self.api.query_private('AddOrder', order_params)
            
            if result and 'error' not in result:
                order_info = {
                    'txid': result.get('result', {}).get('txid', []),
                    'pair': pair,
                    'side': side,
                    'volume': volume,
                    'price': price,
                    'leverage': leverage,
                    'mode': self.mode,
                    'timestamp': datetime.now()
                }
                
                logging.info(f"Ordre limité {side} placé: {order_info}")
                return order_info
            
            return None
            
        except Exception as e:
            logging.error(f"Erreur lors du placement de l'ordre limité {side} pour {pair}: {e}")
            return None
    
    def place_stop_loss_order(self, pair: str, side: str, volume: float, 
                             stop_price: float, leverage: Optional[float] = None) -> Optional[Dict]:
        """
        Placer un ordre stop-loss
        
        Args:
            pair: Paire de trading
            side: 'buy' ou 'sell'
            volume: Volume à trader
            stop_price: Prix de déclenchement
            leverage: Levier (pour les futures)
            
        Returns:
            Dictionnaire avec les détails de l'ordre
        """
        try:
            self._rate_limit()
            
            # Paramètres de l'ordre
            order_params = {
                'pair': pair,
                'type': side,
                'ordertype': 'stop-loss',
                'volume': str(volume),
                'price': str(stop_price)
            }
            
            # Ajouter le levier pour les futures
            if self.mode == 'futures' and leverage:
                order_params['leverage'] = str(leverage)
            
            # Placer l'ordre stop-loss
            result = self.api.query_private('AddOrder', order_params)
            
            if result and result.get('error') == []:
                order_info = {
                    'txid': result.get('result', {}).get('txid', []),
                    'pair': pair,
                    'side': side,
                    'volume': volume,
                    'stop_price': stop_price,
                    'leverage': leverage,
                    'mode': self.mode,
                    'timestamp': datetime.now()
                }
                
                logging.info(f"Ordre stop-loss {side} placé: {order_info}")
                return order_info
            
            return None
            
        except Exception as e:
            logging.error(f"Erreur lors du placement de l'ordre stop-loss {side} pour {pair}: {e}")
            return None
    
    def get_open_orders(self) -> Optional[pd.DataFrame]:
        """Obtenir les ordres ouverts"""
        try:
            self._rate_limit()
            orders = self.kraken.get_open_orders()
            return orders
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des ordres ouverts: {e}")
            return None
    
    def cancel_order(self, txid: str) -> bool:
        """Annuler un ordre"""
        try:
            self._rate_limit()
            result = self.kraken.cancel_order(txid)
            if result:
                logging.info(f"Ordre {txid} annulé avec succès")
                return True
            return False
        except Exception as e:
            logging.error(f"Erreur lors de l'annulation de l'ordre {txid}: {e}")
            return False
    
    def get_trade_history(self, pair: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Obtenir l'historique des trades"""
        try:
            self._rate_limit()
            trades = self.kraken.get_trade_history(pair=pair)
            return trades
        except Exception as e:
            logging.error(f"Erreur lors de la récupération de l'historique: {e}")
            return None
    
    def get_positions(self) -> Optional[Dict]:
        """Obtenir les positions ouvertes (pour les futures)"""
        if self.mode != 'futures':
            logging.warning("Les positions ne sont disponibles qu'en mode futures")
            return None
        
        try:
            self._rate_limit()
            # Utiliser l'API privée pour obtenir les positions futures
            result = self.api.query_private('OpenPositions')
            
            if result and result.get('error') == []:
                positions = result.get('result', {})
                return positions
            
            return None
            
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des positions: {e}")
            return None
    
    def get_futures_balance(self) -> Optional[Dict]:
        """Obtenir le solde du compte futures"""
        if self.mode != 'futures':
            logging.warning("Le solde futures n'est disponible qu'en mode futures")
            return None
        
        try:
            self._rate_limit()
            # Utiliser l'API privée pour obtenir le solde futures
            result = self.api.query_private('Balance')
            
            if result and result.get('error') == []:
                balance = result.get('result', {})
                return balance
            
            return None
            
        except Exception as e:
            logging.error(f"Erreur lors de la récupération du solde futures: {e}")
            return None
    
    def get_order_book(self, pair: str, count: int = 10) -> Optional[Dict]:
        """Obtenir le carnet d'ordres"""
        try:
            self._rate_limit()
            order_book = self.kraken.get_order_book(pair, count=count)
            return order_book
        except Exception as e:
            logging.error(f"Erreur lors de la récupération du carnet d'ordres pour {pair}: {e}")
            return None
    
    def get_recent_trades(self, pair: str, count: int = 100) -> Optional[pd.DataFrame]:
        """Obtenir les trades récents"""
        try:
            self._rate_limit()
            trades = self.kraken.get_recent_trades(pair, count=count)
            return trades
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des trades récents pour {pair}: {e}")
            return None
    
    def get_asset_info(self, asset: str) -> Optional[Dict]:
        """Obtenir les informations sur un actif"""
        try:
            self._rate_limit()
            info = self.kraken.get_asset_information(asset)
            return info
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des infos sur {asset}: {e}")
            return None
    
    def get_tradable_asset_pairs(self) -> Optional[pd.DataFrame]:
        """Obtenir les paires tradables"""
        try:
            self._rate_limit()
            pairs = self.kraken.get_tradable_asset_pairs()
            return pairs
        except Exception as e:
            logging.error(f"Erreur lors de la récupération des paires tradables: {e}")
            return None
    
    def validate_pair(self, pair: str) -> bool:
        """Valider qu'une paire est tradable"""
        try:
            pairs = self.get_tradable_asset_pairs()
            if pairs is not None and not pairs.empty:
                return pair in pairs.index
            return False
        except Exception as e:
            logging.error(f"Erreur lors de la validation de la paire {pair}: {e}")
            return False
    
    def get_server_time(self) -> Optional[Dict]:
        """Obtenir l'heure du serveur Kraken"""
        try:
            self._rate_limit()
            time_info = self.kraken.get_system_status()
            return time_info
        except Exception as e:
            logging.error(f"Erreur lors de la récupération de l'heure serveur: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Tester la connexion à l'API Kraken"""
        try:
            # Essayer de récupérer l'heure du serveur
            server_time = self.get_server_time()
            if server_time is not None:
                logging.info("Connexion à Kraken établie avec succès")
                return True
            return False
        except Exception as e:
            logging.error(f"Échec de la connexion à Kraken: {e}")
            return False 