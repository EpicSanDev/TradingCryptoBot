import krakenex
import pykrakenapi as kraken
import pandas as pd
from datetime import datetime, timedelta
import time
from .config import Config

class KrakenClient:
    """Client pour interagir avec l'API Kraken"""
    
    def __init__(self):
        """Initialiser le client Kraken"""
        self.api = krakenex.API(
            key=Config.KRAKEN_API_KEY,
            secret=Config.KRAKEN_SECRET_KEY
        )
        self.kraken = kraken.KrakenAPI(self.api)
        self.last_api_call = 0
        self.min_call_interval = 1  # 1 seconde minimum entre les appels
        
    def _rate_limit(self):
        """Respecter les limites de fréquence d'appel API"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        
        if time_since_last_call < self.min_call_interval:
            sleep_time = self.min_call_interval - time_since_last_call
            print(f"public call frequency exceeded (seconds={time_since_last_call:.6f}) \n sleeping for {sleep_time:.1f} seconds")
            time.sleep(sleep_time)
        
        self.last_api_call = time.time()
        
    def get_account_balance(self):
        """Obtenir le solde du compte"""
        try:
            self._rate_limit()
            balance = self.kraken.get_account_balance()
            if balance is None or balance.empty:
                print("Aucun solde trouvé ou compte vide")
                return pd.DataFrame()  # Retourner un DataFrame vide au lieu de None
            return balance
        except Exception as e:
            print(f"Erreur lors de la récupération du solde: {e}")
            return pd.DataFrame()  # Retourner un DataFrame vide au lieu de None
    
    def get_ticker_info(self, pair):
        """Obtenir les informations du ticker"""
        try:
            self._rate_limit()
            ticker = self.kraken.get_ticker_information(pair)
            return ticker
        except Exception as e:
            print(f"Erreur lors de la récupération du ticker: {e}")
            return None
    
    def get_ohlc_data(self, pair, interval=1, since=None):
        """Obtenir les données OHLC (Open, High, Low, Close)"""
        try:
            self._rate_limit()
            if since is None:
                since = datetime.now() - timedelta(days=30)
            
            ohlc, last = self.kraken.get_ohlc_data(
                pair, 
                interval=interval, 
                since=since
            )
            return ohlc
        except Exception as e:
            print(f"Erreur lors de la récupération des données OHLC: {e}")
            return None
    
    def place_market_buy_order(self, pair, volume):
        """Placer un ordre d'achat au marché"""
        try:
            self._rate_limit()
            order = self.kraken.create_market_buy_order(pair, volume)
            print(f"Ordre d'achat placé: {order}")
            return order
        except Exception as e:
            print(f"Erreur lors du placement de l'ordre d'achat: {e}")
            return None
    
    def place_market_sell_order(self, pair, volume):
        """Placer un ordre de vente au marché"""
        try:
            self._rate_limit()
            order = self.kraken.create_market_sell_order(pair, volume)
            print(f"Ordre de vente placé: {order}")
            return order
        except Exception as e:
            print(f"Erreur lors du placement de l'ordre de vente: {e}")
            return None
    
    def place_limit_buy_order(self, pair, volume, price):
        """Placer un ordre d'achat à cours limité"""
        try:
            self._rate_limit()
            order = self.kraken.create_limit_buy_order(pair, volume, price)
            print(f"Ordre d'achat limité placé: {order}")
            return order
        except Exception as e:
            print(f"Erreur lors du placement de l'ordre d'achat limité: {e}")
            return None
    
    def place_limit_sell_order(self, pair, volume, price):
        """Placer un ordre de vente à cours limité"""
        try:
            self._rate_limit()
            order = self.kraken.create_limit_sell_order(pair, volume, price)
            print(f"Ordre de vente limité placé: {order}")
            return order
        except Exception as e:
            print(f"Erreur lors du placement de l'ordre de vente limité: {e}")
            return None
    
    def get_open_orders(self):
        """Obtenir les ordres ouverts"""
        try:
            self._rate_limit()
            orders = self.kraken.get_open_orders()
            return orders
        except Exception as e:
            print(f"Erreur lors de la récupération des ordres ouverts: {e}")
            return None
    
    def cancel_order(self, txid):
        """Annuler un ordre"""
        try:
            self._rate_limit()
            result = self.kraken.cancel_order(txid)
            print(f"Ordre annulé: {result}")
            return result
        except Exception as e:
            print(f"Erreur lors de l'annulation de l'ordre: {e}")
            return None
    
    def get_trade_history(self, pair=None):
        """Obtenir l'historique des trades"""
        try:
            self._rate_limit()
            trades = self.kraken.get_trade_history(pair=pair)
            return trades
        except Exception as e:
            print(f"Erreur lors de la récupération de l'historique: {e}")
            return None
    
    def get_current_price(self, pair):
        """Obtenir le prix actuel d'une paire"""
        try:
            self._rate_limit()
            ticker = self.get_ticker_info(pair)
            if ticker is not None and not ticker.empty:
                # Le prix de clôture peut être une liste ou une valeur unique
                close_price = ticker['c'][0]
                if isinstance(close_price, list):
                    return float(close_price[0])
                elif isinstance(close_price, (str, int, float)):
                    return float(close_price)
                else:
                    # Si c'est un DataFrame ou autre, essayer de le convertir
                    return float(str(close_price))
            return None
        except Exception as e:
            print(f"Erreur lors de la récupération du prix actuel: {e}")
            return None 