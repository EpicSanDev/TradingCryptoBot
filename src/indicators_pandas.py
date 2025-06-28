"""
Indicateurs techniques utilisant pandas et numpy uniquement
Alternative à ta-lib pour éviter les dépendances système
"""

import pandas as pd
import numpy as np
from .config import Config

class TechnicalIndicatorsPandas:
    """Classe pour calculer les indicateurs techniques sans ta-lib"""
    
    def __init__(self, data):
        """
        Initialiser avec les données OHLC
        
        Args:
            data (pd.DataFrame): DataFrame avec les colonnes 'open', 'high', 'low', 'close', 'volume'
        """
        self.data = data.copy()
        self.calculate_all_indicators()
    
    def calculate_all_indicators(self):
        """Calculer tous les indicateurs techniques"""
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_bollinger_bands()
        self.calculate_moving_averages()
        self.calculate_stochastic()
        self.calculate_atr()
    
    def calculate_rsi(self, period=None):
        """Calculer le RSI (Relative Strength Index)"""
        if period is None:
            period = Config.RSI_PERIOD
            
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + rs))
    
    def calculate_macd(self, fast=None, slow=None, signal=None):
        """Calculer le MACD (Moving Average Convergence Divergence)"""
        if fast is None:
            fast = Config.MACD_FAST
        if slow is None:
            slow = Config.MACD_SLOW
        if signal is None:
            signal = Config.MACD_SIGNAL
            
        ema_fast = self.data['close'].ewm(span=fast).mean()
        ema_slow = self.data['close'].ewm(span=slow).mean()
        self.data['macd'] = ema_fast - ema_slow
        self.data['macd_signal'] = self.data['macd'].ewm(span=signal).mean()
        self.data['macd_histogram'] = self.data['macd'] - self.data['macd_signal']
    
    def calculate_bollinger_bands(self, period=None, std_dev=None):
        """Calculer les bandes de Bollinger"""
        if period is None:
            period = Config.BOLLINGER_PERIOD
        if std_dev is None:
            std_dev = Config.BOLLINGER_STD
            
        # Moyenne mobile simple
        sma = self.data['close'].rolling(window=period).mean()
        # Écart-type mobile
        std = self.data['close'].rolling(window=period).std()
        
        self.data['bb_upper'] = sma + (std * std_dev)
        self.data['bb_middle'] = sma
        self.data['bb_lower'] = sma - (std * std_dev)
        
        # Largeur des bandes (normalisée)
        self.data['bb_width'] = (self.data['bb_upper'] - self.data['bb_lower']) / self.data['bb_middle']
        
        # Position du prix dans les bandes (0-1)
        self.data['bb_percent'] = (self.data['close'] - self.data['bb_lower']) / (self.data['bb_upper'] - self.data['bb_lower'])
    
    def calculate_moving_averages(self, fast=None, slow=None):
        """Calculer les moyennes mobiles"""
        if fast is None:
            fast = Config.MA_FAST
        if slow is None:
            slow = Config.MA_SLOW
            
        # Moyennes mobiles simples
        self.data['ma_fast'] = self.data['close'].rolling(window=fast).mean()
        self.data['ma_slow'] = self.data['close'].rolling(window=slow).mean()
        
        # Moyennes mobiles exponentielles
        self.data['ema_12'] = self.data['close'].ewm(span=12).mean()
        self.data['ema_26'] = self.data['close'].ewm(span=26).mean()
    
    def calculate_stochastic(self, k_period=14, d_period=3):
        """Calculer l'oscillateur stochastique"""
        # Plus bas et plus haut sur la période
        low_min = self.data['low'].rolling(window=k_period).min()
        high_max = self.data['high'].rolling(window=k_period).max()
        
        # %K
        self.data['stoch_k'] = 100 * ((self.data['close'] - low_min) / (high_max - low_min))
        
        # %D (moyenne mobile de %K)
        self.data['stoch_d'] = self.data['stoch_k'].rolling(window=d_period).mean()
    
    def calculate_atr(self, period=14):
        """Calculer l'ATR (Average True Range)"""
        # True Range
        tr1 = self.data['high'] - self.data['low']
        tr2 = abs(self.data['high'] - self.data['close'].shift())
        tr3 = abs(self.data['low'] - self.data['close'].shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR (moyenne mobile du True Range)
        self.data['atr'] = true_range.rolling(window=period).mean()
    
    def get_latest_indicators(self):
        """Obtenir les dernières valeurs des indicateurs"""
        if self.data.empty:
            return None
        
        latest = self.data.iloc[-1]
        return {
            'price': latest['close'],
            'rsi': latest.get('rsi', np.nan),
            'macd': latest.get('macd', np.nan),
            'macd_signal': latest.get('macd_signal', np.nan),
            'macd_histogram': latest.get('macd_histogram', np.nan),
            'bb_upper': latest.get('bb_upper', np.nan),
            'bb_middle': latest.get('bb_middle', np.nan),
            'bb_lower': latest.get('bb_lower', np.nan),
            'bb_percent': latest.get('bb_percent', np.nan),
            'ma_fast': latest.get('ma_fast', np.nan),
            'ma_slow': latest.get('ma_slow', np.nan),
            'ema_12': latest.get('ema_12', np.nan),
            'ema_26': latest.get('ema_26', np.nan),
            'stoch_k': latest.get('stoch_k', np.nan),
            'stoch_d': latest.get('stoch_d', np.nan),
            'atr': latest.get('atr', np.nan)
        }
    
    def get_rsi_signal(self):
        """Obtenir le signal RSI"""
        if self.data.empty or 'rsi' not in self.data.columns or self.data['rsi'].isna().all():
            return 'NEUTRAL'
        
        latest_rsi = self.data['rsi'].iloc[-1]
        
        if pd.isna(latest_rsi):
            return 'NEUTRAL'
        
        if latest_rsi > Config.RSI_OVERBOUGHT:
            return 'SELL'
        elif latest_rsi < Config.RSI_OVERSOLD:
            return 'BUY'
        else:
            return 'NEUTRAL'
    
    def get_macd_signal(self):
        """Obtenir le signal MACD"""
        if (self.data.empty or 
            'macd' not in self.data.columns or 
            'macd_signal' not in self.data.columns or
            self.data['macd'].isna().all() or 
            self.data['macd_signal'].isna().all()):
            return 'NEUTRAL'
        
        latest_macd = self.data['macd'].iloc[-1]
        latest_signal = self.data['macd_signal'].iloc[-1]
        
        if pd.isna(latest_macd) or pd.isna(latest_signal):
            return 'NEUTRAL'
        
        # Signal basé sur le croisement MACD
        if len(self.data) >= 2:
            prev_macd = self.data['macd'].iloc[-2]
            prev_signal = self.data['macd_signal'].iloc[-2]
            
            if not (pd.isna(prev_macd) or pd.isna(prev_signal)):
                # Croisement haussier
                if prev_macd < prev_signal and latest_macd > latest_signal:
                    return 'BUY'
                # Croisement baissier
                elif prev_macd > prev_signal and latest_macd < latest_signal:
                    return 'SELL'
        
        return 'NEUTRAL'
    
    def get_bollinger_signal(self):
        """Obtenir le signal des bandes de Bollinger"""
        if (self.data.empty or 
            'bb_upper' not in self.data.columns or 
            'bb_lower' not in self.data.columns or
            'bb_percent' not in self.data.columns or
            self.data['bb_upper'].isna().all() or 
            self.data['bb_lower'].isna().all()):
            return 'NEUTRAL'
        
        latest_price = self.data['close'].iloc[-1]
        latest_upper = self.data['bb_upper'].iloc[-1]
        latest_lower = self.data['bb_lower'].iloc[-1]
        latest_percent = self.data['bb_percent'].iloc[-1]
        
        if pd.isna(latest_upper) or pd.isna(latest_lower) or pd.isna(latest_percent):
            return 'NEUTRAL'
        
        if latest_price <= latest_lower or latest_percent <= 0.2:
            return 'BUY'
        elif latest_price >= latest_upper or latest_percent >= 0.8:
            return 'SELL'
        else:
            return 'NEUTRAL'
    
    def get_ma_signal(self):
        """Obtenir le signal des moyennes mobiles"""
        if (self.data.empty or 
            'ma_fast' not in self.data.columns or 
            'ma_slow' not in self.data.columns or
            self.data['ma_fast'].isna().all() or 
            self.data['ma_slow'].isna().all()):
            return 'NEUTRAL'
        
        latest_fast = self.data['ma_fast'].iloc[-1]
        latest_slow = self.data['ma_slow'].iloc[-1]
        
        if pd.isna(latest_fast) or pd.isna(latest_slow):
            return 'NEUTRAL'
        
        if len(self.data) >= 2:
            prev_fast = self.data['ma_fast'].iloc[-2]
            prev_slow = self.data['ma_slow'].iloc[-2]
            
            if not (pd.isna(prev_fast) or pd.isna(prev_slow)):
                # Croisement haussier
                if prev_fast < prev_slow and latest_fast > latest_slow:
                    return 'BUY'
                # Croisement baissier
                elif prev_fast > prev_slow and latest_fast < latest_slow:
                    return 'SELL'
        
        return 'NEUTRAL'
    
    def get_stochastic_signal(self):
        """Obtenir le signal stochastique"""
        if (self.data.empty or 
            'stoch_k' not in self.data.columns or 
            'stoch_d' not in self.data.columns or
            self.data['stoch_k'].isna().all() or 
            self.data['stoch_d'].isna().all()):
            return 'NEUTRAL'
        
        latest_k = self.data['stoch_k'].iloc[-1]
        latest_d = self.data['stoch_d'].iloc[-1]
        
        if pd.isna(latest_k) or pd.isna(latest_d):
            return 'NEUTRAL'
        
        if latest_k < 20 and latest_d < 20:
            return 'BUY'
        elif latest_k > 80 and latest_d > 80:
            return 'SELL'
        else:
            return 'NEUTRAL'
    
    def get_combined_signal(self):
        """Obtenir un signal combiné de tous les indicateurs"""
        signals = {
            'rsi': self.get_rsi_signal(),
            'macd': self.get_macd_signal(),
            'bollinger': self.get_bollinger_signal(),
            'ma': self.get_ma_signal(),
            'stochastic': self.get_stochastic_signal()
        }
        
        # Compter les signaux
        buy_signals = sum(1 for signal in signals.values() if signal == 'BUY')
        sell_signals = sum(1 for signal in signals.values() if signal == 'SELL')
        
        # Décision basée sur la majorité
        if buy_signals >= 3:
            return 'BUY'
        elif sell_signals >= 3:
            return 'SELL'
        else:
            return 'NEUTRAL'
    
    def get_support_resistance_levels(self):
        """Calculer les niveaux de support et résistance"""
        if self.data.empty or 'bb_upper' not in self.data.columns:
            return None
        
        # Utiliser les bandes de Bollinger comme niveaux
        latest = self.data.iloc[-1]
        
        return {
            'support': latest.get('bb_lower', None),
            'resistance': latest.get('bb_upper', None),
            'middle': latest.get('bb_middle', None)
        }