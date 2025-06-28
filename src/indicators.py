import pandas as pd
import numpy as np
import ta
from .config import Config

class TechnicalIndicators:
    """Classe pour calculer les indicateurs techniques"""
    
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
    
    def calculate_rsi(self):
        """Calculer le RSI (Relative Strength Index)"""
        self.data['rsi'] = ta.momentum.RSIIndicator(
            close=self.data['close'],
            window=Config.RSI_PERIOD
        ).rsi()
    
    def calculate_macd(self):
        """Calculer le MACD (Moving Average Convergence Divergence)"""
        macd = ta.trend.MACD(
            close=self.data['close'],
            window_fast=Config.MACD_FAST,
            window_slow=Config.MACD_SLOW,
            window_sign=Config.MACD_SIGNAL
        )
        self.data['macd'] = macd.macd()
        self.data['macd_signal'] = macd.macd_signal()
        self.data['macd_histogram'] = macd.macd_diff()
    
    def calculate_bollinger_bands(self):
        """Calculer les bandes de Bollinger"""
        bb = ta.volatility.BollingerBands(
            close=self.data['close'],
            window=Config.BOLLINGER_PERIOD,
            window_dev=Config.BOLLINGER_STD
        )
        self.data['bb_upper'] = bb.bollinger_hband()
        self.data['bb_middle'] = bb.bollinger_mavg()
        self.data['bb_lower'] = bb.bollinger_lband()
        self.data['bb_width'] = bb.bollinger_wband()
        self.data['bb_percent'] = bb.bollinger_pband()
    
    def calculate_moving_averages(self):
        """Calculer les moyennes mobiles"""
        self.data['ma_fast'] = ta.trend.SMAIndicator(
            close=self.data['close'],
            window=Config.MA_FAST
        ).sma_indicator()
        
        self.data['ma_slow'] = ta.trend.SMAIndicator(
            close=self.data['close'],
            window=Config.MA_SLOW
        ).sma_indicator()
        
        # Moyenne mobile exponentielle
        self.data['ema_12'] = ta.trend.EMAIndicator(
            close=self.data['close'],
            window=12
        ).ema_indicator()
        
        self.data['ema_26'] = ta.trend.EMAIndicator(
            close=self.data['close'],
            window=26
        ).ema_indicator()
    
    def calculate_stochastic(self):
        """Calculer l'oscillateur stochastique"""
        stoch = ta.momentum.StochasticOscillator(
            high=self.data['high'],
            low=self.data['low'],
            close=self.data['close']
        )
        self.data['stoch_k'] = stoch.stoch()
        self.data['stoch_d'] = stoch.stoch_signal()
    
    def calculate_atr(self):
        """Calculer l'ATR (Average True Range)"""
        self.data['atr'] = ta.volatility.AverageTrueRange(
            high=self.data['high'],
            low=self.data['low'],
            close=self.data['close']
        ).average_true_range()
    
    def get_latest_indicators(self):
        """Obtenir les dernières valeurs des indicateurs"""
        if self.data.empty:
            return None
        
        latest = self.data.iloc[-1]
        return {
            'price': latest['close'],
            'rsi': latest['rsi'],
            'macd': latest['macd'],
            'macd_signal': latest['macd_signal'],
            'macd_histogram': latest['macd_histogram'],
            'bb_upper': latest['bb_upper'],
            'bb_middle': latest['bb_middle'],
            'bb_lower': latest['bb_lower'],
            'bb_percent': latest['bb_percent'],
            'ma_fast': latest['ma_fast'],
            'ma_slow': latest['ma_slow'],
            'ema_12': latest['ema_12'],
            'ema_26': latest['ema_26'],
            'stoch_k': latest['stoch_k'],
            'stoch_d': latest['stoch_d'],
            'atr': latest['atr']
        }
    
    def get_rsi_signal(self):
        """Obtenir le signal RSI"""
        if self.data.empty or self.data['rsi'].isna().all():
            return 'NEUTRAL'
        
        latest_rsi = self.data['rsi'].iloc[-1]
        
        if latest_rsi > Config.RSI_OVERBOUGHT:
            return 'SELL'
        elif latest_rsi < Config.RSI_OVERSOLD:
            return 'BUY'
        else:
            return 'NEUTRAL'
    
    def get_macd_signal(self):
        """Obtenir le signal MACD"""
        if (self.data.empty or 
            self.data['macd'].isna().all() or 
            self.data['macd_signal'].isna().all()):
            return 'NEUTRAL'
        
        latest_macd = self.data['macd'].iloc[-1]
        latest_signal = self.data['macd_signal'].iloc[-1]
        latest_histogram = self.data['macd_histogram'].iloc[-1]
        
        # Signal basé sur le croisement MACD
        if len(self.data) >= 2:
            prev_macd = self.data['macd'].iloc[-2]
            prev_signal = self.data['macd_signal'].iloc[-2]
            
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
            self.data['bb_upper'].isna().all() or 
            self.data['bb_lower'].isna().all()):
            return 'NEUTRAL'
        
        latest_price = self.data['close'].iloc[-1]
        latest_upper = self.data['bb_upper'].iloc[-1]
        latest_lower = self.data['bb_lower'].iloc[-1]
        latest_percent = self.data['bb_percent'].iloc[-1]
        
        if latest_price <= latest_lower or latest_percent <= 0.2:
            return 'BUY'
        elif latest_price >= latest_upper or latest_percent >= 0.8:
            return 'SELL'
        else:
            return 'NEUTRAL'
    
    def get_ma_signal(self):
        """Obtenir le signal des moyennes mobiles"""
        if (self.data.empty or 
            self.data['ma_fast'].isna().all() or 
            self.data['ma_slow'].isna().all()):
            return 'NEUTRAL'
        
        latest_fast = self.data['ma_fast'].iloc[-1]
        latest_slow = self.data['ma_slow'].iloc[-1]
        
        if len(self.data) >= 2:
            prev_fast = self.data['ma_fast'].iloc[-2]
            prev_slow = self.data['ma_slow'].iloc[-2]
            
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
            self.data['stoch_k'].isna().all() or 
            self.data['stoch_d'].isna().all()):
            return 'NEUTRAL'
        
        latest_k = self.data['stoch_k'].iloc[-1]
        latest_d = self.data['stoch_d'].iloc[-1]
        
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
        if self.data.empty:
            return None
        
        # Utiliser les bandes de Bollinger comme niveaux
        latest = self.data.iloc[-1]
        
        return {
            'support': latest['bb_lower'],
            'resistance': latest['bb_upper'],
            'middle': latest['bb_middle']
        } 