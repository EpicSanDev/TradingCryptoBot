import os
import json
from dotenv import load_dotenv
from typing import Dict, List, Optional

# Charger les variables d'environnement
load_dotenv('config.env')

class Config:
    """Configuration du bot de trading avancé"""
    
    # === CLÉS API SPOT ===
    SPOT_API_KEY = os.getenv('SPOT_API_KEY')
    SPOT_SECRET_KEY = os.getenv('SPOT_SECRET_KEY')
    
    # === CLÉS API FUTURES ===
    FUTURES_API_KEY = os.getenv('FUTURES_API_KEY')
    FUTURES_SECRET_KEY = os.getenv('FUTURES_SECRET_KEY')
    
    # === CONFIGURATION MULTI-PAIRE ===
    TRADING_PAIRS = json.loads(os.getenv('TRADING_PAIRS', '["XXBTZEUR", "XETHZEUR"]'))
    
    # === CONFIGURATION DU TRADING ===
    TRADING_MODE = os.getenv('TRADING_MODE', 'spot')  # 'spot' ou 'futures'
    INVESTMENT_AMOUNT = float(os.getenv('INVESTMENT_AMOUNT', '1000'))
    TOTAL_CAPITAL = float(os.getenv('INVESTMENT_AMOUNT', '1000'))  # Alias pour compatibilité
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '0.1'))
    FIXED_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '0.1'))  # Alias pour compatibilité
    STOP_LOSS_PERCENTAGE = float(os.getenv('STOP_LOSS_PERCENTAGE', '5'))
    TAKE_PROFIT_PERCENTAGE = float(os.getenv('TAKE_PROFIT_PERCENTAGE', '10'))
    
    # === CONFIGURATION TP/SL AUTOMATIQUES ===
    USE_AUTO_TP_SL = os.getenv('USE_AUTO_TP_SL', 'true').lower() == 'true'
    RISK_REWARD_RATIO = float(os.getenv('RISK_REWARD_RATIO', '2.0'))  # Ratio 1:2 par défaut
    AUTO_TP_SL_METHOD = os.getenv('AUTO_TP_SL_METHOD', 'percentage')  # 'percentage' ou 'atr'
    ATR_MULTIPLIER_TP = float(os.getenv('ATR_MULTIPLIER_TP', '2.0'))
    ATR_MULTIPLIER_SL = float(os.getenv('ATR_MULTIPLIER_SL', '1.0'))
    USE_TRAILING_STOP = os.getenv('USE_TRAILING_STOP', 'false').lower() == 'true'
    TRAILING_STOP_PERCENTAGE = float(os.getenv('TRAILING_STOP_PERCENTAGE', '2.0'))
    
    # === CONFIGURATION POUR COMPATIBILITÉ ===
    TRADING_PAIR = os.getenv('TRADING_PAIR', 'XXBTZEUR')  # Pour compatibilité avec l'ancien bot
    KRAKEN_API_KEY = os.getenv('SPOT_API_KEY')  # Alias pour compatibilité
    KRAKEN_SECRET_KEY = os.getenv('SPOT_SECRET_KEY')  # Alias pour compatibilité
    
    # === MONEY MANAGEMENT AVANCÉ ===
    POSITION_SIZING_METHOD = os.getenv('POSITION_SIZING_METHOD', 'kelly')  # 'fixed', 'kelly', 'martingale'
    MAX_RISK_PER_TRADE = float(os.getenv('MAX_RISK_PER_TRADE', '2'))  # % du capital
    RISK_PER_TRADE = float(os.getenv('MAX_RISK_PER_TRADE', '2'))  # Alias pour compatibilité
    MAX_CORRELATED_RISK = float(os.getenv('MAX_CORRELATED_RISK', '5'))  # % du capital
    MAX_DRAWDOWN = float(os.getenv('MAX_DRAWDOWN', '20'))  # % du capital
    DRAWDOWN_REDUCTION = float(os.getenv('DRAWDOWN_REDUCTION', '0.5'))  # Facteur de réduction
    KELLY_FRACTION = float(os.getenv('KELLY_FRACTION', '0.25'))  # Fraction Kelly à utiliser
    MIN_SIGNAL_CONFIDENCE = float(os.getenv('MIN_SIGNAL_CONFIDENCE', '0.6'))
    MAX_POSITIONS = int(os.getenv('MAX_POSITIONS', '5'))  # Nombre max de positions simultanées
    
    # === CONFIGURATION FUTURES ===
    MAX_LEVERAGE = float(os.getenv('MAX_LEVERAGE', '10'))
    DEFAULT_LEVERAGE = float(os.getenv('DEFAULT_LEVERAGE', '3'))
    
    # === CONFIGURATION DES INDICATEURS TECHNIQUES ===
    RSI_PERIOD = int(os.getenv('RSI_PERIOD', '14'))
    RSI_OVERBOUGHT = float(os.getenv('RSI_OVERBOUGHT', '70'))
    RSI_OVERSOLD = float(os.getenv('RSI_OVERSOLD', '30'))
    
    MACD_FAST = int(os.getenv('MACD_FAST', '12'))
    MACD_SLOW = int(os.getenv('MACD_SLOW', '26'))
    MACD_SIGNAL = int(os.getenv('MACD_SIGNAL', '9'))
    
    BOLLINGER_PERIOD = int(os.getenv('BOLLINGER_PERIOD', '20'))
    BOLLINGER_STD = float(os.getenv('BOLLINGER_STD', '2'))
    
    MA_FAST = int(os.getenv('MA_FAST', '9'))
    MA_SLOW = int(os.getenv('MA_SLOW', '21'))
    
    # === INTERVALLE DE VÉRIFICATION ===
    CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '5'))
    
    # === CONFIGURATION PAR PAIRE ===
    PAIR_CONFIGS = json.loads(os.getenv('PAIR_CONFIGS', '{}'))
    
    # === OPTIMISATION DES PERFORMANCES ===
    CACHE_DURATION = int(os.getenv('CACHE_DURATION', '60'))  # Durée du cache en secondes
    MIN_API_INTERVAL = int(os.getenv('MIN_API_INTERVAL', '3'))  # Intervalle minimum entre appels API
    DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    
    # === NOTIFICATIONS ===
    NOTIFICATIONS_ENABLED = os.getenv('NOTIFICATIONS_ENABLED', 'false').lower() == 'true'
    EMAIL_NOTIFICATIONS = os.getenv('EMAIL_NOTIFICATIONS', 'false').lower() == 'true'
    WEBHOOK_NOTIFICATIONS = os.getenv('WEBHOOK_NOTIFICATIONS', 'false').lower() == 'true'
    
    # Configuration Email
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', '587'))
    EMAIL_FROM = os.getenv('EMAIL_FROM', '')
    EMAIL_TO = os.getenv('EMAIL_TO', '')
    EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')
    
    # Configuration Webhook
    WEBHOOK_URL = os.getenv('WEBHOOK_URL', '')
    
    @classmethod
    def get_spot_credentials(cls) -> dict:
        """Obtenir les credentials pour le mode spot"""
        if not cls.SPOT_API_KEY or not cls.SPOT_SECRET_KEY:
            raise ValueError("Les clés API Spot sont requises dans config.env")
        return {
            'api_key': cls.SPOT_API_KEY,
            'secret_key': cls.SPOT_SECRET_KEY
        }
    
    @classmethod
    def get_futures_credentials(cls) -> dict:
        """Obtenir les credentials pour le mode futures"""
        if not cls.FUTURES_API_KEY or not cls.FUTURES_SECRET_KEY:
            raise ValueError("Les clés API Futures sont requises dans config.env")
        return {
            'api_key': cls.FUTURES_API_KEY,
            'secret_key': cls.FUTURES_SECRET_KEY
        }
    
    @classmethod
    def get_stop_loss_for_pair(cls, pair: str) -> float:
        """Obtenir le stop-loss pour une paire spécifique"""
        if pair in cls.PAIR_CONFIGS:
            return cls.PAIR_CONFIGS[pair].get('stop_loss', cls.STOP_LOSS_PERCENTAGE)
        return cls.STOP_LOSS_PERCENTAGE
    
    @classmethod
    def get_take_profit_for_pair(cls, pair: str) -> float:
        """Obtenir le take-profit pour une paire spécifique"""
        if pair in cls.PAIR_CONFIGS:
            return cls.PAIR_CONFIGS[pair].get('take_profit', cls.TAKE_PROFIT_PERCENTAGE)
        return cls.TAKE_PROFIT_PERCENTAGE
    
    @classmethod
    def calculate_auto_tp_sl_levels(cls, entry_price: float, pair: Optional[str] = None, atr_value: Optional[float] = None) -> dict:
        """
        Calculer automatiquement les niveaux de TP et SL basés sur le ratio RR
        
        Args:
            entry_price (float): Prix d'entrée
            pair (str): Paire de trading (optionnel)
            atr_value (float): Valeur ATR (optionnel)
            
        Returns:
            dict: Niveaux de TP et SL
        """
        if cls.AUTO_TP_SL_METHOD == 'atr' and atr_value:
            # Méthode basée sur ATR
            stop_loss = entry_price - (atr_value * cls.ATR_MULTIPLIER_SL)
            take_profit = entry_price + (atr_value * cls.ATR_MULTIPLIER_TP)
        else:
            # Méthode basée sur pourcentage avec ratio RR
            stop_loss_percent = cls.get_stop_loss_for_pair(pair) if pair else cls.STOP_LOSS_PERCENTAGE
            take_profit_percent = stop_loss_percent * cls.RISK_REWARD_RATIO
            
            stop_loss = entry_price * (1 - stop_loss_percent / 100)
            take_profit = entry_price * (1 + take_profit_percent / 100)
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'stop_loss_percent': ((entry_price - stop_loss) / entry_price) * 100,
            'take_profit_percent': ((take_profit - entry_price) / entry_price) * 100,
            'risk_reward_ratio': cls.RISK_REWARD_RATIO
        }
    
    @classmethod
    def should_use_auto_tp_sl(cls) -> bool:
        """Déterminer si on doit utiliser les TP/SL automatiques"""
        return cls.USE_AUTO_TP_SL
    
    @classmethod
    def get_leverage_for_pair(cls, pair: str) -> float:
        """Obtenir le levier pour une paire spécifique (futures uniquement)"""
        if cls.TRADING_MODE != 'futures':
            return 1.0
        
        if pair in cls.PAIR_CONFIGS:
            return cls.PAIR_CONFIGS[pair].get('leverage', cls.DEFAULT_LEVERAGE)
        return cls.DEFAULT_LEVERAGE
    
    @classmethod
    def get_allocation_for_pair(cls, pair: str) -> float:
        """Obtenir l'allocation pour une paire spécifique"""
        if pair in cls.PAIR_CONFIGS:
            return cls.PAIR_CONFIGS[pair].get('allocation', 100.0 / len(cls.TRADING_PAIRS))
        return 100.0 / len(cls.TRADING_PAIRS)
    
    @classmethod
    def validate(cls):
        """Valider la configuration"""
        # Vérifier les clés API selon le mode
        if cls.TRADING_MODE == 'spot':
            cls.get_spot_credentials()
        elif cls.TRADING_MODE == 'futures':
            cls.get_futures_credentials()
        else:
            raise ValueError("TRADING_MODE doit être 'spot' ou 'futures'")
        
        # Vérifier les paires de trading
        if not cls.TRADING_PAIRS:
            raise ValueError("Au moins une paire de trading doit être configurée")
        
        # Vérifier les montants
        if cls.INVESTMENT_AMOUNT <= 0:
            raise ValueError("Le montant d'investissement doit être positif")
        
        if cls.MAX_RISK_PER_TRADE <= 0 or cls.MAX_RISK_PER_TRADE > 100:
            raise ValueError("MAX_RISK_PER_TRADE doit être entre 0 et 100")
        
        if cls.MAX_DRAWDOWN <= 0 or cls.MAX_DRAWDOWN > 100:
            raise ValueError("MAX_DRAWDOWN doit être entre 0 et 100")
        
        # Vérifier le levier
        if cls.TRADING_MODE == 'futures':
            if cls.DEFAULT_LEVERAGE <= 0 or cls.DEFAULT_LEVERAGE > cls.MAX_LEVERAGE:
                raise ValueError(f"DEFAULT_LEVERAGE doit être entre 0 et {cls.MAX_LEVERAGE}")
        
        return True
    
    @classmethod
    def print_config(cls):
        """Afficher la configuration actuelle"""
        print("\n=== CONFIGURATION ACTUELLE ===")
        print(f"Mode de trading: {cls.TRADING_MODE}")
        print(f"Paires de trading: {', '.join(cls.TRADING_PAIRS)}")
        print(f"Montant d'investissement: {cls.INVESTMENT_AMOUNT}")
        print(f"Taille max position: {cls.MAX_POSITION_SIZE * 100}%")
        print(f"Stop-loss par défaut: {cls.STOP_LOSS_PERCENTAGE}%")
        print(f"Take-profit par défaut: {cls.TAKE_PROFIT_PERCENTAGE}%")
        print(f"Intervalle de vérification: {cls.CHECK_INTERVAL} minutes")
        
        print("\n=== OPTIMISATION DES PERFORMANCES ===")
        print(f"Durée du cache: {cls.CACHE_DURATION} secondes")
        print(f"Intervalle minimum API: {cls.MIN_API_INTERVAL} secondes")
        print(f"Mode debug: {'Activé' if cls.DEBUG_MODE else 'Désactivé'}")
        
        print("\n=== MONEY MANAGEMENT ===")
        print(f"Méthode de sizing: {cls.POSITION_SIZING_METHOD}")
        print(f"Risque max par trade: {cls.MAX_RISK_PER_TRADE}%")
        print(f"Risque corrélé max: {cls.MAX_CORRELATED_RISK}%")
        print(f"Drawdown max: {cls.MAX_DRAWDOWN}%")
        print(f"Fraction Kelly: {cls.KELLY_FRACTION}")
        print(f"Confiance min signal: {cls.MIN_SIGNAL_CONFIDENCE}")
        
        if cls.TRADING_MODE == 'futures':
            print(f"\n=== FUTURES ===")
            print(f"Levier par défaut: {cls.DEFAULT_LEVERAGE}x")
            print(f"Levier max: {cls.MAX_LEVERAGE}x")
        
        print("\n=== INDICATEURS TECHNIQUES ===")
        print(f"RSI période: {cls.RSI_PERIOD} (survente: {cls.RSI_OVERSOLD}, surachat: {cls.RSI_OVERBOUGHT})")
        print(f"MACD: {cls.MACD_FAST}/{cls.MACD_SLOW}/{cls.MACD_SIGNAL}")
        print(f"Bollinger: {cls.BOLLINGER_PERIOD} périodes, {cls.BOLLINGER_STD}σ")
        print(f"Moving Averages: {cls.MA_FAST}/{cls.MA_SLOW}")
        
        if cls.PAIR_CONFIGS:
            print("\n=== CONFIGURATIONS PAR PAIRE ===")
            for pair, config in cls.PAIR_CONFIGS.items():
                print(f"{pair}:")
                if 'stop_loss' in config:
                    print(f"  Stop-loss: {config['stop_loss']}%")
                if 'take_profit' in config:
                    print(f"  Take-profit: {config['take_profit']}%")
                if 'leverage' in config:
                    print(f"  Levier: {config['leverage']}x")
                if 'allocation' in config:
                    print(f"  Allocation: {config['allocation']}%") 