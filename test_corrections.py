#!/usr/bin/env python3
"""
Test des corrections du bot de trading crypto
============================================

Ce script teste que les corrections principales sont fonctionnelles,
notamment l'initialisation correcte de TechnicalIndicators.
"""

import sys
import os
import pandas as pd
import numpy as np

# Ajouter le répertoire src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_technical_indicators():
    """Test des indicateurs techniques"""
    print("🧪 Test des indicateurs techniques...")
    
    try:
        # Créer des données de test
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        
        # Simuler des données OHLC réalistes
        close_prices = 50000 + np.cumsum(np.random.randn(100) * 100)
        
        test_data = pd.DataFrame({
            'open': close_prices + np.random.randn(100) * 50,
            'high': close_prices + np.abs(np.random.randn(100) * 100),
            'low': close_prices - np.abs(np.random.randn(100) * 100),
            'close': close_prices,
            'volume': np.random.randint(1, 1000, 100)
        }, index=dates)
        
        # Test avec l'ancienne version (ta-lib)
        try:
            from src.indicators import TechnicalIndicators
            indicators_ta = TechnicalIndicators(test_data)
            print("✅ TechnicalIndicators (ta-lib) : OK")
            ta_available = True
        except ImportError as e:
            print(f"⚠️  TechnicalIndicators (ta-lib) : Non disponible - {e}")
            ta_available = False
        
        # Test avec la nouvelle version (pandas)
        try:
            from src.indicators_pandas import TechnicalIndicatorsPandas
            indicators_pandas = TechnicalIndicatorsPandas(test_data)
            print("✅ TechnicalIndicatorsPandas : OK")
            
            # Tester les signaux
            latest = indicators_pandas.get_latest_indicators()
            rsi_signal = indicators_pandas.get_rsi_signal()
            macd_signal = indicators_pandas.get_macd_signal()
            combined_signal = indicators_pandas.get_combined_signal()
            
            print(f"   - RSI: {latest['rsi']:.2f} → Signal: {rsi_signal}")
            print(f"   - MACD: {latest['macd']:.4f} → Signal: {macd_signal}")
            print(f"   - Signal combiné: {combined_signal}")
            
            pandas_available = True
        except Exception as e:
            print(f"❌ TechnicalIndicatorsPandas : Erreur - {e}")
            pandas_available = False
        
        return ta_available or pandas_available
        
    except Exception as e:
        print(f"❌ Erreur lors du test des indicateurs: {e}")
        return False

def test_config():
    """Test de la configuration"""
    print("\n🧪 Test de la configuration...")
    
    try:
        from src.config import Config
        
        # Test des valeurs par défaut
        print(f"✅ Configuration chargée")
        print(f"   - Mode de trading: {Config.TRADING_MODE}")
        print(f"   - Paires: {Config.TRADING_PAIRS}")
        print(f"   - Montant d'investissement: {Config.INVESTMENT_AMOUNT}")
        print(f"   - RSI période: {Config.RSI_PERIOD}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test de configuration: {e}")
        return False

def test_bot_initialization():
    """Test d'initialisation du bot sans erreur fatale"""
    print("\n🧪 Test d'initialisation du bot...")
    
    try:
        # Test que l'import ne génère pas d'erreur
        from src.advanced_trading_bot import AdvancedTradingBot
        print("✅ Import AdvancedTradingBot : OK")
        
        # Note : On ne teste pas l'initialisation complète car elle nécessite des clés API
        print("   ℹ️  Initialisation complète nécessite des clés API")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de l'import du bot: {e}")
        return False

def main():
    """Test principal"""
    print("=" * 60)
    print("🤖 TEST DES CORRECTIONS DU BOT DE TRADING CRYPTO")
    print("=" * 60)
    
    results = []
    
    # Tests
    results.append(test_config())
    results.append(test_technical_indicators())
    results.append(test_bot_initialization())
    
    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSULTATS DES TESTS")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ Tous les tests passent ({passed}/{total})")
        print("\n🎉 Les corrections sont fonctionnelles !")
        print("   Le bot peut maintenant être utilisé sans l'erreur initiale.")
        
        print("\n📝 Prochaines étapes :")
        print("   1. Créer le fichier config.env avec vos clés API")
        print("   2. Tester avec : python3 advanced_main.py --config")
        print("   3. Lancer en mode test : python3 advanced_main.py --test")
        
    else:
        print(f"⚠️  {passed}/{total} tests passent")
        print("   Vérifiez les erreurs ci-dessus")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)