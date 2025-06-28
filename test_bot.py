#!/usr/bin/env python3
"""
Script de test pour vérifier le bon fonctionnement du bot de trading
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.kraken_client import KrakenClient
from src.strategy import TradingStrategy

def test_connection():
    """Tester la connexion à Kraken"""
    print("🔌 Test de connexion à Kraken...")
    try:
        client = KrakenClient()
        balance = client.get_account_balance()
        print(f"✅ Connexion réussie")
        print(f"📊 Solde: {balance}")
        return True
    except Exception as e:
        print(f"❌ Erreur de connexion: {e}")
        return False

def test_price_retrieval():
    """Tester la récupération du prix"""
    print("\n💰 Test de récupération du prix...")
    try:
        client = KrakenClient()
        price = client.get_current_price(Config.TRADING_PAIR)
        print(f"✅ Prix récupéré: {price}")
        return True
    except Exception as e:
        print(f"❌ Erreur de récupération du prix: {e}")
        return False

def test_strategy():
    """Tester la stratégie de trading"""
    print("\n📈 Test de la stratégie...")
    try:
        client = KrakenClient()
        strategy = TradingStrategy(client)
        analysis = strategy.analyze_market(Config.TRADING_PAIR)
        
        if analysis:
            print("✅ Analyse de marché réussie")
            print(f"📊 Paire: {analysis['pair']}")
            print(f"💰 Prix actuel: {analysis['current_price']}")
            print(f"🎯 Recommandation: {analysis['recommendation']['action']}")
            print(f"📝 Raison: {analysis['recommendation']['reason']}")
            return True
        else:
            print("❌ Échec de l'analyse de marché")
            return False
    except Exception as e:
        print(f"❌ Erreur de stratégie: {e}")
        return False

def test_performance_summary():
    """Tester le résumé des performances"""
    print("\n📊 Test du résumé des performances...")
    try:
        client = KrakenClient()
        strategy = TradingStrategy(client)
        performance = strategy.get_performance_summary()
        print("✅ Résumé des performances généré")
        print(f"📈 Trades totaux: {performance['total_trades']}")
        print(f"🔓 Trades ouverts: {performance['open_trades']}")
        return True
    except Exception as e:
        print(f"❌ Erreur du résumé des performances: {e}")
        return False

def main():
    """Fonction principale de test"""
    print("🧪 Tests du bot de trading")
    print("=" * 50)
    
    # Valider la configuration
    try:
        Config.validate()
        print("✅ Configuration validée")
    except Exception as e:
        print(f"❌ Erreur de configuration: {e}")
        return
    
    # Tests
    tests = [
        test_connection,
        test_price_retrieval,
        test_strategy,
        test_performance_summary
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Résultats: {passed}/{total} tests réussis")
    
    if passed == total:
        print("🎉 Tous les tests sont passés! Le bot est prêt.")
    else:
        print("⚠️  Certains tests ont échoué. Vérifiez la configuration.")

if __name__ == "__main__":
    main() 