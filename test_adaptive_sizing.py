#!/usr/bin/env python3
"""
Script de test pour l'adaptation automatique de la taille des lots
================================================================

Ce script démontre comment le bot adapte automatiquement la taille des lots
en fonction des fonds réels du compte Kraken.
"""

import os
import sys
from datetime import datetime

# Ajouter le répertoire src au path
sys.path.append('src')

try:
    from dotenv import load_dotenv
    from src.config import Config
    from src.advanced_kraken_client import AdvancedKrakenClient
    from src.money_management import MoneyManager
except ImportError:
    # Fallback pour les imports directs
    from config import Config
    from advanced_kraken_client import AdvancedKrakenClient
    from money_management import MoneyManager

def test_adaptive_sizing():
    """Tester l'adaptation automatique de la taille des lots"""
    
    print("=== TEST DE L'ADAPTATION AUTOMATIQUE DES LOTS ===\n")
    
    try:
        # Charger la configuration
        load_dotenv('config.env')
        Config.validate()
        
        print(f"Configuration chargée:")
        print(f"  - Capital configuré: {Config.TOTAL_CAPITAL} EUR")
        print(f"  - Méthode de sizing: {Config.POSITION_SIZING_METHOD}")
        print(f"  - Risque max par trade: {Config.MAX_RISK_PER_TRADE}%")
        print()
        
        # Initialiser les clients
        print("Initialisation du client Kraken...")
        kraken_client = AdvancedKrakenClient('spot')
        
        if not kraken_client.test_connection():
            print("❌ Erreur : Impossible de se connecter à Kraken")
            return False
        
        print("✅ Connexion à Kraken établie")
        
        # Initialiser le MoneyManager
        money_manager = MoneyManager(kraken_client)
        
        # Test 1: Obtenir le solde réel du compte
        print("\n=== TEST 1: RÉCUPÉRATION DU SOLDE ===")
        success = money_manager.update_account_balance()
        
        if success:
            balance_summary = money_manager.get_balance_summary()
            print("✅ Solde récupéré avec succès:")
            print(f"  - Solde réel du compte: {balance_summary['account_balance']:.2f} EUR")
            print(f"  - Capital effectif: {balance_summary['effective_capital']:.2f} EUR")
            print(f"  - Utilisation du capital: {balance_summary['capital_utilization']:.1f}%")
        else:
            print("❌ Erreur lors de la récupération du solde")
            return False
        
        # Test 2: Calculer les tailles de position pour différentes paires
        print("\n=== TEST 2: CALCUL DES TAILLES DE POSITION ===")
        
        test_pairs = ['XXBTZEUR', 'XETHZEUR']
        current_prices = {}
        
        for pair in test_pairs:
            print(f"\nTest pour {pair}:")
            
            # Obtenir le prix actuel
            price = kraken_client.get_current_price(pair)
            if price is None:
                print(f"  ❌ Impossible d'obtenir le prix pour {pair}")
                continue
            
            current_prices[pair] = price
            print(f"  Prix actuel: {price:.2f} EUR")
            
            # Calculer différentes tailles de position selon la force du signal
            signal_strengths = [0.3, 0.6, 0.9]
            
            for signal_strength in signal_strengths:
                stop_loss_price = price * (1 - Config.get_stop_loss_for_pair(pair) / 100)
                
                position_size = money_manager.calculate_position_size(
                    pair, signal_strength, price, stop_loss_price
                )
                
                volume = position_size / price if price > 0 else 0
                
                print(f"  Signal {signal_strength:.1f}: {position_size:.2f} EUR (volume: {volume:.6f})")
        
        # Test 3: Vérifier les limites de risque
        print("\n=== TEST 3: VÉRIFICATION DES LIMITES DE RISQUE ===")
        
        for pair in test_pairs:
            if pair not in current_prices:
                continue
                
            price = current_prices[pair]
            stop_loss_price = price * (1 - Config.get_stop_loss_for_pair(pair) / 100)
            
            # Tester différentes tailles de position
            test_sizes = [
                balance_summary['effective_capital'] * 0.01,  # 1% du capital
                balance_summary['effective_capital'] * 0.05,  # 5% du capital
                balance_summary['effective_capital'] * 0.1,   # 10% du capital
            ]
            
            print(f"\nLimites de risque pour {pair}:")
            for size in test_sizes:
                risk_ok = money_manager.check_risk_limits(pair, size)
                percentage = (size / balance_summary['effective_capital']) * 100
                status = "✅ Acceptable" if risk_ok else "❌ Risque trop élevé"
                print(f"  {percentage:.1f}% du capital ({size:.2f} EUR): {status}")
        
        # Test 4: Simulation d'évolution du solde
        print("\n=== TEST 4: SIMULATION D'ÉVOLUTION DU SOLDE ===")
        
        # Simuler un trade perdant
        print("\nSimulation d'une perte de 200 EUR:")
        money_manager.update_balance(-200)
        
        new_metrics = money_manager.get_performance_metrics()
        print(f"  Nouveau drawdown: {new_metrics['current_drawdown']:.2f}%")
        
        if money_manager.should_reduce_exposure():
            factor = money_manager.get_position_reduction_factor()
            print(f"  ⚠️ Exposition réduite: facteur {factor:.2f}")
        else:
            print("  ✅ Exposition normale maintenue")
        
        # Test 5: Afficher les métriques complètes
        print("\n=== TEST 5: MÉTRIQUES COMPLÈTES ===")
        
        metrics = money_manager.get_performance_metrics()
        print(f"Balance compte: {metrics['account_balance']:.2f} EUR")
        print(f"Capital effectif: {metrics['effective_capital']:.2f} EUR")
        print(f"Drawdown actuel: {metrics['current_drawdown']:.2f}%")
        
        print("\n=== TEST TERMINÉ AVEC SUCCÈS ===")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False

def main():
    """Fonction principale"""
    
    print("Script de test - Adaptation automatique de la taille des lots")
    print("=" * 60)
    
    # Vérifier que le fichier de configuration existe
    if not os.path.exists('config.env'):
        print("❌ Erreur: Fichier config.env non trouvé")
        print("Copiez config.env.example vers config.env et configurez vos clés API")
        return
    
    # Exécuter les tests
    success = test_adaptive_sizing()
    
    if success:
        print("\n🎉 Tous les tests sont passés avec succès!")
        print("Le bot adaptera automatiquement la taille des lots en fonction des fonds de votre compte.")
    else:
        print("\n❌ Certains tests ont échoué. Vérifiez votre configuration.")

if __name__ == "__main__":
    main()