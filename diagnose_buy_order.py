#!/usr/bin/env python3
"""
Script de diagnostic pour identifier les problèmes d'ordre d'achat
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.advanced_kraken_client import AdvancedKrakenClient
from src.config import Config
from src.money_management import MoneyManager
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def diagnose_buy_order_issue():
    """Diagnostiquer les problèmes d'ordre d'achat"""
    
    print("=== DIAGNOSTIC DES ORDRES D'ACHAT ===\n")
    
    # 1. Vérifier la configuration
    print("1. VÉRIFICATION DE LA CONFIGURATION")
    print(f"   Mode de trading: {Config.TRADING_MODE}")
    print(f"   Capital configuré: {Config.TOTAL_CAPITAL} EUR")
    print(f"   Taille max position: {Config.MAX_POSITION_SIZE}%")
    print(f"   Paires configurées: {Config.TRADING_PAIRS}")
    print(f"   Configuration par paire: {Config.PAIR_CONFIGS}")
    print()
    
    # 2. Tester la connexion à Kraken
    print("2. TEST DE CONNEXION KRAKEN")
    try:
        client = AdvancedKrakenClient(mode=Config.TRADING_MODE)
        if client.test_connection():
            print("   ✓ Connexion à Kraken réussie")
        else:
            print("   ✗ Échec de la connexion à Kraken")
            return
    except Exception as e:
        print(f"   ✗ Erreur de connexion: {e}")
        return
    print()
    
    # 3. Vérifier le solde du compte
    print("3. VÉRIFICATION DU SOLDE")
    try:
        balance = client.get_account_balance()
        if balance is not None and not balance.empty:
            print("   ✓ Solde récupéré")
            print(f"   Solde: {balance}")
        else:
            print("   ✗ Impossible de récupérer le solde")
            print("   Solde vide ou erreur API")
    except Exception as e:
        print(f"   ✗ Erreur lors de la récupération du solde: {e}")
    print()
    
    # 4. Tester la récupération du prix pour XXBTZEUR
    print("4. TEST DU PRIX XXBTZEUR")
    try:
        price = client.get_current_price('XXBTZEUR')
        if price:
            print(f"   ✓ Prix actuel XXBTZEUR: {price} EUR")
        else:
            print("   ✗ Impossible de récupérer le prix XXBTZEUR")
            return
    except Exception as e:
        print(f"   ✗ Erreur lors de la récupération du prix: {e}")
        return
    print()
    
    # 5. Calculer la taille de position
    print("5. CALCUL DE LA TAILLE DE POSITION")
    try:
        money_manager = MoneyManager(client)
        
        # Simuler une analyse
        analysis = {
            'current_price': price,
            'recommendation': {'confidence': 0.7}
        }
        
        # Calculer la taille de position
        position_size = money_manager.calculate_position_size(
            'XXBTZEUR', 0.7, price, price * 0.95  # Stop-loss à -5%
        )
        
        position_value = position_size * price
        
        print(f"   Taille de position calculée: {position_size:.8f} BTC")
        print(f"   Valeur de la position: {position_value:.2f} EUR")
        print(f"   Capital effectif: {money_manager.get_effective_capital():.2f} EUR")
        print(f"   Solde disponible: {money_manager.available_balance:.2f} EUR")
        
        # Vérifier les limites
        if position_value > money_manager.available_balance:
            print("   ⚠️  La position dépasse le solde disponible!")
        elif position_value < 10:  # Kraken minimum order value
            print("   ⚠️  La position est trop petite (minimum ~10 EUR)")
        else:
            print("   ✓ Taille de position acceptable")
            
    except Exception as e:
        print(f"   ✗ Erreur lors du calcul de la taille: {e}")
    print()
    
    # 6. Tester un ordre d'achat minimal
    print("6. TEST D'ORDRE D'ACHAT MINIMAL")
    try:
        # Essayer avec un ordre minimal de 10 EUR
        min_order_value = 10.0
        min_volume = min_order_value / price
        
        print(f"   Tentative d'ordre minimal: {min_volume:.8f} BTC ({min_order_value} EUR)")
        
        # Vérifier si on a assez de fonds
        if min_order_value > money_manager.available_balance:
            print(f"   ✗ Fonds insuffisants: {money_manager.available_balance:.2f} EUR < {min_order_value} EUR")
        else:
            # Essayer de placer l'ordre (en mode test)
            print("   ⚠️  Note: L'ordre ne sera pas réellement placé (mode diagnostic)")
            print("   Pour tester un vrai ordre, utilisez le mode manuel du bot")
            
    except Exception as e:
        print(f"   ✗ Erreur lors du test d'ordre: {e}")
    print()
    
    # 7. Recommandations
    print("7. RECOMMANDATIONS")
    print("   Si les ordres échouent, vérifiez:")
    print("   - Le solde disponible sur votre compte Kraken")
    print("   - Les limites minimales d'ordre (généralement ~10 EUR)")
    print("   - Les permissions de votre clé API (trading activé)")
    print("   - La configuration INVESTMENT_AMOUNT dans config.env")
    print("   - Les paramètres MAX_POSITION_SIZE et POSITION_SIZING_METHOD")
    print()
    
    print("=== FIN DU DIAGNOSTIC ===")

if __name__ == "__main__":
    diagnose_buy_order_issue() 