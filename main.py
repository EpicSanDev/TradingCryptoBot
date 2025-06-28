#!/usr/bin/env python3
"""
Bot de Trading Crypto pour Kraken
================================

Ce bot automatise le trading de cryptomonnaies sur Kraken en utilisant
des indicateurs techniques avancés et une gestion des risques.

Fonctionnalités:
- Analyse technique avec RSI, MACD, Bollinger Bands, Moving Averages
- Gestion automatique des stop-loss et take-profit
- Interface de ligne de commande interactive
- Logging détaillé des opérations
- Gestion des risques avec position sizing

Usage:
    python main.py                    # Démarrer le bot automatique
    python main.py --manual           # Mode manuel
    python main.py --config           # Configurer le bot
    python main.py --status           # Afficher le statut
"""

import argparse
import sys
import os
from datetime import datetime

# Ajouter le répertoire src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.trading_bot import TradingBot
from src.config import Config

def print_banner():
    """Afficher la bannière du bot"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    BOT DE TRADING CRYPTO                     ║
║                        Kraken Edition                        ║
║                                                              ║
║  Indicateurs: RSI, MACD, Bollinger Bands, Moving Averages   ║
║  Gestion des risques: Stop-loss, Take-profit, Position sizing ║
║  Trading automatique avec analyse technique avancée         ║
╚══════════════════════════════════════════════════════════════╝
    """)

def print_config():
    """Afficher la configuration actuelle"""
    print("\n=== CONFIGURATION ACTUELLE ===")
    print(f"Paire de trading: {Config.TRADING_PAIR}")
    print(f"Montant d'investissement: {Config.INVESTMENT_AMOUNT}")
    print(f"Taille max position: {Config.MAX_POSITION_SIZE * 100}%")
    print(f"Stop-loss: {Config.STOP_LOSS_PERCENTAGE}%")
    print(f"Take-profit: {Config.TAKE_PROFIT_PERCENTAGE}%")
    print(f"Intervalle de vérification: {Config.CHECK_INTERVAL} minutes")
    print("\n=== INDICATEURS TECHNIQUES ===")
    print(f"RSI période: {Config.RSI_PERIOD} (survente: {Config.RSI_OVERSOLD}, surachat: {Config.RSI_OVERBOUGHT})")
    print(f"MACD: {Config.MACD_FAST}/{Config.MACD_SLOW}/{Config.MACD_SIGNAL}")
    print(f"Bollinger: {Config.BOLLINGER_PERIOD} périodes, {Config.BOLLINGER_STD}σ")
    print(f"Moving Averages: {Config.MA_FAST}/{Config.MA_SLOW}")

def interactive_mode(bot):
    """Mode interactif pour le trading manuel"""
    print("\n=== MODE INTERACTIF ===")
    print("Commandes disponibles:")
    print("  buy     - Acheter")
    print("  sell    - Vendre")
    print("  status  - Afficher le statut")
    print("  history - Historique des trades")
    print("  positions - Positions actuelles")
    print("  quit    - Quitter")
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command == 'quit':
                break
            elif command == 'buy':
                print("Exécution d'un achat manuel...")
                if bot.manual_buy():
                    print("Achat exécuté avec succès!")
                else:
                    print("Échec de l'achat")
            elif command == 'sell':
                print("Exécution d'une vente manuelle...")
                if bot.manual_sell():
                    print("Vente exécutée avec succès!")
                else:
                    print("Échec de la vente")
            elif command == 'status':
                status = bot.get_status()
                print(f"\nStatut du bot: {'En cours' if status['is_running'] else 'Arrêté'}")
                if status['last_check']:
                    print(f"Dernière vérification: {status['last_check']}")
                
                performance = status['performance']
                print(f"Trades totaux: {performance['total_trades']}")
                print(f"Taux de réussite: {performance['win_rate']:.2f}%")
                print(f"Profit/Perte total: {performance['total_profit_loss']:.2f}")
            elif command == 'history':
                history = bot.get_trade_history()
                if not history:
                    print("Aucun trade dans l'historique")
                else:
                    print("\n=== HISTORIQUE DES TRADES ===")
                    for i, trade in enumerate(history[-10:], 1):  # Afficher les 10 derniers
                        status = "Vendu" if trade.get('sold', False) else "Ouvert"
                        print(f"{i}. {trade['action']} {trade['volume']} {trade['pair']} à {trade['price']} - {status}")
                        if trade.get('profit_loss'):
                            print(f"   Profit/Perte: {trade['profit_loss']:.2f} ({trade['profit_loss_percent']:.2f}%)")
            elif command == 'positions':
                positions = bot.get_current_positions()
                if not positions:
                    print("Aucune position ouverte")
                else:
                    print("\n=== POSITIONS OUVERTES ===")
                    for i, pos in enumerate(positions, 1):
                        print(f"{i}. {pos['pair']}: {pos['volume']} acheté à {pos['price']}")
            else:
                print("Commande inconnue. Tapez 'quit' pour quitter.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Erreur: {e}")

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Bot de Trading Crypto pour Kraken')
    parser.add_argument('--manual', action='store_true', help='Mode manuel interactif')
    parser.add_argument('--config', action='store_true', help='Afficher la configuration')
    parser.add_argument('--status', action='store_true', help='Afficher le statut du bot')
    parser.add_argument('--test', action='store_true', help='Mode test (pas de trading réel)')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Vérifier si le fichier de configuration existe
    if not os.path.exists('config.env'):
        print("❌ Fichier config.env non trouvé!")
        print("📝 Créez le fichier config.env avec vos clés API Kraken:")
        print("   Copiez config.env.example vers config.env et remplissez vos clés")
        sys.exit(1)
    
    try:
        # Initialiser le bot
        bot = TradingBot()
        
        if args.config:
            print_config()
            return
        
        if args.status:
            status = bot.get_status()
            print(f"\nStatut du bot: {'En cours' if status['is_running'] else 'Arrêté'}")
            if status['last_check']:
                print(f"Dernière vérification: {status['last_check']}")
            
            performance = status['performance']
            print(f"Trades totaux: {performance['total_trades']}")
            print(f"Taux de réussite: {performance['win_rate']:.2f}%")
            print(f"Profit/Perte total: {performance['total_profit_loss']:.2f}")
            return
        
        if args.manual:
            interactive_mode(bot)
            return
        
        # Mode automatique
        print("🚀 Démarrage du bot de trading automatique...")
        print("⚠️  ATTENTION: Ce bot effectue des trades réels!")
        print("   Assurez-vous d'avoir configuré correctement vos paramètres.")
        print("   Appuyez sur Ctrl+C pour arrêter le bot.\n")
        
        if args.test:
            print("🧪 MODE TEST ACTIVÉ - Aucun trade réel ne sera effectué")
        
        # Démarrer le bot
        bot.start()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Arrêt du bot demandé par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur fatale: {e}")
        print("Vérifiez votre configuration et vos clés API Kraken")
        sys.exit(1)

if __name__ == "__main__":
    main() 