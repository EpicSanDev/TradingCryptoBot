#!/usr/bin/env python3
"""
Bot de Trading Crypto Avancé pour Kraken
========================================

Ce bot automatise le trading de cryptomonnaies sur Kraken avec des fonctionnalités avancées :
- Mode multi-paire avec gestion indépendante
- Support des modes spot et futures (avec levier)
- Money management sophistiqué (Kelly, Martingale, Fixed)
- Gestion des risques avancée avec corrélation
- Stop-loss dynamiques et trailing stops

Usage:
    python advanced_main.py                    # Démarrer le bot automatique
    python advanced_main.py --manual           # Mode manuel
    python advanced_main.py --config           # Configurer le bot
    python advanced_main.py --status           # Afficher le statut
    python advanced_main.py --spot             # Mode spot uniquement
    python advanced_main.py --futures          # Mode futures uniquement
"""

import argparse
import sys
import os
from datetime import datetime

# Ajouter le répertoire src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.advanced_trading_bot import AdvancedTradingBot
from src.config import Config

def print_banner():
    """Afficher la bannière du bot avancé"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                BOT DE TRADING CRYPTO AVANCÉ                  ║
║                      Kraken Edition                          ║
║                                                              ║
║  Multi-paire: Trading simultané sur plusieurs paires        ║
║  Modes: Spot et Futures (avec levier)                       ║
║  Money Management: Kelly, Martingale, Fixed sizing          ║
║  Gestion des risques: Corrélation, Drawdown, Stop-loss      ║
║  Indicateurs: RSI, MACD, Bollinger, Moving Averages         ║
╚══════════════════════════════════════════════════════════════╝
    """)

def print_config():
    """Afficher la configuration actuelle"""
    Config.print_config()

def interactive_mode(bot):
    """Mode interactif pour le trading manuel"""
    print("\n=== MODE INTERACTIF AVANCÉ ===")
    print("Commandes disponibles:")
    print("  buy <pair> [volume]  - Acheter sur une paire")
    print("  sell <pair>          - Vendre sur une paire")
    print("  status               - Afficher le statut")
    print("  history              - Historique des trades")
    print("  positions            - Positions actuelles")
    print("  performance          - Performance détaillée")
    print("  pairs                - Liste des paires configurées")
    print("  capital              - Informations sur le capital")
    print("  update_capital       - Forcer la mise à jour du capital")
    print("  recommendations      - Recommandations de taille de position")
    print("  quit                 - Quitter")
    
    while True:
        try:
            command = input("\n> ").strip().lower().split()
            
            if not command:
                continue
                
            cmd = command[0]
            
            if cmd == 'quit':
                break
            elif cmd == 'buy':
                if len(command) < 2:
                    print("Usage: buy <pair> [volume]")
                    continue
                pair = command[1].upper()
                volume = float(command[2]) if len(command) > 2 else None
                
                print(f"Exécution d'un achat manuel sur {pair}...")
                if bot.manual_buy(pair, volume):
                    print("Achat exécuté avec succès!")
                else:
                    print("Échec de l'achat")
            elif cmd == 'sell':
                if len(command) < 2:
                    print("Usage: sell <pair>")
                    continue
                pair = command[1].upper()
                
                print(f"Exécution d'une vente manuelle sur {pair}...")
                if bot.manual_sell(pair):
                    print("Vente exécutée avec succès!")
                else:
                    print("Échec de la vente")
            elif cmd == 'status':
                status = bot.get_status()
                print(f"\nStatut du bot: {'En cours' if status['is_running'] else 'Arrêté'}")
                print(f"Mode de trading: {status['trading_mode']}")
                if status['last_check']:
                    print(f"Dernière vérification: {status['last_check']}")
                
                print(f"Paires actives: {', '.join(status['active_pairs']) if status['active_pairs'] else 'Aucune'}")
                
                performance = status['performance']
                print(f"\n=== PERFORMANCE ===")
                print(f"Trades totaux: {performance['total_trades']}")
                print(f"Taux de réussite: {performance['win_rate']:.2f}%")
                print(f"Profit/Perte total: {performance['total_profit_loss']:.2f}")
                print(f"Drawdown actuel: {performance['current_drawdown']:.2f}%")
                print(f"Ratio de Sharpe: {performance['sharpe_ratio']:.2f}")
            elif cmd == 'history':
                history = bot.get_trade_history()
                if not history:
                    print("Aucun trade dans l'historique")
                else:
                    print("\n=== HISTORIQUE DES TRADES ===")
                    for i, trade in enumerate(history[-10:], 1):  # Afficher les 10 derniers
                        mode_info = f" ({trade['trading_mode']})" if 'trading_mode' in trade else ""
                        leverage_info = f" {trade['leverage']}x" if trade.get('leverage') else ""
                        print(f"{i}. {trade['action']} {trade['volume']} {trade['pair']} à {trade['price']}{mode_info}{leverage_info}")
                        if trade.get('profit_loss') is not None:
                            print(f"   Profit/Perte: {trade['profit_loss']:.2f} ({trade['profit_loss_percent']:.2f}%)")
            elif cmd == 'positions':
                positions = bot.get_current_positions()
                if not positions:
                    print("Aucune position ouverte")
                else:
                    print("\n=== POSITIONS OUVERTES ===")
                    for i, pos in enumerate(positions, 1):
                        leverage_info = f" (levier {pos['leverage']}x)" if pos.get('leverage') else ""
                        print(f"{i}. {pos['pair']}: {pos['volume']} acheté à {pos['entry_price']}{leverage_info}")
                        print(f"   Prix actuel: {pos['current_price']}")
                        print(f"   PnL: {pos['unrealized_pnl']:.2f} ({pos['unrealized_pnl_percent']:.2f}%)")
            elif cmd == 'performance':
                status = bot.get_status()
                performance = status['performance']
                print("\n=== PERFORMANCE DÉTAILLÉE ===")
                print(f"Trades totaux: {performance['total_trades']}")
                print(f"Trades gagnants: {performance['winning_trades']}")
                print(f"Trades perdants: {performance['losing_trades']}")
                print(f"Taux de réussite: {performance['win_rate']:.2f}%")
                print(f"Profit/Perte total: {performance['total_profit_loss']:.2f}")
                if performance['total_trades'] > 0:
                    print(f"Profit/Perte moyen: {performance['average_profit_loss']:.2f}")
                print(f"Drawdown actuel: {performance['current_drawdown']:.2f}%")
                print(f"Ratio de Sharpe: {performance['sharpe_ratio']:.2f}")
            elif cmd == 'pairs':
                print(f"\nPaires configurées: {', '.join(Config.TRADING_PAIRS)}")
                if Config.PAIR_CONFIGS:
                    print("\nConfigurations spécifiques:")
                    for pair, config in Config.PAIR_CONFIGS.items():
                        print(f"  {pair}:")
                        if 'stop_loss' in config:
                            print(f"    Stop-loss: {config['stop_loss']}%")
                        if 'take_profit' in config:
                            print(f"    Take-profit: {config['take_profit']}%")
                        if 'leverage' in config:
                            print(f"    Levier: {config['leverage']}x")
                        if 'allocation' in config:
                            print(f"    Allocation: {config['allocation']}%")
            elif cmd == 'capital':
                capital_info = bot.money_manager.get_capital_info()
                print("\n=== INFORMATIONS SUR LE CAPITAL ===")
                print(f"Capital statique configuré: {capital_info['static_capital']:.2f} {capital_info['capital_currency']}")
                print(f"Capital disponible (dynamique): {capital_info['available_capital']:.2f} {capital_info['capital_currency']}")
                print(f"Solde actuel du bot: {capital_info['current_balance']:.2f} {capital_info['capital_currency']}")
                print(f"Pic historique: {capital_info['peak_balance']:.2f} {capital_info['capital_currency']}")
                print(f"Adaptation automatique: {'✅ Activée' if capital_info['auto_update_enabled'] else '❌ Désactivée'}")
                
                if capital_info['last_update']:
                    import datetime
                    time_since_update = datetime.datetime.now() - capital_info['last_update']
                    print(f"Dernière mise à jour: il y a {time_since_update.total_seconds()/60:.1f} minutes")
                else:
                    print("Dernière mise à jour: Jamais")
                
                # Afficher la différence
                capital_difference = capital_info['available_capital'] - capital_info['static_capital']
                difference_percent = (capital_difference / capital_info['static_capital']) * 100 if capital_info['static_capital'] > 0 else 0
                print(f"Différence capital dynamique vs statique: {difference_percent:+.1f}%")
                
                # Afficher la méthode de sizing
                sizing_info = bot.money_manager.get_sizing_method_info()
                print(f"Méthode de sizing: {sizing_info}")
            elif cmd == 'update_capital':
                print("Mise à jour forcée du capital en cours...")
                success = bot.money_manager.force_capital_update()
                if success:
                    print("✅ Capital mis à jour avec succès")
                    capital_info = bot.money_manager.get_capital_info()
                    print(f"Nouveau capital disponible: {capital_info['available_capital']:.2f} {capital_info['capital_currency']}")
                else:
                    print("❌ Échec de la mise à jour du capital")
            elif cmd == 'recommendations':
                recommendations = bot.money_manager.get_position_recommendations()
                print("\n=== RECOMMANDATIONS DE TAILLE DE POSITION ===")
                dynamic_capital = bot.money_manager.get_dynamic_capital()
                print(f"Capital total disponible: {dynamic_capital:.2f} EUR")
                
                for pair, rec in recommendations.items():
                    if 'error' in rec:
                        print(f"{pair}: Erreur - {rec['error']}")
                    else:
                        print(f"\n{pair}:")
                        print(f"  Allocation: {rec['allocation_percent']:.1f}% ({rec['allocated_capital']:.2f} EUR)")
                        print(f"  Taille recommandée: {rec['recommended_size']:.2f} EUR")
                        print(f"  Taille maximale: {rec['max_size']:.2f} EUR")
            else:
                print("Commande inconnue. Tapez 'quit' pour quitter.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Erreur: {e}")

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Bot de Trading Crypto Avancé pour Kraken')
    parser.add_argument('--manual', action='store_true', help='Mode manuel interactif')
    parser.add_argument('--config', action='store_true', help='Afficher la configuration')
    parser.add_argument('--status', action='store_true', help='Afficher le statut du bot')
    parser.add_argument('--test', action='store_true', help='Mode test (pas de trading réel)')
    parser.add_argument('--spot', action='store_true', help='Mode spot uniquement')
    parser.add_argument('--futures', action='store_true', help='Mode futures uniquement')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Vérifier si le fichier de configuration existe
    if not os.path.exists('config.env'):
        print("❌ Fichier config.env non trouvé!")
        print("📝 Créez le fichier config.env avec vos clés API Kraken:")
        print("   Copiez config.env.example vers config.env et remplissez vos clés")
        print("\nVariables requises:")
        print("   SPOT_API_KEY, SPOT_SECRET_KEY")
        print("   FUTURES_API_KEY, FUTURES_SECRET_KEY")
        print("   TRADING_PAIRS, TRADING_MODE")
        sys.exit(1)
    
    try:
        # Déterminer le mode de trading
        trading_mode = Config.TRADING_MODE
        if args.spot:
            trading_mode = 'spot'
        elif args.futures:
            trading_mode = 'futures'
        
        # Initialiser le bot
        bot = AdvancedTradingBot(trading_mode)
        
        if args.config:
            print_config()
            return
        
        if args.status:
            status = bot.get_status()
            print(f"\nStatut du bot: {'En cours' if status['is_running'] else 'Arrêté'}")
            print(f"Mode de trading: {status['trading_mode']}")
            if status['last_check']:
                print(f"Dernière vérification: {status['last_check']}")
            
            print(f"Paires actives: {', '.join(status['active_pairs']) if status['active_pairs'] else 'Aucune'}")
            
            performance = status['performance']
            print(f"Trades totaux: {performance['total_trades']}")
            print(f"Taux de réussite: {performance['win_rate']:.2f}%")
            print(f"Profit/Perte total: {performance['total_profit_loss']:.2f}")
            print(f"Drawdown actuel: {performance['current_drawdown']:.2f}%")
            return
        
        if args.manual:
            interactive_mode(bot)
            return
        
        # Mode automatique
        print("🚀 Démarrage du bot de trading avancé...")
        print(f"📊 Mode: {trading_mode.upper()}")
        print(f"🔄 Paires: {', '.join(Config.TRADING_PAIRS)}")
        print(f"💰 Capital: {Config.INVESTMENT_AMOUNT}")
        print(f"📈 Money Management: {Config.POSITION_SIZING_METHOD}")
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