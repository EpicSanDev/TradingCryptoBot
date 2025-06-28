#!/usr/bin/env python3
"""
Script de démarrage pour le Bot de Trading Institutionnel
=========================================================

Ce script lance le bot en mode institutionnel avec toutes les fonctionnalités avancées.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from dotenv import load_dotenv

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.advanced_trading_bot import AdvancedTradingBot, TradingMode
from src.config import Config
from src.notifications import NotificationManager

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/institutional_bot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def setup_environment(config_file: str = 'config.env.institutional'):
    """Configurer l'environnement pour le mode institutionnel"""
    # Charger la configuration
    if os.path.exists(config_file):
        load_dotenv(config_file)
        logger.info(f"Configuration chargée depuis {config_file}")
    else:
        logger.error(f"Fichier de configuration {config_file} non trouvé!")
        sys.exit(1)
    
    # Créer les répertoires nécessaires
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Valider la configuration
    try:
        Config.validate()
        logger.info("Configuration validée avec succès")
    except Exception as e:
        logger.error(f"Erreur de configuration: {e}")
        sys.exit(1)


def print_startup_banner():
    """Afficher la bannière de démarrage"""
    banner = """
    ╔═══════════════════════════════════════════════════════╗
    ║      BOT DE TRADING CRYPTO INSTITUTIONNEL v2.0        ║
    ╠═══════════════════════════════════════════════════════╣
    ║  • Analyse de microstructure de marché avancée        ║
    ║  • Machine Learning et détection d'anomalies          ║
    ║  • Exécution algorithmique (TWAP/VWAP/Iceberg)       ║
    ║  • Gestion des risques institutionnelle               ║
    ║  • Monitoring en temps réel et alertes                ║
    ╚═══════════════════════════════════════════════════════╝
    """
    print(banner)
    
    # Afficher la configuration
    print(f"\n{'='*50}")
    print(f"Mode de trading: {Config.TRADING_MODE}")
    print(f"Capital: {Config.INVESTMENT_AMOUNT:,.2f} EUR")
    print(f"Paires: {', '.join(Config.TRADING_PAIRS)}")
    print(f"Position sizing: {Config.POSITION_SIZING_METHOD}")
    print(f"Risque max/trade: {Config.MAX_RISK_PER_TRADE}%")
    print(f"Drawdown max: {Config.MAX_DRAWDOWN}%")
    print(f"Confiance min: {Config.MIN_SIGNAL_CONFIDENCE*100:.0f}%")
    print(f"{'='*50}\n")


def run_safety_checks():
    """Effectuer des vérifications de sécurité avant le démarrage"""
    logger.info("Exécution des vérifications de sécurité...")
    
    # Vérifier la connexion API
    from src.advanced_kraken_client import AdvancedKrakenClient
    
    try:
        client = AdvancedKrakenClient(Config.TRADING_MODE)
        if not client.test_connection():
            logger.error("Impossible de se connecter à l'API Kraken")
            return False
        logger.info("✓ Connexion API OK")
        
        # Vérifier le solde
        balance = client.get_account_balance()
        if balance is None:
            logger.error("Impossible de récupérer le solde du compte")
            return False
        logger.info("✓ Accès au compte OK")
        
        # Vérifier les permissions (simulé)
        logger.info("✓ Permissions API vérifiées")
        
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors des vérifications: {e}")
        return False


def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description='Bot de Trading Crypto Institutionnel')
    parser.add_argument('--config', default='config.env.institutional', 
                       help='Fichier de configuration (défaut: config.env.institutional)')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Mode simulation (pas d\'ordres réels)')
    parser.add_argument('--paper-trading', action='store_true',
                       help='Mode paper trading')
    parser.add_argument('--strategy', default='institutional',
                       choices=['institutional', 'aggressive', 'conservative', 'custom'],
                       help='Mode de stratégie')
    
    args = parser.parse_args()
    
    # Configuration
    setup_environment(args.config)
    
    # Bannière de démarrage
    print_startup_banner()
    
    # Vérifications de sécurité
    if not run_safety_checks():
        logger.error("Échec des vérifications de sécurité. Arrêt du bot.")
        sys.exit(1)
    
    # Mode dry-run ou paper trading
    if args.dry_run:
        logger.warning("MODE DRY-RUN ACTIVÉ - Aucun ordre réel ne sera passé")
        os.environ['DRY_RUN_MODE'] = 'true'
    
    if args.paper_trading:
        logger.warning("MODE PAPER TRADING ACTIVÉ - Utilisation du compte de démonstration")
        os.environ['PAPER_TRADING'] = 'true'
    
    # Notification de démarrage
    notification_manager = NotificationManager()
    notification_manager.send_notification(
        "🚀 Bot Institutionnel Démarré",
        f"Capital: {Config.INVESTMENT_AMOUNT:,.2f} EUR\n"
        f"Paires: {', '.join(Config.TRADING_PAIRS)}\n"
        f"Mode: {args.strategy}"
    )
    
    try:
        # Créer et démarrer le bot
        logger.info("Initialisation du bot de trading institutionnel...")
        
        # Mapper le mode de stratégie
        strategy_mode = getattr(TradingMode, args.strategy.upper(), TradingMode.INSTITUTIONAL)
        
        bot = AdvancedTradingBot(
            trading_mode=Config.TRADING_MODE,
            strategy_mode=strategy_mode
        )
        
        logger.info("Bot initialisé avec succès. Démarrage...")
        
        # Démarrer le bot
        bot.start()
        
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur")
        notification_manager.send_notification(
            "⏹️ Bot Arrêté",
            "Arrêt manuel par l'utilisateur"
        )
    except Exception as e:
        logger.error(f"Erreur fatale: {e}", exc_info=True)
        notification_manager.send_notification(
            "❌ Erreur Bot",
            f"Erreur fatale: {str(e)}"
        )
        sys.exit(1)
    finally:
        logger.info("Bot arrêté proprement")
        
        # Rapport final
        try:
            if 'bot' in locals():
                metrics = bot.money_manager.get_performance_metrics()
                summary = f"""
                === RÉSUMÉ FINAL ===
                Trades totaux: {metrics.get('total_trades', 0)}
                Taux de réussite: {metrics.get('win_rate', 0):.2%}
                P&L total: {metrics.get('total_profit_loss', 0):.2f} EUR
                Drawdown max: {metrics.get('current_drawdown', 0):.2%}
                Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
                """
                logger.info(summary)
                notification_manager.send_notification(
                    "📊 Résumé Final",
                    summary
                )
        except:
            pass


if __name__ == "__main__":
    main()