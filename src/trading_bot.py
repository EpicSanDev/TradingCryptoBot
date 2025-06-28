import time
import schedule
import logging
from datetime import datetime
from .config import Config
from .advanced_kraken_client import AdvancedKrakenClient
from .strategy import TradingStrategy

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

class TradingBot:
    """Bot de trading principal"""
    
    def __init__(self):
        """Initialiser le bot de trading"""
        try:
            # Valider la configuration
            Config.validate()
            
            # Initialiser les composants
            self.kraken_client = AdvancedKrakenClient()
            self.strategy = TradingStrategy(self.kraken_client)
            
            # État du bot
            self.is_running = False
            self.last_check = None
            
            logging.info("Bot de trading initialisé avec succès")
            
        except Exception as e:
            logging.error(f"Erreur lors de l'initialisation du bot: {e}")
            raise
    
    def start(self):
        """Démarrer le bot de trading"""
        try:
            logging.info("Démarrage du bot de trading...")
            
            # Vérifier la connexion à Kraken
            balance = self.kraken_client.get_account_balance()
            if balance is None:
                raise Exception("Impossible de se connecter à Kraken")
            
            logging.info("Connexion à Kraken établie")
            logging.info(f"Solde du compte: {balance}")
            
            # Planifier les vérifications
            schedule.every(Config.CHECK_INTERVAL).minutes.do(self.run_trading_cycle)
            
            self.is_running = True
            logging.info(f"Bot démarré - Vérification toutes les {Config.CHECK_INTERVAL} minutes")
            
            # Exécuter la première vérification immédiatement
            self.run_trading_cycle()
            
            # Boucle principale
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)
                
        except KeyboardInterrupt:
            logging.info("Arrêt du bot demandé par l'utilisateur")
            self.stop()
        except Exception as e:
            logging.error(f"Erreur dans la boucle principale: {e}")
            self.stop()
    
    def stop(self):
        """Arrêter le bot de trading"""
        logging.info("Arrêt du bot de trading...")
        self.is_running = False
        schedule.clear()
    
    def run_trading_cycle(self):
        """Exécuter un cycle de trading complet"""
        try:
            logging.info("=== Début du cycle de trading ===")
            
            # Analyser le marché
            analysis = self.strategy.analyze_market(Config.TRADING_PAIR)
            if not analysis:
                logging.warning("Impossible d'analyser le marché")
                return
            
            # Afficher l'analyse
            self._log_analysis(analysis)
            
            # Vérifier les stop-loss et take-profit pour les positions ouvertes
            self._check_risk_management(analysis)
            
            # Prendre des décisions de trading
            self._make_trading_decisions(analysis)
            
            # Afficher le résumé des performances
            self._log_performance_summary()
            
            self.last_check = datetime.now()
            logging.info("=== Fin du cycle de trading ===\n")
            
        except Exception as e:
            logging.error(f"Erreur lors du cycle de trading: {e}")
    
    def _log_analysis(self, analysis):
        """Afficher les résultats de l'analyse"""
        logging.info(f"Paire: {analysis['pair']}")
        logging.info(f"Prix actuel: {analysis['current_price']}")
        
        # Signaux
        signals = analysis['signals']
        logging.info("Signaux:")
        for indicator, signal in signals.items():
            logging.info(f"  {indicator.upper()}: {signal}")
        
        # Recommandation
        recommendation = analysis['recommendation']
        logging.info(f"Recommandation: {recommendation['action']}")
        logging.info(f"Raison: {recommendation['reason']}")
        if 'confidence' in recommendation:
            logging.info(f"Confiance: {recommendation['confidence']:.2f}")
        
        # Niveaux
        levels = analysis['levels']
        if levels:
            logging.info(f"Support: {levels.get('support', 'N/A')}")
            logging.info(f"Résistance: {levels.get('resistance', 'N/A')}")
    
    def _check_risk_management(self, analysis):
        """Vérifier la gestion des risques"""
        current_price = analysis['current_price']
        pair = analysis['pair']
        
        # Vérifier les stop-loss et take-profit
        action = self.strategy.check_stop_loss_take_profit(pair, current_price)
        
        if action == 'SELL':
            logging.warning("Stop-loss ou take-profit atteint - Vente forcée")
            
            # Créer une analyse factice pour la vente
            sell_analysis = analysis.copy()
            sell_analysis['recommendation'] = {
                'action': 'SELL',
                'reason': 'Stop-loss/Take-profit atteint',
                'confidence': 1.0
            }
            
            if self.strategy.execute_sell_order(pair, sell_analysis):
                logging.info("Vente forcée exécutée avec succès")
            else:
                logging.error("Échec de la vente forcée")
    
    def _make_trading_decisions(self, analysis):
        """Prendre des décisions de trading"""
        pair = analysis['pair']
        recommendation = analysis['recommendation']
        
        if recommendation['action'] == 'BUY' and self.strategy.should_buy(analysis):
            logging.info("Signal d'achat détecté - Exécution de l'ordre")
            
            if self.strategy.execute_buy_order(pair, analysis):
                logging.info("Ordre d'achat exécuté avec succès")
            else:
                logging.error("Échec de l'exécution de l'ordre d'achat")
        
        elif recommendation['action'] == 'SELL' and self.strategy.should_sell(analysis):
            logging.info("Signal de vente détecté - Exécution de l'ordre")
            
            if self.strategy.execute_sell_order(pair, analysis):
                logging.info("Ordre de vente exécuté avec succès")
            else:
                logging.error("Échec de l'exécution de l'ordre de vente")
        
        else:
            logging.info("Aucune action de trading requise")
    
    def _log_performance_summary(self):
        """Afficher le résumé des performances"""
        performance = self.strategy.get_performance_summary()
        
        logging.info("=== Résumé des performances ===")
        logging.info(f"Trades totaux: {performance['total_trades']}")
        logging.info(f"Trades ouverts: {performance['open_trades']}")
        logging.info(f"Trades gagnants: {performance['winning_trades']}")
        logging.info(f"Trades perdants: {performance['losing_trades']}")
        logging.info(f"Taux de réussite: {performance['win_rate']:.2f}%")
        logging.info(f"Profit/Perte total: {performance['total_profit_loss']:.2f}")
        if performance['total_trades'] > 0:
            logging.info(f"Profit/Perte moyen: {performance['average_profit_loss']:.2f}")
    
    def get_status(self):
        """Obtenir le statut actuel du bot"""
        return {
            'is_running': self.is_running,
            'last_check': self.last_check,
            'performance': self.strategy.get_performance_summary(),
            'last_analysis': self.strategy.last_analysis
        }
    
    def get_trade_history(self):
        """Obtenir l'historique des trades"""
        return self.strategy.trade_history
    
    def manual_buy(self, pair=None, volume=None):
        """Achat manuel"""
        if not pair:
            pair = Config.TRADING_PAIR
        
        if not volume:
            current_price = self.kraken_client.get_current_price(pair)
            volume = self.strategy.calculate_position_size(current_price, Config.INVESTMENT_AMOUNT)
        
        logging.info(f"Achat manuel: {volume} {pair}")
        
        # Créer une analyse factice
        analysis = {
            'pair': pair,
            'current_price': self.kraken_client.get_current_price(pair),
            'recommendation': {'action': 'BUY', 'confidence': 1.0}
        }
        
        return self.strategy.execute_buy_order(pair, analysis)
    
    def manual_sell(self, pair=None):
        """Vente manuelle"""
        if not pair:
            pair = Config.TRADING_PAIR
        
        logging.info(f"Vente manuelle: {pair}")
        
        # Créer une analyse factice
        analysis = {
            'pair': pair,
            'current_price': self.kraken_client.get_current_price(pair),
            'recommendation': {'action': 'SELL', 'confidence': 1.0}
        }
        
        return self.strategy.execute_sell_order(pair, analysis)
    
    def get_current_positions(self):
        """Obtenir les positions actuelles"""
        positions = []
        for trade in self.strategy.trade_history:
            if not trade.get('sold', False):
                positions.append(trade)
        return positions 