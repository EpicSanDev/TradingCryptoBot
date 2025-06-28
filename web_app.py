#!/usr/bin/env python3
"""
Interface Web pour le Bot de Trading Crypto Avancé
==================================================

Backend Flask pour suivre les actions du bot en temps réel
"""

from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading
import time
import json
from datetime import datetime
import os
import sys

# Ajouter le répertoire src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.advanced_trading_bot import AdvancedTradingBot
from src.config import Config

app = Flask(__name__, static_folder='web/build', static_url_path='')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Instance globale du bot
bot_instance = None
bot_thread = None

class WebInterface:
    def __init__(self, bot):
        self.bot = bot
        self.is_broadcasting = False
        
    def start_broadcasting(self):
        """Diffuser les données en temps réel via WebSocket"""
        self.is_broadcasting = True
        while self.is_broadcasting:
            try:
                # Obtenir le statut du bot
                status = self.bot.get_status()
                
                # Obtenir les prix en temps réel pour chaque paire
                prices = {}
                for pair in Config.TRADING_PAIRS:
                    price = self.bot.active_client.get_current_price(pair)
                    if price:
                        prices[pair] = price
                
                # Obtenir les positions
                positions = self.bot.get_current_positions()
                
                # Obtenir l'historique récent
                history = self.bot.get_trade_history()[-10:]  # 10 derniers trades
                
                # Préparer les données à envoyer
                data = {
                    'timestamp': datetime.now().isoformat(),
                    'status': status,
                    'prices': prices,
                    'positions': positions,
                    'recent_trades': history
                }
                
                # Émettre via WebSocket
                socketio.emit('update', data)
                
                # Attendre avant la prochaine mise à jour
                time.sleep(1)  # Mise à jour chaque seconde
                
            except Exception as e:
                print(f"Erreur lors de la diffusion: {e}")
                time.sleep(5)
    
    def stop_broadcasting(self):
        self.is_broadcasting = False

web_interface = None

@app.route('/')
def index():
    """Page principale"""
    return app.send_static_file('index.html')

@app.route('/api/status')
def get_status():
    """Obtenir le statut du bot"""
    if bot_instance:
        return jsonify(bot_instance.get_status())
    return jsonify({'error': 'Bot not running'})

@app.route('/api/positions')
def get_positions():
    """Obtenir les positions ouvertes"""
    if bot_instance:
        return jsonify(bot_instance.get_current_positions())
    return jsonify([])

@app.route('/api/history')
def get_history():
    """Obtenir l'historique des trades"""
    if bot_instance:
        limit = request.args.get('limit', 50, type=int)
        history = bot_instance.get_trade_history()
        return jsonify(history[-limit:])
    return jsonify([])

@app.route('/api/prices')
def get_prices():
    """Obtenir les prix actuels"""
    if bot_instance:
        prices = {}
        for pair in Config.TRADING_PAIRS:
            price = bot_instance.active_client.get_current_price(pair)
            if price:
                prices[pair] = price
        return jsonify(prices)
    return jsonify({})

@app.route('/api/config')
def get_config():
    """Obtenir la configuration"""
    return jsonify({
        'trading_pairs': Config.TRADING_PAIRS,
        'trading_mode': Config.TRADING_MODE,
        'investment_amount': Config.INVESTMENT_AMOUNT,
        'position_sizing_method': Config.POSITION_SIZING_METHOD,
        'risk_per_trade': Config.RISK_PER_TRADE,
        'max_positions': Config.MAX_POSITIONS
    })

@app.route('/api/start_bot', methods=['POST'])
def start_bot():
    """Démarrer le bot"""
    global bot_instance, bot_thread, web_interface
    
    if bot_instance and bot_instance.is_running:
        return jsonify({'error': 'Bot already running'}), 400
    
    try:
        # Créer une nouvelle instance du bot
        trading_mode = request.json.get('mode', Config.TRADING_MODE)
        bot_instance = AdvancedTradingBot(trading_mode)
        
        # Démarrer le bot dans un thread séparé
        bot_thread = threading.Thread(target=bot_instance.start)
        bot_thread.start()
        
        # Démarrer la diffusion
        web_interface = WebInterface(bot_instance)
        broadcast_thread = threading.Thread(target=web_interface.start_broadcasting)
        broadcast_thread.start()
        
        return jsonify({'status': 'Bot started successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop_bot', methods=['POST'])
def stop_bot():
    """Arrêter le bot"""
    global bot_instance, web_interface
    
    if not bot_instance:
        return jsonify({'error': 'Bot not running'}), 400
    
    try:
        if web_interface:
            web_interface.stop_broadcasting()
        
        bot_instance.stop()
        return jsonify({'status': 'Bot stopped successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/manual_trade', methods=['POST'])
def manual_trade():
    """Exécuter un trade manuel"""
    if not bot_instance:
        return jsonify({'error': 'Bot not running'}), 400
    
    data = request.json
    action = data.get('action')
    pair = data.get('pair')
    volume = data.get('volume')
    
    try:
        if action == 'buy':
            success = bot_instance.manual_buy(pair, volume)
        elif action == 'sell':
            success = bot_instance.manual_sell(pair)
        else:
            return jsonify({'error': 'Invalid action'}), 400
        
        if success:
            return jsonify({'status': 'Trade executed successfully'})
        else:
            return jsonify({'error': 'Trade execution failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """Gérer la connexion WebSocket"""
    emit('connected', {'data': 'Connected to trading bot'})

@socketio.on('disconnect')
def handle_disconnect():
    """Gérer la déconnexion WebSocket"""
    print('Client disconnected')

if __name__ == '__main__':
    print("🚀 Démarrage de l'interface web du bot de trading...")
    print("📊 Accédez à http://localhost:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)