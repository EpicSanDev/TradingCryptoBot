"""
Module de Notifications et Alertes
==================================

Ce module gère les notifications pour les événements importants du bot :
- Trades exécutés
- Alertes de prix
- Erreurs critiques
- Performances exceptionnelles
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Optional
import requests
import json
from .config import Config

class NotificationManager:
    """Gestionnaire de notifications multi-canal"""
    
    def __init__(self):
        self.enabled = Config.NOTIFICATIONS_ENABLED if hasattr(Config, 'NOTIFICATIONS_ENABLED') else False
        self.email_enabled = Config.EMAIL_NOTIFICATIONS if hasattr(Config, 'EMAIL_NOTIFICATIONS') else False
        self.webhook_enabled = Config.WEBHOOK_NOTIFICATIONS if hasattr(Config, 'WEBHOOK_NOTIFICATIONS') else False
        
        # Configuration email
        self.smtp_server = Config.SMTP_SERVER if hasattr(Config, 'SMTP_SERVER') else None
        self.smtp_port = Config.SMTP_PORT if hasattr(Config, 'SMTP_PORT') else 587
        self.email_from = Config.EMAIL_FROM if hasattr(Config, 'EMAIL_FROM') else None
        self.email_to = Config.EMAIL_TO if hasattr(Config, 'EMAIL_TO') else None
        self.email_password = Config.EMAIL_PASSWORD if hasattr(Config, 'EMAIL_PASSWORD') else None
        
        # Configuration webhook (Discord, Slack, etc.)
        self.webhook_url = Config.WEBHOOK_URL if hasattr(Config, 'WEBHOOK_URL') else None
        
        logging.info("Gestionnaire de notifications initialisé")
    
    def send_notification(self, title: str, message: str, level: str = 'info', data: Optional[Dict] = None):
        """
        Envoyer une notification sur tous les canaux configurés
        
        Args:
            title: Titre de la notification
            message: Message détaillé
            level: Niveau d'importance ('info', 'warning', 'error', 'success')
            data: Données supplémentaires
        """
        if not self.enabled:
            return
        
        # Formatter le message avec les données
        full_message = self._format_message(title, message, level, data)
        
        # Envoyer via email
        if self.email_enabled:
            self._send_email(title, full_message, level)
        
        # Envoyer via webhook
        if self.webhook_enabled:
            self._send_webhook(title, full_message, level, data)
        
        # Logger localement
        log_level = getattr(logging, level.upper(), logging.INFO)
        logging.log(log_level, f"Notification: {title} - {message}")
    
    def _format_message(self, title: str, message: str, level: str, data: Optional[Dict]) -> str:
        """Formater le message complet"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        formatted = f"""
🤖 Bot de Trading Crypto - {title}
{'='*50}
⏰ Heure: {timestamp}
📊 Niveau: {level.upper()}

{message}
"""
        
        if data:
            formatted += "\n📈 Détails:\n"
            for key, value in data.items():
                formatted += f"  • {key}: {value}\n"
        
        return formatted
    
    def _send_email(self, subject: str, body: str, level: str):
        """Envoyer une notification par email"""
        if not all([self.smtp_server, self.email_from, self.email_to, self.email_password]):
            logging.warning("Configuration email incomplète")
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_from or ''
            msg['To'] = self.email_to or ''
            msg['Subject'] = f"[CryptoBot] {subject}"
            
            # Ajouter un style selon le niveau
            color = {
                'info': '#3498db',
                'warning': '#f39c12',
                'error': '#e74c3c',
                'success': '#2ecc71'
            }.get(level, '#3498db')
            
            html_body = f"""
            <html>
                <body style="font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px;">
                    <div style="background-color: white; border-radius: 10px; padding: 20px; max-width: 600px; margin: 0 auto; border-top: 4px solid {color};">
                        <h2 style="color: #333; margin-top: 0;">{subject}</h2>
                        <pre style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; white-space: pre-wrap;">{body}</pre>
                    </div>
                </body>
            </html>
            """
            
            msg.attach(MIMEText(html_body, 'html'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_from, self.email_password)
                server.send_message(msg)
            
            logging.debug(f"Email envoyé: {subject}")
            
        except Exception as e:
            logging.error(f"Erreur lors de l'envoi de l'email: {e}")
    
    def _send_webhook(self, title: str, message: str, level: str, data: Optional[Dict]):
        """Envoyer une notification via webhook (Discord, Slack, etc.)"""
        if not self.webhook_url:
            return
        
        try:
            # Déterminer le type de webhook
            if 'discord' in self.webhook_url:
                self._send_discord_webhook(title, message, level, data)
            elif 'slack' in self.webhook_url:
                self._send_slack_webhook(title, message, level, data)
            else:
                # Webhook générique
                payload = {
                    'title': title,
                    'message': message,
                    'level': level,
                    'timestamp': datetime.now().isoformat(),
                    'data': data
                }
                requests.post(self.webhook_url, json=payload)
            
        except Exception as e:
            logging.error(f"Erreur lors de l'envoi du webhook: {e}")
    
    def _send_discord_webhook(self, title: str, message: str, level: str, data: Optional[Dict]):
        """Envoyer vers Discord"""
        color = {
            'info': 3447003,      # Bleu
            'warning': 16776960,  # Jaune
            'error': 15158332,    # Rouge
            'success': 3066993    # Vert
        }.get(level, 3447003)
        
        embed = {
            'title': title,
            'description': message,
            'color': color,
            'timestamp': datetime.now().isoformat(),
            'footer': {'text': 'Bot de Trading Crypto'},
            'fields': []
        }
        
        if data:
            for key, value in data.items():
                embed['fields'].append({
                    'name': key,
                    'value': str(value),
                    'inline': True
                })
        
        payload = {'embeds': [embed]}
        requests.post(self.webhook_url, json=payload)
    
    def _send_slack_webhook(self, title: str, message: str, level: str, data: Optional[Dict]):
        """Envoyer vers Slack"""
        color = {
            'info': '#3498db',
            'warning': '#f39c12',
            'error': '#e74c3c',
            'success': '#2ecc71'
        }.get(level, '#3498db')
        
        attachment = {
            'color': color,
            'title': title,
            'text': message,
            'footer': 'Bot de Trading Crypto',
            'ts': int(datetime.now().timestamp()),
            'fields': []
        }
        
        if data:
            for key, value in data.items():
                attachment['fields'].append({
                    'title': key,
                    'value': str(value),
                    'short': True
                })
        
        payload = {'attachments': [attachment]}
        requests.post(self.webhook_url, json=payload)
    
    # Méthodes de notification spécifiques
    
    def notify_trade_executed(self, pair: str, action: str, volume: float, price: float, profit_loss: Optional[float] = None):
        """Notifier l'exécution d'un trade"""
        title = f"Trade Exécuté - {action.upper()} {pair}"
        
        if profit_loss is not None:
            emoji = "✅" if profit_loss >= 0 else "❌"
            message = f"{emoji} Trade {action} complété avec P&L: {profit_loss:+.2f}€"
            level = 'success' if profit_loss >= 0 else 'warning'
        else:
            message = f"🔄 Ordre {action} exécuté"
            level = 'info'
        
        data = {
            'Paire': pair,
            'Action': action,
            'Volume': f"{volume:.4f}",
            'Prix': f"{price:.2f}€"
        }
        
        if profit_loss is not None:
            data['P&L'] = f"{profit_loss:+.2f}€"
        
        self.send_notification(title, message, level, data)
    
    def notify_price_alert(self, pair: str, current_price: float, alert_type: str, threshold: float):
        """Notifier une alerte de prix"""
        title = f"Alerte Prix - {pair}"
        
        if alert_type == 'above':
            message = f"⬆️ Le prix est passé au-dessus de {threshold:.2f}€"
        else:
            message = f"⬇️ Le prix est passé en dessous de {threshold:.2f}€"
        
        data = {
            'Paire': pair,
            'Prix actuel': f"{current_price:.2f}€",
            'Seuil': f"{threshold:.2f}€",
            'Variation': f"{((current_price - threshold) / threshold * 100):+.2f}%"
        }
        
        self.send_notification(title, message, 'warning', data)
    
    def notify_error(self, error_message: str, error_type: str = "Erreur"):
        """Notifier une erreur"""
        title = f"⚠️ {error_type}"
        self.send_notification(title, error_message, 'error')
    
    def notify_performance_milestone(self, metric: str, value: float, milestone: str):
        """Notifier un jalon de performance atteint"""
        title = f"🎯 Jalon Atteint - {milestone}"
        message = f"Le {metric} a atteint {value:.2f}"
        
        data = {
            'Métrique': metric,
            'Valeur': f"{value:.2f}",
            'Jalon': milestone
        }
        
        self.send_notification(title, message, 'success', data)
    
    def notify_bot_status(self, status: str, details: Optional[str] = None):
        """Notifier un changement de statut du bot"""
        if status == 'started':
            title = "🚀 Bot Démarré"
            message = "Le bot de trading a été démarré avec succès"
            level = 'success'
        elif status == 'stopped':
            title = "⏹️ Bot Arrêté"
            message = "Le bot de trading a été arrêté"
            level = 'warning'
        else:
            title = f"ℹ️ Statut: {status}"
            message = details or f"Le statut du bot est maintenant: {status}"
            level = 'info'
        
        self.send_notification(title, message, level)