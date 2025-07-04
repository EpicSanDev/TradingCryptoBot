"""
Module de Visualisation des Résultats de Backtest
===============================================

Ce module génère des graphiques et rapports visuels des performances
du backtest pour analyser la stratégie de trading.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os
import logging

# Configuration du style matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class BacktestVisualizer:
    """Générateur de visualisations pour les résultats de backtest"""
    
    def __init__(self, results: Dict, output_dir: str = "backtest_reports"):
        """
        Initialiser le visualiseur
        
        Args:
            results: Résultats du backtest
            output_dir: Répertoire de sortie des graphiques
        """
        self.results = results
        self.output_dir = output_dir
        
        # Créer le répertoire de sortie
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Vérifier si les résultats contiennent une erreur
        if 'error' in results:
            logging.warning(f"Erreur dans les résultats du backtest: {results['error']}")
            # Créer des données vides pour éviter les erreurs
            self.equity_df = pd.DataFrame(columns=['timestamp', 'equity', 'drawdown', 'peak_equity'])
            self.trades_df = pd.DataFrame(columns=['timestamp', 'pair', 'action', 'price', 'volume', 
                                                  'profit_loss', 'profit_loss_percent', 'hold_duration', 'fees'])
            logging.info("Visualiseur initialisé avec des données vides (erreur de backtest)")
        else:
            # Préparer les données
            if 'equity_curve' not in results:
                logging.warning("Clé 'equity_curve' manquante dans les résultats")
                self.equity_df = pd.DataFrame(columns=['timestamp', 'equity', 'drawdown', 'peak_equity'])
            else:
                self.equity_df = pd.DataFrame(results['equity_curve'])
            
            if 'trades_df' not in results:
                logging.warning("Clé 'trades_df' manquante dans les résultats")
                self.trades_df = pd.DataFrame(columns=['timestamp', 'pair', 'action', 'price', 'volume', 
                                                      'profit_loss', 'profit_loss_percent', 'hold_duration', 'fees'])
            else:
                self.trades_df = results['trades_df']
            
            logging.info(f"Visualiseur initialisé avec {len(self.trades_df)} trades")
    
    def generate_complete_report(self, title: str = "Rapport de Backtest") -> str:
        """
        Générer un rapport complet avec tous les graphiques
        
        Args:
            title: Titre du rapport
            
        Returns:
            Chemin du fichier HTML généré
        """
        logging.info("Génération du rapport complet de backtest...")
        
        # Créer le rapport HTML interactif
        html_file = self._create_interactive_dashboard(title)
        
        # Générer les graphiques statiques
        self._generate_static_charts()
        
        logging.info(f"Rapport généré: {html_file}")
        return html_file
    
    def _create_interactive_dashboard(self, title: str) -> str:
        """Créer un dashboard interactif avec Plotly"""
        # Créer les sous-graphiques
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                'Courbe d\'Equity', 'Drawdown',
                'Distribution des Profits/Pertes', 'Trades par Paire',
                'Performance Mensuelle', 'Durée des Positions',
                'Évolution du Capital', 'Analyse des Risques'
            ],
            specs=[
                [{"secondary_y": True}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"colspan": 2}, None]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # 1. Courbe d'equity avec drawdown
        self._add_equity_curve(fig, row=1, col=1)
        
        # 2. Graphique de drawdown
        self._add_drawdown_chart(fig, row=1, col=2)
        
        # 3. Distribution des profits/pertes
        self._add_profit_distribution(fig, row=2, col=1)
        
        # 4. Trades par paire
        self._add_trades_by_pair(fig, row=2, col=2)
        
        # 5. Performance mensuelle
        self._add_monthly_performance(fig, row=3, col=1)
        
        # 6. Durée des positions
        self._add_position_duration(fig, row=3, col=2)
        
        # 7. Évolution du capital
        self._add_capital_evolution(fig, row=4, col=1)
        
        # Configuration du layout
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24}
            },
            height=1200,
            showlegend=True,
            template='plotly_white'
        )
        
        # Sauvegarder le dashboard
        html_file = os.path.join(self.output_dir, 'backtest_dashboard.html')
        fig.write_html(html_file)
        
        return html_file
    
    def _add_equity_curve(self, fig, row: int, col: int):
        """Ajouter la courbe d'equity"""
        if self.equity_df.empty:
            # Ajouter un message si pas de données
            fig.add_annotation(
                x=0.5, y=0.5,
                xref=f'x{row}{col}', yref=f'y{row}{col}',
                text="Aucune donnée d'equity disponible",
                showarrow=False,
                font=dict(size=14, color="gray")
            )
            return
        
        fig.add_trace(
            go.Scatter(
                x=self.equity_df['timestamp'],
                y=self.equity_df['equity'],
                mode='lines',
                name='Equity',
                line=dict(color='blue', width=2),
                hovertemplate='%{x}<br>Equity: %{y:.2f}€<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Ajouter la ligne du capital initial
        if 'summary' in self.results and 'initial_capital' in self.results['summary']:
            fig.add_hline(
                y=self.results['summary']['initial_capital'],
                line_dash="dash",
                line_color="gray",
                annotation_text="Capital Initial",
                row=row, col=col
            )
        
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Equity (€)", row=row, col=col)
    
    def _add_drawdown_chart(self, fig, row: int, col: int):
        """Ajouter le graphique de drawdown"""
        if self.equity_df.empty:
            # Ajouter un message si pas de données
            fig.add_annotation(
                x=0.5, y=0.5,
                xref=f'x{row}{col}', yref=f'y{row}{col}',
                text="Aucune donnée de drawdown disponible",
                showarrow=False,
                font=dict(size=14, color="gray")
            )
            return
        
        fig.add_trace(
            go.Scatter(
                x=self.equity_df['timestamp'],
                y=-self.equity_df['drawdown'],  # Négatif pour afficher vers le bas
                mode='lines',
                name='Drawdown',
                fill='tonexty',
                line=dict(color='red', width=1),
                fillcolor='rgba(255,0,0,0.3)',
                hovertemplate='%{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
            ),
            row=row, col=col
        )
        
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Drawdown (%)", row=row, col=col)
    
    def _add_profit_distribution(self, fig, row: int, col: int):
        """Ajouter la distribution des profits/pertes"""
        if self.trades_df.empty:
            # Ajouter un message si pas de trades
            fig.add_annotation(
                x=0.5, y=0.5,
                xref=f'x{row}{col}', yref=f'y{row}{col}',
                text="Aucun trade à afficher",
                showarrow=False,
                font=dict(size=14, color="gray")
            )
            return
        
        sell_trades = self.trades_df[self.trades_df['action'] == 'SELL']
        
        if not sell_trades.empty:
            fig.add_trace(
                go.Histogram(
                    x=sell_trades['profit_loss_percent'],
                    nbinsx=30,
                    name='Distribution P&L',
                    marker_color='lightblue',
                    opacity=0.7,
                    hovertemplate='Rendement: %{x:.2f}%<br>Nombre: %{y}<extra></extra>'
                ),
                row=row, col=col
            )
        else:
            # Ajouter un message si pas de trades de vente
            fig.add_annotation(
                x=0.5, y=0.5,
                xref=f'x{row}{col}', yref=f'y{row}{col}',
                text="Aucun trade de vente",
                showarrow=False,
                font=dict(size=14, color="gray")
            )
        
        fig.update_xaxes(title_text="Rendement (%)", row=row, col=col)
        fig.update_yaxes(title_text="Nombre de Trades", row=row, col=col)
    
    def _add_trades_by_pair(self, fig, row: int, col: int):
        """Ajouter les trades par paire"""
        if self.trades_df.empty:
            # Ajouter un message si pas de trades
            fig.add_annotation(
                x=0.5, y=0.5,
                xref=f'x{row}{col}', yref=f'y{row}{col}',
                text="Aucun trade par paire",
                showarrow=False,
                font=dict(size=14, color="gray")
            )
            return
        
        trades_by_pair = self.trades_df.groupby('pair').size().reset_index(name='count')
        
        if not trades_by_pair.empty:
            fig.add_trace(
                go.Bar(
                    x=trades_by_pair['pair'],
                    y=trades_by_pair['count'],
                    name='Trades par Paire',
                    marker_color='lightgreen',
                    hovertemplate='%{x}<br>Trades: %{y}<extra></extra>'
                ),
                row=row, col=col
            )
        else:
            # Ajouter un message si pas de données
            fig.add_annotation(
                x=0.5, y=0.5,
                xref=f'x{row}{col}', yref=f'y{row}{col}',
                text="Aucune donnée de trades",
                showarrow=False,
                font=dict(size=14, color="gray")
            )
        
        fig.update_xaxes(title_text="Paire", row=row, col=col)
        fig.update_yaxes(title_text="Nombre de Trades", row=row, col=col)
    
    def _add_monthly_performance(self, fig, row: int, col: int):
        """Ajouter la performance mensuelle"""
        if self.equity_df.empty:
            # Ajouter un message si pas de données
            fig.add_annotation(
                x=0.5, y=0.5,
                xref=f'x{row}{col}', yref=f'y{row}{col}',
                text="Aucune donnée de performance",
                showarrow=False,
                font=dict(size=14, color="gray")
            )
            return
        
        self.equity_df['month'] = self.equity_df['timestamp'].dt.to_period('M')
        monthly_returns = self.equity_df.groupby('month')['equity'].agg(['first', 'last'])
        monthly_returns['return'] = ((monthly_returns['last'] - monthly_returns['first']) / monthly_returns['first']) * 100
        
        if not monthly_returns.empty:
            colors = ['green' if x >= 0 else 'red' for x in monthly_returns['return']]
            
            fig.add_trace(
                go.Bar(
                    x=[str(x) for x in monthly_returns.index],
                    y=monthly_returns['return'],
                    name='Performance Mensuelle',
                    marker_color=colors,
                    hovertemplate='%{x}<br>Rendement: %{y:.2f}%<extra></extra>'
                ),
                row=row, col=col
            )
        else:
            # Ajouter un message si pas de données
            fig.add_annotation(
                x=0.5, y=0.5,
                xref=f'x{row}{col}', yref=f'y{row}{col}',
                text="Aucune performance mensuelle",
                showarrow=False,
                font=dict(size=14, color="gray")
            )
        
        fig.update_xaxes(title_text="Mois", row=row, col=col)
        fig.update_yaxes(title_text="Rendement Mensuel (%)", row=row, col=col)
    
    def _add_position_duration(self, fig, row: int, col: int):
        """Ajouter la durée des positions"""
        sell_trades = self.trades_df[self.trades_df['action'] == 'SELL']
        
        if not sell_trades.empty and 'hold_duration' in sell_trades.columns:
            fig.add_trace(
                go.Histogram(
                    x=sell_trades['hold_duration'],
                    nbinsx=20,
                    name='Durée des Positions',
                    marker_color='orange',
                    opacity=0.7,
                    hovertemplate='Durée: %{x:.1f}h<br>Nombre: %{y}<extra></extra>'
                ),
                row=row, col=col
            )
        
        fig.update_xaxes(title_text="Durée (heures)", row=row, col=col)
        fig.update_yaxes(title_text="Nombre de Positions", row=row, col=col)
    
    def _add_capital_evolution(self, fig, row: int, col: int):
        """Ajouter l'évolution du capital"""
        # Calculer l'évolution cumulative
        sell_trades = self.trades_df[self.trades_df['action'] == 'SELL'].sort_values('timestamp')
        
        if not sell_trades.empty:
            sell_trades['cumulative_pnl'] = sell_trades['profit_loss'].cumsum()
            sell_trades['cumulative_capital'] = self.results['summary']['initial_capital'] + sell_trades['cumulative_pnl']
            
            fig.add_trace(
                go.Scatter(
                    x=sell_trades['timestamp'],
                    y=sell_trades['cumulative_capital'],
                    mode='lines+markers',
                    name='Capital Cumulé',
                    line=dict(color='purple', width=2),
                    hovertemplate='%{x}<br>Capital: %{y:.2f}€<extra></extra>'
                ),
                row=row, col=col
            )
        
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Capital (€)", row=row, col=col)
    
    def _generate_static_charts(self):
        """Générer des graphiques statiques avec matplotlib"""
        # Graphique 1: Courbe d'equity et drawdown
        self._create_equity_drawdown_chart()
        
        # Graphique 2: Distribution des rendements
        self._create_returns_distribution()
        
        # Graphique 3: Analyse des trades
        self._create_trades_analysis()
        
        # Graphique 4: Matrice de corrélation des performances
        self._create_correlation_matrix()
        
        # Graphique 5: Rolling Sharpe ratio
        self._create_rolling_metrics()
    
    def _create_equity_drawdown_chart(self):
        """Créer le graphique equity/drawdown"""
        if self.equity_df.empty:
            logging.warning("Pas de données d'equity pour créer le graphique equity/drawdown")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Courbe d'equity
        ax1.plot(self.equity_df['timestamp'], self.equity_df['equity'], 
                linewidth=2, color='blue', label='Equity')
        
        if 'summary' in self.results and 'initial_capital' in self.results['summary']:
            ax1.axhline(y=self.results['summary']['initial_capital'], 
                       color='gray', linestyle='--', alpha=0.7, label='Capital Initial')
        
        ax1.set_ylabel('Equity (€)')
        ax1.set_title('Évolution de l\'Equity', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        ax2.fill_between(self.equity_df['timestamp'], 0, -self.equity_df['drawdown'], 
                        color='red', alpha=0.3, label='Drawdown')
        ax2.plot(self.equity_df['timestamp'], -self.equity_df['drawdown'], 
                color='red', linewidth=1)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Format des dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'equity_drawdown.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_returns_distribution(self):
        """Créer la distribution des rendements"""
        sell_trades = self.trades_df[self.trades_df['action'] == 'SELL']
        
        if sell_trades.empty:
            logging.warning("Pas de trades de vente pour créer la distribution des rendements")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histogramme des rendements
        ax1.hist(sell_trades['profit_loss_percent'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Seuil de rentabilité')
        ax1.set_xlabel('Rendement (%)')
        ax1.set_ylabel('Fréquence')
        ax1.set_title('Distribution des Rendements')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot des rendements par paire
        if len(sell_trades['pair'].unique()) > 1:
            sell_trades.boxplot(column='profit_loss_percent', by='pair', ax=ax2)
            ax2.set_title('Rendements par Paire')
            ax2.set_xlabel('Paire')
            ax2.set_ylabel('Rendement (%)')
        else:
            ax2.text(0.5, 0.5, 'Une seule paire\ntradée', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Rendements par Paire')
        
        # Évolution des rendements dans le temps
        ax3.scatter(sell_trades['timestamp'], sell_trades['profit_loss_percent'], 
                   alpha=0.6, c=sell_trades['profit_loss_percent'], cmap='RdYlGn')
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Rendement (%)')
        ax3.set_title('Évolution des Rendements')
        ax3.grid(True, alpha=0.3)
        
        # Rendements cumulés
        sell_trades_sorted = sell_trades.sort_values('timestamp')
        cumulative_returns = (1 + sell_trades_sorted['profit_loss_percent'] / 100).cumprod()
        ax4.plot(sell_trades_sorted['timestamp'], cumulative_returns, linewidth=2, color='green')
        ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Seuil de rentabilité')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Rendement Cumulé')
        ax4.set_title('Rendements Cumulés')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'returns_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_trades_analysis(self):
        """Créer l'analyse des trades"""
        if self.trades_df.empty:
            logging.warning("Pas de trades pour créer l'analyse des trades")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Trades par paire
        trades_by_pair = self.trades_df.groupby('pair').size()
        if not trades_by_pair.empty:
            trades_by_pair.plot(kind='bar', ax=ax1, color='lightblue')
            ax1.set_title('Nombre de Trades par Paire')
            ax1.set_xlabel('Paire')
            ax1.set_ylabel('Nombre de Trades')
            ax1.tick_params(axis='x', rotation=45)
        
        # Trades par action (BUY/SELL)
        trades_by_action = self.trades_df.groupby('action').size()
        if not trades_by_action.empty:
            trades_by_action.plot(kind='pie', ax=ax2, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
            ax2.set_title('Répartition des Trades')
            ax2.set_ylabel('')
        
        # Volume des trades
        if 'volume' in self.trades_df.columns:
            ax3.scatter(self.trades_df['timestamp'], self.trades_df['volume'], 
                       alpha=0.6, c=self.trades_df['action'].map({'BUY': 'green', 'SELL': 'red'}))
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Volume')
            ax3.set_title('Volume des Trades')
            ax3.grid(True, alpha=0.3)
        
        # Frais des trades
        if 'fees' in self.trades_df.columns:
            ax4.hist(self.trades_df['fees'], bins=20, alpha=0.7, color='orange', edgecolor='black')
            ax4.set_xlabel('Frais (€)')
            ax4.set_ylabel('Fréquence')
            ax4.set_title('Distribution des Frais')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'trades_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_correlation_matrix(self):
        """Créer la matrice de corrélation des performances"""
        if self.equity_df.empty:
            logging.warning("Pas de données d'equity pour créer la matrice de corrélation")
            return
        
        # Calculer les corrélations entre différentes métriques
        correlation_data = self.equity_df[['equity', 'drawdown']].copy()
        
        if len(correlation_data) > 1:
            correlation_matrix = correlation_data.corr()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Matrice de Corrélation des Performances')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_rolling_metrics(self):
        """Créer les métriques mobiles"""
        if len(self.equity_df) < 30:  # Pas assez de données
            logging.warning("Pas assez de données pour créer les métriques mobiles")
            return
        
        # Calculer les métriques mobiles
        self.equity_df['returns'] = self.equity_df['equity'].pct_change()
        
        # Sharpe ratio mobile (fenêtre de 30 jours)
        window = min(30, len(self.equity_df) // 3)
        self.equity_df['rolling_sharpe'] = (
            self.equity_df['returns'].rolling(window).mean() / 
            self.equity_df['returns'].rolling(window).std() * np.sqrt(252)
        )
        
        # Volatilité mobile
        self.equity_df['rolling_volatility'] = self.equity_df['returns'].rolling(window).std() * np.sqrt(252) * 100
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Sharpe ratio mobile
        ax1.plot(self.equity_df['timestamp'], self.equity_df['rolling_sharpe'], 
                linewidth=2, color='blue', label=f'Sharpe Ratio (fenêtre {window}j)')
        ax1.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Bon niveau (1.0)')
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Neutre (0.0)')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.set_title('Sharpe Ratio Mobile', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volatilité mobile
        ax2.plot(self.equity_df['timestamp'], self.equity_df['rolling_volatility'], 
                linewidth=2, color='red', label=f'Volatilité (fenêtre {window}j)')
        ax2.set_ylabel('Volatilité Annualisée (%)')
        ax2.set_xlabel('Date')
        ax2.set_title('Volatilité Mobile', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rolling_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self) -> str:
        """Générer un rapport de synthèse en texte"""
        summary = self.results['summary']
        trades_stats = self.results['trades_stats']
        
        report = f"""
=== RAPPORT DE BACKTEST ===
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RÉSUMÉ FINANCIER:
- Capital initial: {summary['initial_capital']:,.2f}€
- Capital final: {summary['final_capital']:,.2f}€
- Profit/Perte total: {summary['total_profit_loss']:,.2f}€
- Rendement total: {summary['total_return']:.2f}%
- Drawdown maximum: {summary['max_drawdown']:.2f}%
- Ratio de Sharpe: {summary['sharpe_ratio']:.2f}

STATISTIQUES DES TRADES:
- Nombre total de trades: {trades_stats['total_trades']}
- Trades gagnants: {trades_stats['winning_trades']}
- Trades perdants: {trades_stats['losing_trades']}
- Taux de réussite: {trades_stats['win_rate']:.2f}%
- Gain moyen: {trades_stats['avg_win']:.2f}€
- Perte moyenne: {trades_stats['avg_loss']:.2f}€
- Facteur de profit: {trades_stats['profit_factor']:.2f}
- Frais totaux: {trades_stats['total_fees']:.2f}€

ÉVALUATION:
"""
        
        # Évaluation de la performance
        if summary['total_return'] > 10:
            report += "✅ Excellente performance (+10%)\n"
        elif summary['total_return'] > 5:
            report += "✅ Bonne performance (+5%)\n"
        elif summary['total_return'] > 0:
            report += "⚠️  Performance positive mais modeste\n"
        else:
            report += "❌ Performance négative\n"
        
        # Évaluation du risque
        if summary['max_drawdown'] < 5:
            report += "✅ Risque faible (drawdown < 5%)\n"
        elif summary['max_drawdown'] < 15:
            report += "⚠️  Risque modéré (drawdown < 15%)\n"
        else:
            report += "❌ Risque élevé (drawdown > 15%)\n"
        
        # Évaluation de la régularité
        if trades_stats['win_rate'] > 60:
            report += "✅ Très bonne régularité (>60% de réussite)\n"
        elif trades_stats['win_rate'] > 50:
            report += "✅ Bonne régularité (>50% de réussite)\n"
        else:
            report += "⚠️  Régularité à améliorer (<50% de réussite)\n"
        
        # Sauvegarder le rapport
        report_file = os.path.join(self.output_dir, 'summary_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        return report_file