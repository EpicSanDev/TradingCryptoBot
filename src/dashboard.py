import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from .trading_bot import TradingBot
from .config import Config

# Initialiser l'application Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout du dashboard
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🤖 Bot de Trading Crypto - Dashboard", className="text-center mb-4"),
            html.Hr()
        ])
    ]),
    
    # Statut du bot
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Statut du Bot"),
                dbc.CardBody(id="bot-status")
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Performance Globale"),
                dbc.CardBody(id="performance-summary")
            ])
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Dernière Analyse"),
                dbc.CardBody(id="last-analysis")
            ])
        ], width=4)
    ], className="mb-4"),
    
    # Graphiques
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Évolution du Profit/Perte"),
                dbc.CardBody([
                    dcc.Graph(id="profit-loss-chart")
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Répartition des Trades"),
                dbc.CardBody([
                    dcc.Graph(id="trades-pie-chart")
                ])
            ])
        ], width=6)
    ], className="mb-4"),
    
    # Indicateurs techniques
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Indicateurs Techniques"),
                dbc.CardBody(id="technical-indicators")
            ])
        ], width=12)
    ], className="mb-4"),
    
    # Historique des trades
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Historique des Trades"),
                dbc.CardBody(id="trades-table")
            ])
        ], width=12)
    ]),
    
    # Intervalle de mise à jour
    dcc.Interval(
        id='interval-component',
        interval=30*1000,  # Mise à jour toutes les 30 secondes
        n_intervals=0
    )
], fluid=True)

# Callback pour le statut du bot
@app.callback(
    Output('bot-status', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_bot_status(n):
    try:
        bot = TradingBot()
        status = bot.get_status()
        
        status_color = "success" if status['is_running'] else "danger"
        status_text = "🟢 En cours" if status['is_running'] else "🔴 Arrêté"
        
        last_check = status['last_check']
        last_check_text = last_check.strftime("%H:%M:%S") if last_check else "Jamais"
        
        return [
            html.H4(status_text, className=f"text-{status_color}"),
            html.P(f"Dernière vérification: {last_check_text}"),
            html.P(f"Paire: {Config.TRADING_PAIR}"),
            html.P(f"Intervalle: {Config.CHECK_INTERVAL} min")
        ]
    except Exception as e:
        return html.P(f"Erreur: {e}", className="text-danger")

# Callback pour le résumé des performances
@app.callback(
    Output('performance-summary', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_performance_summary(n):
    try:
        bot = TradingBot()
        performance = bot.get_status()['performance']
        
        total_trades = performance['total_trades']
        win_rate = performance['win_rate']
        total_pl = performance['total_profit_loss']
        open_trades = performance['open_trades']
        
        pl_color = "success" if total_pl >= 0 else "danger"
        pl_icon = "📈" if total_pl >= 0 else "📉"
        
        return [
            html.H4(f"{pl_icon} {total_pl:.2f} €", className=f"text-{pl_color}"),
            html.P(f"Trades totaux: {total_trades}"),
            html.P(f"Trades ouverts: {open_trades}"),
            html.P(f"Taux de réussite: {win_rate:.1f}%"),
            html.P(f"Trades gagnants: {performance['winning_trades']}"),
            html.P(f"Trades perdants: {performance['losing_trades']}")
        ]
    except Exception as e:
        return html.P(f"Erreur: {e}", className="text-danger")

# Callback pour la dernière analyse
@app.callback(
    Output('last-analysis', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_last_analysis(n):
    try:
        bot = TradingBot()
        last_analysis = bot.get_status()['last_analysis']
        
        if not last_analysis:
            return html.P("Aucune analyse disponible")
        
        current_price = last_analysis['current_price']
        recommendation = last_analysis['recommendation']
        action = recommendation['action']
        
        action_color = {
            'BUY': 'success',
            'SELL': 'danger',
            'HOLD': 'warning'
        }.get(action, 'secondary')
        
        action_icon = {
            'BUY': '🟢',
            'SELL': '🔴',
            'HOLD': '🟡'
        }.get(action, '⚪')
        
        return [
            html.H4(f"{action_icon} {action}", className=f"text-{action_color}"),
            html.P(f"Prix: {current_price:.2f} €"),
            html.P(f"Confiance: {recommendation.get('confidence', 0):.1%}"),
            html.P(f"Raison: {recommendation.get('reason', 'N/A')}"),
            html.P(f"Timestamp: {last_analysis['timestamp'].strftime('%H:%M:%S')}")
        ]
    except Exception as e:
        return html.P(f"Erreur: {e}", className="text-danger")

# Callback pour le graphique profit/perte
@app.callback(
    Output('profit-loss-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_profit_loss_chart(n):
    try:
        bot = TradingBot()
        trades = bot.get_trade_history()
        
        if not trades:
            return go.Figure().add_annotation(
                text="Aucun trade disponible",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Filtrer les trades complétés
        completed_trades = [t for t in trades if t.get('sold', False)]
        
        if not completed_trades:
            return go.Figure().add_annotation(
                text="Aucun trade complété",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Créer le DataFrame
        df = pd.DataFrame(completed_trades)
        df['timestamp'] = pd.to_datetime(df['sell_timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculer le profit/perte cumulé
        df['cumulative_pl'] = df['profit_loss'].cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['cumulative_pl'],
            mode='lines+markers',
            name='Profit/Perte cumulé',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Évolution du Profit/Perte",
            xaxis_title="Date",
            yaxis_title="Profit/Perte (€)",
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Erreur: {e}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

# Callback pour le graphique en camembert des trades
@app.callback(
    Output('trades-pie-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_trades_pie_chart(n):
    try:
        bot = TradingBot()
        performance = bot.get_status()['performance']
        
        labels = ['Gagnants', 'Perdants', 'Ouverts']
        values = [
            performance['winning_trades'],
            performance['losing_trades'],
            performance['open_trades']
        ]
        colors = ['green', 'red', 'orange']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker_colors=colors
        )])
        
        fig.update_layout(
            title="Répartition des Trades",
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Erreur: {e}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )

# Callback pour les indicateurs techniques
@app.callback(
    Output('technical-indicators', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_technical_indicators(n):
    try:
        bot = TradingBot()
        last_analysis = bot.get_status()['last_analysis']
        
        if not last_analysis or 'signals' not in last_analysis:
            return html.P("Aucune analyse technique disponible")
        
        signals = last_analysis['signals']
        indicators = last_analysis.get('indicators', {})
        
        signal_colors = {
            'BUY': 'success',
            'SELL': 'danger',
            'NEUTRAL': 'secondary'
        }
        
        signal_icons = {
            'BUY': '🟢',
            'SELL': '🔴',
            'NEUTRAL': '⚪'
        }
        
        rows = []
        for indicator, signal in signals.items():
            if indicator != 'combined':
                color = signal_colors.get(signal, 'secondary')
                icon = signal_icons.get(signal, '⚪')
                
                # Ajouter la valeur si disponible
                value_text = ""
                if indicator.upper() in indicators:
                    value = indicators[indicator.upper()]
                    if value is not None:
                        value_text = f" ({value:.2f})"
                
                rows.append(
                    dbc.Row([
                        dbc.Col(f"{icon} {indicator.upper()}{value_text}", width=6),
                        dbc.Col(signal, className=f"text-{color}", width=6)
                    ], className="mb-2")
                )
        
        return rows
        
    except Exception as e:
        return html.P(f"Erreur: {e}", className="text-danger")

# Callback pour le tableau des trades
@app.callback(
    Output('trades-table', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_trades_table(n):
    try:
        bot = TradingBot()
        trades = bot.get_trade_history()
        
        if not trades:
            return html.P("Aucun trade dans l'historique")
        
        # Prendre les 10 derniers trades
        recent_trades = trades[-10:]
        
        table_rows = []
        for trade in reversed(recent_trades):
            timestamp = trade['timestamp'].strftime('%Y-%m-%d %H:%M')
            action = trade['action']
            pair = trade['pair']
            price = f"{trade['price']:.2f}"
            volume = f"{trade['volume']:.6f}"
            
            status = "Vendu" if trade.get('sold', False) else "Ouvert"
            status_color = "success" if trade.get('sold', False) else "warning"
            
            profit_loss = ""
            if trade.get('profit_loss') is not None:
                pl = trade['profit_loss']
                pl_color = "success" if pl >= 0 else "danger"
                profit_loss = html.Span(
                    f"{pl:.2f} € ({trade['profit_loss_percent']:.1f}%)",
                    className=f"text-{pl_color}"
                )
            
            table_rows.append(
                html.Tr([
                    html.Td(timestamp),
                    html.Td(action),
                    html.Td(pair),
                    html.Td(price),
                    html.Td(volume),
                    html.Td(status, className=f"text-{status_color}"),
                    html.Td(profit_loss)
                ])
            )
        
        return dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Date"),
                    html.Th("Action"),
                    html.Th("Paire"),
                    html.Th("Prix"),
                    html.Th("Volume"),
                    html.Th("Statut"),
                    html.Th("Profit/Perte")
                ])
            ]),
            html.Tbody(table_rows)
        ], striped=True, bordered=True, hover=True)
        
    except Exception as e:
        return html.P(f"Erreur: {e}", className="text-danger")

def run_dashboard(host='0.0.0.0', port=8050, debug=False):
    """Lancer le dashboard web"""
    print(f"🚀 Démarrage du dashboard sur http://{host}:{port}")
    app.run_server(host=host, port=port, debug=debug)

if __name__ == '__main__':
    run_dashboard() 