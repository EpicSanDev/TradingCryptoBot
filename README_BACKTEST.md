# 📊 Système de Backtest - Bot de Trading Crypto

Ce système de backtest permet de tester et d'optimiser automatiquement votre stratégie de trading crypto en utilisant des données historiques de Kraken.

## 🚀 Fonctionnalités

- **📈 Téléchargement automatique** des données historiques de Kraken
- **🔄 Simulation complète** de trading avec votre stratégie existante  
- **📊 Visualisations interactives** des performances avec graphiques détaillés
- **🎯 Optimisation automatique** des paramètres de stratégie
- **🧪 Analyse walk-forward** pour tester la robustesse
- **📝 Rapports détaillés** HTML et texte

## 📦 Installation

### 1. Installer les dépendances

```bash
pip install -r requirements_backtest.txt
```

### 2. Vérifier la configuration

```bash
python backtest_main.py --config-test
```

## 🎯 Utilisation

### Backtest de Base

Tester votre stratégie sur 180 jours avec un capital de 10 000€ :

```bash
python backtest_main.py --pairs BTCEUR --days 180 --capital 10000
```

Tester plusieurs paires :

```bash
python backtest_main.py --pairs BTCEUR ETHEUR ADAEUR --days 365 --capital 20000
```

Changer l'intervalle (1h par défaut) :

```bash
python backtest_main.py --pairs BTCEUR --interval 240 --days 90  # 4 heures
```

### Optimisation des Paramètres

#### Optimisation Bayésienne (Recommandé)

```bash
python backtest_main.py --optimize --method optuna --trials 100 --metric total_return
```

#### Recherche par Grille

```bash
python backtest_main.py --optimize --method grid --trials 500 --metric sharpe_ratio
```

#### Évolution Différentielle

```bash
python backtest_main.py --optimize --method differential --trials 50 --metric calmar_ratio
```

### Analyse Walk-Forward

Tester la robustesse de la stratégie dans le temps :

```bash
python backtest_main.py --walk-forward --window 6 --step 1 --days 365
```

## 📊 Métriques d'Optimisation

- `total_return` : Rendement total (%)
- `sharpe_ratio` : Ratio de Sharpe
- `profit_factor` : Facteur de profit
- `win_rate` : Taux de réussite (%)
- `calmar_ratio` : Ratio de Calmar (rendement/drawdown)
- `max_drawdown_adjusted` : Inverse du drawdown maximum

## 📈 Résultats et Rapports

### Rapports Générés

1. **Dashboard HTML interactif** (`backtest_dashboard.html`)
   - Courbe d'equity et drawdown
   - Distribution des profits/pertes
   - Performance mensuelle
   - Analyse détaillée des trades

2. **Graphiques statiques** (PNG)
   - `equity_drawdown.png`
   - `returns_analysis.png`
   - `trades_analysis.png`
   - `correlation_matrix.png`
   - `rolling_metrics.png`

3. **Rapport de synthèse** (`summary_report.txt`)
   - Métriques clés
   - Évaluation de la performance
   - Recommandations

### Exemple de Résultats

```
=== RAPPORT DE BACKTEST ===
RÉSUMÉ FINANCIER:
- Capital initial: 10,000.00€
- Capital final: 12,350.00€
- Profit/Perte total: 2,350.00€
- Rendement total: 23.50%
- Drawdown maximum: 8.30%
- Ratio de Sharpe: 1.45

STATISTIQUES DES TRADES:
- Nombre total de trades: 45
- Trades gagnants: 28
- Trades perdants: 17
- Taux de réussite: 62.22%
- Gain moyen: 156.30€
- Perte moyenne: 89.50€
- Facteur de profit: 2.13
```

## ⚙️ Paramètres Optimisables

Le système peut optimiser automatiquement ces paramètres :

### Indicateurs Techniques
- `RSI_PERIOD` (10-30) : Période RSI
- `RSI_OVERSOLD` (20-35) : Seuil de survente RSI
- `RSI_OVERBOUGHT` (65-80) : Seuil de surachat RSI
- `MACD_FAST` (8-15) : MACD rapide
- `MACD_SLOW` (20-30) : MACD lent
- `MACD_SIGNAL` (6-12) : Signal MACD
- `BOLLINGER_PERIOD` (15-25) : Période Bollinger
- `BOLLINGER_STD` (1.5-2.5) : Écart-type Bollinger
- `MA_FAST` (5-15) : Moyenne mobile rapide
- `MA_SLOW` (20-50) : Moyenne mobile lente

### Gestion des Risques
- `STOP_LOSS_PERCENTAGE` (2.0-10.0) : Stop-loss (%)
- `TAKE_PROFIT_PERCENTAGE` (3.0-15.0) : Take-profit (%)
- `MIN_SIGNAL_CONFIDENCE` (0.3-0.8) : Confiance minimale du signal

## 🔧 Configuration Avancée

### Intervalles Supportés

- `1` : 1 minute
- `5` : 5 minutes  
- `15` : 15 minutes
- `30` : 30 minutes
- `60` : 1 heure (défaut)
- `240` : 4 heures
- `1440` : 1 jour

### Paires Supportées

Toutes les paires disponibles sur Kraken, par exemple :
- `BTCEUR`, `ETHEUR`, `ADAEUR`
- `XRPEUR`, `DOTEUR`, `LINKEUR`
- `BTCUSD`, `ETHUSD`, etc.

## 📋 Exemples d'Utilisation

### 1. Test Rapide (7 jours)

```bash
python backtest_main.py --pairs BTCEUR --days 7 --capital 1000
```

### 2. Backtest Complet (1 an)

```bash
python backtest_main.py --pairs BTCEUR ETHEUR --days 365 --capital 50000 --interval 60
```

### 3. Optimisation Multi-Métriques

```bash
# Optimiser pour le rendement
python backtest_main.py --optimize --metric total_return --trials 200

# Optimiser pour le ratio de Sharpe
python backtest_main.py --optimize --metric sharpe_ratio --trials 200

# Optimiser pour minimiser le risque
python backtest_main.py --optimize --metric max_drawdown_adjusted --trials 200
```

### 4. Analyse de Robustesse

```bash
python backtest_main.py --walk-forward --window 3 --step 1 --days 365
```

## 🐛 Dépannage

### Problèmes Courants

1. **Erreur de connexion Kraken**
   ```bash
   python backtest_main.py --config-test
   ```

2. **Manque de données**
   - Vérifiez que les paires sont correctement écrites
   - Réduisez la période de test
   - Vérifiez votre connexion internet

3. **Optimisation lente**
   - Réduisez le nombre d'essais
   - Utilisez un intervalle plus grand (240 ou 1440)
   - Réduisez la période de test

### Mode Debug

```bash
python backtest_main.py --debug --pairs BTCEUR --days 30
```

## 📊 Interprétation des Résultats

### Métriques Clés

- **Rendement Total** > 10% : Excellente performance
- **Ratio de Sharpe** > 1.0 : Bon ratio rendement/risque
- **Drawdown Max** < 15% : Risque acceptable
- **Taux de Réussite** > 50% : Stratégie cohérente
- **Facteur de Profit** > 1.5 : Gains supérieurs aux pertes

### Signaux d'Alerte

⚠️ **Attention si :**
- Drawdown > 20%
- Moins de 10 trades sur la période
- Ratio de Sharpe < 0.5
- Taux de réussite < 40%

## 🔄 Workflow Recommandé

1. **Test initial** : Backtest de base sur 90 jours
2. **Optimisation** : Optimiser avec Optuna (100 essais)
3. **Validation** : Analyse walk-forward sur 1 an
4. **Application** : Appliquer les paramètres optimaux au bot

## 📁 Structure des Fichiers

```
backtest_reports/
├── backtest_dashboard.html     # Dashboard interactif
├── equity_drawdown.png         # Graphique equity/drawdown
├── returns_analysis.png        # Analyse des rendements
├── trades_analysis.png         # Analyse des trades
├── correlation_matrix.png      # Matrice de corrélation
├── rolling_metrics.png         # Métriques mobiles
└── summary_report.txt          # Rapport de synthèse

optimization_optuna_total_return.json  # Résultats d'optimisation
walk_forward_results.json              # Résultats walk-forward
backtest.log                           # Logs détaillés
```

## 🤝 Support

Pour toute question ou problème :

1. Vérifiez ce README
2. Consultez les logs (`backtest.log`)
3. Utilisez le mode debug (`--debug`)
4. Testez la configuration (`--config-test`)

Bon trading ! 🚀📈