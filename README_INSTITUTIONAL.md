# Bot de Trading Crypto Institutionnel

## 🚀 Introduction

Ce bot de trading crypto a été transformé en une solution institutionnelle de niveau professionnel, intégrant les techniques les plus avancées utilisées par les hedge funds et les desks de trading institutionnels.

## 📋 Nouvelles Fonctionnalités Institutionnelles

### 1. **Analyse de Microstructure du Marché**
- Analyse approfondie du carnet d'ordres
- Détection des murs d'ordres et niveaux de liquidité
- Calcul du spread effectif et de l'impact de marché
- Score de liquidité en temps réel

### 2. **Machine Learning et IA**
- Random Forest pour la classification des signaux
- Détection d'anomalies avec Isolation Forest
- Prédictions adaptatives basées sur l'historique
- Feature engineering automatique

### 3. **Analyse Multi-Timeframe**
- Analyse simultanée sur 1m, 5m, 15m, 1h, 4h
- Confluence des signaux multi-timeframe
- Détection automatique des régimes de marché
- Structure de marché (HH, HL, LL, LH)

### 4. **Exécution Algorithmique**
- TWAP (Time Weighted Average Price)
- VWAP (Volume Weighted Average Price)
- Iceberg orders pour minimiser l'impact
- Smart order routing avec slicing automatique

### 5. **Gestion des Risques Avancée**
- VaR (Value at Risk) et CVaR
- Matrice de corrélation entre paires
- Stops dynamiques avec trailing et breakeven
- Mode défensif automatique en cas de drawdown

## 🛠️ Installation

### Prérequis
- Python 3.8+
- Compte Kraken avec API activée
- Capital minimum recommandé : 10,000 EUR

### Installation des dépendances
```bash
pip install -r requirements.txt
```

### Configuration

1. **Copier le fichier de configuration institutionnelle :**
```bash
cp config.env.institutional config.env
```

2. **Éditer config.env avec vos clés API :**
```env
SPOT_API_KEY=votre_cle_api
SPOT_SECRET_KEY=votre_secret_api
```

3. **Personnaliser les paramètres selon vos besoins**

## 🚀 Démarrage

### Mode Institutionnel (Recommandé)
```bash
python start_institutional.py
```

### Options de démarrage
```bash
# Mode dry-run (simulation sans ordres réels)
python start_institutional.py --dry-run

# Mode paper trading
python start_institutional.py --paper-trading

# Changer de stratégie
python start_institutional.py --strategy conservative
```

## 📊 Configuration Recommandée

### Pour un Capital de 100,000 EUR

```env
# Capital et risque
INVESTMENT_AMOUNT=100000
MAX_RISK_PER_TRADE=1
MAX_DRAWDOWN=15
MIN_SIGNAL_CONFIDENCE=0.7

# Position sizing
POSITION_SIZING_METHOD=kelly
KELLY_FRACTION=0.25

# Paires diversifiées
TRADING_PAIRS=["XXBTZEUR", "XETHZEUR", "ADAEUR", "DOTEUR", "LINKEUR"]
```

### Pour un Capital de 10,000 EUR

```env
# Capital et risque
INVESTMENT_AMOUNT=10000
MAX_RISK_PER_TRADE=2
MAX_DRAWDOWN=20
MIN_SIGNAL_CONFIDENCE=0.75

# Position sizing
POSITION_SIZING_METHOD=fixed
MAX_POSITION_SIZE=0.1

# Paires principales uniquement
TRADING_PAIRS=["XXBTZEUR", "XETHZEUR"]
```

## 📈 Métriques de Performance

Le bot surveille en temps réel :

- **Sharpe Ratio** : Rendement ajusté au risque
- **Sortino Ratio** : Focus sur la volatilité négative
- **Win Rate** : Pourcentage de trades gagnants
- **Profit Factor** : Ratio gains/pertes
- **Max Drawdown** : Perte maximale depuis un pic
- **VaR 95%** : Perte maximale avec 95% de confiance

## 🔧 Paramètres Avancés

### Modes de Trading

1. **Institutional** (Par défaut)
   - Seuils de confiance élevés
   - Exécution algorithmique
   - Gestion des risques stricte

2. **Aggressive**
   - Seuils de confiance réduits
   - Positions plus larges
   - Plus de trades

3. **Conservative**
   - Seuils très élevés
   - Positions réduites
   - Focus sur la préservation du capital

### Méthodes de Position Sizing

1. **Kelly Criterion**
   - Optimisation mathématique
   - Ajustement selon le win rate
   - Fraction Kelly configurable

2. **Fixed**
   - Pourcentage fixe du capital
   - Simple et prévisible
   - Bon pour débuter

3. **Martingale**
   - Augmentation après pertes
   - Risqué mais potentiellement profitable
   - Limité à 3x maximum

## 🛡️ Sécurité et Risques

### Mécanismes de Protection

- **Stops automatiques** : Stop-loss sur chaque position
- **Limites de corrélation** : Évite la surexposition
- **Mode défensif** : Activation automatique si drawdown > seuil
- **Time-based exits** : Fermeture des positions stagnantes
- **Circuit breakers** : Arrêt si conditions anormales

### Recommandations

1. **Toujours commencer en dry-run** pour valider la configuration
2. **Paper trading minimum 1 mois** avant le trading réel
3. **Commencer avec 10% du capital** prévu
4. **Augmenter progressivement** selon les performances
5. **Surveiller quotidiennement** les métriques

## 📱 Notifications

Le bot supporte :
- Email (SMTP)
- Webhooks (Discord, Slack, etc.)
- Logs détaillés

Configuration dans config.env :
```env
NOTIFICATIONS_ENABLED=true
WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_URL
```

## 🔍 Monitoring

### Dashboard Web (Optionnel)
```bash
# Démarrer le dashboard
python web_app.py
```
Accéder à http://localhost:5000

### Logs
Les logs sont sauvegardés dans `/logs` avec rotation automatique.

### État du Bot
L'état est sauvegardé dans `bot_state.json` à chaque cycle.

## 🧪 Backtesting

Pour tester la stratégie sur des données historiques :
```bash
python backtest_main.py
```

## 🆘 Support et Dépannage

### Problèmes Courants

1. **"Fonds insuffisants"**
   - Vérifier le solde du compte
   - Réduire INVESTMENT_AMOUNT
   - Vérifier les minimums Kraken (10 EUR)

2. **"Connexion API perdue"**
   - Vérifier les clés API
   - Vérifier la connexion internet
   - Le bot tentera de reconnecter automatiquement

3. **"Limites de risque dépassées"**
   - Réduire MAX_RISK_PER_TRADE
   - Augmenter MIN_SIGNAL_CONFIDENCE
   - Vérifier les corrélations

### Logs de Debug
```bash
# Activer le mode debug
DEBUG_MODE=true
```

## 📜 Licence et Avertissement

**AVERTISSEMENT IMPORTANT** : Le trading de cryptomonnaies comporte des risques significatifs. Ce bot est fourni à titre éducatif. Les performances passées ne garantissent pas les résultats futurs. N'investissez que ce que vous pouvez vous permettre de perdre.

## 🤝 Contribution

Les contributions sont bienvenues ! Voir CONTRIBUTING.md pour les directives.

## 📞 Contact

Pour toute question ou support professionnel, créez une issue sur GitHub.

---

*Bot de Trading Crypto Institutionnel v2.0 - Trading professionnel pour tous*