# Bot de Trading Crypto Avancé pour Kraken

Un bot de trading automatisé sophistiqué pour Kraken avec interface web moderne, support multi-paire, modes spot et futures, et gestion avancée des risques.

## 🚀 Fonctionnalités Principales

### Trading Avancé
- **Multi-paire** : Trading simultané sur plusieurs paires de cryptomonnaies
- **Modes Spot et Futures** : Support complet avec gestion du levier
- **Money Management** : Méthodes Kelly, Martingale et Fixed sizing
- **Gestion des Risques** : Stop-loss dynamiques, trailing stops, corrélation entre positions
- **Indicateurs Techniques** : RSI, MACD, Bollinger Bands, Moving Averages

### Interface Web Moderne
- **Dashboard en temps réel** : Suivi des positions et performances
- **WebSocket** : Mise à jour en temps réel des prix et positions
- **Graphiques interactifs** : Visualisation de la performance
- **Contrôle complet** : Démarrage/arrêt du bot, trades manuels
- **Responsive** : Interface adaptée à tous les écrans

## � Prérequis

- Python 3.8+
- Node.js 14+ et npm
- Compte Kraken avec API activée
- Clés API pour Spot et/ou Futures

## 🔧 Installation

### 1. Cloner le repository
```bash
git clone https://github.com/yourusername/TradingCryptoBot.git
cd TradingCryptoBot
```

### 2. Installer les dépendances Python
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 3. Installer les dépendances de l'interface web
```bash
cd web
npm install
cd ..
```

### 4. Configuration
```bash
cp config.env.example config.env
```

Éditez `config.env` avec vos paramètres :

```env
# API Kraken Spot
SPOT_API_KEY=votre_cle_api_spot
SPOT_SECRET_KEY=votre_cle_secrete_spot

# API Kraken Futures
FUTURES_API_KEY=votre_cle_api_futures
FUTURES_SECRET_KEY=votre_cle_secrete_futures

# Configuration Multi-Paire
TRADING_PAIRS=["XXBTZEUR", "XETHZEUR", "XLTCZEUR"]

# Mode de Trading
TRADING_MODE=spot  # ou 'futures'

# Capital et Risques
INVESTMENT_AMOUNT=1000
MAX_RISK_PER_TRADE=2
MAX_DRAWDOWN=20

# Money Management
POSITION_SIZING_METHOD=kelly  # 'fixed', 'kelly', ou 'martingale'
```

## 🚀 Utilisation

### Lancer l'interface web

```bash
# Terminal 1 - Backend
python web_app.py

# Terminal 2 - Frontend
cd web
npm start
```

L'interface sera accessible à http://localhost:3000

### Lancer le bot en ligne de commande

```bash
# Mode automatique
python advanced_main.py

# Mode manuel interactif
python advanced_main.py --manual

# Mode spot uniquement
python advanced_main.py --spot

# Mode futures uniquement
python advanced_main.py --futures
```

## 🌐 Interface Web

### Dashboard Principal
- Vue d'ensemble des performances
- Métriques en temps réel (P&L, taux de réussite, Sharpe ratio)
- Statut du bot et mode de trading

### Fonctionnalités
1. **Panneau de Contrôle** : Démarrer/arrêter le bot, choisir le mode
2. **Paires de Trading** : Prix en temps réel et variations
3. **Positions Ouvertes** : Suivi des P&L non réalisés
4. **Historique** : Tous les trades avec détails
5. **Graphique de Performance** : Évolution du P&L cumulatif

## 📊 Stratégies de Trading

### Indicateurs Utilisés
- **RSI** : Détection des zones de surachat/survente
- **MACD** : Identification des tendances et momentum
- **Bollinger Bands** : Volatilité et niveaux de support/résistance
- **Moving Averages** : Confirmation de tendance

### Gestion des Risques
- Stop-loss automatiques basés sur l'ATR ou pourcentage
- Take-profit avec ratio risque/récompense configurable
- Trailing stops pour protéger les profits
- Réduction de l'exposition en cas de drawdown

## � Sécurité

- Les clés API sont stockées localement dans `config.env`
- Jamais de commit des clés dans le repository
- Support des permissions API limitées (trading uniquement)
- Logs détaillés pour audit

## � Performance et Optimisation

- Cache intelligent pour réduire les appels API
- Exécution parallèle pour le multi-paire
- WebSocket pour les mises à jour temps réel
- Money management adaptatif selon les performances

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add AmazingFeature'`)
4. Push la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## ⚠️ Avertissement

Le trading de cryptomonnaies comporte des risques importants. Ce bot est fourni à titre éducatif. Utilisez-le à vos propres risques et ne tradez que ce que vous pouvez vous permettre de perdre.

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 📞 Support

Pour toute question ou problème :
- Ouvrir une issue sur GitHub
- Consulter la documentation dans `README_ADVANCED.md`
- Vérifier les logs dans le terminal pour debug 