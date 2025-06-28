# Bot de Trading Crypto Avancé - Kraken Edition

Un bot de trading automatique sophistiqué pour Kraken avec support multi-paire, modes spot/futures, et money management avancé.

## 🚀 Fonctionnalités Avancées

### 📊 Mode Multi-Paire
- Trading simultané sur plusieurs paires de cryptomonnaies
- Gestion indépendante de chaque paire
- Configuration spécifique par paire (stop-loss, take-profit, levier, allocation)

### 💰 Money Management Sophistiqué
- **Calcul Kelly** : Optimisation mathématique de la taille des positions
- **Méthode Martingale** : Augmentation progressive après les pertes
- **Sizing Fixe** : Taille de position constante
- Gestion du drawdown avec réduction automatique des positions
- Contrôle des risques corrélés entre paires

### 🔄 Modes de Trading
- **Mode Spot** : Trading classique sans levier
- **Mode Futures** : Trading avec levier (jusqu'à 10x)
- Clés API séparées pour chaque mode
- Gestion automatique des marges et liquidations

### 🛡️ Gestion des Risques Avancée
- Stop-loss dynamiques basés sur l'ATR
- Trailing stops automatiques
- Take-profit partiels
- Limitation du risque par trade et par corrélation
- Surveillance continue du drawdown

### 📈 Indicateurs Techniques
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Moving Averages (rapide et lente)
- Support et résistance automatiques

## 🛠️ Installation

### Prérequis
- Python 3.8+
- Compte Kraken avec API activée
- Compte Kraken Futures (pour le mode futures)

### Installation des dépendances
```bash
pip install -r requirements.txt
```

### Configuration
1. Copiez le fichier de configuration :
```bash
cp config.env.example config.env
```

2. Éditez `config.env` avec vos clés API et paramètres :
```env
# Clés API Spot
SPOT_API_KEY=votre_clé_api_spot
SPOT_SECRET_KEY=votre_clé_secrète_spot

# Clés API Futures
FUTURES_API_KEY=votre_clé_api_futures
FUTURES_SECRET_KEY=votre_clé_secrète_futures

# Mode de trading
TRADING_MODE=spot

# Paires de trading
TRADING_PAIRS=["XXBTZEUR", "XETHZEUR"]

# Capital d'investissement
INVESTMENT_AMOUNT=1000
```

## 🎯 Utilisation

### Démarrage rapide
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

### Commandes disponibles
```bash
# Afficher la configuration
python advanced_main.py --config

# Afficher le statut
python advanced_main.py --status

# Mode test (pas de trading réel)
python advanced_main.py --test
```

### Mode interactif
Le mode manuel offre des commandes avancées :
```
buy <pair> [volume]  - Acheter sur une paire
sell <pair>          - Vendre sur une paire
status               - Afficher le statut
history              - Historique des trades
positions            - Positions actuelles
performance          - Performance détaillée
pairs                - Liste des paires configurées
quit                 - Quitter
```

## ⚙️ Configuration Avancée

### Configuration par paire
Vous pouvez définir des paramètres spécifiques pour chaque paire :

```env
PAIR_CONFIGS={
  "XXBTZEUR": {
    "stop_loss": 3,
    "take_profit": 8,
    "leverage": 2,
    "allocation": 60
  },
  "XETHZEUR": {
    "stop_loss": 5,
    "take_profit": 12,
    "leverage": 3,
    "allocation": 40
  }
}
```

### Money Management
Choisissez votre stratégie de sizing :

- **Kelly** : Optimisation mathématique (recommandé pour les traders expérimentés)
- **Martingale** : Augmentation après les pertes (risqué)
- **Fixed** : Taille constante (conservateur)

### Paramètres de risque
```env
# Risque maximum par trade (2% du capital)
MAX_RISK_PER_TRADE=2

# Risque maximum pour paires corrélées (5% du capital)
MAX_CORRELATED_RISK=5

# Drawdown maximum autorisé (20%)
MAX_DRAWDOWN=20

# Réduction des positions en cas de drawdown (50%)
DRAWDOWN_REDUCTION=0.5
```

## 📊 Monitoring et Performance

### Métriques suivies
- Taux de réussite des trades
- Profit/Perte total et moyen
- Drawdown maximum et actuel
- Ratio de Sharpe
- Performance par paire

### Logs détaillés
Le bot génère des logs complets dans `trading_bot.log` :
- Décisions de trading
- Exécution des ordres
- Gestion des risques
- Performances

## 🔒 Sécurité

### Bonnes pratiques
1. **Clés API limitées** : Utilisez des clés avec permissions minimales
2. **Authentification 2FA** : Activez la 2FA sur votre compte Kraken
3. **Test en premier** : Commencez avec de petits montants
4. **Surveillance** : Vérifiez régulièrement les performances
5. **Sauvegarde** : Gardez une copie de votre configuration

### Permissions API recommandées
- **Spot** : View, Trade
- **Futures** : View, Trade, Transfer (si nécessaire)

## ⚠️ Avertissements

### Risques du trading
- Le trading de cryptomonnaies est hautement spéculatif
- Les pertes peuvent dépasser votre investissement initial
- Le mode futures avec levier amplifie les risques
- Ne tradez que ce que vous pouvez vous permettre de perdre

### Limitations techniques
- Le bot dépend de la connectivité internet
- Les API Kraken peuvent avoir des limitations de fréquence
- Les conditions de marché peuvent changer rapidement
- Aucune garantie de profit

## 🧪 Tests

### Mode test
```bash
python advanced_main.py --test
```
Le mode test simule les trades sans effectuer d'ordres réels.

### Tests unitaires
```bash
pytest test_bot.py
```

## 📈 Exemples de Configuration

### Configuration conservatrice (débutant)
```env
TRADING_MODE=spot
TRADING_PAIRS=["XXBTZEUR"]
POSITION_SIZING_METHOD=fixed
MAX_RISK_PER_TRADE=1
MAX_DRAWDOWN=10
INVESTMENT_AMOUNT=100
```

### Configuration agressive (expérimenté)
```env
TRADING_MODE=futures
TRADING_PAIRS=["XXBTZEUR", "XETHZEUR", "ADAUSD"]
POSITION_SIZING_METHOD=kelly
MAX_RISK_PER_TRADE=3
MAX_DRAWDOWN=25
DEFAULT_LEVERAGE=5
INVESTMENT_AMOUNT=1000
```

## 🤝 Contribution

Les contributions sont les bienvenues ! Voici comment contribuer :

1. Fork le projet
2. Créez une branche pour votre fonctionnalité
3. Committez vos changements
4. Poussez vers la branche
5. Ouvrez une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## 📞 Support

Pour obtenir de l'aide :
1. Consultez la documentation
2. Vérifiez les issues existantes
3. Créez une nouvelle issue avec les détails de votre problème

## 🔄 Mises à jour

Le bot est régulièrement mis à jour avec :
- Nouvelles fonctionnalités
- Corrections de bugs
- Améliorations de performance
- Nouvelles stratégies de trading

Restez informé des mises à jour en surveillant ce repository.

---

**⚠️ DISCLAIMER : Ce bot est fourni à des fins éducatives. Le trading de cryptomonnaies comporte des risques importants. Utilisez-le à vos propres risques.** 