# 🤖 Bot de Trading Crypto pour Kraken

Un bot de trading automatique pour Kraken qui utilise des indicateurs techniques avancés pour prendre des décisions d'achat et de vente intelligentes.

## ✨ Fonctionnalités

### 📊 Indicateurs Techniques
- **RSI (Relative Strength Index)** - Détecte les conditions de surachat/survente
- **MACD (Moving Average Convergence Divergence)** - Identifie les changements de tendance
- **Bandes de Bollinger** - Détermine les niveaux de support et résistance
- **Moyennes Mobiles** - Analyse les tendances à court et long terme
- **Oscillateur Stochastique** - Confirme les signaux de retournement
- **ATR (Average True Range)** - Mesure la volatilité

### 🛡️ Gestion des Risques
- **Stop-loss automatique** - Limite les pertes
- **Take-profit automatique** - Sécurise les gains
- **Position sizing** - Contrôle la taille des positions
- **Analyse multi-indicateurs** - Confirmation des signaux

### 🔄 Trading Automatique
- **Analyse continue** - Surveillance 24/7 du marché
- **Exécution automatique** - Ordres placés sans intervention
- **Logging détaillé** - Suivi complet des opérations
- **Mode manuel** - Contrôle manuel des trades

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- Compte Kraken avec API activée
- Clés API Kraken (publique et privée)

### 1. Cloner le repository
```bash
git clone <repository-url>
cd TradingCryptoBot
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Configuration
```bash
# Copier le fichier de configuration
cp config.env.example config.env

# Éditer le fichier config.env avec vos clés API
nano config.env
```

### 4. Configuration des clés API Kraken

1. Connectez-vous à votre compte Kraken
2. Allez dans **Settings > API**
3. Créez une nouvelle clé API avec les permissions suivantes :
   - **View Funds** ✅
   - **Trade** ✅
   - **Query Funds** ✅
4. Copiez la **API Key** et la **Private Key**
5. Ajoutez-les dans `config.env`

## ⚙️ Configuration

### Variables d'environnement (`config.env`)

```env
# Kraken API Credentials
KRAKEN_API_KEY=your_api_key_here
KRAKEN_SECRET_KEY=your_secret_key_here

# Trading Configuration
TRADING_PAIR=XXBTZEUR          # Paire de trading (Bitcoin/Euro)
INVESTMENT_AMOUNT=100          # Montant par trade en EUR
MAX_POSITION_SIZE=0.1          # 10% du solde maximum par position
STOP_LOSS_PERCENTAGE=5         # Stop-loss à 5%
TAKE_PROFIT_PERCENTAGE=10      # Take-profit à 10%

# Technical Indicators Configuration
RSI_PERIOD=14                  # Période RSI
RSI_OVERBOUGHT=70              # Niveau de surachat
RSI_OVERSOLD=30                # Niveau de survente
MACD_FAST=12                   # MACD rapide
MACD_SLOW=26                   # MACD lent
MACD_SIGNAL=9                  # Signal MACD
BOLLINGER_PERIOD=20            # Période Bollinger
BOLLINGER_STD=2                # Écart-type Bollinger
MA_FAST=9                      # Moyenne mobile rapide
MA_SLOW=21                     # Moyenne mobile lente

# Trading Schedule
CHECK_INTERVAL=5               # Vérification toutes les 5 minutes
```

### Paires de trading disponibles

| Paire | Description |
|-------|-------------|
| `XXBTZEUR` | Bitcoin/Euro |
| `XETHZEUR` | Ethereum/Euro |
| `XLTCZEUR` | Litecoin/Euro |
| `XXRPZEUR` | Ripple/Euro |
| `XADAZEUR` | Cardano/Euro |

## 🎯 Utilisation

### Mode automatique (recommandé)
```bash
python main.py
```

### Mode manuel interactif
```bash
python main.py --manual
```

### Afficher la configuration
```bash
python main.py --config
```

### Afficher le statut
```bash
python main.py --status
```

### Mode test (pas de trading réel)
```bash
python main.py --test
```

## 📈 Stratégie de Trading

### Signaux d'achat
Le bot achète quand :
- **RSI** < 30 (survente)
- **MACD** croise au-dessus du signal
- **Prix** touche la bande inférieure de Bollinger
- **Moyenne rapide** croise au-dessus de la moyenne lente
- **Stochastique** < 20

### Signaux de vente
Le bot vend quand :
- **RSI** > 70 (surachat)
- **MACD** croise en dessous du signal
- **Prix** touche la bande supérieure de Bollinger
- **Moyenne rapide** croise en dessous de la moyenne lente
- **Stochastique** > 80

### Gestion des risques
- **Stop-loss** : Vente automatique si perte > 5%
- **Take-profit** : Vente automatique si gain > 10%
- **Position sizing** : Maximum 10% du solde par trade

## 📊 Monitoring

### Logs
Les logs sont sauvegardés dans `trading_bot.log` :
```
2024-01-15 10:30:00 - INFO - === Début du cycle de trading ===
2024-01-15 10:30:01 - INFO - Paire: XXBTZEUR
2024-01-15 10:30:01 - INFO - Prix actuel: 45000.0
2024-01-15 10:30:01 - INFO - Signaux:
2024-01-15 10:30:01 - INFO -   RSI: BUY
2024-01-15 10:30:01 - INFO -   MACD: NEUTRAL
2024-01-15 10:30:01 - INFO - Recommandation: BUY
```

### Interface en ligne de commande
En mode manuel, utilisez ces commandes :
- `buy` - Acheter manuellement
- `sell` - Vendre manuellement
- `status` - Afficher le statut
- `history` - Historique des trades
- `positions` - Positions ouvertes
- `quit` - Quitter

## 🔧 Personnalisation

### Ajuster les indicateurs
Modifiez les paramètres dans `config.env` :

```env
# RSI plus sensible
RSI_OVERBOUGHT=75
RSI_OVERSOLD=25

# MACD plus rapide
MACD_FAST=8
MACD_SLOW=21

# Bollinger plus large
BOLLINGER_STD=2.5
```

### Ajouter de nouveaux indicateurs
1. Modifiez `src/indicators.py`
2. Ajoutez le calcul dans `calculate_all_indicators()`
3. Créez une méthode de signal dans `get_*_signal()`
4. Intégrez dans `get_combined_signal()`

## ⚠️ Avertissements

### Risques financiers
- **Le trading de crypto est risqué** - Vous pouvez perdre de l'argent
- **Testez d'abord** - Utilisez le mode test avant le trading réel
- **Commencez petit** - Utilisez de petits montants au début
- **Surveillez** - Vérifiez régulièrement les performances

### Sécurité
- **Protégez vos clés API** - Ne les partagez jamais
- **Limitez les permissions** - Utilisez seulement les permissions nécessaires
- **Surveillez les logs** - Vérifiez les activités suspectes

## 🐛 Dépannage

### Erreurs courantes

**"Impossible de se connecter à Kraken"**
- Vérifiez vos clés API
- Assurez-vous que l'API est activée
- Vérifiez votre connexion internet

**"Solde insuffisant"**
- Vérifiez votre solde sur Kraken
- Réduisez `INVESTMENT_AMOUNT`
- Assurez-vous d'avoir la devise de base (EUR)

**"Erreur lors de l'analyse du marché"**
- Vérifiez la paire de trading
- Assurez-vous qu'elle est disponible sur Kraken
- Vérifiez les permissions API

### Logs de débogage
Activez le mode debug en modifiant `src/trading_bot.py` :
```python
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Ressources

### Documentation
- [API Kraken](https://www.kraken.com/features/api)
- [Indicateurs techniques](https://www.investopedia.com/technical-analysis-4689657)
- [Gestion des risques](https://www.investopedia.com/risk-management-4689657)

### Communauté
- [Reddit r/algotrading](https://www.reddit.com/r/algotrading/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/cryptocurrency)

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🤝 Contribution

Les contributions sont les bienvenues ! 

1. Fork le projet
2. Créez une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📞 Support

Si vous avez des questions ou des problèmes :

1. Vérifiez la section [Dépannage](#-dépannage)
2. Consultez les [Issues](https://github.com/username/TradingCryptoBot/issues)
3. Créez une nouvelle issue avec les détails du problème

---

**⚠️ Disclaimer : Ce bot est fourni à des fins éducatives. Le trading de cryptomonnaies comporte des risques. Utilisez à vos propres risques.** 