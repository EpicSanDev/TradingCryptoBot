# Adaptation Automatique du Capital

## Vue d'ensemble

Le bot de trading a été amélioré avec une fonctionnalité d'**adaptation automatique de la taille des positions** en fonction des fonds réellement disponibles sur votre compte Kraken. Cette fonctionnalité permet au bot de s'ajuster dynamiquement plutôt que d'utiliser une valeur statique configurée.

## Fonctionnalités Principales

### 🔄 Synchronisation Automatique du Solde
- Le bot récupère automatiquement le solde de votre compte Kraken toutes les 5 minutes
- Calcule le capital total en convertissant toutes les devises en EUR (configurable)
- Applique une marge de sécurité de 5% pour les frais de trading

### 💰 Calcul Dynamique des Positions
- **Méthode Fixed** : Taille basée sur le capital disponible réel × allocation × pourcentage fixe
- **Méthode Kelly** : Optimisation mathématique basée sur le capital dynamique
- **Méthode Martingale** : Augmentation progressive basée sur le capital réel

### 🌍 Support Multi-Devises
- Conversion automatique de toutes les devises du compte vers une devise principale (EUR par défaut)
- Support des principales cryptomonnaies (BTC, ETH, ADA, DOT, LINK)
- Taux de change en temps réel via l'API Kraken
- Taux de fallback en cas d'indisponibilité des données de marché

## Configuration

### Variables d'Environnement
Aucune nouvelle variable n'est requise. Le système utilise automatiquement vos clés API existantes pour récupérer le solde.

### Devise Principale
Par défaut, tous les calculs sont effectués en EUR. Vous pouvez changer cela via :
```python
bot.money_manager.set_capital_currency('USD')  # ou 'XBT', etc.
```

## Utilisation

### Mode Automatique
Lorsque vous lancez le bot normalement, l'adaptation automatique est activée par défaut :
```bash
python advanced_main.py
```

### Mode Manuel Interactif
De nouvelles commandes sont disponibles :

#### `capital` - Afficher les informations du capital
```
> capital
=== INFORMATIONS SUR LE CAPITAL ===
Capital statique configuré: 1000.00 EUR
Capital disponible (dynamique): 1250.00 EUR
Solde actuel du bot: 1250.00 EUR
Pic historique: 1250.00 EUR
Adaptation automatique: ✅ Activée
Dernière mise à jour: il y a 2.3 minutes
Différence capital dynamique vs statique: +25.0%
Méthode de sizing: Méthode KELLY: Optimisation mathématique (fraction: 0.25)
```

#### `update_capital` - Forcer la mise à jour
```
> update_capital
Mise à jour forcée du capital en cours...
✅ Capital mis à jour avec succès
Nouveau capital disponible: 1275.50 EUR
```

#### `recommendations` - Voir les recommandations de position
```
> recommendations
=== RECOMMANDATIONS DE TAILLE DE POSITION ===
Capital total disponible: 1275.50 EUR

XXBTZEUR:
  Allocation: 50.0% (637.75 EUR)
  Taille recommandée: 127.55 EUR
  Taille maximale: 255.10 EUR

XETHZEUR:
  Allocation: 50.0% (637.75 EUR)
  Taille recommandée: 127.55 EUR
  Taille maximale: 255.10 EUR
```

## Logs et Surveillance

### Logs Automatiques
Le bot affiche automatiquement le statut du capital dans le résumé global :
```
=== STATUT DU CAPITAL ===
Capital statique configuré: 1000.00 EUR
Capital disponible (dynamique): 1275.50 EUR
Solde actuel: 1275.50 EUR
Pic historique: 1275.50 EUR
Adaptation automatique: Activée
Dernière mise à jour: il y a 3.2 minutes
⚠️  Capital dynamique supérieur au capital statique: +27.6%
```

### Notifications de Changements
Le système notifie automatiquement les changements significatifs (>10%) :
```
INFO: Changement significatif du capital: 1200.00 → 1350.00 (+12.5%)
```

## Avantages

### ✅ Gestion Optimale du Risque
- Positions automatiquement ajustées selon les fonds réels
- Évite le sur-trading quand les fonds sont insuffisants
- Maximise l'utilisation du capital quand plus de fonds sont disponibles

### ✅ Transparence Totale
- Logs détaillés de tous les calculs
- Traçabilité des conversions de devises
- Comparaison constante entre capital statique et dynamique

### ✅ Robustesse
- Fallback vers le capital statique en cas de problème API
- Gestion d'erreur gracieuse
- Marge de sécurité intégrée pour les frais

## Sécurité

### 🔒 Marge de Sécurité
- 5% du capital total réservé pour les frais
- Jamais 100% du capital utilisé pour le trading

### 🔒 Validation Continue
- Vérification des limites de risque à chaque trade
- Respect des pourcentages de risque configurés (MAX_RISK_PER_TRADE)
- Protection contre les positions corrélées excessives

### 🔒 Mode Dégradé
Si l'API Kraken n'est pas disponible, le bot continue de fonctionner avec les valeurs statiques configurées.

## Exemples Concrets

### Scenario 1 : Dépôt de Fonds
```
Capital initial configuré: 1000 EUR
Dépôt: +500 EUR sur Kraken
→ Le bot détecte automatiquement 1500 EUR disponibles
→ Augmente proportionnellement la taille des positions
→ Log: "Capital dynamique supérieur au capital statique: +50.0%"
```

### Scenario 2 : Après des Gains
```
Capital configuré: 1000 EUR
Gains de trading: +200 EUR
→ Capital disponible: 1200 EUR
→ Positions futures plus importantes grâce aux gains
→ Croissance composée automatique
```

### Scenario 3 : Diversification Multi-Devises
```
Compte Kraken:
- 0.5 BTC (~22500 EUR)
- 2.0 ETH (~5000 EUR) 
- 1000 USD (~920 EUR)
- 500 ADA (~175 EUR)
→ Capital total calculé: ~28595 EUR
→ Positions dimensionnées sur cette base réelle
```

## Dépannage

### Capital non mis à jour
1. Vérifiez vos clés API Kraken
2. Vérifiez la connectivité internet
3. Utilisez `update_capital` pour forcer la mise à jour

### Conversions de devises incorrectes
1. Le bot utilise les taux Kraken en temps réel
2. En cas d'échec, des taux de fallback sont appliqués
3. Vérifiez les logs pour voir quelles conversions sont utilisées

### Différence majeure capital statique/dynamique
C'est normal et souhaitable ! Cela signifie que le bot s'adapte correctement à votre situation réelle.

## Configuration Avancée

### Changer la Devise Principale
```python
# En mode interactif ou dans le code
bot.money_manager.set_capital_currency('USD')  # USD au lieu d'EUR
bot.money_manager.set_capital_currency('XBT')  # Bitcoin comme référence
```

### Ajuster le Seuil de Notification
```python
# Notifier pour des changements >5% au lieu de 10%
bot.money_manager.capital_change_threshold = 0.05
```

### Désactiver l'Adaptation Automatique
Si vous voulez revenir au mode statique :
```python
# Créer le bot sans client pour le money manager
bot.money_manager = MoneyManager()  # Sans client Kraken
```

## Conclusion

L'adaptation automatique du capital transforme votre bot de trading en un système vraiment intelligent qui :
- S'adapte à votre situation financière réelle
- Optimise l'utilisation de vos fonds
- Maintient la sécurité et la gestion des risques
- Fournit une transparence totale sur ses calculs

Cette fonctionnalité est particulièrement utile pour :
- Les traders qui font des dépôts/retraits réguliers
- Ceux qui tradent avec plusieurs devises
- Les utilisateurs qui veulent une croissance composée automatique
- Tous ceux qui veulent une gestion optimale de leur capital