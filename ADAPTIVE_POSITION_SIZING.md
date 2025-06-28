# Adaptation Automatique de la Taille des Lots

## Vue d'ensemble

Le bot de trading a été amélioré pour adapter automatiquement la taille des lots en fonction des fonds réels disponibles sur votre compte Kraken. Cette fonctionnalité permet une gestion plus précise du capital et une meilleure gestion des risques.

## Comment ça fonctionne

### 1. Récupération Automatique du Solde

Le bot récupère automatiquement le solde de votre compte Kraken :
- **Fréquence** : Mise à jour toutes les 30 minutes ou à chaque demande
- **Devises supportées** : EUR, USD, BTC, ETH et autres cryptomonnaies configurées
- **Conversion automatique** : Toutes les devises sont converties en EUR pour les calculs

### 2. Capital Effectif

Le système calcule le **capital effectif** utilisé pour le trading :
```
Capital Effectif = min(Solde Réel du Compte, Capital Configuré)
```

Cela signifie que :
- Si votre compte a plus de fonds que configuré, seul le montant configuré sera utilisé
- Si votre compte a moins de fonds, seul le solde disponible sera utilisé
- Cela limite l'exposition même si vous avez plus de capital

### 3. Calcul Adaptatif des Positions

La taille des positions est calculée dynamiquement :

#### Position Fixe
```
Taille Position = (Capital Effectif × Allocation Paire) × Taille Position Fixe
```

#### Critère de Kelly
```
Taille Position = Capital Effectif × Fraction Kelly × Force du Signal
```

#### Martingale
```
Taille Position = Taille Base × (1.5 ^ Pertes Consécutives)
```

## Configuration

### Variables d'environnement importantes

```env
# Capital maximum à utiliser (même si le compte a plus)
INVESTMENT_AMOUNT=1000

# Méthode de sizing des positions
POSITION_SIZING_METHOD=kelly  # 'fixed', 'kelly', 'martingale'

# Risque maximum par trade (% du capital effectif)
MAX_RISK_PER_TRADE=2

# Fraction Kelly à utiliser (prudence recommandée)
KELLY_FRACTION=0.25
```

### Exemple de configuration optimale

```env
# Configuration pour un compte de 5000 EUR
INVESTMENT_AMOUNT=5000
POSITION_SIZING_METHOD=kelly
MAX_RISK_PER_TRADE=1.5
KELLY_FRACTION=0.2
MAX_POSITION_SIZE=0.1
STOP_LOSS_PERCENTAGE=5
TAKE_PROFIT_PERCENTAGE=10
```

## Avantages

### 1. **Sécurité Améliorée**
- Limite l'exposition au capital configuré
- Adaptation automatique si le solde diminue
- Gestion des risques en temps réel

### 2. **Précision du Capital**
- Utilise le solde réel du compte
- Évite le sur-trading
- Calculs basés sur les fonds disponibles

### 3. **Gestion Dynamique**
- Ajustement automatique après chaque trade
- Prise en compte des profits/pertes
- Réduction automatique de l'exposition en cas de drawdown

## Informations Affichées

Le bot affiche maintenant des informations détaillées sur le solde :

```
=== SOLDE DU COMPTE ===
Solde réel du compte: 5247.83 EUR
Capital effectif utilisé: 5000.00 EUR
Capital configuré: 5000.00 EUR
Utilisation du capital: 100.0%
Dernière mise à jour: 14:30:25
```

## Métriques de Performance Étendues

Les métriques incluent maintenant :
- `account_balance` : Solde réel du compte
- `available_balance` : Solde disponible pour trading
- `effective_capital` : Capital effectif utilisé
- `capital_utilization` : Pourcentage d'utilisation du capital

## Exemples d'Utilisation

### Scenario 1 : Compte avec 3000 EUR, Configuration 5000 EUR
```
Capital Effectif = min(3000, 5000) = 3000 EUR
→ Le bot utilisera 3000 EUR maximum
```

### Scenario 2 : Compte avec 8000 EUR, Configuration 5000 EUR
```
Capital Effectif = min(8000, 5000) = 5000 EUR
→ Le bot utilisera 5000 EUR maximum (sécurité)
```

### Scenario 3 : Après une perte de 500 EUR
```
Nouveau solde : 4500 EUR
Capital Effectif = min(4500, 5000) = 4500 EUR
→ Réduction automatique des positions futures
```

## Surveillance et Alertes

Le bot surveille automatiquement :
- **Solde insuffisant** : Alerte si le capital effectif devient trop bas
- **Drawdown excessif** : Réduction automatique de l'exposition
- **Limites de risque** : Vérification avant chaque trade

## Recommandations

### Pour débuter
```env
INVESTMENT_AMOUNT=1000
POSITION_SIZING_METHOD=fixed
MAX_POSITION_SIZE=0.05
MAX_RISK_PER_TRADE=1
```

### Pour traders expérimentés
```env
INVESTMENT_AMOUNT=10000
POSITION_SIZING_METHOD=kelly
KELLY_FRACTION=0.15
MAX_RISK_PER_TRADE=2
MAX_CORRELATED_RISK=5
```

## Support Multi-Devises

Le système convertit automatiquement :
- **EUR/USD** : Conversion approximative (0.85)
- **BTC/ETH** : Prix en temps réel via Kraken
- **Autres cryptos** : Extension possible

## Troubleshooting

### Problème : "Solde insuffisant"
- Vérifiez que votre compte Kraken a des fonds
- Vérifiez les clés API et permissions
- Réduisez `INVESTMENT_AMOUNT` si nécessaire

### Problème : "Capital effectif = 0"
- Problème de connexion à Kraken
- Clés API incorrectes
- Compte vide ou gelé

### Problème : Positions trop petites
- Augmentez `MAX_POSITION_SIZE`
- Vérifiez `INVESTMENT_AMOUNT`
- Augmentez `MAX_RISK_PER_TRADE`

Cette fonctionnalité rend le bot plus robuste et adaptatif aux conditions réelles de votre compte de trading.