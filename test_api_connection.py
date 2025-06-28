#!/usr/bin/env python3
"""
Script de test pour diagnostiquer les problèmes de connexion API Kraken
"""

import os
import sys
from dotenv import load_dotenv
import krakenex
import pykrakenapi as kraken

def test_api_connection():
    """Tester la connexion à l'API Kraken"""
    
    # Charger les variables d'environnement
    load_dotenv('config.env')
    
    # Récupérer les clés
    spot_api_key = os.getenv('SPOT_API_KEY')
    spot_secret_key = os.getenv('SPOT_SECRET_KEY')
    
    print("=== DIAGNOSTIC API KRAKEN ===")
    print(f"Clé API trouvée: {'Oui' if spot_api_key else 'Non'}")
    print(f"Clé secrète trouvée: {'Oui' if spot_secret_key else 'Non'}")
    
    if spot_api_key:
        print(f"Longueur clé API: {len(spot_api_key)} caractères")
        print(f"Préfixe clé API: {spot_api_key[:10]}...")
    
    if spot_secret_key:
        print(f"Longueur clé secrète: {len(spot_secret_key)} caractères")
        print(f"Préfixe clé secrète: {spot_secret_key[:10]}...")
    
    print("\n=== TEST DE CONNEXION ===")
    
    try:
        # Test 1: Connexion sans authentification (API publique)
        print("1. Test API publique...")
        api_public = krakenex.API()
        time_info = api_public.query_public('Time')
        if time_info and time_info.get('error') == []:
            print("✅ API publique fonctionne")
            print(f"   Heure serveur: {time_info['result']['unixtime']}")
        else:
            print("❌ API publique échoue")
            print(f"   Erreur: {time_info}")
        
        # Test 2: Connexion avec authentification
        print("\n2. Test API privée...")
        if not spot_api_key or not spot_secret_key:
            print("❌ Clés API manquantes")
            return False
        
        # Nettoyer les clés (enlever les espaces et caractères invisibles)
        clean_api_key = spot_api_key.strip()
        clean_secret_key = spot_secret_key.strip()
        
        print(f"Clé API nettoyée: {clean_api_key[:10]}...")
        print(f"Clé secrète nettoyée: {clean_secret_key[:10]}...")
        
        api_private = krakenex.API(
            key=clean_api_key,
            secret=clean_secret_key
        )
        
        # Test de l'API privée - Balance
        balance_result = api_private.query_private('Balance')
        
        if balance_result and balance_result.get('error') == []:
            print("✅ API privée fonctionne")
            print("   Solde récupéré avec succès")
            balance = balance_result['result']
            if balance:
                print("   Actifs trouvés:")
                for asset, amount in balance.items():
                    if float(amount) > 0:
                        print(f"     {asset}: {amount}")
            else:
                print("   Aucun solde trouvé")
        else:
            print("❌ API privée échoue")
            if balance_result:
                print(f"   Erreur: {balance_result.get('error', 'Erreur inconnue')}")
            else:
                print("   Aucune réponse de l'API")
        
        # Test 3: Test avec pykrakenapi
        print("\n3. Test avec pykrakenapi...")
        kraken_api = kraken.KrakenAPI(api_private)
        
        try:
            balance_df = kraken_api.get_account_balance()
            if balance_df is not None and not balance_df.empty:
                print("✅ pykrakenapi fonctionne")
                print(f"   Nombre d'actifs: {len(balance_df)}")
            else:
                print("⚠️  pykrakenapi retourne un DataFrame vide")
        except Exception as e:
            print(f"❌ Erreur pykrakenapi: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur générale: {e}")
        return False

def check_config_file():
    """Vérifier le fichier de configuration"""
    print("\n=== VÉRIFICATION DU FICHIER CONFIG ===")
    
    if not os.path.exists('config.env'):
        print("❌ Fichier config.env non trouvé")
        return False
    
    print("✅ Fichier config.env trouvé")
    
    # Lire le contenu du fichier
    with open('config.env', 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Vérifier les clés
    spot_api_found = any('SPOT_API_KEY=' in line for line in lines)
    spot_secret_found = any('SPOT_SECRET_KEY=' in line for line in lines)
    
    print(f"SPOT_API_KEY trouvée: {'Oui' if spot_api_found else 'Non'}")
    print(f"SPOT_SECRET_KEY trouvée: {'Oui' if spot_secret_found else 'Non'}")
    
    # Afficher les lignes avec les clés (masquées)
    for line in lines:
        if 'SPOT_API_KEY=' in line:
            parts = line.split('=', 1)
            if len(parts) == 2:
                key_value = parts[1].strip()
                if key_value:
                    print(f"SPOT_API_KEY: {key_value[:10]}...{key_value[-5:] if len(key_value) > 15 else ''}")
                else:
                    print("SPOT_API_KEY: (vide)")
        elif 'SPOT_SECRET_KEY=' in line:
            parts = line.split('=', 1)
            if len(parts) == 2:
                key_value = parts[1].strip()
                if key_value:
                    print(f"SPOT_SECRET_KEY: {key_value[:10]}...{key_value[-5:] if len(key_value) > 15 else ''}")
                else:
                    print("SPOT_SECRET_KEY: (vide)")
    
    return True

if __name__ == "__main__":
    print("🔍 Diagnostic des problèmes de connexion API Kraken")
    print("=" * 60)
    
    # Vérifier le fichier de configuration
    config_ok = check_config_file()
    
    if config_ok:
        # Tester la connexion API
        api_ok = test_api_connection()
        
        if api_ok:
            print("\n✅ Diagnostic terminé - API fonctionne")
        else:
            print("\n❌ Diagnostic terminé - Problèmes détectés")
            print("\n🔧 Solutions possibles:")
            print("1. Vérifiez que vos clés API sont correctes")
            print("2. Assurez-vous que les clés ont les bonnes permissions")
            print("3. Vérifiez que votre compte Kraken est actif")
            print("4. Essayez de régénérer vos clés API")
    else:
        print("\n❌ Problème avec le fichier de configuration")
        print("🔧 Solutions:")
        print("1. Vérifiez que config.env existe")
        print("2. Vérifiez le format des clés API")
        print("3. Assurez-vous qu'il n'y a pas d'espaces en trop") 