#!/usr/bin/env python3
"""
Script pour lancer le dashboard web du bot de trading
"""

import sys
import os

# Ajouter le répertoire src au path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.dashboard import run_dashboard
    print("🚀 Lancement du dashboard web...")
    run_dashboard(host='0.0.0.0', port=8050, debug=False)
except ImportError as e:
    print(f"❌ Erreur d'importation: {e}")
    print("📦 Installez les dépendances avec: pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Erreur lors du lancement du dashboard: {e}")
    sys.exit(1) 