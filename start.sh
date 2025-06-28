#!/bin/bash

# Script de démarrage pour le Bot de Trading Crypto Avancé
# ========================================================

echo "🚀 Démarrage du Bot de Trading Crypto Avancé..."

# Vérifier si Python est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 n'est pas installé. Veuillez l'installer."
    exit 1
fi

# Vérifier si Node.js est installé
if ! command -v node &> /dev/null; then
    echo "❌ Node.js n'est pas installé. Veuillez l'installer."
    exit 1
fi

# Vérifier si le fichier config.env existe
if [ ! -f "config.env" ]; then
    echo "❌ Fichier config.env non trouvé!"
    echo "📝 Création du fichier config.env depuis config.env.example..."
    cp config.env.example config.env
    echo "⚠️  Veuillez éditer config.env avec vos clés API Kraken"
    exit 1
fi

# Créer l'environnement virtuel Python si nécessaire
if [ ! -d "venv" ]; then
    echo "📦 Création de l'environnement virtuel Python..."
    python3 -m venv venv
fi

# Activer l'environnement virtuel
echo "🐍 Activation de l'environnement virtuel..."
source venv/bin/activate

# Installer les dépendances Python si nécessaire
echo "📦 Vérification des dépendances Python..."
pip install -q -r requirements.txt

# Installer les dépendances Node.js si nécessaire
if [ ! -d "web/node_modules" ]; then
    echo "📦 Installation des dépendances Node.js..."
    cd web
    npm install
    cd ..
fi

# Fonction pour arrêter proprement les processus
cleanup() {
    echo -e "\n⏹️  Arrêt des services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    exit 0
}

# Capturer Ctrl+C
trap cleanup INT

# Démarrer le backend Flask
echo "🔧 Démarrage du backend Flask..."
python web_app.py &
BACKEND_PID=$!

# Attendre que le backend soit prêt
sleep 3

# Démarrer le frontend React
echo "🎨 Démarrage du frontend React..."
cd web
npm start &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Bot de Trading Crypto Avancé démarré!"
echo "📊 Interface web: http://localhost:3000"
echo "🔌 API Backend: http://localhost:5000"
echo ""
echo "📝 Pour arrêter, appuyez sur Ctrl+C"

# Attendre indéfiniment
wait