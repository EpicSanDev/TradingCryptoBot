@echo off
REM Script de démarrage pour le Bot de Trading Crypto Avancé (Windows)
REM ==================================================================

echo 🚀 Demarrage du Bot de Trading Crypto Avance...

REM Vérifier si Python est installé
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python n'est pas installe. Veuillez l'installer.
    pause
    exit /b 1
)

REM Vérifier si Node.js est installé
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Node.js n'est pas installe. Veuillez l'installer.
    pause
    exit /b 1
)

REM Vérifier si le fichier config.env existe
if not exist "config.env" (
    echo ❌ Fichier config.env non trouve!
    echo 📝 Creation du fichier config.env depuis config.env.example...
    copy config.env.example config.env
    echo ⚠️  Veuillez editer config.env avec vos cles API Kraken
    pause
    exit /b 1
)

REM Créer l'environnement virtuel Python si nécessaire
if not exist "venv" (
    echo 📦 Creation de l'environnement virtuel Python...
    python -m venv venv
)

REM Activer l'environnement virtuel
echo 🐍 Activation de l'environnement virtuel...
call venv\Scripts\activate.bat

REM Installer les dépendances Python si nécessaire
echo 📦 Verification des dependances Python...
pip install -q -r requirements.txt

REM Installer les dépendances Node.js si nécessaire
if not exist "web\node_modules" (
    echo 📦 Installation des dependances Node.js...
    cd web
    npm install
    cd ..
)

REM Démarrer le backend Flask dans une nouvelle fenêtre
echo 🔧 Demarrage du backend Flask...
start "Backend Flask" cmd /k "venv\Scripts\activate && python web_app.py"

REM Attendre que le backend soit prêt
timeout /t 3 /nobreak >nul

REM Démarrer le frontend React dans une nouvelle fenêtre
echo 🎨 Demarrage du frontend React...
cd web
start "Frontend React" cmd /k "npm start"
cd ..

echo.
echo ✅ Bot de Trading Crypto Avance demarre!
echo 📊 Interface web: http://localhost:3000
echo 🔌 API Backend: http://localhost:5000
echo.
echo 📝 Pour arreter, fermez les fenetres Backend et Frontend
echo.
pause