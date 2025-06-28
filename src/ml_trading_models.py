"""
Module de Machine Learning pour Trading Crypto
=============================================

Ce module implémente des modèles ML réels pour la prédiction de signaux de trading,
avec collecte de données, entraînement, validation et mise à jour automatique.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
import logging
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier, StackingClassifier

try:
    from ta import add_all_ta_features
    from ta.utils import dropna
except ImportError:
    logging.warning("ta library not installed, using pandas_ta")
    import pandas_ta as ta


class MLTradingModels:
    """Gestionnaire de modèles ML pour le trading"""
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialiser le gestionnaire de modèles
        
        Args:
            model_dir: Répertoire pour sauvegarder les modèles
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.performance_history = {}
        
        # Configuration des features
        self.feature_config = {
            'price_features': True,
            'volume_features': True,
            'technical_features': True,
            'microstructure_features': True,
            'time_features': True,
            'rolling_features': True
        }
        
        # Fenêtres temporelles pour les features
        self.lookback_periods = [5, 10, 20, 50, 100]
        
        # Seuils pour la classification
        self.profit_threshold = 0.01  # 1% pour un signal d'achat
        self.loss_threshold = -0.01   # -1% pour un signal de vente
        
        logging.info("MLTradingModels initialisé")
    
    def collect_training_data(self, kraken_client, pairs: List[str], 
                            days: int = 365) -> pd.DataFrame:
        """
        Collecter les données historiques pour l'entraînement
        
        Args:
            kraken_client: Client Kraken pour récupérer les données
            pairs: Liste des paires à collecter
            days: Nombre de jours d'historique
            
        Returns:
            DataFrame avec toutes les données collectées
        """
        all_data = []
        
        for pair in pairs:
            logging.info(f"Collecte des données pour {pair}...")
            
            try:
                # Récupérer les données OHLC sur différentes timeframes
                for interval in [1, 5, 15, 60, 240]:  # 1m, 5m, 15m, 1h, 4h
                    ohlc = kraken_client.get_ohlc_data(pair, interval=interval, since=days*24*60)
                    
                    if ohlc is not None and not ohlc.empty:
                        # Ajouter les métadonnées
                        ohlc['pair'] = pair
                        ohlc['interval'] = interval
                        
                        # Calculer le label (signal de trading)
                        ohlc = self._calculate_labels(ohlc)
                        
                        # Générer les features
                        ohlc = self._generate_features(ohlc)
                        
                        all_data.append(ohlc)
                
                # Récupérer aussi les données de microstructure si disponibles
                order_book = kraken_client.get_order_book(pair, count=100)
                if order_book:
                    micro_features = self._extract_microstructure_features(order_book, pair)
                    # Ajouter aux données les plus récentes
                    if all_data and not all_data[-1].empty:
                        for col, val in micro_features.items():
                            all_data[-1].loc[all_data[-1].index[-1], col] = val
                
            except Exception as e:
                logging.error(f"Erreur lors de la collecte pour {pair}: {e}")
                continue
        
        if not all_data:
            raise ValueError("Aucune donnée collectée")
        
        # Combiner toutes les données
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Nettoyer les données
        combined_data = self._clean_data(combined_data)
        
        logging.info(f"Données collectées: {len(combined_data)} lignes, {len(combined_data.columns)} colonnes")
        
        return combined_data
    
    def _calculate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculer les labels pour l'apprentissage supervisé
        
        Labels:
        - 0: HOLD (pas de signal clair)
        - 1: BUY (le prix va monter)
        - 2: SELL (le prix va descendre)
        """
        df = df.copy()
        
        # Calculer les rendements futurs sur différents horizons
        for horizon in [5, 10, 20]:  # 5, 10, 20 périodes dans le futur
            df[f'future_return_{horizon}'] = df['close'].pct_change(horizon).shift(-horizon)
        
        # Label principal basé sur le rendement futur moyen
        df['avg_future_return'] = df[[f'future_return_{h}' for h in [5, 10, 20]]].mean(axis=1)
        
        # Classification
        df['label'] = 0  # HOLD par défaut
        df.loc[df['avg_future_return'] > self.profit_threshold, 'label'] = 1  # BUY
        df.loc[df['avg_future_return'] < self.loss_threshold, 'label'] = 2    # SELL
        
        return df
    
    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Générer toutes les features pour le ML"""
        df = df.copy()
        
        # Features de prix
        if self.feature_config['price_features']:
            df = self._add_price_features(df)
        
        # Features de volume
        if self.feature_config['volume_features']:
            df = self._add_volume_features(df)
        
        # Indicateurs techniques
        if self.feature_config['technical_features']:
            df = self._add_technical_indicators(df)
        
        # Features temporelles
        if self.feature_config['time_features']:
            df = self._add_time_features(df)
        
        # Features rolling (statistiques glissantes)
        if self.feature_config['rolling_features']:
            df = self._add_rolling_features(df)
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajouter les features basées sur les prix"""
        # Rendements
        for period in [1, 5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
        
        # Ratios de prix
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Position dans la range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Distance aux extremes
        for period in self.lookback_periods:
            df[f'dist_to_high_{period}'] = (df['high'].rolling(period).max() - df['close']) / df['close']
            df[f'dist_to_low_{period}'] = (df['close'] - df['low'].rolling(period).min()) / df['close']
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajouter les features basées sur le volume"""
        # Volume relatif
        for period in [5, 10, 20]:
            df[f'volume_ratio_{period}'] = df['volume'] / df['volume'].rolling(period).mean()
        
        # VWAP
        df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['price_to_vwap'] = df['close'] / df['vwap']
        
        # On-Balance Volume
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        
        # Volume par dollar
        df['dollar_volume'] = df['close'] * df['volume']
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajouter les indicateurs techniques"""
        try:
            # Si ta library est disponible
            df = add_all_ta_features(
                df, open="open", high="high", low="low", close="close", volume="volume"
            )
        except:
            # Sinon utiliser pandas_ta ou calculs manuels
            # RSI
            for period in [14, 21]:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_diff'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            for period in [20, 50]:
                rolling_mean = df['close'].rolling(period).mean()
                rolling_std = df['close'].rolling(period).std()
                df[f'bb_upper_{period}'] = rolling_mean + (rolling_std * 2)
                df[f'bb_lower_{period}'] = rolling_mean - (rolling_std * 2)
                df[f'bb_width_{period}'] = df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']
                df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / df[f'bb_width_{period}']
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            df['atr_14'] = true_range.rolling(14).mean()
            
            # Stochastic
            for period in [14, 21]:
                low_min = df['low'].rolling(window=period).min()
                high_max = df['high'].rolling(window=period).max()
                df[f'stoch_k_{period}'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
                df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(window=3).mean()
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajouter les features temporelles"""
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['day_of_month'] = pd.to_datetime(df['timestamp']).dt.day
            df['month'] = pd.to_datetime(df['timestamp']).dt.month
            
            # Features cycliques
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajouter les features statistiques glissantes"""
        for period in [10, 20, 50]:
            # Statistiques de prix
            df[f'mean_price_{period}'] = df['close'].rolling(period).mean()
            df[f'std_price_{period}'] = df['close'].rolling(period).std()
            df[f'skew_price_{period}'] = df['close'].rolling(period).skew()
            df[f'kurt_price_{period}'] = df['close'].rolling(period).kurt()
            
            # Statistiques de volume
            df[f'mean_volume_{period}'] = df['volume'].rolling(period).mean()
            df[f'std_volume_{period}'] = df['volume'].rolling(period).std()
            
            # Corrélation prix-volume
            df[f'corr_price_volume_{period}'] = df['close'].rolling(period).corr(df['volume'])
        
        return df
    
    def _extract_microstructure_features(self, order_book: Dict, pair: str) -> Dict:
        """Extraire les features de microstructure du marché"""
        features = {}
        
        try:
            # Profondeur du carnet
            bid_depth = sum(float(order[1]) for order in order_book['bids'][:20])
            ask_depth = sum(float(order[1]) for order in order_book['asks'][:20])
            
            features['bid_depth'] = bid_depth
            features['ask_depth'] = ask_depth
            features['depth_imbalance'] = (bid_depth - ask_depth) / (bid_depth + ask_depth)
            
            # Spread
            best_bid = float(order_book['bids'][0][0])
            best_ask = float(order_book['asks'][0][0])
            mid_price = (best_bid + best_ask) / 2
            
            features['spread'] = best_ask - best_bid
            features['spread_pct'] = features['spread'] / mid_price
            
            # Distribution des ordres
            bid_prices = [float(order[0]) for order in order_book['bids'][:50]]
            ask_prices = [float(order[0]) for order in order_book['asks'][:50]]
            
            features['bid_price_std'] = np.std(bid_prices)
            features['ask_price_std'] = np.std(ask_prices)
            
            # Concentration
            bid_volumes = [float(order[1]) for order in order_book['bids'][:50]]
            ask_volumes = [float(order[1]) for order in order_book['asks'][:50]]
            
            features['bid_concentration'] = max(bid_volumes) / sum(bid_volumes) if bid_volumes else 0
            features['ask_concentration'] = max(ask_volumes) / sum(ask_volumes) if ask_volumes else 0
            
        except Exception as e:
            logging.error(f"Erreur extraction microstructure: {e}")
        
        return features
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoyer les données"""
        # Supprimer les colonnes avec trop de NaN
        threshold = 0.5
        df = df.dropna(thresh=len(df) * threshold, axis=1)
        
        # Supprimer les lignes avec NaN dans les colonnes critiques
        critical_cols = ['close', 'volume', 'label']
        df = df.dropna(subset=[col for col in critical_cols if col in df.columns])
        
        # Remplir les NaN restants
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Supprimer les outliers extrêmes
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in ['label', 'pair', 'interval']:
                q1 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                df = df[(df[col] >= q1) & (df[col] <= q99)]
        
        return df
    
    def train_models(self, data: pd.DataFrame, models_to_train: List[str] = None):
        """
        Entraîner plusieurs modèles sur les données
        
        Args:
            data: DataFrame avec features et labels
            models_to_train: Liste des modèles à entraîner
        """
        if models_to_train is None:
            models_to_train = ['rf', 'xgb', 'lgb', 'mlp', 'ensemble']
        
        # Préparer les données
        X, y = self._prepare_training_data(data)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scaler
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Sélection de features
        selector = SelectKBest(f_classif, k=min(50, X.shape[1]))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        # Sauvegarder le scaler et selector
        self.scalers['main'] = scaler
        self.feature_selectors['main'] = selector
        
        # Entraîner chaque modèle
        trained_models = {}
        
        if 'rf' in models_to_train:
            logging.info("Entraînement Random Forest...")
            rf_model = self._train_random_forest(X_train_selected, y_train, X_test_selected, y_test)
            trained_models['rf'] = rf_model
        
        if 'xgb' in models_to_train:
            logging.info("Entraînement XGBoost...")
            xgb_model = self._train_xgboost(X_train_selected, y_train, X_test_selected, y_test)
            trained_models['xgb'] = xgb_model
        
        if 'lgb' in models_to_train:
            logging.info("Entraînement LightGBM...")
            lgb_model = self._train_lightgbm(X_train_selected, y_train, X_test_selected, y_test)
            trained_models['lgb'] = lgb_model
        
        if 'mlp' in models_to_train:
            logging.info("Entraînement Neural Network...")
            mlp_model = self._train_neural_network(X_train_selected, y_train, X_test_selected, y_test)
            trained_models['mlp'] = mlp_model
        
        if 'ensemble' in models_to_train and len(trained_models) > 1:
            logging.info("Création du modèle ensemble...")
            ensemble_model = self._create_ensemble_model(trained_models, X_train_selected, y_train, X_test_selected, y_test)
            trained_models['ensemble'] = ensemble_model
        
        # Sauvegarder les modèles
        for name, model in trained_models.items():
            self.models[name] = model
            self._save_model(model, name)
        
        # Évaluer et logger les performances
        self._evaluate_models(X_test_selected, y_test)
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Préparer les données pour l'entraînement"""
        # Colonnes à exclure
        exclude_cols = ['label', 'pair', 'interval', 'timestamp', 'avg_future_return'] + \
                      [col for col in data.columns if 'future_return' in col]
        
        # Features
        feature_cols = [col for col in data.columns if col not in exclude_cols and data[col].dtype in ['float64', 'int64']]
        X = data[feature_cols]
        
        # Labels
        y = data['label']
        
        logging.info(f"Features préparées: {X.shape}")
        logging.info(f"Distribution des labels: {y.value_counts().to_dict()}")
        
        return X, y
    
    def _train_random_forest(self, X_train, y_train, X_test, y_test):
        """Entraîner un Random Forest optimisé"""
        # Hyperparamètres à tester
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Grid search avec validation croisée
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        
        # Évaluation
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logging.info(f"Random Forest - Accuracy: {accuracy:.4f}")
        logging.info(f"Best params: {grid_search.best_params_}")
        
        return best_model
    
    def _train_xgboost(self, X_train, y_train, X_test, y_test):
        """Entraîner un modèle XGBoost"""
        # Paramètres
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'mlogloss',
            'seed': 42
        }
        
        # Conversion en DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Entraînement avec early stopping
        evallist = [(dtrain, 'train'), (dtest, 'eval')]
        
        model = xgb.train(
            params, dtrain, 
            num_boost_round=1000,
            evals=evallist,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Évaluation
        y_pred = model.predict(dtest).argmax(axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        
        logging.info(f"XGBoost - Accuracy: {accuracy:.4f}")
        
        return model
    
    def _train_lightgbm(self, X_train, y_train, X_test, y_test):
        """Entraîner un modèle LightGBM"""
        # Paramètres
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }
        
        # Datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Entraînement
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Évaluation
        y_pred = model.predict(X_test, num_iteration=model.best_iteration).argmax(axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        
        logging.info(f"LightGBM - Accuracy: {accuracy:.4f}")
        
        return model
    
    def _train_neural_network(self, X_train, y_train, X_test, y_test):
        """Entraîner un réseau de neurones"""
        # Architecture du réseau
        hidden_layers = (100, 50, 25)
        
        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        
        # Entraînement
        mlp.fit(X_train, y_train)
        
        # Évaluation
        y_pred = mlp.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logging.info(f"Neural Network - Accuracy: {accuracy:.4f}")
        
        return mlp
    
    def _create_ensemble_model(self, base_models: Dict, X_train, y_train, X_test, y_test):
        """Créer un modèle ensemble à partir des modèles de base"""
        # Préparer les estimateurs
        estimators = []
        
        for name, model in base_models.items():
            if name != 'xgb' and name != 'lgb':  # Ces modèles nécessitent un wrapper
                estimators.append((name, model))
        
        # Voting Classifier (vote majoritaire)
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Utilise les probabilités
        )
        
        # Entraînement
        voting_clf.fit(X_train, y_train)
        
        # Évaluation
        y_pred = voting_clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logging.info(f"Ensemble Model - Accuracy: {accuracy:.4f}")
        
        return voting_clf
    
    def _evaluate_models(self, X_test, y_test):
        """Évaluer tous les modèles entraînés"""
        results = {}
        
        for name, model in self.models.items():
            if name == 'xgb':
                dtest = xgb.DMatrix(X_test)
                y_pred = model.predict(dtest).argmax(axis=1)
            elif name == 'lgb':
                y_pred = model.predict(X_test, num_iteration=model.best_iteration).argmax(axis=1)
            else:
                y_pred = model.predict(X_test)
            
            # Métriques
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            logging.info(f"\n{name} Performance:")
            logging.info(f"  Accuracy: {accuracy:.4f}")
            logging.info(f"  Precision: {precision:.4f}")
            logging.info(f"  Recall: {recall:.4f}")
            logging.info(f"  F1-Score: {f1:.4f}")
        
        self.performance_history[datetime.now()] = results
        
        return results
    
    def predict(self, features: Union[pd.DataFrame, np.ndarray], 
               model_name: str = 'ensemble') -> Dict:
        """
        Faire une prédiction avec un modèle spécifique
        
        Args:
            features: Features pour la prédiction
            model_name: Nom du modèle à utiliser
            
        Returns:
            Dictionnaire avec prédiction et probabilités
        """
        if model_name not in self.models:
            # Essayer de charger le modèle
            self._load_model(model_name)
            
            if model_name not in self.models:
                raise ValueError(f"Modèle {model_name} non trouvé")
        
        model = self.models[model_name]
        
        # Préparer les features
        if isinstance(features, pd.DataFrame):
            features = features.values
        
        # Reshape si nécessaire
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scaler et sélection
        if 'main' in self.scalers:
            features = self.scalers['main'].transform(features)
        
        if 'main' in self.feature_selectors:
            features = self.feature_selectors['main'].transform(features)
        
        # Prédiction
        if model_name == 'xgb':
            dmatrix = xgb.DMatrix(features)
            probabilities = model.predict(dmatrix)
            prediction = probabilities.argmax(axis=1)[0]
        elif model_name == 'lgb':
            probabilities = model.predict(features, num_iteration=model.best_iteration)
            prediction = probabilities.argmax(axis=1)[0]
        else:
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
        
        # Mapper les prédictions
        label_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        
        result = {
            'prediction': label_map[prediction],
            'confidence': float(max(probabilities)),
            'probabilities': {
                'HOLD': float(probabilities[0]),
                'BUY': float(probabilities[1]),
                'SELL': float(probabilities[2])
            }
        }
        
        return result
    
    def update_models(self, new_data: pd.DataFrame, retrain_full: bool = False):
        """
        Mettre à jour les modèles avec de nouvelles données
        
        Args:
            new_data: Nouvelles données pour la mise à jour
            retrain_full: Si True, réentraîne complètement les modèles
        """
        if retrain_full:
            # Réentraînement complet
            self.train_models(new_data)
        else:
            # Mise à jour incrémentale (pour certains modèles)
            X, y = self._prepare_training_data(new_data)
            
            # Scaler et sélection
            if 'main' in self.scalers:
                X = self.scalers['main'].transform(X)
            
            if 'main' in self.feature_selectors:
                X = self.feature_selectors['main'].transform(X)
            
            # Mise à jour des modèles qui le supportent
            for name, model in self.models.items():
                if hasattr(model, 'partial_fit'):
                    model.partial_fit(X, y)
                    logging.info(f"Modèle {name} mis à jour de manière incrémentale")
    
    def _save_model(self, model, name: str):
        """Sauvegarder un modèle"""
        filepath = os.path.join(self.model_dir, f"{name}_model.pkl")
        joblib.dump(model, filepath)
        
        # Sauvegarder aussi les preprocessors
        if 'main' in self.scalers:
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            joblib.dump(self.scalers['main'], scaler_path)
        
        if 'main' in self.feature_selectors:
            selector_path = os.path.join(self.model_dir, "feature_selector.pkl")
            joblib.dump(self.feature_selectors['main'], selector_path)
        
        logging.info(f"Modèle {name} sauvegardé dans {filepath}")
    
    def _load_model(self, name: str):
        """Charger un modèle"""
        filepath = os.path.join(self.model_dir, f"{name}_model.pkl")
        
        if os.path.exists(filepath):
            self.models[name] = joblib.load(filepath)
            logging.info(f"Modèle {name} chargé depuis {filepath}")
            
            # Charger aussi les preprocessors
            scaler_path = os.path.join(self.model_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                self.scalers['main'] = joblib.load(scaler_path)
            
            selector_path = os.path.join(self.model_dir, "feature_selector.pkl")
            if os.path.exists(selector_path):
                self.feature_selectors['main'] = joblib.load(selector_path)
    
    def get_feature_importance(self, model_name: str = 'rf') -> pd.DataFrame:
        """Obtenir l'importance des features pour un modèle"""
        if model_name not in self.models:
            raise ValueError(f"Modèle {model_name} non trouvé")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif model_name == 'xgb':
            importances = model.get_score(importance_type='gain')
            importances = list(importances.values())
        elif model_name == 'lgb':
            importances = model.feature_importance(importance_type='gain')
        else:
            return pd.DataFrame()
        
        # Créer un DataFrame
        feature_names = [f"feature_{i}" for i in range(len(importances))]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df