"""
Module de Stratégie de Trading Institutionnelle
==============================================

Cette stratégie implémente des techniques de trading institutionnelles avancées incluant:
- Analyse du carnet d'ordres et microstructure du marché
- Indicateurs de flux d'ordres et volume profile
- Machine Learning pour la prédiction et détection de régimes
- Gestion des risques quantitative avancée (VaR, CVaR, Kelly optimal)
- Exécution algorithmique optimisée
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from .config import Config
from .money_management import MoneyManager
try:
    from .indicators import TechnicalIndicators
except ImportError:
    from .indicators_pandas import TechnicalIndicatorsPandas as TechnicalIndicators


class MarketRegime:
    """Énumération des régimes de marché"""
    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"


class InstitutionalStrategy:
    """Stratégie de trading institutionnelle avancée"""
    
    def __init__(self, kraken_client):
        """
        Initialiser la stratégie institutionnelle
        
        Args:
            kraken_client: Instance du client Kraken
        """
        self.kraken_client = kraken_client
        self.money_manager = MoneyManager(kraken_client)
        
        # État du marché
        self.market_regime = MarketRegime.SIDEWAYS
        self.volatility_regime = MarketRegime.LOW_VOLATILITY
        self.market_microstructure = {}
        
        # Historique et cache
        self.order_book_history = []
        self.trade_flow_history = []
        self.volume_profile_cache = {}
        self.correlation_matrix = {}
        
        # Machine Learning
        self.ml_model = None
        self.anomaly_detector = None
        self.feature_scaler = StandardScaler()
        self._initialize_ml_models()
        
        # Paramètres institutionnels
        self.min_liquidity_ratio = 0.1  # Ratio minimum de liquidité pour trader
        self.max_market_impact = 0.002  # Impact maximum accepté (0.2%)
        self.institutional_confidence_threshold = 0.7
        
        logging.info("Stratégie institutionnelle initialisée")
    
    def _initialize_ml_models(self):
        """Initialiser les modèles de Machine Learning"""
        try:
            # Modèle de classification pour les signaux
            self.ml_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            
            # Détecteur d'anomalies pour identifier les conditions inhabituelles
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42
            )
            
            logging.info("Modèles ML initialisés")
            
        except Exception as e:
            logging.error(f"Erreur lors de l'initialisation des modèles ML: {e}")
    
    def analyze_market_advanced(self, pair: str) -> Optional[Dict]:
        """
        Analyser le marché avec des techniques institutionnelles avancées
        
        Args:
            pair: Paire de trading
            
        Returns:
            Analyse complète du marché
        """
        try:
            # Récupérer toutes les données nécessaires
            market_data = self._gather_market_data(pair)
            if not market_data:
                return None
            
            # Analyse de la microstructure du marché
            microstructure = self._analyze_market_microstructure(pair, market_data)
            
            # Analyse du flux d'ordres et volume
            order_flow = self._analyze_order_flow(pair, market_data)
            
            # Détection du régime de marché
            regime = self._detect_market_regime(market_data)
            
            # Analyse technique avancée
            technical_signals = self._advanced_technical_analysis(market_data)
            
            # Analyse du sentiment et corrélations
            sentiment = self._analyze_market_sentiment(pair, market_data)
            
            # Machine Learning predictions
            ml_signals = self._get_ml_predictions(market_data, technical_signals)
            
            # Calcul du score institutionnel composite
            institutional_score = self._calculate_institutional_score(
                microstructure, order_flow, technical_signals, ml_signals, sentiment
            )
            
            # Génération de la recommandation finale
            recommendation = self._generate_institutional_recommendation(
                institutional_score, regime, microstructure
            )
            
            return {
                'timestamp': datetime.now(),
                'pair': pair,
                'current_price': market_data['current_price'],
                'microstructure': microstructure,
                'order_flow': order_flow,
                'regime': regime,
                'technical_signals': technical_signals,
                'sentiment': sentiment,
                'ml_signals': ml_signals,
                'institutional_score': institutional_score,
                'recommendation': recommendation,
                'confidence': institutional_score['confidence']
            }
            
        except Exception as e:
            logging.error(f"Erreur dans l'analyse avancée du marché: {e}")
            return None
    
    def _gather_market_data(self, pair: str) -> Optional[Dict]:
        """Rassembler toutes les données de marché nécessaires"""
        try:
            # Données OHLC multi-timeframe
            ohlc_1m = self.kraken_client.get_ohlc_data(pair, interval=1)
            ohlc_5m = self.kraken_client.get_ohlc_data(pair, interval=5)
            ohlc_15m = self.kraken_client.get_ohlc_data(pair, interval=15)
            ohlc_1h = self.kraken_client.get_ohlc_data(pair, interval=60)
            ohlc_4h = self.kraken_client.get_ohlc_data(pair, interval=240)
            
            # Prix actuel et spread
            ticker = self.kraken_client.get_ticker(pair)
            if not ticker:
                return None
            
            current_price = float(ticker['c'][0])
            bid = float(ticker['b'][0])
            ask = float(ticker['a'][0])
            spread = ask - bid
            
            # Carnet d'ordres
            order_book = self.kraken_client.get_order_book(pair, count=100)
            
            # Trades récents
            recent_trades = self.kraken_client.get_recent_trades(pair)
            
            return {
                'ohlc_1m': ohlc_1m,
                'ohlc_5m': ohlc_5m,
                'ohlc_15m': ohlc_15m,
                'ohlc_1h': ohlc_1h,
                'ohlc_4h': ohlc_4h,
                'current_price': current_price,
                'bid': bid,
                'ask': ask,
                'spread': spread,
                'order_book': order_book,
                'recent_trades': recent_trades,
                'ticker': ticker
            }
            
        except Exception as e:
            logging.error(f"Erreur lors de la collecte des données de marché: {e}")
            return None
    
    def _analyze_market_microstructure(self, pair: str, market_data: Dict) -> Dict:
        """Analyser la microstructure du marché"""
        try:
            order_book = market_data['order_book']
            recent_trades = market_data['recent_trades']
            
            # Analyser la profondeur du carnet d'ordres
            bid_depth = self._calculate_order_book_depth(order_book['bids'])
            ask_depth = self._calculate_order_book_depth(order_book['asks'])
            
            # Calculer l'imbalance du carnet d'ordres
            order_book_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)
            
            # Analyser la distribution des ordres
            bid_levels = self._analyze_order_distribution(order_book['bids'])
            ask_levels = self._analyze_order_distribution(order_book['asks'])
            
            # Détecter les murs d'ordres (walls)
            bid_walls = self._detect_order_walls(order_book['bids'])
            ask_walls = self._detect_order_walls(order_book['asks'])
            
            # Calculer le spread effectif et l'impact de marché
            effective_spread = self._calculate_effective_spread(recent_trades)
            market_impact = self._estimate_market_impact(order_book, 10000)  # Pour 10k EUR
            
            # Analyser la liquidité
            liquidity_score = self._calculate_liquidity_score(order_book, recent_trades)
            
            return {
                'bid_depth': bid_depth,
                'ask_depth': ask_depth,
                'order_book_imbalance': order_book_imbalance,
                'bid_levels': bid_levels,
                'ask_levels': ask_levels,
                'bid_walls': bid_walls,
                'ask_walls': ask_walls,
                'effective_spread': effective_spread,
                'market_impact': market_impact,
                'liquidity_score': liquidity_score
            }
            
        except Exception as e:
            logging.error(f"Erreur dans l'analyse de microstructure: {e}")
            return {}
    
    def _analyze_order_flow(self, pair: str, market_data: Dict) -> Dict:
        """Analyser le flux d'ordres et le volume"""
        try:
            recent_trades = market_data['recent_trades']
            ohlc_1m = market_data['ohlc_1m']
            
            # Calculer le Volume Delta cumulatif (CVD)
            cvd = self._calculate_cumulative_volume_delta(recent_trades)
            
            # Analyser le volume profile
            volume_profile = self._calculate_volume_profile(ohlc_1m)
            
            # Identifier les niveaux de Point of Control (POC)
            poc_level = self._find_point_of_control(volume_profile)
            
            # Calculer le VWAP
            vwap = self._calculate_vwap(ohlc_1m)
            
            # Analyser l'agressivité des acheteurs/vendeurs
            trade_aggression = self._analyze_trade_aggression(recent_trades)
            
            # Détecter les large trades (institutional footprints)
            large_trades = self._detect_large_trades(recent_trades)
            
            return {
                'cvd': cvd,
                'volume_profile': volume_profile,
                'poc_level': poc_level,
                'vwap': vwap,
                'trade_aggression': trade_aggression,
                'large_trades': large_trades,
                'buy_pressure': trade_aggression['buy_ratio'],
                'sell_pressure': trade_aggression['sell_ratio']
            }
            
        except Exception as e:
            logging.error(f"Erreur dans l'analyse du flux d'ordres: {e}")
            return {}
    
    def _detect_market_regime(self, market_data: Dict) -> Dict:
        """Détecter le régime de marché actuel"""
        try:
            ohlc_1h = market_data['ohlc_1h']
            ohlc_4h = market_data['ohlc_4h']
            
            # Calculer les tendances sur différentes timeframes
            trend_1h = self._calculate_trend_strength(ohlc_1h)
            trend_4h = self._calculate_trend_strength(ohlc_4h)
            
            # Calculer la volatilité
            volatility_1h = self._calculate_volatility_regime(ohlc_1h)
            volatility_4h = self._calculate_volatility_regime(ohlc_4h)
            
            # Déterminer le régime principal
            if trend_4h['strength'] > 0.6 and trend_4h['direction'] > 0:
                main_regime = MarketRegime.BULL_TREND
            elif trend_4h['strength'] > 0.6 and trend_4h['direction'] < 0:
                main_regime = MarketRegime.BEAR_TREND
            else:
                main_regime = MarketRegime.SIDEWAYS
            
            # Déterminer le régime de volatilité
            if volatility_4h['current'] > volatility_4h['average'] * 1.5:
                vol_regime = MarketRegime.HIGH_VOLATILITY
            else:
                vol_regime = MarketRegime.LOW_VOLATILITY
            
            return {
                'main_regime': main_regime,
                'volatility_regime': vol_regime,
                'trend_1h': trend_1h,
                'trend_4h': trend_4h,
                'volatility_1h': volatility_1h,
                'volatility_4h': volatility_4h
            }
            
        except Exception as e:
            logging.error(f"Erreur dans la détection du régime de marché: {e}")
            return {'main_regime': MarketRegime.SIDEWAYS, 'volatility_regime': MarketRegime.LOW_VOLATILITY}
    
    def _advanced_technical_analysis(self, market_data: Dict) -> Dict:
        """Analyse technique avancée multi-timeframe"""
        try:
            signals = {}
            
            # Analyse sur chaque timeframe
            for tf_name, ohlc_data in [('1m', market_data['ohlc_1m']),
                                       ('5m', market_data['ohlc_5m']),
                                       ('15m', market_data['ohlc_15m']),
                                       ('1h', market_data['ohlc_1h']),
                                       ('4h', market_data['ohlc_4h'])]:
                
                if ohlc_data is None or ohlc_data.empty:
                    continue
                
                # Indicateurs standards
                indicators = TechnicalIndicators(ohlc_data)
                
                # Indicateurs avancés
                advanced = self._calculate_advanced_indicators(ohlc_data)
                
                # Structure de marché
                market_structure = self._analyze_market_structure(ohlc_data)
                
                signals[tf_name] = {
                    'standard': indicators.get_combined_signal(),
                    'advanced': advanced,
                    'structure': market_structure
                }
            
            # Confluence des signaux multi-timeframe
            confluence = self._calculate_signal_confluence(signals)
            
            return {
                'timeframe_signals': signals,
                'confluence': confluence,
                'strength': confluence['score']
            }
            
        except Exception as e:
            logging.error(f"Erreur dans l'analyse technique avancée: {e}")
            return {}
    
    def _analyze_market_sentiment(self, pair: str, market_data: Dict) -> Dict:
        """Analyser le sentiment du marché"""
        try:
            # Ratio long/short basé sur l'ordre book
            order_book = market_data['order_book']
            long_short_ratio = self._calculate_long_short_ratio(order_book)
            
            # Sentiment basé sur le volume et momentum
            ohlc_1h = market_data['ohlc_1h']
            volume_sentiment = self._analyze_volume_sentiment(ohlc_1h)
            
            # Fear & Greed approximatif
            fear_greed = self._calculate_fear_greed_index(market_data)
            
            return {
                'long_short_ratio': long_short_ratio,
                'volume_sentiment': volume_sentiment,
                'fear_greed': fear_greed,
                'overall_sentiment': self._aggregate_sentiment(long_short_ratio, volume_sentiment, fear_greed)
            }
            
        except Exception as e:
            logging.error(f"Erreur dans l'analyse du sentiment: {e}")
            return {}
    
    def _get_ml_predictions(self, market_data: Dict, technical_signals: Dict) -> Dict:
        """Obtenir les prédictions ML"""
        try:
            # Préparer les features
            features = self._prepare_ml_features(market_data, technical_signals)
            
            if features is None or len(features) == 0:
                return {'signal': 'NEUTRAL', 'confidence': 0.0}
            
            # Normaliser les features
            features_scaled = self.feature_scaler.fit_transform(features.reshape(1, -1))
            
            # Détecter les anomalies
            is_anomaly = self.anomaly_detector.fit_predict(features_scaled)[0] == -1
            
            # Si anomalie détectée, réduire la confiance
            if is_anomaly:
                return {'signal': 'NEUTRAL', 'confidence': 0.0, 'anomaly': True}
            
            # Pour la démo, simuler une prédiction
            # Dans la réalité, il faudrait entraîner le modèle sur des données historiques
            ml_signal = self._simulate_ml_prediction(features)
            
            return {
                'signal': ml_signal['signal'],
                'confidence': ml_signal['confidence'],
                'anomaly': False
            }
            
        except Exception as e:
            logging.error(f"Erreur dans les prédictions ML: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0.0}
    
    def _calculate_institutional_score(self, microstructure: Dict, order_flow: Dict,
                                     technical_signals: Dict, ml_signals: Dict,
                                     sentiment: Dict) -> Dict:
        """Calculer le score institutionnel composite"""
        try:
            scores = []
            weights = []
            
            # Score de microstructure (25%)
            micro_score = self._score_microstructure(microstructure)
            scores.append(micro_score)
            weights.append(0.25)
            
            # Score de flux d'ordres (25%)
            flow_score = self._score_order_flow(order_flow)
            scores.append(flow_score)
            weights.append(0.25)
            
            # Score technique (20%)
            tech_score = technical_signals.get('strength', 0.5)
            scores.append(tech_score)
            weights.append(0.20)
            
            # Score ML (20%)
            ml_score = ml_signals.get('confidence', 0.5)
            scores.append(ml_score)
            weights.append(0.20)
            
            # Score de sentiment (10%)
            sentiment_score = sentiment.get('overall_sentiment', 0.5)
            scores.append(sentiment_score)
            weights.append(0.10)
            
            # Calculer le score pondéré
            composite_score = np.average(scores, weights=weights)
            
            # Calculer la confiance basée sur la cohérence des signaux
            confidence = self._calculate_signal_confidence(scores)
            
            return {
                'composite_score': composite_score,
                'confidence': confidence,
                'components': {
                    'microstructure': micro_score,
                    'order_flow': flow_score,
                    'technical': tech_score,
                    'ml': ml_score,
                    'sentiment': sentiment_score
                }
            }
            
        except Exception as e:
            logging.error(f"Erreur dans le calcul du score institutionnel: {e}")
            return {'composite_score': 0.5, 'confidence': 0.0}
    
    def _generate_institutional_recommendation(self, institutional_score: Dict,
                                            regime: Dict, microstructure: Dict) -> Dict:
        """Générer une recommandation institutionnelle"""
        try:
            score = institutional_score['composite_score']
            confidence = institutional_score['confidence']
            
            # Vérifier les conditions de liquidité
            if microstructure.get('liquidity_score', 0) < self.min_liquidity_ratio:
                return {
                    'action': 'HOLD',
                    'reason': 'Liquidité insuffisante pour une exécution institutionnelle',
                    'confidence': 0.0,
                    'risk_level': 'HIGH'
                }
            
            # Ajuster selon le régime de marché
            regime_adjustment = self._get_regime_adjustment(regime)
            adjusted_score = score * regime_adjustment
            
            # Déterminer l'action
            if adjusted_score > 0.7 and confidence > self.institutional_confidence_threshold:
                action = 'BUY'
                reason = 'Signal institutionnel fort avec confluence multi-facteurs'
            elif adjusted_score < 0.3 and confidence > self.institutional_confidence_threshold:
                action = 'SELL'
                reason = 'Signal de vente institutionnel avec risque élevé'
            else:
                action = 'HOLD'
                reason = 'Conditions insuffisantes pour une position institutionnelle'
            
            # Calculer les niveaux de risque
            risk_levels = self._calculate_institutional_risk_levels(
                microstructure, regime, institutional_score
            )
            
            return {
                'action': action,
                'reason': reason,
                'confidence': confidence,
                'adjusted_score': adjusted_score,
                'risk_levels': risk_levels,
                'execution_strategy': self._recommend_execution_strategy(microstructure, action)
            }
            
        except Exception as e:
            logging.error(f"Erreur dans la génération de recommandation: {e}")
            return {'action': 'HOLD', 'reason': 'Erreur d'analyse', 'confidence': 0.0}
    
    # === Méthodes auxiliaires ===
    
    def _calculate_order_book_depth(self, orders: List) -> float:
        """Calculer la profondeur du carnet d'ordres"""
        total_volume = sum(float(order[1]) for order in orders[:20])  # Top 20 niveaux
        return total_volume
    
    def _analyze_order_distribution(self, orders: List) -> Dict:
        """Analyser la distribution des ordres"""
        if not orders:
            return {}
        
        prices = [float(order[0]) for order in orders[:50]]
        volumes = [float(order[1]) for order in orders[:50]]
        
        return {
            'mean_price': np.mean(prices),
            'std_price': np.std(prices),
            'total_volume': sum(volumes),
            'mean_volume': np.mean(volumes),
            'max_volume': max(volumes),
            'concentration': max(volumes) / sum(volumes) if sum(volumes) > 0 else 0
        }
    
    def _detect_order_walls(self, orders: List, threshold: float = 3.0) -> List[Dict]:
        """Détecter les murs d'ordres significatifs"""
        if not orders:
            return []
        
        volumes = [float(order[1]) for order in orders[:50]]
        mean_volume = np.mean(volumes)
        std_volume = np.std(volumes)
        
        walls = []
        for i, order in enumerate(orders[:50]):
            volume = float(order[1])
            if volume > mean_volume + (threshold * std_volume):
                walls.append({
                    'price': float(order[0]),
                    'volume': volume,
                    'significance': (volume - mean_volume) / std_volume
                })
        
        return walls
    
    def _calculate_effective_spread(self, recent_trades: pd.DataFrame) -> float:
        """Calculer le spread effectif basé sur les trades récents"""
        if recent_trades is None or recent_trades.empty:
            return 0.0
        
        # Calculer le midpoint pour chaque trade
        midpoints = []
        for _, trade in recent_trades.iterrows():
            # Estimation simplifiée
            midpoints.append(float(trade['price']))
        
        if len(midpoints) > 1:
            return np.std(midpoints) * 2  # 2 * écart-type comme proxy
        return 0.0
    
    def _estimate_market_impact(self, order_book: Dict, trade_size: float) -> float:
        """Estimer l'impact sur le marché d'un trade de taille donnée"""
        cumulative_volume = 0
        weighted_price = 0
        
        # Simuler l'exécution à travers le carnet d'ordres
        for order in order_book['asks']:
            price = float(order[0])
            volume = float(order[1])
            
            if cumulative_volume + volume >= trade_size:
                # Ordre partiellement rempli
                remaining = trade_size - cumulative_volume
                weighted_price += price * remaining
                cumulative_volume = trade_size
                break
            else:
                # Ordre complètement consommé
                weighted_price += price * volume
                cumulative_volume += volume
        
        if cumulative_volume > 0:
            avg_execution_price = weighted_price / cumulative_volume
            current_price = float(order_book['asks'][0][0])
            impact = (avg_execution_price - current_price) / current_price
            return impact
        
        return 0.0
    
    def _calculate_liquidity_score(self, order_book: Dict, recent_trades: pd.DataFrame) -> float:
        """Calculer un score de liquidité"""
        # Profondeur du carnet
        bid_depth = self._calculate_order_book_depth(order_book['bids'])
        ask_depth = self._calculate_order_book_depth(order_book['asks'])
        
        # Spread
        spread = float(order_book['asks'][0][0]) - float(order_book['bids'][0][0])
        mid_price = (float(order_book['asks'][0][0]) + float(order_book['bids'][0][0])) / 2
        spread_bps = (spread / mid_price) * 10000  # en basis points
        
        # Volume de trading récent
        if recent_trades is not None and not recent_trades.empty:
            recent_volume = recent_trades['volume'].sum()
        else:
            recent_volume = 0
        
        # Score composite (normalisé entre 0 et 1)
        depth_score = min((bid_depth + ask_depth) / 10000, 1.0)  # Normaliser sur 10k
        spread_score = max(0, 1 - (spread_bps / 100))  # 100 bps = score 0
        volume_score = min(recent_volume / 1000, 1.0)  # Normaliser sur 1k
        
        liquidity_score = (depth_score * 0.4 + spread_score * 0.3 + volume_score * 0.3)
        
        return liquidity_score
    
    def _calculate_cumulative_volume_delta(self, recent_trades: pd.DataFrame) -> float:
        """Calculer le Volume Delta Cumulatif"""
        if recent_trades is None or recent_trades.empty:
            return 0.0
        
        buy_volume = recent_trades[recent_trades['type'] == 'buy']['volume'].sum()
        sell_volume = recent_trades[recent_trades['type'] == 'sell']['volume'].sum()
        
        return buy_volume - sell_volume
    
    def _calculate_volume_profile(self, ohlc_data: pd.DataFrame) -> Dict:
        """Calculer le profil de volume"""
        if ohlc_data is None or ohlc_data.empty:
            return {}
        
        # Créer des bins de prix
        price_range = ohlc_data['high'].max() - ohlc_data['low'].min()
        n_bins = 50
        bin_size = price_range / n_bins
        
        volume_profile = {}
        
        for _, row in ohlc_data.iterrows():
            # Distribuer le volume uniformément dans la range de la bougie
            avg_price = (row['high'] + row['low']) / 2
            bin_index = int((avg_price - ohlc_data['low'].min()) / bin_size)
            
            if bin_index not in volume_profile:
                volume_profile[bin_index] = {
                    'price': ohlc_data['low'].min() + (bin_index * bin_size),
                    'volume': 0
                }
            
            volume_profile[bin_index]['volume'] += row['volume']
        
        return volume_profile
    
    def _find_point_of_control(self, volume_profile: Dict) -> float:
        """Trouver le Point of Control (niveau de prix avec le plus de volume)"""
        if not volume_profile:
            return 0.0
        
        max_volume = 0
        poc_price = 0
        
        for profile in volume_profile.values():
            if profile['volume'] > max_volume:
                max_volume = profile['volume']
                poc_price = profile['price']
        
        return poc_price
    
    def _calculate_vwap(self, ohlc_data: pd.DataFrame) -> float:
        """Calculer le VWAP"""
        if ohlc_data is None or ohlc_data.empty:
            return 0.0
        
        typical_price = (ohlc_data['high'] + ohlc_data['low'] + ohlc_data['close']) / 3
        vwap = (typical_price * ohlc_data['volume']).sum() / ohlc_data['volume'].sum()
        
        return vwap
    
    def _analyze_trade_aggression(self, recent_trades: pd.DataFrame) -> Dict:
        """Analyser l'agressivité des trades"""
        if recent_trades is None or recent_trades.empty:
            return {'buy_ratio': 0.5, 'sell_ratio': 0.5, 'aggression_score': 0.0}
        
        buy_count = len(recent_trades[recent_trades['type'] == 'buy'])
        sell_count = len(recent_trades[recent_trades['type'] == 'sell'])
        total_count = buy_count + sell_count
        
        if total_count == 0:
            return {'buy_ratio': 0.5, 'sell_ratio': 0.5, 'aggression_score': 0.0}
        
        buy_ratio = buy_count / total_count
        sell_ratio = sell_count / total_count
        
        # Score d'agressivité (-1 à 1)
        aggression_score = buy_ratio - sell_ratio
        
        return {
            'buy_ratio': buy_ratio,
            'sell_ratio': sell_ratio,
            'aggression_score': aggression_score
        }
    
    def _detect_large_trades(self, recent_trades: pd.DataFrame) -> List[Dict]:
        """Détecter les large trades (empreintes institutionnelles)"""
        if recent_trades is None or recent_trades.empty:
            return []
        
        volumes = recent_trades['volume'].values
        mean_volume = np.mean(volumes)
        std_volume = np.std(volumes)
        
        large_trades = []
        threshold = mean_volume + (2 * std_volume)  # 2 écarts-types
        
        for _, trade in recent_trades.iterrows():
            if trade['volume'] > threshold:
                large_trades.append({
                    'timestamp': trade['time'],
                    'price': trade['price'],
                    'volume': trade['volume'],
                    'type': trade['type'],
                    'significance': (trade['volume'] - mean_volume) / std_volume
                })
        
        return large_trades
    
    def _calculate_trend_strength(self, ohlc_data: pd.DataFrame) -> Dict:
        """Calculer la force de la tendance"""
        if ohlc_data is None or ohlc_data.empty:
            return {'strength': 0.0, 'direction': 0}
        
        # Utiliser l'ADX pour la force
        high = ohlc_data['high']
        low = ohlc_data['low']
        close = ohlc_data['close']
        
        # Calcul simplifié de l'ADX
        tr = np.maximum(high - low, np.abs(high - close.shift(1)), np.abs(low - close.shift(1)))
        atr = tr.rolling(14).mean()
        
        # Direction basée sur la pente de la SMA
        sma_20 = close.rolling(20).mean()
        if len(sma_20) > 20:
            direction = 1 if sma_20.iloc[-1] > sma_20.iloc[-20] else -1
        else:
            direction = 0
        
        # Force basée sur l'angle de la tendance
        if len(close) > 20:
            trend_angle = np.arctan((close.iloc[-1] - close.iloc[-20]) / 20)
            strength = abs(np.sin(trend_angle))
        else:
            strength = 0.0
        
        return {'strength': strength, 'direction': direction}
    
    def _calculate_volatility_regime(self, ohlc_data: pd.DataFrame) -> Dict:
        """Calculer le régime de volatilité"""
        if ohlc_data is None or ohlc_data.empty:
            return {'current': 0.0, 'average': 0.0, 'regime': 'LOW'}
        
        # Calculer la volatilité réalisée
        returns = np.log(ohlc_data['close'] / ohlc_data['close'].shift(1))
        current_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)  # Annualisée
        avg_vol = returns.rolling(100).std().mean() * np.sqrt(252)
        
        # Déterminer le régime
        if current_vol > avg_vol * 1.5:
            regime = 'HIGH'
        elif current_vol < avg_vol * 0.5:
            regime = 'LOW'
        else:
            regime = 'NORMAL'
        
        return {
            'current': current_vol,
            'average': avg_vol,
            'regime': regime
        }
    
    def _calculate_advanced_indicators(self, ohlc_data: pd.DataFrame) -> Dict:
        """Calculer des indicateurs techniques avancés"""
        if ohlc_data is None or ohlc_data.empty:
            return {}
        
        close = ohlc_data['close']
        high = ohlc_data['high']
        low = ohlc_data['low']
        volume = ohlc_data['volume']
        
        indicators = {}
        
        # Market Profile Value Area
        indicators['value_area_high'] = high.rolling(20).quantile(0.7)
        indicators['value_area_low'] = low.rolling(20).quantile(0.3)
        
        # Accumulation/Distribution
        clv = ((close - low) - (high - close)) / (high - low)
        indicators['accumulation_distribution'] = (clv * volume).cumsum()
        
        # On-Balance Volume
        indicators['obv'] = (np.sign(close.diff()) * volume).cumsum()
        
        # Chaikin Money Flow
        mfv = clv * volume
        indicators['cmf'] = mfv.rolling(20).sum() / volume.rolling(20).sum()
        
        # Williams %R
        indicators['williams_r'] = ((high.rolling(14).max() - close) / 
                                  (high.rolling(14).max() - low.rolling(14).min())) * -100
        
        return {k: v.iloc[-1] for k, v in indicators.items() if not v.empty}
    
    def _analyze_market_structure(self, ohlc_data: pd.DataFrame) -> Dict:
        """Analyser la structure du marché"""
        if ohlc_data is None or ohlc_data.empty:
            return {}
        
        high = ohlc_data['high']
        low = ohlc_data['low']
        
        # Identifier les swing highs et lows
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(high) - 2):
            if high.iloc[i] > high.iloc[i-1] and high.iloc[i] > high.iloc[i-2] and \
               high.iloc[i] > high.iloc[i+1] and high.iloc[i] > high.iloc[i+2]:
                swing_highs.append({'index': i, 'price': high.iloc[i]})
            
            if low.iloc[i] < low.iloc[i-1] and low.iloc[i] < low.iloc[i-2] and \
               low.iloc[i] < low.iloc[i+1] and low.iloc[i] < low.iloc[i+2]:
                swing_lows.append({'index': i, 'price': low.iloc[i]})
        
        # Déterminer la structure (HH, HL, LL, LH)
        structure = 'NEUTRAL'
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            if swing_highs[-1]['price'] > swing_highs[-2]['price'] and \
               swing_lows[-1]['price'] > swing_lows[-2]['price']:
                structure = 'BULLISH'  # Higher Highs, Higher Lows
            elif swing_highs[-1]['price'] < swing_highs[-2]['price'] and \
                 swing_lows[-1]['price'] < swing_lows[-2]['price']:
                structure = 'BEARISH'  # Lower Highs, Lower Lows
        
        return {
            'structure': structure,
            'swing_highs': swing_highs[-3:] if swing_highs else [],
            'swing_lows': swing_lows[-3:] if swing_lows else []
        }
    
    def _calculate_signal_confluence(self, signals: Dict) -> Dict:
        """Calculer la confluence des signaux multi-timeframe"""
        buy_count = 0
        sell_count = 0
        total_count = 0
        
        timeframe_weights = {
            '1m': 0.1,
            '5m': 0.15,
            '15m': 0.2,
            '1h': 0.3,
            '4h': 0.25
        }
        
        weighted_score = 0
        
        for tf, signal in signals.items():
            if 'standard' in signal:
                weight = timeframe_weights.get(tf, 0.2)
                
                if signal['standard'] == 'BUY':
                    buy_count += 1
                    weighted_score += weight
                elif signal['standard'] == 'SELL':
                    sell_count += 1
                    weighted_score -= weight
                
                total_count += 1
        
        if total_count > 0:
            confluence_ratio = max(buy_count, sell_count) / total_count
        else:
            confluence_ratio = 0
        
        return {
            'score': (weighted_score + 1) / 2,  # Normaliser entre 0 et 1
            'buy_signals': buy_count,
            'sell_signals': sell_count,
            'confluence_ratio': confluence_ratio
        }
    
    def _calculate_long_short_ratio(self, order_book: Dict) -> float:
        """Calculer le ratio long/short basé sur le carnet d'ordres"""
        bid_volume = sum(float(order[1]) for order in order_book['bids'][:50])
        ask_volume = sum(float(order[1]) for order in order_book['asks'][:50])
        
        if ask_volume > 0:
            return bid_volume / ask_volume
        return 1.0
    
    def _analyze_volume_sentiment(self, ohlc_data: pd.DataFrame) -> float:
        """Analyser le sentiment basé sur le volume"""
        if ohlc_data is None or ohlc_data.empty:
            return 0.5
        
        # Volume sur bougies vertes vs rouges
        green_volume = ohlc_data[ohlc_data['close'] > ohlc_data['open']]['volume'].sum()
        red_volume = ohlc_data[ohlc_data['close'] <= ohlc_data['open']]['volume'].sum()
        
        total_volume = green_volume + red_volume
        if total_volume > 0:
            return green_volume / total_volume
        return 0.5
    
    def _calculate_fear_greed_index(self, market_data: Dict) -> float:
        """Calculer un index Fear & Greed approximatif"""
        # Simplifié - dans la réalité, utiliser plus de facteurs
        ohlc_1h = market_data['ohlc_1h']
        if ohlc_1h is None or ohlc_1h.empty:
            return 50
        
        # Momentum
        momentum = (ohlc_1h['close'].iloc[-1] - ohlc_1h['close'].iloc[-24]) / ohlc_1h['close'].iloc[-24]
        
        # Volatilité (inverse)
        volatility = ohlc_1h['close'].pct_change().std()
        
        # Score combiné (0-100)
        fear_greed = 50 + (momentum * 100) - (volatility * 200)
        fear_greed = max(0, min(100, fear_greed))
        
        return fear_greed
    
    def _aggregate_sentiment(self, long_short: float, volume_sentiment: float, fear_greed: float) -> float:
        """Agréger les scores de sentiment"""
        # Normaliser fear_greed de 0-100 à 0-1
        fear_greed_normalized = fear_greed / 100
        
        # Moyenne pondérée
        sentiment = (long_short * 0.3 + volume_sentiment * 0.4 + fear_greed_normalized * 0.3)
        
        # Ramener entre 0 et 1
        return max(0, min(1, sentiment))
    
    def _prepare_ml_features(self, market_data: Dict, technical_signals: Dict) -> np.ndarray:
        """Préparer les features pour le ML"""
        features = []
        
        try:
            # Features de prix
            current_price = market_data['current_price']
            features.append(current_price)
            
            # Spread
            features.append(market_data['spread'])
            
            # Features techniques
            if 'confluence' in technical_signals:
                features.append(technical_signals['confluence']['score'])
            
            # Volume
            if market_data['recent_trades'] is not None:
                features.append(len(market_data['recent_trades']))
            
            # Ajouter d'autres features selon les besoins...
            
            return np.array(features)
            
        except Exception as e:
            logging.error(f"Erreur dans la préparation des features ML: {e}")
            return np.array([])
    
    def _simulate_ml_prediction(self, features: np.ndarray) -> Dict:
        """Simuler une prédiction ML (remplacer par un vrai modèle en production)"""
        # Pour la démo, utiliser une logique simple
        if len(features) > 0:
            # Simulation basée sur les features
            score = np.mean(features) % 1  # Ramener entre 0 et 1
            
            if score > 0.6:
                return {'signal': 'BUY', 'confidence': score}
            elif score < 0.4:
                return {'signal': 'SELL', 'confidence': 1 - score}
            else:
                return {'signal': 'NEUTRAL', 'confidence': 1 - abs(score - 0.5) * 2}
        
        return {'signal': 'NEUTRAL', 'confidence': 0.0}
    
    def _score_microstructure(self, microstructure: Dict) -> float:
        """Scorer la microstructure du marché"""
        score = 0.5  # Score neutre par défaut
        
        # Ajuster selon l'imbalance du carnet d'ordres
        if 'order_book_imbalance' in microstructure:
            imbalance = microstructure['order_book_imbalance']
            score += imbalance * 0.3  # ±0.3 max
        
        # Ajuster selon la liquidité
        if 'liquidity_score' in microstructure:
            liquidity = microstructure['liquidity_score']
            score += (liquidity - 0.5) * 0.2  # ±0.1 max
        
        # Pénaliser si l'impact est trop élevé
        if 'market_impact' in microstructure:
            impact = microstructure['market_impact']
            if impact > self.max_market_impact:
                score -= 0.2
        
        return max(0, min(1, score))
    
    def _score_order_flow(self, order_flow: Dict) -> float:
        """Scorer le flux d'ordres"""
        score = 0.5
        
        # Ajuster selon la pression d'achat/vente
        if 'buy_pressure' in order_flow:
            buy_pressure = order_flow['buy_pressure']
            score = buy_pressure  # Déjà entre 0 et 1
        
        # Bonus si large trades détectés dans la direction
        if 'large_trades' in order_flow and len(order_flow['large_trades']) > 0:
            # Analyser la direction des large trades
            buy_large = sum(1 for t in order_flow['large_trades'] if t['type'] == 'buy')
            sell_large = sum(1 for t in order_flow['large_trades'] if t['type'] == 'sell')
            
            if buy_large > sell_large:
                score += 0.1
            elif sell_large > buy_large:
                score -= 0.1
        
        return max(0, min(1, score))
    
    def _calculate_signal_confidence(self, scores: List[float]) -> float:
        """Calculer la confiance basée sur la cohérence des signaux"""
        if not scores:
            return 0.0
        
        # Calculer l'écart-type des scores
        std_dev = np.std(scores)
        
        # Plus les scores sont cohérents (faible std), plus la confiance est élevée
        confidence = 1 - (std_dev * 2)  # std_dev max ~0.5 donne confiance 0
        
        # Ajuster selon la force moyenne
        mean_score = np.mean(scores)
        strength_factor = abs(mean_score - 0.5) * 2  # 0 à 1
        
        confidence = confidence * strength_factor
        
        return max(0, min(1, confidence))
    
    def _get_regime_adjustment(self, regime: Dict) -> float:
        """Obtenir l'ajustement selon le régime de marché"""
        main_regime = regime.get('main_regime', MarketRegime.SIDEWAYS)
        vol_regime = regime.get('volatility_regime', MarketRegime.LOW_VOLATILITY)
        
        adjustment = 1.0
        
        # Ajuster selon le régime principal
        if main_regime == MarketRegime.SIDEWAYS:
            adjustment *= 0.7  # Réduire les signaux en range
        elif main_regime == MarketRegime.BEAR_TREND:
            adjustment *= 0.8  # Être plus prudent en tendance baissière
        
        # Ajuster selon la volatilité
        if vol_regime == MarketRegime.HIGH_VOLATILITY:
            adjustment *= 0.8  # Réduire en haute volatilité
        
        return adjustment
    
    def _calculate_institutional_risk_levels(self, microstructure: Dict, regime: Dict,
                                           institutional_score: Dict) -> Dict:
        """Calculer les niveaux de risque institutionnels"""
        current_price = microstructure.get('current_price', 0)
        
        # Stop-loss basé sur l'ATR et la microstructure
        atr_multiplier = 2.0
        if regime.get('volatility_regime') == MarketRegime.HIGH_VOLATILITY:
            atr_multiplier = 3.0
        
        # Utiliser la liquidité pour ajuster les stops
        liquidity_factor = microstructure.get('liquidity_score', 0.5)
        stop_distance = current_price * 0.02 * (1 + (1 - liquidity_factor))  # 2-4%
        
        # Take-profit basé sur la structure du marché et le R:R
        risk_reward_ratio = 2.5  # Ratio institutionnel standard
        if institutional_score.get('confidence', 0) > 0.8:
            risk_reward_ratio = 3.0
        
        return {
            'stop_loss': current_price - stop_distance,
            'take_profit': current_price + (stop_distance * risk_reward_ratio),
            'position_size_factor': liquidity_factor,  # Réduire la taille si faible liquidité
            'max_slippage': 0.001 * (1 + (1 - liquidity_factor))  # 0.1-0.2%
        }
    
    def _recommend_execution_strategy(self, microstructure: Dict, action: str) -> Dict:
        """Recommander une stratégie d'exécution institutionnelle"""
        liquidity = microstructure.get('liquidity_score', 0.5)
        market_impact = microstructure.get('market_impact', 0.001)
        
        if liquidity > 0.7 and market_impact < 0.001:
            # Haute liquidité, faible impact
            strategy = 'AGGRESSIVE'
            method = 'MARKET_ORDER'
        elif liquidity > 0.5:
            # Liquidité moyenne
            strategy = 'PASSIVE'
            method = 'LIMIT_ORDER'
        else:
            # Faible liquidité
            strategy = 'PATIENT'
            method = 'ICEBERG'
        
        return {
            'strategy': strategy,
            'method': method,
            'execution_timeframe': '5-15 minutes' if strategy == 'PATIENT' else '1-5 minutes',
            'slice_size': 0.1 if method == 'ICEBERG' else 1.0,
            'price_improvement_target': 0.0005 if strategy == 'PASSIVE' else 0
        }
    
    def calculate_position_size_institutional(self, pair: str, analysis: Dict) -> float:
        """
        Calculer la taille de position selon les méthodes institutionnelles
        
        Args:
            pair: Paire de trading
            analysis: Analyse complète du marché
            
        Returns:
            Taille de position optimale
        """
        try:
            # Récupérer les paramètres nécessaires
            current_price = analysis['current_price']
            risk_levels = analysis['recommendation']['risk_levels']
            confidence = analysis['confidence']
            liquidity = analysis['microstructure']['liquidity_score']
            
            # Capital effectif
            effective_capital = self.money_manager.get_effective_capital()
            
            # Risque par trade ajusté selon la confiance
            base_risk = Config.MAX_RISK_PER_TRADE / 100
            adjusted_risk = base_risk * confidence * risk_levels['position_size_factor']
            
            # Calculer la taille basée sur le stop-loss
            stop_distance = abs(current_price - risk_levels['stop_loss'])
            stop_percentage = stop_distance / current_price
            
            # Position size = (Capital * Risk%) / Stop%
            position_value = (effective_capital * adjusted_risk) / stop_percentage
            
            # Ajuster selon la liquidité
            position_value *= min(liquidity, 1.0)
            
            # Vérifier les limites
            max_position = effective_capital * 0.1  # Max 10% du capital par position
            position_value = min(position_value, max_position)
            
            # Convertir en volume
            volume = position_value / current_price
            
            return volume
            
        except Exception as e:
            logging.error(f"Erreur dans le calcul de position institutionnelle: {e}")
            return 0.0
    
    def get_risk_metrics(self, positions: Dict) -> Dict:
        """Calculer les métriques de risque du portefeuille"""
        try:
            total_exposure = sum(p['value'] for p in positions.values())
            
            # VaR et CVaR simplifiés
            returns = [p.get('unrealized_pnl', 0) / p['value'] for p in positions.values() if p['value'] > 0]
            
            if returns:
                var_95 = np.percentile(returns, 5)
                cvar_95 = np.mean([r for r in returns if r <= var_95])
            else:
                var_95 = 0
                cvar_95 = 0
            
            return {
                'total_exposure': total_exposure,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'exposure_ratio': total_exposure / self.money_manager.get_effective_capital(),
                'position_count': len(positions)
            }
            
        except Exception as e:
            logging.error(f"Erreur dans le calcul des métriques de risque: {e}")
            return {}