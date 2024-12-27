import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import ta
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class CryptoFeatureEngineer:
    """
    Implements comprehensive feature engineering for cryptocurrency data,
    combining price-based technical indicators with sentiment features
    as described in the paper's Section III.A.
    """
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_names = []
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates temporal features to capture time-based patterns in trading
        behavior and sentiment.
        """
        # Convert date to datetime if it's not already
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Extract basic time features
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        df['year'] = df['Date'].dt.year
        df['is_weekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)
        
        # Create cyclical features for day of week and month
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        
        self.feature_names.extend([
            'day_sin', 'day_cos', 'month_sin', 'month_cos', 'is_weekend'
        ])
        
        return df

    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates technical indicators based on price data as specified
        in the paper's methodology.
        """
        # Calculate returns at different frequencies
        df['returns_1d'] = df['Close'].pct_change()
        df['returns_3d'] = df['Close'].pct_change(periods=3)
        df['returns_7d'] = df['Close'].pct_change(periods=7)
        
        # Calculate volatility features
        for window in [7, 14, 30]:
            df[f'volatility_{window}d'] = df['returns_1d'].rolling(window).std()
            df[f'volatility_ratio_{window}d'] = (
                df[f'volatility_{window}d'] / 
                df[f'volatility_{window}d'].rolling(window).mean()
            )
        
        # Calculate momentum indicators
        df['rsi_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['macd'], df['macd_signal'], df['macd_hist'] = ta.trend.MACD(
            df['Close'].values,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        
        # Calculate moving averages and crossovers
        for window in [7, 14, 30, 50]:
            df[f'ma_{window}'] = df['Close'].rolling(window).mean()
            df[f'ma_ratio_{window}'] = df['Close'] / df[f'ma_{window}']
        
        # Price channels and trend features
        df['upper_bb'], df['middle_bb'], df['lower_bb'] = ta.volatility.BollingerBands(
            df['Close'].values,
            window=20
        ).bollinger_hband(), ta.volatility.BollingerBands(
            df['Close'].values,
            window=20
        ).bollinger_mavg, ta.volatility.BollingerBands(
            df['Close'].values,
            window=20
        ).bollinger_lband
        df['bb_width'] = (df['upper_bb'] - df['lower_bb']) / df['middle_bb']
        
        price_features = [col for col in df.columns if col.startswith((
            'returns', 'volatility', 'rsi', 'macd', 'ma_', 'bb_'
        ))]
        self.feature_names.extend(price_features)
        
        return df

    def create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates advanced sentiment features based on the BERT sentiment scores
        as described in the paper.
        """
        # Calculate rolling sentiment statistics
        windows = [3, 7, 14, 30]
        for window in windows:
            # Rolling mean and standard deviation
            df[f'sentiment_ma_{window}d'] = (
                df['sentiment_score'].rolling(window).mean()
            )
            df[f'sentiment_std_{window}d'] = (
                df['sentiment_score'].rolling(window).std()
            )
            
            # Sentiment momentum (rate of change)
            df[f'sentiment_roc_{window}d'] = (
                df['sentiment_score'].diff(window) / window
            )
            
            # Sentiment relative strength
            df[f'sentiment_rs_{window}d'] = (
                df['sentiment_score'] / 
                df[f'sentiment_ma_{window}d']
            )
        
        # Calculate sentiment divergence with price
        df['sentiment_price_divergence'] = (
            stats.zscore(df['sentiment_score']) - 
            stats.zscore(df['returns_1d'])
        )
        
        # Create sentiment regime features
        df['sentiment_regime'] = pd.qcut(
            df['sentiment_score'], 
            q=5, 
            labels=['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
        )
        
        # One-hot encode sentiment regimes
        sentiment_dummies = pd.get_dummies(
            df['sentiment_regime'], 
            prefix='sentiment_regime'
        )
        df = pd.concat([df, sentiment_dummies], axis=1)
        
        sentiment_features = [col for col in df.columns if col.startswith((
            'sentiment_ma', 'sentiment_std', 'sentiment_roc', 
            'sentiment_rs', 'sentiment_regime_'
        ))]
        self.feature_names.extend(sentiment_features)
        
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates features capturing interactions between price and sentiment
        as specified in the paper.
        """
        # Price-Sentiment Correlations
        windows = [7, 14, 30]
        for window in windows:
            df[f'price_sentiment_corr_{window}d'] = (
                df['returns_1d'].rolling(window)
                .corr(df['sentiment_score'])
            )
        
        # Sentiment-Volatility Interaction
        df['sentiment_volatility_interaction'] = (
            df['sentiment_score'] * df['volatility_7d']
        )
        
        # Sentiment-Momentum Interaction
        df['sentiment_momentum_interaction'] = (
            df['sentiment_score'] * df['rsi_14']
        )
        
        # Create lagged interaction terms
        lags = [1, 2, 3, 5, 7]
        for lag in lags:
            df[f'sentiment_return_interaction_lag_{lag}'] = (
                df['sentiment_score'].shift(lag) * df['returns_1d']
            )
        
        interaction_features = [col for col in df.columns if col.startswith((
            'price_sentiment_corr', 'sentiment_volatility',
            'sentiment_momentum', 'sentiment_return_interaction'
        ))]
        self.feature_names.extend(interaction_features)
        
        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handles missing values in engineered features using appropriate methods.
        """
        # Forward fill price-based features
        price_cols = [col for col in df.columns if col.startswith((
            'returns', 'volatility', 'rsi', 'macd', 'ma_', 'bb_'
        ))]
        df[price_cols] = df[price_cols].fillna(method='ffill')
        
        # Fill sentiment features with neutral values
        sentiment_cols = [col for col in df.columns if col.startswith('sentiment_')]
        df[sentiment_cols] = df[sentiment_cols].fillna(0)
        
        # Drop any remaining rows with missing values
        df = df.dropna()
        
        return df

    def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes all features to the same scale using Min-Max scaling.
        """
        feature_cols = self.feature_names
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        return df

    def create_all_features(self, 
                          price_data: pd.DataFrame, 
                          sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """
        Main function to create all features and prepare final dataset.
        """
        logger.info("Starting feature engineering process...")
        
        # Merge price and sentiment data
        df = pd.merge(price_data, sentiment_data, on='Date', how='inner')
        
        # Create features
        logger.info("Creating time features...")
        df = self.create_time_features(df)
        
        logger.info("Creating price features...")
        df = self.create_price_features(df)
        
        logger.info("Creating sentiment features...")
        df = self.create_sentiment_features(df)
        
        logger.info("Creating interaction features...")
        df = self.create_interaction_features(df)
        
        # Handle missing values
        logger.info("Handling missing values...")
        df = self.handle_missing_values(df)
        
        # Normalize features
        logger.info("Normalizing features...")
        df = self.normalize_features(df)
        
        # Save feature names and importance info
        feature_info = {
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names)
        }
        
        with open('feature_info.json', 'w') as f:
            json.dump(feature_info, f, indent=4)
        
        logger.info(f"Feature engineering complete. Created {len(self.feature_names)} features.")
        return df

def main():
    """
    Main function to demonstrate feature engineering process.
    """
    # Load data
    price_data = pd.read_csv('bitcoin.csv')
    sentiment_data = pd.read_csv('processed_tweets.csv')
    
    # Initialize feature engineer
    engineer = CryptoFeatureEngineer()
    
    # Create features
    final_dataset = engineer.create_all_features(price_data, sentiment_data)
    
    # Save processed dataset
    final_dataset.to_csv('engineered_features.csv', index=False)
    logger.info("Features saved to 'engineered_features.csv'")

if __name__ == "__main__":
    main()