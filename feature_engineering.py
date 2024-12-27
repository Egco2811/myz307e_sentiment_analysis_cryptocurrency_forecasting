import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import talib
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import json

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
   as described in Section III.A.
   """
   def __init__(self, config: Dict):
       self.config = config
       self.scaler = MinMaxScaler()
       self.feature_names = []
       
   def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
       """
       Creates temporal features to capture time-based patterns
       """
       df['Date'] = pd.to_datetime(df['Date'])
       
       # Basic time features
       df['day_of_week'] = df['Date'].dt.dayofweek
       df['month'] = df['Date'].dt.month
       df['quarter'] = df['Date'].dt.quarter
       df['year'] = df['Date'].dt.year
       df['is_weekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)
       
       # Cyclical features
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
       Creates technical indicators based on price data
       """
       # Returns at different frequencies
       df['returns_1d'] = df['Close'].pct_change()
       df['returns_3d'] = df['Close'].pct_change(periods=3)
       df['returns_7d'] = df['Close'].pct_change(periods=7)
       
       # Volatility features
       for window in self.config['feature_engineering']['price_features']['moving_averages']:
           df[f'volatility_{window}d'] = df['returns_1d'].rolling(window).std()
           df[f'volatility_ratio_{window}d'] = (
               df[f'volatility_{window}d'] / 
               df[f'volatility_{window}d'].rolling(window).mean()
           )
       
       # Technical indicators
       df['rsi'] = talib.RSI(
           df['Close'].values, 
           timeperiod=self.config['feature_engineering']['price_features']['rsi_period']
       )
       
       macd_config = self.config['feature_engineering']['price_features']['macd_params']
       df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
           df['Close'].values,
           fastperiod=macd_config['fast_period'],
           slowperiod=macd_config['slow_period'],
           signalperiod=macd_config['signal_period']
       )
       
       # Moving averages and crossovers
       for window in self.config['feature_engineering']['price_features']['moving_averages']:
           df[f'ma_{window}'] = df['Close'].rolling(window).mean()
           df[f'ma_ratio_{window}'] = df['Close'] / df[f'ma_{window}']
       
       # Bollinger Bands
       bb_config = self.config['feature_engineering']['price_features']['bollinger_bands']
       df['upper_bb'], df['middle_bb'], df['lower_bb'] = talib.BBANDS(
           df['Close'].values,
           timeperiod=bb_config['window'],
           nbdevup=bb_config['num_std'],
           nbdevdn=bb_config['num_std']
       )
       
       df['bb_width'] = (df['upper_bb'] - df['lower_bb']) / df['middle_bb']
       
       # Update feature names
       self.feature_names.extend([col for col in df.columns if col.startswith((
           'returns', 'volatility', 'rsi', 'macd', 'ma_', 'bb_'
       ))])
       
       return df

   def create_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
       """
       Creates advanced sentiment features based on BERT sentiment scores
       """
       sent_config = self.config['feature_engineering']['sentiment_features']
       
       # Rolling statistics
       for window in sent_config['rolling_windows']:
           df[f'sentiment_ma_{window}d'] = (
               df['sentiment_score'].rolling(window).mean()
           )
           df[f'sentiment_std_{window}d'] = (
               df['sentiment_score'].rolling(window).std()
           )
           df[f'sentiment_roc_{window}d'] = (
               df['sentiment_score'].diff(window) / window
           )
           df[f'sentiment_rs_{window}d'] = (
               df['sentiment_score'] / 
               df[f'sentiment_ma_{window}d']
           )
       
       # Sentiment momentum
       for period in sent_config['momentum_periods']:
           df[f'sentiment_momentum_{period}d'] = (
               df['sentiment_score'] - df['sentiment_score'].shift(period)
           )
       
       # Sentiment-price divergence
       df['sentiment_price_divergence'] = (
           stats.zscore(df['sentiment_score']) - 
           stats.zscore(df['returns_1d'])
       )
       
       # Sentiment regimes
       df['sentiment_regime'] = pd.qcut(
           df['sentiment_score'], 
           q=5, 
           labels=['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
       )
       
       # One-hot encode regimes
       sentiment_dummies = pd.get_dummies(
           df['sentiment_regime'], 
           prefix='sentiment_regime'
       )
       df = pd.concat([df, sentiment_dummies], axis=1)
       
       self.feature_names.extend([col for col in df.columns if col.startswith((
           'sentiment_ma', 'sentiment_std', 'sentiment_momentum',
           'sentiment_roc', 'sentiment_rs', 'sentiment_regime_'
       ))])
       
       return df

   def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
       """
       Creates features capturing interactions between price and sentiment
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
           df['sentiment_score'] * df['rsi']
       )
       
       # Lagged interactions
       lags = [1, 2, 3, 5, 7]
       for lag in lags:
           df[f'sentiment_return_interaction_lag_{lag}'] = (
               df['sentiment_score'].shift(lag) * df['returns_1d']
           )
       
       self.feature_names.extend([col for col in df.columns if col.startswith((
           'price_sentiment_corr', 'sentiment_volatility',
           'sentiment_momentum', 'sentiment_return_interaction'
       ))])
       
       return df

   def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
       """
       Handles missing values in engineered features
       """
       # Forward fill price features
       price_cols = [col for col in df.columns if col.startswith((
           'returns', 'volatility', 'rsi', 'macd', 'ma_', 'bb_'
       ))]
       df[price_cols] = df[price_cols].fillna(method='ffill')
       
       # Fill sentiment features with neutral values
       sentiment_cols = [col for col in df.columns if col.startswith('sentiment_')]
       df[sentiment_cols] = df[sentiment_cols].fillna(0)
       
       return df.dropna()

   def normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
       """
       Normalizes features using Min-Max scaling
       """
       feature_cols = self.feature_names
       df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
       return df

   def create_all_features(self, 
                         price_data: pd.DataFrame, 
                         sentiment_data: pd.DataFrame) -> pd.DataFrame:
       """
       Main function to create all features and prepare final dataset
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
       
       # Save feature info
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
   Main function to run feature engineering pipeline
   """
   # Load configuration
   with open('config.json', 'r') as f:
       config = json.load(f)
   
   # Load data
   price_data = pd.read_csv(config['raw_data']['price_file'])
   sentiment_data = pd.read_csv(config['raw_data']['tweets_file'])
   
   # Initialize feature engineer
   engineer = CryptoFeatureEngineer(config)
   
   # Create features
   final_dataset = engineer.create_all_features(price_data, sentiment_data)
   
   # Save processed dataset
   final_dataset.to_csv('engineered_features.csv', index=False)
   logger.info("Features saved to 'engineered_features.csv'")

if __name__ == "__main__":
   main()
