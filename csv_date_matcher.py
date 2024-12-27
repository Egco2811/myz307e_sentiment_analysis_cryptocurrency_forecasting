import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataMatcher:
   """
   Implements data alignment and preprocessing for combining price and sentiment data
   as described in Section III.A.1 of the paper.
   """
   def __init__(self, config: Dict):
       self.config = config
       self.price_scaler = MinMaxScaler()
       self.sentiment_scaler = MinMaxScaler()

   def load_bitcoin_data(self, price_file: str) -> pd.DataFrame:
       """
       Load and preprocess Bitcoin price data
       """
       try:
           bitcoin_df = pd.read_csv(
               price_file,
               parse_dates=['Date'],
               date_format='%Y-%m-%d'
           )
           
           # Verify required columns
           required_cols = ['Date', 'Close', 'Open', 'High', 'Low', 'Volume']
           missing_cols = set(required_cols) - set(bitcoin_df.columns)
           if missing_cols:
               raise ValueError(f"Missing required columns: {missing_cols}")
           
           # Calculate daily metrics
           bitcoin_df['returns'] = bitcoin_df['Close'].pct_change()
           bitcoin_df['high_low_spread'] = (bitcoin_df['High'] - bitcoin_df['Low']) / bitcoin_df['Low']
           bitcoin_df['volatility'] = bitcoin_df['returns'].rolling(window=30).std()
           
           # Handle missing values
           bitcoin_df = bitcoin_df.fillna(method='ffill').fillna(method='bfill')
           
           return bitcoin_df
           
       except Exception as e:
           logger.error(f"Error loading Bitcoin data: {str(e)}")
           raise

   def load_sentiment_data(self, sentiment_file: str) -> pd.DataFrame:
       """
       Load and preprocess sentiment data
       """
       try:
           sentiment_df = pd.read_csv(
               sentiment_file,
               parse_dates=['Date']
           )
           
           # Verify required columns
           if 'sentiment_score' not in sentiment_df.columns:
               raise ValueError("Missing sentiment_score column")
           
           return sentiment_df
           
       except Exception as e:
           logger.error(f"Error loading sentiment data: {str(e)}")
           raise

   def align_dates(self, 
                  price_df: pd.DataFrame, 
                  sentiment_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
       """
       Align price and sentiment data by date
       """
       # Convert dates to datetime if needed
       price_df['Date'] = pd.to_datetime(price_df['Date'])
       sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
       
       # Filter date range
       start_date = pd.to_datetime(self.config['raw_data']['start_date'])
       end_date = pd.to_datetime(self.config['raw_data']['end_date'])
       
       price_df = price_df[
           (price_df['Date'] >= start_date) & 
           (price_df['Date'] <= end_date)
       ]
       sentiment_df = sentiment_df[
           (sentiment_df['Date'] >= start_date) & 
           (sentiment_df['Date'] <= end_date)
       ]
       
       # Get common dates
       common_dates = pd.Index(
           sorted(set(price_df['Date']) & set(sentiment_df['Date']))
       )
       
       return (
           price_df[price_df['Date'].isin(common_dates)],
           sentiment_df[sentiment_df['Date'].isin(common_dates)]
       )

   def handle_missing_values(self, 
                           price_df: pd.DataFrame, 
                           sentiment_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
       """
       Handle missing values according to Section III.A.1
       """
       # Forward fill price data
       price_df = price_df.fillna(method='ffill')
       
       # Fill sentiment gaps with global mean
       sentiment_mean = sentiment_df['sentiment_score'].mean()
       sentiment_df = sentiment_df.fillna(sentiment_mean)
       
       return price_df, sentiment_df

   def normalize_data(self, 
                     price_df: pd.DataFrame, 
                     sentiment_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
       """
       Normalize price and sentiment data
       """
       # Normalize price data
       price_cols = ['Close', 'Open', 'High', 'Low', 'returns', 'volatility']
       price_df[price_cols] = self.price_scaler.fit_transform(price_df[price_cols])
       
       # Normalize sentiment data
       sentiment_df['sentiment_score'] = self.sentiment_scaler.fit_transform(
           sentiment_df[['sentiment_score']]
       )
       
       return price_df, sentiment_df

   def process_data(self) -> Dict[str, pd.DataFrame]:
       """
       Main processing pipeline
       """
       try:
           logger.info("Starting data processing...")
           
           # Load data
           price_df = self.load_bitcoin_data(self.config['raw_data']['price_file'])
           sentiment_df = self.load_sentiment_data(self.config['raw_data']['tweets_file'])
           
           # Align dates
           logger.info("Aligning dates...")
           price_df, sentiment_df = self.align_dates(price_df, sentiment_df)
           
           # Handle missing values
           logger.info("Handling missing values...")
           price_df, sentiment_df = self.handle_missing_values(price_df, sentiment_df)
           
           # Normalize data
           logger.info("Normalizing data...")
           price_df, sentiment_df = self.normalize_data(price_df, sentiment_df)
           
           # Save processed data
           self.save_processed_data(price_df, sentiment_df)
           
           logger.info(
               f"Data processing complete. "
               f"Final dataset contains {len(price_df)} rows"
           )
           
           return {
               'price_data': price_df,
               'sentiment_data': sentiment_df
           }
           
       except Exception as e:
           logger.error(f"Error in data processing: {str(e)}")
           raise

   def save_processed_data(self,
                         price_df: pd.DataFrame,
                         sentiment_df: pd.DataFrame):
       """
       Save processed datasets
       """
       output_dir = Path(self.config['output_directory'])
       output_dir.mkdir(exist_ok=True)
       
       price_df.to_csv(output_dir / 'processed_price_data.csv', index=False)
       sentiment_df.to_csv(output_dir / 'processed_sentiment_data.csv', index=False)

def main():
   """
   Main execution function
   """
   # Load configuration
   with open('config.json', 'r') as f:
       config = json.load(f)
   
   # Initialize data matcher
   matcher = DataMatcher(config)
   
   # Process data
   processed_data = matcher.process_data()
   
   logger.info("Processing complete!")

if __name__ == "__main__":
   main()
