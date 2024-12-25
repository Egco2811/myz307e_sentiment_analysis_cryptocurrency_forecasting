import pandas as pd
import re
from datetime import datetime
import numpy as np

class DataMatcher:
    def __init__(self, bitcoin_file, tweets_file):
        self.bitcoin_file = bitcoin_file
        self.tweets_file = tweets_file
        
    def load_bitcoin_data(self):
        """
        Load and preprocess Bitcoin price data
        """
        bitcoin_df = pd.read_csv(
            self.bitcoin_file,
            parse_dates=["Date"],
            date_format="%Y-%m-%d"
        )
        
        # Calculate daily returns
        bitcoin_df['returns'] = bitcoin_df['Close'].pct_change()
        
        # Add volatility measure
        bitcoin_df['volatility'] = bitcoin_df['returns'].rolling(window=30).std()
        
        return bitcoin_df
    
    def process_tweets_chunk(self, chunk, bitcoin_df):
        """
        Process a chunk of tweets and merge with Bitcoin data
        """
        # Convert tweet dates to datetime
        chunk["Date"] = pd.to_datetime(
            chunk[self.tweets_date_col].str.split().str[0],
            format="%Y-%m-%d",
            errors="coerce"
        )
        
        # Group tweets by date and calculate daily sentiment statistics
        daily_sentiment = chunk.groupby("Date")['sentiment'].agg([
            'mean',
            'count',
            'std'
        ]).reset_index()
        
        # Merge with bitcoin data
        merged_chunk = pd.merge(
            bitcoin_df,
            daily_sentiment,
            on="Date",
            how="inner"
        )
        
        return merged_chunk

    def process_data(self):
        """
        Process entire dataset
        """
        bitcoin_df = self.load_bitcoin_data()
        
        tweets_df_iter = pd.read_csv(
            self.tweets_file,
            iterator=True,
            chunksize=5000,
            on_bad_lines="error",
            encoding="utf-8",
            engine="python",
            low_memory=True
        )
        
        # Process first chunk to get column names
        tweets_chunk_sample = next(tweets_df_iter)
        self.tweets_date_col = tweets_chunk_sample.columns[8]
        
        # Reset iterator
        tweets_df_iter = pd.read_csv(
            self.tweets_file,
            iterator=True,
            chunksize=5000,
            on_bad_lines="error",
            encoding="utf-8",
            engine="python",
            low_memory=True
        )
        
        merged_chunks = []
        for chunk in tweets_df_iter:
            merged_chunk = self.process_tweets_chunk(chunk, bitcoin_df)
            if not merged_chunk.empty:
                merged_chunks.append(merged_chunk)
        
        # Combine all chunks
        final_df = pd.concat(merged_chunks)
        
        # Save processed data
        final_df.to_csv("merged_data.csv", index=False)
        print("Data processing complete. Results saved to merged_data.csv")

if __name__ == "__main__":
    matcher = DataMatcher("bitcoin.csv", "processed_tweets.csv")
    matcher.process_data()
