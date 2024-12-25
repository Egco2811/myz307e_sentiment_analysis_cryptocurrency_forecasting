import csv
import os
import shutil
from transformers import pipeline
import pandas as pd
import torch

class SentimentLabeler:
    def __init__(self):
        # Initialize BERT sentiment analyzer
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="bert-base-uncased",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Define sentiment mapping
        self.sentiment_map = {
            'POSITIVE': 1,
            'NEGATIVE': 2,
            'NEUTRAL': 3
        }

    def get_bert_sentiment(self, text):
        """
        Get sentiment prediction from BERT
        """
        try:
            result = self.sentiment_analyzer(text)[0]
            score = result['score']
            
            # Define thresholds for neutral sentiment
            if 0.4 <= score <= 0.6:
                return 3  # Neutral
            return 1 if result['label'] == 'POSITIVE' else 2
        except:
            return None

    def manual_labeling(self, text):
        """
        Get manual sentiment label from user
        """
        print("\nText:", text)
        while True:
            sentiment = input(
                "1. Positive\n2. Negative\n3. Neutral\n"
                "Type 'end' to finish current batch.\nType the number: "
            )
            if sentiment.lower() == 'end':
                return None
            try:
                sentiment_num = int(sentiment)
                if sentiment_num in [1, 2, 3]:
                    return sentiment_num
                print("Please enter 1, 2, or 3.")
            except ValueError:
                print("Invalid input. Please enter 1, 2, 3, or 'end'.")

def process_file(csv_filepath, batch_size=100):
    """
    Process CSV file for sentiment labeling
    """
    labeler = SentimentLabeler()
    
    df = pd.read_csv(csv_filepath)
    
    if 'sentiment' not in df.columns:
        df['sentiment'] = None
        
    # Get indices of unlabeled rows
    unlabeled_indices = df[df['sentiment'].isnull()].index
    
    for i in range(0, len(unlabeled_indices), batch_size):
        batch_indices = unlabeled_indices[i:i+batch_size]
        
        for idx in batch_indices:
            text = df.loc[idx, 'text']
            
            # Try automatic labeling first
            sentiment = labeler.get_bert_sentiment(text)
            
            # If automatic labeling fails, do manual labeling
            if sentiment is None:
                sentiment = labeler.manual_labeling(text)
                if sentiment is None:  # User wants to end
                    break
            
            df.loc[idx, 'sentiment'] = sentiment
        
        # Save progress after each batch
        df.to_csv(csv_filepath, index=False)
        
        continue_labeling = input("\nContinue with next batch? (yes/no): ")
        if continue_labeling.lower() != 'yes':
            break
    
    return df

if __name__ == "__main__":
    csv_file_path = "processed_tweets.csv"
    process_file(csv_file_path)
