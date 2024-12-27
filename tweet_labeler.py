import torch
from transformers import pipeline, BertTokenizer, BertModel
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
   """
   Implements BERT-based sentiment analysis as described in Section III.B.1
   """
   def __init__(self, config):
       model_name = config['sentiment_analysis']['bert_model']
       self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       self.tokenizer = BertTokenizer.from_pretrained(model_name)
       self.model = BertModel.from_pretrained(model_name).to(self.device)
       self.sentiment_classifier = pipeline(
           "sentiment-analysis",
           model=self.model,
           tokenizer=self.tokenizer,
           device=0 if torch.cuda.is_available() else -1
       )

   def calculate_probabilities(self, text: str) -> Dict[str, float]:
       """
       Calculate sentiment probabilities as defined in Section III.B.1.a
       """
       encoding = self.tokenizer(
           text,
           return_tensors='pt',
           max_length=128,
           truncation=True,
           padding='max_length'
       ).to(self.device)

       with torch.no_grad():
           outputs = self.model(**encoding)
           logits = outputs.logits
           probs = torch.softmax(logits, dim=-1)
           
           return {
               'positive': probs[0][2].item(),
               'neutral': probs[0][1].item(),
               'negative': probs[0][0].item()
           }

   def calculate_sentiment_score(self, text: str) -> float:
       """
       Calculate sentiment score using equation (3) from the paper
       """
       probs = self.calculate_probabilities(text)
       return probs['positive'] - probs['negative']

   def determine_sentiment_state(self, 
                               probs: Dict[str, float]) -> Tuple[str, float]:
       """
       Implement state definitions from Section III.B.1.c
       """
       p_pos = probs['positive']
       p_neg = probs['negative']
       p_neu = probs['neutral']
       
       # Calculate sentiment score
       score = p_pos - p_neg
       
       # Determine state based on paper criteria
       if score > 0 and p_pos >= p_neu:
           state = 'Positive'
       elif score < 0 and p_neg >= p_neu:
           state = 'Negative'
       else:
           state = 'Neutral'
           
       return state, score

class BatchSentimentProcessor:
   """
   Processes large volumes of tweets with efficient batching
   """
   def __init__(self, 
                analyzer: SentimentAnalyzer,
                batch_size: int = 32):
       self.analyzer = analyzer
       self.batch_size = batch_size

   def process_dataframe(self, 
                        df: pd.DataFrame,
                        text_column: str) -> pd.DataFrame:
       """
       Process entire dataframe of tweets with batching
       """
       results = []
       for i in range(0, len(df), self.batch_size):
           batch = df[text_column].iloc[i:i + self.batch_size]
           batch_results = self._process_batch(batch)
           results.extend(batch_results)
           
       df['sentiment_score'] = [r['score'] for r in results]
       df['sentiment_state'] = [r['state'] for r in results]
       return df

   def _process_batch(self, texts: List[str]) -> List[Dict]:
       """
       Process a batch of texts
       """
       results = []
       for text in texts:
           probs = self.analyzer.calculate_probabilities(text)
           state, score = self.analyzer.determine_sentiment_state(probs)
           results.append({
               'score': score,
               'state': state,
               'probabilities': probs
           })
       return results

class DailySentimentAggregator:
   """
   Implements daily sentiment aggregation as described in Section III.B.1.e
   """
   def __init__(self):
       self.required_columns = ['Date', 'sentiment_score', 'sentiment_state']

   def aggregate_daily_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
       """
       Calculate daily sentiment metrics according to equation (4)
       """
       self._validate_dataframe(df)
       
       daily_metrics = df.groupby('Date').agg({
           'sentiment_score': [
               'mean',                    # Daily sentiment score
               'std',                     # Score variation
               'count'                    # Number of tweets
           ],
           'sentiment_state': lambda x: x.value_counts().to_dict()
       }).reset_index()

       # Flatten column names
       daily_metrics.columns = [
           'Date', 'sentiment_score', 'sentiment_std', 
           'tweet_count', 'sentiment_distribution'
       ]
       
       return daily_metrics

   def _validate_dataframe(self, df: pd.DataFrame):
       """
       Ensure dataframe has required columns
       """
       missing_cols = set(self.required_columns) - set(df.columns)
       if missing_cols:
           raise ValueError(f"Missing required columns: {missing_cols}")

def process_tweets_file(input_file: str,
                      output_file: str,
                      batch_size: int = 32) -> None:
   """
   Process entire tweets dataset with sentiment analysis
   """
   try:
       # Initialize components
       analyzer = SentimentAnalyzer()
       processor = BatchSentimentProcessor(analyzer, batch_size)
       aggregator = DailySentimentAggregator()
       
       # Read and process data
       df = pd.read_csv(input_file)
       logger.info(f"Processing {len(df)} tweets...")
       
       # Process individual tweets
       df = processor.process_dataframe(df, 'text')
       
       # Calculate daily aggregations
       daily_sentiment = aggregator.aggregate_daily_sentiment(df)
       
       # Save results
       daily_sentiment.to_csv(output_file, index=False)
       logger.info(f"Results saved to {output_file}")
       
   except Exception as e:
       logger.error(f"Error processing tweets: {str(e)}")
       raise

if __name__ == "__main__":
   input_file = "processed_tweets.csv"
   output_file = "sentiment_scores.csv"
   process_tweets_file(input_file, output_file)
