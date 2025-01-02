import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BERTSentimentAnalyzer:
    def __init__(self, checkpoint_dir='sentiment_checkpoints'):
        """Initialize BERT model for sentiment analysis"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        try:
            logger.info("Loading FinBERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                "ProsusAI/finbert",
                torch_dtype=torch.float32
            )
        except Exception as e:
            logger.warning(f"FinBERT loading failed: {e}")
            logger.info("Falling back to base BERT...")
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.model = AutoModelForSequenceClassification.from_pretrained(
                'bert-base-uncased',
                num_labels=3,
                torch_dtype=torch.float32
            )
        
        self.model = self.model.to(self.device)
        
    def get_sentiment_scores(self, texts: list, batch_size: int = 16) -> np.ndarray:
        """
        Calculate sentiment scores for a list of texts
        Returns array of scores in range [-1, 1]
        """
        self.model.eval()
        scores = []
        
        # Clean and validate texts
        texts = [str(text) if pd.notna(text) else "" for text in texts]
        
        try:
            for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
                batch = texts[i:i + batch_size]
                
                # Tokenize inputs
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = F.softmax(outputs.logits, dim=1)
                    
                    # Extract probabilities for each class
                    pos_probs = probs[:, 2].cpu().numpy()
                    neu_probs = probs[:, 1].cpu().numpy()
                    neg_probs = probs[:, 0].cpu().numpy()
                    
                    # Calculate scores with neutral threshold
                    sentiment_scores = np.where(
                        neu_probs > 0.3,  # Neutral threshold
                        0,
                        pos_probs - neg_probs
                    )
                    scores.extend(sentiment_scores)
                
                
                torch.cuda.empty_cache()
                
                # Save checkpoint
                if i % 1000 == 0:
                    self._save_checkpoint(scores, len(texts))
                    
        except Exception as e:
            logger.error(f"Error during scoring: {e}")
            raise
            
        return np.array(scores)
    
    def _save_checkpoint(self, scores: list, total_texts: int):
        """Save checkpoint during processing"""
        checkpoint_path = os.path.join(self.checkpoint_dir, 'scores_checkpoint.npy')
        meta_path = os.path.join(self.checkpoint_dir, 'meta.json')
        
        np.save(checkpoint_path, np.array(scores))
        pd.DataFrame({
            'timestamp': [datetime.now().isoformat()],
            'processed_texts': [len(scores)],
            'total_texts': [total_texts]
        }).to_json(meta_path)

def process_tweets(input_file: str, start_date: str = '2021-03-01', end_date: str = '2022-07-31'):
    """Process tweets and extract sentiment scores"""
    logger.info("Loading tweet data...")
    
    try:
        # Load and preprocess data
        df = pd.read_csv(input_file, low_memory=False)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter date range
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        df = df[mask].copy()
        df = df.dropna(subset=['cleaned_text'])
        
        logger.info(f"Processing {len(df)} tweets between {start_date} and {end_date}")
        
        # Get sentiment scores
        analyzer = BERTSentimentAnalyzer()
        scores = analyzer.get_sentiment_scores(df['cleaned_text'].values)
        
        # Verify scores
        if len(scores) != len(df):
            raise ValueError(f"Score length mismatch: got {len(scores)}, expected {len(df)}")
        
        # Add sentiment analysis results
        df['sentiment_score'] = scores
        df['sentiment_state'] = pd.cut(
            df['sentiment_score'],
            bins=[-1, -0.2, 0.2, 1],
            labels=['Negative', 'Neutral', 'Positive']
        )
        
        # Calculate daily aggregations
        daily = df.groupby(df['date'].dt.date).agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'sentiment_state': lambda x: x.value_counts().to_dict()
        }).reset_index()
        
        daily.columns = ['date', 'sentiment_score', 'sentiment_std', 'tweet_count', 'sentiment_dist']
        
        # Save results
        logger.info("Saving processed data...")
        df.to_csv('sentiment_analyzed_tweets.csv', index=False)
        daily.to_csv('daily_sentiment.csv', index=False)
        
        return df, daily
        
    except Exception as e:
        logger.error(f"Error processing tweets: {e}")
        raise

def verify_results(df: pd.DataFrame, daily: pd.DataFrame) -> bool:
    """Verify the sentiment analysis results"""
    logger.info("Verifying results...")
    
    try:
        # Check basic statistics
        logger.info(f"\nProcessed {len(df)} tweets across {len(daily)} days")
        
        # Check sentiment distribution
        sentiment_dist = df['sentiment_state'].value_counts(normalize=True)
        logger.info("\nSentiment distribution:")
        logger.info(sentiment_dist)
        
        # Check score statistics
        logger.info("\nScore statistics:")
        logger.info(df['sentiment_score'].describe())
        
        # Validation checks
        checks = {
            "Score range": df['sentiment_score'].between(-1, 1).all(),
            "No missing values": not df['sentiment_score'].isna().any(),
            "Valid distribution": all(x > 0.05 for x in sentiment_dist)
        }
        
        for check, result in checks.items():
            logger.info(f"{check}: {'✓' if result else '×'}")
        
        return all(checks.values())
        
    except Exception as e:
        logger.error(f"Error in verification: {e}")
        return False

if __name__ == "__main__":
    try:
        # Process tweets
        df, daily = process_tweets('processed_tweets.csv')
        
        # Verify results
        if verify_results(df, daily):
            logger.info("Processing completed successfully!")
        else:
            logger.warning("Processing completed with validation warnings")
            
    except Exception as e:
        logger.error(f"Processing failed: {e}")
