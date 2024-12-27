import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import BertTokenizer
import logging
from typing import List, Dict, Optional
from pathlib import Path
import json
from tqdm import tqdm
import emoji
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TweetPreprocessor:
   """
   Implements comprehensive tweet preprocessing pipeline as described in the paper
   """
   def __init__(self, config: Dict):
       self.config = config
       self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
       self.stop_words = set(stopwords.words('english'))
       
       # Download required NLTK data
       nltk.download('punkt', quiet=True)
       nltk.download('stopwords', quiet=True)
       nltk.download('wordnet', quiet=True)
       
       # Compile regex patterns
       self.url_pattern = re.compile(r'http\S+|www\S+|https\S+')
       self.username_pattern = re.compile(r'@\w+')
       self.hashtag_pattern = re.compile(r'#(\w+)')
       
       # Load crypto-specific terms
       self.crypto_terms = self._load_crypto_terms()

   def _load_crypto_terms(self) -> List[str]:
       """
       Load and return crypto-specific terms
       """
       return [
           'btc', 'eth', 'hodl', 'fomo', 'dyor', 'bullish', 'bearish',
           'moon', 'dump', 'pump', 'whale', 'fud', 'lambo', 'rekt',
           'shill', 'alt', 'blockchain', 'mining', 'satoshi'
       ]

   def clean_text(self, text: str) -> str:
       """
       Clean and preprocess tweet text according to paper specifications
       """
       if not isinstance(text, str):
           return ""
           
       # Convert to lowercase
       text = text.lower()
       
       # Replace emoji with text representation
       text = emoji.demojize(text)
       
       # Remove URLs
       text = self.url_pattern.sub('', text)
       
       # Process hashtags but keep the text
       text = self.hashtag_pattern.sub(r'\1', text)
       
       # Remove usernames
       text = self.username_pattern.sub('', text)
       
       # Split into words
       words = text.split()
       
       # Process each word
       processed_words = []
       for word in words:
           # Keep crypto terms intact
           if word in self.crypto_terms:
               processed_words.append(word)
           else:
               # Remove non-alphabetic characters
               word = re.sub(r'[^a-zA-Z\s]', '', word)
               if word:
                   processed_words.append(word)
       
       # Join words and remove extra whitespace
       text = ' '.join(processed_words)
       text = re.sub(r'\s+', ' ', text).strip()
       
       # Remove stop words except those relevant to sentiment
       important_words = {'not', 'no', 'nor', 'but', 'very', 'most', 'more'}
       words = word_tokenize(text)
       text = ' '.join([
           word for word in words 
           if word not in self.stop_words or word in important_words
       ])
       
       return text

   def prepare_for_bert(self, text: str) -> Dict:
       """
       Prepare text for BERT model input
       """
       encoding = self.bert_tokenizer.encode_plus(
           text,
           add_special_tokens=True,
           max_length=self.config['preprocessing']['max_tweet_length'],
           padding='max_length',
           truncation=True,
           return_attention_mask=True,
           return_tensors='pt'
       )
       return encoding

   def filter_tweet(self, tweet: Dict) -> bool:
       """
       Check if tweet meets filtering criteria
       """
       if not tweet.get('text'):
           return False
           
       text_length = len(str(tweet['text']).split())
       
       return (
           text_length >= self.config['preprocessing']['min_tweet_length'] and
           text_length <= self.config['preprocessing']['max_tweet_length'] and
           tweet.get('lang') == self.config['preprocessing']['language']
       )

   def process_tweet_batch(self, tweets: List[Dict]) -> List[Dict]:
       """
       Process a batch of tweets
       """
       processed_tweets = []
       
       for tweet in tweets:
           if self.filter_tweet(tweet):
               cleaned_text = self.clean_text(tweet['text'])
               if cleaned_text:
                   processed_tweets.append({
                       'text': tweet['text'],
                       'cleaned_text': cleaned_text,
                       'created_at': tweet['created_at'],
                       'bert_encoding': self.prepare_for_bert(cleaned_text)
                   })
       
       return processed_tweets

   def process_tweets_file(self, input_file: str,
                          output_file: str,
                          batch_size: int = 1000) -> None:
        """
        Process entire tweets dataset
        """
        try:
            # Read data in chunks
            chunks = pd.read_csv(
                input_file,
                chunksize=batch_size,
                on_bad_lines='skip'
            )
            
            processed_tweets = []
            for chunk in tqdm(chunks, desc="Processing tweets"):
                tweets = chunk.to_dict('records')
                batch_processed = self.process_tweet_batch(tweets)
                processed_tweets.extend(batch_processed)
                
            # Convert to DataFrame
            df = pd.DataFrame(processed_tweets)
            
            # Save processed data
            df.to_csv(output_file, index=False)
            logger.info(
                f"Processed {len(processed_tweets)} tweets. "
                f"Results saved to {output_file}"
            )
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing tweets: {str(e)}")
            raise

def main():
    with open('config.json', 'r') as f:
        config = json.load(f)
   
    input_file = config['raw_data']['tweets_file']
    output_file = Path(config['output_directory']) / 'processed_tweets.csv'
   
    # Create preprocessor instance and call its method
    preprocessor = TweetPreprocessor(config)
    preprocessor.process_tweets_file(input_file, output_file)

if __name__ == "__main__":
    main()