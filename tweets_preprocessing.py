import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import BertTokenizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TweetPreprocessor:
    def __init__(self):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """
        Clean and preprocess tweet text according to paper specifications
        """
        if isinstance(text, str):
            # Remove URLs
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            
            # Keep important crypto terms before removing special characters
            crypto_terms = ['btc', 'eth', 'hodl', 'fomo', 'dyor', 'bullish', 'bearish']
            text = text.lower()
            
            # Remove hashtags but keep the text
            text = re.sub(r'#(\w+)', r'\1', text)
            
            # Remove @ mentions
            text = re.sub(r'@\w+', '', text)
            
            # Remove non-alphabetic characters while preserving crypto terms
            text = ' '.join(word if word in crypto_terms else 
                          re.sub(r'[^a-zA-Z\s]', '', word) 
                          for word in text.split())
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Remove stop words except those relevant to sentiment
            important_words = {'not', 'no', 'nor', 'but'}
            words = word_tokenize(text)
            text = ' '.join([word for word in words if word not in self.stop_words or word in important_words])
            
            return text
        return ""

    def prepare_for_bert(self, text):
        """
        Prepare text for BERT model input
        """
        encoding = self.bert_tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return encoding

def process_tweets_file(input_file, output_file):
    """
    Process entire tweets dataset
    """
    processor = TweetPreprocessor()
    
    # Read data in chunks to handle large files
    chunk_size = 10000
    chunks = []
    
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        # Drop unnecessary columns
        columns_to_drop = ["user_name", "user_location", "user_description", 
                          "user_created", "user_followers", "user_friends",
                          "user_favourites", "user_verified", "hashtags", 
                          "source", "is_retweet"]
        
        chunk = chunk.drop(columns=[col for col in columns_to_drop if col in chunk.columns])
        
        # Clean text
        chunk["cleaned_text"] = chunk["text"].apply(processor.clean_text)
        
        # Remove empty tweets after cleaning
        chunk = chunk[chunk["cleaned_text"].str.len() > 0]
        
        chunks.append(chunk)
    
    # Combine all chunks
    df = pd.concat(chunks)
    
    # Save processed data
    df.to_csv(output_file, index=False)
    print(f"Processed {len(df)} tweets and saved to {output_file}")

if __name__ == "__main__":
    input_file = "filtered_tweets.csv"
    output_file = "processed_tweets.csv"
    process_tweets_file(input_file, output_file)
