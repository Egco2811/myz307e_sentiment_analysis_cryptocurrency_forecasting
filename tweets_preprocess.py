import pandas as pd
import numpy as np
import re
from datetime import datetime
from tqdm import tqdm

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'@\w+', '', text)
    
    crypto_terms = ['btc', 'eth', 'hodl', 'fomo', 'dyor', 'bullish', 
                   'bearish', 'moon', 'dump', 'pump', 'whale', 'fud']
    
    words = text.split()
    processed_words = []
    
    for word in words:
        if word.lower() in crypto_terms:
            processed_words.append(word)
        else:
            word = re.sub(r'[^a-zA-Z\s]', '', word)
            if word:
                processed_words.append(word)
    
    text = ' '.join(processed_words)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_tweets(file_path):
    # Read CSV with explicit date parsing
    chunks = []
    chunk_size = 10000
    date_parser = lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S', errors='coerce')
    
    for chunk in tqdm(pd.read_csv(file_path, 
                                 chunksize=chunk_size,
                                 on_bad_lines='skip')):
        # Parse dates after reading chunk
        if 'date' in chunk.columns:
            chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
        chunks.append(chunk)
    
    df = pd.concat(chunks, ignore_index=True)
    
    # Convert date range to timestamps
    start_date = pd.Timestamp('2021-03-01')
    end_date = pd.Timestamp('2022-07-31')
    
    # Filter date range
    df = df.dropna(subset=['date'])  # Remove rows with invalid dates
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    df = df[mask]
    
    print(f"Initial tweets in date range: {len(df)}")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['text', 'date'])
    
    # Clean text
    print("Cleaning text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Filter by length
    df['word_count'] = df['cleaned_text'].str.split().str.len()
    df = df[(df['word_count'] >= 3) & (df['word_count'] <= 280)]
    
    # Sort by date
    df = df.sort_values('date')
    
    print(f"Final processed tweets: {len(df)}")
    
    return df

# Run preprocessing
if __name__ == "__main__":
    input_file = '/content/Bitcoin_tweets.csv'
    processed_df = process_tweets(input_file)
    
    # Save processed data
    processed_df.to_csv('processed_tweets.csv', index=False)
    
    # Display statistics
    print("\nDate range:")
    print("Start:", processed_df['date'].min())
    print("End:", processed_df['date'].max())
    
    # Sample processed tweets
    print("\nSample processed tweets:")
    sample_df = processed_df[['date', 'cleaned_text']].head()
    for _, row in sample_df.iterrows():
        print(f"{row['date']}: {row['cleaned_text']}")
    
    # Daily tweet counts
    daily_counts = processed_df.groupby(processed_df['date'].dt.date)['cleaned_text'].count()
    print("\nDaily tweet statistics:")
    print(daily_counts.describe())
