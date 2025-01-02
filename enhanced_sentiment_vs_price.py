import pandas as pd
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def plot_sentiment_vs_price(price_df: pd.DataFrame,
                            sentiment_df: pd.DataFrame,
                            scale_factor: float = 20000.0,
                            start_date: str = None,
                            end_date: str = None,
                            out_file: str = 'sentiment_vs_price.png'):
  
    # 1) Convert to datetime
    price_df = price_df.copy()
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    sentiment_df = sentiment_df.copy()
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

    # 2) Optional date filtering
    if start_date:
        price_df = price_df[price_df['Date'] >= start_date]
        sentiment_df = sentiment_df[sentiment_df['date'] >= start_date]
    if end_date:
        price_df = price_df[price_df['Date'] <= end_date]
        sentiment_df = sentiment_df[sentiment_df['date'] <= end_date]

    # 3) Merge on date
    merged = pd.merge(
        price_df[['Date', 'Close']],
        sentiment_df[['date', 'sentiment_score']],
        left_on='Date',
        right_on='date',
        how='inner'
    ).sort_values('Date')

    if len(merged) < 1:
        logger.warning("No overlapping dates found after filtering.")
        return

    # 4) Scale sentiment for visualization
    merged['scaled_sentiment'] = merged['sentiment_score'] * scale_factor

    # 5) Plot overlay
    plt.figure(figsize=(12, 6))
    plt.plot(merged['Date'], merged['Close'], label='Price (USD)', color='blue')
    plt.plot(merged['Date'], merged['scaled_sentiment'], label=f'Sentiment x {scale_factor:g}', color='red')
    
    plt.title('Overlay of Bitcoin Price and Daily Sentiment')
    plt.xlabel('Date')
    plt.ylabel('Price (USD) / Scaled Sentiment')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(out_file, dpi=300)
    plt.close()

    logger.info(f"Overlay plot saved to: {out_file}")
   main()
