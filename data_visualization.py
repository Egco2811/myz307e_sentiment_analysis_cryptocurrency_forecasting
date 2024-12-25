import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
import torch
from sklearn.preprocessing import MinMaxScaler
import json

# Configure logging and style settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
plt.style.use('seaborn')
sns.set_palette("husl")

class CryptoVisualizer:
    """
    Creates comprehensive visualizations for cryptocurrency analysis,
    integrating price data, sentiment scores, and model predictions.
    """
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize the visualizer with an output directory for saving plots.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set consistent style parameters
        self.plot_style = {
            'figsize': (12, 6),
            'title_fontsize': 14,
            'label_fontsize': 12,
            'legend_fontsize': 10
        }
        
    def save_plot(self, fig, filename: str):
        """
        Save plot with consistent formatting and both PNG and interactive HTML formats.
        """
        # Save static version
        fig.savefig(self.output_dir / f"{filename}.png", 
                   bbox_inches='tight', dpi=300)
        plt.close(fig)

    def plot_price_sentiment_correlation(self, 
                                      price_data: pd.DataFrame, 
                                      sentiment_data: pd.DataFrame):
        """
        Create price vs sentiment correlation plot as described in Section V.D
        of the paper.
        """
        # Merge price and sentiment data
        merged_data = pd.merge(price_data, sentiment_data, on='Date')
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Price and sentiment over time
        ax1.plot(merged_data['Date'], merged_data['Close'], 
                label='Price', color='blue')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(merged_data['Date'], merged_data['sentiment_score'], 
                     label='Sentiment', color='red', alpha=0.7)
        
        ax1.set_title('Price and Sentiment Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (USD)', color='blue')
        ax1_twin.set_ylabel('Sentiment Score', color='red')
        
        # Correlation scatter plot
        sns.scatterplot(data=merged_data, x='sentiment_score', y='Close', ax=ax2)
        ax2.set_title('Price vs Sentiment Correlation')
        
        # Calculate and display correlation coefficient
        corr = merged_data['Close'].corr(merged_data['sentiment_score'])
        ax2.text(0.05, 0.95, f'Correlation: {corr:.2f}', 
                transform=ax2.transAxes)
        
        plt.tight_layout()
        self.save_plot(fig, 'price_sentiment_correlation')

    def plot_sentiment_distribution(self, sentiment_data: pd.DataFrame):
        """
        Visualize the distribution of sentiment scores and their evolution
        over time.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Sentiment distribution
        sns.histplot(sentiment_data['sentiment_score'], kde=True, ax=ax1)
        ax1.set_title('Distribution of Sentiment Scores')
        ax1.set_xlabel('Sentiment Score')
        ax1.set_ylabel('Frequency')
        
        # Rolling mean of sentiment
        window_size = 7  # 7-day rolling average
        rolling_mean = sentiment_data['sentiment_score'].rolling(window=window_size).mean()
        ax2.plot(sentiment_data['Date'], rolling_mean, label=f'{window_size}-day Moving Average')
        ax2.set_title('Sentiment Moving Average')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Sentiment Score')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        self.save_plot(fig, 'sentiment_distribution')

    def plot_prediction_analysis(self, 
                               true_values: np.array, 
                               predictions: np.array, 
                               dates: np.array):
        """
        Create comprehensive prediction analysis plots as specified in
        Section V.A of the paper.
        """
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        
        # Predicted vs Actual Values
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(dates, true_values, label='Actual', alpha=0.7)
        ax1.plot(dates, predictions, label='Predicted', alpha=0.7)
        ax1.set_title('Predicted vs Actual Values')
        ax1.legend()
        plt.xticks(rotation=45)
        
        # Prediction Error Distribution
        ax2 = fig.add_subplot(gs[1, 0])
        errors = predictions - true_values
        sns.histplot(errors, kde=True, ax=ax2)
        ax2.set_title('Prediction Error Distribution')
        
        # Scatter plot of Predicted vs Actual
        ax3 = fig.add_subplot(gs[1, 1])
        sns.scatterplot(x=true_values, y=predictions, alpha=0.5, ax=ax3)
        ax3.plot([min(true_values), max(true_values)], 
                [min(true_values), max(true_values)], 
                'r--', label='Perfect Prediction')
        ax3.set_title('Predicted vs Actual Scatter Plot')
        ax3.set_xlabel('Actual Values')
        ax3.set_ylabel('Predicted Values')
        ax3.legend()
        
        plt.tight_layout()
        self.save_plot(fig, 'prediction_analysis')

    def plot_trading_performance(self, 
                               price_data: pd.DataFrame, 
                               trading_signals: np.array):
        """
        Visualize trading strategy performance as described in Section V.C
        of the paper.
        """
        # Calculate cumulative returns
        strategy_returns = np.cumprod(1 + trading_signals * price_data['returns'])
        buy_hold_returns = np.cumprod(1 + price_data['returns'])
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Trading signals and price
        ax1.plot(price_data['Date'], price_data['Close'], 
                label='Price', color='blue')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(price_data['Date'], trading_signals, 
                     label='Trading Signal', color='red', alpha=0.5)
        ax1.set_title('Trading Signals and Price')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (USD)', color='blue')
        ax1_twin.set_ylabel('Trading Signal', color='red')
        
        # Cumulative returns comparison
        ax2.plot(price_data['Date'], strategy_returns, 
                label='Strategy Returns', color='green')
        ax2.plot(price_data['Date'], buy_hold_returns, 
                label='Buy & Hold Returns', color='blue', alpha=0.7)
        ax2.set_title('Cumulative Returns Comparison')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Cumulative Returns')
        ax2.legend()
        
        plt.tight_layout()
        self.save_plot(fig, 'trading_performance')

    def create_interactive_dashboard(self, 
                                  price_data: pd.DataFrame, 
                                  sentiment_data: pd.DataFrame, 
                                  predictions: np.array):
        """
        Create an interactive dashboard combining all key visualizations
        using Plotly.
        """
        # Merge all data
        dashboard_data = pd.merge(price_data, sentiment_data, on='Date')
        dashboard_data['predictions'] = predictions
        
        # Create interactive figure
        fig = go.Figure()
        
        # Add price trace
        fig.add_trace(go.Scatter(
            x=dashboard_data['Date'],
            y=dashboard_data['Close'],
            name='Price',
            yaxis='y1'
        ))
        
        # Add sentiment trace
        fig.add_trace(go.Scatter(
            x=dashboard_data['Date'],
            y=dashboard_data['sentiment_score'],
            name='Sentiment',
            yaxis='y2'
        ))
        
        # Add predictions trace
        fig.add_trace(go.Scatter(
            x=dashboard_data['Date'],
            y=dashboard_data['predictions'],
            name='Predictions',
            yaxis='y1',
            line=dict(dash='dot')
        ))
        
        # Update layout
        fig.update_layout(
            title='Cryptocurrency Analysis Dashboard',
            yaxis=dict(title='Price (USD)', side='left'),
            yaxis2=dict(title='Sentiment Score', 
                       side='right', 
                       overlaying='y'),
            hovermode='x unified'
        )
        
        # Save interactive dashboard
        fig.write_html(self.output_dir / 'interactive_dashboard.html')

def main():
    """
    Main function to demonstrate the visualization capabilities.
    """
    logger.info("Loading data...")
    
    # Load necessary data
    price_data = pd.read_csv('merged_data.csv')
    sentiment_data = pd.read_csv('processed_tweets.csv')
    predictions = np.load('model_predictions.npy')
    
    # Initialize visualizer
    visualizer = CryptoVisualizer()
    
    logger.info("Creating visualizations...")
    
    # Create all visualizations
    visualizer.plot_price_sentiment_correlation(price_data, sentiment_data)
    visualizer.plot_sentiment_distribution(sentiment_data)
    visualizer.plot_prediction_analysis(
        price_data['Close'].values,
        predictions,
        price_data['Date'].values
    )
    
    # Create trading signals (example)
    trading_signals = np.sign(np.diff(predictions, prepend=predictions[0]))
    visualizer.plot_trading_performance(price_data, trading_signals)
    
    # Create interactive dashboard
    visualizer.create_interactive_dashboard(
        price_data,
        sentiment_data,
        predictions
    )
    
    logger.info("Visualizations complete! Check the 'visualizations' directory.")

if __name__ == "__main__":
    main()