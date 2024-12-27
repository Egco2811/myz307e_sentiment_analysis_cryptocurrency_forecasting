import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging
from typing import Dict, List
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoVisualizer:
   def __init__(self, config: Dict):
       self.config = config
       self.output_dir = Path(config['output_directory']) / "visualizations"
       self.output_dir.mkdir(exist_ok=True, parents=True)
       
       # Set style configurations
       plt.style.use(config['visualization']['style'])
       self.colors = config['visualization']['color_scheme']
       self.dpi = config['visualization']['dpi']

   def save_plot(self, fig, filename: str):
       """Save plot in multiple formats with consistent settings"""
       for fmt in self.config['visualization']['plot_formats']:
           filepath = self.output_dir / f"{filename}.{fmt}"
           fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
       plt.close(fig)

   def plot_price_sentiment_correlation(self, df: pd.DataFrame):
       """Plot price vs sentiment correlation analysis"""
       fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
       
       # Price and sentiment over time
       ax1.plot(df['Date'], df['Close'], 
               label='Price', color=self.colors['price'])
       ax1_twin = ax1.twinx()
       ax1_twin.plot(df['Date'], df['sentiment_score'], 
                    label='Sentiment', color=self.colors['sentiment'])
       
       ax1.set_title('Price and Sentiment Over Time')
       ax1.set_xlabel('Date')
       ax1.set_ylabel('Price (USD)', color=self.colors['price'])
       ax1_twin.set_ylabel('Sentiment Score', color=self.colors['sentiment'])
       
       # Correlation scatter plot
       sns.scatterplot(data=df, x='sentiment_score', y='Close', ax=ax2)
       corr = df['Close'].corr(df['sentiment_score'])
       ax2.text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=ax2.transAxes)
       ax2.set_title('Price vs Sentiment Correlation')
       
       plt.tight_layout()
       self.save_plot(fig, 'price_sentiment_correlation')

   def plot_prediction_analysis(self, df: pd.DataFrame):
       """Create comprehensive prediction analysis plots"""
       fig = plt.figure(figsize=(15, 10))
       gs = fig.add_gridspec(2, 2)
       
       # Predicted vs Actual Values
       ax1 = fig.add_subplot(gs[0, :])
       ax1.plot(df['Date'], df['Close'], 
               label='Actual', color=self.colors['price'])
       ax1.plot(df['Date'], df['prediction'], 
               label='Predicted', color=self.colors['prediction'])
       ax1.set_title('Predicted vs Actual Values')
       ax1.legend()
       
       # Prediction Error Distribution
       ax2 = fig.add_subplot(gs[1, 0])
       errors = df['prediction'] - df['Close']
       sns.histplot(errors, kde=True, ax=ax2)
       ax2.set_title('Prediction Error Distribution')
       
       # Scatter plot
       ax3 = fig.add_subplot(gs[1, 1])
       sns.scatterplot(data=df, x='Close', y='prediction', alpha=0.5, ax=ax3)
       
       plt.tight_layout()
       self.save_plot(fig, 'prediction_analysis')

   def plot_trading_performance(self, results: Dict):
       """Visualize trading strategy performance"""
       fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
       
       # Equity curve
       ax1.plot(results['equity_curve'], 
               label='Portfolio Value', color=self.colors['price'])
       ax1.set_title('Portfolio Equity Curve')
       ax1.set_xlabel('Days')
       ax1.set_ylabel('Portfolio Value')
       
       # Drawdown analysis
       drawdowns = results['drawdowns']
       ax2.plot(drawdowns, label='Drawdown', color='red')
       ax2.set_title('Portfolio Drawdown')
       ax2.set_xlabel('Days')
       ax2.set_ylabel('Drawdown')
       
       plt.tight_layout()
       self.save_plot(fig, 'trading_performance')

   def create_interactive_dashboard(self, df: pd.DataFrame, results: Dict):
       """Create interactive dashboard using Plotly"""
       if not self.config['visualization']['generate_interactive']:
           return
           
       fig = go.Figure()
       
       # Add price trace
       fig.add_trace(go.Scatter(
           x=df['Date'],
           y=df['Close'],
           name='Price',
           yaxis='y1'
       ))
       
       # Add sentiment trace
       fig.add_trace(go.Scatter(
           x=df['Date'],
           y=df['sentiment_score'],
           name='Sentiment',
           yaxis='y2'
       ))
       
       # Add predictions trace
       fig.add_trace(go.Scatter(
           x=df['Date'],
           y=df['prediction'],
           name='Predictions',
           yaxis='y1',
           line=dict(dash='dot')
       ))
       
       # Update layout
       fig.update_layout(
           title='Cryptocurrency Analysis Dashboard',
           yaxis=dict(title='Price (USD)', side='left'),
           yaxis2=dict(title='Sentiment Score', side='right', overlaying='y'),
           hovermode='x unified'
       )
       
       fig.write_html(self.output_dir / 'interactive_dashboard.html')

   def plot_feature_importance(self, feature_importance: Dict):
       """Plot feature importance analysis"""
       plt.figure(figsize=(12, 6))
       features = list(feature_importance.keys())
       importances = list(feature_importance.values())
       
       sns.barplot(x=importances, y=features)
       plt.title('Feature Importance Analysis')
       plt.xlabel('Importance Score')
       
       self.save_plot(plt.gcf(), 'feature_importance')

   def create_all_visualizations(self, df: pd.DataFrame, results: Dict):
       """Create all visualizations"""
       logger.info("Creating visualizations...")
       
       self.plot_price_sentiment_correlation(df)
       self.plot_prediction_analysis(df)
       self.plot_trading_performance(results)
       self.create_interactive_dashboard(df, results)
       
       if 'feature_importance' in results:
           self.plot_feature_importance(results['feature_importance'])
           
       logger.info("Visualizations complete!")

def main():
   """Main execution function"""
   with open('config.json', 'r') as f:
       config = json.load(f)
       
   # Load data
   df = pd.read_csv(
       Path(config['output_directory']) / 'processed_results.csv',
       parse_dates=['Date']
   )
   
   with open(Path(config['output_directory']) / 'trading_results.json', 'r') as f:
       results = json.load(f)
   
   # Create visualizations
   visualizer = CryptoVisualizer(config)
   visualizer.create_all_visualizations(df, results)

if __name__ == "__main__":
   main()
