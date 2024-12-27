#!/usr/bin/env python3
"""
Main execution script for BERT-LSTM Cryptocurrency Price Prediction Project.
This script orchestrates the entire pipeline from data preprocessing through 
model training and trading strategy evaluation.
"""

import os
import logging
import json
from pathlib import Path
import torch
from datetime import datetime
import pandas as pd
from typing import Dict, Optional, Tuple

from tweets_preprocessing import TweetPreprocessor
from tweet_labeler import SentimentAnalyzer  
from csv_date_matcher import DataMatcher
from model_training import BertLSTM, CryptoDataset
from hyperparameter_tuning import HyperparameterOptimizer
from evaluation import ModelEvaluator
from data_visualization import CryptoVisualizer
from trading_strategy import CryptoTradingStrategy

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s',
   handlers=[
       logging.FileHandler('execution.log'),
       logging.StreamHandler()
   ]
)
logger = logging.getLogger(__name__)

class PipelineExecutor:
   def __init__(self, config_path: str = "config.json"):
       self.config = self._load_config(config_path)
       self.output_dir = Path(self.config['output_directory'])
       self.output_dir.mkdir(exist_ok=True, parents=True)
       self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       logger.info(f"Using device: {self.device}")

   def _load_config(self, config_path: str) -> Dict:
       try:
           with open(config_path, 'r') as f:
               config = json.load(f)
           self._validate_config(config)
           return config
       except FileNotFoundError:
           logger.error(f"Configuration file not found: {config_path}")
           raise
       except json.JSONDecodeError:
           logger.error("Invalid JSON in configuration file")
           raise

   def _validate_config(self, config: Dict):
       required_sections = [
           'raw_data', 'preprocessing', 'model', 'hyperparameter_tuning',
           'feature_engineering', 'trading', 'evaluation', 'visualization'
       ]
       
       for section in required_sections:
           if section not in config:
               raise ValueError(f"Missing required config section: {section}")

   def process_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
       logger.info("Starting data preprocessing...")
       
       # Preprocess tweets
       tweet_processor = TweetPreprocessor(self.config)
       processed_tweets = tweet_processor.process_tweets_file(
           self.config['raw_data']['tweets_file'],
           self.output_dir / 'processed_tweets.csv'
       )
       
       # Analyze sentiment
       sentiment_analyzer = SentimentAnalyzer(self.config)
       sentiment_data = sentiment_analyzer.analyze_tweets(processed_tweets)
       
       # Align and merge data
       data_matcher = DataMatcher(self.config)
       price_data, sentiment_data = data_matcher.process_data()
       
       return price_data, sentiment_data

   def train_model(self, train_data, val_data) -> BertLSTM:
       logger.info("Starting model training...")
       
       # Hyperparameter optimization
       optimizer = HyperparameterOptimizer(
           train_data=train_data,
           val_data=val_data,
           device=self.device,
           config=self.config
       )
       
       best_params = optimizer.optimize(
           n_trials=self.config['hyperparameter_tuning']['n_trials']
       )
       
       # Train final model with best parameters
       model = BertLSTM(
           bert_model=self.bert_model,
           **best_params
       ).to(self.device)
       
       return model, best_params

   def evaluate_model(self, 
                     model: BertLSTM, 
                     test_data,
                     price_data: pd.DataFrame,
                     sentiment_data: pd.DataFrame):
       logger.info("Starting model evaluation...")
       
       # Model evaluation
       evaluator = ModelEvaluator(
           model=model,
           config=self.config,
           device=self.device,
           test_loader=test_data
       )
       evaluation_results = evaluator.evaluate_model()
       
       # Trading strategy evaluation
       strategy = CryptoTradingStrategy(self.config)
       trading_results = strategy.run_backtest(
           price_data=price_data,
           sentiment_data=sentiment_data,
           predictions=evaluation_results['predictions']
       )
       
       # Combine results
       results = {
           'model_evaluation': evaluation_results,
           'trading_performance': trading_results
       }
       
       self._save_results(results)
       return results

   def create_visualizations(self, 
                           data: pd.DataFrame,
                           results: Dict):
       logger.info("Creating visualizations...")
       
       visualizer = CryptoVisualizer(self.config)
       visualizer.create_all_visualizations(data, results)

   def _save_results(self, results: Dict):
       results_path = self.output_dir / 'results.json'
       with open(results_path, 'w') as f:
           json.dump(results, f, indent=4)

   def run_pipeline(self):
       try:
           # Process data
           price_data, sentiment_data = self.process_data()
           
           # Create datasets
           train_data, val_data, test_data = self._create_datasets(
               price_data, sentiment_data
           )
           
           # Train model
           model, best_params = self.train_model(train_data, val_data)
           
           # Evaluate model and trading strategy
           results = self.evaluate_model(
               model, test_data, price_data, sentiment_data
           )
           
           # Create visualizations
           self.create_visualizations(price_data, results)
           
           logger.info("Pipeline execution completed successfully!")
           return results
           
       except Exception as e:
           logger.error(f"Pipeline execution failed: {str(e)}")
           raise

   def _create_datasets(self, 
                       price_data: pd.DataFrame,
                       sentiment_data: pd.DataFrame):
       dataset = CryptoDataset(
           price_data=price_data,
           sentiment_data=sentiment_data,
           sequence_length=self.config['model']['sequence_length'],
           bert_tokenizer=self.bert_tokenizer
       )
       
       # Split data
       train_size = int(len(dataset) * 0.8)
       val_size = int(len(dataset) * 0.1)
       test_size = len(dataset) - train_size - val_size
       
       train_data, val_data, test_data = torch.utils.data.random_split(
           dataset, [train_size, val_size, test_size]
       )
       
       return train_data, val_data, test_data

def main():
   import argparse
   parser = argparse.ArgumentParser(
       description='Run Cryptocurrency Analysis Pipeline'
   )
   parser.add_argument(
       '--config', 
       type=str,
       default='config.json',
       help='Path to configuration file'
   )
   args = parser.parse_args()
   
   executor = PipelineExecutor(args.config)
   executor.run_pipeline()

if __name__ == "__main__":
   main()
