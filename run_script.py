#!/usr/bin/env python3
"""
Main execution script for BERT-LSTM Cryptocurrency Price Prediction Project.
This script orchestrates the entire pipeline from data preprocessing through 
model training and trading strategy evaluation.
"""

import os
import sys
import logging
import time
from pathlib import Path
import json
from datetime import datetime
import argparse
import torch
import pandas as pd
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Import our custom modules
from tweets_preprocessing import TweetPreprocessor
from tweet_labeler import SentimentLabeler
from csv_date_matcher import DataMatcher
from model_training import BertLSTM, CryptoDataset
from evaluation import ModelEvaluator
from data_visualization import CryptoVisualizer
from feature_engineering import CryptoFeatureEngineer
from hyperparameter_tuning import HyperparameterOptimizer
from trading_strategy import CryptoTradingStrategy

# Configure logging
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
    """
    Orchestrates the execution of the entire cryptocurrency analysis pipeline.
    """
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize pipeline with configuration settings.
        """
        self.start_time = time.time()
        self.config = self._load_config(config_path)
        self.output_dir = Path(self.config['output_directory'])
        self.output_dir.mkdir(exist_ok=True)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration settings from JSON file.
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)

    def _check_dependencies(self) -> None:
        """
        Verify all required data files and dependencies exist.
        """
        required_files = [
            self.config['raw_data']['tweets_file'],
            self.config['raw_data']['price_file']
        ]
        
        for file_path in required_files:
            if not Path(file_path).exists():
                logger.error(f"Required file not found: {file_path}")
                sys.exit(1)

    def run_pipeline(self) -> None:
        """
        Execute the complete analysis pipeline.
        """
        try:
            logger.info("Starting pipeline execution...")
            self._check_dependencies()

            # 1. Data Preprocessing
            logger.info("Step 1: Data Preprocessing")
            preprocessor = TweetPreprocessor()
            processed_tweets = preprocessor.process_tweets_file(
                self.config['raw_data']['tweets_file'],
                self.output_dir / 'processed_tweets.csv'
            )

            # 2. Sentiment Analysis
            logger.info("Step 2: Sentiment Analysis")
            sentiment_labeler = SentimentLabeler()
            labeled_tweets = sentiment_labeler.process_file(
                self.output_dir / 'processed_tweets.csv'
            )

            # 3. Data Matching and Alignment
            logger.info("Step 3: Data Matching")
            data_matcher = DataMatcher(
                self.config['raw_data']['price_file'],
                self.output_dir / 'processed_tweets.csv'
            )
            merged_data = data_matcher.process_data()

            # 4. Feature Engineering
            logger.info("Step 4: Feature Engineering")
            feature_engineer = CryptoFeatureEngineer()
            engineered_data = feature_engineer.create_all_features(
                merged_data['price_data'],
                merged_data['sentiment_data']
            )

            # 5. Model Training and Hyperparameter Tuning
            logger.info("Step 5: Model Training")
            train_data = CryptoDataset(engineered_data['train'])
            val_data = CryptoDataset(engineered_data['val'])
            
            # Optimize hyperparameters
            optimizer = HyperparameterOptimizer(
                train_data=train_data,
                val_data=val_data,
                device=self.device
            )
            best_params = optimizer.optimize_hyperparameters()

            # Train model with best parameters
            model = BertLSTM(**best_params).to(self.device)
            # Training code here...

            # Create test dataset and loader
            test_data = CryptoDataset(engineered_data['test'])
            test_loader = torch.utils.data.DataLoader(
                test_data, 
                batch_size=self.config['training']['batch_size']
            )

            # 6. Model Evaluation
            logger.info("Step 6: Model Evaluation")
            evaluator = ModelEvaluator(model, self.device, test_loader)
            evaluation_results = evaluator.evaluate_model()
            model_predictions = evaluation_results['predictions']  # Get predictions from evaluation results

            # 7. Trading Strategy Implementation
            logger.info("Step 7: Trading Strategy")
            strategy = CryptoTradingStrategy(
                initial_capital=self.config['trading']['initial_capital']
            )
            trading_results = strategy.run_backtest(
                prices=test_data['prices'],
                dates=test_data['dates'],
                predictions=model_predictions,
                sentiment_scores=labeled_tweets['sentiment_scores']
            )

            # 8. Visualization
            logger.info("Step 8: Generating Visualizations")
            visualizer = CryptoVisualizer(self.output_dir)
            visualizer.create_all_visualizations(
                price_data=merged_data['price_data'],
                sentiment_data=merged_data['sentiment_data'],
                predictions=model_predictions,
                trading_results=trading_results
            )

            # Save results
            self._save_results(evaluation_results, trading_results)
            
            execution_time = time.time() - self.start_time
            logger.info(f"Pipeline execution completed in {execution_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise

    def _save_results(self, evaluation_results: dict, trading_results: dict) -> None:
        """
        Save all results to output directory.
        """
        results = {
            'evaluation': evaluation_results,
            'trading': trading_results,
            'execution_time': time.time() - self.start_time,
            'config': self.config,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=4)

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Cryptocurrency Price Prediction Pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Path to configuration file'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    executor = PipelineExecutor(args.config)
    executor.run_pipeline()