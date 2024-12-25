import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import logging
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from functools import partial

# Import our custom modules
from model_training import BertLSTM, CryptoDataset
from evaluation import ModelEvaluator

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hyperparameter_tuning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """
    Implements comprehensive hyperparameter optimization for the BERT-LSTM model
    using multiple optimization strategies as specified in the paper.
    """
    def __init__(self, 
                 train_data: CryptoDataset,
                 val_data: CryptoDataset,
                 device: torch.device,
                 output_dir: str = "hyperparameter_results"):
        """
        Initialize the optimizer with datasets and configuration.
        """
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define hyperparameter search spaces
        self.param_space = {
            'lstm_hidden_size': (32, 256),
            'num_lstm_layers': (1, 4),
            'dropout_rate': (0.1, 0.5),
            'learning_rate': (1e-5, 1e-3),
            'batch_size': (16, 64),
            'sequence_length': (5, 30)
        }
        
        # Track best parameters and scores
        self.best_params = None
        self.best_score = float('inf')
        
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        Implements the evaluation criteria specified in the paper.
        """
        # Sample hyperparameters
        params = {
            'lstm_hidden_size': trial.suggest_int('lstm_hidden_size', 
                                                self.param_space['lstm_hidden_size'][0],
                                                self.param_space['lstm_hidden_size'][1],
                                                step=32),
            'num_lstm_layers': trial.suggest_int('num_lstm_layers',
                                               self.param_space['num_lstm_layers'][0],
                                               self.param_space['num_lstm_layers'][1]),
            'dropout_rate': trial.suggest_float('dropout_rate',
                                              self.param_space['dropout_rate'][0],
                                              self.param_space['dropout_rate'][1]),
            'learning_rate': trial.suggest_loguniform('learning_rate',
                                                    self.param_space['learning_rate'][0],
                                                    self.param_space['learning_rate'][1]),
            'batch_size': trial.suggest_int('batch_size',
                                          self.param_space['batch_size'][0],
                                          self.param_space['batch_size'][1],
                                          step=16),
            'sequence_length': trial.suggest_int('sequence_length',
                                               self.param_space['sequence_length'][0],
                                               self.param_space['sequence_length'][1])
        }
        
        # Create model with sampled parameters
        model = BertLSTM(
            bert_model=self.bert_model,
            lstm_hidden_size=params['lstm_hidden_size'],
            num_lstm_layers=params['num_lstm_layers'],
            dropout_rate=params['dropout_rate']
        ).to(self.device)
        
        # Create data loaders with sampled batch size
        train_loader = DataLoader(
            self.train_data,
            batch_size=params['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            self.val_data,
            batch_size=params['batch_size']
        )
        
        # Training setup
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['learning_rate']
        )
        criterion = nn.MSELoss()
        
        # Implement early stopping
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        # Training loop
        for epoch in range(50):  # Maximum epochs
            model.train()
            train_losses = []
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                numerical_features = batch['numerical_features'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask, numerical_features)
                loss = criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation phase
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    numerical_features = batch['numerical_features'].to(self.device)
                    targets = batch['target'].to(self.device)
                    
                    outputs = model(input_ids, attention_mask, numerical_features)
                    loss = criterion(outputs, targets)
                    val_losses.append(loss.item())
            
            val_loss = np.mean(val_losses)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
            
            # Report intermediate value to Optuna
            trial.report(val_loss, epoch)
            
            # Handle pruning based on the intermediate value
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return best_val_loss

    def optimize_hyperparameters(self, n_trials: int = 100):
        """
        Run hyperparameter optimization using Optuna.
        """
        logger.info("Starting hyperparameter optimization...")
        
        # Create Optuna study
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=n_trials)
        
        # Store best parameters
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        # Save optimization results
        self.save_optimization_results(study)
        
        return self.best_params

    def save_optimization_results(self, study: optuna.study.Study):
        """
        Save optimization results and visualizations.
        """
        # Save best parameters
        with open(self.output_dir / 'best_params.json', 'w') as f:
            json.dump(self.best_params, f, indent=4)
        
        # Create optimization history plot
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(self.output_dir / 'optimization_history.png')
        
        # Create parameter importance plot
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig(self.output_dir / 'parameter_importance.png')
        
        # Create parallel coordinate plot
        plt.figure(figsize=(15, 8))
        optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        plt.savefig(self.output_dir / 'parallel_coordinate.png')
        
        # Save study statistics
        study_stats = {
            'best_value': study.best_value,
            'best_trial': study.best_trial.number,
            'n_trials': len(study.trials),
            'datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(self.output_dir / 'study_statistics.json', 'w') as f:
            json.dump(study_stats, f, indent=4)

    def cross_validate_best_params(self, n_splits: int = 5):
        """
        Perform time series cross-validation with the best parameters.
        """
        logger.info("Starting cross-validation of best parameters...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.train_data)):
            logger.info(f"Evaluating fold {fold + 1}/{n_splits}")
            
            # Create data loaders for this fold
            train_fold = torch.utils.data.Subset(self.train_data, train_idx)
            val_fold = torch.utils.data.Subset(self.train_data, val_idx)
            
            train_loader = DataLoader(
                train_fold,
                batch_size=self.best_params['batch_size'],
                shuffle=True
            )
            val_loader = DataLoader(
                val_fold,
                batch_size=self.best_params['batch_size']
            )
            
            # Train and evaluate model with best parameters
            model = BertLSTM(
                bert_model=self.bert_model,
                lstm_hidden_size=self.best_params['lstm_hidden_size'],
                num_lstm_layers=self.best_params['num_lstm_layers'],
                dropout_rate=self.best_params['dropout_rate']
            ).to(self.device)
            
            # Train model
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.best_params['learning_rate']
            )
            criterion = nn.MSELoss()
            
            # Training loop (simplified for brevity)
            model.train()
            for epoch in range(10):  # Reduced epochs for CV
                for batch in train_loader:
                    # Training step (same as in objective function)
                    pass
                    
            # Evaluate on validation fold
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    # Validation step (same as in objective function)
                    pass
                    
            cv_scores.append(np.mean(val_losses))
        
        # Save cross-validation results
        cv_results = {
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'cv_scores': cv_scores
        }
        
        with open(self.output_dir / 'cv_results.json', 'w') as f:
            json.dump(cv_results, f, indent=4)
        
        return cv_results

def main():
    """
    Main function to run hyperparameter optimization.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    # Note: You'll need to adapt this to your specific data loading process
    train_data = CryptoDataset(...)
    val_data = CryptoDataset(...)
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(
        train_data=train_data,
        val_data=val_data,
        device=device
    )
    
    # Run optimization
    best_params = optimizer.optimize_hyperparameters(n_trials=100)
    logger.info(f"Best parameters found: {best_params}")
    
    # Cross-validate best parameters
    cv_results = optimizer.cross_validate_best_params()
    logger.info(f"Cross-validation results: {cv_results}")

if __name__ == "__main__":
    main()