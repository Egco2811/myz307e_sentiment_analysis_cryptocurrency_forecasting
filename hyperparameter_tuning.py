import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import optuna
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import json
from datetime import datetime
import time
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
   def __init__(self, 
                train_data,
                val_data, 
                device: torch.device,
                config: Dict,
                alpha: float = 0.6,    # MSE weight
                beta: float = 0.3,     # Directional Accuracy weight
                gamma: float = 0.1):   # Computational Cost weight
       
       self.train_data = train_data
       self.val_data = val_data
       self.device = device
       self.config = config
       self.alpha = alpha
       self.beta = beta
       self.gamma = gamma
       
       self.output_dir = Path(config['output_directory']) / "hyperparameter_tuning"
       self.output_dir.mkdir(exist_ok=True, parents=True)

       # Define hyperparameter search spaces
       self.param_space = {
           'lstm_hidden_size': [64, 128, 256],
           'batch_size': (16, 64, 8),  # min, max, step
           'learning_rate': (1e-5, 1e-3),
           'dropout_rate': (0.1, 0.5),
           'num_lstm_layers': [2, 3, 4]
       }

       self.best_params = None
       self.best_score = float('inf')

   def calculate_directional_accuracy(self, true_values: np.array, predictions: np.array) -> float:
       true_direction = np.sign(np.diff(true_values))
       pred_direction = np.sign(np.diff(predictions))
       return np.mean(true_direction == pred_direction)

   def calculate_computational_cost(self, start_time: float) -> float:
       elapsed_time = time.time() - start_time
       max_allowed_time = 3600
       return min(elapsed_time / max_allowed_time, 1.0)

   def objective(self, trial: optuna.Trial) -> float:
       start_time = time.time()
       
       params = {
           'lstm_hidden_size': trial.suggest_categorical('lstm_hidden_size', 
                                                       self.param_space['lstm_hidden_size']),
           'batch_size': trial.suggest_int('batch_size', 
                                         self.param_space['batch_size'][0],
                                         self.param_space['batch_size'][1],
                                         step=self.param_space['batch_size'][2]),
           'learning_rate': trial.suggest_loguniform('learning_rate',
                                                   self.param_space['learning_rate'][0],
                                                   self.param_space['learning_rate'][1]),
           'dropout_rate': trial.suggest_float('dropout_rate',
                                             self.param_space['dropout_rate'][0],
                                             self.param_space['dropout_rate'][1]),
           'num_lstm_layers': trial.suggest_categorical('num_lstm_layers',
                                                      self.param_space['num_lstm_layers'])
       }

       train_loader = DataLoader(
           self.train_data,
           batch_size=params['batch_size'],
           shuffle=True
       )
       val_loader = DataLoader(
           self.val_data,
           batch_size=params['batch_size']
       )

       model = self.create_model(params)
       metrics = self.train_and_evaluate(model, train_loader, val_loader, params)
       
       computational_cost = self.calculate_computational_cost(start_time)
       
       objective_value = (
           self.alpha * metrics['mse'] + 
           self.beta * (1 - metrics['directional_accuracy']) + 
           self.gamma * computational_cost
       )

       trial.set_user_attr('metrics', metrics)
       trial.set_user_attr('computational_cost', computational_cost)
       
       return objective_value

   def create_model(self, params: Dict) -> nn.Module:
       from model_training import BertLSTM
       bert_model = BertModel.from_pretrained('bert-base-uncased')
       return BertLSTM(
           bert_model=bert_model,
           lstm_hidden_size=params['lstm_hidden_size'],
           num_lstm_layers=params['num_lstm_layers'],
           dropout_rate=params['dropout_rate']
       ).to(self.device)

   def train_and_evaluate(self, 
                         model: nn.Module, 
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         params: Dict) -> Dict:
       
       optimizer = torch.optim.AdamW(
           model.parameters(),
           lr=params['learning_rate']
       )
       criterion = nn.MSELoss()
       
       best_val_loss = float('inf')
       patience = 5
       patience_counter = 0
       
       for epoch in range(50):
           model.train()
           train_losses = []
           
           for batch in train_loader:
               optimizer.zero_grad()
               
               # Move batch to device and get predictions
               predictions = self.process_batch(model, batch)
               loss = criterion(predictions, batch['target'].to(self.device))
               
               loss.backward()
               torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
               optimizer.step()
               
               train_losses.append(loss.item())

           # Validation
           val_metrics = self.validate_model(model, val_loader)
           
           if val_metrics['mse'] < best_val_loss:
               best_val_loss = val_metrics['mse']
               patience_counter = 0
           else:
               patience_counter += 1
               
           if patience_counter >= patience:
               break

       return val_metrics

   def process_batch(self, model: nn.Module, batch: Dict) -> torch.Tensor:
       return model(
           batch['input_ids'].to(self.device),
           batch['attention_mask'].to(self.device),
           batch['numerical_features'].to(self.device)
       )

   def validate_model(self, model: nn.Module, val_loader: DataLoader) -> Dict:
       model.eval()
       criterion = nn.MSELoss()
       val_losses = []
       predictions = []
       true_values = []
       
       with torch.no_grad():
           for batch in val_loader:
               predictions_batch = self.process_batch(model, batch)
               targets = batch['target'].to(self.device)
               
               loss = criterion(predictions_batch, targets)
               val_losses.append(loss.item())
               
               predictions.extend(predictions_batch.cpu().numpy())
               true_values.extend(targets.cpu().numpy())

       predictions = np.array(predictions)
       true_values = np.array(true_values)
       
       return {
           'mse': np.mean(val_losses),
           'directional_accuracy': self.calculate_directional_accuracy(true_values, predictions)
       }

   def optimize(self, n_trials: int = 100) -> Dict:
       study = optuna.create_study(
           direction="minimize",
           sampler=optuna.samplers.TPESampler(seed=42),
           pruner=optuna.pruners.MedianPruner()
       )
       
       study.optimize(self.objective, n_trials=n_trials)
       
       self.best_params = study.best_params
       self.best_score = study.best_value
       
       self.save_results(study)
       
       return self.best_params

   def save_results(self, study: optuna.study.Study):
       results = {
           'best_params': self.best_params,
           'best_value': float(self.best_score),
           'n_trials': len(study.trials),
           'optimization_weights': {
               'alpha': self.alpha,
               'beta': self.beta,
               'gamma': self.gamma
           }
       }
       
       with open(self.output_dir / 'optimization_results.json', 'w') as f:
           json.dump(results, f, indent=4)
           
       self.create_visualizations(study)

   def create_visualizations(self, study: optuna.study.Study):
       # Optimization history
       plt.figure(figsize=(10, 6))
       optuna.visualization.matplotlib.plot_optimization_history(study)
       plt.savefig(self.output_dir / 'optimization_history.png')
       plt.close()
       
       # Parameter importance
       plt.figure(figsize=(10, 6))
       optuna.visualization.matplotlib.plot_param_importances(study)
       plt.savefig(self.output_dir / 'parameter_importance.png')
       plt.close()
       
       # Parallel coordinate plot
       plt.figure(figsize=(15, 8))
       optuna.visualization.matplotlib.plot_parallel_coordinate(study)
       plt.savefig(self.output_dir / 'parallel_coordinate.png')
       plt.close()

def main():
   with open('config.json', 'r') as f:
       config = json.load(f)
   
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
   from model_training import CryptoDataset
   train_data = CryptoDataset(config['train_data_path'])
   val_data = CryptoDataset(config['val_data_path'])
   
   optimizer = HyperparameterOptimizer(
       train_data=train_data,
       val_data=val_data,
       device=device,
       config=config
   )
   
   best_params = optimizer.optimize(n_trials=config['hyperparameter_tuning']['n_trials'])
   logger.info(f"Best parameters found: {best_params}")

if __name__ == "__main__":
   main()
    
