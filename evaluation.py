import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging
from pathlib import Path
from torch.utils.data import DataLoader
import json
from datetime import datetime
from scipy import stats

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
   def __init__(self, 
                model,
                config: Dict,
                device: torch.device,
                test_loader: DataLoader):
       self.model = model.to(device)
       self.device = device
       self.config = config
       self.test_loader = test_loader
       self.results_dir = Path(config['output_directory']) / "evaluation_results"
       self.results_dir.mkdir(exist_ok=True, parents=True)

   def calculate_basic_metrics(self, 
                             true_values: np.array, 
                             predictions: np.array) -> Dict:
       """Calculate standard evaluation metrics"""
       mse = mean_squared_error(true_values, predictions)
       rmse = np.sqrt(mse)
       mape = mean_absolute_percentage_error(true_values, predictions) * 100
       
       return {
           'MSE': mse,
           'RMSE': rmse,
           'MAPE': mape
       }

   def calculate_directional_accuracy(self, 
                                   true_values: np.array, 
                                   predictions: np.array) -> float:
       """Calculate directional accuracy of predictions"""
       direction_true = np.sign(np.diff(true_values))
       direction_pred = np.sign(np.diff(predictions))
       return np.mean(direction_true == direction_pred) * 100

   def calculate_trading_metrics(self, 
                               true_values: np.array, 
                               predictions: np.array,
                               config: Dict) -> Dict:
       """Calculate trading strategy performance metrics"""
       position_sizes = self._calculate_position_sizes(predictions, config)
       returns = self._calculate_returns(true_values, position_sizes, config)
       
       metrics = {
           'Total_Return': float(np.prod(1 + returns) - 1),
           'Annual_Return': float(np.mean(returns) * 252),
           'Sharpe_Ratio': float(self._calculate_sharpe_ratio(returns)),
           'Max_Drawdown': float(self._calculate_max_drawdown(returns)),
           'Win_Rate': float(np.mean(returns > 0) * 100)
       }
       
       return metrics

   def _calculate_position_sizes(self, 
                               predictions: np.array, 
                               config: Dict) -> np.array:
       """Calculate trading position sizes based on predictions"""
       confidence = np.abs(predictions)
       max_pos = config['trading']['position_sizing']['max_position_size']
       min_pos = config['trading']['position_sizing']['min_position_size']
       
       position_sizes = np.clip(confidence, min_pos, max_pos)
       position_sizes *= np.sign(predictions)
       
       return position_sizes

   def _calculate_returns(self, 
                        true_values: np.array,
                        position_sizes: np.array,
                        config: Dict) -> np.array:
       """Calculate trading returns including transaction costs"""
       price_returns = np.diff(true_values) / true_values[:-1]
       position_changes = np.diff(position_sizes) != 0
       transaction_costs = config['trading']['transaction_fee'] * position_changes
       
       strategy_returns = position_sizes[:-1] * price_returns - transaction_costs
       return strategy_returns

   def _calculate_sharpe_ratio(self, 
                             returns: np.array, 
                             risk_free_rate: float = 0.0) -> float:
       """Calculate annualized Sharpe ratio"""
       excess_returns = returns - risk_free_rate
       if len(excess_returns) > 1:
           return np.sqrt(252) * (np.mean(excess_returns) / np.std(excess_returns))
       return 0.0

   def _calculate_max_drawdown(self, returns: np.array) -> float:
       """Calculate maximum drawdown"""
       cumulative = np.cumprod(1 + returns)
       running_max = np.maximum.accumulate(cumulative)
       drawdowns = (cumulative - running_max) / running_max
       return np.min(drawdowns)

   def evaluate_model(self) -> Dict:
       """Perform complete model evaluation"""
       self.model.eval()
       predictions = []
       true_values = []
       
       with torch.no_grad():
           for batch in self.test_loader:
               # Process batch
               input_ids = batch['input_ids'].to(self.device)
               attention_mask = batch['attention_mask'].to(self.device)
               numerical_features = batch['numerical_features'].to(self.device)
               targets = batch['target'].to(self.device)
               
               outputs = self.model(input_ids, attention_mask, numerical_features)
               
               predictions.extend(outputs.cpu().numpy())
               true_values.extend(targets.cpu().numpy())
       
       predictions = np.array(predictions)
       true_values = np.array(true_values)
       
       # Calculate all metrics
       basic_metrics = self.calculate_basic_metrics(true_values, predictions)
       directional_accuracy = self.calculate_directional_accuracy(
           true_values, predictions)
       trading_metrics = self.calculate_trading_metrics(
           true_values, predictions, self.config)
       
       # Combine all metrics
       metrics = {
           **basic_metrics,
           'Directional_Accuracy': directional_accuracy,
           **trading_metrics
       }
       
       # Save results
       self._save_results(predictions, true_values, metrics)
       
       # Create visualizations
       self._create_visualizations(predictions, true_values, metrics)
       
       return metrics

   def _save_results(self, 
                    predictions: np.array,
                    true_values: np.array,
                    metrics: Dict):
       """Save evaluation results"""
       # Save predictions
       pd.DataFrame({
           'Predicted': predictions.flatten(),
           'Actual': true_values.flatten()
       }).to_csv(self.results_dir / 'predictions.csv', index=False)
       
       # Save metrics
       with open(self.results_dir / 'metrics.json', 'w') as f:
           json.dump(metrics, f, indent=4)

   def _create_visualizations(self, 
                            predictions: np.array,
                            true_values: np.array,
                            metrics: Dict):
       """Create evaluation visualizations"""
       # 1. Price Comparison Plot
       plt.figure(figsize=(12, 6))
       plt.plot(true_values, label='Actual', alpha=0.7)
       plt.plot(predictions, label='Predicted', alpha=0.7)
       plt.title('Predicted vs Actual Price Movement')
       plt.xlabel('Time')
       plt.ylabel('Price')
       plt.legend()
       plt.savefig(self.results_dir / 'price_comparison.png')
       plt.close()
       
       # 2. Error Distribution
       errors = predictions - true_values
       plt.figure(figsize=(10, 6))
       sns.histplot(errors, kde=True)
       plt.title('Prediction Error Distribution')
       plt.xlabel('Error')
       plt.ylabel('Frequency')
       plt.savefig(self.results_dir / 'error_distribution.png')
       plt.close()
       
       # 3. Trading Performance
       returns = self._calculate_returns(
           true_values, 
           self._calculate_position_sizes(predictions, self.config),
           self.config
       )
       cumulative_returns = np.cumprod(1 + returns)
       
       plt.figure(figsize=(12, 6))
       plt.plot(cumulative_returns, label='Strategy Returns')
       plt.title('Cumulative Trading Returns')
       plt.xlabel('Time')
       plt.ylabel('Cumulative Returns')
       plt.legend()
       plt.savefig(self.results_dir / 'trading_returns.png')
       plt.close()

def main():
    """
    Main function to run model evaluation
    """
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize BERT and LSTM models
    from model_training import BertLSTM
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = BertLSTM(
        bert_model=bert_model,
        lstm_hidden_size=config['model']['lstm_hidden_size'],
        num_lstm_layers=config['model']['num_lstm_layers'],
        dropout_rate=config['model']['dropout_rate']
    )
    
    # Load best model weights
    model.load_state_dict(torch.load(
        Path(config['output_directory']) / 'best_model.pt'
    ))
    
    # Load test data
    test_data = pd.read_csv(
        Path(config['output_directory']) / 'processed_test_data.csv'
    )
    
    from model_training import CryptoDataset
    test_dataset = CryptoDataset(
        price_data=test_data,
        sentiment_data=test_data,
        sequence_length=config['model']['sequence_length'],
        bert_tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['model']['batch_size'],
        shuffle=False
    )
    
    # Initialize evaluator and run evaluation
    evaluator = ModelEvaluator(model, config, device, test_loader)
    metrics = evaluator.evaluate_model()
    
    logger.info("Evaluation metrics:")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
        
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()
