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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Implements comprehensive evaluation metrics for the BERT-LSTM model as specified
    in Section V of the paper. Handles both price prediction accuracy and trading
    performance evaluation.
    """
    def __init__(self, model, device, test_loader: DataLoader):
        self.model = model.to(device)
        self.device = device
        self.test_loader = test_loader
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def calculate_metrics(self, true_values: np.array, predictions: np.array) -> Dict:
        """
        Calculate all evaluation metrics specified in the paper's Section V.D
        """
        # Calculate basic metrics
        mse = mean_squared_error(true_values, predictions)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(true_values, predictions) * 100
        
        # Calculate directional accuracy
        direction_true = np.sign(np.diff(true_values))
        direction_pred = np.sign(np.diff(predictions))
        directional_accuracy = np.mean(direction_true == direction_pred) * 100
        
        # Calculate trading metrics
        returns = self.calculate_trading_returns(true_values, predictions)
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        max_drawdown = self.calculate_max_drawdown(returns)
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Returns': float(returns[-1])  # Cumulative return
        }
        
        return metrics

    def calculate_trading_returns(self, true_values: np.array, 
                                predictions: np.array) -> np.array:
        """
        Calculate returns based on the trading strategy described in the paper
        """
        position = np.zeros(len(predictions))
        # Take long position when predicted return is positive
        position[:-1] = np.where(np.diff(predictions) > 0, 1, -1)
        
        # Calculate strategy returns
        asset_returns = np.diff(true_values) / true_values[:-1]
        strategy_returns = position[:-1] * asset_returns
        
        # Apply transaction costs as mentioned in the paper (0.1%)
        trades = np.diff(position) != 0
        transaction_costs = 0.001 * trades  # 0.1% per trade
        strategy_returns = strategy_returns - transaction_costs
        
        return np.cumprod(1 + strategy_returns)

    def calculate_sharpe_ratio(self, returns: np.array, risk_free_rate=0.0) -> float:
        """
        Calculate the Sharpe ratio of the trading strategy
        """
        excess_returns = returns - risk_free_rate
        if len(excess_returns) > 1:
            return np.sqrt(252) * (np.mean(excess_returns) / np.std(excess_returns))
        return 0.0

    def calculate_max_drawdown(self, returns: np.array) -> float:
        """
        Calculate the maximum drawdown of the trading strategy
        """
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return np.min(drawdowns)

    def evaluate_model(self) -> Dict:
        """
        Perform full model evaluation including predictions and metric calculations
        """
        self.model.eval()
        predictions = []
        true_values = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                numerical_features = batch['numerical_features'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Get predictions
                outputs = self.model(input_ids, attention_mask, numerical_features)
                
                predictions.extend(outputs.cpu().numpy())
                true_values.extend(targets.cpu().numpy())
        
        predictions = np.array(predictions)
        true_values = np.array(true_values)
        
        # Calculate all metrics
        metrics = self.calculate_metrics(true_values, predictions)
        
        # Save predictions and true values
        self.save_predictions(predictions, true_values)
        
        # Generate visualizations
        self.create_visualizations(predictions, true_values)
        
        return metrics

    def create_visualizations(self, predictions: np.array, true_values: np.array):
        """
        Create all visualizations specified in the paper
        """
        # Set style for plots
        plt.style.use('seaborn')
        
        # 1. Predicted vs Actual Prices
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
        returns = self.calculate_trading_returns(true_values, predictions)
        plt.figure(figsize=(12, 6))
        plt.plot(returns, label='Strategy Returns')
        plt.title('Cumulative Trading Returns')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.savefig(self.results_dir / 'trading_returns.png')
        plt.close()

    def save_predictions(self, predictions: np.array, true_values: np.array):
        """
        Save predictions and true values for further analysis
        """
        results_df = pd.DataFrame({
            'Predicted': predictions.flatten(),
            'Actual': true_values.flatten()
        })
        results_df.to_csv(self.results_dir / 'predictions.csv', index=False)

    def perform_ablation_study(self, model_variants: Dict) -> Dict:
        """
        Perform ablation study as described in the paper's Section V.B
        """
        ablation_results = {}
        
        for variant_name, model in model_variants.items():
            logger.info(f"Evaluating model variant: {variant_name}")
            model = model.to(self.device)
            metrics = self.evaluate_model()
            ablation_results[variant_name] = metrics
            
        # Save ablation results
        with open(self.results_dir / 'ablation_results.json', 'w') as f:
            json.dump(ablation_results, f, indent=4)
            
        return ablation_results

def main():
    """
    Main function to run the evaluation
    """
    # Load the trained model and test data
    from model_training import BertLSTM, CryptoDataset
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load the best model
    model = BertLSTM(bert_model)  # You'll need to define bert_model
    model.load_state_dict(torch.load('best_model.pt'))
    
    # Create test dataset and dataloader
    test_dataset = CryptoDataset(...)  # You'll need to provide appropriate parameters
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, device, test_loader)
    
    # Run evaluation
    metrics = evaluator.evaluate_model()
    logger.info("Evaluation metrics:")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
    
    # Run ablation study if needed
    model_variants = {
        'BERT-LSTM': model,
        'LSTM-only': LSTM_only_model,  # You'll need to define these variant models
        'BERT-only': BERT_only_model
    }
    ablation_results = evaluator.perform_ablation_study(model_variants)
    
    logger.info("Evaluation complete! Results saved in the evaluation_results directory.")

if __name__ == "__main__":
    main()