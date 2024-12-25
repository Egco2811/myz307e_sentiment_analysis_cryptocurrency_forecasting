import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
from scipy import stats
from dataclasses import dataclass
import torch
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradePosition:
    """
    Represents a trading position with entry and exit information.
    """
    entry_price: float
    entry_date: datetime
    position_size: float
    position_type: str  # 'long' or 'short'
    exit_price: float = None
    exit_date: datetime = None
    pnl: float = None

class CryptoTradingStrategy:
    """
    Implements the trading strategy based on BERT-LSTM model predictions
    as described in the paper.
    """
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 risk_per_trade: float = 0.02,
                 stop_loss_pct: float = 0.02,
                 take_profit_pct: float = 0.04,
                 transaction_fee: float = 0.001):
        """
        Initialize trading strategy with risk management parameters.
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.transaction_fee = transaction_fee
        
        self.positions = []
        self.current_position = None
        self.performance_metrics = {}
        
        # Performance tracking
        self.daily_returns = []
        self.equity_curve = [initial_capital]
        self.trade_history = []
        
    def calculate_position_size(self, entry_price: float) -> float:
        """
        Calculate position size based on risk management rules.
        """
        risk_amount = self.current_capital * self.risk_per_trade
        position_size = risk_amount / (entry_price * self.stop_loss_pct)
        return position_size

    def execute_trade(self, 
                     signal: int,  # 1 for buy, -1 for sell, 0 for hold
                     price: float,
                     date: datetime,
                     sentiment_score: float) -> None:
        """
        Execute trading decision based on model prediction and sentiment.
        """
        # Don't trade if sentiment and signal disagree strongly
        if abs(sentiment_score) > 0.5 and np.sign(sentiment_score) != signal:
            logger.info(f"Skipping trade due to sentiment divergence: {sentiment_score}")
            return
        
        if self.current_position is None and signal != 0:
            # Open new position
            position_size = self.calculate_position_size(price)
            
            # Apply transaction fee
            cost = position_size * price * (1 + self.transaction_fee)
            
            if cost <= self.current_capital:
                position_type = 'long' if signal == 1 else 'short'
                self.current_position = TradePosition(
                    entry_price=price,
                    entry_date=date,
                    position_size=position_size,
                    position_type=position_type
                )
                self.current_capital -= cost
                logger.info(f"Opened {position_type} position at {price}")
                
        elif self.current_position is not None:
            # Check exit conditions
            pnl = self.calculate_unrealized_pnl(price)
            pnl_pct = pnl / (self.current_position.entry_price * 
                            self.current_position.position_size)
            
            should_exit = (
                (signal == -self.get_position_signal()) or  # Signal reversed
                (pnl_pct <= -self.stop_loss_pct) or        # Stop loss hit
                (pnl_pct >= self.take_profit_pct)          # Take profit hit
            )
            
            if should_exit:
                self.close_position(price, date)

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized profit/loss for current position.
        """
        if self.current_position is None:
            return 0.0
            
        price_diff = current_price - self.current_position.entry_price
        if self.current_position.position_type == 'short':
            price_diff = -price_diff
            
        return price_diff * self.current_position.position_size

    def close_position(self, price: float, date: datetime) -> None:
        """
        Close current position and update metrics.
        """
        if self.current_position is None:
            return
            
        self.current_position.exit_price = price
        self.current_position.exit_date = date
        
        # Calculate PnL including transaction fees
        entry_cost = (self.current_position.entry_price * 
                     self.current_position.position_size * 
                     (1 + self.transaction_fee))
        exit_value = (price * self.current_position.position_size * 
                     (1 - self.transaction_fee))
        
        if self.current_position.position_type == 'short':
            self.current_position.pnl = entry_cost - exit_value
        else:
            self.current_position.pnl = exit_value - entry_cost
            
        self.current_capital += exit_value
        self.positions.append(self.current_position)
        self.current_position = None
        
        logger.info(f"Closed position at {price}, PnL: {self.current_position.pnl}")

    def get_position_signal(self) -> int:
        """
        Get current position signal (1 for long, -1 for short, 0 for none).
        """
        if self.current_position is None:
            return 0
        return 1 if self.current_position.position_type == 'long' else -1

    def generate_signals(self, 
                        predictions: np.array, 
                        sentiment_scores: np.array,
                        threshold: float = 0.02) -> np.array:
        """
        Generate trading signals based on model predictions and sentiment.
        """
        signals = np.zeros_like(predictions)
        
        # Calculate prediction changes
        pred_changes = np.diff(predictions, prepend=predictions[0])
        
        # Generate signals based on prediction changes and sentiment
        signals[pred_changes > threshold] = 1  # Buy signal
        signals[pred_changes < -threshold] = -1  # Sell signal
        
        # Modify signals based on sentiment agreement
        sentiment_agreement = np.sign(sentiment_scores) == np.sign(pred_changes)
        signals[~sentiment_agreement] = 0  # No trade if sentiment disagrees
        
        return signals

    def run_backtest(self, 
                    prices: np.array,
                    dates: np.array,
                    predictions: np.array,
                    sentiment_scores: np.array) -> Dict:
        """
        Run backtest of trading strategy.
        """
        logger.info("Starting strategy backtest...")
        
        # Generate trading signals
        signals = self.generate_signals(predictions, sentiment_scores)
        
        # Reset performance tracking
        self.current_capital = self.initial_capital
        self.positions = []
        self.current_position = None
        self.daily_returns = []
        self.equity_curve = [self.initial_capital]
        
        # Run simulation
        for i in range(len(prices)):
            # Execute trading logic
            self.execute_trade(
                signal=signals[i],
                price=prices[i],
                date=dates[i],
                sentiment_score=sentiment_scores[i]
            )
            
            # Track daily performance
            daily_pnl = self.calculate_unrealized_pnl(prices[i])
            self.daily_returns.append(
                daily_pnl / self.equity_curve[-1] if self.equity_curve[-1] > 0 else 0
            )
            self.equity_curve.append(self.current_capital)
            
        # Calculate performance metrics
        self.calculate_performance_metrics()
        
        return self.performance_metrics

    def calculate_performance_metrics(self) -> None:
        """
        Calculate comprehensive performance metrics for the strategy.
        """
        returns = np.array(self.daily_returns)
        equity = np.array(self.equity_curve)
        
        # Basic return metrics
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        annual_return = total_return * (252 / len(returns))
        
        # Risk metrics
        daily_vol = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annual_return / daily_vol if daily_vol > 0 else 0
        
        # Drawdown analysis
        rolling_max = np.maximum.accumulate(equity)
        drawdowns = (equity - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns)
        
        # Trade analysis
        profitable_trades = len([p for p in self.positions if p.pnl > 0])
        total_trades = len(self.positions)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        self.performance_metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'daily_vol': daily_vol
        }

    def plot_results(self, output_dir: str = "trading_results"):
        """
        Create comprehensive visualization of trading results.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(self.equity_curve, label='Portfolio Value')
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Days')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.savefig(output_dir / 'equity_curve.png')
        plt.close()
        
        # Plot drawdown
        equity = np.array(self.equity_curve)
        rolling_max = np.maximum.accumulate(equity)
        drawdowns = (equity - rolling_max) / rolling_max
        
        plt.figure(figsize=(12, 6))
        plt.plot(drawdowns, label='Drawdown')
        plt.title('Portfolio Drawdown')
        plt.xlabel('Days')
        plt.ylabel('Drawdown')
        plt.legend()
        plt.savefig(output_dir / 'drawdown.png')
        plt.close()
        
        # Plot trade distribution
        pnls = [position.pnl for position in self.positions]
        plt.figure(figsize=(10, 6))
        sns.histplot(pnls, kde=True)
        plt.title('Trade PnL Distribution')
        plt.xlabel('PnL')
        plt.ylabel('Frequency')
        plt.savefig(output_dir / 'pnl_distribution.png')
        plt.close()

def main():
    """
    Main function to demonstrate the trading strategy.
    """
    # Load necessary data
    price_data = pd.read_csv('price_data.csv')
    predictions = np.load('model_predictions.npy')
    sentiment_scores = np.load('sentiment_scores.npy')
    
    # Initialize strategy
    strategy = CryptoTradingStrategy(
        initial_capital=100000.0,
        risk_per_trade=0.02,
        stop_loss_pct=0.02,
        take_profit_pct=0.04
    )
    
    # Run backtest
    results = strategy.run_backtest(
        prices=price_data['Close'].values,
        dates=pd.to_datetime(price_data['Date']).values,
        predictions=predictions,
        sentiment_scores=sentiment_scores
    )
    
    # Log results
    logger.info("Trading Results:")
    for metric, value in results.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Create visualizations
    strategy.plot_results()

if __name__ == "__main__":
    main()