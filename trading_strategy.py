import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dataclasses import dataclass
import torch
from sklearn.preprocessing import MinMaxScaler
import json

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TradePosition:
   """Represents a trading position with entry and exit information"""
   entry_price: float
   entry_date: datetime
   position_size: float
   position_type: str  # 'long' or 'short'
   entry_sentiment: float
   exit_price: float = None
   exit_date: datetime = None
   exit_sentiment: float = None
   pnl: float = None

class CryptoTradingStrategy:
   def __init__(self, config: Dict):
       """Initialize trading strategy with configuration parameters"""
       trade_config = config['trading']
       self.initial_capital = trade_config['initial_capital']
       self.risk_per_trade = trade_config['risk_per_trade']
       self.stop_loss_pct = trade_config['stop_loss_pct']
       self.take_profit_pct = trade_config['take_profit_pct']
       self.transaction_fee = trade_config['transaction_fee']
       self.max_position_size = trade_config['position_sizing']['max_position_size']
       self.min_position_size = trade_config['position_sizing']['min_position_size']
       
       self.current_capital = self.initial_capital
       self.positions: List[TradePosition] = []
       self.current_position = None
       
       # Performance tracking
       self.daily_returns = []
       self.equity_curve = [self.initial_capital]
       self.performance_metrics = {}

   def calculate_position_size(self, 
                             entry_price: float, 
                             sentiment_score: float) -> float:
       """Calculate position size based on risk management and sentiment"""
       risk_amount = self.current_capital * self.risk_per_trade
       base_position = risk_amount / (entry_price * self.stop_loss_pct)
       
       # Adjust position size based on sentiment strength
       sentiment_multiplier = 1 + abs(sentiment_score)
       position_size = base_position * sentiment_multiplier
       
       # Apply position size limits
       return np.clip(
           position_size,
           self.min_position_size * self.current_capital,
           self.max_position_size * self.current_capital
       )

   def should_enter_trade(self, 
                         prediction: float, 
                         sentiment_score: float, 
                         volatility: float) -> bool:
       """Determine if we should enter a trade based on signals"""
       # Strong sentiment-prediction agreement
       signal_agreement = np.sign(prediction) == np.sign(sentiment_score)
       
       # Adequate volatility for trading
       sufficient_volatility = volatility > np.mean(self.volatility_history)
       
       return signal_agreement and sufficient_volatility

   def execute_trade(self, 
                    price: float,
                    date: datetime,
                    prediction: float,
                    sentiment_score: float,
                    volatility: float):
       """Execute trading decision based on signals"""
       if self.current_position is None:
           if self.should_enter_trade(prediction, sentiment_score, volatility):
               position_size = self.calculate_position_size(price, sentiment_score)
               cost = position_size * price * (1 + self.transaction_fee)
               
               if cost <= self.current_capital:
                   position_type = 'long' if prediction > 0 else 'short'
                   self.current_position = TradePosition(
                       entry_price=price,
                       entry_date=date,
                       position_size=position_size,
                       position_type=position_type,
                       entry_sentiment=sentiment_score
                   )
                   self.current_capital -= cost
                   logger.info(f"Opened {position_type} position at {price}")
       
       else:
           self._check_exit_conditions(
               price, date, prediction, sentiment_score, volatility
           )

   def _check_exit_conditions(self,
                            price: float,
                            date: datetime,
                            prediction: float,
                            sentiment_score: float,
                            volatility: float):
       """Check and execute position exit conditions"""
       pnl = self.calculate_unrealized_pnl(price)
       pnl_pct = pnl / (self.current_position.entry_price * 
                       self.current_position.position_size)
       
       # Exit conditions
       stop_loss_hit = pnl_pct <= -self.stop_loss_pct
       take_profit_hit = pnl_pct >= self.take_profit_pct
       sentiment_reversal = (np.sign(sentiment_score) != 
                           np.sign(self.current_position.entry_sentiment))
       
       if stop_loss_hit or take_profit_hit or sentiment_reversal:
           self.close_position(price, date, sentiment_score)

   def calculate_unrealized_pnl(self, current_price: float) -> float:
       """Calculate unrealized profit/loss for current position"""
       if self.current_position is None:
           return 0.0
           
       price_diff = current_price - self.current_position.entry_price
       if self.current_position.position_type == 'short':
           price_diff = -price_diff
           
       return price_diff * self.current_position.position_size

   def close_position(self, 
                     price: float, 
                     date: datetime,
                     sentiment_score: float):
       """Close current position and update metrics"""
       if self.current_position is None:
           return
           
       self.current_position.exit_price = price
       self.current_position.exit_date = date
       self.current_position.exit_sentiment = sentiment_score
       
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
       
       logger.info(
           f"Closed position at {price}, "
           f"PnL: {self.current_position.pnl:.2f}"
       )
       
       self.current_position = None

   def run_backtest(self, 
                   df: pd.DataFrame) -> Dict:
       """Run backtest of trading strategy"""
       logger.info("Starting strategy backtest...")
       
       # Initialize tracking variables
       self.volatility_history = df['volatility'].values
       
       # Run simulation
       for idx, row in df.iterrows():
           self.execute_trade(
               price=row['Close'],
               date=row['Date'],
               prediction=row['prediction'],
               sentiment_score=row['sentiment_score'],
               volatility=row['volatility']
           )
           
           # Track daily performance
           daily_pnl = self.calculate_unrealized_pnl(row['Close'])
           self.daily_returns.append(
               daily_pnl / self.equity_curve[-1] 
               if self.equity_curve[-1] > 0 else 0
           )
           self.equity_curve.append(self.current_capital)
       
       # Calculate final performance metrics
       self.calculate_performance_metrics()
       return self.performance_metrics

   def calculate_performance_metrics(self):
       """Calculate comprehensive performance metrics"""
       returns = np.array(self.daily_returns)
       equity = np.array(self.equity_curve)
       
       # Return metrics
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
           'profitable_trades': profitable_trades
       }

   def plot_results(self, output_dir: Path):
       """Create comprehensive visualization of trading results"""
       # Ensure output directory exists
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
   """Main function to run trading strategy"""
   # Load configuration
   with open('config.json', 'r') as f:
       config = json.load(f)
   
   # Load data
   data = pd.read_csv(
       Path(config['output_directory']) / 'processed_test_data.csv',
       parse_dates=['Date']
   )
   
   # Initialize strategy
   strategy = CryptoTradingStrategy(config)
   
   # Run backtest
   results = strategy.run_backtest(data)
   
   # Create visualizations
   output_dir = Path(config['output_directory']) / 'trading_results'
   strategy.plot_results(output_dir)
   
   # Log results
   logger.info("Trading Results:")
   for metric, value in results.items():
       logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
   main()
