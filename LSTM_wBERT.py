import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

########################################
# 1) Plot Optimization History
########################################
def plot_optimization_history(objective_values: List[float], best_values: List[float]):
    plt.figure(figsize=(12, 6))
    plt.plot(objective_values, 'o', color='lightblue', alpha=0.5, label='Trial Value')
    plt.plot(best_values, 'r-', linewidth=2, label='Best Value')
    plt.xlabel('Number of Trials')
    plt.ylabel('Objective Value')
    plt.title('Convergence of Hyperparameter Optimization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('optimization_history.png', dpi=300, bbox_inches='tight')
    plt.close()

########################################
# 2) Technical Indicators
########################################
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    
    Use min_periods=1 so partial windows don't turn entire columns into NaNs.
    Then fill with forward-fill or zeros to ensure no NaN remains.
    """
    df = df.copy()

    # 1) RSI (14-day)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-9)  # Add small epsilon to avoid division by zero
    df['rsi'] = 100 - (100 / (1 + rs))

    # 2) MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # 3) Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
    std = df['Close'].rolling(window=20, min_periods=1).std()
    df['bb_upper'] = df['bb_middle'] + (std * 2)
    df['bb_lower'] = df['bb_middle'] - (std * 2)

    # 4) Volatility
    df['volatility'] = df['Close'].pct_change().rolling(window=10, min_periods=1).std()

    # 5) Returns
    df['returns'] = df['Close'].pct_change()

    # Forward fill leftover NaNs, then fill any remaining with 0
    df = df.ffill().fillna(0)
    return df

########################################
# 3) CryptoDataset
########################################
class CryptoDataset(Dataset):
    def __init__(self, price_data: pd.DataFrame, sentiment_data: pd.DataFrame, sequence_length=10):
        self.sequence_length = sequence_length

        
        self.all_dates = price_data['Date'].values  # or price_data.index

        # Initialize scalers
        self.price_scaler = MinMaxScaler()
        self.sentiment_scaler = MinMaxScaler()
        self.volume_scaler = MinMaxScaler()
        
        # Add technical indicators
        price_data = calculate_technical_indicators(price_data)
        
        # Normalize features
        self.normalized_data = self.normalize_features(price_data, sentiment_data)
        
        # Create sequences (X, y) pairs
        self.X, self.y = self._create_sequences(self.normalized_data)

    def normalize_features(self, price_data: pd.DataFrame, sentiment_data: pd.DataFrame):
       
        price_features = self.price_scaler.fit_transform(
            price_data[['Close']].values.reshape(-1, 1)
        )
        volume = self.volume_scaler.fit_transform(
            price_data[['Volume']].values.reshape(-1, 1)
        )
        sentiment = self.sentiment_scaler.fit_transform(
            sentiment_data[['sentiment_score']].values.reshape(-1, 1)
        )
        
        # Combine all features:
        
        technical_features = np.column_stack([
            price_features,
            volume,
            price_data[['rsi', 'macd', 'macd_signal', 
                        'bb_upper', 'bb_lower', 'volatility', 'returns']].values,
            sentiment
        ])
        
        # Check for NaNs or Infs
        if np.isnan(technical_features).any() or np.isinf(technical_features).any():
            logger.warning("NaN or Inf found in normalized features!")
        
        return technical_features
    
    def _create_sequences(self, data: np.ndarray):
       
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length, 0])  # 0 => normalized close price
        return np.array(X), np.array(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor([self.y[idx]])
    
    def inverse_transform_price(self, data: np.ndarray):
        # Inverse transform for the price column
        return self.price_scaler.inverse_transform(data.reshape(-1, 1))
    
    def get_date(self, idx):
      
        return self.all_dates[idx]


########################################
# 4) Bidirectional LSTM Model
########################################
class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_layers=2, dropout=0.5):
        super(BidirectionalLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc1 = nn.Linear(hidden_size * 2, 32)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        # Take the last time step
        last_out = lstm_out[:, -1]
        
        x = self.fc1(last_out)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

########################################
# 5) Training Function (with Early Stopping)
########################################
def train_model(
    model: BidirectionalLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    num_epochs: int,
    device: torch.device,
    dropout: float = 0.5,
    l2_lambda: float = 0.01,
    patience: int = 10
):
  
    train_losses = []
    val_losses = []

    # Dynamically set dropout if desired
    model.dropout.p = dropout

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        # ---------------------------
        #      TRAIN
        # ---------------------------
        model.train()
        epoch_train_loss = 0.0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            
            outputs = model(X)   
            loss = criterion(outputs, y)
            if torch.isnan(loss):
                logger.warning(f"NaN loss encountered on epoch {epoch+1}")
                break
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * X.size(0)
        
        # Average training loss for the epoch
        if len(train_loader.dataset) > 0:
            epoch_train_loss /= len(train_loader.dataset)
        else:
            epoch_train_loss = float('inf')
        train_losses.append(epoch_train_loss)

        # ---------------------------
        #      VALIDATION
        # ---------------------------
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                epoch_val_loss += val_loss.item() * X_val.size(0)
        
        # Average validation loss for the epoch
        if len(val_loader.dataset) > 0:
            epoch_val_loss /= len(val_loader.dataset)
        else:
            epoch_val_loss = float('inf')
        val_losses.append(epoch_val_loss)

        logger.info(f"[Epoch {epoch+1}/{num_epochs}] "
                    f"Train Loss (normalized): {epoch_train_loss:.6f}, "
                    f"Val Loss (normalized): {epoch_val_loss:.6f}")
        
        # Early Stopping Check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    return train_losses, val_losses

########################################
# 6) Directional Accuracy
########################################
def calculate_directional_accuracy(model: BidirectionalLSTM,
                                   dataloader: DataLoader,
                                   device: torch.device) -> float:
   
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        all_outputs = []
        all_targets = []
        
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            all_outputs.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
        
        # Flatten or concatenate
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # For directional accuracy, we need consecutive pairs
        for i in range(len(all_outputs) - 1):
            pred_dir = all_outputs[i+1] - all_outputs[i]
            true_dir = all_targets[i+1] - all_targets[i]
            if pred_dir * true_dir > 0:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0

########################################
# 7) Hyperparameter Optimization
########################################
def train_with_optimization(model: BidirectionalLSTM, 
                            train_loader: DataLoader,
                            val_loader: DataLoader,
                            device: torch.device,
                            num_trials: int = 5) -> Dict:
   
    objective_values = []
    best_values = []
    best_value = float('inf')
    best_params = {}
    
    for trial in range(num_trials):
        # Randomly sample hyperparameters
        lr = np.exp(np.random.uniform(np.log(1e-5), np.log(1e-3)))
        dropout = np.random.uniform(0.3, 0.7)
        l2_lambda = np.exp(np.random.uniform(np.log(1e-5), np.log(1e-2)))
        
        # Re-initialize the model each trial
        trial_model = BidirectionalLSTM(
            input_size=model.lstm.input_size,
            hidden_size=model.hidden_size,
            num_layers=model.num_layers,
            dropout=dropout
        ).to(device)
        
        optimizer = torch.optim.Adam(trial_model.parameters(), lr=lr, weight_decay=l2_lambda)
        
        # Train model (on normalized scale)
        train_losses, val_losses = train_model(
            trial_model,
            train_loader,
            val_loader,
            nn.MSELoss(),
            optimizer,
            num_epochs=30,
            device=device,
            dropout=dropout,
            l2_lambda=l2_lambda
        )
        
        # Use the average of the last 5 val losses
        if len(val_losses) >= 5:
            val_loss = np.mean(val_losses[-5:])
        else:
            val_loss = val_losses[-1] if len(val_losses) > 0 else float('inf')

        # Directional accuracy on normalized data
        direction_acc = calculate_directional_accuracy(trial_model, val_loader, device)

        # Weighted objective: lower val_loss is better, higher direction_acc is better
        objective = val_loss + 0.1 * (1 - direction_acc)
        objective_values.append(objective)
        
        # Update best if improved
        if objective < best_value:
            best_value = objective
            best_params = {
                'lr': lr,
                'dropout': dropout,
                'l2_lambda': l2_lambda
            }
        best_values.append(best_value)
        
        logger.info(
            f"Trial {trial+1}/{num_trials} | "
            f"lr={lr:.6f}, dropout={dropout:.3f}, l2_lambda={l2_lambda:.6f} | "
            f"Val Loss={val_loss:.6f}, DirAcc={direction_acc:.3f}, "
            f"Objective={objective:.6f}"
        )
        
        # Plot progress each trial
        plot_optimization_history(objective_values, best_values)
    
    return best_params, objective_values, best_values

########################################
# 8) K-Fold Time-Series Split
########################################
def k_fold_cross_validation(dataset: Dataset, k=5, batch_size=32):
    tscv = TimeSeriesSplit(n_splits=k)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(dataset)):
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
        
        fold_results.append((train_loader, val_loader))
    
    return fold_results

########################################
# 9) Utility: Plot Final Test Predictions vs. Actual
########################################
def plot_final_predictions(dates, actuals, predictions, title='Bitcoin Price Prediction vs Actual'):
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actuals, label='Actual', marker='o')
    plt.plot(dates, predictions, label='Predicted', marker='o')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('final_prediction_plot.png', dpi=300)
    plt.close()

########################################
# 10) Main
########################################
def main():
    # Load data
    price_data = pd.read_csv('bitcoin.csv')  # Must contain columns: Date, Close, Volume, etc.
    sentiment_data = pd.read_csv('daily_sentiment.csv')  # Must contain columns: date, sentiment_score
    
    # Date filtering
    start_date = '2021-03-01'
    end_date = '2022-07-31'
    
    price_data['Date'] = pd.to_datetime(price_data['Date'])
    sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
    
    mask = (price_data['Date'] >= start_date) & (price_data['Date'] <= end_date)
    price_data = price_data[mask]
    
    logger.info(f"Filtered price_data rows: {len(price_data)}")
    logger.info(f"Full sentiment_data rows: {len(sentiment_data)}")
    
    # Merge price + sentiment on date
    data = pd.merge(
        price_data, 
        sentiment_data[['date', 'sentiment_score']], 
        left_on='Date', 
        right_on='date',
        how='inner'
    )
    logger.info(f"Merged rows: {len(data)}")
    
    if len(data) < 20:
        logger.warning("WARNING: Very few rows after merge. Rolling indicators may be mostly NaN!")
    
    # Create dataset
    dataset = CryptoDataset(data, data, sequence_length=10)
    
    # If dataset is too short for sequence_length=10, handle it
    if len(dataset) < 1:
        logger.error("Not enough data to create a single sequence. Exiting.")
        return None, None
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cross-validation
    folds = k_fold_cross_validation(dataset, k=5, batch_size=32)
    cv_results = []
    
    for fold, (train_loader, val_loader) in enumerate(folds):
        logger.info(f"\n--- Training Fold {fold+1} / {len(folds)} ---")
        
        model = BidirectionalLSTM(input_size=10).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)
        criterion = nn.MSELoss()
        
        
        train_losses, val_losses = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs=30,
            device=device,
            patience=5  # shorter patience for CV
        )
        
        final_val_loss = val_losses[-1] if len(val_losses) > 0 else float('inf')
        cv_results.append({
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_val_loss': final_val_loss
        })
    
    # Save cross-validation results
    np.save('cv_results.npy', cv_results, allow_pickle=True)

    # Prepare single train_loader / val_loader for hyperparam optimization
    if len(folds) > 0:
        train_loader, val_loader = folds[0]
    else:
        logger.error("No folds created. Possibly not enough data for TimeSeriesSplit.")
        return None, None
    
    # Hyperparameter optimization on a fresh model
    base_model = BidirectionalLSTM(input_size=10).to(device)
    logger.info("Starting hyperparameter optimization...")
    best_params, obj_values, best_values = train_with_optimization(
        base_model, train_loader, val_loader, device, num_trials=5
    )
    logger.info(f"Best hyperparameters found: {best_params}")
    
    # Provide fallback if keys missing (avoid KeyError)
    if 'dropout' not in best_params:
        best_params['dropout'] = 0.5
    if 'lr' not in best_params:
        best_params['lr'] = 1e-4
    if 'l2_lambda' not in best_params:
        best_params['l2_lambda'] = 1e-3
    
    # Plot final optimization history
    plot_optimization_history(obj_values, best_values)
    
    # Train final model with best hyperparameters on the full dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, len(dataset)))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    final_model = BidirectionalLSTM(
        input_size=10,
        hidden_size=64,
        dropout=best_params['dropout']
    ).to(device)
    
    criterion = nn.MSELoss()
    final_optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=best_params['lr'],
        weight_decay=best_params['l2_lambda']
    )
    
    logger.info("Training final model on the full dataset (normalized scale)...")
    train_losses, val_losses = train_model(
        final_model,
        train_loader,
        val_loader,
        criterion,
        final_optimizer,
        num_epochs=30,
        device=device,
        dropout=best_params['dropout'],
        l2_lambda=best_params['l2_lambda'],
        patience=10
    )
    
    # Evaluate on Test set
    final_model.eval()
    predictions = []
    actuals = []
    test_dates = []  
    
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            outputs = final_model(X)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y.cpu().numpy())
            
            
    predictions_arr = np.array(predictions).flatten()  # shape [N]
    actuals_arr = np.array(actuals).flatten()
    
    # Inverse transform from normalized back to USD scale
    predictions_inv = dataset.inverse_transform_price(predictions_arr)
    actuals_inv = dataset.inverse_transform_price(actuals_arr)
    
    # We'll construct the date range for each test sample
    
    test_indices = range(train_size + val_size, len(dataset))
    
    test_dates_list = []
    for idx in test_indices:
        
        actual_idx_for_label = idx + dataset.sequence_length
        if actual_idx_for_label < len(dataset.all_dates):
            test_dates_list.append(dataset.get_date(actual_idx_for_label))
        else:
            test_dates_list.append(dataset.get_date(len(dataset.all_dates)-1))
    
    
    test_dates_list = test_dates_list[:len(predictions_inv)]
    
    # Calculate final metrics in the real (USD) scale
    mse = np.mean((predictions_inv - actuals_inv) ** 2) if len(predictions_inv) > 0 else float('inf')
    rmse = np.sqrt(mse) if mse != float('inf') else float('inf')
    if np.any(actuals_inv == 0):
        mape = float('inf')
    else:
        mape = np.mean(np.abs((actuals_inv - predictions_inv) / actuals_inv)) * 100
    
    results = {
        'predictions': predictions_inv,
        'actuals': actuals_inv,
        'metrics': {
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape)
        },
        'cv_results': cv_results
    }
    
    np.save('final_results.npy', results, allow_pickle=True)
    
    # Plot training/validation history
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss (normalized)')
    plt.plot(val_losses, label='Validation Loss (normalized)')
    plt.title('Training History (normalized scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    if len(actuals_inv) > 0:
        plt.plot(actuals_inv, label='Actual (test set)', marker='o')
    if len(predictions_inv) > 0:
        plt.plot(predictions_inv, label='Predicted (test set)', marker='o')
    plt.title('Bitcoin Price Prediction (Test Set) - Original Scale')
    plt.xlabel('Sample Index (Test Set)')
    plt.ylabel('Price (USD)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()
    
    # Also plot final predictions over time (dates)
    plot_final_predictions(test_dates_list, actuals_inv, predictions_inv,
                           title='Bitcoin Price Prediction vs Actual (Test Set)')

    logger.info(f'\n[Final Test Metrics on Real Price Scale]:')
    logger.info(f'MSE: {mse:.2f}')
    logger.info(f'RMSE: {rmse:.2f}')
    logger.info(f'MAPE: {mape:.2f}%')
    
    return final_model, results

if __name__ == "__main__":
    model, results = main()
