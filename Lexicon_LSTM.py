import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import List, Dict
from datetime import datetime

from textblob import TextBlob
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


########################################
# 1) Compute Lexicon-Based Daily Sentiment
########################################
def compute_lexicon_sentiment(text: str) -> float:
    """
    Compute a basic polarity score from -1.0 (negative) to +1.0 (positive) 
    using TextBlob's lexicon-based approach.
    """
    if not isinstance(text, str):
        text = ""
    blob = TextBlob(text)
    return blob.sentiment.polarity

def aggregate_daily_sentiment(lexicon_df: pd.DataFrame) -> pd.DataFrame:
    
    # 1) Calculate polarity
    logger.info("Calculating lexicon-based sentiment for each tweet...")
    lexicon_df['sentiment_polarity'] = lexicon_df['cleaned_text'].apply(compute_lexicon_sentiment)

    # 2) Group by date (ensure 'date' is daily or you do .dt.date)
    daily_sent = (
        lexicon_df.groupby(lexicon_df['date'].dt.date)['sentiment_polarity']
        .mean()
        .reset_index()
        .rename(columns={'sentiment_polarity': 'sentiment_score'})
    )

    # Convert date back to datetime
    daily_sent['date'] = pd.to_datetime(daily_sent['date'])
    logger.info(f"Aggregated daily sentiment shape: {daily_sent.shape}")
    return daily_sent


########################################
# 2) Technical Indicators (Optional)
########################################
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    df['bb_middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
    std = df['Close'].rolling(window=20, min_periods=1).std()
    df['bb_upper'] = df['bb_middle'] + 2 * std
    df['bb_lower'] = df['bb_middle'] - 2 * std

    df['volatility'] = df['Close'].pct_change().rolling(window=10, min_periods=1).std()
    df['returns'] = df['Close'].pct_change()

    df = df.ffill().fillna(0)
    return df


########################################
# 3) LexiconLSTMDataset
########################################
class LexiconLSTMDataset(Dataset):
    
    def __init__(self, price_df: pd.DataFrame, sentiment_df: pd.DataFrame, sequence_length=10):
        self.sequence_length = sequence_length

        # Compute technical indicators
        price_df = calculate_technical_indicators(price_df)

        # Merge daily sentiment with price data
        merged = pd.merge(
            price_df, 
            sentiment_df, 
            left_on='Date', 
            right_on='date',
            how='inner'
        ).dropna(subset=['Close','sentiment_score']).reset_index(drop=True)

        # Keep track of dates for final plotting
        self.all_dates = merged['Date'].values

        # Build feature matrix
        
        features = merged[['Close','Volume','rsi','macd','macd_signal','bb_upper',
                           'bb_lower','volatility','returns','sentiment_score']].values

        # Scale these features
        self.scaler = MinMaxScaler()
        scaled_features = self.scaler.fit_transform(features)

        # Create sequences (X,y)
        self.X, self.y = self._create_sequences(scaled_features)

    def _create_sequences(self, data: np.ndarray):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i : i+self.sequence_length])
            
            y.append(data[i + self.sequence_length, 0])
        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor([self.y[idx]])

    def inverse_transform_price(self, data: np.ndarray):
        
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        dummy = np.zeros((len(data), 10))
        dummy[:, 0] = data[:, 0]
        inv = self.scaler.inverse_transform(dummy)
        return inv[:, 0]

    def get_date(self, dataset_index: int):
        
        offset_idx = dataset_index + self.sequence_length
        if offset_idx < len(self.all_dates):
            return self.all_dates[offset_idx]
        else:
            return self.all_dates[-1]


########################################
# 4) A Simple BiLSTM Model
########################################
class BiLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_layers=2, dropout=0.3):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size*2, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers*2, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers*2, batch_size, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        
        last_out = out[:, -1, :]
        y_hat = self.fc(last_out)
        return y_hat


########################################
# 5) Train Function
########################################
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    num_epochs: int,
    device: torch.device,
    patience: int=5
):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * X.size(0)

        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                pred_val = model(X_val)
                val_loss = criterion(pred_val, y_val)
                epoch_val_loss += val_loss.item() * X_val.size(0)

        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        logger.info(f"[Epoch {epoch+1}/{num_epochs}] "
                    f"TrainLoss={epoch_train_loss:.5f}, "
                    f"ValLoss={epoch_val_loss:.5f}")

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
# 6) Time-Series Cross Validation
########################################
def time_series_cv(dataset: Dataset, k=5, batch_size=32):
    
    tscv = TimeSeriesSplit(n_splits=k)
    results = []
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(dataset), start=1):
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler   = SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader   = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

        results.append((fold_idx, train_loader, val_loader))
    return results


########################################
# 7) Plotting
########################################
def plot_prediction_vs_actual(dates, actuals, preds, title='Lexicon-LSTM: Prediction vs. Actual'):
    plt.figure(figsize=(12,6))
    plt.plot(dates, actuals, marker='o', label='Actual')
    plt.plot(dates, preds, marker='o', label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('lexicon_lstm_prediction_plot.png', dpi=300)
    plt.close()


########################################
# 8) Main
########################################
def main():
    # 1) Read pre-processed tweets from 'processed_tweets.csv'
    
    logger.info("Loading processed tweets from 'processed_tweets.csv'...")
    tweets_df = pd.read_csv('processed_tweets.csv')
    
    tweets_df['date'] = pd.to_datetime(tweets_df['date'])
    logger.info(f"Loaded {len(tweets_df)} tweets after cleaning.")

    # 2) Compute daily lexicon-based sentiment & save
    daily_lex_sent = aggregate_daily_sentiment(tweets_df)
    daily_lex_sent = daily_lex_sent.sort_values('date').reset_index(drop=True)
    daily_lex_sent.to_csv('daily_lexicon_sentiment.csv', index=False)
    logger.info("Saved 'daily_lexicon_sentiment.csv' with daily lexicon-based sentiment scores.")

    # 3) Load bitcoin data
    logger.info("Loading 'bitcoin.csv' ...")
    price_df = pd.read_csv('bitcoin.csv')
    price_df['Date'] = pd.to_datetime(price_df['Date'])

    
    start_date = '2021-03-01'
    end_date   = '2022-07-31'
    mask = (price_df['Date'] >= start_date) & (price_df['Date'] <= end_date)
    price_df = price_df[mask].copy().reset_index(drop=True)
    logger.info(f"Price rows after filtering: {len(price_df)}")

    # 4) Create LexiconLSTMDataset
    dataset = LexiconLSTMDataset(price_df, daily_lex_sent, sequence_length=10)
    if len(dataset) < 1:
        logger.error("Not enough data to form sequences. Exiting.")
        return

    # 5) Time-series CV
    folds = time_series_cv(dataset, k=5, batch_size=32)
    cv_results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.MSELoss()

    for (fold_idx, train_loader, val_loader) in folds:
        logger.info(f"\n--- Fold {fold_idx}/{len(folds)} ---")
        model = BiLSTM(input_size=10, hidden_size=64, num_layers=2, dropout=0.3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        train_losses, val_losses = train_model(
            model, train_loader, val_loader,
            criterion, optimizer,
            num_epochs=20,
            device=device,
            patience=5
        )

        fold_val_loss = val_losses[-1] if len(val_losses) else float('inf')
        cv_results.append({
            'fold': fold_idx,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_val_loss': fold_val_loss
        })

    # Save CV results
    np.save('cv_results_lexicon_lstm.npy', cv_results, allow_pickle=True)
    logger.info("Time-series CV completed. Saved to 'cv_results_lexicon_lstm.npy'.")

    # 6) Final Train/Test
    full_size = len(dataset)
    train_size = int(0.8 * full_size)
    val_size   = int(0.1 * full_size)
    test_size  = full_size - (train_size + val_size)

    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset   = torch.utils.data.Subset(dataset, range(train_size, train_size+val_size))
    test_dataset  = torch.utils.data.Subset(dataset, range(train_size+val_size, full_size))

    logger.info(f"Final splits => Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=1,  shuffle=False)

    final_model = BiLSTM(input_size=10, hidden_size=64, num_layers=2, dropout=0.3).to(device)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-4)

    logger.info("Training final Lexicon-LSTM on (train+val) portion...")
    train_losses, val_losses = train_model(
        final_model, train_loader, val_loader,
        criterion, final_optimizer,
        num_epochs=30,
        device=device,
        patience=5
    )

    # Plot training history
    plt.figure(figsize=(12,6))
    plt.subplot(2,1,1)
    plt.plot(train_losses, label='Train Loss (scaled)')
    plt.plot(val_losses, label='Val Loss (scaled)')
    plt.title('Lexicon-LSTM Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 7) Evaluate on Test Set
    final_model.eval()
    preds_list = []
    actuals_list = []
    dates_list = []

    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            pred = final_model(X)
            preds_list.append(pred.item())
            actuals_list.append(y.item())

            global_idx = train_size + val_size + i
            date_val = dataset.get_date(global_idx)
            dates_list.append(date_val)

    preds_arr = np.array(preds_list)
    actuals_arr = np.array(actuals_list)

    # Inverse transform to original scale
    inv_preds = dataset.inverse_transform_price(preds_arr)
    inv_actuals = dataset.inverse_transform_price(actuals_arr)

    mse = np.mean((inv_preds - inv_actuals)**2) if len(inv_preds) else float('inf')
    rmse = np.sqrt(mse)
    if np.any(inv_actuals == 0):
        mape = float('inf')
    else:
        mape = np.mean(np.abs((inv_preds - inv_actuals) / inv_actuals)) * 100

    plt.subplot(2,1,2)
    plt.plot(inv_actuals, marker='o', label='Actual (Test)')
    plt.plot(inv_preds, marker='o', label='Predicted (Test)')
    plt.title('Lexicon-LSTM: Test Prediction (Original Scale)')
    plt.xlabel('Sample Index (Test Set)')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('lexicon_lstm_training_history.png', dpi=300)
    plt.close()

    # Final date-based plot
    plot_prediction_vs_actual(dates_list, inv_actuals, inv_preds,
                              title='Lexicon-LSTM: Prediction vs Actual (Test)')

    logger.info(f"[Lexicon-LSTM Final Test Metrics]: MSE={mse:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")

    final_results = {
        'predictions': inv_preds,
        'actuals': inv_actuals,
        'metrics': {
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape)
        },
        'cv_results': cv_results
    }
    np.save('final_results_lexicon_lstm.npy', final_results, allow_pickle=True)
    logger.info("Saved 'final_results_lexicon_lstm.npy' for comparison with BERT+LSTM.")


if __name__ == "__main__":
    main()
