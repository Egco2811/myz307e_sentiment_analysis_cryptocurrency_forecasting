import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    df['bb_upper'] = df['bb_middle'] + 2*std
    df['bb_lower'] = df['bb_middle'] - 2*std

    df['volatility'] = df['Close'].pct_change().rolling(window=10, min_periods=1).std()
    df['returns'] = df['Close'].pct_change()

    df = df.ffill().fillna(0)
    return df


class LSTMOnlyDataset(Dataset):
    """
    Uses only price + technical indicators, no sentiment.
    
    """
    def __init__(self, df: pd.DataFrame, sequence_length=10):
        self.sequence_length = sequence_length
        df = df.copy().sort_values('Date').reset_index(drop=True)
        self.dates = df['Date'].values

        df = calculate_technical_indicators(df)
        feats = df[['Close','Volume','rsi','macd','macd_signal',
                    'bb_upper','bb_lower','volatility','returns']].values

        self.scaler = MinMaxScaler()
        scaled = self.scaler.fit_transform(feats)

        self.X, self.y = self._create_sequences(scaled)

    def _create_sequences(self, data: np.ndarray):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i : i+self.sequence_length])
            y.append(data[i + self.sequence_length, 0])  # next-step close
        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor([self.y[idx]])

    def inverse_transform_price(self, arr: np.ndarray):
        if arr.ndim == 1:
            arr = arr.reshape(-1,1)
        dummy = np.zeros((len(arr), 9))
        dummy[:,0] = arr[:,0]
        inv = self.scaler.inverse_transform(dummy)
        return inv[:,0]


class StandardLSTM(nn.Module):
    def __init__(self, input_size=9, hidden_size=64, num_layers=1, dropout=0.0):
        super(StandardLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=(dropout if num_layers>1 else 0.0),
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        last_out = out[:, -1, :]
        y_hat = self.fc(last_out)
        return y_hat


def main():
    # 1) Load and filter data
    df = pd.read_csv('bitcoin.csv')
    df['Date'] = pd.to_datetime(df['Date'])

    start_date = '2021-03-01'
    end_date   = '2022-07-31'
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df = df[mask].copy().reset_index(drop=True)
    logger.info(f"LSTM-only: Rows after filtering: {len(df)}")

    if len(df) < 20:
        logger.error("Not enough data for demonstration. Exiting.")
        return

    # 2) Build dataset with e.g. seq_length=10
    seq_len = 10
    dataset = LSTMOnlyDataset(df, sequence_length=seq_len)

    # 3) We define the final test set to be exactly the last data points
    
    test_size = 3
    if test_size >= len(dataset):
        logger.error("Not enough total sequences to carve out 3 test points. Exiting.")
        return

    
    train_val_size = len(dataset) - test_size

    # 4) Prepare train+val loaders, test loader
    
    train_ids = range(train_val_size)
    test_ids  = range(train_val_size, len(dataset))

    logger.info(f"Train+Val sequences: {len(train_ids)}, Test sequences: {len(test_ids)}")

    from torch.utils.data import DataLoader, SubsetRandomSampler
    train_loader = DataLoader(dataset, batch_size=32, sampler=SubsetRandomSampler(train_ids))
    test_loader  = DataLoader(dataset, batch_size=1,  sampler=SubsetRandomSampler(test_ids))

    # 5) Build LSTM model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StandardLSTM(input_size=9, hidden_size=64, num_layers=1, dropout=0.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # 6) Train for e.g. 30 epochs
    
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        total_samples = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X.size(0)
            total_samples += X.size(0)

        epoch_loss /= total_samples
        if (epoch+1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss={epoch_loss:.6f}")

    # 7) Evaluate on the final test set
    model.eval()
    preds_list = []
    actuals_list = []
    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            preds_list.append(pred.item())
            actuals_list.append(y.item())

    preds_arr = np.array(preds_list)
    acts_arr  = np.array(actuals_list)

    # Inverse transform
    inv_preds = dataset.inverse_transform_price(preds_arr)
    inv_acts  = dataset.inverse_transform_price(acts_arr)

    # Compute metrics on these  test points
    mse  = np.mean((inv_preds - inv_acts)**2)
    rmse = np.sqrt(mse)
    if np.any(inv_acts==0):
        mape = float('inf')
    else:
        mape = np.mean(np.abs((inv_preds - inv_acts)/inv_acts))*100

    logger.info(f"LSTM-only (3-point test) => MSE={mse:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")

    # 8) Simple index-based plot: x=0..2
    x_idx = range(len(inv_preds))
    plt.figure(figsize=(6,4))
    plt.plot(x_idx, inv_acts,  label='Actual', color='black', marker='o')
    plt.plot(x_idx, inv_preds, label='Predicted', color='orange', marker='o', linestyle='--')
    plt.xlabel('Test Sample Index (last 3 points)')
    plt.ylabel('BTC Price (USD)')
    plt.title('LSTM-Only: Last 3-Point Test Prediction')
    plt.legend()
    plt.tight_layout()
    plt.savefig('standard_lstm_pred_plot.png', dpi=300)
    plt.close()

    # 9) Save final results
    results = {
        'predictions': inv_preds,
        'actuals': inv_acts,
        'metrics': {
            'mse':  float(mse),
            'rmse': float(rmse),
            'mape': float(mape)
        },
        'cv_results': None  # or any partial CV info if you ran it
    }
    np.save('final_results_standard_lstm.npy', results, allow_pickle=True)
    logger.info("Saved final_results_standard_lstm.npy. Done.")

if __name__ == "__main__":
    main()

# Display the plot
plt.show()
