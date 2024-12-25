import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import TimeSeriesSplit
import logging
import warnings
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class CryptoDataset(Dataset):
    """
    Custom dataset class for cryptocurrency data that combines price and sentiment
    features from our preprocessed data.
    """
    def __init__(self, price_data, sentiment_data, sequence_length, tokenizer, max_length=128):
        self.sequence_length = sequence_length
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Combine price and sentiment data
        self.data = self._prepare_data(price_data, sentiment_data)
        
    def _prepare_data(self, price_data, sentiment_data):
        """
        Prepare combined dataset with both price and sentiment features.
        Implements the data alignment described in Section III.A of the paper.
        """
        # Merge price and sentiment data on date
        combined_data = pd.merge(price_data, sentiment_data, on='Date', how='inner')
        
        # Calculate daily returns
        combined_data['returns'] = combined_data['Close'].pct_change()
        
        # Calculate rolling sentiment metrics as described in paper
        combined_data['rolling_sentiment'] = combined_data['sentiment_score'].rolling(
            window=7).mean()
        
        # Normalize data as described in Section III.A.3
        numerical_columns = ['Close', 'returns', 'sentiment_score', 'rolling_sentiment']
        for column in numerical_columns:
            min_val = combined_data[column].min()
            max_val = combined_data[column].max()
            combined_data[column] = (combined_data[column] - min_val) / (max_val - min_val)
            
        return combined_data
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        # Get sequence of data
        sequence = self.data.iloc[idx:idx + self.sequence_length]
        
        # Prepare BERT input for sentiment text
        text_sequence = sequence['cleaned_text'].values
        encoded_text = self.tokenizer(
            text_sequence.tolist(),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Prepare price and sentiment features
        numerical_features = sequence[['Close', 'returns', 'sentiment_score', 
                                    'rolling_sentiment']].values
        
        # Prepare target (next day's return)
        target = self.data.iloc[idx + self.sequence_length]['returns']
        
        return {
            'input_ids': encoded_text['input_ids'],
            'attention_mask': encoded_text['attention_mask'],
            'numerical_features': torch.FloatTensor(numerical_features),
            'target': torch.FloatTensor([target])
        }

class BertLSTM(nn.Module):
    """
    Combined BERT-LSTM model as described in Section III.B of the paper.
    Implements the architecture that processes both text sentiment and price data.
    """
    def __init__(self, bert_model, lstm_hidden_size=128, num_lstm_layers=3, dropout_rate=0.3):
        super(BertLSTM, self).__init__()
        
        # BERT for sentiment analysis
        self.bert = bert_model
        self.bert_dropout = nn.Dropout(dropout_rate)
        
        # Dimension of BERT output
        bert_output_dim = self.bert.config.hidden_size
        
        # LSTM for sequential processing
        self.lstm = nn.LSTM(
            input_size=bert_output_dim + 4,  # BERT output + numerical features
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0
        )
        
        # Final prediction layers
        self.fc1 = nn.Linear(lstm_hidden_size, 64)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, input_ids, attention_mask, numerical_features):
        # Process text through BERT
        bert_output = self.bert(input_ids, attention_mask=attention_mask)[0]
        bert_output = self.bert_dropout(bert_output)
        
        # Combine BERT output with numerical features
        combined_features = torch.cat(
            [bert_output, numerical_features.unsqueeze(-1)], dim=-1)
        
        # Process through LSTM
        lstm_out, _ = self.lstm(combined_features)
        
        # Take the last output
        last_lstm = lstm_out[:, -1, :]
        
        # Final prediction
        x = torch.relu(self.fc1(last_lstm))
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output

class ModelTrainer:
    """
    Handles the training process for the BERT-LSTM model, implementing the training
    objective described in Section III.B.3 of the paper.
    """
    def __init__(self, model, learning_rate=2e-5, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            self.optimizer.zero_grad()
            
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            numerical_features = batch['numerical_features'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask, numerical_features)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                numerical_features = batch['numerical_features'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, numerical_features)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load preprocessed data
    price_data = pd.read_csv('bitcoin.csv')
    sentiment_data = pd.read_csv('processed_tweets.csv')
    
    # Initialize BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    
    # Create dataset
    sequence_length = 10  # As specified in paper
    dataset = CryptoDataset(price_data, sentiment_data, sequence_length, tokenizer)
    
    # Create time series cross-validation splits
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Training parameters
    batch_size = 32
    num_epochs = 100
    learning_rate = 2e-5
    
    # Initialize model
    model = BertLSTM(bert_model)
    trainer = ModelTrainer(model, learning_rate=learning_rate, device=device)
    
    # Training loop with cross-validation
    for fold, (train_idx, val_idx) in enumerate(tscv.split(dataset)):
        logger.info(f"Starting fold {fold + 1}")
        
        # Create data loaders for this fold
        train_loader = DataLoader(
            torch.utils.data.Subset(dataset, train_idx),
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            torch.utils.data.Subset(dataset, val_idx),
            batch_size=batch_size
        )
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            train_loss = trainer.train_epoch(train_loader)
            val_loss = trainer.validate(val_loader)
            
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"Training Loss: {train_loss:.4f}")
            logger.info(f"Validation Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'best_model_fold_{fold}.pt')
                
            # Early stopping logic could be added here
                
    logger.info("Training complete!")

if __name__ == "__main__":
    main()