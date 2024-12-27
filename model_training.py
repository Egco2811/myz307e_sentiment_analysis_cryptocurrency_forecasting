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
from typing import Dict, List, Tuple
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class CryptoDataset(Dataset):
    """
    Custom dataset class for cryptocurrency data that combines price and sentiment
    features as specified in Section III.A of the paper.
    """
    def __init__(self, price_data: pd.DataFrame, 
                 sentiment_data: pd.DataFrame, 
                 sequence_length: int,
                 bert_tokenizer,
                 max_length: int = 128):
        self.sequence_length = sequence_length
        self.tokenizer = bert_tokenizer
        self.max_length = max_length
        
        # Combine price and sentiment data
        self.data = self._prepare_data(price_data, sentiment_data)
        
    def _prepare_data(self, price_data: pd.DataFrame, 
                     sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare combined dataset with both price and sentiment features.
        Implements data alignment described in Section III.A.
        """
        # Merge price and sentiment data
        combined_data = pd.merge(price_data, sentiment_data, 
                               on='Date', how='inner')
        
        # Calculate daily returns
        combined_data['returns'] = combined_data['Close'].pct_change()
        
        # Calculate rolling sentiment metrics
        combined_data['rolling_sentiment'] = combined_data['sentiment_score'].rolling(
            window=7).mean()
        
        # Normalize data
        numerical_columns = ['Close', 'returns', 'sentiment_score', 
                           'rolling_sentiment']
        for column in numerical_columns:
            min_val = combined_data[column].min()
            max_val = combined_data[column].max()
            combined_data[column] = (combined_data[column] - min_val) / (
                max_val - min_val)
            
        return combined_data
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
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
        
        # Prepare numerical features
        numerical_features = sequence[
            ['Close', 'returns', 'sentiment_score', 'rolling_sentiment']
        ].values
        
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
    def __init__(self, 
                 bert_model: BertModel,
                 lstm_hidden_size: int = 128,
                 num_lstm_layers: int = 3,
                 dropout_rate: float = 0.3):
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
    def __init__(self, model: BertLSTM, config: dict, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['model']['learning_rate']
        )
        self.criterion = nn.MSELoss()
        self.early_stopping_patience = config['model']['early_stopping_patience']
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model and calculate directional accuracy.
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                numerical_features = batch['numerical_features'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, numerical_features)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.cpu().numpy())
        
        # Calculate directional accuracy
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actuals)
        directional_accuracy = np.mean(pred_direction == actual_direction)
        
        return total_loss / len(val_loader), directional_accuracy

    def train_with_early_stopping(self, 
                                train_loader: DataLoader,
                                val_loader: DataLoader,
                                num_epochs: int) -> Dict:
        """
        Train the model with early stopping and return training history.
        """
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'directional_accuracy': []
        }
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, directional_acc = self.validate(val_loader)
            
            # Update training history
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_loss)
            training_history['directional_accuracy'].append(directional_acc)
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Directional Accuracy: {directional_acc:.4f}"
            )
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model('best_model.pt')
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
                
        return training_history
    
    def save_model(self, filepath: str):
        """Save model state dict."""
        torch.save(self.model.state_dict(), filepath)
        
    def load_model(self, filepath: str):
        """Load model state dict."""
        self.model.load_state_dict(torch.load(filepath))

def main():
    """Main function to demonstrate the training process."""
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
        
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    
    # Load and prepare data
    price_data = pd.read_csv(config['raw_data']['price_file'])
    sentiment_data = pd.read_csv(config['raw_data']['tweets_file'])
    
    # Create datasets
    train_dataset = CryptoDataset(
        price_data=price_data,
        sentiment_data=sentiment_data,
        sequence_length=config['model']['sequence_length'],
        bert_tokenizer=tokenizer
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['model']['batch_size'],
        shuffle=True
    )
    
    # Initialize model and trainer
    model = BertLSTM(bert_model)
    trainer = ModelTrainer(model, config, device)
    
    # Train model
    history = trainer.train_with_early_stopping(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['model']['num_epochs']
    )
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
