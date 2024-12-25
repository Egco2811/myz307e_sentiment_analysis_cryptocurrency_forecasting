# Cryptocurrency Price Prediction using BERT-LSTM and Sentiment Analysis

## Overview
This project implements a sophisticated cryptocurrency price prediction system that combines BERT-based sentiment analysis of social media data with LSTM-based time series forecasting. The implementation follows the methodology described in our research paper, providing a complete pipeline from data preprocessing through model training to trading strategy evaluation.

## Features
- Sentiment analysis of cryptocurrency-related tweets using BERT
- Advanced feature engineering combining price and sentiment data
- LSTM-based price prediction with bidirectional processing
- Hyperparameter optimization using Optuna
- Comprehensive trading strategy implementation
- Detailed visualization and evaluation metrics

## Installation

### Requirements
- Python 3.8+
- PyTorch 1.9+
- Transformers 4.5+
- pandas
- numpy
- scikit-learn
- optuna
- matplotlib
- seaborn

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/crypto-prediction.git
cd crypto-prediction

Install dependencies:

bashCopypip install -r requirements.txt

Download required NLTK data:

pythonCopyimport nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
Project Structure
Copycrypto-prediction/
├── data/                      # Data directory
├── models/                    # Saved models
├── outputs/                   # Results and visualizations
├── src/                      # Source code
│   ├── tweets_preprocessing.py
│   ├── tweet_labeler.py
│   ├── csv_date_matcher.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── data_visualization.py
│   ├── feature_engineering.py
│   ├── hyperparameter_tuning.py
│   ├── trading_strategy.py
│   └── run_script.py
├── config.json               # Configuration file
├── requirements.txt
└── README.md
Usage
Configuration
Before running the pipeline, configure the parameters in config.json:
jsonCopy{
    "output_directory": "outputs",
    "raw_data": {
        "tweets_file": "data/raw/tweets.csv",
        "price_file": "data/raw/prices.csv"
    },
    "model": {
        "bert_model": "bert-base-uncased",
        "sequence_length": 10,
        "batch_size": 32
    },
    "trading": {
        "initial_capital": 100000,
        "risk_per_trade": 0.02
    }
}
Running the Pipeline
Execute the complete pipeline:
bashCopypython src/run_script.py --config config.json
For individual components:
bashCopy# Preprocess tweets
python src/tweets_preprocessing.py

# Train model
python src/model_training.py

# Run trading strategy
python src/trading_strategy.py
Component Descriptions
1. Data Preprocessing

Cleans and normalizes tweet text
Removes duplicates and irrelevant content
Aligns tweet timestamps with price data

2. Sentiment Analysis

Uses BERT for tweet sentiment classification
Generates sentiment scores for each time period
Handles multiple languages and emoji content

3. Feature Engineering

Creates technical indicators from price data
Combines sentiment and price features
Implements rolling windows and momentum indicators

4. Model Architecture

Bidirectional LSTM network
BERT-based sentiment feature extraction
Attention mechanism for temporal dependencies

5. Trading Strategy

Risk management implementation
Position sizing based on volatility
Stop-loss and take-profit mechanisms

Output Files
The pipeline generates the following outputs:

Processed datasets
Trained model weights
Performance metrics
Trading strategy results
Visualization plots

Troubleshooting
Common Issues

CUDA Out of Memory

CopySolution: Reduce batch size in config.json

Missing Data Files

CopySolution: Ensure all required files are in the data/ directory

Training Instability

CopySolution: Adjust learning rate and gradient clipping parameters
Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Citation
If you use this code in your research, please cite:
Copy@article{your_paper,
    title={Advanced Framework for Cryptocurrency Price Forecasting Using BERT and LSTM},
    author={Your Name},
    journal={Your Journal},
    year={2024}
}
Contact
For questions and feedback, please contact your.email@domain.com
