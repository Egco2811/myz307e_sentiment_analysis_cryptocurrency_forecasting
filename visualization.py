import numpy as np
import matplotlib.pyplot as plt

def load_predictions_and_actuals(npy_file):
   
    data = np.load(npy_file, allow_pickle=True).item()
    pred = np.array(data['predictions']).flatten()
    act  = np.array(data['actuals']).flatten()
    return pred, act

def main():
    
    arima_file   = 'final_results_arima.npy'
    lstm_file    = 'final_results_standard_lstm.npy'
    lexicon_file = 'final_results_lexicon_lstm.npy'
    bert_file    = 'final_results .npy'  

    # 2) Load them
    arima_pred, arima_act     = load_predictions_and_actuals(arima_file)
    lstm_pred, lstm_act       = load_predictions_and_actuals(lstm_file)
    lexicon_pred, lexicon_act = load_predictions_and_actuals(lexicon_file)
    bert_pred, bert_act       = load_predictions_and_actuals(bert_file)

    # 3) Setup figure with 4 subplots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # axes is a 2D array: axes[row][col]

    # 4) ARIMA subplot
    ax_arima = axes[0][0]
    x_arima = range(len(arima_pred))  # index 0..N-1
    ax_arima.plot(x_arima, arima_act, label='Actual', color='black')
    ax_arima.plot(x_arima, arima_pred, label='Prediction', color='blue', linestyle='--')
    ax_arima.set_title('ARIMA (Baseline)')
    ax_arima.set_xlabel('Sample Index')
    ax_arima.set_ylabel('BTC Price')
    ax_arima.legend()
    ax_arima.grid(True, alpha=0.3)

    # 5) LSTM-only subplot
    ax_lstm = axes[0][1]
    x_lstm = range(len(lstm_pred))
    ax_lstm.plot(x_lstm, lstm_act, label='Actual', color='black')
    ax_lstm.plot(x_lstm, lstm_pred, label='Prediction', color='orange', linestyle='--')
    ax_lstm.set_title('LSTM-only')
    ax_lstm.set_xlabel('Sample Index')
    ax_lstm.set_ylabel('BTC Price')
    ax_lstm.legend()
    ax_lstm.grid(True, alpha=0.3)

    # 6) Lexicon-LSTM subplot
    ax_lex = axes[1][0]
    x_lex = range(len(lexicon_pred))
    ax_lex.plot(x_lex, lexicon_act, label='Actual', color='black')
    ax_lex.plot(x_lex, lexicon_pred, label='Prediction', color='green', linestyle='--')
    ax_lex.set_title('Lexicon-LSTM')
    ax_lex.set_xlabel('Sample Index')
    ax_lex.set_ylabel('BTC Price')
    ax_lex.legend()
    ax_lex.grid(True, alpha=0.3)

    # 7) BERT-LSTM subplot
    ax_bert = axes[1][1]
    x_bert = range(len(bert_pred))
    ax_bert.plot(x_bert, bert_act, label='Actual', color='black')
    ax_bert.plot(x_bert, bert_pred, label='Prediction', color='red', linestyle='--')
    ax_bert.set_title('BERT-LSTM (Ours)')
    ax_bert.set_xlabel('Sample Index')
    ax_bert.set_ylabel('BTC Price')
    ax_bert.legend()
    ax_bert.grid(True, alpha=0.3)

    # 8) Final layout & save
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.close()

    print("Saved figure 'model_comparison.png' with four subplots (ARIMA, LSTM-only, "
          "Lexicon-LSTM, and BERT-LSTM) comparing predicted vs. actual on index-based axes.")

if __name__ == "__main__":
    main()
