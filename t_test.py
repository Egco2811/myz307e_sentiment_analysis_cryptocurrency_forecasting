
import numpy as np
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_cv_mses(npy_file: str, key: str = 'cv_results'):
   
    data = np.load(npy_file, allow_pickle=True).item()
    if key not in data:
        logger.warning(f"{key} not found in {npy_file}.")
        return None
    
    cv_info = data[key]
    mses = []
    # We'll assume each element is a dict with 'final_val_loss' or similar
    for fold_item in cv_info:
        if 'final_val_loss' in fold_item:
            mses.append(fold_item['final_val_loss'])
        elif 'mse' in fold_item:
            mses.append(fold_item['mse'])
        else:
            logger.warning(f"No recognized MSE key in fold_item: {fold_item}")
    return np.array(mses, dtype=float)

def main():
    # 1) Load fold-level results for each model
    logger.info("Loading cross-validation / rolling-window metrics from each model...")
    
    # ARIMA
    #arima_file = 'final_results_arima.npy'
    #arima_cv_mses = load_cv_mses(arima_file)
    
    # LSTM-only
    #lstm_file = 'final_results_standard_lstm.npy'
    #lstm_cv_mses = load_cv_mses(lstm_file)
    
    # Lexicon-LSTM
    lex_file = 'cv_results_lexicon_lstm.npy'
    lex_cv_mses = load_cv_mses(lex_file)
    
    # BERT-LSTM
    bert_file = 'c_results.npy'  # BERT-LSTM
    bert_cv_mses = load_cv_mses(bert_file)
    
    # 2) Check for None or length mismatch
    def safe_len(x):
        return len(x) if x is not None else 0
    
    logger.info(f"Lengths: ARIMA={safe_len(arima_cv_mses)}, "
                f"LSTM-only={safe_len(lstm_cv_mses)}, "
                f"Lexicon-LSTM={safe_len(lex_cv_mses)}, "
                f"BERT-LSTM={safe_len(bert_cv_mses)}")

    

    # 3) Paired t-test for each baseline vs. BERT
    def paired_ttest(baseline_name, baseline_array, bert_array):
        if baseline_array is None or bert_array is None:
            logger.warning(f"Skipping {baseline_name} due to missing data.")
            return
        if len(baseline_array) != len(bert_array):
            logger.warning(f"Skipping {baseline_name}, length mismatch with BERT-LSTM.")
            return
        
        
        t_stat, p_val = stats.ttest_rel(baseline_array, bert_array)
        diff_mean = np.mean(baseline_array - bert_array)
        logger.info(f"\nPaired t-test: {baseline_name} vs. BERT-LSTM")
        logger.info(f"Mean difference (baseline - BERT) = {diff_mean:.6f}")
        logger.info(f"t-statistic = {t_stat:.4f}, p-value = {p_val:.6f}")
        if p_val < 0.05:
            logger.info("=> The difference is statistically significant at p < 0.05.\n")
        else:
            logger.info("=> Not significant at p < 0.05.\n")
    
    # 4) Run tests
   
    paired_ttest("Lexicon-LSTM", lex_cv_mses, bert_cv_mses)

    logger.info("Significance test complete.")

if __name__ == "__main__":
    main()
