import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # 1) Load and filter data
    df = pd.read_csv('bitcoin.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    start_date = '2021-03-01'
    end_date   = '2022-07-31'
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df = df[mask].copy().reset_index(drop=True)
    logger.info(f"ARIMA: Rows after filtering: {len(df)}")

    if len(df) < 10:
        logger.error("Not enough data points to proceed for ARIMA. Exiting.")
        return

    
    close_prices = df['Close'].values

    # 2) We define the final test set 
    test_size = 3
    if test_size > len(close_prices):
        logger.error("Not enough data to carve out 3 test points. Exiting.")
        return

    train_val_data = close_prices[:-test_size]  # everything except the last 3
    test_data      = close_prices[-test_size:]  # last 3 points

    logger.info(f"ARIMA Train+Val size: {len(train_val_data)}, Test size: {len(test_data)}")

    # 3) Fit ARIMA on entire train+val portion
    # Example ARIMA(1,1,1) for demonstration
    try:
        model = sm.tsa.ARIMA(train_val_data, order=(1,1,1))
        model_fit = model.fit()
    except Exception as e:
        logger.error(f"ARIMA fit failed: {e}")
        return

    # 4) Forecast exactly 3 steps (the test set length)
    forecast_vals = model_fit.forecast(steps=test_size)
    if len(forecast_vals) != test_size:
        logger.error("Forecast length mismatch.")
        return

    # 5) Compute metrics on these 3 points
    mse  = np.mean((forecast_vals - test_data) ** 2)
    rmse = np.sqrt(mse)
    if np.any(test_data == 0):
        mape = float('inf')
    else:
        mape = np.mean(np.abs((forecast_vals - test_data) / test_data)) * 100

    # 6) Plot index-based comparison for these 3 data points
    x_idx = range(test_size)
    plt.figure(figsize=(6,4))
    plt.plot(x_idx, test_data, label='Actual', marker='o', color='black')
    plt.plot(x_idx, forecast_vals, label='Predicted', marker='o', color='blue', linestyle='--')
    plt.xlabel('Test Sample Index (last 3 points)')
    plt.ylabel('BTC Price (USD)')
    plt.title('ARIMA: Last 3-Point Test Prediction')
    plt.legend()
    plt.tight_layout()
    plt.savefig('arima_prediction_plot.png', dpi=300)
    plt.close()

    logger.info(f"ARIMA (3-point test) => MSE={mse:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")

    # 7) Save final results to .npy
    final_results = {
        'predictions': forecast_vals,
        'actuals': test_data,
        'metrics': {
            'mse': float(mse),
            'rmse': float(rmse),
            'mape': float(mape)
        },
        'cv_results': None  # or store cross-validation info if desired
    }
    np.save('final_results_arima.npy', final_results, allow_pickle=True)
    logger.info("Saved final_results_arima.npy. Done.")

if __name__ == "__main__":
    main()
