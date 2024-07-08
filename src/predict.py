import argparse
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch
import torch.nn as nn
import time
from sklearn.preprocessing import StandardScaler
from model import PricePredictor
from data_loader import preprocess_data
from feature_engineering import calculate_technical_indicators
from typing import List
from utils import load_json

# Configure logging
logging.basicConfig(filename='../predict.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Set device
device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# Load the trained model
def load_model(path: str, input_shape: int) -> nn.Module:
    model = PricePredictor(input_shape).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# Get historical data
def get_data(_ticker: str, start: str, end: str) -> pd.DataFrame:
    historical_data = yf.download(_ticker, start=start, end=end)
    historical_data = calculate_technical_indicators(historical_data)
    return historical_data


# Make predictions
def predict(_model: nn.Module, _x: np.ndarray, _scaler: StandardScaler, future_days: int, _features: List) \
        -> tuple[np.ndarray, np.ndarray]:
    _model.eval()
    with torch.no_grad():
        _X_tensor = torch.tensor(_x, dtype=torch.float32).to(device)
        predictions = _model(_X_tensor).cpu().numpy()

        # Prepare for inverse transformation
        predictions_reshaped = np.zeros((_x.shape[0], len(_features)))
        predictions_reshaped[:, 0] = predictions[:, 0]
        predictions = _scaler.inverse_transform(predictions_reshaped)[:, 0]

        # Forecast future prices
        future_predictions = []
        for _ in range(future_days):
            _X_tensor = torch.tensor(_x[-1:], dtype=torch.float32).to(device)
            future_pred = _model(_X_tensor).cpu().numpy()[0][0]
            future_predictions.append(future_pred)

            new_row = np.zeros((1, _x.shape[2]))
            new_row[0, 0] = future_pred
            new_row[0, 1:] = _x[-1, -1, 1:]

            _x = np.append(_x, [np.append(_x[-1][1:], new_row, axis=0)], axis=0)

        future_predictions_reshaped = np.zeros((future_days, len(_features)))
        future_predictions_reshaped[:, 0] = future_predictions
        future_predictions = _scaler.inverse_transform(future_predictions_reshaped)[:, 0]

    return predictions, future_predictions


# Plot predictions aggregated with historical data
def plot_predictions(_historical_data: np.ndarray, _predictions: np.ndarray, _future_predictions: np.ndarray,
                     _data: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 7))

    # Keep 60 days of historical data for better visualization
    plt.plot(_data.index[-60:], _historical_data[-60:], label='Historical Prices')

    # Debug: print last date from historical data
    print('Last date from historical data:', _data.index[-1])

    # Create future dates for future predictions
    future_dates = pd.date_range(_data.index[-1], periods=len(_future_predictions) + 1, freq='D')[1:]

    # Debug: print first date from future dates
    print('First date from future dates:', future_dates[0])
    plt.plot(future_dates, _future_predictions, label='Predicted Future Prices', linestyle='dashed')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # Save the plot
    plt.savefig('prediction.png')


# Main function for prediction
def main(_ticker: str, _target: str, _start_date: str, _model_path: str,
         _look_back: int, _look_forward: int, _features: List) -> None:
    data = get_data(_ticker, start=_start_date, end=time.strftime('%Y-%m-%d'))
    x, _, scaler, selected_features = preprocess_data(data, _target, look_back=_look_back, look_forward=_look_forward, features=_features)
    model = load_model(_model_path, input_shape=len(selected_features))
    predictions, future_predictions = predict(model, x, scaler, _look_forward, selected_features)

    plot_predictions(data['Close'].values, predictions, future_predictions, data)

    logging.info('Predictions completed and plotted')


# Execute prediction
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    args = parser.parse_args()

    # Load configuration
    config = load_json(args.config)

    ticker = config['ticker']
    model_path = config['model_path']
    start_date = config['start_date']
    look_back = config['look_back']
    look_forward = config['look_forward']
    features = config['features']
    target = config['target']

    main(ticker, target, start_date, model_path, look_back, look_forward, features)
