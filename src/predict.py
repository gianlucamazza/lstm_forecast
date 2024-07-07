import argparse
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch
import torch.nn as nn
import time
import json
from sklearn.preprocessing import StandardScaler
from model import PricePredictor
from data_loader import preprocess_data

# Configure logging
logging.basicConfig(filename='../predict.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Set device
device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# Load the trained model
def load_model(path: str) -> nn.Module:
    model = PricePredictor().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# Get historical data
def get_data(_ticker: str, start: str, end: str) -> pd.DataFrame:
    historical_data = yf.download(_ticker, start=start, end=end)
    return historical_data


# Make predictions
def predict(_model: nn.Module, _x: np.ndarray, _scaler: StandardScaler, future_days: int) \
        -> tuple[np.ndarray, np.ndarray]:
    _model.eval()
    with torch.no_grad():
        _X_tensor = torch.tensor(_x, dtype=torch.float32).to(device)
        predictions = _model(_X_tensor).cpu().numpy()

        # Prepare for inverse transformation
        predictions_reshaped = np.zeros((_x.shape[0], 3))
        predictions_reshaped[:, 0] = predictions[:, 0]
        predictions = _scaler.inverse_transform(predictions_reshaped)[:, 0]

        # Forecast future prices
        future_predictions = []
        for _ in range(future_days):
            _X_tensor = torch.tensor(_x[-1:], dtype=torch.float32).to(device)
            future_pred = _model(_X_tensor).cpu().numpy()[0][0]
            future_predictions.append(future_pred)

            new_row = np.array([[future_pred, _x[-1][-1][1], _x[-1][-1][2]]])
            _x = np.append(_x, [np.append(_x[-1][1:], new_row, axis=0)], axis=0)

        future_predictions_reshaped = np.zeros((future_days, 3))
        future_predictions_reshaped[:, 0] = future_predictions
        future_predictions = _scaler.inverse_transform(future_predictions_reshaped)[:, 0]

    return predictions, future_predictions


# Plot predictions aggregated with historical data
def plot_predictions(_historical_data: np.ndarray, _predictions: np.ndarray, _future_predictions: np.ndarray,
                     _data: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 7))
    plt.plot(_data.index, _historical_data, label='Historical Data')

    # Create future dates for future predictions
    future_dates = pd.date_range(_data.index[-1], periods=len(_future_predictions) + 1, freq='D')[1:]
    plt.plot(future_dates, _future_predictions, label='Predicted Future Prices', linestyle='dashed')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # Save the plot
    plt.savefig('prediction.png')


# Main function for prediction
def main(_ticker: str, _start_date: str, _model_path: str, _look_back: int, _look_forward: int) -> None:
    data = get_data(_ticker, start=_start_date, end=time.strftime('%Y-%m-%d'))
    x, _, scaler = preprocess_data(data, look_back=_look_back, look_forward=_look_forward)
    model = load_model(_model_path)
    predictions, future_predictions = predict(model, x, scaler, _look_forward)

    plot_predictions(data['Close'].values, predictions, future_predictions, data)

    logging.info('Predictions completed and plotted')


# Execute prediction
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, required=True, help='The stock ticker symbol')
    parser.add_argument('--model_path', type=str, required=True, help='The path to the trained model')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = json.load(config_file)

    start_date = config['start_date']
    look_back = config['look_back']
    look_forward = config['look_forward']

    main(args.ticker, start_date, args.model_path, look_back, look_forward)
