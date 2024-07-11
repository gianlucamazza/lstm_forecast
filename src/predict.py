import argparse
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
import time
import os
import sys
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import load_model
from src.data_loader import preprocess_data, get_data
from src.utils import load_json
from src.logger import setup_logger
from typing import List

# Set up logger
logger = setup_logger('predict_logger', 'logs/predict.log')

# Set device
device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Device: {device}")

def predict(_model: nn.Module, _x: np.ndarray, _scaler: StandardScaler, future_days: int, _features: List) -> tuple[np.ndarray, np.ndarray]:
    _model.eval()
    with torch.no_grad():
        logger.info("Making predictions on the input data.")
        _X_tensor = torch.tensor(_x, dtype=torch.float32).to(device)
        predictions = _model(_X_tensor).cpu().numpy()

        # Prepare for inverse transformation
        predictions_reshaped = np.zeros((_x.shape[0], len(_features) + 1))
        predictions_reshaped[:, 0] = predictions[:, 0]
        predictions_reshaped = np.pad(predictions_reshaped, ((0, 0), (0, len(_scaler.scale_) - len(predictions_reshaped[0]))), 'constant')
        predictions = _scaler.inverse_transform(predictions_reshaped)[:, 0]
        logger.info(f"Predictions: {predictions}")

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

        future_predictions_reshaped = np.zeros((future_days, len(_features) + 1))
        future_predictions_reshaped[:, 0] = future_predictions
        future_predictions_reshaped = np.pad(future_predictions_reshaped, ((0, 0), (0, len(_scaler.scale_) - len(future_predictions_reshaped[0]))), 'constant')
        future_predictions = _scaler.inverse_transform(future_predictions_reshaped)[:, 0]
        logger.info(f"Future predictions: {future_predictions}")

    return predictions, future_predictions

def plot_predictions(symbol: str, filename: str, candles: pd.DataFrame, _predictions: np.ndarray, _future_predictions: np.ndarray, _data: pd.DataFrame, _freq: str) -> None:
    logger.info("Plotting predictions")

    # Use only the necessary portion of the historical data for the given data_sampling_interval
    interval_value = int(_freq[:-1]) if _freq[:-1].isdigit() else 1
    interval_unit = _freq[-1]
    interval = pd.Timedelta(f"{interval_value}{interval_unit}")
    start_date = _data.index[-1] - (len(_predictions) * interval)
    filtered_data = _data.loc[start_date:]

    # Check for missing data
    if filtered_data.isnull().values.any():
        logger.warning("There are missing values in the historical data which may affect the accuracy of the plot.")
        filtered_data = filtered_data.dropna()

    # Ensure there is enough data to plot
    if len(filtered_data) < len(_predictions):
        logger.error("Not enough historical data to plot the predictions.")
        return

    # Create candlestick chart using historical data (Open, High, Low, Close)
    fig = go.Figure(data=[go.Candlestick(x=filtered_data.index, open=filtered_data['Open'], high=filtered_data['High'], low=filtered_data['Low'], close=filtered_data['Close'], name='Market Data')])

    # Add predictions
    prediction_dates = filtered_data.index[-len(_predictions):]
    future_dates = pd.date_range(prediction_dates[-1], periods=len(_future_predictions) + 1, freq=_freq)[1:]

    fig.add_trace(go.Scatter(x=prediction_dates, y=_predictions, mode='lines', name='Predictions', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=future_dates, y=_future_predictions, mode='lines', name='Future Predictions', line=dict(color='orange', dash='dot')))

    # Update layout for a professional appearance
    fig.update_layout(
        title=f'{symbol} Predictions',
        xaxis_title='Date' if 'd' in _freq else 'Date/Time',
        yaxis_title='Price',
        xaxis_rangeslider_visible=True,
        template='plotly_white'
    )

    # Save the plot as an HTML file for interactivity
    fig.write_html(filename)
    logger.info(f"Interactive plot saved to {filename}")

def main(_ticker: str, _symbol: str, _asset_type: str, _data_sampling_interval: str, _targets: List[str], _start_date: str, _model_dir: str, _model_params: dict, _look_back: int, _look_forward: int, _best_features: List, _indicator_windows: dict, _data_resampling_frequency: str) -> None:
    logger.info(f"Getting data for {_symbol} from {_start_date}")
    historical_data, features = get_data(_ticker, _symbol, asset_type=_asset_type, start=_start_date, end=time.strftime('%Y-%m-%d'), windows=_indicator_windows, data_sampling_interval=_data_sampling_interval, data_resampling_frequency=_data_resampling_frequency)
    logger.info(f"Preprocessing data")
    x, _, scaler, selected_features = preprocess_data(historical_data, _targets, look_back=_look_back, look_forward=_look_forward, features=features, best_features=_best_features)
    logger.info(f"Loaded model from {_model_dir}")
    model = load_model(_symbol, _model_dir, len(selected_features), _model_params)
    logger.info(f"Making predictions")
    predictions, future_predictions = predict(model, x, scaler, _look_forward, selected_features)

    # Create candles DataFrame
    candles = historical_data[['Open', 'High', 'Low', 'Close']]

    # Plot predictions
    plot_predictions(_symbol, f'html/{_symbol}_predictions.html', candles, predictions, future_predictions, historical_data, _data_resampling_frequency)

    # Create report
    report = pd.DataFrame({
        'Date': pd.date_range(historical_data.index[-1], periods=_look_forward + 1, freq=_data_resampling_frequency)[1:],
        'Predicted Price': future_predictions
    })

    report.to_csv(f'reports/{_symbol}_predictions.csv', index=False)
    logger.info(f"Predictions report saved to reports/{_symbol}_predictions.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    args = parser.parse_args()

    # Load configuration
    config = load_json(args.config)

    ticker = config['ticker']
    symbol = config['symbol']
    asset_type = config['asset_type']
    data_sampling_interval = config['data_sampling_interval']
    model_dir = config['model_dir']
    model_params = config.get('model_params', {})
    start_date = config['start_date']
    look_back = config['look_back']
    look_forward = config['look_forward']
    best_features = config.get('best_features', [])
    targets = config.get('targets', ['Close'])
    data_resampling_frequency = config['data_resampling_frequency']
    indicator_windows = config['indicator_windows']

    logger.info(f"Starting prediction for {ticker}")
    main(_ticker=ticker, _symbol=symbol, _asset_type=asset_type, _data_sampling_interval=data_sampling_interval, _targets=targets, _start_date=start_date, _model_dir=model_dir, _model_params=model_params, _look_back=look_back, _look_forward=look_forward, _best_features=best_features, _indicator_windows=indicator_windows, _data_resampling_frequency=data_resampling_frequency)
    logger.info(f"Prediction for {symbol} completed")
