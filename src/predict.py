import argparse
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
from sklearn.preprocessing import StandardScaler
from model import PricePredictor
from data_loader import preprocess_data, get_data
from typing import List
from utils import load_json
from logger import setup_logger

# Set up logger
logger = setup_logger('predict_logger', 'logs/predict.log')


# Set device
device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Device: {device}")

def load_model(path: str, input_shape: int) -> nn.Module:
    """
    Load the trained model from a given path.

    Args:
        path (str): The path to the trained model.
        input_shape (int): The input shape of the model.

    Returns:
        nn.Module: The trained model.
    """
    logger.info(f"Loading model from {path} with input shape {input_shape}")
    model = PricePredictor(input_shape).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    logger.info("Model loaded and set to evaluation mode.")
    return model

def predict(_model: nn.Module, _x: np.ndarray, _scaler: StandardScaler, future_days: int, _features: List) \
        -> tuple[np.ndarray, np.ndarray]:
    """
    Make predictions using the trained model.

    Args:
        _model (nn.Module): The trained model.
        _x (np.ndarray): The input data.
        _scaler (StandardScaler): The scaler used to scale the data.
        future_days (int): The number of days to forecast.
        _features (List): The list of features.

    Returns:
        tuple[np.ndarray, np.ndarray]: The predictions and future predictions.
    """
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


def plot_predictions(filename: str, _historical_data: np.ndarray, _predictions: np.ndarray, _future_predictions: np.ndarray,
                     _data: pd.DataFrame, _freq: str = 'B') -> None:
    """
    Plot the historical data, predictions, and future predictions.

    Args:
        filename (str): The filename to save the plot.
        _historical_data (np.ndarray): The historical data.
        _predictions (np.ndarray): The predictions.
        _future_predictions (np.ndarray): The future predictions.
        _data (pd.DataFrame): The data.

    Returns:
        None
    """
    plt.figure(figsize=(14, 7))

    plt.plot(_data.index, _historical_data, label='Historical Prices')

    # Create future dates for future predictions
    aligned_predictions = np.zeros_like(_historical_data)
    aligned_predictions[-len(_predictions):] = _predictions

    plt.plot(_data.index, aligned_predictions, label='Predicted Prices', color='red')

    # Create future dates for future predictions
    future_dates = pd.date_range(_data.index[-1], periods=len(_future_predictions) + 1, freq=_freq)[1:]

    plt.plot(future_dates, _future_predictions, label='Predicted Future Prices', linestyle='dashed')

    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # Save the plot
    plt.savefig(filename)
    logger.info(f'Plot saved to {filename}')


def main(_ticker: str, _target: str, _start_date: str, _model_path: str,
         _look_back: int, _look_forward: int, _features: List, _best_features: List, _indicator_windows: dict) -> None:
    """
    Main function for prediction.

    Args:
        _ticker (str): The ticker symbol.
        _target (str): The target feature.
        _start_date (str): The start date.
        _model_path (str): The path to the trained model.
        _look_back (int): The look back window.
        _look_forward (int): The look forward window.
        _features (List): The list of features.
        _best_features (List): The list of best features.

    Returns:
        None
    """
    logger.info(f"Getting data for {_ticker} from {_start_date}")
    data = get_data(_ticker, start=_start_date, end=time.strftime('%Y-%m-%d'), windows=_indicator_windows)
    logger.info(f"Preprocessing data")
    x, _, scaler, selected_features = preprocess_data(data, _target, look_back=_look_back, look_forward=_look_forward, features=_features, best_features=_best_features)
    logger.info(f"Loaded model from {_model_path}")
    model = load_model(_model_path, input_shape=len(selected_features))
    logger.info(f"Making predictions")
    predictions, future_predictions = predict(model, x, scaler, _look_forward, selected_features)

    plot_predictions(f'png/{_ticker}_90_days.png', data['Close'].values[-90:], predictions[-90:], future_predictions, data[-90:])
    plot_predictions(f'png/{_ticker}_365_days.png', data['Close'].values[-365:], predictions[-365:], future_predictions, data[-365:])
    plot_predictions(f'png/{_ticker}_full.png', data['Close'].values, predictions, future_predictions, data)

    logger.info('Predictions completed and plotted')


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
    best_features = config.get('best_features', None)
    target = config['target']
    indicator_windows = config['indicator_windows']

    logger.info(f"Starting prediction for {ticker}")
    main(ticker, target, start_date, model_path, look_back, look_forward, features, best_features, indicator_windows)
    logger.info(f"Prediction for {ticker} completed")
