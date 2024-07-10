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

def load_model(symbol: str, path: str, input_shape: int) -> nn.Module:
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
    model.load_state_dict(torch.load(path + f'/{symbol}_model.pth'))
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


def plot_predictions(symbol: str, filename: str, _historical_data: np.ndarray, _predictions: np.ndarray, _future_predictions: np.ndarray,
                     _data: pd.DataFrame, _freq: str) -> None:
    """
    Plot the historical data, predictions, and future predictions.

    Args:
        symbol (str): The stock symbol.
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

    plt.title(f'{symbol} {_data.index[0].strftime("%Y-%m-%d")} to {_data.index[-1].strftime("%Y-%m-%d")}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()

    # Save the plot
    plt.savefig(filename)
    logger.info(f'Plot saved to {filename}')


def main(_ticker: str, _symbol: str, _asset_type: str, _interval: str, _target: str, _start_date: str, _model_path: str,
         _look_back: int, _look_forward: int, _best_features: List, _indicator_windows: dict, freq: str) -> None:
    """
    Main function for prediction.

    Args:
        _ticker (str): The stock ticker.
        _symbol (str): The stock symbol.
        _asset_type (str): The asset type.
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
    logger.info(f"Getting data for {_symbol} from {_start_date}")
    historical_data, features = get_data(_ticker, _symbol, asset_type=_asset_type, start=_start_date, end=time.strftime('%Y-%m-%d'), windows=_indicator_windows, interval=_interval)
    logger.info(f"Preprocessing data")
    x, _, scaler, selected_features = preprocess_data(historical_data, _target, look_back=_look_back, look_forward=_look_forward, features=features, best_features=_best_features)
    logger.info(f"Loaded model from {_model_path}")
    model = load_model(_symbol, _model_path, input_shape=len(selected_features))
    logger.info(f"Making predictions")
    predictions, future_predictions = predict(model, x, scaler, _look_forward, selected_features)

    # 7, 30, 90, 365 days
    plot_predictions(_symbol, f'png/{_symbol}_7_days.png', historical_data['Close'].values[-7:], predictions[-7:], future_predictions, historical_data[-7:], freq)
    plot_predictions(_symbol, f'png/{_symbol}_30_days.png', historical_data['Close'].values[-30:], predictions[-30:], future_predictions, historical_data[-30:], freq)
    plot_predictions(_symbol, f'png/{_symbol}_90_days.png', historical_data['Close'].values[-90:], predictions[-90:], future_predictions, historical_data[-90:], freq)
    plot_predictions(_symbol, f'png/{_symbol}_365_days.png', historical_data['Close'].values[-365:], predictions[-365:], future_predictions, historical_data[-365:], freq)
    plot_predictions(_symbol, f'png/{_symbol}_full.png', historical_data['Close'].values, predictions, future_predictions, historical_data, freq)

    logger.info('Predictions completed and plotted')
    
    # Create report
    report = pd.DataFrame({
        'Date': pd.date_range(historical_data.index[-1], periods=_look_forward + 1, freq=freq)[1:],
        'Predicted Price': future_predictions
    })
    
    report.to_csv(f'reports/{_symbol}_predictions.csv', index=False)
    logger.info(f'Report saved to reports/{_symbol}_predictions.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    args = parser.parse_args()

    # Load configuration
    config = load_json(args.config)

    ticker = config['ticker']
    symbol = config['symbol']
    asset_type = config['asset_type']
    interval = config['interval']
    model_path = config['model_path']
    start_date = config['start_date']
    look_back = config['look_back']
    look_forward = config['look_forward']
    best_features = config.get('best_features', None)
    target = config['target']
    freq = config['frequency']
    indicator_windows = config['indicator_windows']

    logger.info(f"Starting prediction for {ticker}")
    main(ticker, symbol, asset_type, interval, target, start_date, model_path, look_back, look_forward, best_features, indicator_windows, freq)
    logger.info(f"Prediction for {symbol} completed")
