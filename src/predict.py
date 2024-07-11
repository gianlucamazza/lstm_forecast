import argparse
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
from src.logger import setup_logger
from src.config import load_config
from typing import List, Tuple

# Set up logger
logger = setup_logger('predict_logger', 'logs/predict.log')

# Set device
device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Device: {device}")


def predict(_model: nn.Module, _x: np.ndarray, _scaler: StandardScaler, future_days: int, _features: List[str],
            _targets: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    _model.eval()
    num_targets = len(_targets)

    logger.info(f"Starting prediction for {num_targets} targets over {future_days} future days.")

    with torch.no_grad():
        logger.info("Making predictions on the input data.")
        _X_tensor = torch.tensor(_x, dtype=torch.float32).to(device)
        predictions = _model(_X_tensor).cpu().numpy()

        logger.info(f"Initial predictions shape: {predictions.shape}")

        # Prepare for inverse transformation
        predictions_reshaped = np.zeros((_x.shape[0], len(_features) + num_targets))
        predictions_reshaped[:, :num_targets] = predictions
        logger.debug(f"Predictions reshaped for inverse transform: {predictions_reshaped.shape}")

        predictions_reshaped = np.pad(predictions_reshaped,
                                      ((0, 0),
                                       (0, len(_scaler.scale_) - predictions_reshaped.shape[1])), 'constant')
        predictions = _scaler.inverse_transform(predictions_reshaped)[:, :num_targets]
        logger.info(f"Inverse transformed predictions: {predictions.shape}")

        # Forecast future prices
        future_predictions = []
        for day in range(future_days):
            _X_tensor = torch.tensor(_x[-1:], dtype=torch.float32).to(device)
            future_pred = _model(_X_tensor).cpu().numpy()[0]
            future_predictions.append(future_pred)

            logger.debug(f"Day {day + 1} future prediction: {future_pred}")

            new_row = np.zeros((1, _x.shape[2]))
            logger.debug(f"New row shape: {new_row.shape}, future_pred shape: {future_pred.shape}")

            if new_row.shape[1] < len(future_pred):
                new_row = np.zeros((1, len(future_pred)))
                logger.debug(f"Adjusted new row shape: {new_row.shape}")

            new_row[0, :num_targets] = future_pred
            if new_row.shape[1] > num_targets:
                new_row[0, num_targets:] = _x[-1, -1, num_targets:]

            logger.debug(f"New row for future prediction: {new_row}")

            if _x[-1][1:].shape[1] != new_row.shape[1]:
                logger.debug(
                    f"Adjusting shapes for concatenation. _x[-1][1:].shape: {_x[-1][1:].shape}, new_row.shape: {new_row.shape}")
                if _x[-1][1:].shape[1] < new_row.shape[1]:
                    new_row = new_row[:, :_x[-1][1:].shape[1]]
                else:
                    new_row_padded = np.zeros((_x[-1][1:].shape[0], _x[-1][1:].shape[1]))
                    new_row_padded[:, :new_row.shape[1]] = new_row
                    new_row = new_row_padded

            # Debug the dimensions before concatenation
            logger.debug(f"_x[-1][1:].shape: {_x[-1][1:].shape}, new_row.shape: {new_row.shape}")

            _x = np.append(_x, [np.vstack((_x[-1][1:], new_row))], axis=0)
            logger.debug(f"Updated input data shape: {_x.shape}")

        future_predictions_reshaped = np.zeros((future_days, len(_features) + num_targets))
        future_predictions_reshaped[:, :num_targets] = future_predictions
        logger.debug(f"Future predictions reshaped: {future_predictions_reshaped.shape}")

        future_predictions_reshaped = np.pad(future_predictions_reshaped,
                                             ((0, 0), (0, len(_scaler.scale_) - future_predictions_reshaped.shape[1])),
                                             'constant')
        future_predictions = _scaler.inverse_transform(future_predictions_reshaped)[:, :num_targets]
        logger.info(f"Inverse transformed future predictions: {future_predictions.shape}")

    logger.info("Prediction completed.")
    return predictions, future_predictions


def plot_predictions(symbol: str, filename: str, candles: pd.DataFrame, _predictions: np.ndarray,
                     _future_predictions: np.ndarray, _data: pd.DataFrame, _freq: str, _interval: str,
                     _targets: list[str]) -> None:
    logger.info("Plotting predictions")

    # Define the start and end date for the future predictions
    start_date = candles.index[-1]
    end_date = pd.to_datetime(candles.index[-1]) + pd.Timedelta(
        days=len(_future_predictions) * (1 if 'd' in _interval else 1 / 24))
    date_range = pd.date_range(start=start_date, periods=len(_future_predictions), freq=_freq)

    # Create the candlestick chart
    fig = go.Figure()

    # Add candlestick data
    fig.add_trace(go.Candlestick(
        x=candles.index,
        open=candles['Open'],
        high=candles['High'],
        low=candles['Low'],
        close=candles['Close'],
        name='Candlestick'
    ))

    # Add historical predictions for each target
    target_names = _targets
    colors = ['blue', 'green', 'orange', 'purple']
    for i, target_name in enumerate(target_names):
        fig.add_trace(go.Scatter(
            x=candles.index[-len(_predictions):],
            y=_predictions[:, i],
            mode='lines',
            name=f'Predictions {target_name}',
            line=dict(color=colors[i])
        ))

    # Add future predictions for each target
    for i, target_name in enumerate(target_names):
        fig.add_trace(go.Scatter(
            x=date_range,
            y=_future_predictions[:, i],
            mode='lines',
            name=f'Future Predictions {target_name}',
            line=dict(color=colors[i], dash='dash')
        ))

    # Update layout for a professional appearance
    fig.update_layout(
        title=f'{symbol} - Predictions',
        xaxis_title='Date' if 'd' in _interval else 'Date/Time',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        xaxis=dict(
            tickformat='%Y-%m-%d %H:%M' if 'h' in _interval else '%Y-%m-%d'
        )
    )

    # Save the plot as an HTML file for interactivity
    fig.write_html(filename)
    logger.info(f"Predictions plot saved to {filename}")


def main(_ticker: str, _symbol: str, _asset_type: str, _data_sampling_interval: str, _targets: List[str],
         _start_date: str, _model_dir: str, _model_params: dict, _look_back: int, _look_forward: int,
         _best_features: List, _indicator_windows: dict, _data_resampling_frequency: str) -> None:
    logger.info(f"Getting data for {_symbol} from {_start_date}")
    historical_data, features = get_data(_ticker, _symbol, asset_type=_asset_type, start=_start_date,
                                         end=time.strftime('%Y-%m-%d'), windows=_indicator_windows,
                                         data_sampling_interval=_data_sampling_interval,
                                         data_resampling_frequency=_data_resampling_frequency)
    logger.info(f"Preprocessing data")
    x, _, scaler, selected_features = preprocess_data(historical_data, _targets, look_back=_look_back,
                                                      look_forward=_look_forward, features=features,
                                                      best_features=_best_features)
    logger.info(f"Loaded model from {_model_dir}")
    model = load_model(_symbol, _model_dir, len(selected_features), _model_params)
    logger.info(f"Making predictions")
    predictions, future_predictions = predict(model, x, scaler, _look_forward, selected_features, _targets)

    # Create candles DataFrame
    candles = historical_data[['Open', 'High', 'Low', 'Close']]

    # Plot predictions
    plot_predictions(_symbol, f'docs/{_symbol}_{config.best_features}_predictions.html', candles, predictions,
                     future_predictions, historical_data, _data_resampling_frequency, _data_sampling_interval, _targets)

    # Create report
    report = pd.DataFrame(data=future_predictions, columns=_targets)
    report.index = pd.date_range(start=candles.index[-1], periods=len(future_predictions),
                                 freq=_data_resampling_frequency)

    report.to_csv(f'reports/{_symbol}_predictions.csv', index=False)
    logger.info(f"Predictions report saved to reports/{_symbol}_predictions.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    logger.info(f"Starting prediction for {config.ticker}")
    main(_ticker=config.ticker, _symbol=config.symbol, _asset_type=config.asset_type,
         _data_sampling_interval=config.data_sampling_interval, _targets=config.targets,
         _start_date=config.start_date, _model_dir=config.model_dir, _model_params=config.model_params,
         _look_back=config.look_back, _look_forward=config.look_forward, _best_features=config.best_features,
         _indicator_windows=config.indicator_windows, _data_resampling_frequency=config.data_resampling_frequency)
    logger.info(f"Prediction for {config.symbol} completed")
