import argparse
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from lstm_forecast.model import load_model
from lstm_forecast.data_loader import preprocess_data, get_data
from lstm_forecast.predict_utils import inverse_transform_predictions, plot_predictions, save_predictions_report
from lstm_forecast.logger import setup_logger
from lstm_forecast.config import Config

# Set up logger
logger = setup_logger("predict_logger", "logs/predict.log")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

def predict(
    _model: nn.Module,
    _x: np.ndarray,
    scaler_prices: StandardScaler,
    scaler_volume: MinMaxScaler,
    future_days: int,
    _features: List[str],
    _targets: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    _model.eval()
    num_targets = len(_targets)

    logger.info(
        f"Starting prediction for {num_targets} targets over {future_days} future days."
    )

    with torch.no_grad():
        logger.info("Making predictions on the input data.")
        _X_tensor = torch.tensor(_x, dtype=torch.float32).to(device)
        predictions = _model(_X_tensor).cpu().numpy()

        logger.info(f"Initial predictions shape: {predictions.shape}")

        predictions = inverse_transform_predictions(
            predictions, scaler_prices, scaler_volume, _features, num_targets
        )
        logger.info(f"Inverse transformed predictions: {predictions.shape}")

        # Forecast future prices
        last_x = _x[-1]
        future_predictions = []

        for _ in range(future_days):
            _X_tensor = torch.tensor(last_x[np.newaxis, :], dtype=torch.float32).to(device)
            future_pred = _model(_X_tensor).cpu().numpy()[0]
            future_predictions.append(future_pred)

            new_row = np.zeros((1, last_x.shape[1]))
            new_row[0, :num_targets] = future_pred

            if new_row.shape[1] > num_targets:
                new_row[0, num_targets:] = last_x[-1, num_targets:]

            last_x = np.vstack((last_x[1:], new_row))

        future_predictions = np.array(future_predictions)
        logger.debug(f"Future predictions array shape: {future_predictions.shape}")

        future_predictions_reshaped = np.zeros(
            (future_days, len(_features) + num_targets)
        )
        future_predictions_reshaped[:, :num_targets] = future_predictions
        logger.debug(
            f"Future predictions reshaped: {future_predictions_reshaped.shape}"
        )

        padding_width = len(scaler_prices.scale_) - future_predictions_reshaped.shape[1]
        if padding_width > 0:
            future_predictions_reshaped = np.pad(
                future_predictions_reshaped, ((0, 0), (0, padding_width)), "constant"
            )

        future_predictions = scaler_prices.inverse_transform(
            future_predictions_reshaped[:, : num_targets - 1]
        )
        future_predictions = np.hstack(
            (
                future_predictions,
                scaler_volume.inverse_transform(
                    future_predictions_reshaped[:, num_targets - 1 : num_targets]
                ),
            )
        )
        logger.info(
            f"Inverse transformed future predictions: {future_predictions.shape}"
        )

    logger.info("Prediction completed.")
    return predictions, future_predictions

def get_historical_data(config) -> Tuple[pd.DataFrame, List[str]]:
    logger.info(f"Getting data for {config.symbol} from {config.start_date}")
    return get_data(config)

def main(config: Config):
    logger.info(f"Starting prediction for {config.data_settings['ticker']}")

    selected_features = config.data_settings["selected_features"]

    historical_data, features = get_historical_data(config)
    
    x, y, scaler_prices, scaler_volume, _ = preprocess_data(
        symbol=config.data_settings["symbol"],
        data_sampling_interval=config.data_settings["data_sampling_interval"],
        historical_data=historical_data,
        targets=config.data_settings["targets"],
        look_back=config.training_settings["look_back"],
        look_forward=config.training_settings["look_forward"],
        features=features,
        disabled_features=config.data_settings.get("disabled_features", []),
    )

    logger.info(f"Loaded historical data for {config.data_settings['symbol']}")
    logger.info(f"Selected features: {selected_features}")

    # Filter x to only include selected features
    feature_indices = [features.index(feature) for feature in selected_features]
    x_selected = x[:, :, feature_indices]

    logger.info(f"Shape of x after selecting features: {x_selected.shape}")

    model = load_model(
        config.data_settings["symbol"],
        config.training_settings["model_dir"],
        config.model_settings,
        len(selected_features)
    )

    logger.info(f"Loaded model for {config.data_settings['symbol']}")

    predictions, future_predictions = predict(
        _model=model,
        _x=x_selected,
        scaler_prices=scaler_prices,
        scaler_volume=scaler_volume,
        future_days=config.training_settings["look_forward"],
        _features=selected_features,
        _targets=config.data_settings["targets"],
    )

    candles = historical_data[["Open", "High", "Low", "Close"]]
    plot_predictions(
        config.symbol,
        f"docs/{config.symbol}_{selected_features}_predictions.html",
        candles,
        predictions,
        future_predictions,
        config.data_resampling_frequency,
        config.data_sampling_interval,
    )

    save_predictions_report(
        future_predictions,
        config.targets,
        candles.index[-1],
        config.data_resampling_frequency,
        config.symbol,
    )
    logger.info(f"Prediction for {config.symbol} completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration JSON file"
    )
    args = parser.parse_args()
    main(args.config)