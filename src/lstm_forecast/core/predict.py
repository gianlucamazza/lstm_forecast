from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from lstm_forecast.model import load_model
from lstm_forecast.data_loader import main as prepare_data
from lstm_forecast.predict_utils import (
    inverse_transform_predictions,
    plot_predictions,
    save_predictions_report,
)
from lstm_forecast.logger import setup_logger
from lstm_forecast.config import Config

# Set up logger
logger = setup_logger("predict_logger", "logs/predict.log")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")


def predict(
    model: nn.Module,
    x: np.ndarray,
    scaler_prices: StandardScaler,
    scaler_volume: MinMaxScaler,
    future_days: int,
    features: List[str],
    targets: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    num_targets = len(targets)

    logger.info(
        f"Starting prediction for {num_targets} targets over {future_days} future days."
    )

    with torch.no_grad():
        logger.info("Making predictions on the input data.")
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        predictions = model(x_tensor).cpu().numpy()

        logger.info(f"Initial predictions shape: {predictions.shape}")

        predictions = inverse_transform_predictions(
            predictions, scaler_prices, scaler_volume, features, num_targets
        )
        logger.info(f"Inverse transformed predictions: {predictions.shape}")

        # Forecast future prices
        last_x = x[-1]
        future_predictions = []

        for _ in range(future_days):
            x_tensor = torch.tensor(
                last_x[np.newaxis, :], dtype=torch.float32
            ).to(device)
            future_pred = model(x_tensor).cpu().numpy()[0]
            future_predictions.append(future_pred)

            new_row = np.zeros((1, last_x.shape[1]))
            new_row[0, :num_targets] = future_pred

            if new_row.shape[1] > num_targets:
                new_row[0, num_targets:] = last_x[-1, num_targets:]

            last_x = np.vstack((last_x[1:], new_row))

        future_predictions = np.array(future_predictions)
        logger.debug(
            f"Future predictions array shape: {future_predictions.shape}"
        )

        future_predictions_reshaped = np.zeros(
            (future_days, len(features) + num_targets)
        )
        future_predictions_reshaped[:, :num_targets] = future_predictions
        logger.debug(
            f"Future predictions reshaped: {future_predictions_reshaped.shape}"
        )

        padding_width = (
            len(scaler_prices.scale_) - future_predictions_reshaped.shape[1]
        )
        if padding_width > 0:
            future_predictions_reshaped = np.pad(
                future_predictions_reshaped,
                ((0, 0), (0, padding_width)),
                "constant",
            )

        future_predictions = scaler_prices.inverse_transform(
            future_predictions_reshaped[:, : num_targets - 1]
        )
        future_predictions = np.hstack(
            (
                future_predictions,
                scaler_volume.inverse_transform(
                    future_predictions_reshaped[
                        :, num_targets - 1 : num_targets
                    ]
                ),
            )
        )
        logger.info(
            f"Inverse transformed future predictions: {future_predictions.shape}"
        )

    logger.info("Prediction completed.")
    return predictions, future_predictions


def main(config: Config):
    logger.info(f"Starting prediction for {config.data_settings['ticker']}")

    (
        train_val_loaders,
        features,
        scaler_prices,
        scaler_volume,
        historical_data,
        _,
        _,
    ) = prepare_data(config)

    logger.info(f"Loaded historical data for {config.data_settings['symbol']}")
    logger.info(f"Available features: {features}")

    model, trained_features = load_model(config)

    logger.info(f"Loaded model for {config.data_settings['symbol']}")
    logger.info(f"Model was trained with features: {trained_features}")

    feature_indices = [features.index(feature) for feature in trained_features]
    x = train_val_loaders[0][0].dataset.tensors[0].numpy()
    x_selected = x[:, :, feature_indices]

    logger.info(f"Shape of x after selecting features: {x_selected.shape}")

    predictions, future_predictions = predict(
        model=model,
        x=x_selected,
        scaler_prices=scaler_prices,
        scaler_volume=scaler_volume,
        future_days=config.training_settings["look_forward"],
        features=trained_features,
        targets=config.data_settings["targets"],
    )

    candles = historical_data[["Open", "High", "Low", "Close"]]
    plot_predictions(
        config.data_settings["symbol"],
        f"docs/{config.data_settings['symbol']}_{'_'.join(trained_features)}_predictions.html",
        candles,
        predictions,
        future_predictions,
        config.data_settings["data_resampling_frequency"],
        config.data_settings["data_sampling_interval"],
    )

    save_predictions_report(
        future_predictions,
        config.data_settings["targets"],
        candles.index[-1],
        config.data_settings["data_resampling_frequency"],
        config.data_settings["symbol"],
    )
    logger.info(f"Prediction for {config.data_settings['symbol']} completed")


if __name__ == "__main__":
    main()
