import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import EarlyStopping, PricePredictor, init_weights
from src.logger import setup_logger
from src.data_loader import get_data, preprocess_data, split_data
from src.config import load_config, update_config
from src.model_utils import run_training_epoch, run_validation_epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = setup_logger("train_logger", "logs/train.log")


def initialize_model(config):
    """ Initialize the model with the given configuration. """
    hidden_size = config.model_params.get("hidden_size", 64)
    num_layers = config.model_params.get("num_layers", 2)
    dropout = config.model_params.get("dropout", 0.2)
    input_size = len(config.feature_settings["best_features"])
    fc_output_size = len(config.data_settings["targets"])

    model = PricePredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        fc_output_size=fc_output_size,
    ).to(device)

    model.apply(init_weights)

    return model


def train_model(
    symbol,
    model,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate,
    model_dir,
    weight_decay,
):
    """Train the model using the given data loaders."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )
    early_stopping = EarlyStopping(patience=10, delta=0)

    best_model = None
    best_val_loss = float("inf")

    model.train()
    for epoch in range(num_epochs):
        train_loss = run_training_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss = run_validation_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        early_stopping(val_loss)

        logger.info(
            f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

        if early_stopping.early_stop:
            logger.info("Early stopping triggered.")
            break

    save_best_model(best_model, model_dir, symbol)


def save_best_model(best_model, model_dir, symbol):
    """Save the best model to the given directory."""
    if best_model:
        torch.save(best_model, f"{model_dir}/{symbol}_model.pth")
        logger.info(f"Best model saved to {model_dir}/{symbol}_model.pth")


def evaluate_model(symbol, model, X, y, scaler_prices, scaler_volume, dates):
    """Evaluate the model using the given data."""
    model.eval()
    with torch.no_grad():
        predictions = (
            model(
                torch.tensor(
                    X,
                    dtype=torch.float32).to(device)).cpu().numpy())
        predictions = inverse_transform(
            predictions, scaler_prices, scaler_volume)
        y_true = inverse_transform(y, scaler_prices, scaler_volume)

    plot_evaluation(symbol, predictions, y_true, dates)


def inverse_transform(data, scaler_prices, scaler_volume):
    """Inverse transform the given data using the given scalers."""
    inverse_data = np.zeros_like(data)
    inverse_data[:, :-1] = scaler_prices.inverse_transform(data[:, :-1])
    inverse_data[:, -1] = scaler_volume.inverse_transform(
        data[:, -1].reshape(-1, 1)
    ).flatten()
    return inverse_data


def plot_evaluation(symbol, predictions, y_true, dates):
    aligned_dates = dates[-len(y_true):]

    plt.figure(figsize=(14, 7))
    plt.title(f"{symbol} - Model Evaluation")
    plt.plot(aligned_dates, y_true[:, 0], label="True Price", color="blue")
    plt.plot(aligned_dates, predictions[:, 0],
             label="Predicted Prices", color="red")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"png/{symbol}_evaluation.png")
    plt.close()
    logger.info("Model evaluation completed and plot saved.")


def main():
    args = parse_arguments()
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    logger.info(f"Configuration: {config}")
    if args.rebuild_features:
        rebuild_features(config)

    historical_data, features = get_historical_data(config)
    x, y, scaler_features, scaler_prices, scaler_volume, selected_features = preprocess_data(
        config.data_settings["symbol"],
        config.data_settings["data_sampling_interval"],
        historical_data,
        config.data_settings["targets"],
        config.training_settings["look_back"],
        config.training_settings["look_forward"],
        features,
        config.data_settings["disabled_features"],
        config.feature_settings["best_features"],
    )

    update_config_with_best_features(config, selected_features)

    train_loader, val_loader = split_data(x, y, batch_size=config.training_settings["batch_size"])
    model = initialize_model(config)

    train_model(
        config.data_settings["symbol"],
        model,
        train_loader,
        val_loader,
        num_epochs=config.training_settings["epochs"],
        learning_rate=config.model_settings.get("learning_rate", 0.001),
        model_dir=config.training_settings["model_dir"],
        weight_decay=config.model_settings.get("weight_decay", 0.0),
    )

    evaluate_model(
        config.data_settings["symbol"],
        model,
        x,
        y,
        scaler_prices,
        scaler_volume,
        historical_data.index)


def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration JSON file")
    arg_parser.add_argument(
        "--rebuild-features", action="store_true", help="Rebuild features"
    )
    return arg_parser.parse_args()


def rebuild_features(config):
    update_config(config, "feature_settings.best_features", [])
    config.save()
    logger.info("Rebuilding features")


def get_historical_data(config):
    logger.info(
        f"Getting historical data for {config.data_settings['ticker']} from {config.data_settings['start_date']} to {config.data_settings['end_date']}"
    )
    return get_data(
        config.data_settings["ticker"],
        config.data_settings["symbol"],
        config.data_settings["asset_type"],
        config.data_settings["start_date"],
        config.data_settings["end_date"],
        config.data_settings["technical_indicators"],
        config.data_settings["data_sampling_interval"],
        config.data_settings["data_resampling_frequency"],
    )


def update_config_with_best_features(config, selected_features):
    logger.info(f"Selected features: {selected_features}")
    update_config(config, "feature_settings.best_features", selected_features)
    config.save()


if __name__ == "__main__":
    main()
