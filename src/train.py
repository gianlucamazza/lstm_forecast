import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import PricePredictor, init_weights
from src.early_stopping import EarlyStopping
from src.logger import setup_logger
from src.data_loader import load_and_preprocess_data
from src.config import load_config, update_config
from src.model_utils import run_training_epoch, run_validation_epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = setup_logger("train_logger", "logs/train.log")


def initialize_model(config):
    """Initialize the model with the given configuration."""
    hidden_size = config.model_settings.get("hidden_size", 64)
    num_layers = config.model_settings.get("num_layers", 2)
    dropout = config.model_settings.get("dropout", 0.2)
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


def save_best_model(best_model, model_dir, symbol):
    """Save the best model to the given directory."""
    if best_model:
        torch.save(best_model, f"{model_dir}/{symbol}_model.pth")
        logger.info(f"Best model saved to {model_dir}/{symbol}_model.pth")


def save_model_checkpoint(symbol, model, model_dir, epoch):
    """Save a checkpoint of the given model."""
    torch.save(model.state_dict(), f"{model_dir}/{symbol}_checkpoint_{epoch}.pth")
    logger.info(f"Model checkpoint saved to {model_dir}/{symbol}_checkpoint_{epoch}.pth")


def evaluate_model(model, data_loader, loss_fn, device):
    """Evaluate the model on the given data loader."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            x_batch, y_batch = batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            total_loss += loss.item()

    return total_loss / len(data_loader)


def inverse_transform(data, scaler_prices, scaler_volume):
    """Inverse transform the given data using the given scalers."""
    inverse_data = np.zeros_like(data)
    inverse_data[:, :-1] = scaler_prices.inverse_transform(data[:, :-1])
    inverse_data[:, -1] = scaler_volume.inverse_transform(
        data[:, -1].reshape(-1, 1)
    ).flatten()
    return inverse_data


def train_model(symbol, model, train_loader, val_loader, num_epochs, learning_rate, model_dir, weight_decay, device):
    """Train the model with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    early_stopping = EarlyStopping(patience=10, verbose=True, path=f"{model_dir}/{symbol}_best_model.pth")

    for epoch in range(num_epochs):
        train_loss = run_training_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = run_validation_epoch(model, val_loader, loss_fn, device)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Check early stopping condition
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            logger.info("Early stopping triggered. Stopping training.")
            break

        save_model_checkpoint(symbol, model, model_dir, epoch)


def plot_evaluation(symbol, predictions, y_true, dates):
    """Plot the evaluation results."""
    aligned_dates = dates[-len(y_true):]

    plt.figure(figsize=(14, 7))
    plt.title(f"{symbol} - Model Evaluation")
    plt.plot(aligned_dates, y_true[:, 0], label="True Price", color="blue")
    plt.plot(aligned_dates, predictions[:, 0], label="Predicted Prices", color="red")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"png/{symbol}_evaluation.png")
    plt.close()
    logger.info("Model evaluation completed and plot saved.")


def parse_arguments():
    """Parse command-line arguments."""
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
    """Rebuild features if specified."""
    update_config(config, "feature_settings.best_features", [])
    config.save()
    logger.info("Rebuilding features")


def update_config_with_best_features(config, selected_features):
    """Update configuration with the best features."""
    logger.info(f"Selected features: {selected_features}")
    update_config(config, "feature_settings.best_features", selected_features)
    config.save()


def main():
    """Main function to run the training and evaluation."""
    args = parse_arguments()
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    logger.info(f"Configuration: {config}")
    if args.rebuild_features:
        rebuild_features(config)

    train_val_loaders, selected_features, scaler_prices, scaler_volume, historical_data, scaler_features = (
        load_and_preprocess_data(config))

    model = initialize_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for train_loader, val_loader in train_val_loaders:
        train_model(
            config.data_settings["symbol"],
            model,
            train_loader,
            val_loader,
            num_epochs=config.training_settings["epochs"],
            learning_rate=config.model_settings.get("learning_rate", 0.001),
            model_dir=config.training_settings["model_dir"],
            weight_decay=config.model_settings.get("weight_decay", 0.0),
            device=device
        )

        x, y = [], []
        for data, target in train_loader:
            x.append(data)
            y.append(target)
        x = torch.cat(x).to(device)
        y = torch.cat(y).to(device)

        evaluate_model(
            model,
            train_loader,
            torch.nn.MSELoss(),
            device
        )


if __name__ == "__main__":
    main()
