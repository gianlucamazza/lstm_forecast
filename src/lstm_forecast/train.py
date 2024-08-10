import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import onnx
from lstm_forecast.data_loader import main as prepare_data
from lstm_forecast.model import PricePredictor, init_weights
from lstm_forecast.early_stopping import EarlyStopping
from lstm_forecast.logger import setup_logger
from lstm_forecast.config import Config
from lstm_forecast.model_utils import run_training_epoch, run_validation_epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = setup_logger("train_logger", "logs/train.log")


def initialize_model(config: Config, num_features: int) -> PricePredictor:
    """Initialize the model with the given configuration and number of features."""
    dropout = (
        config.model_settings["dropout"]
        if config.model_settings["num_layers"] > 1
        else 0
    )

    model = PricePredictor(
        input_size=num_features,
        hidden_size=config.model_settings["hidden_size"],
        num_layers=config.model_settings["num_layers"],
        dropout=dropout,
        fc_output_size=len(config.data_settings["targets"]),
    ).to(device)

    model.apply(init_weights)

    return model


def save_model_checkpoint(
    symbol: str, model: torch.nn.Module, checkpoint_dir: str, epoch: int
) -> None:
    """Save a checkpoint of the given model."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir, f"{symbol}_checkpoint_{epoch}.pth"
    )
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"Model checkpoint saved to {checkpoint_path}")


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    _device: torch.device,
) -> float:
    """Evaluate the model on the given data loader."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(_device), y_batch.to(_device)
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            total_loss += loss.item()

    return total_loss / len(data_loader)


def save_training_state(model, optimizer, epoch, best_val_loss, config):
    """Save the model state, optimizer state, and other training artifacts."""
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }
    save_path = os.path.join(
        config.training_settings["model_dir"],
        f"{config.data_settings['symbol']}_training_state.pth",
    )
    torch.save(state, save_path)
    logger.info(f"Training state saved to {save_path}")


def train_model(
    config,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    learning_rate: float,
    model_dir: str,
    weight_decay: float,
    _device: torch.device,
    fold_idx: int = None,
) -> None:
    """Train the model with early stopping."""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = torch.nn.MSELoss()

    early_stopping = EarlyStopping(
        patience=10,
        verbose=True,
        path=f"{model_dir}/{config.data_settings['symbol']}_fold_{fold_idx}.pth",
    )
    checkpoint_dir = os.path.join(model_dir, "checkpoints")

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = run_training_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            _device,
            clip_value=config.model_settings.get("clip_value"),
        )
        val_loss = run_validation_epoch(model, val_loader, loss_fn, _device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(
            f"Fold {fold_idx}, Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, "
            f"Validation Loss: {val_loss:.4f}"
        )

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            logger.info("Early stopping triggered. Stopping training.")
            break

        save_model_checkpoint(
            config.data_settings["symbol"], model, checkpoint_dir, epoch
        )

    plot_training_history(train_losses, val_losses, config)


def plot_evaluation(
    symbol: str, predictions: np.ndarray, y_true: np.ndarray, dates: np.ndarray
) -> None:
    """Plot the evaluation results."""
    aligned_dates = dates[-len(y_true) :]
    os.makedirs("png", exist_ok=True)

    plt.figure(figsize=(14, 7))
    plt.title(f"{symbol} - Model Evaluation")
    plt.plot(aligned_dates, y_true[:, 0], label="True Price", color="blue")
    plt.plot(
        aligned_dates, predictions[:, 0], label="Predicted Prices", color="red"
    )
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"png/{symbol}_evaluation.png")
    plt.close()
    logger.info("Model evaluation completed and plot saved.")


def plot_training_history(train_losses, val_losses, config):
    """Plot the training and validation loss over epochs."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        f"{config.training_settings['plot_dir']}/training_validation_loss.png"
    )
    plt.close()
    logger.info("Training history plot saved.")


def export_to_onnx(model, config, num_features) -> None:
    """Export the model to ONNX format."""
    try:
        sequence_length = config.model_settings.get("sequence_length", 120)
        dummy_input = torch.randn(1, sequence_length, num_features).to(device)

        onnx_file = f"{config.training_settings['model_dir']}/{config.data_settings['symbol']}_best_model.onnx"

        torch.onnx.export(
            model,
            dummy_input,
            onnx_file,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

        # Verify the model
        onnx_model = onnx.load(onnx_file)
        onnx.checker.check_model(onnx_model)

        logger.info(f"Model exported to ONNX format: {onnx_file}")
    except Exception as e:
        logger.error(f"Error exporting model to ONNX: {str(e)}")


def main(config: Config):
    """Main function to run the training and evaluation."""
    try:
        logger.info(f"Loaded configuration from {config}")
        logger.info(f"Configuration: {config}")

        # Prepare data
        train_val_loaders, _, _, _, _, _, num_features = prepare_data(config)

        best_val_loss = float("inf")
        best_model = None

        # Initialize the model only once, outside the loop
        model = initialize_model(config, num_features)
        logger.info("Model initialized")

        # Define the optimizer outside of the fold loop
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.model_settings.get("learning_rate", 0.001),
            weight_decay=config.model_settings.get("weight_decay", 0.0),
        )

        for fold_idx, (train_loader, val_loader) in enumerate(
            train_val_loaders, 1
        ):
            logger.info(f"Training fold {fold_idx}")
            # Use the same model instance for all folds, just move it to the correct device
            model.to(device)

            # Train the model
            for epoch in range(config.training_settings["epochs"]):
                train_model(
                    config,
                    model,
                    train_loader,
                    val_loader,
                    num_epochs=config.training_settings["epochs"],
                    learning_rate=config.model_settings.get(
                        "learning_rate", 0.001
                    ),
                    model_dir=config.training_settings["model_dir"],
                    weight_decay=config.model_settings.get(
                        "weight_decay", 0.0
                    ),
                    _device=device,
                    fold_idx=fold_idx,
                )

                # Evaluate the model on the validation set
                val_loss = evaluate_model(
                    model, val_loader, torch.nn.MSELoss(), device
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model.state_dict()

                save_training_state(
                    model, optimizer, epoch, best_val_loss, config
                )

        if best_model is not None:
            # Save and export the best model
            final_model = model
            final_model.load_state_dict(best_model)
            final_model.to(device)

            export_to_onnx(final_model, config, num_features)
            pth_model_path = f"{config.training_settings['model_dir']}/{config.data_settings['symbol']}_best.pth"
            torch.save(final_model.state_dict(), pth_model_path)
            logger.info(f"Best model saved to {pth_model_path}")

            # Save the training state including the optimizer state
            save_training_state(
                final_model, optimizer, epoch, best_val_loss, config
            )
        else:
            logger.error("No best model found to export.")

    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
