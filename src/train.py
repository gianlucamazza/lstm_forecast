import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import os
import sys
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import EarlyStopping, PricePredictor
from src.data_loader import get_data, preprocess_data, split_data
from src.logger import setup_logger
from src.config import load_config, update_config

device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = setup_logger('train_logger', 'logs/train.log')


def train_model(symbol: str, _model: nn.Module, _train_loader: DataLoader, _val_loader: DataLoader, num_epochs: int,
                _learning_rate: float, _model_dir: str) -> None:
    """
    Train the model.

    Args:
        symbol (str): The stock symbol.
        _model (nn.Module): The model to train.
        _train_loader (DataLoader): The training data loader.
        _val_loader (DataLoader): The validation data loader.
        num_epochs (int): The number of epochs to train the model for.
        _learning_rate (float): The learning rate for the optimizer.
        _model_dir (str): The path to save the best model.

    Returns:
        None
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(_model.parameters(), lr=_learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    early_stopping = EarlyStopping(patience=10, delta=0)

    best_model = None
    best_val_loss = float('inf')

    _model.train()

    for epoch in range(num_epochs):
        train_loss = 0.0
        for X_batch, y_batch in _train_loader:
            x_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = _model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0.0
        _model.eval()
        with torch.no_grad():
            for X_val_batch, y_val_batch in _val_loader:
                x_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                val_outputs = _model(x_val_batch)
                val_loss += criterion(val_outputs, y_val_batch).item()

        train_loss /= len(_train_loader)
        val_loss /= len(_val_loader)

        scheduler.step(val_loss)
        early_stopping(val_loss)

        logger.info(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = _model.state_dict()

        if early_stopping.early_stop:
            logger.info("Early stopping triggered.")
            break

        _model.train()

    # Save the best model
    if best_model is not None:
        torch.save(best_model, f'{_model_dir}/{symbol}_model.pth')
        logger.info(f'Best model saved to {_model_dir}/{symbol}_best_model.pth')


def evaluate_model(symbol: str, _model: nn.Module, _x: np.ndarray, _y: np.ndarray,
                   _scaler: StandardScaler, feature_names: list, dates: pd.DatetimeIndex) -> None:
    """
    Evaluate the model.

    Args:
        symbol (str): The stock symbol.
        _model (nn.Module): The model to evaluate.
        _x (np.ndarray): The input data.
        _y (np.ndarray): The target data.
        _scaler (StandardScaler): The scaler used to scale the data.
        feature_names (List): The list of feature names.
        dates (pd.DatetimeIndex): The dates corresponding to the data.

    Returns:
        None
    """

    def inverse_transform(data: np.ndarray, scaler: StandardScaler, feature_names: list) -> np.ndarray:
        """
        Inverse transform the scaled data.

        Args:
            data (np.ndarray): The scaled data.
            scaler (StandardScaler): The scaler used for scaling.
            feature_names (list): The list of feature names.

        Returns:
            np.ndarray: The inverse transformed data.
        """
        if data.ndim == 1:
            reshaped_data = np.zeros((data.shape[0], len(feature_names) + 1))
            reshaped_data[:, 0] = data  # 1-dimensional data handling
        else:
            reshaped_data = np.zeros((data.shape[0], len(feature_names) + 1))
            reshaped_data[:, 0] = data[:, 0]  # 2-dimensional data handling

        if reshaped_data.shape[1] < len(scaler.scale_):
            reshaped_data = np.pad(reshaped_data,
                                   ((0, 0),
                                    (0, len(scaler.scale_) - reshaped_data.shape[1])), 'constant')

        return scaler.inverse_transform(reshaped_data)[:, 0]

    _model.eval()
    with torch.no_grad():
        predictions = _model(torch.tensor(_x, dtype=torch.float32).to(device)).cpu().numpy()

        # Inverse transform predictions and true values
        predictions = inverse_transform(predictions, _scaler, feature_names)
        y_true = inverse_transform(_y, _scaler, feature_names)

    aligned_dates = dates[-len(y_true):]

    plt.figure(figsize=(14, 7))
    plt.title(f'{symbol} - Model Evaluation')
    plt.plot(aligned_dates, y_true, label='True Price', color='blue')
    plt.plot(aligned_dates, predictions, label='Predicted Prices', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'png/{symbol}_evaluation.png')
    plt.close()
    logger.info('Model evaluation completed and plot saved.')


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    arg_parser.add_argument('--rebuild-features', action='store_true', help='Rebuild features')
    args = arg_parser.parse_args()
    print(args)
    # Load configuration
    config = load_config(args.config)
    logger.info(f'Loaded configuration from {args.config}')

    if args.rebuild_features:
        update_config(config, 'best_features', [])
        config.save()
        logger.info('Rebuilding features')
    else:
        logger.info('Using existing features')

    # Get historical data
    logger.info(f'Getting historical data for {config.ticker} from {config.start_date} to {config.end_date}')
    historical_data, features = get_data(config.ticker, config.symbol, config.asset_type, config.start_date,
                                         config.end_date, config.indicator_windows, config.data_sampling_interval,
                                         config.data_resampling_frequency)
    dates = historical_data.index

    # Preprocess data
    logger.info('Preprocessing data')
    X, y, scaler, selected_features = preprocess_data(historical_data=historical_data, targets=config.targets,
                                                      look_back=config.look_back, look_forward=config.look_forward,
                                                      features=features, best_features=config.best_features,
                                                      max_iter=100)

    # Debug: print selected features
    logger.info(f'Selected features: {selected_features}')

    # Update configuration with best features
    update_config(config, 'best_features', selected_features)
    config.save()

    # Split data
    logger.info('Splitting data')
    train_loader, val_loader = split_data(X, y, batch_size=config.batch_size)

    # Initialize model
    logger.info('Initializing model')
    hidden_size = config.model_params.get('hidden_size', 64)
    num_layers = config.model_params.get('num_layers', 2)
    dropout = config.model_params.get('dropout', 0.2)
    learning_rate = config.model_params.get('learning_rate', 0.001)

    model = PricePredictor(
        input_size=len(selected_features),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        fc_output_size=len(config.targets)
    ).to(device)

    # Train model
    logger.info('Training model')
    train_model(config.symbol, model, train_loader, val_loader, num_epochs=config.epochs,
                _learning_rate=learning_rate, _model_dir=config.model_dir)

    # Evaluate model
    logger.info('Evaluating model')
    evaluate_model(config.symbol, model, X, y, scaler, selected_features, dates)
