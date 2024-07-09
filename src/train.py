import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import time
import json
from torch.utils.data import DataLoader
from model import EarlyStopping, PricePredictor
from sklearn.preprocessing import StandardScaler
from data_loader import get_data, preprocess_data, split_data
from utils import load_json
from logger import setup_logger

device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = setup_logger('train_logger')

def train_model(_model: nn.Module, _train_loader: DataLoader, _val_loader: DataLoader, num_epochs: int,
                _learning_rate: float, _model_path: str) -> None:
    """
    Train the model.

    Args:
        _model (nn.Module): The model to train.
        _train_loader (DataLoader): The training data loader.
        _val_loader (DataLoader): The validation data loader.
        num_epochs (int): The number of epochs to train the model for.
        _learning_rate (float): The learning rate for the optimizer.
        _model_path (str): The path to save the best model.

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
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0.0
        _model.eval()
        with torch.no_grad():
            for X_val_batch, y_val_batch in _val_loader:
                x_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                val_outputs = _model(x_val_batch)
                val_loss += criterion(val_outputs, y_val_batch.unsqueeze(1)).item()

        train_loss /= len(_train_loader)
        val_loss /= len(_val_loader)

        scheduler.step(val_loss)
        early_stopping(val_loss)

        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = _model.state_dict()

        if early_stopping.early_stop:
            print("Early stopping")
            break

        _model.train()

    # Save the best model
    if best_model is not None:
        torch.save(best_model, _model_path)


def evaluate_model(_ticker: str, _model: nn.Module, _x: np.ndarray, _y: np.ndarray,
                   _scaler: StandardScaler, feature_names: list, dates: pd.DatetimeIndex) -> None:
    """
    Evaluate the model.

    Args:
        _ticker (str): The ticker symbol.
        _model (nn.Module): The model to evaluate.
        _x (np.ndarray): The input data.
        _y (np.ndarray): The target data.
        _scaler (StandardScaler): The scaler used to scale the data.
        feature_indices (list): The indices of the selected features.
        dates (pd.DatetimeIndex): The dates corresponding to the input data.

    Returns:
        None
    """
    _model.eval()
    with torch.no_grad():
        predictions = _model(torch.tensor(_x, dtype=torch.float32).to(device)).cpu().numpy()
        
        predictions_reshaped = np.zeros((_y.shape[0], len(feature_names) + 1))
        predictions_reshaped[:, 0] = predictions[:, 0]

        if len(predictions_reshaped[0]) < len(_scaler.scale_):
            predictions_reshaped = np.pad(predictions_reshaped, ((0, 0), (0, len(_scaler.scale_) - len(predictions_reshaped[0]))), 'constant')

        predictions = _scaler.inverse_transform(predictions_reshaped)[:, 0]

        y_true_reshaped = np.zeros((_y.shape[0], len(feature_names) + 1))
        y_true_reshaped[:, 0] = _y

        if len(y_true_reshaped[0]) < len(_scaler.scale_):
            y_true_reshaped = np.pad(y_true_reshaped, ((0, 0), (0, len(_scaler.scale_) - len(y_true_reshaped[0]))), 'constant')

        y_true = _scaler.inverse_transform(y_true_reshaped)[:, 0]
    
    aligned_dates = dates[-len(y_true):]

    plt.figure(figsize=(14, 7))
    plt.title(f'{_ticker} - Model Evaluation')
    plt.plot(aligned_dates, y_true, label='True Price', color='blue')
    plt.plot(aligned_dates, predictions, label='Predicted Price', color='red')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('png/evaluation.png')
    plt.show()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    args = arg_parser.parse_args()

    # Load configuration
    config = load_json(args.config)

    ticker = config['ticker']
    start_date = config['start_date']
    end_date = time.strftime('%Y-%m-%d')
    look_back = config['look_back']
    look_forward = config['look_forward']
    epochs = config['epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    model_path = config['model_path']
    features = config['features']
    target = config['target']
    indicator_windows = config['indicator_windows']
    best_features = config.get('best_features', None)

    # Get historical data
    logger.info(f'Getting historical data for {ticker} from {start_date} to {end_date}')
    historical_data = get_data(ticker, start_date, end_date, indicator_windows)
    dates = historical_data.index
    
    # Preprocess data
    logger.info('Preprocessing data')
    X, y, scaler, selected_features = preprocess_data(historical_data, target, look_back=look_back,
                                                      look_forward=look_forward, features=features, best_features=best_features,max_iter=100)

    # Debug: print selected features
    logger.info(f'Selected features: {selected_features}')

    # Update config.json with selected features
    config['best_features'] = selected_features
    with open(args.config, 'w') as f:
        json.dump(config, f, indent=4)

    # Split data
    logger.info('Splitting data')
    train_loader, val_loader = split_data(X, y, batch_size=batch_size)

    # Initialize model
    logger.info('Initializing model')
    model = PricePredictor(
        input_size=len(selected_features)
    ).to(device)

    # Train model
    logger.info('Training model')
    train_model(model, train_loader, val_loader, num_epochs=epochs,
                _learning_rate=learning_rate, _model_path=model_path)

    # Evaluate model
    logger.info('Evaluating model')
    evaluate_model(ticker, model, X, y, scaler, selected_features, dates)
