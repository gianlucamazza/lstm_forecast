import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import EarlyStopping, PricePredictor
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse
import logging
import time
from data_loader import get_data, preprocess_data, split_data
from utils import load_json

device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(_model: nn.Module, _train_loader: DataLoader, _val_loader: DataLoader, num_epochs: int,
                _learning_rate: float, _model_path: str) -> None:
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
                   _scaler: StandardScaler, feature_indices: list) -> None:
    _model.eval()
    with torch.no_grad():
        predictions = _model(torch.tensor(_x, dtype=torch.float32).to(device)).cpu().numpy()
        predictions_reshaped = np.zeros((_y.shape[0], len(feature_indices)))
        predictions_reshaped[:, 0] = predictions[:, 0]
        predictions = _scaler.inverse_transform(predictions_reshaped)[:, 0]

        y_true_reshaped = np.zeros((_y.shape[0], len(feature_indices)))
        y_true_reshaped[:, 0] = _y
        y_true = _scaler.inverse_transform(y_true_reshaped)[:, 0]

    plt.figure(figsize=(14, 7))
    plt.title(f'{_ticker} - Model Evaluation')
    plt.plot(y_true, label='True Price', color='blue')
    plt.plot(predictions, label='Predicted Price', color='red')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('evaluation.png')
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

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

    # Log parameters
    logging.info(f'Ticker: {ticker}')
    logging.info(f'Start Date: {start_date}')
    logging.info(f'End Date: {end_date}')
    logging.info(f'Look Back: {look_back}')
    logging.info(f'Look Forward: {look_forward}')
    logging.info(f'Epochs: {epochs}')
    logging.info(f'Batch Size: {batch_size}')
    logging.info(f'Learning Rate: {learning_rate}')
    logging.info(f'Model Path: {model_path}')
    logging.info(f'Features: {features}')
    logging.info(f'Target: {target}')

    # Get historical data
    logging.info(f'Getting historical data for {ticker} from {start_date} to {end_date}')
    historical_data = get_data(ticker, start_date, end_date)

    # Preprocess data
    logging.info('Preprocessing data')
    X, y, scaler, selected_features = preprocess_data(historical_data, target, look_back=look_back,
                                                      look_forward=look_forward, features=features)

    # Debug: print selected features
    logging.info(f'Selected features: {selected_features}')

    # Split data
    logging.info('Splitting data')
    train_loader, val_loader = split_data(X, y, batch_size=batch_size)

    input_size = len(selected_features)
    logging.info(f'Input size: {input_size}')

    # Initialize model
    logging.info('Initializing model')
    model = PricePredictor(
        input_size=input_size
    ).to(device)

    # Train model
    logging.info('Training model')
    train_model(model, train_loader, val_loader, num_epochs=epochs,
                _learning_rate=learning_rate, _model_path=model_path)

    # Evaluate model
    logging.info('Evaluating model')
    evaluate_model(ticker, model, X, y, scaler, selected_features)
