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
import json

device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(_model: nn.Module, _train_loader: DataLoader, _val_loader: DataLoader, num_epochs: int,
                learning_rate: float, model_path: str) -> None:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(_model.parameters(), lr=learning_rate)
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
        torch.save(best_model, model_path)


def evaluate_model(_ticker: str, _model: nn.Module, _x: np.ndarray, _y: np.ndarray, _scaler: StandardScaler) -> None:
    _model.eval()
    with torch.no_grad():
        predictions = _model(torch.tensor(_x, dtype=torch.float32).to(device)).cpu().numpy()
        predictions_reshaped = np.zeros((_y.shape[0], 3))
        predictions_reshaped[:, 0] = predictions[:, 0]
        predictions = _scaler.inverse_transform(predictions_reshaped)[:, 0]
        y_true = _scaler.inverse_transform(np.concatenate([_y.reshape(-1, 1), np.zeros((_y.shape[0], 2))], axis=1))[:,
                 0]

    plt.figure(figsize=(14, 7))
    plt.title(f'{_ticker} - Model Evaluation')
    plt.plot(y_true, label='True Price', color='blue')
    plt.plot(predictions, label='Predicted Price', color='red')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('evaluation.png')


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    args = arg_parser.parse_args()

    with open(args.config, 'r') as config_file:
        config = json.load(config_file)

    ticker = config['ticker']
    start_date = config['start_date']
    end_date = time.strftime('%Y-%m-%d')
    look_back = config['look_back']
    look_forward = config['look_forward']
    epochs = config['epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    model_path = config['model_path']

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

    # Get historical data
    logging.info(f'Getting historical data for {ticker} from {start_date} to {end_date}')
    historical_data = get_data(ticker, start_date, end_date)

    # Preprocess data
    logging.info('Preprocessing data')
    X, y, scaler = preprocess_data(historical_data, look_back=look_back, look_forward=look_forward)

    # Split data
    logging.info('Splitting data')
    train_loader, val_loader = split_data(X, y, batch_size=batch_size)

    # Initialize model
    logging.info('Initializing model')
    model = PricePredictor().to(device)

    # Train model
    logging.info('Training model')
    train_model(model, train_loader, val_loader, num_epochs=epochs, learning_rate=learning_rate, model_path=model_path)

    # Evaluate model
    logging.info('Evaluating model')
    evaluate_model(ticker, model, X, y, scaler)
