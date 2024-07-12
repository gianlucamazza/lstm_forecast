import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import EarlyStopping, PricePredictor, init_weights
from src.data_loader import get_data, preprocess_data, split_data
from src.logger import setup_logger
from src.config import load_config, update_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = setup_logger('train_logger', 'logs/train.log')


def initialize_model(config):
    """ Initialize the model with the given configuration. """
    hidden_size = config.model_params.get('hidden_size', 64)
    num_layers = config.model_params.get('num_layers', 2)
    dropout = config.model_params.get('dropout', 0.2)
    input_size = len(config.best_features)
    fc_output_size = len(config.targets)

    model = PricePredictor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        fc_output_size=fc_output_size
    ).to(device)

    model.apply(init_weights)

    return model


def train_model(symbol, model, train_loader, val_loader, num_epochs, learning_rate, model_dir, weight_decay):
    """ Train the model using the given data loaders. """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    early_stopping = EarlyStopping(patience=10, delta=0)

    best_model = None
    best_val_loss = float('inf')

    model.train()
    for epoch in range(num_epochs):
        train_loss = run_training_epoch(model, train_loader, criterion, optimizer)
        val_loss = run_validation_epoch(model, val_loader, criterion)

        scheduler.step(val_loss)
        early_stopping(val_loss)

        logger.info(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

        if early_stopping.early_stop:
            logger.info("Early stopping triggered.")
            break

    save_best_model(best_model, model_dir, symbol)


def run_training_epoch(model, data_loader, criterion, optimizer):
    """ Run a single training epoch using the given data loader. """
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in data_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def run_validation_epoch(model, data_loader, criterion):
    """ Run a single validation epoch using the given data loader. """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_val_batch, y_val_batch in data_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            val_outputs = model(X_val_batch)
            total_loss += criterion(val_outputs, y_val_batch).item()
    return total_loss / len(data_loader)


def save_best_model(best_model, model_dir, symbol):
    """ Save the best model to the given directory. """
    if best_model:
        torch.save(best_model, f'{model_dir}/{symbol}_model.pth')
        logger.info(f'Best model saved to {model_dir}/{symbol}_model.pth')


def evaluate_model(symbol, model, X, y, scaler_prices, scaler_volume, dates):
    """ Evaluate the model using the given data. """
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
        predictions = inverse_transform(predictions, scaler_prices, scaler_volume)
        y_true = inverse_transform(y, scaler_prices, scaler_volume)

    plot_evaluation(symbol, predictions, y_true, dates)


def inverse_transform(data, scaler_prices, scaler_volume):
    """ Inverse transform the given data using the given scalers. """
    inverse_data = np.zeros_like(data)
    inverse_data[:, :-1] = scaler_prices.inverse_transform(data[:, :-1])
    inverse_data[:, -1] = scaler_volume.inverse_transform(data[:, -1].reshape(-1, 1)).flatten()
    return inverse_data


def plot_evaluation(symbol, predictions, y_true, dates):
    aligned_dates = dates[-len(y_true):]

    plt.figure(figsize=(14, 7))
    plt.title(f'{symbol} - Model Evaluation')
    plt.plot(aligned_dates, y_true[:, 0], label='True Price', color='blue')
    plt.plot(aligned_dates, predictions[:, 0], label='Predicted Prices', color='red')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'png/{symbol}_evaluation.png')
    plt.close()
    logger.info('Model evaluation completed and plot saved.')


def main():
    args = parse_arguments()
    config = load_config(args.config)
    logger.info(f'Loaded configuration from {args.config}')
    logger.info(f'Configuration: {config}')
    if args.rebuild_features:
        rebuild_features(config)

    historical_data, features = get_historical_data(config)
    x, y, scaler_features, scaler_prices, scaler_volume, selected_features = preprocess_data(
        symbol=config.symbol,
        historical_data=historical_data,
        targets=config.targets,
        look_back=config.look_back,
        look_forward=config.look_forward,
        features=features,
        best_features=config.best_features,
        max_iter=100,
        feature_selection_algo=config.feature_selection_algo
    )

    update_config_with_best_features(config, selected_features)

    train_loader, val_loader = split_data(x, y, batch_size=config.batch_size)
    model = initialize_model(config)

    train_model(config.symbol, model, train_loader, val_loader, num_epochs=config.epochs,
                learning_rate=config.model_params.get('learning_rate', 0.001), model_dir=config.model_dir,
                weight_decay=config.model_params.get('weight_decay', 0.0))

    evaluate_model(config.symbol, model, x, y, scaler_prices, scaler_volume, historical_data.index)


def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    arg_parser.add_argument('--rebuild-features', action='store_true', help='Rebuild features')
    return arg_parser.parse_args()


def rebuild_features(config):
    update_config(config, 'best_features', [])
    config.save()
    logger.info('Rebuilding features')


def get_historical_data(config):
    logger.info(f'Getting historical data for {config.ticker} from {config.start_date} to {config.end_date}')
    return get_data(config.ticker, config.symbol, config.asset_type, config.start_date,
                    config.end_date, config.indicator_windows, config.data_sampling_interval,
                    config.data_resampling_frequency)


def update_config_with_best_features(config, selected_features):
    logger.info(f'Selected features: {selected_features}')
    update_config(config, 'best_features', selected_features)
    config.save()


if __name__ == "__main__":
    main()
