from typing import List, Tuple
import argparse
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

from lstm_forecast.feature_engineering import calculate_technical_indicators
from lstm_forecast.logger import setup_logger
from lstm_forecast.config import load_config

logger = setup_logger("data_loader_logger", "logs/data_loader.log")

def get_data(config) -> Tuple[pd.DataFrame, List[str]]:
    """Download historical stock data and calculate technical indicators."""
    logger.info(f"Downloading data for {config.ticker} from {config.start_date} to {config.end_date}")
    historical_data = yf.download(
        config.ticker, 
        start=config.start_date, 
        end=config.end_date, 
        interval=config.data_sampling_interval
    )

    historical_data, features = calculate_technical_indicators(
        historical_data,
        windows=config.indicator_windows,
        frequency=config.data_resampling_frequency,
    )

    historical_data.to_csv(f"data/{config.symbol}_{config.data_sampling_interval}.csv")
    logger.info(f"Data for {config.symbol} saved to data/{config.symbol}_{config.data_sampling_interval}.csv")
    return historical_data, features

def preprocess_data(
    symbol: str,
    data_sampling_interval: str,
    historical_data: pd.DataFrame,
    targets: List[str],
    look_back: int,
    look_forward: int,
    features: List[str],
    disabled_features: List[str],
) -> Tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler, MinMaxScaler]:
    logger.info("Starting preprocessing of data")
    log_preprocessing_params(targets, look_back, look_forward, features)

    features = [f for f in features if f not in targets and f not in disabled_features]
    logger.info(f"Features after removing targets and disabled features: {features}")

    target_data, feature_data = split_data_into_targets_and_features(historical_data, targets, features)
    scaled_targets, scaler_prices, scaler_volume = scale_targets(target_data)
    scaled_features, scaler_features = scale_features(feature_data)

    save_scaled_data(symbol, data_sampling_interval, scaled_features, scaled_targets, features, targets)

    X, y = create_dataset(scaled_features, scaled_targets, look_back, look_forward, targets)

    validate_data(X, y)

    return X, y, scaler_features, scaler_prices, scaler_volume

def log_preprocessing_params(targets: List[str], look_back: int, look_forward: int, features: List[str]) -> None:
    logger.info(f"Targets: {targets}")
    logger.info(f"Look back: {look_back}, Look forward: {look_forward}")
    logger.info(f"Selected features: {features}")

def split_data_into_targets_and_features(
    historical_data: pd.DataFrame, targets: List[str], features: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target_data = historical_data[targets]
    feature_data = historical_data[features]
    return target_data, feature_data

def scale_targets(target_data: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler, MinMaxScaler]:
    scaler_prices = StandardScaler()
    scaled_prices = scaler_prices.fit_transform(target_data.drop(columns=["Volume"]))

    scaler_volume = MinMaxScaler()
    scaled_volume = scaler_volume.fit_transform(target_data[["Volume"]])

    scaled_targets = np.concatenate((scaled_prices, scaled_volume), axis=1)
    return scaled_targets, scaler_prices, scaler_volume

def scale_features(feature_data: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    scaler_features = StandardScaler()
    scaled_features = scaler_features.fit_transform(feature_data)
    return scaled_features, scaler_features

def save_scaled_data(
    symbol: str,
    interval: str,
    scaled_features: np.ndarray,
    scaled_targets: np.ndarray,
    features: List[str],
    targets: List[str],
) -> None:
    scaled_data = np.concatenate((scaled_features, scaled_targets), axis=1)
    scaled_df = pd.DataFrame(scaled_data, columns=features + targets)
    scaled_df.to_csv(f"data/{symbol}_{interval}_scaled_data.csv", index=False)
    logger.info(f"Scaled dataset saved to data/{symbol}_{interval}_scaled_data.csv")

def create_dataset(scaled_features, scaled_targets, look_back, look_forward, targets):
    _X, _y = [], []
    for i in range(look_back, len(scaled_features) - look_forward + 1):
        _X.append(scaled_features[i - look_back: i])
        _y.append(scaled_targets[i + look_forward - 1, : len(targets)])

    _X = np.array(_X)
    _y = np.array(_y)
    logger.info(f"Shape of _X: {_X.shape}")
    logger.info(f"Shape of _y: {_y.shape}")
    return _X, _y


def create_timeseries_window(_x: np.ndarray, _y: np.ndarray, n_splits: int):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for train_index, val_index in tscv.split(_x):
        x_train, x_val = _x[train_index], _x[val_index]
        y_train, y_val = _y[train_index], _y[val_index]
        train_data = (torch.tensor(x_train).float(), torch.tensor(y_train).float())
        val_data = (torch.tensor(x_val).float(), torch.tensor(y_val).float())
        splits.append((train_data, val_data))
    logger.info(f"Created {len(splits)} time series windows for k-fold cross-validation")
    return splits

def validate_data(_x: np.ndarray, _y: np.ndarray) -> None:
    if np.any(np.isnan(_x)) or np.any(np.isnan(_y)):
        logger.error("NaN values found in input data.")
        raise ValueError("NaN values found in input data.")
    if np.any(np.isinf(_x)) or np.any(np.isinf(_y)):
        logger.error("Infinite values found in input data.")
        raise ValueError("Infinite values found in input data.")
    logger.info("Data validation passed without any NaN or infinite values.")

def split_data(
    _x: np.ndarray, _y: np.ndarray, batch_size: int, test_size: float = 0.15
) -> Tuple[DataLoader, DataLoader]:
    logger.info("Splitting data into training and validation sets")
    x_train, x_val, y_train, y_val = train_test_split(_x, _y, test_size=test_size, random_state=42)
    logger.info(f"Training data shape: {x_train.shape}, Validation data shape: {x_val.shape}")
    train_data = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_data = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True),
        DataLoader(val_data, batch_size=batch_size, shuffle=False),
    )

def create_dataloader(data, batch_size=32):
    dataset = torch.utils.data.TensorDataset(*data)
    logger.info(f"Creating DataLoader with batch size: {batch_size}")
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

def load_and_preprocess_data(config, selected_features=None):
    historical_data, all_features = get_data(config)
    
    if selected_features is None:
        selected_features = all_features
    else:
        # Ensure all selected features are present in the data
        selected_features = [f for f in selected_features if f in all_features]
    
    logger.info(f"Processing data with selected features: {selected_features}")

    x, y, scaler_features, scaler_prices, scaler_volume = preprocess_data(
        config.symbol,
        config.data_sampling_interval,
        historical_data,
        config.targets,
        config.look_back,
        config.look_forward,
        selected_features,
        config.disabled_features,
    )

    if config.training_settings.get("use_time_series_split", False):
        logger.info("Using k-fold cross-validation with TimeSeriesSplit")
        logger.info(f"Number of splits: {config.training_settings.get('time_series_splits', 5)}")
        windows = create_timeseries_window(x, y, n_splits=config.training_settings.get("time_series_splits", 5))
        batch_size = config.model_settings['batch_size']
        train_val_loaders = [(create_dataloader(train_data, batch_size),
                              create_dataloader(val_data, batch_size)) for train_data, val_data in windows]
        logger.info("Data loading and preprocessing completed successfully using time series split.")
        return train_val_loaders, selected_features, scaler_prices, scaler_volume, historical_data, scaler_features

    train_loader, val_loader = split_data(x, y, batch_size=config.batch_size)
    logger.info("Data loading and preprocessing completed successfully.")
    return ([(train_loader, val_loader)], selected_features, scaler_prices, scaler_volume, historical_data,
            scaler_features)

def parse_arguments() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration JSON file")
    return arg_parser.parse_args()

def main():
    args = parse_arguments()
    config = load_config(args.config)
    train_val_loaders, features, scaler_prices, scaler_volume, historical_data, scaler_features = load_and_preprocess_data(config)
    logger.info("Data loading and preprocessing completed successfully.")
    logger.info("Exiting data loader.")
    return train_val_loaders, features, scaler_prices, scaler_volume, historical_data, scaler_features

if __name__ == "__main__":
    main()