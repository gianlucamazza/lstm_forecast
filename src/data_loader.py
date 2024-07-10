import torch
import yfinance as yf
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from feature_engineering import calculate_technical_indicators
from typing import List, Tuple, Dict
from logger import setup_logger

logger = setup_logger('data_loader_logger', 'logs/data_loader.log')


def get_data(_ticker: str, asset_type: str, start: str, end: str, windows: Dict[str, int], interval: str) -> tuple[pd.DataFrame, List[str]]:
    """Download historical stock data from Yahoo Finance and calculate technical indicators.
    
    Args:
        _ticker (str): The stock ticker.
        asset_type (str): The type of asset.
        start (str): The start date for the historical data.
        end (str): The end date for the historical data.
        windows (Dict[str, int]): The window sizes for the technical indicators.
        interval (str): The interval for the historical data.
        
    Returns:
        tuple[pd.DataFrame, List[str]]: The historical stock data and the list of calculated feature names.
    """
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    
    # Adjust the start date if the interval is not daily and the date range is more than 2 years
    if interval != '1d':
        deltatime = pd.to_datetime(end) - pd.to_datetime(start)
        if deltatime.days > 365:
            logger.warning("Interval is not 1d and the time range is more than 1 years. Changing the start date to 1 year before the end date.")
            # Set the start date to 1 year before the end date minus 1 day
            start = end_date - pd.DateOffset(years=1)
            start = start.strftime('%Y-%m-%d')
            logger.info(f"New start date: {start}")
            
    logger.info(f"Downloading data for {_ticker} from {start} to {end}")
    historical_data = yf.download(_ticker, start=start, end=end, interval=interval)
    historical_data, features = calculate_technical_indicators(historical_data, windows=windows, asset_type=asset_type)
    historical_data.to_csv(f'data/{_ticker}.csv')
    logger.info(f"Data for {_ticker} saved to data/{_ticker}.csv")
    return historical_data, features


def preprocess_data(historical_data: pd.DataFrame, target: str, look_back: int = 60,
                    look_forward: int = 30, features: List[str] = None, best_features: List[str] = None, max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """
    Preprocess the historical stock data for training the model.
    
    Args:
        historical_data (pd.DataFrame): The historical stock data.
        target (str): The target feature to predict.
        look_back (int): The number of past days to consider for each input sample.
        look_forward (int): The number of future days to predict.
        features (List[str]): List of selected feature names.
        best_features (List[str]): List of best feature names.
        max_iter (int): The maximum number of iterations for feature selection.

    Returns:
        tuple: Processed input features (X), target values (y), and the scaler used for normalization.
    """
    logger.info("Starting preprocessing of data")
    logger.info(f"Target feature: {target}")
    logger.info(f"Look back: {look_back}, Look forward: {look_forward}")
    logger.info(f"Selected features: {features}")
    
    for feature in features:
        if feature not in historical_data.columns:
            logger.error(f"Feature {feature} is not in the historical data.")
            raise ValueError(f"Feature {feature} is not in the historical data.")

    historical_data = historical_data.dropna()
    _scaler = StandardScaler()
    columns = [target] + features
    scaled_data = _scaler.fit_transform(historical_data[columns])
    logger.debug(f"Scaled data: {scaled_data[:5]}") 
    
    _X, _y = [], []
    for i in range(look_back, len(scaled_data) - look_forward):
        _X.append(scaled_data[i - look_back:i])
        _y.append(scaled_data[i + look_forward - 1, 0])

    _X = np.array(_X)
    _y = np.array(_y)

    logger.info(f"Shape of _X: {_X.shape}")
    logger.info(f"Shape of _y: {_y.shape}")

    if np.any(np.isnan(_X)) or np.any(np.isnan(_y)):
        logger.error("NaN values found in input data.")
        raise ValueError("NaN values found in input data.")
    if np.any(np.isinf(_X)) or np.any(np.isinf(_y)):
        logger.error("Infinite values found in input data.")
        raise ValueError("Infinite values found in input data.")
    
    if best_features:
        logger.info(f"Using predefined best features: {best_features}")
        feature_columns = columns[1:]
        feature_indices = [feature_columns.index(feature) for feature in best_features]
        _X_selected = _X[:, :, feature_indices]
        logger.info(f"Shape of _X_selected: {_X_selected.shape}")
        return _X_selected, _y, _scaler, best_features

    _X_reshaped = _X.reshape(_X.shape[0], -1)
    logger.info(f"Shape of _X_reshaped: {_X_reshaped.shape}")

    forest = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    logger.info("Starting feature selection using RandomForestRegressor")
    forest.fit(_X_reshaped, _y)
    
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    selected_features_indices = indices[:min(len(indices), max_iter)]
    logger.info(f"Selected feature indices: {selected_features_indices}")

    max_index = len(features) - 1
    selected_features_indices = [i for i in selected_features_indices if i <= max_index]
    logger.info(f"Validated selected feature indices: {selected_features_indices}")

    selected_features = [features[i] for i in selected_features_indices]
    logger.info(f"Selected features: {selected_features}")

    feature_columns = columns[1:]
    feature_indices = [feature_columns.index(feature) for feature in selected_features]
    _X_selected = _X[:, :, feature_indices]
    logger.info(f"Shape of _X_selected: {_X_selected.shape}")

    return _X_selected, _y, _scaler, selected_features


def split_data(_x: np.ndarray, _y: np.ndarray, batch_size: int, test_size: float = 0.15) -> tuple[DataLoader, DataLoader]:
    """Split the data into training and validation sets.

    Args:
        _x (np.ndarray): The input features.
        _y (np.ndarray): The target values.
        batch_size (int): The batch size for the data loaders.
        test_size (float): The fraction of data to use for validation.

    Returns:
        tuple[DataLoader, DataLoader]: Training and validation data loaders.
    """
    logger.info("Splitting data into training and validation sets")
    x_train, x_val, y_train, y_val = train_test_split(_x, _y, test_size=test_size, random_state=42)
    logger.info(f"Training data shape: {x_train.shape}, Validation data shape: {x_val.shape}")
    train_data = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_data = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    return (DataLoader(train_data, batch_size=batch_size, shuffle=True),
            DataLoader(val_data, batch_size=batch_size, shuffle=False))
