import torch
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from feature_engineering import calculate_technical_indicators
from typing import List, Tuple, Dict


def get_data(_ticker: str, start: str, end: str, windows: Dict[str, int]) -> pd.DataFrame:
    historical_data = yf.download(_ticker, start=start, end=end)
    historical_data = calculate_technical_indicators(historical_data, windows=windows)
    historical_data.to_csv(f'data/{_ticker}.csv')
    return historical_data


def preprocess_data(historical_data: pd.DataFrame, target: str, look_back: int = 60,
                    look_forward: int = 30, features: List[str] = None) -> Tuple[np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """Preprocess the data for model training.

    Args:
        historical_data (pd.DataFrame): The historical stock data.
        target (str): The target feature to predict.
        look_back (int): The number of past days to consider for each input sample.
        look_forward (int): The number of future days to predict.
        features (List[str]): List of selected feature names.

    Returns:
        tuple: Processed input features (X), target values (y), and the scaler used for normalization.
    """
    for feature in features:
        if feature not in historical_data.columns:
            raise ValueError(f"Feature {feature} is not in the historical data.")

    historical_data = historical_data.dropna()
    _scaler = StandardScaler()
    columns = [target] + [feature for feature in features if feature != target]
    scaled_data = _scaler.fit_transform(historical_data[columns])

    _X, _y = [], []
    for i in range(look_back, len(scaled_data) - look_forward):
        _X.append(scaled_data[i - look_back:i])
        _y.append(scaled_data[i + look_forward, 0])

    _X = np.array(_X)
    _y = np.array(_y)

    return _X, _y, _scaler, columns


def split_data(_x: np.ndarray, _y: np.ndarray, batch_size: int,test_size: float = 0.15) -> tuple[DataLoader, DataLoader]:
    """Split the data into training and validation sets.

    Args:
        _x (np.ndarray): The input features.
        _y (np.ndarray): The target values.
        batch_size (int): The batch size for the data loaders.
        test_size (float): The fraction of data to use for validation.

    Returns:
        tuple[DataLoader, DataLoader]: Training and validation data loaders.
    """
    x_train, x_val, y_train, y_val = train_test_split(_x, _y, test_size=test_size, random_state=42)
    train_data = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_data = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    return (DataLoader(train_data, batch_size=batch_size, shuffle=True),
            DataLoader(val_data, batch_size=batch_size, shuffle=False))
