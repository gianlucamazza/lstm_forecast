import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
import torch


def get_data(_ticker: str, start: str, end: str) -> pd.DataFrame:
    historical_data = yf.download(_ticker, start=start, end=end)
    return historical_data


def preprocess_data(historical_data: pd.DataFrame, look_back: int = 60) -> tuple[
    np.ndarray, np.ndarray, StandardScaler]:
    historical_data['SMA_20'] = SMAIndicator(historical_data['Close'], window=20).sma_indicator()
    historical_data['RSI_14'] = RSIIndicator(historical_data['Close'], window=14).rsi()

    historical_data = historical_data.dropna()

    _scaler = StandardScaler()
    scaled_data = _scaler.fit_transform(historical_data[['Close', 'SMA_20', 'RSI_14']])

    _X = []
    _y = []

    for i in range(look_back, len(scaled_data) - 30):
        _X.append(scaled_data[i - look_back:i])
        _y.append(scaled_data[i + 30, 0])  # Predict the close price 30 days into the future

    _X = np.array(_X)
    _y = np.array(_y)

    return _X, _y, _scaler


def split_data(_x: np.ndarray, _y: np.ndarray, test_size: float = 0.15) -> tuple[DataLoader, DataLoader]:
    x_train, x_val, y_train, y_val = train_test_split(_x, _y, test_size=test_size, random_state=42)
    train_data = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_data = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

    return DataLoader(train_data, batch_size=32, shuffle=True), DataLoader(val_data, batch_size=32, shuffle=False)
