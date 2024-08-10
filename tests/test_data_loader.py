import pytest
import pandas as pd
import numpy as np
import torch
from unittest.mock import patch, MagicMock
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from lstm_forecast.config import Config
from lstm_forecast.data_loader import (
    get_data,
    preprocess_data,
    create_dataset,
    create_dataloader,
    load_and_preprocess_data,
)


@pytest.fixture
def mock_config():
    config = MagicMock(spec=Config)
    config.ticker = "AAPL"
    config.start_date = "2020-01-01"
    config.end_date = "2021-01-01"
    config.data_sampling_interval = "1d"
    config.symbol = "AAPL"
    config.indicator_windows = [5, 10]
    config.data_resampling_frequency = "D"
    config.targets = ["Close", "Volume"]
    config.all_features = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SMA_5",
        "SMA_10",
    ]
    config.look_back = 30
    config.look_forward = 1
    config.disabled_features = []
    config.data_settings = {
        "selected_features": ["Open", "High", "Low", "Close", "Volume"]
    }
    config.model_settings = {"batch_size": 32}
    return config


@pytest.fixture
def mock_historical_data():
    dates = pd.date_range(start="2020-01-01", end="2021-01-01", freq="D")
    data = {
        "Open": np.random.rand(len(dates)),
        "High": np.random.rand(len(dates)),
        "Low": np.random.rand(len(dates)),
        "Close": np.random.rand(len(dates)),
        "Volume": np.random.randint(1000, 10000, len(dates)),
        "SMA_5": np.random.rand(len(dates)),
        "SMA_10": np.random.rand(len(dates)),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def mock_features():
    return ["Open", "High", "Low", "Close", "Volume", "SMA_5", "SMA_10"]


@patch("lstm_forecast.data_loader.yf.download")
@patch("lstm_forecast.data_loader.calculate_technical_indicators")
def test_get_data(
    mock_calculate_indicators,
    mock_yf_download,
    mock_config,
    mock_historical_data,
    mock_features,
):
    mock_yf_download.return_value = mock_historical_data
    mock_calculate_indicators.return_value = (
        mock_historical_data,
        mock_features,
    )

    historical_data, features = get_data(mock_config)

    assert isinstance(historical_data, pd.DataFrame)
    assert isinstance(features, list)
    assert len(features) == len(mock_features)
    mock_yf_download.assert_called_once()
    mock_calculate_indicators.assert_called_once()


def test_preprocess_data(mock_config, mock_historical_data):
    selected_features = mock_config.data_settings["selected_features"]
    X, y, scaler_features, scaler_prices, scaler_volume = preprocess_data(
        mock_config.symbol,
        mock_config.data_sampling_interval,
        mock_historical_data,
        mock_config.targets,
        mock_config.look_back,
        mock_config.look_forward,
        selected_features,
        mock_config.disabled_features,
    )

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(scaler_features, StandardScaler)
    assert isinstance(scaler_prices, StandardScaler)
    assert isinstance(scaler_volume, MinMaxScaler)
    assert X.shape[1] == mock_config.look_back
    assert X.shape[2] == len(selected_features) - len(mock_config.targets)
    assert y.shape[1] == len(mock_config.targets)


def test_create_dataset(mock_config):
    selected_features = mock_config.data_settings["selected_features"]
    features = np.random.rand(100, len(selected_features))
    targets = np.random.rand(100, len(mock_config.targets))
    look_back = mock_config.look_back
    look_forward = mock_config.look_forward

    X, y = create_dataset(
        features, targets, look_back, look_forward, mock_config.targets
    )

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape == (
        100 - look_back - look_forward + 1,
        look_back,
        len(selected_features),
    )
    assert y.shape == (
        100 - look_back - look_forward + 1,
        len(mock_config.targets),
    )


def test_create_dataloader(mock_config):
    selected_features = mock_config.data_settings["selected_features"]
    X = torch.randn(100, mock_config.look_back, len(selected_features))
    y = torch.randn(100, len(mock_config.targets))
    batch_size = mock_config.model_settings["batch_size"]

    dataloader = create_dataloader((X, y), batch_size)

    assert isinstance(dataloader, torch.utils.data.DataLoader)
    assert dataloader.batch_size == batch_size


@patch("lstm_forecast.data_loader.get_data")
@patch("lstm_forecast.data_loader.preprocess_data")
@patch("lstm_forecast.data_loader.train_test_split")
def test_load_and_preprocess_data(
    mock_train_test_split,
    mock_preprocess_data,
    mock_get_data,
    mock_config,
    mock_historical_data,
    mock_features,
):
    mock_get_data.return_value = (mock_historical_data, mock_features)
    selected_features = mock_config.data_settings["selected_features"]
    mock_preprocess_data.return_value = (
        np.random.rand(100, mock_config.look_back, len(selected_features)),
        np.random.rand(100, len(mock_config.targets)),
        MagicMock(spec=StandardScaler),
        MagicMock(spec=StandardScaler),
        MagicMock(spec=MinMaxScaler),
    )
    mock_train_test_split.return_value = (
        np.random.rand(80, mock_config.look_back, len(selected_features)),
        np.random.rand(20, mock_config.look_back, len(selected_features)),
        np.random.rand(80, len(mock_config.targets)),
        np.random.rand(20, len(mock_config.targets)),
    )

    result = load_and_preprocess_data(mock_config)

    assert isinstance(result, tuple)
    assert len(result) == 7
    assert isinstance(result[0], list)
    assert len(result[0]) == 1
    assert isinstance(result[0][0], tuple)
    assert len(result[0][0]) == 2
    assert isinstance(result[0][0][0], torch.utils.data.DataLoader)
    assert isinstance(result[0][0][1], torch.utils.data.DataLoader)
    assert result[1] == selected_features
    assert isinstance(result[2], StandardScaler)
    assert isinstance(result[3], MinMaxScaler)
    assert isinstance(result[4], pd.DataFrame)
    assert isinstance(result[5], StandardScaler)
    assert isinstance(result[6], int)
    assert result[6] == len(selected_features)


if __name__ == "__main__":
    pytest.main()
