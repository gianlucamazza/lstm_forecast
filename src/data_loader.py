from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from numpy import ndarray, dtype
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from src.feature_engineering import calculate_technical_indicators
from src.logger import setup_logger

logger = setup_logger("data_loader_logger", "logs/data_loader.log")


def get_data(
        _ticker: str,
        symbol: str,
        asset_type: str,
        start: str,
        end: str,
        windows: Dict[str, int],
        data_sampling_interval: str,
        data_resampling_frequency: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """Download historical stock data and calculate technical indicators."""
    logger.info(f"Downloading data for {_ticker} from {start} to {end}")
    historical_data = yf.download(
        _ticker, start=start, end=end, interval=data_sampling_interval
    )

    historical_data, features = calculate_technical_indicators(
        historical_data,
        windows=windows,
        asset_type=asset_type,
        frequency=data_resampling_frequency,
    )
    save_historical_data(symbol, data_sampling_interval, historical_data)
    return historical_data, features


def adjust_start_date_if_needed(start: str, end: str, end_date: pd.Timestamp) -> str:
    """Adjust the start date if the interval is not daily and the date range is more than 1 year."""
    deltatime = pd.to_datetime(end) - pd.to_datetime(start)
    if deltatime.days > 365:
        logger.warning(
            "Interval is not 1d and the time range is more than 1 year. Changing the start date to 1 "
            "year before the end date."
        )
        start = (end_date - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
        logger.info(f"New start date: {start}")
    return start


def save_historical_data(symbol: str, interval: str, historical_data: pd.DataFrame) -> None:
    """Save historical data to CSV."""
    historical_data.to_csv(f"data/{symbol}_{interval}.csv")
    logger.info(f"Data for {symbol} saved to data/{symbol}.csv")


def preprocess_data(
        symbol: str,
        data_sampling_interval: str,
        historical_data: pd.DataFrame,
        targets: List[str],
        look_back: int = 60,
        look_forward: int = 30,
        features: List[str] = None,
        disabled_features: List[str] = None,
        best_features: List[str] = None,
) -> Tuple[
    ndarray[Any, dtype[Any]],
    ndarray[Any, dtype[Any]],
    StandardScaler,
    StandardScaler,
    MinMaxScaler,
    List[str],
]:
    """Preprocess the historical stock data for training the model."""
    logger.info("Starting preprocessing of data")
    log_preprocessing_params(targets, look_back, look_forward, features)

    features = [f for f in features if f not in targets]
    if disabled_features:
        features = [f for f in features if f not in disabled_features]
    logger.info(f"Features after removing targets and disabled features: {features}")

    target_data, feature_data = split_data_into_targets_and_features(
        historical_data, targets, features
    )
    scaled_targets, scaler_prices, scaler_volume = scale_targets(target_data)
    scaled_features, scaler_features = scale_features(feature_data)

    save_scaled_data(symbol, data_sampling_interval, scaled_features, scaled_targets, features, targets)

    _X, _y = create_dataset(
        scaled_features, scaled_targets, look_back, look_forward, targets
    )

    validate_data(_X, _y)

    if best_features:
        return select_best_features(
            _X,
            _y,
            best_features,
            features,
            scaled_features,
            scaler_features,
            scaler_prices,
            scaler_volume,
        )

    selected_features = select_features(
        _X, _y
    )
    _X_selected = _X[:, :, selected_features]

    return (
        _X_selected,
        _y,
        scaler_features,
        scaler_prices,
        scaler_volume,
        [features[i] for i in selected_features],
    )


def select_features(_x: np.ndarray, _y: np.ndarray) -> ndarray:
    """Perform feature selection using XGBoost and return the selected feature indices."""
    _X_reshaped = _x.reshape(_x.shape[0], -1)
    logger.info(f"Shape of _X_reshaped: {_X_reshaped.shape}")

    model = XGBRegressor()
    model.fit(_X_reshaped, _y)
    feature_importances = model.feature_importances_
    logger.info(f"Feature importances: {feature_importances}")

    num_features = _x.shape[2]
    selected_features_indices = np.argsort(feature_importances)[::-1][:num_features]
    original_feature_indices = np.unique(selected_features_indices % num_features)
    logger.info(f"Selected feature indices: {original_feature_indices}")

    return original_feature_indices


def log_preprocessing_params(
        targets: List[str], look_back: int, look_forward: int, features: List[str]
) -> None:
    """Log preprocessing parameters."""
    logger.info(f"Targets: {targets}")
    logger.info(f"Look back: {look_back}, Look forward: {look_forward}")
    logger.info(f"Selected features: {features}")


def split_data_into_targets_and_features(
        historical_data: pd.DataFrame, targets: List[str], features: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split historical data into targets and features."""
    target_data = historical_data[targets]
    feature_data = historical_data[features]
    return target_data, feature_data


def scale_targets(
        target_data: pd.DataFrame,
) -> Tuple[np.ndarray, StandardScaler, MinMaxScaler]:
    """Scale target data."""
    scaler_prices = StandardScaler()
    scaled_prices = scaler_prices.fit_transform(target_data.drop(columns=["Volume"]))

    scaler_volume = MinMaxScaler()
    scaled_volume = scaler_volume.fit_transform(target_data[["Volume"]])

    scaled_targets = np.concatenate((scaled_prices, scaled_volume), axis=1)
    return scaled_targets, scaler_prices, scaler_volume


def scale_features(feature_data: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """Scale feature data."""
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
    """Save scaled data to CSV."""
    scaled_data = np.concatenate((scaled_features, scaled_targets), axis=1)
    scaled_df = pd.DataFrame(scaled_data, columns=features + targets)
    scaled_df.to_csv(f"data/{symbol}_{interval}_scaled_data.csv", index=False)
    logger.info(f"Scaled dataset saved to data/{symbol}_scaled_data.csv")


def create_dataset(
        scaled_features: np.ndarray,
        scaled_targets: np.ndarray,
        look_back: int,
        look_forward: int,
        targets: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Create dataset from scaled features and targets."""
    _X, _y = [], []
    for i in range(look_back, len(scaled_features) - look_forward):
        _X.append(scaled_features[i - look_back: i])
        _y.append(scaled_targets[i + look_forward - 1, : len(targets)])

    _X = np.array(_X)
    _y = np.array(_y)
    _y = _y.reshape(-1, len(targets))
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
    return splits


def validate_data(_x: np.ndarray, _y: np.ndarray) -> None:
    """Validate dataset to ensure there are no NaN or infinite values."""
    if np.any(np.isnan(_x)) or np.any(np.isnan(_y)):
        logger.error("NaN values found in input data.")
        raise ValueError("NaN values found in input data.")
    if np.any(np.isinf(_x)) or np.any(np.isinf(_y)):
        logger.error("Infinite values found in input data.")
        raise ValueError("Infinite values found in input data.")


def select_best_features(
        _x: np.ndarray,
        _y: np.ndarray,
        best_features: List[str],
        features: List[str],
        scaled_features: np.ndarray,
        scaler_features: StandardScaler,
        scaler_prices: StandardScaler,
        scaler_volume: MinMaxScaler,
) -> Tuple[
    np.ndarray, np.ndarray, StandardScaler, StandardScaler, MinMaxScaler, List[str]
]:
    """Select predefined best features."""
    logger.info(f"Using predefined best features: {best_features}")
    feature_indices = [features.index(feature) for feature in best_features]
    validate_feature_indices(feature_indices, scaled_features.shape[1])
    _X_selected = _x[:, :, feature_indices]
    logger.info(f"Shape of _X_selected: {_X_selected.shape}")
    return _X_selected, _y, scaler_features, scaler_prices, scaler_volume, best_features


def validate_feature_indices(feature_indices: List[int], max_index: int) -> None:
    """Validate feature indices to ensure they are within bounds."""
    if any(idx >= max_index for idx in feature_indices):
        logger.error("One or more feature indices are out of bounds.")
        raise ValueError("One or more feature indices are out of bounds.")


def split_data(
        _x: np.ndarray, _y: np.ndarray, batch_size: int, test_size: float = 0.15
) -> Tuple[DataLoader, DataLoader]:
    """Split data into training and validation sets."""
    logger.info("Splitting data into training and validation sets")
    x_train, x_val, y_train, y_val = train_test_split(
        _x, _y, test_size=test_size, random_state=42
    )
    logger.info(
        f"Training data shape: {x_train.shape}, Validation data shape: {x_val.shape}"
    )
    train_data = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_data = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True),
        DataLoader(val_data, batch_size=batch_size, shuffle=False),
    )


def create_dataloader(data, batch_size=32):
    dataset = torch.utils.data.TensorDataset(*data)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_and_preprocess_data(config):
    historical_data, features = get_data(
        config.ticker,
        config.symbol,
        config.asset_type,
        config.start_date,
        config.end_date,
        config.indicator_windows,
        config.data_sampling_interval,
        config.data_resampling_frequency,
    )

    x, y, scaler_features, scaler_prices, scaler_volume, selected_features = preprocess_data(
        config.symbol,
        config.data_sampling_interval,
        historical_data,
        config.targets,
        config.look_back,
        config.look_forward,
        features,
        config.disabled_features,
        config.best_features,
    )

    if config.training_settings.get("use_time_series_split", False):
        logger.info("Using k-fold cross-validation with TimeSeriesSplit")
        logger.info(f"Number of splits: {config.training_settings.get('time_series_splits', 5)}")
        windows = create_timeseries_window(x, y, n_splits=config.training_settings.get("time_series_splits", 5))
        batch_size = config.batch_size
        train_val_loaders = [(create_dataloader(train_data, batch_size),
                              create_dataloader(val_data, batch_size)) for train_data, val_data in windows]
        return train_val_loaders, selected_features, scaler_prices, scaler_volume, historical_data, scaler_features

    train_loader, val_loader = split_data(x, y, batch_size=config.batch_size)
    return ([(train_loader, val_loader)], selected_features, scaler_prices, scaler_volume, historical_data,
            scaler_features)
