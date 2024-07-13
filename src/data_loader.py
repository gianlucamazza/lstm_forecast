from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from numpy import ndarray, dtype
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV, MultiTaskLassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from src.feature_engineering import calculate_technical_indicators
from src.logger import setup_logger

logger = setup_logger("data_loader_logger", "logs/data_loader.log")

FEATURE_SELECTION_ALGOS = {
    "none": "none",  # No feature selection
    "random_forest": RandomForestRegressor(
        n_estimators=100, random_state=42, max_depth=10
    ),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "extra_trees": ExtraTreesRegressor(n_estimators=100, random_state=42, max_depth=10),
    "lasso": LassoCV(cv=5, random_state=42, max_iter=10000, tol=1e-6),
    "multi_task_lasso": MultiTaskLassoCV(
        cv=5, random_state=42, max_iter=20000, tol=1e-6
    ),
    "xgboost": XGBRegressor(n_estimators=100, random_state=42),
    "lightgbm": LGBMRegressor(n_estimators=100, random_state=42),
    "catboost": CatBoostRegressor(
        iterations=100, random_seed=42, logging_level="Silent"
    ),
    "svr": RFE(estimator=SVR(kernel="linear"), n_features_to_select=10),
    "rfe": RFE(
        estimator=RandomForestRegressor(
            n_estimators=100, random_state=42, max_depth=10
        ),
        n_features_to_select=10,
    ),
}


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
    """Download and preprocess historical data."""
    end_date = pd.to_datetime(end)

    if data_sampling_interval != "1d":
        start = adjust_start_date_if_needed(start, end, end_date)

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
    save_historical_data(symbol, historical_data)
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


def save_historical_data(symbol: str, historical_data: pd.DataFrame) -> None:
    """Save historical data to CSV."""
    historical_data.to_csv(f"data/{symbol}.csv")
    logger.info(f"Data for {symbol} saved to data/{symbol}.csv")


def preprocess_data(
        symbol: str,
        historical_data: pd.DataFrame,
        targets: List[str],
        look_back: int = 60,
        look_forward: int = 30,
        features: List[str] = None,
        best_features: List[str] = None,
        max_iter: int = 100,
        feature_selection_algo: str = "random_forest",
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

    features = [f for f in features if f not in targets]  # Remove targets from features
    logger.info(f"Features after removing targets: {features}")

    target_data, feature_data = split_data_into_targets_and_features(
        historical_data, targets, features
    )
    scaled_targets, scaler_prices, scaler_volume = scale_targets(target_data)
    scaled_features, scaler_features = scale_features(feature_data)

    save_scaled_data(symbol, scaled_features, scaled_targets, features, targets)

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
        _X, _y, features, max_iter, feature_selection_algo
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


def select_features(
        _x: np.ndarray, _y: np.ndarray, features: List[str], max_iter: int, algo: str
) -> List[int]:
    """Perform feature selection using the selected algorithm."""
    _X_reshaped = _x.reshape(_x.shape[0], -1)
    logger.info(f"Shape of _X_reshaped: {_X_reshaped.shape}")

    if algo not in FEATURE_SELECTION_ALGOS:
        logger.error(f"Feature selection algorithm {algo} is not recognized.")
        raise ValueError(f"Feature selection algorithm {algo} is not recognized.")

    if algo == "none":
        logger.info("No feature selection algorithm specified. Using all features.")
        return list(range(len(features)))

    logger.info(f"Starting feature selection using {algo}")
    model = FEATURE_SELECTION_ALGOS[algo]

    selected_features_indices = []

    if algo in [
        "gradient_boosting",
        "random_forest",
        "extra_trees",
        "xgboost",
        "lightgbm",
        "catboost",
        "svr",
        "rfe",
    ]:
        for i in range(_y.shape[1]):
            logger.info(f"Feature selection for target column {i + 1}/{_y.shape[1]}")
            model.fit(_X_reshaped, _y[:, i])
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "ranking_"):  # For RFE
                importances = -model.ranking_
            else:
                raise ValueError(
                    "Model does not have feature importances or coefficients."
                )

            indices = np.argsort(importances)[::-1]
            selected_features_indices.extend(indices[: min(len(indices), max_iter)])

        selected_features_indices = list(set(selected_features_indices))
        selected_features_indices.sort()

    else:
        if algo == "lasso" and _y.ndim > 1 and _y.shape[1] > 1:
            logger.info("Using MultiTaskLassoCV for multi-task output.")
            model = MultiTaskLassoCV(cv=5, random_state=42, max_iter=20000, tol=1e-6)

        if algo == "rfe":
            model = RFE(estimator=model, n_features_to_select=max_iter)

        model.verbose = 1
        model.fit(_X_reshaped, _y)

        if hasattr(model, "coef_"):  # For Lasso and MultiTaskLasso
            importances = np.abs(model.coef_).sum(axis=0)
        elif hasattr(model, "feature_importances_"):  # For RandomForest and XGBoost
            importances = model.feature_importances_
        elif hasattr(model, "ranking_"):  # For RFE
            importances = -model.ranking_
        else:
            raise ValueError("Model does not have feature importances or coefficients.")

        indices = np.argsort(importances)[::-1]
        selected_features_indices = indices[: min(len(indices), max_iter)].tolist()

    logger.info(f"Selected feature indices: {selected_features_indices}")

    max_index = len(features) - 1
    selected_features_indices = [i for i in selected_features_indices if i <= max_index]
    logger.info(f"Validated selected feature indices: {selected_features_indices}")

    return selected_features_indices


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
        scaled_features: np.ndarray,
        scaled_targets: np.ndarray,
        features: List[str],
        targets: List[str],
) -> None:
    """Save scaled data to CSV."""
    scaled_data = np.concatenate((scaled_features, scaled_targets), axis=1)
    scaled_df = pd.DataFrame(scaled_data, columns=features + targets)
    scaled_df.to_csv(f"data/{symbol}_scaled_data.csv", index=False)
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
