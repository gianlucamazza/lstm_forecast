from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from lstm_forecast.feature_engineering import calculate_technical_indicators
from lstm_forecast.logger import setup_logger
from lstm_forecast.config import Config

logger = setup_logger("data_loader_logger", "logs/data_loader.log")


def get_data(config: Config) -> Tuple[pd.DataFrame, List[str]]:
    """
    Download historical data for the given symbol and preprocess it.

    Args:
        config (Config): Configuration object containing necessary settings.

    Returns:
        Tuple[pd.DataFrame, List[str]]: A tuple containing the historical data and the list of features.
    """
    logger.info(
        f"Downloading data for {config.ticker} from {config.start_date} to {config.end_date}"
    )
    historical_data = yf.download(
        config.ticker,
        start=config.start_date,
        end=config.end_date,
        interval=config.data_sampling_interval,
    )

    historical_data, features = calculate_technical_indicators(
        historical_data,
        windows=config.indicator_windows,
        frequency=config.data_resampling_frequency,
    )

    historical_data.to_csv(
        f"data/{config.symbol}_{config.data_sampling_interval}.csv"
    )
    logger.info(
        f"Data for {config.symbol} saved to data/{config.symbol}_{config.data_sampling_interval}.csv"
    )
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
) -> Tuple[
    np.ndarray, np.ndarray, StandardScaler, StandardScaler, MinMaxScaler
]:
    """
    Preprocess the historical data for the model.

    Args:
        symbol (str): The symbol of the asset.
        data_sampling_interval (str): The data sampling interval.
        historical_data (pd.DataFrame): The historical data.
        targets (List[str]): The target features.
        look_back (int): The number of time steps to look back.
        look_forward (int): The number of time steps to look forward.
        features (List[str]): The list of features.
        disabled_features (List[str]): The list of disabled features.

    Returns:
        Tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler, MinMaxScaler]: A tuple containing the preprocessed data.
    """
    logger.info("Starting preprocessing of data")
    logger.info(f"Targets: {targets}")
    logger.info(f"Look back: {look_back}, Look forward: {look_forward}")

    features = [
        f for f in features if f not in targets and f not in disabled_features
    ]
    logger.info(
        f"Features after removing targets and disabled features: {features}"
    )

    target_data = historical_data[targets]
    feature_data = historical_data[features]

    scaler_prices = StandardScaler()
    scaled_prices = scaler_prices.fit_transform(
        target_data.drop(columns=["Volume"])
    )

    scaler_volume = MinMaxScaler()
    scaled_volume = scaler_volume.fit_transform(target_data[["Volume"]])

    scaled_targets = np.concatenate((scaled_prices, scaled_volume), axis=1)

    scaler_features = StandardScaler()
    scaled_features = scaler_features.fit_transform(feature_data)

    scaled_data = np.concatenate((scaled_features, scaled_targets), axis=1)
    scaled_df = pd.DataFrame(scaled_data, columns=features + targets)
    scaled_df.to_csv(
        f"data/{symbol}_{data_sampling_interval}_scaled_data.csv", index=False
    )
    logger.info(
        f"Scaled dataset saved to data/{symbol}_{data_sampling_interval}_scaled_data.csv"
    )

    X, y = create_dataset(
        scaled_features, scaled_targets, look_back, look_forward, targets
    )

    if (
        np.any(np.isnan(X))
        or np.any(np.isnan(y))
        or np.any(np.isinf(X))
        or np.any(np.isinf(y))
    ):
        logger.error("NaN or infinite values found in input data.")
        raise ValueError("NaN or infinite values found in input data.")
    logger.info("Data validation passed without any NaN or infinite values.")

    return X, y, scaler_features, scaler_prices, scaler_volume


def create_dataset(
    scaled_features: np.ndarray,
    scaled_targets: np.ndarray,
    look_back: int,
    look_forward: int,
    targets: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create the input and target datasets for the model.

    Args:
        scaled_features (np.ndarray): The scaled feature data.
        scaled_targets (np.ndarray): The scaled target data.
        look_back (int): The number of time steps to look back.
        look_forward (int): The number of time steps to look forward.
        targets (List[str]): The target features.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the input and target datasets.
    """
    X, y = [], []
    for i in range(look_back, len(scaled_features) - look_forward + 1):
        X.append(scaled_features[i - look_back : i])
        y.append(scaled_targets[i + look_forward - 1, : len(targets)])

    X = np.array(X)
    y = np.array(y)
    logger.info(f"Shape of X: {X.shape}")
    logger.info(f"Shape of y: {y.shape}")
    return X, y


def create_dataloader(
    data: Tuple[torch.Tensor, torch.Tensor],
    batch_size: int,
    name: str = "DataLoader",
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader from the input data.

    Args:
        data (Tuple[torch.Tensor, torch.Tensor]): The input data.
        batch_size (int): The batch size.
        name (str): The name of the DataLoader.

    Returns:
        torch.utils.data.DataLoader: The DataLoader object.
    """
    dataset = torch.utils.data.TensorDataset(*data)
    logger.info(f"Creating {name} with batch size: {batch_size}")
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )


def load_and_preprocess_data(
    config: Config, selected_features: List[str] = None
) -> Tuple[
    List[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]],
    List[str],
    StandardScaler,
    MinMaxScaler,
    pd.DataFrame,
    StandardScaler,
]:
    """
    Load and preprocess the data for the model.

    Args:
        config (Config): The configuration object.
        selected_features (List[str]): The list of selected features.

    Returns:
        Tuple[List[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]], List[str], StandardScaler, MinMaxScaler, pd.DataFrame, StandardScaler, int]: A tuple containing the processed data.
    """
    historical_data, all_features = get_data(config)

    if selected_features is None:
        selected_features = config.data_settings.get(
            "selected_features", all_features
        )

    # Ensure all selected features are actually present in the data
    selected_features = [f for f in selected_features if f in all_features]

    if not selected_features:
        logger.warning(
            "No valid features selected. Using all available features."
        )
        selected_features = all_features

    logger.info(f"Processing data with selected features: {selected_features}")

    X, y, scaler_features, scaler_prices, scaler_volume = preprocess_data(
        config.symbol,
        config.data_sampling_interval,
        historical_data,
        config.targets,
        config.look_back,
        config.look_forward,
        selected_features,
        config.disabled_features,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    batch_size = config.model_settings.get("batch_size", 32)
    train_loader = create_dataloader(
        (torch.tensor(X_train).float(), torch.tensor(y_train).float()),
        batch_size=batch_size,
        name="train_loader",
    )
    val_loader = create_dataloader(
        (torch.tensor(X_val).float(), torch.tensor(y_val).float()),
        batch_size=batch_size,
        name="val_loader",
    )

    num_features = X.shape[2]
    logger.info(f"Number of features in the input data: {num_features}")

    return (
        [(train_loader, val_loader)],
        selected_features,
        scaler_prices,
        scaler_volume,
        historical_data,
        scaler_features,
        num_features,
    )


def main(config: Config, selected_features: List[str] = None) -> Tuple[
    List[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]],
    List[str],
    StandardScaler,
    MinMaxScaler,
    pd.DataFrame,
    StandardScaler,
    int,
]:
    (
        train_val_loaders,
        features,
        scaler_prices,
        scaler_volume,
        historical_data,
        scaler_features,
        num_features,
    ) = load_and_preprocess_data(config, selected_features)
    logger.info("Data loading and preprocessing completed successfully.")
    return (
        train_val_loaders,
        features,
        scaler_prices,
        scaler_volume,
        historical_data,
        scaler_features,
        num_features,
    )


if __name__ == "__main__":
    main()
