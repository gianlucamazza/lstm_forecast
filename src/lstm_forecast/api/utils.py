import numpy as np
from typing import Dict, Any, List
from config import Config
from lstm_forecast.feature_engineering import calculate_technical_indicators
import pandas as pd


def preprocess_data(data: Dict[str, Any]) -> np.ndarray:
    """
    Preprocess the input data for the model.

    Args:
    data (Dict[str, Any]): A dictionary containing the input data.
        Expected to have 'ohlcv' key with OHLCV data.

    Returns:
    np.ndarray: Preprocessed data ready for model input.
    """
    config = Config()

    df = pd.DataFrame(
        data["ohlcv"], columns=["Open", "High", "Low", "Close", "Volume"]
    )

    df, features = calculate_technical_indicators(
        df,
        windows=config.indicator_windows,
        asset_type=config.asset_type,
        frequency=config.data_resampling_frequency,
    )

    selected_features = config.data_settings["selected_features"]
    feature_data = df[selected_features]

    scaler = config.get_scaler()
    normalized_data = scaler.transform(feature_data)

    sequence_length = config.model_settings.get("sequence_length", 120)
    if len(normalized_data) < sequence_length:
        raise ValueError(f"Insufficient data. Need {sequence_length}+ points.")

    model_input = np.array(
        [normalized_data[-sequence_length:]], dtype=np.float32
    )

    return model_input


def postprocess_prediction(
    prediction: np.ndarray, config: Config
) -> List[float]:
    """
    Postprocess the model's prediction.

    Args:
    prediction (np.ndarray): Raw prediction from the model.
    config (Config): Configuration object containing necessary settings.

    Returns:
    List[float]: Postprocessed prediction.
    """
    scaler = config.get_target_scaler()
    denormalized_prediction = scaler.inverse_transform(prediction)

    processed_prediction = [
        round(float(x), 2) for x in denormalized_prediction.flatten()
    ]

    return processed_prediction
