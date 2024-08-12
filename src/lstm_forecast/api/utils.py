import numpy as np
from typing import Dict, Any, List
from lstm_forecast.config import Config
from lstm_forecast.feature_engineering import calculate_technical_indicators
import pandas as pd


def preprocess_data(data: Dict[str, Any], config: Config) -> np.ndarray:
    """
    Preprocess the data for the model.

    Args:
    data (Dict[str, Any]): Data to be preprocessed.
    config (Config): Configuration object containing necessary settings.

    Returns:
    np.ndarray: Preprocessed data.
    """
    df = pd.DataFrame(data)
    df = calculate_technical_indicators(df)
    df = df.dropna()

    features = config.get_features()
    target = config.get_target()

    df = df[features + [target]]

    scaler = config.get_feature_scaler()
    normalized_data = scaler.transform(df)

    return normalized_data


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
