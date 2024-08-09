import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from typing import Dict, Any, List
from config import Config
from feature_engineering import calculate_technical_indicators
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
    
    # Converti i dati di input in un DataFrame
    df = pd.DataFrame(data['ohlcv'], columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    # Calcola gli indicatori tecnici
    df, features = calculate_technical_indicators(
        df,
        windows=config.indicator_windows,
        asset_type=config.asset_type,
        frequency=config.data_resampling_frequency,
    )
    
    # Seleziona solo le feature utilizzate dal modello
    selected_features = config.data_settings["selected_features"]
    feature_data = df[selected_features]
    
    # Normalizza i dati (assumendo che usi StandardScaler come nel tuo codice di training)
    scaler = config.get_scaler()  # Assumi che ci sia un metodo per ottenere lo scaler salvato
    normalized_data = scaler.transform(feature_data)
    
    # Reshape per il modello LSTM (assumendo una sequenza di input)
    sequence_length = config.model_settings.get("sequence_length", 120)
    if len(normalized_data) < sequence_length:
        raise ValueError(f"Not enough data. Expected at least {sequence_length} data points.")
    
    model_input = np.array([normalized_data[-sequence_length:]], dtype=np.float32)
    
    return model_input

def postprocess_prediction(prediction: np.ndarray, config: Config) -> List[float]:
    """
    Postprocess the model's prediction.
    
    Args:
    prediction (np.ndarray): Raw prediction from the model.
    config (Config): Configuration object containing necessary settings.
    
    Returns:
    List[float]: Postprocessed prediction.
    """
    # Assumiamo che la predizione sia per i prossimi n timestep dei target
    # dove n è la lunghezza dell'output del modello
    
    # De-normalizza la predizione
    scaler = config.get_target_scaler()  # Assumi che ci sia un metodo per ottenere lo scaler dei target
    denormalized_prediction = scaler.inverse_transform(prediction)
    
    # Converti in lista e arrotonda a 2 decimali
    processed_prediction = [round(float(x), 2) for x in denormalized_prediction.flatten()]
    
    return processed_prediction
