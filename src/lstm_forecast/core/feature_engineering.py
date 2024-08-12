from typing import List, Tuple, Dict

import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import (
    SMAIndicator,
    EMAIndicator,
    MACD,
    ADXIndicator,
    AroonIndicator,
    CCIIndicator,
    IchimokuIndicator,
)
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator

from lstm_forecast.logger import setup_logger

logger = setup_logger(
    "feature_engineering_logger", "logs/feature_engineering.log"
)


def calculate_common_indicators(
    historical_data: pd.DataFrame, windows: Dict[str, int]
) -> pd.DataFrame:
    """
    Calculate common technical indicators for all asset types.

    Args:
        historical_data (pd.DataFrame): Historical data containing OHLCV values.
        windows (Dict[str, int]): Dictionary containing window sizes for technical indicators.

    Returns:
        pd.DataFrame: Historical data with calculated common technical indicators.
    """
    logger.info("Calculating common indicators")

    historical_data["SMA_50"] = SMAIndicator(
        historical_data["Close"], window=50
    ).sma_indicator()
    historical_data["SMA_200"] = SMAIndicator(
        historical_data["Close"], window=200
    ).sma_indicator()
    historical_data["EMA"] = EMAIndicator(
        historical_data["Close"], window=windows.get("EMA", 20)
    ).ema_indicator()

    macd = MACD(historical_data["Close"])
    historical_data["MACD"] = macd.macd()
    historical_data["MACD_Signal"] = macd.macd_signal()

    historical_data["RSI"] = RSIIndicator(
        historical_data["Close"], window=windows.get("RSI", 14)
    ).rsi()

    bollinger = BollingerBands(
        historical_data["Close"], window=windows.get("Bollinger", 20)
    )
    historical_data["Bollinger_High"] = bollinger.bollinger_hband()
    historical_data["Bollinger_Low"] = bollinger.bollinger_lband()

    historical_data["ADX"] = ADXIndicator(
        high=historical_data["High"],
        low=historical_data["Low"],
        close=historical_data["Close"],
        window=windows.get("ADX", 14),
    ).adx()

    return historical_data


def calculate_all_specific_indicators(
    historical_data: pd.DataFrame, windows: Dict[str, int]
) -> pd.DataFrame:
    """
    Calculate all specific technical indicators, regardless of asset type.

    Args:
        historical_data (pd.DataFrame): Historical data containing OHLCV values.
        windows (Dict[str, int]): Dictionary containing window sizes for technical indicators.

    Returns:
        pd.DataFrame: Historical data with calculated specific technical indicators.
    """
    logger.info("Calculating all specific indicators")

    historical_data["OBV"] = OnBalanceVolumeIndicator(
        close=historical_data["Close"], volume=historical_data["Volume"]
    ).on_balance_volume()
    historical_data["VWAP"] = (
        historical_data["Close"] * historical_data["Volume"]
    ).cumsum() / historical_data["Volume"].cumsum()
    historical_data["Stochastic"] = StochasticOscillator(
        historical_data["High"],
        historical_data["Low"],
        historical_data["Close"],
        window=windows.get("Stochastic", 14),
    ).stoch()

    aroon_indicator = AroonIndicator(
        high=historical_data["High"],
        low=historical_data["Low"],
        window=windows.get("Aroon", 25),
    )
    historical_data["Aroon_Up"] = aroon_indicator.aroon_up()
    historical_data["Aroon_Down"] = aroon_indicator.aroon_down()

    historical_data["CCI"] = CCIIndicator(
        high=historical_data["High"],
        low=historical_data["Low"],
        close=historical_data["Close"],
        window=windows.get("CCI", 20),
    ).cci()
    historical_data["CMF"] = ChaikinMoneyFlowIndicator(
        high=historical_data["High"],
        low=historical_data["Low"],
        close=historical_data["Close"],
        volume=historical_data["Volume"],
        window=windows.get("CMF", 20),
    ).chaikin_money_flow()

    return historical_data


def calculate_ichimoku(historical_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Ichimoku indicators.

    Args:
        historical_data (pd.DataFrame): Historical data containing OHLCV values.

    Returns:
        pd.DataFrame: Historical data with calculated Ichimoku indicators.
    """
    logger.info("Calculating Ichimoku indicators")

    ichimoku = IchimokuIndicator(
        historical_data["High"],
        historical_data["Low"],
        window1=9,
        window2=26,
        window3=52,
    )
    historical_data["Ichimoku_Tenkan"] = ichimoku.ichimoku_conversion_line()
    historical_data["Ichimoku_Kijun"] = ichimoku.ichimoku_base_line()
    historical_data["Ichimoku_Senkou_Span_A"] = ichimoku.ichimoku_a()
    historical_data["Ichimoku_Senkou_Span_B"] = ichimoku.ichimoku_b()

    return historical_data


def calculate_volume_indicators(
    historical_data: pd.DataFrame, windows: Dict[str, int]
) -> pd.DataFrame:
    """
    Calculate volume-based indicators.

    Args:
        historical_data (pd.DataFrame): Historical data containing OHLCV values.
        windows (Dict[str, int]): Dictionary containing window sizes for technical indicators.

    Returns:
        pd.DataFrame: Historical data with calculated volume-based indicators.
    """
    logger.info("Calculating volume-based indicators")

    historical_data["Volume_SMA"] = SMAIndicator(
        historical_data["Volume"], window=windows.get("Volume_SMA", 20)
    ).sma_indicator()
    historical_data["Volume_Change"] = historical_data["Volume"].pct_change()
    historical_data["Volume_RSI"] = RSIIndicator(
        historical_data["Volume"], window=windows.get("Volume_RSI", 14)
    ).rsi()

    return historical_data


def calculate_technical_indicators(
    historical_data: pd.DataFrame, windows: Dict[str, int], frequency: str
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Calculate all technical indicators and add them to the historical data.

    Args:
        historical_data (pd.DataFrame): Historical data containing OHLCV values.
        windows (Dict[str, int]): Dictionary containing window sizes for technical indicators.
        frequency (str): Frequency of the data.

    Returns:
        Tuple[pd.DataFrame, List[str]]:
            - Historical data with calculated technical indicators.
            - List of feature names.
    """
    logger.info("Starting calculation of all technical indicators")

    historical_data.index = pd.to_datetime(historical_data.index)
    historical_data = historical_data.asfreq(frequency)

    historical_data = calculate_common_indicators(historical_data, windows)
    historical_data = calculate_all_specific_indicators(
        historical_data, windows
    )
    historical_data = calculate_ichimoku(historical_data)
    historical_data = calculate_volume_indicators(historical_data, windows)

    historical_data = historical_data.dropna()
    logger.info("Dropped NA values from historical data")

    features = list(historical_data.columns)
    return historical_data, features
