import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands


def calculate_technical_indicators(historical_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators and add them to the historical data.

    Args:
        historical_data (pd.DataFrame): The historical stock data.

    Returns:
        pd.DataFrame: The historical stock data with added technical indicators.
    """
    historical_data['SMA_20'] = SMAIndicator(historical_data['Close'], window=20).sma_indicator()
    historical_data['EMA_20'] = EMAIndicator(historical_data['Close'], window=20).ema_indicator()
    macd = MACD(historical_data['Close'])
    historical_data['MACD'] = macd.macd()
    historical_data['MACD_Signal'] = macd.macd_signal()
    historical_data['RSI_14'] = RSIIndicator(historical_data['Close'], window=14).rsi()
    stochastic = StochasticOscillator(historical_data['High'], historical_data['Low'], historical_data['Close'],
                                      window=14)
    historical_data['Stochastic'] = stochastic.stoch()
    bollinger = BollingerBands(historical_data['Close'], window=20)
    historical_data['Bollinger_High'] = bollinger.bollinger_hband()
    historical_data['Bollinger_Low'] = bollinger.bollinger_lband()

    historical_data = historical_data.dropna()
    return historical_data
