import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands


def calculate_technical_indicators(historical_data: pd.DataFrame, windows: dict) -> pd.DataFrame:
    """Calculate technical indicators with variable windows and add them to the historical data.

    Args:
        historical_data (pd.DataFrame): The historical stock data.
        windows (dict): A dictionary containing the window sizes for each indicator.

    Returns:
        pd.DataFrame: The historical stock data with added technical indicators.
    """
    historical_data['SMA'] = SMAIndicator(historical_data['Close'], window=windows.get('SMA', 20)).sma_indicator()
    historical_data['EMA'] = EMAIndicator(historical_data['Close'], window=windows.get('EMA', 20)).ema_indicator()
    macd = MACD(historical_data['Close'])
    historical_data['MACD'] = macd.macd()
    historical_data['MACD_Signal'] = macd.macd_signal()
    historical_data['RSI'] = RSIIndicator(historical_data['Close'], window=windows.get('RSI', 14)).rsi()
    stochastic = StochasticOscillator(historical_data['High'], historical_data['Low'], historical_data['Close'],
                                      window=windows.get('Stochastic', 14))
    historical_data['Stochastic'] = stochastic.stoch()
    bollinger = BollingerBands(historical_data['Close'], window=windows.get('Bollinger', 20))
    historical_data['Bollinger_High'] = bollinger.bollinger_hband()
    historical_data['Bollinger_Low'] = bollinger.bollinger_lband()

    historical_data = historical_data.dropna()
    return historical_data