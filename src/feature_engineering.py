import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator


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
    historical_data['ATR'] = AverageTrueRange(historical_data['High'], historical_data['Low'], historical_data['Close'], 
                                              window=windows.get('ATR', 14)).average_true_range()
    historical_data['ADX'] = ADXIndicator(historical_data['High'], historical_data['Low'], historical_data['Close'], 
                                          window=windows.get('ADX', 14)).adx()
    historical_data['OBV'] = OnBalanceVolumeIndicator(historical_data['Close'], historical_data['Volume']).on_balance_volume()
    historical_data['MFI'] = MFIIndicator(historical_data['High'], historical_data['Low'], historical_data['Close'], 
                                          historical_data['Volume'], window=windows.get('MFI', 14)).money_flow_index()

    historical_data = historical_data.dropna()
    return historical_data