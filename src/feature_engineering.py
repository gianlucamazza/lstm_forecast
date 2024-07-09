import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, AroonIndicator, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator


def calculate_technical_indicators(historical_data: pd.DataFrame, windows: dict, features: list) -> pd.DataFrame:
    """Calculate technical indicators with variable windows and add them to the historical data if specified in features.

    Args:
        historical_data (pd.DataFrame): The historical stock data.
        windows (dict): A dictionary containing the window sizes for each indicator.
        features (list): A list of features to be calculated and added.

    Returns:
        pd.DataFrame: The historical stock data with added technical indicators.
    """
    if 'SMA_20' in features:
        historical_data['SMA_20'] = SMAIndicator(historical_data['Close'], window=20).sma_indicator()

    if 'SMA_50' in features:
        historical_data['SMA_50'] = SMAIndicator(historical_data['Close'], window=50).sma_indicator()

    if 'SMA_200' in features:
        historical_data['SMA_200'] = SMAIndicator(historical_data['Close'], window=200).sma_indicator()

    if 'EMA' in features:
        historical_data['EMA'] = EMAIndicator(historical_data['Close'], window=windows.get('EMA', 20)).ema_indicator()

    if 'MACD' in features or 'MACD_Signal' in features:
        macd = MACD(historical_data['Close'])
        if 'MACD' in features:
            historical_data['MACD'] = macd.macd()
        if 'MACD_Signal' in features:
            historical_data['MACD_Signal'] = macd.macd_signal()

    if 'RSI' in features:
        historical_data['RSI'] = RSIIndicator(historical_data['Close'], window=windows.get('RSI', 14)).rsi()

    if 'Stochastic' in features:
        stochastic = StochasticOscillator(historical_data['High'], historical_data['Low'], historical_data['Close'],
                                          window=windows.get('Stochastic', 14))
        historical_data['Stochastic'] = stochastic.stoch()

    if 'Bollinger_High' in features or 'Bollinger_Low' in features:
        bollinger = BollingerBands(historical_data['Close'], window=windows.get('Bollinger', 20))
        if 'Bollinger_High' in features:
            historical_data['Bollinger_High'] = bollinger.bollinger_hband()
        if 'Bollinger_Low' in features:
            historical_data['Bollinger_Low'] = bollinger.bollinger_lband()

    if 'ADX' in features:
        historical_data['ADX'] = ADXIndicator(historical_data['High'], historical_data['Low'],
                                              historical_data['Close'], window=windows.get('ADX', 14)).adx()

    if 'OBV' in features:
        obv = OnBalanceVolumeIndicator(close=historical_data['Close'], volume=historical_data['Volume'])
        historical_data['OBV'] = obv.on_balance_volume()

    if 'VWAP' in features:
        typical_price = (historical_data['High'] + historical_data['Low'] + historical_data['Close']) / 3
        historical_data['VWAP'] = (typical_price * historical_data['Volume']).cumsum() / historical_data[
            'Volume'].cumsum()

    if 'Aroon_Up' in features or 'Aroon_Down' in features:
        aroon = AroonIndicator(high=historical_data['High'], low=historical_data['Low'],
                               window=windows.get('Aroon', 25))
        if 'Aroon_Up' in features:
            historical_data['Aroon_Up'] = aroon.aroon_up()
        if 'Aroon_Down' in features:
            historical_data['Aroon_Down'] = aroon.aroon_down()

    if 'Williams_R' in features:
        historical_data['Williams_R'] = WilliamsRIndicator(historical_data['High'], historical_data['Low'],
                                                           historical_data['Close'],
                                                           lbp=windows.get('Williams_R', 14)).williams_r()

    if 'CMF' in features:
        historical_data['CMF'] = ChaikinMoneyFlowIndicator(historical_data['High'], historical_data['Low'],
                                                           historical_data['Close'], historical_data['Volume'],
                                                           window=windows.get('CMF', 20)).chaikin_money_flow()

    if 'ATR' in features:
        historical_data['ATR'] = AverageTrueRange(high=historical_data['High'], low=historical_data['Low'],
                                                  close=historical_data['Close'],
                                                  window=windows.get('ATR', 14)).average_true_range()

    if 'CCI' in features:
        historical_data['CCI'] = CCIIndicator(high=historical_data['High'], low=historical_data['Low'],
                                              close=historical_data['Close'],
                                              window=windows.get('CCI', 20)).cci()

    historical_data = historical_data.dropna()
    return historical_data
