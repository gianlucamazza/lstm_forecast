import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, AroonIndicator, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from logger import setup_logger

# Configura il logger
logger = setup_logger('feature_engineering_logger', 'logs/feature_engineering.log')

def aggregate_features(df):
    """
    Aggrega feature correlate per ridurre la dimensionalità e mantenere informazioni cruciali.

    Args:
        df (pd.DataFrame): Il DataFrame contenente i dati storici e gli indicatori calcolati.

    Returns:
        pd.DataFrame: Il DataFrame con feature aggregate.
    """
    logger.info("Starting aggregation of technical indicators")

    df['SMA_Agg'] = df[['SMA_50', 'SMA_200']].mean(axis=1)
    df['MACD_Agg'] = df[['MACD', 'MACD_Signal']].mean(axis=1)
    df['Bollinger_Bandwidth'] = df['Bollinger_High'] - df['Bollinger_Low']

    return df

def calculate_technical_indicators(historical_data: pd.DataFrame, windows: dict) -> pd.DataFrame:
    """Calculate technical indicators with variable windows and add them to the historical data if specified in features.

    Args:
        historical_data (pd.DataFrame): The historical stock data.
        windows (dict): A dictionary containing the window sizes for each indicator.

    Returns:
        pd.DataFrame: The historical stock data with added technical indicators.
    """
    logger.info("Starting calculation of technical indicators")

    try:
        historical_data['SMA_20'] = SMAIndicator(historical_data['Close'], window=20).sma_indicator()
        logger.debug(f"SMA_20 values: {historical_data['SMA_20'].head()}")

        historical_data['SMA_50'] = SMAIndicator(historical_data['Close'], window=50).sma_indicator()
        logger.debug(f"SMA_50 values: {historical_data['SMA_50'].head()}")

        historical_data['SMA_200'] = SMAIndicator(historical_data['Close'], window=200).sma_indicator()
        logger.debug(f"SMA_200 values: {historical_data['SMA_200'].head()}")

        historical_data['EMA'] = EMAIndicator(historical_data['Close'], window=windows.get('EMA', 20)).ema_indicator()
        logger.debug(f"EMA values: {historical_data['EMA'].head()}")

        macd = MACD(historical_data['Close'])
        historical_data['MACD'] = macd.macd()
        historical_data['MACD_Signal'] = macd.macd_signal()
        logger.debug(f"MACD values: {historical_data['MACD'].head()}")
        logger.debug(f"MACD_Signal values: {historical_data['MACD_Signal'].head()}")

        historical_data['RSI'] = RSIIndicator(historical_data['Close'], window=windows.get('RSI', 14)).rsi()
        logger.debug(f"RSI values: {historical_data['RSI'].head()}")

        stochastic = StochasticOscillator(historical_data['High'], historical_data['Low'], historical_data['Close'],
                                          window=windows.get('Stochastic', 14))
        historical_data['Stochastic'] = stochastic.stoch()
        logger.debug(f"Stochastic values: {historical_data['Stochastic'].head()}")

        bollinger = BollingerBands(historical_data['Close'], window=windows.get('Bollinger', 20))
        historical_data['Bollinger_High'] = bollinger.bollinger_hband()
        historical_data['Bollinger_Low'] = bollinger.bollinger_lband()
        logger.debug(f"Bollinger_High values: {historical_data['Bollinger_High'].head()}")
        logger.debug(f"Bollinger_Low values: {historical_data['Bollinger_Low'].head()}")

        historical_data['ADX'] = ADXIndicator(historical_data['High'], historical_data['Low'],
                                              historical_data['Close'], window=windows.get('ADX', 14)).adx()
        logger.debug(f"ADX values: {historical_data['ADX'].head()}")

        obv = OnBalanceVolumeIndicator(close=historical_data['Close'], volume=historical_data['Volume'])
        historical_data['OBV'] = obv.on_balance_volume()
        logger.debug(f"OBV values: {historical_data['OBV'].head()}")

        typical_price = (historical_data['High'] + historical_data['Low'] + historical_data['Close']) / 3
        historical_data['VWAP'] = (typical_price * historical_data['Volume']).cumsum() / historical_data[
            'Volume'].cumsum()
        logger.debug(f"VWAP values: {historical_data['VWAP'].head()}")

        aroon = AroonIndicator(high=historical_data['High'], low=historical_data['Low'],
                               window=windows.get('Aroon', 25))
        historical_data['Aroon_Up'] = aroon.aroon_up()
        historical_data['Aroon_Down'] = aroon.aroon_down()
        logger.debug(f"Aroon_Up values: {historical_data['Aroon_Up'].head()}")
        logger.debug(f"Aroon_Down values: {historical_data['Aroon_Down'].head()}")

        historical_data['Williams_R'] = WilliamsRIndicator(historical_data['High'], historical_data['Low'],
                                                           historical_data['Close'],
                                                           lbp=windows.get('Williams_R', 14)).williams_r()
        logger.debug(f"Williams_R values: {historical_data['Williams_R'].head()}")

        historical_data['CMF'] = ChaikinMoneyFlowIndicator(historical_data['High'], historical_data['Low'],
                                                           historical_data['Close'], historical_data['Volume'],
                                                           window=windows.get('CMF', 20)).chaikin_money_flow()
        logger.debug(f"CMF values: {historical_data['CMF'].head()}")

        historical_data['ATR'] = AverageTrueRange(high=historical_data['High'], low=historical_data['Low'],
                                                  close=historical_data['Close'],
                                                  window=windows.get('ATR', 14)).average_true_range()
        logger.debug(f"ATR values: {historical_data['ATR'].head()}")

        historical_data['CCI'] = CCIIndicator(high=historical_data['High'], low=historical_data['Low'],
                                              close=historical_data['Close'],
                                              window=windows.get('CCI', 20)).cci()
        logger.debug(f"CCI values: {historical_data['CCI'].head()}")

        historical_data = historical_data.dropna()
        logger.info("Dropped NA values from historical data")

        # Aggregazione delle feature
        historical_data = aggregate_features(historical_data)

    except Exception as e:
        logger.error(f"An error occurred while calculating technical indicators: {e}")
        raise

    logger.info("Finished calculation of technical indicators")
    return historical_data
