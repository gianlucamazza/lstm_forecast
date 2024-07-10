import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, AroonIndicator, CCIIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
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
    
    df = df.copy()
    
    if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
        df.loc[:, 'SMA_Agg'] = df[['SMA_50', 'SMA_200']].mean(axis=1)
    
    if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
        df.loc[:, 'MACD_Agg'] = df[['MACD', 'MACD_Signal']].mean(axis=1)
    
    if 'Bollinger_High' in df.columns and 'Bollinger_Low' in df.columns:
        df.loc[:, 'Bollinger_Bandwidth'] = df['Bollinger_High'] - df['Bollinger_Low']

    return df

def calculate_stock_indicators(historical_data: pd.DataFrame, windows: dict) -> tuple:
    logger.info("Calculating stock indicators")
    
    historical_data['SMA_50'] = SMAIndicator(historical_data['Close'], window=50).sma_indicator()
    historical_data['SMA_200'] = SMAIndicator(historical_data['Close'], window=200).sma_indicator()
    historical_data['EMA'] = EMAIndicator(historical_data['Close'], window=windows.get('EMA', 20)).ema_indicator()

    macd = MACD(historical_data['Close'])
    historical_data['MACD'] = macd.macd()
    historical_data['MACD_Signal'] = macd.macd_signal()

    historical_data['RSI'] = RSIIndicator(historical_data['Close'], window=windows.get('RSI', 14)).rsi()

    bollinger = BollingerBands(historical_data['Close'], window=windows.get('Bollinger', 20))
    historical_data['Bollinger_High'] = bollinger.bollinger_hband()
    historical_data['Bollinger_Low'] = bollinger.bollinger_lband()

    historical_data['ADX'] = ADXIndicator(historical_data['High'], historical_data['Low'], historical_data['Close'], window=windows.get('ADX', 14)).adx()
    historical_data['OBV'] = OnBalanceVolumeIndicator(close=historical_data['Close'], volume=historical_data['Volume']).on_balance_volume()
    historical_data['VWAP'] = (historical_data['Close'] * historical_data['Volume']).cumsum() / historical_data['Volume'].cumsum()

    ichimoku = IchimokuIndicator(historical_data['High'], historical_data['Low'], window1=9, window2=26, window3=52)
    historical_data['Ichimoku_Tenkan'] = ichimoku.ichimoku_conversion_line()
    historical_data['Ichimoku_Kijun'] = ichimoku.ichimoku_base_line()
    historical_data['Ichimoku_Senkou_Span_A'] = ichimoku.ichimoku_a()
    historical_data['Ichimoku_Senkou_Span_B'] = ichimoku.ichimoku_b()

    historical_data = historical_data.dropna()
    logger.info("Dropped NA values from historical data")

    historical_data = aggregate_features(historical_data)

    return historical_data, [
        'SMA_50', 'SMA_200', 'EMA', 'MACD', 'MACD_Signal', 'RSI', 
        'Bollinger_High', 'Bollinger_Low', 'ADX', 'OBV', 'VWAP', 
        'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_Senkou_Span_A', 
        'Ichimoku_Senkou_Span_B', 'SMA_Agg', 'MACD_Agg', 'Bollinger_Bandwidth'
    ]


def calculate_forex_indicators(historical_data: pd.DataFrame, windows: dict) -> tuple:
    logger.info("Calculating forex indicators")
    
    historical_data['SMA_20'] = SMAIndicator(historical_data['Close'], window=20).sma_indicator()
    historical_data['EMA'] = EMAIndicator(historical_data['Close'], window=windows.get('EMA', 20)).ema_indicator()

    macd = MACD(historical_data['Close'])
    historical_data['MACD'] = macd.macd()
    historical_data['MACD_Signal'] = macd.macd_signal()

    historical_data['RSI'] = RSIIndicator(historical_data['Close'], window=windows.get('RSI', 14)).rsi()
    historical_data['Stochastic'] = StochasticOscillator(historical_data['High'], historical_data['Low'], historical_data['Close'], window=windows.get('Stochastic', 14)).stoch()
    
    bollinger = BollingerBands(historical_data['Close'], window=windows.get('Bollinger', 20))
    historical_data['Bollinger_High'] = bollinger.bollinger_hband()
    historical_data['Bollinger_Low'] = bollinger.bollinger_lband()

    historical_data['ADX'] = ADXIndicator(historical_data['High'], historical_data['Low'], historical_data['Close'], window=windows.get('ADX', 14)).adx()
    historical_data['Aroon_Up'] = AroonIndicator(high=historical_data['High'], low=historical_data['Low'], window=windows.get('Aroon', 25)).aroon_up()
    historical_data['Aroon_Down'] = AroonIndicator(high=historical_data['High'], low=historical_data['Low'], window=windows.get('Aroon', 25)).aroon_down()

    ichimoku = IchimokuIndicator(historical_data['High'], historical_data['Low'], window1=9, window2=26, window3=52)
    historical_data['Ichimoku_Tenkan'] = ichimoku.ichimoku_conversion_line()
    historical_data['Ichimoku_Kijun'] = ichimoku.ichimoku_base_line()
    historical_data['Ichimoku_Senkou_Span_A'] = ichimoku.ichimoku_a()
    historical_data['Ichimoku_Senkou_Span_B'] = ichimoku.ichimoku_b()

    historical_data = historical_data.dropna()
    logger.info("Dropped NA values from historical data")

    historical_data = aggregate_features(historical_data)

    return historical_data, [
        'SMA_20', 'EMA', 'MACD', 'MACD_Signal', 'RSI', 'Stochastic', 
        'Bollinger_High', 'Bollinger_Low', 'ADX', 'Aroon_Up', 'Aroon_Down', 
        'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_Senkou_Span_A', 
        'Ichimoku_Senkou_Span_B', 'MACD_Agg', 'Bollinger_Bandwidth'
    ]
    
    
def calculate_commodity_indicators(historical_data: pd.DataFrame, windows: dict) -> tuple:
    logger.info("Calculating commodity indicators")
    
    historical_data['SMA_20'] = SMAIndicator(historical_data['Close'], window=20).sma_indicator()
    historical_data['EMA'] = EMAIndicator(historical_data['Close'], window=windows.get('EMA', 20)).ema_indicator()

    macd = MACD(historical_data['Close'])
    historical_data['MACD'] = macd.macd()
    historical_data['MACD_Signal'] = macd.macd_signal()

    historical_data['RSI'] = RSIIndicator(historical_data['Close'], window=windows.get('RSI', 14)).rsi()
    historical_data['Stochastic'] = StochasticOscillator(historical_data['High'], historical_data['Low'], historical_data['Close'], window=windows.get('Stochastic', 14)).stoch()
    
    bollinger = BollingerBands(historical_data['Close'], window=windows.get('Bollinger', 20))
    historical_data['Bollinger_High'] = bollinger.bollinger_hband()
    historical_data['Bollinger_Low'] = bollinger.bollinger_lband()

    historical_data['ADX'] = ADXIndicator(historical_data['High'], historical_data['Low'], historical_data['Close'], window=windows.get('ADX', 14)).adx()
    historical_data['CCI'] = CCIIndicator(high=historical_data['High'], low=historical_data['Low'], close=historical_data['Close'], window=windows.get('CCI', 20)).cci()
    historical_data['CMF'] = ChaikinMoneyFlowIndicator(high=historical_data['High'], low=historical_data['Low'], close=historical_data['Close'], volume=historical_data['Volume'], window=windows.get('CMF', 20)).chaikin_money_flow()

    ichimoku = IchimokuIndicator(historical_data['High'], historical_data['Low'], window1=9, window2=26, window3=52)
    historical_data['Ichimoku_Tenkan'] = ichimoku.ichimoku_conversion_line()
    historical_data['Ichimoku_Kijun'] = ichimoku.ichimoku_base_line()
    historical_data['Ichimoku_Senkou_Span_A'] = ichimoku.ichimoku_a()
    historical_data['Ichimoku_Senkou_Span_B'] = ichimoku.ichimoku_b()

    historical_data = historical_data.dropna()
    logger.info("Dropped NA values from historical data")

    historical_data = aggregate_features(historical_data)

    return historical_data, [
        'SMA_20', 'EMA', 'MACD', 'MACD_Signal', 'RSI', 'Stochastic', 
        'Bollinger_High', 'Bollinger_Low', 'ADX', 'CCI', 'CMF', 
        'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_Senkou_Span_A', 
        'Ichimoku_Senkou_Span_B', 'MACD_Agg', 'Bollinger_Bandwidth'
    ]


def calculate_etf_indicators(historical_data: pd.DataFrame, windows: dict) -> tuple:
    logger.info("Calculating ETF indicators")
    
    historical_data['SMA_50'] = SMAIndicator(historical_data['Close'], window=50).sma_indicator()
    historical_data['SMA_200'] = SMAIndicator(historical_data['Close'], window=200).sma_indicator()
    historical_data['EMA'] = EMAIndicator(historical_data['Close'], window=windows.get('EMA', 20)).ema_indicator()

    macd = MACD(historical_data['Close'])
    historical_data['MACD'] = macd.macd()
    historical_data['MACD_Signal'] = macd.macd_signal()

    historical_data['RSI'] = RSIIndicator(historical_data['Close'], window=windows.get('RSI', 14)).rsi()

    bollinger = BollingerBands(historical_data['Close'], window=windows.get('Bollinger', 20))
    historical_data['Bollinger_High'] = bollinger.bollinger_hband()
    historical_data['Bollinger_Low'] = bollinger.bollinger_lband()

    historical_data['ADX'] = ADXIndicator(historical_data['High'], historical_data['Low'], historical_data['Close'], window=windows.get('ADX', 14)).adx()
    historical_data['CMF'] = ChaikinMoneyFlowIndicator(high=historical_data['High'], low=historical_data['Low'], close=historical_data['Close'], volume=historical_data['Volume'], window=windows.get('CMF', 20)).chaikin_money_flow()

    ichimoku = IchimokuIndicator(historical_data['High'], historical_data['Low'], window1=9, window2=26, window3=52)
    historical_data['Ichimoku_Tenkan'] = ichimoku.ichimoku_conversion_line()
    historical_data['Ichimoku_Kijun'] = ichimoku.ichimoku_base_line()
    historical_data['Ichimoku_Senkou_Span_A'] = ichimoku.ichimoku_a()
    historical_data['Ichimoku_Senkou_Span_B'] = ichimoku.ichimoku_b()

    historical_data = historical_data.dropna()
    logger.info("Dropped NA values from historical data")

    historical_data = aggregate_features(historical_data)

    return historical_data, [
        'SMA_50', 'SMA_200', 'EMA', 'MACD', 'MACD_Signal', 'RSI', 
        'Bollinger_High', 'Bollinger_Low', 'ADX', 'CMF', 
        'Ichimoku_Tenkan', 'Ichimoku_Kijun', 'Ichimoku_Senkou_Span_A', 
        'Ichimoku_Senkou_Span_B', 'SMA_Agg', 'MACD_Agg', 'Bollinger_Bandwidth'
    ]

def calculate_technical_indicators(historical_data: pd.DataFrame, windows: dict, asset_type: str, frequency: str) -> tuple:
    """
    Calculate technical indicators based on asset type and add them to the historical data.

    Args:
        historical_data (pd.DataFrame): The historical stock data.
        windows (dict): A dictionary containing the window sizes for each indicator.
        asset_type (str): The type of asset ('stocks', 'forex', 'commodity', 'etf').

    Returns:
        pd.DataFrame: The historical stock data with added technical indicators.
        list: The list of features used.
    """
    logger.info(f"Starting calculation of technical indicators for {asset_type}")
    historical_data.index = pd.to_datetime(historical_data.index)
    historical_data = historical_data.asfreq(frequency)

    if asset_type == 'stocks':
        return calculate_stock_indicators(historical_data, windows)
    elif asset_type == 'forex':
        return calculate_forex_indicators(historical_data, windows)
    elif asset_type == 'commodity':
        return calculate_commodity_indicators(historical_data, windows)
    elif asset_type == 'etf':
        return calculate_etf_indicators(historical_data, windows)
    else:
        logger.error(f"Unsupported asset type: {asset_type}")
        raise ValueError(f"Unsupported asset type: {asset_type}")
    