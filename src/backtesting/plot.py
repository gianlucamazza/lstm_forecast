import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_predictions_with_orders(symbol: str, filename: str, historical_data: np.ndarray, predictions: np.ndarray,
                                 future_predictions: np.ndarray, data: pd.DataFrame, freq: str, transactions: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, historical_data, label='Historical Prices')
    aligned_predictions = np.zeros_like(historical_data)
    aligned_predictions[-len(predictions):] = predictions
    plt.plot(data.index, aligned_predictions, label='Predicted Prices', color='red')
    future_dates = pd.date_range(data.index[-1], periods=len(future_predictions) + 1, freq=freq)[1:]
    plt.plot(future_dates, future_predictions, label='Predicted Future Prices', linestyle='dashed')

    buys = transactions[transactions['type'] == 'buy']
    sells = transactions[transactions['type'] == 'sell']
    plt.scatter(buys['date'], buys['price'].apply(lambda x: float(x.replace('$', '').replace(',', ''))), marker='^',
                color='green', label='Buy', alpha=1)
    plt.scatter(sells['date'], sells['price'].apply(lambda x: float(x.replace('$', '').replace(',', ''))), marker='v',
                color='red', label='Sell', alpha=1)

    plt.title(f'{symbol} {data.index[0].strftime("%Y-%m-%d")} to {data.index[-1].strftime("%Y-%m-%d")}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(filename)
