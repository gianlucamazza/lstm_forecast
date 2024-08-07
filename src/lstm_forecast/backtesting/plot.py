import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_predictions_with_orders(symbol: str, filename: str, historical_data: np.ndarray, predictions: np.ndarray,
                                 future_predictions: np.ndarray, data: pd.DataFrame, freq: str, transactions: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 7))
    
    # Plot historical prices
    plt.plot(data.index, historical_data, label='Historical Prices', color='blue', linewidth=1.5)
    
    # Plot predictions
    aligned_predictions = np.full_like(historical_data, np.nan)
    aligned_predictions[-len(predictions):] = predictions
    plt.plot(data.index, aligned_predictions, label='Predicted Prices', color='red', linestyle='dashed', linewidth=1.5)
    
    # Plot future predictions
    future_dates = pd.date_range(data.index[-1], periods=len(future_predictions) + 1, freq=freq)[1:]
    plt.plot(future_dates, future_predictions, label='Predicted Future Prices', color='orange', linestyle='dotted', linewidth=1.5)

    # Plot buy and sell transactions
    buys = transactions[transactions['type'] == 'buy']
    sells = transactions[transactions['type'] == 'sell']
    
    plt.scatter(buys['date'], buys['price'], marker='^', color='green', label='Buy', alpha=1, edgecolors='k', zorder=5)
    plt.scatter(sells['date'], sells['price'], marker='v', color='red', label='Sell', alpha=1, edgecolors='k', zorder=5)

    # Annotations for buys
    for idx, row in buys.iterrows():
        plt.annotate('Buy', (row['date'], row['price']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='green')

    # Annotations for sells
    for idx, row in sells.iterrows():
        plt.annotate('Sell', (row['date'], row['price']), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8, color='red')

    # Titles and labels
    plt.title(f'{symbol} Price Prediction and Trading Signals ({data.index[0].strftime("%Y-%m-%d")} to {data.index[-1].strftime("%Y-%m-%d")})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    
    # Adding grid
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Adding legend
    plt.legend()
    
    # Save plot to file
    plt.savefig(filename)
    plt.close()
