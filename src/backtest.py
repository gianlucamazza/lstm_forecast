import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
from typing import List
from sklearn.preprocessing import StandardScaler
from model import PricePredictor
from data_loader import preprocess_data, get_data
from utils import load_json
from logger import setup_logger

# Set up logger
logger = setup_logger('backtest_logger', 'logs/backtest.log')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Device: {device}")

class FakePortfolio:
    def __init__(self, initial_cash, trading_fee=0.001):
        self.cash = initial_cash
        self.holdings = {}  # {symbol: quantity}
        self.transaction_history = []
        self.trading_fee = trading_fee

    def buy(self, symbol, quantity, price, date):
        cost = quantity * price
        fee = cost * self.trading_fee
        total_cost = cost + fee
        if self.cash >= total_cost:
            self.cash -= total_cost
            self.holdings[symbol] = self.holdings.get(symbol, 0) + quantity
            self.transaction_history.append({
                'type': 'buy',
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'total_cost': total_cost,
                'fee': fee,
                'cash_remaining': self.cash,
                'date': date
            })
        else:
            logger.warning("Not enough cash to buy")

    def sell(self, symbol, quantity, price, date):
        if self.holdings.get(symbol, 0) >= quantity:
            revenue = quantity * price
            fee = revenue * self.trading_fee
            total_revenue = revenue - fee
            self.holdings[symbol] -= quantity
            self.cash += total_revenue
            self.transaction_history.append({
                'type': 'sell',
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'total_revenue': total_revenue,
                'fee': fee,
                'cash_remaining': self.cash,
                'date': date
            })
        else:
            logger.warning("Not enough holdings to sell")

    def portfolio_value(self, current_prices):
        total_value = self.cash
        for symbol, quantity in self.holdings.items():
            total_value += quantity * current_prices.get(symbol, 0)
        return total_value

    def transaction_log(self):
        return pd.DataFrame(self.transaction_history)

    def performance_report(self, current_prices):
        total_value = self.portfolio_value(current_prices)
        profit = total_value - self.cash
        roi = (profit / self.cash) * 100
        return pd.DataFrame([{
            'initial_cash': self.cash,
            'final_cash': self.cash,
            'total_value': total_value,
            'profit': profit,
            'ROI (%)': roi
        }])

def load_model(symbol: str, path: str, input_shape: int) -> nn.Module:
    model = PricePredictor(input_shape).to(device)
    model.load_state_dict(torch.load(f'{path}/{symbol}_model.pth'))
    model.eval()
    return model

def predict(model: nn.Module, x: np.ndarray, scaler: StandardScaler, future_days: int, features: List) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        predictions = model(x_tensor).cpu().numpy()

        predictions_reshaped = np.zeros((x.shape[0], len(features) + 1))
        predictions_reshaped[:, 0] = predictions[:, 0]
        predictions_reshaped = np.pad(predictions_reshaped, ((0, 0), (0, len(scaler.scale_) - len(predictions_reshaped[0]))), 'constant')
        predictions = scaler.inverse_transform(predictions_reshaped)[:, 0]
        
        future_predictions = []
        for _ in range(future_days):
            x_tensor = torch.tensor(x[-1:], dtype=torch.float32).to(device)
            future_pred = model(x_tensor).cpu().numpy()[0][0]
            future_predictions.append(future_pred)

            new_row = np.zeros((1, x.shape[2]))
            new_row[0, 0] = future_pred
            new_row[0, 1:] = x[-1, -1, 1:]

            x = np.append(x, [np.append(x[-1][1:], new_row, axis=0)], axis=0)

        future_predictions_reshaped = np.zeros((future_days, len(features) + 1))
        future_predictions_reshaped[:, 0] = future_predictions
        future_predictions_reshaped = np.pad(future_predictions_reshaped, ((0, 0), (0, len(scaler.scale_) - len(future_predictions_reshaped[0]))), 'constant')
        future_predictions = scaler.inverse_transform(future_predictions_reshaped)[:, 0]

    return predictions, future_predictions

def plot_predictions(symbol: str, filename: str, historical_data: np.ndarray, predictions: np.ndarray, future_predictions: np.ndarray, data: pd.DataFrame, freq: str, transactions: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, historical_data, label='Historical Prices')
    aligned_predictions = np.zeros_like(historical_data)
    aligned_predictions[-len(predictions):] = predictions
    plt.plot(data.index, aligned_predictions, label='Predicted Prices', color='red')
    future_dates = pd.date_range(data.index[-1], periods=len(future_predictions) + 1, freq=freq)[1:]
    plt.plot(future_dates, future_predictions, label='Predicted Future Prices', linestyle='dashed')

    # Plot buy and sell markers
    buys = transactions[transactions['type'] == 'buy']
    sells = transactions[transactions['type'] == 'sell']
    plt.scatter(buys['date'], buys['price'], marker='^', color='green', label='Buy', alpha=1)
    plt.scatter(sells['date'], sells['price'], marker='v', color='red', label='Sell', alpha=1)

    plt.title(f'{symbol} {data.index[0].strftime("%Y-%m-%d")} to {data.index[-1].strftime("%Y-%m-%d")}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(filename)

def backtest(ticker: str, symbol: str, asset_type: str, data_sampling_interval: str, target: str, start_date: str, model_dir: str, look_back: int, look_forward: int, best_features: List, indicator_windows: dict, data_resampling_frequency: str, initial_cash: float, trading_fee: float) -> None:
    logger.info(f"Getting data for {symbol} from {start_date}")
    historical_data, features = get_data(ticker, symbol, asset_type=asset_type, start=start_date, end=time.strftime('%Y-%m-%d'), windows=indicator_windows, data_sampling_interval=data_sampling_interval, data_resampling_frequency=data_resampling_frequency)
    logger.info(f"Preprocessing data")
    x, _, scaler, selected_features = preprocess_data(historical_data, target, look_back=look_back, look_forward=look_forward, features=features, best_features=best_features)
    logger.info(f"Loaded model from {model_dir}")
    model = load_model(symbol, model_dir, input_shape=len(selected_features))
    logger.info(f"Making predictions")
    predictions, future_predictions = predict(model, x, scaler, look_forward, selected_features)

    portfolio = FakePortfolio(initial_cash, trading_fee)
    portfolio_values = []
    daily_trades = []

    for i in range(len(historical_data)):
        current_prices = { symbol: historical_data['Close'].values[i] }
        portfolio_value = portfolio.portfolio_value(current_prices)
        portfolio_values.append(portfolio_value)
        daily_trade_count = 0

        if i < len(predictions) - 1:
            if predictions[i] > historical_data['Close'].values[i] * 1.01:
                portfolio.buy(symbol, 1, historical_data['Close'].values[i], historical_data.index[i])
                daily_trade_count += 1
            elif predictions[i] < historical_data['Close'].values[i] * 0.99:
                portfolio.sell(symbol, 1, historical_data['Close'].values[i], historical_data.index[i])
                daily_trade_count += 1

        daily_trades.append(daily_trade_count)

    current_prices = { symbol: historical_data['Close'].values[-1] }
    final_portfolio_value = portfolio.portfolio_value(current_prices)
    logger.info(f"Final Portfolio Value: {final_portfolio_value}")

    transaction_log = portfolio.transaction_log()
    transaction_log.to_csv(f'reports/{symbol}_transaction_log.csv', index=False)
    logger.info(f'Transaction log saved to reports/{symbol}_transaction_log.csv')

    portfolio_report = pd.DataFrame({
        'Date': historical_data.index,
        'Portfolio Value': portfolio_values,
        'Daily Trades': daily_trades
    })
    portfolio_report.to_csv(f'reports/{symbol}_portfolio_value.csv', index=False)
    logger.info(f'Portfolio value report saved to reports/{symbol}_portfolio_value.csv')

    performance_report = portfolio.performance_report(current_prices)
    performance_report.to_csv(f'reports/{symbol}_performance_report.csv', index=False)
    logger.info(f'Performance report saved to reports/{symbol}_performance_report.csv')

    plot_predictions(symbol, f'png/{symbol}_7_days.png', historical_data['Close'].values[-7:], predictions[-7:], future_predictions, historical_data[-7:], data_resampling_frequency, transaction_log)
    plot_predictions(symbol, f'png/{symbol}_30_days.png', historical_data['Close'].values[-30:], predictions[-30:], future_predictions, historical_data[-30:], data_resampling_frequency, transaction_log)
    plot_predictions(symbol, f'png/{symbol}_90_days.png', historical_data['Close'].values[-90:], predictions[-90:], future_predictions, historical_data[-90:], data_resampling_frequency, transaction_log)
    plot_predictions(symbol, f'png/{symbol}_365_days.png', historical_data['Close'].values[-365:], predictions[-365:], future_predictions, historical_data[-365:], data_resampling_frequency, transaction_log)
    plot_predictions(symbol, f'png/{symbol}_full.png', historical_data['Close'].values, predictions, future_predictions, historical_data, data_resampling_frequency, transaction_log)

    logger.info('Backtesting completed and plotted')
    
    future_dates = pd.date_range(historical_data.index[-1], periods=look_forward + 1, freq=data_resampling_frequency)[1:]
    report = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': future_predictions
    })
    report.to_csv(f'reports/{symbol}_predictions.csv', index=False)
    logger.info(f'Report saved to reports/{symbol}_predictions.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
    args = parser.parse_args()

    config = load_json(args.config)

    ticker = config['ticker']
    symbol = config['symbol']
    asset_type = config['asset_type']
    data_sampling_interval = config['data_sampling_interval']
    model_dir = config['model_dir']
    start_date = config['start_date']
    look_back = config['look_back']
    look_forward = config['look_forward']
    best_features = config.get('best_features', None)
    target = config['target']
    data_resampling_frequency = config['data_resampling_frequency']
    indicator_windows = config['indicator_windows']
    initial_cash = config.get('initial_cash', 100000)
    trading_fee = config.get('trading_fee', 0.001)

    logger.info(f"Starting backtest for {ticker}")
    backtest(ticker, symbol, asset_type, data_sampling_interval, target, start_date, model_dir, look_back, look_forward, best_features, indicator_windows, data_resampling_frequency, initial_cash, trading_fee)
    logger.info(f"Backtest for {symbol} completed")
