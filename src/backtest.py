import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
from typing import List, Tuple, Dict
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


def calculate_sortino_ratio(historical_data: pd.DataFrame) -> float:
    returns = historical_data['Close'].pct_change().dropna()
    mean_return = returns.mean()
    negative_returns = returns[returns < 0]
    std_dev_neg = negative_returns.std()
    sortino_ratio = (mean_return / std_dev_neg) * np.sqrt(252)  # Annualized Sortino Ratio
    return sortino_ratio


def calculate_sharpe_ratio(historical_data: pd.DataFrame) -> float:
    returns = historical_data['Close'].pct_change().dropna()
    mean_return = returns.mean()
    std_dev = returns.std()
    sharpe_ratio = (mean_return / std_dev) * np.sqrt(252)  # Annualized Sharpe Ratio
    return sharpe_ratio


class FakePortfolio:
    def __init__(self, initial_cash: float, trading_fee: float = 0.001, take_profit: float = None,
                 stop_loss: float = None, trade_allocation: float = 0.1, max_open_trades: int = 5):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.holdings = {}  # {symbol: quantity}
        self.transaction_history = []
        self.trading_fee = trading_fee
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.trade_allocation = trade_allocation
        self.max_open_trades = max_open_trades

    def buy(self, symbol: str, quantity: float, price: float, date: pd.Timestamp) -> str:
        open_trades = sum(1 for h in self.holdings.values() if h > 0)
        if open_trades >= self.max_open_trades:
            logger.warning("Reached max open trades limit")
            self._log_transaction('buy', symbol, quantity, price, date, 'failed', 'max_open_trades')
            return 'failed'

        cost = quantity * price
        fee = cost * self.trading_fee
        total_cost = cost + fee
        if self.cash >= total_cost:
            self.cash -= total_cost
            self.holdings[symbol] = self.holdings.get(symbol, 0) + quantity
            self._log_transaction('buy', symbol, quantity, price, date, 'executed')
            return 'executed'
        else:
            logger.warning("Not enough cash to buy")
            self._log_transaction('buy', symbol, quantity, price, date, 'failed')
            return 'failed'

    def sell(self, symbol: str, quantity: float, price: float, date: pd.Timestamp, reason: str = 'normal') -> str:
        if self.holdings.get(symbol, 0) >= quantity:
            revenue = quantity * price
            fee = revenue * self.trading_fee
            total_revenue = revenue - fee
            self.holdings[symbol] -= quantity
            self.cash += total_revenue
            self._log_transaction('sell', symbol, quantity, price, date, 'executed', reason)
            return 'executed'
        else:
            logger.warning("Not enough holdings to sell")
            self._log_transaction('sell', symbol, quantity, price, date, 'failed', reason)
            return 'failed'

    def _log_transaction(self, type_: str, symbol: str, quantity: float, price: float, date: pd.Timestamp, status: str,
                         reason: str = None) -> None:
        transaction = {
            'type': type_,
            'symbol': symbol,
            'quantity': quantity,
            'price': f"${price:,.2f}",
            'total_cost': f"${quantity * price:,.2f}",
            'fee': f"${(quantity * price * self.trading_fee):,.2f}",
            'cash_remaining': f"${self.cash:,.2f}",
            'date': date,
            'status': status
        }
        if reason:
            transaction['reason'] = reason
        self.transaction_history.append(transaction)

    def portfolio_value(self, current_prices: Dict[str, float]) -> float:
        total_value = self.cash
        for symbol, quantity in self.holdings.items():
            total_value += quantity * current_prices.get(symbol, 0)
        return total_value

    def transaction_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.transaction_history)

    def performance_report(self, current_prices: Dict[str, float]) -> pd.DataFrame:
        total_value = self.portfolio_value(current_prices)
        profit = total_value - self.initial_cash
        roi = (profit / self.initial_cash) * 100
        return pd.DataFrame([{
            'initial_cash': f"${self.initial_cash:,.2f}",
            'final_cash': f"${self.cash:,.2f}",
            'total_value': f"${total_value:,.2f}",
            'profit': f"${profit:,.2f}",
            'ROI (%)': f"{roi:.2f}%"
        }])

    def calculate_metrics(self, historical_data: pd.DataFrame, current_prices: Dict[str, float]) -> Dict[str, float]:
        total_trades = len(self.transaction_history) // 2  # Each buy and sell pair counts as one trade
        winning_trades = 0
        losing_trades = 0

        for sell in (t for t in self.transaction_history if t['type'] == 'sell' and t['status'] == 'executed'):
            corresponding_buys = [t for t in self.transaction_history
                                  if t['type'] == 'buy' and t['symbol'] == sell['symbol'] and t['date'] <= sell['date']]
            if corresponding_buys:
                total_cost = sum(
                    float(buy['total_cost'].replace('$', '').replace(',', '')) for buy in corresponding_buys)
                total_revenue = float(sell['total_revenue'].replace('$', '').replace(',', ''))

                if total_revenue > total_cost:
                    winning_trades += 1
                else:
                    losing_trades += 1

        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        drawdown = self.calculate_drawdown(historical_data)
        sharpe_ratio = calculate_sharpe_ratio(historical_data)
        sortino_ratio = calculate_sortino_ratio(historical_data)
        calmar_ratio = self.calculate_calmar_ratio(historical_data)
        performance = self.performance_report(current_prices).iloc[0]
        return {
            'Total Trades': total_trades,
            'Winning Trades': winning_trades,
            'Losing Trades': losing_trades,
            'Win Rate (%)': win_rate,
            'Max Drawdown (%)': drawdown,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Final Portfolio Value ($)': performance['total_value'],
            'Total Profit ($)': performance['profit'],
            'ROI (%)': performance['ROI (%)']
        }

    def calculate_calmar_ratio(self, historical_data: pd.DataFrame) -> float:
        cagr = (historical_data['Close'].iloc[-1] / historical_data['Close'].iloc[0]) ** (
                1 / (len(historical_data) / 252)) - 1
        max_drawdown = self.calculate_drawdown(historical_data)
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan
        return calmar_ratio

    def calculate_drawdown(self, historical_data: pd.DataFrame) -> float:
        portfolio_values = [self.initial_cash]
        for i in range(len(historical_data)):
            current_prices = {symbol: historical_data['Close'].values[i]}
            portfolio_values.append(self.portfolio_value(current_prices))

        portfolio_values = np.array(portfolio_values)
        peaks = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - peaks) / peaks
        max_drawdown = np.min(drawdowns) * 100  # as percentage
        return max_drawdown


def load_model(symbol: str, path: str, input_shape: int) -> nn.Module:
    model = PricePredictor(input_shape).to(device)
    model.load_state_dict(torch.load(f'{path}/{symbol}_model.pth'))
    model.eval()
    return model


def predict(model: nn.Module, x_data: np.ndarray, scaler: StandardScaler, future_days: int, features: List) -> Tuple[
    np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_data, dtype=torch.float32).to(device)
        predictions = model(x_tensor).cpu().numpy()

        predictions_reshaped = np.zeros((x_data.shape[0], len(features) + 1))
        predictions_reshaped[:, 0] = predictions[:, 0]
        predictions_reshaped = np.pad(predictions_reshaped,
                                      ((0, 0), (0, len(scaler.scale_) - len(predictions_reshaped[0]))), 'constant')
        predictions = scaler.inverse_transform(predictions_reshaped)[:, 0]

        future_predictions = []
        for _ in range(future_days):
            x_tensor = torch.tensor(x_data[-1:], dtype=torch.float32).to(device)
            future_pred = model(x_tensor).cpu().numpy()[0][0]
            future_predictions.append(future_pred)

            new_row = np.zeros((1, x_data.shape[2]))
            new_row[0, 0] = future_pred
            new_row[0, 1:] = x_data[-1, -1, 1:]

            x_data = np.append(x_data, [np.append(x_data[-1][1:], new_row, axis=0)], axis=0)

        future_predictions_reshaped = np.zeros((future_days, len(features) + 1))
        future_predictions_reshaped[:, 0] = future_predictions
        future_predictions_reshaped = np.pad(future_predictions_reshaped,
                                             ((0, 0), (0, len(scaler.scale_) - len(future_predictions_reshaped[0]))),
                                             'constant')
        future_predictions = scaler.inverse_transform(future_predictions_reshaped)[:, 0]

    return predictions, future_predictions


def plot_predictions(symbol: str, filename: str, historical_data: np.ndarray, predictions: np.ndarray,
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


def backtest(ticker: str, symbol: str, asset_type: str, data_sampling_interval: str, target: str, start_date: str,
             model_dir: str, look_back: int, look_forward: int, best_features: List, indicator_windows: dict,
             data_resampling_frequency: str, initial_cash: float, trading_fee: float, take_profit: float = None,
             stop_loss: float = None, trade_allocation: float = 0.1, max_open_trades: int = 5) -> None:
    logger.info(f"Getting data for {symbol} from {start_date}")
    historical_data, features = get_data(ticker, symbol, asset_type=asset_type, start=start_date,
                                         end=time.strftime('%Y-%m-%d'), windows=indicator_windows,
                                         data_sampling_interval=data_sampling_interval,
                                         data_resampling_frequency=data_resampling_frequency)
    logger.info(f"Preprocessing data")
    x_data, _, scaler, selected_features = preprocess_data(historical_data, target, look_back=look_back,
                                                           look_forward=look_forward, features=features,
                                                           best_features=best_features)
    logger.info(f"Loaded model from {model_dir}")
    model = load_model(symbol, model_dir, input_shape=len(selected_features))
    logger.info(f"Making predictions")
    predictions, future_predictions = predict(model, x_data, scaler, look_forward, selected_features)

    portfolio = FakePortfolio(initial_cash, trading_fee, take_profit, stop_loss, trade_allocation, max_open_trades)
    portfolio_values = []
    daily_trades = []

    for i in range(len(historical_data)):
        current_prices = {symbol: historical_data['Close'].values[i]}
        portfolio_value = portfolio.portfolio_value(current_prices)
        portfolio_values.append(portfolio_value)
        daily_trade_count = 0

        if i < len(predictions) - 1:
            trade_amount = portfolio.portfolio_value(current_prices) * trade_allocation
            trade_quantity = trade_amount / historical_data['Close'].values[i]

            if predictions[i] > historical_data['Close'].values[i] * 1.01:
                order_status = portfolio.buy(symbol, trade_quantity, historical_data['Close'].values[i],
                                             historical_data.index[i])
                if order_status == 'executed':
                    daily_trade_count += 1
            elif predictions[i] < historical_data['Close'].values[i] * 0.99:
                order_status = portfolio.sell(symbol, trade_quantity, historical_data['Close'].values[i],
                                              historical_data.index[i])
                if order_status == 'executed':
                    daily_trade_count += 1

            if portfolio.take_profit and portfolio.holdings.get(symbol, 0) > 0:
                if historical_data['Close'].values[i] >= portfolio.holdings.get(symbol, 0) * portfolio.take_profit:
                    portfolio.sell(symbol, portfolio.holdings.get(symbol, 0), historical_data['Close'].values[i],
                                   historical_data.index[i], reason='take_profit')
            if portfolio.stop_loss and portfolio.holdings.get(symbol, 0) > 0:
                if historical_data['Close'].values[i] <= portfolio.holdings.get(symbol, 0) * portfolio.stop_loss:
                    portfolio.sell(symbol, portfolio.holdings.get(symbol, 0), historical_data['Close'].values[i],
                                   historical_data.index[i], reason='stop_loss')

        daily_trades.append(daily_trade_count)

    current_prices = {symbol: historical_data['Close'].values[-1]}
    final_portfolio_value = portfolio.portfolio_value(current_prices)
    logger.info(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")

    transaction_log_df = portfolio.transaction_log()
    transaction_log_df.to_csv(f'reports/{symbol}_transaction_log.csv', index=False)
    logger.info(f'Transaction log saved to reports/{symbol}_transaction_log.csv')

    portfolio_report_df = pd.DataFrame({
        'Date': historical_data.index,
        'Portfolio Value': [f"${val:,.2f}" for val in portfolio_values],
        'Daily Trades': daily_trades
    })
    portfolio_report_df.to_csv(f'reports/{symbol}_portfolio_value.csv', index=False)
    logger.info(f'Portfolio value report saved to reports/{symbol}_portfolio_value.csv')

    performance_report_df = portfolio.performance_report(current_prices)
    performance_report_df.to_csv(f'reports/{symbol}_performance_report.csv', index=False)
    logger.info(f'Performance report saved to reports/{symbol}_performance_report.csv')

    plot_predictions(symbol, f'png/{symbol}_7_days.png', historical_data['Close'].values[-7:], predictions[-7:],
                     future_predictions, historical_data[-7:], data_resampling_frequency, transaction_log_df)
    plot_predictions(symbol, f'png/{symbol}_30_days.png', historical_data['Close'].values[-30:], predictions[-30:],
                     future_predictions, historical_data[-30:], data_resampling_frequency, transaction_log_df)
    plot_predictions(symbol, f'png/{symbol}_90_days.png', historical_data['Close'].values[-90:], predictions[-90:],
                     future_predictions, historical_data[-90:], data_resampling_frequency, transaction_log_df)
    plot_predictions(symbol, f'png/{symbol}_365_days.png', historical_data['Close'].values[-365:], predictions[-365:],
                     future_predictions, historical_data[-365:], data_resampling_frequency, transaction_log_df)
    plot_predictions(symbol, f'png/{symbol}_full.png', historical_data['Close'].values, predictions, future_predictions,
                     historical_data, data_resampling_frequency, transaction_log_df)

    logger.info('Backtesting completed and plotted')

    future_dates = pd.date_range(historical_data.index[-1], periods=look_forward + 1, freq=data_resampling_frequency)[
                   1:]
    future_report_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Price': future_predictions
    })
    future_report_df.to_csv(f'reports/{symbol}_predictions.csv', index=False)
    logger.info(f'Report saved to reports/{symbol}_predictions.csv')

    metrics = portfolio.calculate_metrics(historical_data, current_prices)
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")


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
    best_features = config.get('best_features', [])
    target = config['target']
    data_resampling_frequency = config['data_resampling_frequency']
    indicator_windows = config['indicator_windows']
    initial_cash = config.get('initial_cash', 100000)
    trading_fee = config.get('trading_fee', 0.001)
    take_profit = config.get('take_profit', 1.6)
    stop_loss = config.get('stop_loss', 1.2)
    trade_allocation = config.get('trade_allocation', 0.1)
    max_open_trades = config.get('max_open_trades', 1)

    logger.info(f"Starting backtest for {ticker}")
    backtest(ticker=ticker, symbol=symbol, asset_type=asset_type, data_sampling_interval=data_sampling_interval,
             target=target, start_date=start_date, model_dir=model_dir, look_back=look_back, look_forward=look_forward,
             best_features=best_features, indicator_windows=indicator_windows,
             data_resampling_frequency=data_resampling_frequency, initial_cash=initial_cash,
             trading_fee=trading_fee, take_profit=take_profit, stop_loss=stop_loss, trade_allocation=trade_allocation,
             max_open_trades=max_open_trades)

    logger.info(f"Backtest for {symbol} completed")
