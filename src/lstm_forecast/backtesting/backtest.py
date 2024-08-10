import time
import pandas as pd
import torch
from lstm_forecast.data_loader import preprocess_data, get_data
from lstm_forecast.model import predict, load_model
from lstm_forecast.backtesting.portfolio import FakePortfolio
from lstm_forecast.backtesting.plot import plot_predictions_with_orders
from lstm_forecast.logger import setup_logger
from lstm_forecast.config import Config
from typing import List

# Set up logger
logger = setup_logger("backtest_logger", "logs/backtest.log")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")


def backtest(
    ticker: str,
    symbol: str,
    asset_type: str,
    data_sampling_interval: str,
    target: str,
    start_date: str,
    model_dir: str,
    model_params: dict,
    look_back: int,
    look_forward: int,
    selected_features: List,
    indicator_windows: dict,
    data_resampling_frequency: str,
    initial_cash: float,
    trading_fee: float,
    take_profit: float = None,
    stop_loss: float = None,
    trade_allocation: float = 0.1,
    max_open_trades: int = 5,
) -> None:
    logger.info(f"Getting data for {symbol} from {start_date}")
    historical_data, features = get_data(
        ticker,
        symbol,
        asset_type=asset_type,
        start=start_date,
        end=time.strftime("%Y-%m-%d"),
        windows=indicator_windows,
        data_sampling_interval=data_sampling_interval,
        data_resampling_frequency=data_resampling_frequency,
    )

    logger.info("Preprocessing data")
    x_data, _, scaler, selected_features = preprocess_data(
        historical_data,
        target,
        look_back=look_back,
        look_forward=look_forward,
        features=features,
        selected_features=selected_features,
    )

    logger.info(f"Loaded model from {model_dir}")
    model = load_model(
        symbol,
        model_dir,
        input_shape=len(selected_features),
        model_params=model_params,
    )

    logger.info("Making predictions")
    predictions, future_predictions = predict(
        model, x_data, scaler, look_forward, selected_features
    )

    portfolio = FakePortfolio(
        initial_cash,
        trading_fee,
        take_profit,
        stop_loss,
        trade_allocation,
        max_open_trades,
    )
    portfolio_values = []
    daily_trades = []

    for i in range(len(historical_data)):
        current_prices = {symbol: historical_data["Close"].values[i]}
        portfolio_value = portfolio.portfolio_value(current_prices)
        portfolio_values.append(portfolio_value)
        daily_trade_count = 0

        if i < len(predictions) - 1:
            trade_amount = (
                portfolio.portfolio_value(current_prices) * trade_allocation
            )
            trade_quantity = trade_amount / historical_data["Close"].values[i]

            if predictions[i] > historical_data["Close"].values[i] * 1.01:
                order_status = portfolio.buy(
                    symbol,
                    trade_quantity,
                    historical_data["Close"].values[i],
                    historical_data.index[i],
                )
                if order_status == "executed":
                    daily_trade_count += 1
            elif predictions[i] < historical_data["Close"].values[i] * 0.99:
                order_status = portfolio.sell(
                    symbol,
                    trade_quantity,
                    historical_data["Close"].values[i],
                    historical_data.index[i],
                )
                if order_status == "executed":
                    daily_trade_count += 1

            if portfolio.take_profit and portfolio.holdings.get(symbol, 0) > 0:
                avg_buy_price = sum(
                    t["price"] * t["quantity"]
                    for t in portfolio.transaction_history
                    if t["symbol"] == symbol and t["type"] == "buy"
                ) / sum(
                    t["quantity"]
                    for t in portfolio.transaction_history
                    if t["symbol"] == symbol and t["type"] == "buy"
                )
                if historical_data["Close"].values[i] >= avg_buy_price * (
                    1 + portfolio.take_profit
                ):
                    portfolio.sell(
                        symbol,
                        portfolio.holdings.get(symbol, 0),
                        historical_data["Close"].values[i],
                        historical_data.index[i],
                        reason="take_profit",
                    )
            if portfolio.stop_loss and portfolio.holdings.get(symbol, 0) > 0:
                avg_buy_price = sum(
                    t["price"] * t["quantity"]
                    for t in portfolio.transaction_history
                    if t["symbol"] == symbol and t["type"] == "buy"
                ) / sum(
                    t["quantity"]
                    for t in portfolio.transaction_history
                    if t["symbol"] == symbol and t["type"] == "buy"
                )
                if historical_data["Close"].values[i] <= avg_buy_price * (
                    1 - portfolio.stop_loss
                ):
                    portfolio.sell(
                        symbol,
                        portfolio.holdings.get(symbol, 0),
                        historical_data["Close"].values[i],
                        historical_data.index[i],
                        reason="stop_loss",
                    )

        daily_trades.append(daily_trade_count)

    current_prices = {symbol: historical_data["Close"].values[-1]}
    final_portfolio_value = portfolio.portfolio_value(current_prices)
    logger.info(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")

    transaction_log_df = portfolio.transaction_log()
    transaction_log_df.to_csv(
        f"reports/{symbol}_transaction_log.csv", index=False
    )
    logger.info(
        f"Transaction log saved to reports/{symbol}_transaction_log.csv"
    )

    portfolio_report_df = pd.DataFrame(
        {
            "Date": historical_data.index,
            "Portfolio Value": portfolio_values,
            "Daily Trades": daily_trades,
        }
    )
    portfolio_report_df.to_csv(
        f"reports/{symbol}_portfolio_value.csv", index=False
    )
    logger.info(
        f"Portfolio value report saved to reports/{symbol}_portfolio_value.csv"
    )

    performance_report_df = portfolio.performance_report(current_prices)
    performance_report_df.to_csv(
        f"reports/{symbol}_performance_report.csv", index=False
    )
    logger.info(
        f"Performance report saved to reports/{symbol}_performance_report.csv"
    )

    plot_predictions_with_orders(
        symbol,
        f"png/{symbol}_backtesting_7_days.png",
        historical_data["Close"].values[-7:],
        predictions[-7:],
        future_predictions,
        historical_data[-7:],
        data_resampling_frequency,
        transaction_log_df,
    )
    plot_predictions_with_orders(
        symbol,
        f"png/{symbol}_backtesting_30_days.png",
        historical_data["Close"].values[-30:],
        predictions[-30:],
        future_predictions,
        historical_data[-30:],
        data_resampling_frequency,
        transaction_log_df,
    )
    plot_predictions_with_orders(
        symbol,
        f"png/{symbol}_backtesting_90_days.png",
        historical_data["Close"].values[-90:],
        predictions[-90:],
        future_predictions,
        historical_data[-90:],
        data_resampling_frequency,
        transaction_log_df,
    )
    plot_predictions_with_orders(
        symbol,
        f"png/{symbol}_backtesting_365_days.png",
        historical_data["Close"].values[-365:],
        predictions[-365:],
        future_predictions,
        historical_data[-365:],
        data_resampling_frequency,
        transaction_log_df,
    )
    plot_predictions_with_orders(
        symbol,
        f"png/{symbol}_backtesting_full.png",
        historical_data["Close"].values,
        predictions,
        future_predictions,
        historical_data,
        data_resampling_frequency,
        transaction_log_df,
    )

    logger.info("Backtesting completed and plotted")

    future_dates = pd.date_range(
        historical_data.index[-1],
        periods=look_forward + 1,
        freq=data_resampling_frequency,
    )[1:]
    future_report_df = pd.DataFrame(
        {"Date": future_dates, "Predicted Price": future_predictions}
    )
    future_report_df.to_csv(
        f"reports/{symbol}_backtesting_predictions.csv", index=False
    )
    logger.info(
        f"Report saved to reports/{symbol}_backtesting_predictions.csv"
    )

    metrics = portfolio.calculate_metrics(
        historical_data, current_prices, symbol
    )
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")


def main(config: Config) -> None:
    ticker = config["ticker"]
    symbol = config["symbol"]
    asset_type = config["asset_type"]
    data_sampling_interval = config["data_sampling_interval"]
    model_dir = config["model_dir"]
    model_params = config.get("model_params", {})
    start_date = config["start_date"]
    look_back = config.look_back
    look_forward = config.look_forward
    selected_features = config.selected_features
    target = config.get("targets", ["Close"])
    data_resampling_frequency = config["data_resampling_frequency"]
    indicator_windows = config["indicator_windows"]
    backtesting_params = config.get("backtesting_params", {})

    logger.info(f"Starting backtest for {ticker}")
    backtest(
        ticker=ticker,
        symbol=symbol,
        asset_type=asset_type,
        data_sampling_interval=data_sampling_interval,
        target=target,
        start_date=start_date,
        model_dir=model_dir,
        model_params=model_params,
        look_back=look_back,
        look_forward=look_forward,
        selected_features=selected_features,
        indicator_windows=indicator_windows,
        data_resampling_frequency=data_resampling_frequency,
        initial_cash=backtesting_params["initial_cash"],
        trading_fee=backtesting_params["trading_fee"],
        take_profit=backtesting_params.get("take_profit"),
        stop_loss=backtesting_params.get("stop_loss"),
        trade_allocation=backtesting_params["trade_allocation"],
        max_open_trades=backtesting_params["max_open_trades"],
    )

    logger.info(f"Backtest for {symbol} completed")
