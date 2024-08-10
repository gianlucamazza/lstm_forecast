import pandas as pd
import numpy as np
from typing import Dict
from lstm_forecast.logger import setup_logger
from metrics import calculate_sharpe_ratio, calculate_sortino_ratio

logger = setup_logger("trading_engine_logger", "logs/trading_engine.log")


class TradingEngine:
    def __init__(
        self,
        symbol: str,
        initial_cash: float,
        trading_fee: float = 0.001,
        take_profit: float = None,
        stop_loss: float = None,
        trade_allocation: float = 0.1,
        max_open_trades: int = 5,
    ):
        self.symbol = symbol
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.holdings = {}
        self.transaction_history = []
        self.trading_fee = trading_fee
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.trade_allocation = trade_allocation
        self.max_open_trades = max_open_trades

    def create_candlestick_data(
        self, historical_data: pd.DataFrame, look_back: int, look_forward: int
    ) -> pd.DataFrame:
        """
        Create candlestick data from historical data.

        Args:
            historical_data (pd.DataFrame): The historical data.
            look_back (int): The number of previous days to consider.
            look_forward (int): The number of future days to predict.

        Returns:
            pd.DataFrame: The candlestick data.
        """
        candlestick_data = historical_data[
            ["Open", "High", "Low", "Close"]
        ].copy()
        for i in range(1, look_back + 1):
            for col in ["Open", "High", "Low", "Close"]:
                candlestick_data[f"{col} - {i}d"] = candlestick_data[
                    col
                ].shift(i)
        for i in range(1, look_forward + 1):
            for col in ["Open", "High", "Low", "Close"]:
                candlestick_data[f"{col} + {i}d"] = candlestick_data[
                    col
                ].shift(-i)
        return candlestick_data

    def buy(
        self, symbol: str, quantity: float, price: float, date: pd.Timestamp
    ) -> str:
        """
        Buy a stock.

        Args:
            symbol (str): The stock symbol.
            quantity (float): The quantity to buy.
            price (float): The price of the stock.
            date (pd.Timestamp): The date of the transaction.

        Returns:
            str: The status of the transaction.
        """
        open_trades = sum(1 for h in self.holdings.values() if h > 0)
        if open_trades >= self.max_open_trades:
            logger.warning("Reached max open trades limit")
            self._log_transaction(
                "buy",
                symbol,
                quantity,
                price,
                date,
                "failed",
                "max_open_trades",
            )
            return "failed"

        cost = quantity * price
        fee = cost * self.trading_fee
        total_cost = cost + fee
        if self.cash >= total_cost:
            self.cash -= total_cost
            self.holdings[symbol] = self.holdings.get(symbol, 0) + quantity
            self._log_transaction(
                "buy",
                symbol,
                quantity,
                price,
                date,
                "executed",
                total_cost=total_cost,
                fee=fee,
            )
            return "executed"
        else:
            logger.warning("Not enough cash to buy")
            self._log_transaction(
                "buy", symbol, quantity, price, date, "failed"
            )
            return "failed"

    def sell(
        self,
        symbol: str,
        quantity: float,
        price: float,
        date: pd.Timestamp,
        reason: str = "normal",
    ) -> str:
        """
        Sell a stock.

        Args:
            symbol (str): The stock symbol.
            quantity (float): The quantity to sell.
            price (float): The price of the stock.
            date (pd.Timestamp): The date of the transaction.
            reason (str): The reason for selling the stock.

        Returns:
            str: The status of the transaction.
        """
        if self.holdings.get(symbol, 0) >= quantity:
            revenue = quantity * price
            fee = revenue * self.trading_fee
            total_revenue = revenue - fee
            self.holdings[symbol] -= quantity
            self.cash += total_revenue
            self._log_transaction(
                "sell",
                symbol,
                quantity,
                price,
                date,
                "executed",
                reason=reason,
                total_revenue=total_revenue,
                fee=fee,
            )
            return "executed"
        else:
            logger.warning("Not enough holdings to sell")
            self._log_transaction(
                "sell", symbol, quantity, price, date, "failed", reason
            )
            return "failed"

    def _log_transaction(
        self,
        type_: str,
        symbol: str,
        quantity: float,
        price: float,
        date: pd.Timestamp,
        status: str,
        reason: str = None,
        total_cost: float = None,
        total_revenue: float = None,
        fee: float = None,
    ) -> None:
        """
        Log a transaction.

        Args:
            type_ (str): The type of transaction (buy/sell).
            symbol (str): The stock symbol.
            quantity (float): The quantity of the transaction.
            price (float): The price of the stock.
            date (pd.Timestamp): The date of the transaction.
            status (str): The status of the transaction.
            reason (str): The reason for the transaction.
            total_cost (float): The total cost of the transaction.
            total_revenue (float): The total revenue of the transaction.
            fee (float): The transaction fee.
        """
        transaction = {
            "type": type_,
            "symbol": symbol,
            "quantity": quantity,
            "price": float(price),  # Store price as float
            "cash_remaining": self.cash,
            "date": date,
            "status": status,
        }
        if total_cost is not None:
            transaction["total_cost"] = total_cost
        if total_revenue is not None:
            transaction["total_revenue"] = total_revenue
        if fee is not None:
            transaction["fee"] = fee
        if reason:
            transaction["reason"] = reason
        self.transaction_history.append(transaction)

    def portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate portfolio value.

        Args:
            current_prices (Dict[str, float]): Current stock prices.

        Returns:
            float: Portfolio value.
        """
        total_value = self.cash
        for symbol, quantity in self.holdings.items():
            total_value += quantity * current_prices.get(symbol, 0)
        return total_value

    def transaction_log(self) -> pd.DataFrame:
        """
        Get the transaction log.

        Returns:
            pd.DataFrame: The transaction log.
        """
        return pd.DataFrame(self.transaction_history)

    def performance_report(
        self, current_prices: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Generate a performance report.

        Args:
            current_prices (Dict[str, float]): Current stock prices.

        Returns:
            pd.DataFrame: The performance report.
        """
        transaction_log_df = self.transaction_log()
        portfolio_value = self.portfolio_value(current_prices)
        sharpe_ratio = calculate_sharpe_ratio(
            transaction_log_df, portfolio_value
        )
        sortino_ratio = calculate_sortino_ratio(
            transaction_log_df, portfolio_value
        )
        performance_report = {
            "Portfolio Value": portfolio_value,
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
        }
        return pd.DataFrame([performance_report])

    def backtest(
        self,
        historical_data: pd.DataFrame,
        predictions: np.ndarray,
        future_predictions: np.ndarray,
    ) -> pd.DataFrame:
        """
        Backtest the trading strategy.

        Args:
            historical_data (pd.DataFrame): The historical data.
            predictions (np.ndarray): The predictions.
            future_predictions (np.ndarray): The future predictions.

        Returns:
            pd.DataFrame: The performance report.
        """
        portfolio_values = []
        daily_trades = []

        for i in range(len(historical_data)):
            current_prices = {self.symbol: historical_data["Close"].values[i]}
            portfolio_value = self.portfolio_value(current_prices)
            portfolio_values.append(portfolio_value)
            daily_trade_count = 0

            if i < len(predictions) - 1:
                trade_amount = (
                    self.portfolio_value(current_prices)
                    * self.trade_allocation
                )
                trade_quantity = (
                    trade_amount / historical_data["Close"].values[i]
                )

                if predictions[i] > historical_data["Close"].values[i] * 1.01:
                    order_status = self.buy(
                        self.symbol,
                        trade_quantity,
                        historical_data["Close"].values[i],
                        historical_data.index[i],
                    )
                    if order_status == "executed":
                        daily_trade_count += 1
                elif (
                    predictions[i] < historical_data["Close"].values[i] * 0.99
                ):
                    order_status = self.sell(
                        self.symbol,
                        trade_quantity,
                        historical_data["Close"].values[i],
                        historical_data.index[i],
                    )
                    if order_status == "executed":
                        daily_trade_count += 1

                if self.take_profit and self.holdings.get(self.symbol, 0) > 0:
                    avg_buy_price = sum(
                        t["price"] * t["quantity"]
                        for t in self.transaction_history
                        if t["symbol"] == self.symbol and t["type"] == "buy"
                    ) / sum(
                        t["quantity"]
                        for t in self.transaction_history
                        if t["symbol"] == self.symbol and t["type"] == "buy"
                    )
                    if historical_data["Close"].values[i] >= avg_buy_price * (
                        1 + self.take_profit
                    ):
                        self.sell(
                            self.symbol,
                            self.holdings.get(self.symbol, 0),
                            historical_data["Close"].values[i],
                            historical_data.index[i],
                            reason="take_profit",
                        )
                if self.stop_loss and self.holdings.get(self.symbol, 0) > 0:
                    avg_buy_price = sum(
                        t["price"] * t["quantity"]
                        for t in self.transaction_history
                        if t["symbol"] == self.symbol and t["type"] == "buy"
                    ) / sum(
                        t["quantity"]
                        for t in self.transaction_history
                        if t["symbol"] == self.symbol and t["type"] == "buy"
                    )
                    if historical_data["Close"].values[i] <= avg_buy_price * (
                        1 - self.stop_loss
                    ):
                        self.sell(
                            self.symbol,
                            self.holdings.get(self.symbol, 0),
                            historical_data["Close"].values[i],
                            historical_data.index[i],
                            reason="stop_loss",
                        )

            daily_trades.append(daily_trade_count)

        current_prices = {self.symbol: historical_data["Close"].values[-1]}
        final_portfolio_value = self.portfolio_value(current_prices)
        logger.info(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")

        transaction_log_df = self.transaction_log()
        transaction_log_df.to_csv(
            f"reports/{self.symbol}_transaction_log.csv", index=False
        )
        logger.info(
            f"Transaction log saved to reports/"
            f"{self.symbol}_transaction_log.csv"
        )

        portfolio_report_df = pd.DataFrame(
            {
                "Date": historical_data.index,
                "Portfolio Value": portfolio_values,
                "Daily Trades": daily_trades,
            }
        )

        return portfolio_report_df

    def calculate_drawdown(self, historical_data: pd.DataFrame) -> float:
        """
        Calculate the maximum drawdown of the portfolio.

        Args:
            historical_data (pd.DataFrame): The historical data.

        Returns:
            float: The maximum drawdown.
        """
        portfolio_values = [self.initial_cash]
        for i in range(len(historical_data)):
            current_prices = {self.symbol: historical_data["Close"].values[i]}
            portfolio_values.append(self.portfolio_value(current_prices))
        portfolio_values = np.array(portfolio_values)
        drawdown = (
            portfolio_values - np.maximum.accumulate(portfolio_values)
        ) / np.maximum.accumulate(portfolio_values)
        max_drawdown = np.min(drawdown)
        return max_drawdown

    def calculate_return(self, historical_data: pd.DataFrame) -> float:
        """
        Calculate the return on investment (ROI) of the portfolio.

        Args:
            historical_data (pd.DataFrame): The historical data.

        Returns:
            float: The return on investment.
        """
        current_prices = {self.symbol: historical_data["Close"].values[-1]}
        final_portfolio_value = self.portfolio_value(current_prices)
        profit = final_portfolio_value - self.initial_cash
        roi = (profit / self.initial_cash) * 100
        return roi

    def calculate_sharpe_ratio(self, historical_data: pd.DataFrame) -> float:
        """
        Calculate the Sharpe ratio of the portfolio.

        Args:
            historical_data (pd.DataFrame): The historical data.

        Returns:
            float: The Sharpe ratio.
        """
        portfolio_values = [self.initial_cash]
        for i in range(len(historical_data)):
            current_prices = {self.symbol: historical_data["Close"].values[i]}
            portfolio_values.append(self.portfolio_value(current_prices))
        portfolio_values = np.array(portfolio_values)
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = calculate_sharpe_ratio(daily_returns)
        return sharpe_ratio

    def calculate_sortino_ratio(self, historical_data: pd.DataFrame) -> float:
        """
        Calculate the Sortino ratio of the portfolio.

        Args:
            historical_data (pd.DataFrame): The historical data.

        Returns:
            float: The Sortino ratio.
        """
        portfolio_values = [self.initial_cash]
        for i in range(len(historical_data)):
            current_prices = {self.symbol: historical_data["Close"].values[i]}
            portfolio_values.append(self.portfolio_value(current_prices))
        portfolio_values = np.array(portfolio_values)
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sortino_ratio = calculate_sortino_ratio(daily_returns)
        return sortino_ratio

    def calculate_calmar_ratio(self, historical_data: pd.DataFrame) -> float:
        """
        Calculate the Calmar ratio of the portfolio.

        Args:
            historical_data (pd.DataFrame): The historical data.

        Returns:
            float: The Calmar ratio.
        """
        cagr = self.calculate_return(historical_data) / 100
        max_drawdown = self.calculate_drawdown(historical_data)
        calmar_ratio = cagr / abs(max_drawdown)
        return calmar_ratio

    def calculate_mdd(self, historical_data: pd.DataFrame) -> float:
        """
        Calculate the maximum drawdown of the portfolio.

        Args:
            historical_data (pd.DataFrame): The historical data.

        Returns:
            float: The maximum drawdown.
        """
        portfolio_values = [self.initial_cash]
        for i in range(len(historical_data)):
            current_prices = {self.symbol: historical_data["Close"].values[i]}
            portfolio_values.append(self.portfolio_value(current_prices))
        portfolio_values = np.array(portfolio_values)
        peaks = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - peaks) / peaks
        max_drawdown = np.min(drawdowns) * 100

        return max_drawdown
