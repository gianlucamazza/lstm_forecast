import pandas as pd
import numpy as np
from typing import Dict
from src.logger import setup_logger
from metrics import calculate_sharpe_ratio, calculate_sortino_ratio

logger = setup_logger("portfolio_logger", "logs/portfolio.log")


class FakePortfolio:
    def __init__(
        self,
        initial_cash: float,
        trading_fee: float = 0.001,
        take_profit: float = None,
        stop_loss: float = None,
        trade_allocation: float = 0.1,
        max_open_trades: int = 5,
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.holdings = {}  # {symbol: quantity}
        self.transaction_history = []
        self.trading_fee = trading_fee
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.trade_allocation = trade_allocation
        self.max_open_trades = max_open_trades

    def buy(
        self, symbol: str, quantity: float, price: float, date: pd.Timestamp
    ) -> str:
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
        total_value = self.cash
        for symbol, quantity in self.holdings.items():
            total_value += quantity * current_prices.get(symbol, 0)
        return total_value

    def transaction_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.transaction_history)

    def performance_report(
        self, current_prices: Dict[str, float]
    ) -> pd.DataFrame:
        total_value = self.portfolio_value(current_prices)
        profit = total_value - self.initial_cash
        roi = (profit / self.initial_cash) * 100
        return pd.DataFrame(
            [
                {
                    "initial_cash": self.initial_cash,
                    "final_cash": self.cash,
                    "total_value": total_value,
                    "profit": profit,
                    "ROI (%)": roi,
                }
            ]
        )

    def calculate_drawdown(
        self, historical_data: pd.DataFrame, symbol: str
    ) -> float:
        portfolio_values = [self.initial_cash]
        for i in range(len(historical_data)):
            current_prices = {symbol: historical_data["Close"].values[i]}
            portfolio_values.append(self.portfolio_value(current_prices))

        portfolio_values = np.array(portfolio_values)
        peaks = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - peaks) / peaks
        max_drawdown = np.min(drawdowns) * 100  # as percentage
        return max_drawdown

    def calculate_calmar_ratio(
        self, historical_data: pd.DataFrame, symbol: str
    ) -> float:
        cagr = (
            historical_data["Close"].iloc[-1]
            / historical_data["Close"].iloc[0]
        ) ** (1 / (len(historical_data) / 252)) - 1
        max_drawdown = self.calculate_drawdown(historical_data, symbol)
        calmar_ratio = (
            cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan
        )
        return calmar_ratio

    def calculate_metrics(
        self,
        historical_data: pd.DataFrame,
        current_prices: Dict[str, float],
        symbol: str,
    ) -> Dict[str, float]:
        total_trades = (
            len(self.transaction_history) // 2
        )  # Each buy and sell pair counts as one trade
        winning_trades = 0
        losing_trades = 0

        for sell in (
            t
            for t in self.transaction_history
            if t["type"] == "sell" and t["status"] == "executed"
        ):
            corresponding_buys = [
                t
                for t in self.transaction_history
                if t["type"] == "buy"
                and t["symbol"] == sell["symbol"]
                and t["date"] <= sell["date"]
            ]
            if corresponding_buys:
                total_cost = sum(
                    buy["total_cost"]
                    for buy in corresponding_buys
                    if "total_cost" in buy
                )
                total_revenue = (
                    sell["total_revenue"] if "total_revenue" in sell else 0
                )

                if total_revenue > total_cost:
                    winning_trades += 1
                else:
                    losing_trades += 1

        win_rate = (
            (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        )
        drawdown = self.calculate_drawdown(historical_data, symbol)
        sharpe_ratio = calculate_sharpe_ratio(historical_data)
        sortino_ratio = calculate_sortino_ratio(historical_data)
        calmar_ratio = self.calculate_calmar_ratio(historical_data, symbol)
        performance = self.performance_report(current_prices).iloc[0]
        return {
            "Total Trades": total_trades,
            "Winning Trades": winning_trades,
            "Losing Trades": losing_trades,
            "Win Rate (%)": win_rate,
            "Max Drawdown (%)": drawdown,
            "Sharpe Ratio": sharpe_ratio,
            "Sortino Ratio": sortino_ratio,
            "Calmar Ratio": calmar_ratio,
            "Final Portfolio Value ($)": performance["total_value"],
            "Total Profit ($)": performance["profit"],
            "ROI (%)": performance["ROI (%)"],
        }
