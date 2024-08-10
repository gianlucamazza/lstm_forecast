import pandas as pd
import numpy as np


def calculate_sortino_ratio(historical_data: pd.DataFrame) -> float:
    returns = historical_data["Close"].pct_change().dropna()
    mean_return = returns.mean()
    negative_returns = returns[returns < 0]
    std_dev_neg = negative_returns.std()
    sortino_ratio = (mean_return / std_dev_neg) * np.sqrt(
        252
    )  # Annualized Sortino Ratio
    return sortino_ratio


def calculate_sharpe_ratio(historical_data: pd.DataFrame) -> float:
    returns = historical_data["Close"].pct_change().dropna()
    mean_return = returns.mean()
    std_dev = returns.std()
    sharpe_ratio = (mean_return / std_dev) * np.sqrt(
        252
    )  # Annualized Sharpe Ratio
    return sharpe_ratio
