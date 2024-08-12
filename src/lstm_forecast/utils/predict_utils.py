import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def inverse_transform_predictions(
    predictions: np.ndarray,
    scaler_prices: StandardScaler,
    scaler_volume: MinMaxScaler,
    features: List[str],
    num_targets: int,
) -> np.ndarray:
    predictions_reshaped = np.zeros(
        (predictions.shape[0], len(features) + num_targets)
    )
    predictions_reshaped[:, :num_targets] = predictions

    predictions_reshaped[:, : num_targets - 1] = (
        scaler_prices.inverse_transform(predictions[:, : num_targets - 1])
    )
    predictions_reshaped[:, num_targets - 1 : num_targets] = (
        scaler_volume.inverse_transform(
            predictions[:, num_targets - 1 : num_targets]
        )
    )

    return predictions_reshaped[:, :num_targets]


def create_candles(
    predictions: np.ndarray, freq: str, start_date: pd.Timestamp
) -> pd.DataFrame:
    columns = ["Open", "High", "Low", "Close"]
    if predictions.shape[1] == 5:
        columns.append("Volume")
    df = pd.DataFrame(predictions, columns=columns)
    df.index = pd.date_range(start=start_date, periods=len(df), freq=freq)
    return df


def plot_predictions(
    symbol: str,
    filename: str,
    candles: pd.DataFrame,
    predictions: np.ndarray,
    future_predictions: np.ndarray,
    freq: str,
    interval: str,
    logger,
) -> None:
    logger.info("Plotting predictions")

    start_date = candles.index[-1]

    fig = go.Figure()
    add_candlestick_trace(fig, candles, "Actual Candlestick")

    historical_candles = create_candles(
        predictions, freq, candles.index[-len(predictions)]
    )
    add_candlestick_trace(fig, historical_candles, "Predicted Candlestick")

    future_candles = create_candles(future_predictions, freq, start_date)
    add_candlestick_trace(
        fig,
        future_candles,
        "Future Predicted Candlestick",
        increasing_color="blue",
        decreasing_color="orange",
    )

    update_layout(fig, symbol, interval)

    fig.write_html(filename)
    logger.info(f"Predictions plot saved to {filename}")


def add_candlestick_trace(
    fig: go.Figure,
    candles: pd.DataFrame,
    name: str,
    increasing_color="green",
    decreasing_color="red",
) -> None:
    fig.add_trace(
        go.Candlestick(
            x=candles.index,
            open=candles["Open"],
            high=candles["High"],
            low=candles["Low"],
            close=candles["Close"],
            name=name,
            increasing=dict(line=dict(color=increasing_color)),
            decreasing=dict(line=dict(color=decreasing_color)),
        )
    )


def update_layout(fig: go.Figure, symbol: str, interval: str) -> None:
    fig.update_layout(
        title=f"{symbol} - Predictions",
        xaxis_title="Date" if "d" in interval else "Date/Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        xaxis=dict(
            tickformat="%Y-%m-%d %H:%M" if "h" in interval else "%Y-%m-%d"
        ),
    )


def save_predictions_report(
    predictions: np.ndarray,
    targets: List[str],
    start_date: pd.Timestamp,
    freq: str,
    symbol: str,
) -> None:
    if predictions.shape[1] < len(targets):
        padding_width = len(targets) - predictions.shape[1]
        predictions = np.pad(
            predictions, ((0, 0), (0, padding_width)), "constant"
        )

    report = pd.DataFrame(data=predictions, columns=targets)
    report.index = pd.date_range(
        start=start_date, periods=len(predictions), freq=freq
    )
    report.to_csv(f"reports/{symbol}_predictions.csv", index=False)
