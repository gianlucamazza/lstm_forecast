# [PYTORCH] Price Prediction with LSTM

## Overview

This project aims to predict stock prices using Long Short-Term Memory (LSTM) networks.
The model is trained on historical stock data, which includes various technical indicators.
The pipeline includes data preprocessing, feature engineering, model training, and prediction.

## Published models/predictions

[Live website]("https://gianlucamazza.github.io/pytorch-lstm-price_prediction/")

## Evaluation

The model is evaluated using the Mean Squared Error (MSE) and Mean Absolute Error (MAE) metrics.
The evaluation results are displayed in the training logs and the evaluation plot.

![Evaluation](static/S&P_500_evaluation.png)

## Setup

### Requirements

Ensure you have Python 3.11 installed. Install the necessary packages using:

```bash
pip install -r requirements.txt
```

## Configuration

The config.json file contains all the configuration parameters required for training and prediction.

Example config.json:

```json
{
    "ticker": "BTC-USD",
    "symbol": "BTCUSD",
    "asset_type": "commodity",
    "data_sampling_interval": "1d",
    "start_date": "2013-01-01",
    "log_dir": "logs",
    "model_params": {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 0.005,
        "fc_output_size": 5
    },
    "look_back": 120,
    "look_forward": 30,
    "epochs": 100,
    "batch_size": 64,
    "model_dir": "models",
    "indicator_windows": {
        "SMA_50": 50,
        "SMA_200": 200,
        "EMA": 20,
        "MACD": [
            12,
            26,
            9
        ],
        "RSI": 14,
        "Bollinger": 20,
        "ATR": 14,
        "Stochastic": 14,
        "Aroon": 25,
        "Williams_R": 14,
        "CMF": 20,
        "CCI": 20,
        "Ichimoku": {
            "window1": 9,
            "window2": 26,
            "window3": 52
        }
    },
    "data_resampling_frequency": "D",
    "targets": [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ],
    "backtesting_params": {
        "initial_cash": 100000,
        "trading_fee": 0.001,
        "take_profit": 1.5,
        "stop_loss": 1.2,
        "trade_allocation": 0.1,
        "max_open_trades": 5
    },
    "best_features": [
        "MACD_Signal",
        "ADX",
        "Bollinger_Low"
    ]
}
```

## Usage

### Configuration

Update the config.json file with the desired parameters.

- `ticker`: Stock ticker symbol.
- `symbol`: Stock symbol.
- `asset_type`: Type of asset, e.g., 'stocks', 'forex', 'commodity'.
- `data_sampling_interval`: Interval for the historical data, e.g., '1d' for daily or '1h' for hourly. [1h has been limited to 1 year of data]
- `start_date`: Start date for the historical data.
- `log_dir`: Directory to store the logs.
- `look_back`: Number of days to look back for the LSTM model.
- `look_forward`: Number of days to predict.
- `epochs`: Number of epochs for training.
- `batch_size`: Batch size for training.
- `learning_rate`: Learning rate for the optimizer.
- `model_path`: Directory to store the trained model.
- `model_params`: Dictionary containing the model parameters.
    - `hidden_size`: Number of hidden units in the LSTM layer.
    - `num_layers`: Number of LSTM layers.
    - `dropout`: Dropout rate.
    - `fc_output_size`: Output size of the fully connected layer.
- `indicator_windows`: Dictionary containing the window sizes for the technical indicators.
- `data_resampling_frequency`: Frequency of the data, e.g., 'B' for business days or 'D' for daily.
- `target`: Target variable for prediction.
- `best_features`: List of best features selected for training from the feature selection process.

if `best_features` is not present in the config file, the feature discovery process will be executed.

### Features

The model uses various technical indicators as features for training. The feature engineering script calculates these indicators from historical data based on the asset type.

The following technical indicators are used:

#### Simple Moving Average (SMA)
- **SMA 50**: The simple moving average calculated over the past 50 periods. It indicates the medium-term trend.
- **SMA 200**: The simple moving average calculated over the past 200 periods. It indicates the long-term trend.
- **SMA Agg**: The average of SMA 50 and SMA 200, used to reduce dimensionality and retain crucial information.

#### Exponential Moving Average (EMA)
- **EMA**: The exponential moving average calculated over a user-defined window (typically 20 periods). It gives more weight to recent prices compared to the SMA.

#### Moving Average Convergence Divergence (MACD)
- **MACD**: The difference between the 12-period EMA and the 26-period EMA. Used to identify trends and turning points.
- **MACD Signal**: The 9-period EMA of the MACD. It serves as a signal for trading operations.
- **MACD Agg**: The average of MACD and MACD Signal, used to aggregate information.

#### Relative Strength Index (RSI)
- **RSI**: A momentum indicator that measures the speed and change of price movements. Calculated over a 14-period window, it indicates whether an asset is overbought or oversold.

#### Bollinger Bands
- **Bollinger High**: The upper band calculated as the 20-period simple moving average (SMA) plus two standard deviations.
- **Bollinger Low**: The lower band calculated as the 20-period SMA minus two standard deviations.
- **Bollinger Bandwidth**: The difference between Bollinger High and Bollinger Low, used to measure volatility.

#### Average True Range (ATR)
- **ATR**: A measure of volatility calculated as the exponential moving average (EMA) of the true range over a defined period (typically 14 periods).

#### Stochastic Oscillator
- **Stochastic**: Indicates the closing position relative to the high-low range over a 14-period window. Used to identify overbought and oversold conditions.

#### Aroon Indicator
- **Aroon Up**: Calculated over a 25-period window, it measures the number of periods since the last highest high.
- **Aroon Down**: Calculated over a 25-period window, it measures the number of periods since the last lowest low.

#### Chaikin Money Flow (CMF)
- **CMF**: A measure of the money flow in an asset calculated over a 20-period window. Used to confirm trends and identify buying or selling pressure.

#### Commodity Channel Index (CCI)
- **CCI**: Calculated over a 20-period window, it measures the variation of price from its average. Used to identify overbought and oversold conditions.

#### Ichimoku Cloud
- **Ichimoku Tenkan**: The conversion line calculated as the average of the highest high and lowest low over the past 9 periods.
- **Ichimoku Kijun**: The base line calculated as the average of the highest high and lowest low over the past 26 periods.
- **Ichimoku Senkou Span A**: The average of the Tenkan and Kijun lines, projected 26 periods ahead.
- **Ichimoku Senkou Span B**: The average of the highest high and lowest low over the past 52 periods, projected 26 periods ahead.

### Description of Indicators and Features

1. **SMA 50** and **SMA 200**: Used to identify medium- and long-term trends. The crossover between SMA 50 and SMA 200 is often used as a trading signal.
2. **EMA**: Responds more quickly to price changes compared to SMA, useful for detecting recent trends.
3. **MACD and MACD Signal**: Used together to identify trend reversals and trend strength.
4. **RSI**: A momentum indicator that helps identify overbought and oversold conditions.
5. **Bollinger Bands**: Used to measure volatility. The bands expand during periods of high volatility and contract during periods of low volatility.
6. **ATR**: A measure of market volatility, useful for setting stop loss levels.
7. **Stochastic Oscillator**: Helps identify potential trend reversals in overbought and oversold conditions.
8. **Aroon Indicator**: Measures the strength of the current trend and the likelihood of a possible reversal.
9. **CMF**: Indicates buying and selling pressure, useful for confirming trends.
10. **CCI**: Indicates the strength of a trend and possible overbought or oversold conditions.
11. **Ichimoku Cloud**: Provides information on support, resistance, momentum, and future trends.

These technical indicators are calculated using the `ta` library and added to historical data to improve the accuracy of the prediction model.

### Training

To train the model, run:

```bash
python src/train.py  --config config.json
```

### Prediction

To predict stock prices, run:

```bash
python src/predict.py --config config.json
```

### Run all

To run the entire pipeline, run:

Options:
- `--config`: Path to the config file.
- `--skip-training`: Skip the training step.
- `--rebuild-features`: Rebuild the features.

```bash
run.sh 
```

## Pipeline

The pipeline consists of the following steps:

1. **Data Loading**: Load historical stock data using the `yfinance` library.
2. **Data Preprocessing**: Preprocess the data by filling missing values and resampling the data.
3. **Feature Engineering**: Calculate technical indicators and add them to the dataset.
4. **Feature Selection**: Select the best features using the `feature_engineering.py` script.
5. **Model Training**: Train the LSTM model using the selected features.
6. **Model Evaluation**: Evaluate the model using the Mean Squared Error (MSE) and Mean Absolute Error (MAE) metrics.
7. **Prediction**: Generate predictions using the trained model.

#### Logs

`logs/data_loader.log` contains the data loading and preprocessing logs.

`logs/feature_engineering.log` contains the feature engineering logs.

`logs/model.log` contains the model training logs.

`logs/train.log` contains the training logs.

`logs/predict.log` contains the prediction logs.

`logs/main.log` contains the main logs.

#### Models

`models/{symbol}_model.pth` contains the trained model.

#### Evaluation

`png/{symbol}_evaluation.png` contains the training and validation loss.

#### Predictions

`png/{symbol}_full.png` contains the prediction plot for the entire dataset.

`png/{symbol}_365_days.png` contains the prediction plot zoomed in on the last 365 days.

`png/{symbol}_90_days.png` contains the prediction plot zoomed in on the last 90 days.

## Contributors

- `Gianluca Mazza` - [LinkedIn](https://www.linkedin.com/in/gianlucamazza/)
- `Matteo Garbelli` - [LinkedIn](https://www.linkedin.com/in/matteo-garbelli-1a0bb3b1/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
