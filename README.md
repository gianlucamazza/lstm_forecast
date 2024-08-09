# [PYTORCH] LSTM Forecast

![Test and Build](https://github.com/username/repository-name/workflows/Test%20and%20Build/badge.svg)
![Deploy to PyPI](https://github.com/username/repository-name/workflows/Deploy%20to%20PyPI/badge.svg)

## Overview

This project aims to predict stock prices using Long Short-Term Memory (LSTM) networks. The model is trained on historical stock data, which includes various technical indicators. The pipeline includes data preprocessing, feature engineering, model training, and prediction.

## Pipeline

1. **Data Loading**: Load historical stock data using the `yfinance` library.
2. **Data Preprocessing**: Preprocess the data by filling missing values and resampling the data.
3. **Feature Engineering**: Calculate technical indicators and add them to the dataset.
4. **Feature Selection**: Select the best features using the `feature_engineering.py` script.
5. **Model Training**: Train the LSTM model using the selected features.
6. **Model Evaluation**: Evaluate the model using Mean Squared Error (MSE) and Mean Absolute Error (MAE) metrics.
7. **Prediction**: Generate predictions using the trained model.
8. **Hyperparameter Optimization**: Optimize the hyperparameters of the LSTM model using the Optuna library.
9. **Backtesting**: Backtest the model on historical data to evaluate its performance.
10. **Visualization**: Plot the predictions and evaluation metrics using Plotly.
11. **Logging**: Log the training and prediction results using the `logging` library.

## Data

The historical stock data is loaded using the `yfinance` library. The data includes the following columns:

- `Date`: The date of the stock price.
- `Open`: The opening price of the stock.
- `High`: The highest price of the stock.
- `Low`: The lowest price of the stock.
- `Close`: The closing price of the stock.
- `Adj Close`: The adjusted closing price of the stock.
- `Volume`: The volume of the stock traded.

## Setup

### Requirements

Ensure you have Python 3.11 installed. Install the necessary packages using:

```bash
pip install .
```

## Usage

```bash
$ lstm_forecast -h
usage: lstm_forecast [-h] {optimize,train,predict,server} ...

positional arguments:
  {optimize,train,predict,server}
    optimize            Optimize feature selection and hyperparameters
    train               Train the model
    predict             Make predictions
    server              Start the API server

optional arguments:
  -h, --help            show this help message and exit
```

### Configuration

Update the `config.json` file with the desired parameters.

#### `data_settings`

Settings related to data:

- `ticker`: Stock ticker symbol.
- `symbol`: Stock symbol.
- `data_path`: Path to the historical data CSV file.
- `scaled_data_path`: Path to the scaled data CSV file.
- `asset_type`: Type of asset (e.g., 'commodity').
- `data_sampling_interval`: Interval for the historical data (e.g., '1d' for daily).
- `start_date`: Start date for the historical data.
- `data_resampling_frequency`: Frequency of the data (e.g., 'D' for daily).
- `technical_indicators`: Dictionary containing the window sizes for the technical indicators:
  - `SMA_50`: Window size for the 50-day Simple Moving Average.
  - `SMA_200`: Window size for the 200-day Simple Moving Average.
  - `EMA`: Window size for the Exponential Moving Average.
  - `MACD`: List of three values representing the windows for the MACD indicator.
  - `RSI`: Window size for the Relative Strength Index.
  - `Bollinger`: Window size for the Bollinger Bands.
  - `ATR`: Window size for the Average True Range.
  - `Stochastic`: Window size for the Stochastic Oscillator.
  - `Aroon`: Window size for the Aroon indicator.
  - `Williams_R`: Window size for the Williams %R indicator.
  - `CMF`: Window size for the Chaikin Money Flow indicator.
  - `CCI`: Window size for the Commodity Channel Index.
  - `Ichimoku`: Dictionary containing the window sizes for the Ichimoku Cloud:
    - `window1`: Window size for the Tenkan-sen.
    - `window2`: Window size for the Kijun-sen.
    - `window3`: Window size for the Senkou Span B.
- `targets`: List of target variables for prediction.
- `disabled_features`: List of features to be disabled.
- `all_features`: List of all available features.
- `selected_features`: List of best features selected for training.

#### `model_settings`

Settings related to the model:

- `hidden_size`: Number of hidden units in the LSTM layer.
- `num_layers`: Number of LSTM layers.
- `dropout`: Dropout rate.
- `learning_rate`: Learning rate for the optimizer.
- `fc_output_size`: Output size of the fully connected layer.
- `weight_decay`: Weight decay for the optimizer.
- `clip_value`: Gradient clipping value.
- `batch_size`: Batch size for training.
- `sequence_length`: Sequence length for the LSTM model.

#### `training_settings`

Settings related to training:

- `look_back`: Number of days to look back for the LSTM model.
- `look_forward`: Number of days to predict.
- `epochs`: Number of epochs for training.
- `model_dir`: Directory to store the trained model.
- `use_time_series_split`: Boolean to use time series split for cross-validation.
- `time_series_splits`: Number of time series splits for cross-validation.

#### `logging_settings`

Settings related to logging:

- `log_dir`: Directory to store the logs.

#### `backtesting_params`

Parameters for backtesting:

- `initial_cash`: Initial cash for backtesting.
- `trading_fee`: Trading fee for backtesting.
- `take_profit`: Take profit ratio.
- `stop_loss`: Stop loss ratio.
- `trade_allocation`: Trade allocation percentage.
- `max_open_trades`: Maximum number of open trades.

### Models

`models/{symbol}_model.pth` contains the trained model.

## Contributors

- **Gianluca Mazza** - [LinkedIn](https://www.linkedin.com/in/gianlucamazza/)
- **Matteo Garbelli** - [LinkedIn](https://www.linkedin.com/in/matteo-garbelli-1a0bb3b1/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
