Configuration
=============

configuration example (config.json):

.. code-block:: json

    {
        "data_settings": {
            "ticker": "BTC-USD",
            "symbol": "BTCUSD",
            "data_path": "data/BTCUSD_1d.csv",
            "scaled_data_path": "data/BTCUSD_1d_scaled_data.csv",
            "data_sampling_interval": "1d",
            "start_date": "2013-01-01",
            "data_resampling_frequency": "D",
            "technical_indicators": {
                "SMA_50": 50,
                "SMA_200": 200,
                "EMA": 20,
                "MACD": [12, 26, 9],
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
            "targets": ["Open", "High", "Low", "Close", "Volume"],
            "disabled_features": ["Adj Close"],
            "all_features": [
                "SMA_50",
                "SMA_200",
                "EMA",
                "MACD",
                "MACD_Signal",
                "RSI",
                "Bollinger_High",
                "Bollinger_Low",
                "ADX",
                "OBV",
                "VWAP",
                "Stochastic",
                "Aroon_Up",
                "Aroon_Down",
                "CCI",
                "CMF",
                "Ichimoku_Tenkan",
                "Ichimoku_Kijun",
                "Ichimoku_Senkou_Span_A",
                "Ichimoku_Senkou_Span_B",
                "Volume_SMA",
                "Volume_Change",
                "Volume_RSI",
                "Vortex_Positive",
                "Vortex_Negative",
                "TRIX",
                "PSAR",
                "PSAR_Up",
                "PSAR_Down"
            ],
            "selected_features": ["MACD", "SMA_200", "EMA"]
        },
        "model_settings": {
            "hidden_size": 391,
            "num_layers": 1,
            "dropout": 0.012239709228895848,
            "learning_rate": 0.00020025239898961104,
            "fc_output_size": 5,
            "weight_decay": 8.99288758616325e-5,
            "clip_value": 3.974825341474775,
            "batch_size": 20,
            "sequence_length": 16
        },
        "training_settings": {
            "look_back": 120,
            "look_forward": 30,
            "epochs": 150,
            "model_dir": "models",
            "plot_dir": "plots",
            "use_time_series_split": true,
            "time_series_splits": 5,
            "max_lag": 5
        },
        "logging_settings": {
            "log_dir": "logs"
        },
        "backtesting_params": {
            "initial_cash": 100000,
            "trading_fee": 0.001,
            "take_profit": 1.5,
            "stop_loss": 1.2,
            "trade_allocation": 0.1,
            "max_open_trades": 5
        }
    }

Explanation of the configuration file:

- **data_settings**: Contains the settings for the data loader, including the path to the data file, the technical indicators to use, the target variables, and the selected features.
- **model_settings**: Contains the hyperparameters for the LSTM model, including the hidden size, number of layers, dropout rate, learning rate, etc.
- **training_settings**: Contains the settings for training the model, including the look-back and look-forward periods, number of epochs, model directory, etc.
- **logging_settings**: Contains the settings for logging, including the log directory.

The configuration file is in JSON format and can be easily modified to suit your needs. You can create multiple configuration files for different datasets or experiments.
