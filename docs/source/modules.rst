# LSTM Forecast Documentation

![Test and Build](https://github.com/gianlucamazza/lstm_forecast/workflows/Test%20and%20Build/badge.svg)
![Deploy to PyPI](https://github.com/gianlucamazza/lstm_forecast/workflows/Deploy%20to%20PyPI/badge.svg)

## Overview
LSTM-based model for predicting cryptocurrency prices. It includes features for data loading, preprocessing, model training, hyperparameter optimization, backtesting, and API deployment.

## Installation
There are two ways to install the LSTM Forecast project:

### Install with pip
```
pip install lstm-forecast
```

### Install from sources
1. Clone the repository:
   ```
   git clone https://github.com/gianlucamazza/lstm_forecast.git
   cd lstm_forecast
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Install the project in editable mode:
   ```
   pip install -e .
   ```

## Usage
The `lstm_forecast` command-line interface provides several subcommands:

- Prepare Data: `lstm_forecast prepare --config path/to/config.json`
- Optimize Hyperparameters: `lstm_forecast optimize --config path/to/config.json [OPTIONS]`
- Train the Model: `lstm_forecast train --config path/to/config.json`
- Make Predictions: `lstm_forecast predict --config path/to/config.json`
- Start the API Server: `lstm_forecast server --config path/to/config.json`

For more information on any command, use the `--help` option:
```
lstm_forecast [COMMAND] --help
```

## Configuration
The `config.json` file contains all necessary settings for data processing, model architecture, training, and prediction.

## API
Start the API server:
```
uvicorn lstm_forecast.api.app:app --reload
```

API endpoints:
- `/predict`: Make predictions using the trained model
- `/backtest`: Run backtesting on historical data

## Backtesting
Run a backtest:
```
lstm_forecast backtest --config path/to/config.json --model path/to/model.pth
```

## Testing
Run tests:
```
pytest tests/
```

## Modules

### Core Functionality
- [Model](lstm_forecast.model.html)
- [Train](lstm_forecast.train.html)
- [Predict](lstm_forecast.predict.html)

### Data Processing
- [Data Loader](lstm_forecast.data_loader.html)
- [Feature Engineering](lstm_forecast.feature_engineering.html)
- [Feature Selection](lstm_forecast.feature_selection.html)

### Model Optimization
- [Hyperparameter Optimization](lstm_forecast.hyperparameter_optimization.html)
- [Early Stopping](lstm_forecast.early_stopping.html)

### Utilities
- [Config](lstm_forecast.config.html)
- [Logger](lstm_forecast.logger.html)
- [CLI](lstm_forecast.cli.html)
- [Model Utils](lstm_forecast.model_utils.html)
- [Predict Utils](lstm_forecast.predict_utils.html)

### API
- [App](lstm_forecast.api.app.html)
- [Routes](lstm_forecast.api.routes.html)
- [Models](lstm_forecast.api.models.html)
- [Utils](lstm_forecast.api.utils.html)

### Backtesting
- [Backtest](lstm_forecast.backtesting.backtest.html)
- [Metrics](lstm_forecast.backtesting.metrics.html)
- [Plot](lstm_forecast.backtesting.plot.html)
- [Portfolio](lstm_forecast.backtesting.portfolio.html)
- [Trading Engine](lstm_forecast.backtesting.trading_engine.html)

## Module Details

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   lstm_forecast.model
   lstm_forecast.train
   lstm_forecast.predict
   lstm_forecast.data_loader
   lstm_forecast.feature_engineering
   lstm_forecast.feature_selection
   lstm_forecast.hyperparameter_optimization
   lstm_forecast.early_stopping
   lstm_forecast.config
   lstm_forecast.logger
   lstm_forecast.cli
   lstm_forecast.model_utils
   lstm_forecast.predict_utils
   lstm_forecast.api.app
   lstm_forecast.api.routes
   lstm_forecast.api.models
   lstm_forecast.api.utils
   lstm_forecast.backtesting.backtest
   lstm_forecast.backtesting.metrics
   lstm_forecast.backtesting.plot
   lstm_forecast.backtesting.portfolio
   lstm_forecast.backtesting.trading_engine

## Indices and tables

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.