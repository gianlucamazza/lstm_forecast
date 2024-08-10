# LSTM Forecast

## Overview

This project implements an LSTM-based model for predicting cryptocurrency prices. It includes features for data loading, preprocessing, model training, hyperparameter optimization, backtesting, and API deployment.

## Table of Contents

- [LSTM Forecast](#lstm-forecast)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Prepare Data](#prepare-data)
    - [Optimize Hyperparameters and Feature Selection](#optimize-hyperparameters-and-feature-selection)
    - [Train the Model](#train-the-model)
    - [Make Predictions](#make-predictions)
    - [Start the API Server](#start-the-api-server)
    - [General Usage](#general-usage)
  - [Configuration](#configuration)
  - [API](#api)
  - [Backtesting](#backtesting)
  - [Testing](#testing)
  - [License](#license)

## Project Structure

```
.
├── LICENSE
├── README.md
├── config.json
├── data/
├── docs/
├── logs/
├── models/
│   ├── checkpoints/
│   └── optuna/
├── png/
├── pyproject.toml
├── reports/
├── requirements.txt
├── setup.py
├── src/
│   └── lstm_forecast/
│       ├── __init__.py
│       ├── api/
│       ├── backtesting/
│       ├── cli.py
│       ├── config.py
│       ├── data_loader.py
│       ├── early_stopping.py
│       ├── feature_engineering.py
│       ├── feature_selection.py
│       ├── generate_html.py
│       ├── hyperparameter_optimization.py
│       ├── logger.py
│       ├── model.py
│       ├── model_utils.py
│       ├── predict.py
│       ├── predict_utils.py
│       └── train.py
├── static/
└── tests/
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/lstm-price-prediction.git
   cd lstm-price-prediction
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

The `lstm_forecast` command-line interface provides several subcommands for different functionalities:

### Prepare Data

To prepare the data for training and prediction:

```
lstm_forecast prepare --config path/to/config.json
```

### Optimize Hyperparameters and Feature Selection

To run hyperparameter optimization and feature selection:

```
lstm_forecast optimize --config path/to/config.json [OPTIONS]
```

Options:
- `--n_trials INTEGER`: Number of trials for hyperparameter tuning (default: 100)
- `--n_feature_trials INTEGER`: Number of trials for feature selection (default: 15)
- `--min_features INTEGER`: Minimum number of features to select (default: 5)
- `--force`: Force re-run of Optuna study

### Train the Model

To train the model:

```
lstm_forecast train --config path/to/config.json
```

### Make Predictions

To make predictions using a trained model:

```
lstm_forecast predict --config path/to/config.json
```

### Start the API Server

To start the API server:

```
lstm_forecast server --config path/to/config.json
```

### General Usage

All commands require a configuration file specified with the `--config` option. This JSON file contains all the necessary settings for data processing, model architecture, training, and prediction.

For more information on any command, you can use the `--help` option:

```
lstm_forecast [COMMAND] --help
```

Replace `[COMMAND]` with any of the available commands (prepare, optimize, train, predict, server) to see specific help for that command.

## Configuration

The `config.json` file contains all the necessary settings for data processing, model architecture, training, and prediction. Modify this file to adjust parameters such as:

- Data settings (ticker, date range, features)
- Model settings (hidden size, number of layers, dropout)
- Training settings (epochs, learning rate, batch size)
- Backtesting parameters

## API

The project includes a FastAPI-based API for model inference. To start the API server:

```
uvicorn lstm_forecast.api.app:app --reload
```

API endpoints:
- `/predict`: Make predictions using the trained model
- `/backtest`: Run backtesting on historical data

## Backtesting

The backtesting module allows you to evaluate the model's performance on historical data. It includes:

- Trading engine simulation
- Performance metrics calculation
- Visualization of results

To run a backtest:

```
lstm_forecast backtest --config path/to/config.json --model path/to/model.pth
```

## Testing

To run the tests:

```
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.