# LSTM Forecast

![Test and Build](https://github.com/gianlucamazza/lstm_forecast/workflows/Test%20and%20Build/badge.svg)
![Deploy to PyPI](https://github.com/gianlucamazza/lstm_forecast/workflows/Deploy%20to%20PyPI/badge.svg)

## Overview

**LSTM Forecast** is an advanced tool designed for predicting cryptocurrency prices using Long Short-Term Memory (LSTM) networks. This project combines cutting-edge deep learning techniques with comprehensive financial analysis, making it a robust solution for traders and data scientists alike.

With a focus on modularity and flexibility, LSTM Forecast provides a full suite of features including data loading, preprocessing, model training, hyperparameter optimization, backtesting, and API deployment. These tools enable users to build, evaluate, and deploy predictive models tailored to their specific needs, whether for academic research, financial analysis, or active trading.

## Key Features

-   **Data Loading and Preprocessing**: Easily load and preprocess time series data from various sources, with built-in support for technical indicators.
-   **Model Training**: Train LSTM models with customizable architectures and hyperparameters to fit your specific use case.
-   **Hyperparameter Optimization**: Leverage advanced techniques like Optuna for automated hyperparameter tuning and feature selection.
-   **Backtesting**: Evaluate model performance using historical data with a fully integrated backtesting engine.
-   **API Deployment**: Deploy your trained models as RESTful APIs using FastAPI, enabling real-time predictions and integrations.
-   **Command-Line Interface**: A powerful CLI that simplifies all aspects of model training, evaluation, and deployment.

LSTM Forecast is designed to be both user-friendly and powerful, providing everything you need to create and deploy sophisticated forecasting models for cryptocurrencies and financial assets.

## Table of Contents

-   [LSTM Forecast](#lstm-forecast)
    -   [Overview](#overview)
    -   [Key Features](#key-features)
    -   [Table of Contents](#table-of-contents)
    -   [Project Structure](#project-structure)
    -   [Installation](#installation)
        -   [Install with pip](#install-with-pip)
        -   [Install from sources](#install-from-sources)
    -   [Usage](#usage)
        -   [Prepare Data](#prepare-data)
        -   [Optimize Hyperparameters and Feature Selection](#optimize-hyperparameters-and-feature-selection)
        -   [Train the Model](#train-the-model)
        -   [Make Predictions](#make-predictions)
        -   [Start the API Server](#start-the-api-server)
        -   [General Usage](#general-usage)
    -   [Configuration](#configuration)
    -   [API](#api)
    -   [Backtesting](#backtesting)
    -   [Testing](#testing)
    -   [Documentation](#documentation)
    -   [License](#license)

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

### Install with pip

If you want to use the latest stable version, you can install the package directly from PyPI:

```
pip install lstm-forecast
```

### Install from sources

If you want to use the latest development version or contribute to the project, you can install from the source:

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

This way, you'll have the latest version of the code and be able to make changes if needed.

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

-   `--n_trials INTEGER`: Number of trials for hyperparameter tuning (default: 100)
-   `--n_feature_trials INTEGER`: Number of trials for feature selection (default: 15)
-   `--min_features INTEGER`: Minimum number of features to select (default: 5)
-   `--force`: Force re-run of Optuna study

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
uvicorn lstm_forecast.api.app:app --reload
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

-   Data settings (ticker, date range, features)
-   Model settings (hidden size, number of layers, dropout)
-   Training settings (epochs, learning rate, batch size)
-   Backtesting parameters

## API

The project includes a FastAPI-based API for model inference. To start the API server:

```
uvicorn lstm_forecast.api.app:app --reload
```

API endpoints:

-   `/predict`: Make predictions using the trained model
-   `/backtest`: Run backtesting on historical data

## Backtesting

The backtesting module allows you to evaluate the model's performance on historical data. It includes:

-   Trading engine simulation
-   Performance metrics calculation
-   Visualization of results

To run a backtest:

```
lstm_forecast backtest --config path/to/config.json --model path/to/model.pth
```

## Testing

To run the tests:

```
pytest tests/
```

## Documentation

Comprehensive technical documentation is available at the following link:

[View LSTM Forecast Documentation](https://gianlucamazza.github.io/lstm_forecast/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
