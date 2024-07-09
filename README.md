# [PYTORCH] Price Prediction with LSTM

## Overview

This project aims to predict stock prices using Long Short-Term Memory (LSTM) networks.
The model is trained on historical stock data, which includes various technical indicators.
The pipeline includes data preprocessing, feature engineering, model training, and prediction.

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
    "ticker": "^GSPC",
    "symbol": "S&P_500",
    "start_date": "2018-01-01",
    "log_dir": "logs",
    "look_back": 90,
    "look_forward": 30,
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 0.001,
    "model_path": "models",
    "features": [
        "Close",
        "SMA_50",
        "SMA_200",
        "EMA",
        "MACD",
        "MACD_Signal",
        "RSI",
        "Bollinger_High",
        "Bollinger_Low",
        "ATR",
        "OBV",
        "VWAP",
        "ADX",
        "Stochastic",
        "Aroon_Up",
        "Aroon_Down",
        "Williams_R",
        "CMF",
        "CCI",
        "SMA_Agg",
        "MACD_Agg",
        "Bollinger_Bandwidth"
    ],
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
        "CCI": 20
    },
    "frequency": "B",
    "target": "Close",
    "best_features": [
        "ADX",
        "EMA",
        "SMA_Agg",
        "MACD_Agg",
        "Bollinger_Bandwidth"
    ]
}
```

## Usage

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

```bash
run.sh
```

### Structure

`src/data_loader.py` handles data loading, preprocessing, and feature selection.

`src/feature_engineering.py` calculates various technical indicators used as features for the model.

`src/model.py` contains the definition of the LSTM model and the early stopping class.

`src/train.py` script preprocesses the data, trains the model, and evaluates it.

`src/predict.py` script preprocesses the data, loads the trained model, and generates predictions.

### Output

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

`png/{symbol}_prediction.png` contains the predicted stock prices.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
