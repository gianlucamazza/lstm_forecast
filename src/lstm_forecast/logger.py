import os
import logging

LOG_DIR = "logs"
LOG_FILE = "logs/lstm_forecast.log"


def setup_logger(name, log_file=LOG_FILE, level=logging.INFO):
    """Function to setup a logger with the given name and log file."""
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger_module = logging.getLogger(name)
    logger_module.setLevel(level)

    if not logger_module.hasHandlers():
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger_module.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger_module.addHandler(console_handler)

    return logger_module

    return logger_module


model_logger = setup_logger("model_logger", os.path.join(LOG_DIR, "model.log"))
train_logger = setup_logger("train_logger", os.path.join(LOG_DIR, "train.log"))
optuna_logger = setup_logger("optuna_logger", os.path.join(LOG_DIR, "optuna.log"))
early_stopping_logger = setup_logger("early_stopping_logger", os.path.join(LOG_DIR, "early_stopping.log"))
feature_engineering_logger = setup_logger("feature_engineering_logger", os.path.join(LOG_DIR, "feature_engineering.log"))
predict_logger = setup_logger("predict_logger", os.path.join(LOG_DIR, "predict.log"))