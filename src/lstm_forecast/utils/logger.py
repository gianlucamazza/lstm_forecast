import os
import sys
import logging

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "lstm_forecast.log")


def setup_logger(
    name, log_file=LOG_FILE, level=logging.INFO
) -> logging.Logger:
    if (
        "sphinx" in sys.modules
    ):  # Disabilita la creazione dei logger durante la costruzione della documentazione
        return logging.getLogger(name)

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

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
