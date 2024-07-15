import logging
import os
import json

# Load logging configuration from config.json
with open(os.path.join(os.path.dirname(__file__), "..", "config.json"), "r") as f:
    config = json.load(f)

LOG_DIR = config.get("log_dir", "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FILE = os.path.join(LOG_DIR, "main.log")


def setup_logger(name, log_file=LOG_FILE, level=logging.INFO):
    """Function to setup a logger with the given name and log file."""
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger_module = logging.getLogger(name)
    logger_module.setLevel(level)
    logger_module.addHandler(handler)
    logger_module.addHandler(console_handler)

    return logger_module
