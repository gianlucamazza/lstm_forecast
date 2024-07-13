import json
import time


class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.ticker = self.config.get("ticker")
        self.symbol = self.config.get("symbol")
        self.asset_type = self.config.get("asset_type")
        self.data_sampling_interval = self.config.get("data_sampling_interval")
        self.start_date = self.config.get("start_date")
        self.end_date = self.config.get("end_date", time.strftime("%Y-%m-%d"))
        self.log_dir = self.config.get("log_dir")
        self.model_params = self.config.get("model_params", {})
        self.look_back = self.config.get("look_back")
        self.look_forward = self.config.get("look_forward")
        self.epochs = self.config.get("epochs")
        self.batch_size = self.config.get("batch_size")
        self.model_dir = self.config.get("model_dir")
        self.indicator_windows = self.config.get("indicator_windows", {})
        self.data_resampling_frequency = self.config.get("data_resampling_frequency")
        self.targets = self.config.get("targets", [])
        self.backtesting_params = self.config.get("backtesting_params", {})
        self.best_features = self.config.get("best_features", [])
        self.feature_selection_algo = self.config.get(
            "feature_selection_algo", "random_forest"
        )
        self.model_params["fc_output_size"] = len(self.targets)

    def save(self):
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=4)


def load_config(config_path):
    return Config(config_path)


def update_config(config, key, value):
    if isinstance(config, Config):
        setattr(config, key, value)
        config.config[key] = value
    else:
        raise TypeError("The provided config is not an instance of the Config class")
    return config
