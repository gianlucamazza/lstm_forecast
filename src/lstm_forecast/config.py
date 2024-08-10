import json
import time


class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        self.load_config()

    def load_config(self):
        with open(self.config_path, "r") as f:
            self.config = json.load(f)
        self._parse_config()

    def _parse_config(self):
        self.data_settings = self.config.get("data_settings", {})
        self.model_settings = self.config.get("model_settings", {})
        self.training_settings = self.config.get("training_settings", {})
        self.logging_settings = self.config.get("logging_settings", {})
        self.backtesting_params = self.config.get("backtesting_params", {})

        self.ticker = self.data_settings.get("ticker")
        self.symbol = self.data_settings.get("symbol")
        self.asset_type = self.data_settings.get("asset_type")
        self.data_path = self.data_settings.get("data_path")
        self.data_sampling_interval = self.data_settings.get(
            "data_sampling_interval"
        )
        self.start_date = self.data_settings.get("start_date")
        self.end_date = self.data_settings.get(
            "end_date", time.strftime("%Y-%m-%d")
        )
        self.indicator_windows = self.data_settings.get(
            "technical_indicators", {}
        )
        self.data_resampling_frequency = self.data_settings.get(
            "data_resampling_frequency"
        )
        self.targets = self.data_settings.get("targets", [])
        self.disabled_features = self.data_settings.get(
            "disabled_features", []
        )
        self.all_features = self.data_settings.get("all_features", [])
        self.selected_features = self.data_settings.get(
            "selected_features", []
        )
        self.log_dir = self.logging_settings.get("log_dir")

        self.look_back = self.training_settings.get("look_back")
        self.look_forward = self.training_settings.get("look_forward")
        self.epochs = self.training_settings.get("epochs")
        self.batch_size = self.training_settings.get("batch_size")
        self.model_dir = self.training_settings.get("model_dir")

        self.model_settings["fc_output_size"] = len(self.targets)

    def save(self):
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=4)

    def update(self, key, value):
        keys = key.split(".")
        d = self.config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
        self._parse_config()  # Update attributes to reflect the change

    def get(self, key, default=None):
        keys = key.split(".")
        d = self.config
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k, default)
            else:
                return default
        return d


def load_config(config_path):
    return Config(config_path)


def update_config(config, key, value):
    if isinstance(config, Config):
        config.update(key, value)
    else:
        raise TypeError(
            "The provided config is not an instance of the Config class"
        )
    return config
