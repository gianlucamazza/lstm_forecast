import json


def load_json(path: str) -> dict:
    with open(path, 'r') as config_file:
        config = json.load(config_file)
    return config
