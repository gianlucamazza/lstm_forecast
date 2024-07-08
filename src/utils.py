import json


def load_json(path: str) -> dict:
    """
    Load a JSON file from a given path.

    Args:
        path (str): The path to the JSON file.

    Returns:
        dict: The JSON file as a dictionary.
    """
    with open(path, 'r') as config_file:
        config = json.load(config_file)
    return config
