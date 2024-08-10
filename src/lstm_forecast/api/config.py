import os


class Config:
    MODEL_PATH = os.environ.get("MODEL_PATH") or "models/best_model.onnx"
    SECRET_KEY = os.environ.get("SECRET_KEY") or "you-will-never-guess"
    # Add other configuration variables as needed
