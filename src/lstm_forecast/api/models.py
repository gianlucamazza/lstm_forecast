import os
import sys
import onnxruntime as ort
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.config import Config

def load_model():
    return ort.InferenceSession(Config.MODEL_PATH)

def make_prediction(model, data):
    ort_inputs = {model.get_inputs()[0].name: data}
    return model.run(None, ort_inputs)[0]