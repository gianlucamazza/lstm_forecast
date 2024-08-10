import onnxruntime as ort
from api.config import Config


def load_model():
    return ort.InferenceSession(Config.MODEL_PATH)


def make_prediction(model, data):
    ort_inputs = {model.get_inputs()[0].name: data}
    return model.run(None, ort_inputs)[0]
