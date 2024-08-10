from flask import Blueprint, request, jsonify
from .utils import preprocess_data, postprocess_prediction
from .models import load_model, make_prediction

main = Blueprint("main", __name__)


@main.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        processed_data = preprocess_data(data)
        model = load_model()
        prediction = make_prediction(model, processed_data)
        result = postprocess_prediction(prediction)
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@main.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200
