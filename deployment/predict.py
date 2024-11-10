from flask import Flask, jsonify, request

from .prediction_service import TrotroDurationPredictionService

DEPLOYMENT_MODEL_URI = "./model"
if not DEPLOYMENT_MODEL_URI:
    raise ValueError(f"Deployment model url is not defined: {DEPLOYMENT_MODEL_URI}")

duration_prediction = TrotroDurationPredictionService(model_uri=DEPLOYMENT_MODEL_URI)

app = Flask("Trotro-duration-prediction")


@app.route("/predict", methods=["POST"])
def prediction_endpont():
    """Handles prediction requests and returns the predicted duration."""
    request_body = request.get_json()
    prediction = dict(request_body)
    prediction["duration"] = duration_prediction.predict(dict(request_body))
    return jsonify(prediction)


if __name__ == "__main__":

    app.run(debug=True, host="0.0.0.0", port=9696)
