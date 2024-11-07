import os
from flask import Flask, request, jsonify
import mlflow
import sys
from pathlib import Path

# Add the parent directory to the system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from orchestration.scripts.prepare_data import compute_features

import pandas as pd


class TrotroDurationPrediction(object):

    def __init__(self, model_uri) -> None:
        self.model = mlflow.pyfunc.load_model(model_uri)

    def predict(self, trotro_data: dict):
        input_data = pd.DataFrame([trotro_data])
        features = compute_features(input_data)
        features.drop(columns=["tpep_pickup_datetime"], inplace=True)
        preds = self.model.predict(features)
        return float(preds[0])


print(os.getcwd())
deployment_model_uri = "./model"
if not deployment_model_uri:
    raise ValueError(f"Deployment model url is not defined: {deployment_model_uri}")

duration_prediction = TrotroDurationPrediction(model_uri=deployment_model_uri)

app = Flask("Trotro-duration-prediction")


@app.route("/predict", methods=["POST"])
def prediction_endpont():
    request_body = request.get_json()
    prediction = dict(request_body)
    prediction["duration"] = duration_prediction.predict(dict(request_body))
    return jsonify(prediction)


if __name__ == "__main__":

    app.run(debug=True, host="0.0.0.0", port=9696)
