import os

import mlflow
import pandas as pd
from flask import Flask, jsonify, request


class TrotroDurationPrediction(object):

    def __init__(self, model_uri) -> None:
        self.model = mlflow.pyfunc.load_model(model_uri)

    def compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Computes features from the given DataFrame."""

        data["tpep_pickup_datetime"] = pd.to_datetime(data.tpep_pickup_datetime)
        data["hour_of_day"] = data.tpep_pickup_datetime.dt.hour
        data["hour_of_day"] = data["hour_of_day"].astype(float)
        data["day_of_week"] = data.tpep_pickup_datetime.dt.dayofweek
        data["day_of_week"] = data["day_of_week"].astype(float)

        return data

    def predict(self, trotro_data: dict):
        input_data = pd.DataFrame([trotro_data])
        features = self.compute_features(input_data)
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
