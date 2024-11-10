import mlflow
import pandas as pd


class TrotroDurationPredictionService:
    """Service for predicting Trotro duration based on input features."""

    def __init__(self, model_uri, model=None) -> None:
        if model:
            self.model = model
        else:
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
        """Predicts the duration based on the provided trotro data."""
        input_data = pd.DataFrame([trotro_data])
        features = self.compute_features(input_data)
        features.drop(columns=["tpep_pickup_datetime"], inplace=True)
        preds = self.model.predict(features)
        return float(preds[0])
